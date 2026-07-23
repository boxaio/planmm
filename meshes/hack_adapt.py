import os
import os.path as osp
import sys
import re
import glob
import cv2
import numpy as np
import trimesh
from PIL import Image
import igl
import open3d as o3d
import pandas as pd
import torch
import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, FrozenSet

import polyscope as ps

from meshes.hackhead import vertices2landmarks, valid_lmks_ids
from configs.env_paths import BASE_DIR, HACK_LMKS3D_PATH
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_gather, knn_points

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_LARGE_STEPS = _REPO / "meshes" / "large-steps"
if _LARGE_STEPS.is_dir() and str(_LARGE_STEPS) not in sys.path:
    sys.path.insert(0, str(_LARGE_STEPS))

from meshes.largesteps.parameterize import from_differential


def vert_area(verts, faces, eps=1e-18):
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)
    s = 0.5 * (A + B + C)
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=eps).sqrt()
    idx = faces.view(-1)
    v_areas = torch.zeros(verts.shape[0], dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    v_areas.scatter_add_(0, idx, val)
    return v_areas

def full_area(verts, faces, eps=1e-18):
    face_verts = verts[faces]
    barycentric = face_verts.mean(-2).unsqueeze(-2)
    fv_vec = face_verts - barycentric
        
    area = 0.5 * (fv_vec[:,0].cross(fv_vec[:,1]) + fv_vec[:,1].cross(fv_vec[:,2]) + fv_vec[:,2].cross(fv_vec[:,0])).norm(dim=1).abs().clamp_(min=eps)
    idx = faces.view(-1)
    v_areas = torch.zeros(verts.shape[0], dtype=torch.float32, device=verts.device)
    val = torch.stack([area] * 3, dim=1).view(-1)
    v_areas.scatter_add_(0, idx, val)
    return v_areas

def massmatrix_voronoi_approx(verts, faces):
    """
    Compute the area of the Voronoi cell around each vertex in the mesh.
    https://mathworld.wolfram.com/BarycentricCoordinates.html
    """
    l0 = (verts[faces[:,1]] - verts[faces[:,2]]).norm(dim=1)
    l1 = (verts[faces[:,2]] - verts[faces[:,0]]).norm(dim=1)
    l2 = (verts[faces[:,0]] - verts[faces[:,1]]).norm(dim=1)
    l = torch.stack((l0, l1, l2), dim=1)
    return torch.zeros_like(verts).scatter_add_(0, faces, l, ).mean(dim=1)

areaarea = massmatrix_voronoi_approx

def mass_loss(V, F, L, mat):
    tmp_Mass = areaarea(V, F)
    with torch.no_grad():
        m_mean = tmp_Mass.mean()
    with torch.no_grad():
        lap_c = (L @ V) / m_mean
        
        density = torch.linalg.norm(lap_c, dim=-1).unsqueeze(-1)
        density = from_differential(mat, density)
        density = from_differential(mat, density)
        density = from_differential(mat, density)
        
        density = density / density.mean()
        density = torch.reciprocal(density)
        density = torch.clamp(density, 0.0, 1.0)
        MM_d = tmp_Mass * density.squeeze()
    mass_mean_loss = (tmp_Mass - m_mean).square().mean() / m_mean
    mass_lap_loss =  (tmp_Mass - MM_d).square().mean() / m_mean

    return mass_mean_loss, mass_lap_loss


def _prepare_mesh_for_obj(vertices, faces):
    """Center and normalize mesh vertices to ~[-1, 1]^3 for OBJ export."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)

    min_coords = v.min(axis=0)
    max_coords = v.max(axis=0)
    center = (min_coords + max_coords) / 2.0
    extent = np.max(max_coords - min_coords)
    if extent > 0:
        v = (v - center) * (2.0 / extent)

    return v.astype(np.float32), f


# ------------

HACK_FITTING_INDICES = np.array([
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 168, 197, 5, 4,
    98, 97, 2, 326, 327, 246, 159, 157, 173, 153, 144, 398, 385, 387,
    263, 373, 381, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 38, 12, 268, 308, 316, 15, 86,
], dtype=np.int64)


# ------------- key landmark indices for eye lids, lips and nose -------------
EYE_LIDS_INDIECS = np.array([
    381, 384, 380, 385, 374, 386, 373, 387, 390, 388,
    153, 158, 145, 159, 144, 160, 163, 161, 7, 246,
])
LIPS_INDICES = np.array([
    41, 42, 183, 82, 13, 312, 271, 272, 407, 
    96, 89, 179, 86, 15, 316, 403, 319, 325,
])
NOSE_INDICES = np.array([168, 6, 197, 195, 5, 4, 129, 98, 97, 2, 326, 327, 358, 115, 438])

FACIAL_REGION_LMK_INDICES = np.unique(
    np.concatenate([EYE_LIDS_INDIECS, LIPS_INDICES, NOSE_INDICES]).astype(np.int64),
)

PHACK_RENDER_FOCAL = 1015.0
PHACK_RENDER_IMAGE_SIZE = 224

_DEFAULT_BNI_LMKS_DIR = "/media/ubuntu/xb/FFHQ_Dataset/bni_lmks/"
_DEFAULT_BNI_SEG_DIR = "/media/ubuntu/xb/FFHQ_Dataset/bni_seg/"

FACE_PARSE_LABELS: FrozenSet[int] = frozenset({
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15,
})
_FACE_PARSE_LABELS = FACE_PARSE_LABELS

# jonathandinu-face-parsing: 4=l_eye, 5=r_eye, 10=mouth
EXCLUDE_EYE_MOUTH_PARSE_LABELS: FrozenSet[int] = frozenset({4, 5, 10})
BNI_MESH_MASK_PARSE_LABELS: FrozenSet[int] = frozenset(
    lbl for lbl in FACE_PARSE_LABELS if lbl not in EXCLUDE_EYE_MOUTH_PARSE_LABELS
)


def face_mask_from_parsing(
    parsing: np.ndarray,
    face_labels: Optional[FrozenSet[int]] = None,
) -> np.ndarray:
    """Segformer 语义标签图 → 脸部区域 bool mask。"""
    if face_labels is None:
        face_labels = FACE_PARSE_LABELS
    parsing = np.asarray(parsing)
    return np.isin(parsing, list(face_labels))


def stem_from_normal_map_filename(normal_map_name: str) -> str:
    stem = Path(normal_map_name).stem
    stem = re.sub(r"normal_map", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[._-]+", "_", stem).strip("_")
    return stem or "mesh"


def pil_or_array_to_hwc_rgb_uint8(img) -> np.ndarray:
    """统一为 uint8 ``(H, W, 3)``。"""
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)

    arr = np.asarray(img)
    if arr.ndim == 1:
        side = int(round(np.sqrt(arr.size // 3))) if arr.size % 3 == 0 else 0
        if side > 0 and side * side * 3 == arr.size:
            arr = arr.reshape(side, side, 3)
        else:
            raise ValueError(f"cannot reshape 1D image buffer, shape={arr.shape}")
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] >= 4:
            arr = arr[..., :3]
    else:
        raise ValueError(f"expected HxWx3 image, got shape {arr.shape}")

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 image, got shape {arr.shape}")
    return np.ascontiguousarray(arr.astype(np.uint8))


def rgb_from_normal_map_path(normal_map_path: str) -> np.ndarray:
    """将 BNI 用的 normal_map 图像转为 Segformer 输入 RGB (H,W,3)。"""
    raw = cv2.imread(normal_map_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"failed to read normal map: {normal_map_path}")
    if raw.ndim == 2:
        rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(np.asarray(raw)[..., :3], cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)


def segformer_predict_parsing(
    rgb_uint8: np.ndarray,
    image_processor,
    model,
    device: str,
) -> np.ndarray:
    """RGB uint8 (H,W,3) → Segformer 语义标签 (H,W) uint8。"""
    import torch.nn.functional as F

    rgb_uint8 = pil_or_array_to_hwc_rgb_uint8(rgb_uint8)
    H, W = int(rgb_uint8.shape[0]), int(rgb_uint8.shape[1])
    img = Image.fromarray(rgb_uint8, mode="RGB")
    inputs = image_processor(images=[img], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        upsampled = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False,
        )
        parsing = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    return parsing


def ensure_bni_segformer_face_masks(
    normal_files: List[str],
    dataset_dir: str,
    seg_dir: str,
    *,
    segformer_model_dir: Optional[str] = None,
    skip_existing: bool = True,
    rgb_loader: Optional[Callable[[str, str], Optional[np.ndarray]]] = None,
) -> Dict[str, str]:
    """
    为每条 normal_map 生成 Segformer 脸部分区，保存到 ``seg_dir``：
    ``{stem}.png``（语义标签）、``{stem}_face_mask.png``（脸部 mask）。

    默认用 ``dataset_dir`` 下 normal_map 转 RGB；若提供 ``rgb_loader(stem, normal_path)``
    则改用外部 RGB（如 FFHQ 对齐图）。

    Returns:
        仅包含已成功写入 face_mask 的项：normal_map_name -> face_mask_path
    """
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from tqdm import tqdm

    dataset_dir = osp.abspath(dataset_dir)
    seg_dir = osp.abspath(seg_dir)
    os.makedirs(seg_dir, exist_ok=True)

    if segformer_model_dir is None:
        segformer_model_dir = str(
            _REPO / "external" / "jonathandinu-face-parsing"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    need_infer: List[Tuple[str, str, str]] = []
    mask_paths: Dict[str, str] = {}
    num_skipped = 0

    for normal_map_name in normal_files:
        stem = stem_from_normal_map_filename(normal_map_name)
        normal_path = osp.join(dataset_dir, normal_map_name)
        parsing_path = osp.join(seg_dir, f"{stem}.png")
        face_mask_path = osp.join(seg_dir, f"{stem}_face_mask.png")

        if skip_existing and osp.isfile(face_mask_path) and osp.isfile(parsing_path):
            mask_paths[normal_map_name] = face_mask_path
            num_skipped += 1
            continue

        if not osp.isfile(normal_path):
            print(f"[seg] skip {stem}: normal_map not found: {normal_path}", flush=True)
            continue

        need_infer.append((normal_map_name, stem, normal_path))

    num_saved = 0
    if need_infer:
        image_processor = SegformerImageProcessor.from_pretrained(
            segformer_model_dir, local_files_only=True,
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            segformer_model_dir, local_files_only=True,
        )
        model.to(device)
        model.eval()

        for normal_map_name, stem, normal_path in tqdm(need_infer, desc="Segformer face"):
            try:
                if rgb_loader is not None:
                    rgb = rgb_loader(stem, normal_path)
                else:
                    rgb = rgb_from_normal_map_path(normal_path)
            except Exception as e:
                print(f"[seg] skip {stem}: load rgb failed: {e}", flush=True)
                continue

            if rgb is None:
                print(f"[seg] skip {stem}: rgb_loader returned None", flush=True)
                continue

            parsing = segformer_predict_parsing(rgb, image_processor, model, device)
            face_mask = face_mask_from_parsing(parsing)

            parsing_path = osp.join(seg_dir, f"{stem}.png")
            face_mask_path = osp.join(seg_dir, f"{stem}_face_mask.png")
            cv2.imwrite(parsing_path, parsing)
            cv2.imwrite(face_mask_path, (face_mask.astype(np.uint8) * 255))
            mask_paths[normal_map_name] = face_mask_path
            num_saved += 1

    print(
        f"[seg] saved={num_saved}, skipped={num_skipped}, "
        f"pending_failed={len(need_infer) - num_saved}, total={len(normal_files)}",
        flush=True,
    )
    return mask_paths


def make_ffhq_lmk_render_cfg() -> Dict[str, object]:
    image_size = PHACK_RENDER_IMAGE_SIZE
    focal = PHACK_RENDER_FOCAL
    center = image_size / 2.0
    return {
        "image_size": float(image_size),
        "focal": float(focal),
        "center": float(center),
        "fov": float(2 * np.arctan(center / focal) * 180 / np.pi),
        "camera_distance": 10.0,
        "znear": 5.0,
        "zfar": 15.0,
        "cx": float(center),
        "cy": float(center),
        "fx": float(focal),
        "fy": float(focal),
        "t_eps": 1e-5,
        "landmarker_path": f"{BASE_DIR}/assets/face_landmarker.task",
    }


_make_ffhq_lmk_render_cfg = make_ffhq_lmk_render_cfg


def stem_from_image_path(image_path: str) -> str:
    return Path(image_path).stem.split("_")[0]


_stem_from_image_path = stem_from_image_path


def image_paths_from_lmks_dir(
    lmks_dir: str,
    images_dir: str,
    lmks_suffix: str = "_bni_lmks.npz",
    image_ext: str = ".png",
) -> List[str]:
    """根据 ``lmks_dir`` 中的 ``{stem}_bni_lmks.npz`` 文件，收集对应图像路径。"""
    lmks_dir = os.path.abspath(lmks_dir)
    images_dir = os.path.abspath(images_dir)
    image_paths: List[str] = []
    for name in sorted(os.listdir(lmks_dir)):
        if not name.endswith(lmks_suffix):
            continue
        stem = name[: -len(lmks_suffix)]
        image_paths.append(osp.join(images_dir, stem + image_ext))
    return image_paths


def stems_from_lmks_dir(
    lmks_dir: str,
    lmks_suffix: str = "_bni_lmks.npz",
) -> List[str]:
    """根据 ``lmks_dir`` 中的 ``{stem}_bni_lmks.npz`` 文件名，返回 stem 列表。"""
    lmks_dir = os.path.abspath(lmks_dir)
    if not osp.isdir(lmks_dir):
        raise FileNotFoundError(f"lmks_dir not found: {lmks_dir}")
    stems: List[str] = []
    for name in sorted(os.listdir(lmks_dir)):
        if name.endswith(lmks_suffix):
            stems.append(name[: -len(lmks_suffix)])
    return stems


_image_paths_from_lmks_dir = image_paths_from_lmks_dir
_stems_from_lmks_dir = stems_from_lmks_dir


def bni_lmks_npz_to_lm478(lmks_npz_path: str) -> np.ndarray:
    """将 ``bni_lmks`` npz 中的 ``vertex_coords`` 整理为 ``(478, 3)`` mediapipe 坐标。"""
    with np.load(lmks_npz_path) as data:
        if "vertex_coords" not in data:
            raise KeyError(f"'vertex_coords' missing in {lmks_npz_path}")
        coords = np.asarray(data["vertex_coords"], dtype=np.float64).reshape(-1, 3)
        lmk_indices = (
            np.asarray(data["lmk_indices"], dtype=np.int64)
            if "lmk_indices" in data
            else np.arange(coords.shape[0], dtype=np.int64)
        )

    if coords.shape[0] == 0:
        raise ValueError(f"No landmarks stored in {lmks_npz_path}")
    if lmk_indices.shape[0] != coords.shape[0]:
        raise ValueError(
            f"lmk_indices length {lmk_indices.shape[0]} != vertex_coords rows {coords.shape[0]}"
        )

    lm478 = np.full((478, 3), np.nan, dtype=np.float64)
    lm478[lmk_indices] = coords
    return lm478


_bni_lmks_npz_to_lm478 = bni_lmks_npz_to_lm478


def extract_vertex_masked_submesh(
    verts: np.ndarray,
    faces: np.ndarray,
    vertex_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """提取三个顶点均在 ``vertex_mask`` 内的三角面子网格。"""
    vertex_mask = np.asarray(vertex_mask, dtype=bool).reshape(-1)
    if vertex_mask.shape[0] != verts.shape[0]:
        raise ValueError(
            f"vertex_mask length {vertex_mask.shape[0]} != num vertices {verts.shape[0]}"
        )

    face_keep = vertex_mask[faces].all(axis=1)
    if not np.any(face_keep):
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int64),
        )

    f_sub = faces[face_keep]
    used = np.unique(f_sub.reshape(-1))
    old_to_new = np.full(verts.shape[0], -1, dtype=np.int64)
    old_to_new[used] = np.arange(used.shape[0], dtype=np.int64)
    v_sub = np.ascontiguousarray(verts[used], dtype=np.float64)
    f_idx = np.ascontiguousarray(old_to_new[f_sub], dtype=np.int64)
    return v_sub, f_idx


# _extract_vertex_masked_submesh = extract_vertex_masked_submesh


def build_vertex_adjacency(n_verts: int, faces: np.ndarray) -> List[set]:
    adj: List[set] = [set() for _ in range(n_verts)]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return adj


def largest_connected_masked_component(
    vertex_mask: np.ndarray,
    adj: List[set],
) -> np.ndarray:
    """在 mask 顶点子图上保留最大连通分量，剔除孤立离群块。"""
    vertex_mask = vertex_mask.astype(bool, copy=False)
    masked_idx = np.flatnonzero(vertex_mask)
    if masked_idx.size == 0:
        return vertex_mask

    masked_set = set(int(i) for i in masked_idx)
    visited: set = set()
    largest: List[int] = []

    for start in masked_idx:
        start = int(start)
        if start in visited:
            continue
        component: List[int] = []
        stack = [start]
        visited.add(start)
        while stack:
            u = stack.pop()
            component.append(u)
            for v in adj[u]:
                if v in masked_set and v not in visited:
                    visited.add(v)
                    stack.append(v)
        if len(component) > len(largest):
            largest = component

    refined = np.zeros_like(vertex_mask, dtype=bool)
    refined[largest] = True
    return refined


def compute_mesh_vertex_mask_from_parsing(
    verts: np.ndarray,
    faces: np.ndarray,
    parsing: np.ndarray,
    render_cfg: Optional[Dict[str, object]] = None,
    face_labels: Optional[FrozenSet[int]] = None,
) -> np.ndarray:
    if render_cfg is None:
        render_cfg = make_ffhq_lmk_render_cfg()
    if face_labels is None:
        face_labels = _FACE_PARSE_LABELS

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    parsing = np.asarray(parsing)
    vertex_labels = np.full(verts.shape[0], -1, dtype=np.int32)

    camera_distance = float(render_cfg["camera_distance"])
    t_eps = float(render_cfg.get("t_eps", 1e-5))
    cx = float(render_cfg["cx"])
    cy = float(render_cfg["cy"])
    fx = float(render_cfg["fx"])
    fy = float(render_cfg["fy"])

    verts_render = verts.copy()
    verts_render[:, 2] = camera_distance - verts_render[:, 2]
    z = verts_render[:, 2]
    valid_z = z > t_eps

    u = fx * verts_render[:, 0] / z + cx
    v = fy * verts_render[:, 1] / z + cy
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)

    H, W = parsing.shape[:2]
    in_bounds = (
        valid_z
        & (ui >= 0) & (ui < W)
        & (vi >= 0) & (vi < H)
    )
    if np.any(in_bounds):
        vertex_labels[in_bounds] = parsing[vi[in_bounds], ui[in_bounds]].astype(np.int32)

    vertex_mask = np.isin(vertex_labels, list(face_labels))
    if not np.any(vertex_mask):
        return vertex_mask

    adj = build_vertex_adjacency(verts.shape[0], faces)
    return largest_connected_masked_component(vertex_mask, adj)


_compute_mesh_vertex_mask_from_parsing = compute_mesh_vertex_mask_from_parsing


def polyscope_show_ffhq_mesh_lmks(
    mesh_path: str,
    lmks_npz_path: Optional[str] = None,
    lmks_dir: str = _DEFAULT_BNI_LMKS_DIR,
    mesh_name: Optional[str] = None,
    lmks_name: str = "bni_lmks",
    point_radius_rel: float = 2e-3,
    point_radius_abs: Optional[float] = None,
    show_facial_regions_only: bool = False,
    facial_region_indices: Optional[np.ndarray] = None,
    coords_key: str = "hit_coords",
):
    """用 Polyscope 显示 FFHQ/BNI 网格与 landmarks。

    ``show_facial_regions_only=True`` 时仅显示 ``EYE_LIDS_INDIECS``、``LIPS_INDICES``、
    ``NOSE_INDICES`` 并集中的 mediapipe 点（按 npz 内 ``lmk_indices`` 过滤）。
    """
    mesh_path = osp.abspath(mesh_path)
    if not osp.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    stem = Path(mesh_path).stem.split("_")[0]
    if lmks_npz_path is None:
        lmks_npz_path = osp.join(lmks_dir, f"{stem}_bni_lmks.npz")
    lmks_npz_path = osp.abspath(lmks_npz_path)
    if not osp.isfile(lmks_npz_path):
        raise FileNotFoundError(f"Landmarks npz not found: {lmks_npz_path}")

    mesh = trimesh.load(mesh_path, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    with np.load(lmks_npz_path) as data:
        if coords_key not in data:
            if coords_key == "hit_coords" and "vertex_coords" in data:
                print(
                    f"[polyscope] warning: 'hit_coords' missing in {lmks_npz_path}; "
                    "falling back to 'vertex_coords'. Regenerate bni_lmks to show "
                    "ray-mesh intersections.",
                    flush=True,
                )
                coords_key = "vertex_coords"
            else:
                raise KeyError(f"'{coords_key}' missing in {lmks_npz_path}")
        lmks = np.asarray(data[coords_key], dtype=np.float64).reshape(-1, 3)
        lmk_indices = (
            np.asarray(data["lmk_indices"], dtype=np.int64)
            if "lmk_indices" in data
            else np.arange(lmks.shape[0], dtype=np.int64)
        )

    if lmks.shape[0] == 0:
        raise ValueError(f"No landmarks stored in {lmks_npz_path}")
    if lmk_indices.shape[0] != lmks.shape[0]:
        raise ValueError(
            f"lmk_indices length {lmk_indices.shape[0]} != vertex_coords rows {lmks.shape[0]}"
        )

    ok = np.all(np.isfinite(lmks), axis=1)
    if show_facial_regions_only:
        if facial_region_indices is None:
            facial_region_indices = FACIAL_REGION_LMK_INDICES
        region_ids = set(np.asarray(facial_region_indices, dtype=np.int64).reshape(-1).tolist())
        ok = ok & np.isin(lmk_indices, list(region_ids))

    lmks_vis = lmks[ok]
    idx_vis = lmk_indices[ok].astype(np.float64)
    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    rad_rel = max(diag * float(point_radius_rel), 1e-6)
    rad = float(point_radius_abs) if point_radius_abs is not None else rad_rel

    if mesh_name is None:
        mesh_name = stem

    ps.init()
    ps.set_program_name("ffhq_mesh_lmks")
    ps.set_ground_plane_mode("none")
    ps.register_surface_mesh(
        mesh_name,
        verts,
        faces,
        color=(0.9, 0.9, 0.9),
        edge_width=1.0,
        material="clay",
        smooth_shade=False,
    )
    if lmks_vis.shape[0] == 0:
        print(
            f"[polyscope] 警告: 无符合条件的 landmarks，仅显示网格 "
            f"(facial_regions_only={show_facial_regions_only})",
            flush=True,
        )
    else:
        pcs = ps.register_point_cloud(
            lmks_name,
            lmks_vis,
            radius=max(rad, 1e-5),
            color=(0.95, 0.05, 0.05),
        )
        pcs.add_scalar_quantity("lmk_idx", idx_vis, enabled=False)
    print(
        f"[polyscope] FFHQ mesh V={verts.shape[0]} F={faces.shape[0]} | "
        f"lmks {lmks_vis.shape[0]}/{lmks.shape[0]} | coords_key={coords_key} | "
        f"facial_regions_only={show_facial_regions_only}",
        flush=True,
    )
    ps.show()



def polyscope_show_hack_mesh_fitting_lmks(
    verts: np.ndarray,
    faces: np.ndarray,
    fitting_indices: np.ndarray = HACK_FITTING_INDICES,
    hack_lmks3d_path: str = HACK_LMKS3D_PATH,
    mesh_name: str = "hack_raw",
    lmks_name: str = "hack_fitting_lmks",
    point_radius_rel: float = 2e-3,
    point_radius_abs: Optional[float] = None,
    device: str = "cpu",
):
    """用 Polyscope 显示 HACK 网格及拟合用 landmarks（``vertices2landmarks`` + ``HACK_FITTING_INDICES``）。"""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    idx = np.asarray(fitting_indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise ValueError("fitting_indices is empty")
    if idx.min() < 0 or idx.max() >= 478:
        raise ValueError(
            f"fitting_indices must index 478 mediapipe landmarks, got "
            f"min={idx.min()}, max={idx.max()}"
        )

    dev = torch.device(device)
    dtype = torch.float32
    with np.load(hack_lmks3d_path) as nz:
        lmk_faces_indices = torch.as_tensor(
            nz["hack_indices"], dtype=torch.long, device=dev,
        )
        lmk_bary_coords = torch.as_tensor(
            nz["hack_bary_coords"], dtype=dtype, device=dev,
        )

    v_t = torch.as_tensor(verts, dtype=dtype, device=dev).unsqueeze(0)
    f_t = torch.as_tensor(faces, dtype=torch.long, device=dev)
    lm_interp = vertices2landmarks(
        vertices=v_t,
        faces=f_t,
        lmk_faces_indices=lmk_faces_indices.unsqueeze(0),
        lmk_bary_coords=lmk_bary_coords.unsqueeze(0),
    )[0].detach().cpu().numpy()

    lm478 = np.full((478, 3), np.nan, dtype=np.float64)
    lm478[np.asarray(valid_lmks_ids, dtype=np.int64)] = lm_interp
    lmks_all = lm478[idx]
    ok = np.all(np.isfinite(lmks_all), axis=1)
    lmks_vis = lmks_all[ok]
    idx_ok = idx[ok].astype(np.float64)

    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    rad_rel = max(diag * float(point_radius_rel), 1e-6)
    rad = float(point_radius_abs) if point_radius_abs is not None else rad_rel

    ps.init()
    ps.set_program_name("hack_mesh_fitting_lmks")
    ps.set_ground_plane_mode("none")
    ps.register_surface_mesh(
        mesh_name,
        verts,
        faces,
        color=(0.9, 0.9, 0.9),
        edge_width=1.0,
        material="clay",
        smooth_shade=False,
    )
    if lmks_vis.shape[0] == 0:
        print(
            f"[polyscope] 警告: 无数有限 landmarks，仅显示网格 "
            f"(V={verts.shape[0]} F={faces.shape[0]})",
            flush=True,
        )
    else:
        pcs = ps.register_point_cloud(
            lmks_name,
            lmks_vis,
            radius=max(rad, 1e-5),
            color=(0.95, 0.05, 0.05),
        )
        pcs.add_scalar_quantity("lmk478_idx", idx_ok, enabled=False)
    print(
        f"[polyscope] HACK raw | mesh V={verts.shape[0]} F={faces.shape[0]} | "
        f"finite fitting lmks {int(ok.sum())}/{idx.shape[0]} (vertices2landmarks)",
        flush=True,
    )
    ps.show()


def polyscope_show_tgt_mesh_bni_lmks(
    mesh_path: str,
    lmks_npz_path: Optional[str] = None,
    lmks_dir: str = _DEFAULT_BNI_LMKS_DIR,
    fitting_indices: np.ndarray = HACK_FITTING_INDICES,
    mesh_name: Optional[str] = None,
    lmks_name: str = "tgt_bni_fitting_lmks",
    point_radius_rel: float = 2e-3,
    point_radius_abs: Optional[float] = None,
):
    """用 Polyscope 显示 target 网格及 ``bni_lmks`` 中对应拟合 landmarks。"""
    mesh_path = osp.abspath(mesh_path)
    if not osp.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    stem = Path(mesh_path).stem.split("_")[0]
    if lmks_npz_path is None:
        lmks_npz_path = osp.join(lmks_dir, f"{stem}_bni_lmks.npz")
    lmks_npz_path = osp.abspath(lmks_npz_path)
    if not osp.isfile(lmks_npz_path):
        raise FileNotFoundError(f"Landmarks npz not found: {lmks_npz_path}")

    mesh = trimesh.load(mesh_path, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    idx = np.asarray(fitting_indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise ValueError("fitting_indices is empty")
    if idx.min() < 0 or idx.max() >= 478:
        raise ValueError(
            f"fitting_indices must index 478 mediapipe landmarks, got "
            f"min={idx.min()}, max={idx.max()}"
        )

    lm478 = bni_lmks_npz_to_lm478(lmks_npz_path)
    lmks_all = lm478[idx]
    ok = np.all(np.isfinite(lmks_all), axis=1)
    lmks_vis = lmks_all[ok]
    idx_ok = idx[ok].astype(np.float64)

    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    rad_rel = max(diag * float(point_radius_rel), 1e-6)
    rad = float(point_radius_abs) if point_radius_abs is not None else rad_rel

    if mesh_name is None:
        mesh_name = stem

    ps.init()
    ps.set_program_name("tgt_mesh_bni_lmks")
    ps.set_ground_plane_mode("none")
    ps.register_surface_mesh(
        mesh_name,
        verts,
        faces,
        color=(0.82, 0.86, 0.9),
        edge_width=1.0,
        material="clay",
        smooth_shade=False,
    )
    if lmks_vis.shape[0] == 0:
        print(
            f"[polyscope] 警告: 无数有限 landmarks，仅显示 target 网格 "
            f"(V={verts.shape[0]} F={faces.shape[0]})",
            flush=True,
        )
    else:
        pcs = ps.register_point_cloud(
            lmks_name,
            lmks_vis,
            radius=max(rad, 1e-5),
            color=(0.05, 0.55, 0.95),
        )
        pcs.add_scalar_quantity("lmk478_idx", idx_ok, enabled=False)
    print(
        f"[polyscope] target BNI | mesh V={verts.shape[0]} F={faces.shape[0]} | "
        f"finite fitting lmks {int(ok.sum())}/{idx.shape[0]} | npz={lmks_npz_path}",
        flush=True,
    )
    ps.show()


def build_face_colors_from_vertex_mask(
    faces: np.ndarray,
    vertex_mask: np.ndarray,
    *,
    mask_color: Tuple[float, float, float] = (0.839, 0.153, 0.157),
    unmasked_color: Tuple[float, float, float] = (0.75, 0.75, 0.78),
    boundary_color: Tuple[float, float, float] = (1.0, 0.733, 0.055),
) -> Tuple[np.ndarray, np.ndarray]:
    """根据顶点 mask 生成逐面 RGB 颜色与面片类别（0=外部, 1=边界, 2=内部）。"""
    vertex_mask = np.asarray(vertex_mask, dtype=bool).reshape(-1)
    faces = np.asarray(faces, dtype=np.int64)

    n_faces = int(faces.shape[0])
    masked_on_face = vertex_mask[faces]
    n_masked = masked_on_face.sum(axis=1)

    face_class = np.zeros(n_faces, dtype=np.int32)
    face_class[n_masked == 3] = 2
    face_class[(n_masked > 0) & (n_masked < 3)] = 1

    face_rgb = np.tile(np.asarray(unmasked_color, dtype=np.float64), (n_faces, 1))
    face_rgb[face_class == 2] = np.asarray(mask_color, dtype=np.float64)
    face_rgb[face_class == 1] = np.asarray(boundary_color, dtype=np.float64)
    return face_rgb, face_class


def polyscope_show_mesh_vertex_mask(
    mesh_path: str,
    mesh_mask_npz_path: Optional[str] = None,
    mesh_mask_dir: str = _DEFAULT_BNI_SEG_DIR,
    mesh_name: Optional[str] = None,
    color: Tuple[float, float, float] = (0.839, 0.153, 0.157),
    unmasked_color: Tuple[float, float, float] = (0.75, 0.75, 0.78),
    boundary_color: Tuple[float, float, float] = (1.0, 0.733, 0.055),
    show_edge: bool = True,
):
    """用 Polyscope 显示完整网格，并按 ``vertex_mask`` 着色区分 mask 区域。

    - 三个顶点均在 mask 内：``color``（默认红）
    - 部分顶点在 mask 内：``boundary_color``（默认橙，边界）
    - 其余面片：``unmasked_color``（默认灰）
    """
    mesh_path = osp.abspath(mesh_path)
    if not osp.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    stem = Path(mesh_path).stem.split("_")[0]
    if mesh_mask_npz_path is None:
        mesh_mask_npz_path = osp.join(mesh_mask_dir, f"{stem}_mesh_mask.npz")
    mesh_mask_npz_path = osp.abspath(mesh_mask_npz_path)
    if not osp.isfile(mesh_mask_npz_path):
        raise FileNotFoundError(f"Mesh mask npz not found: {mesh_mask_npz_path}")

    mesh = trimesh.load(mesh_path, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    with np.load(mesh_mask_npz_path) as data:
        if "vertex_mask" not in data:
            raise KeyError(f"'vertex_mask' missing in {mesh_mask_npz_path}")
        vertex_mask = np.asarray(data["vertex_mask"], dtype=bool).reshape(-1)

    if vertex_mask.shape[0] != verts.shape[0]:
        raise ValueError(
            f"vertex_mask length {vertex_mask.shape[0]} != num vertices {verts.shape[0]}"
        )

    face_rgb, face_class = build_face_colors_from_vertex_mask(
        faces,
        vertex_mask,
        mask_color=color,
        unmasked_color=unmasked_color,
        boundary_color=boundary_color,
    )
    n_inside = int(np.sum(face_class == 2))
    n_boundary = int(np.sum(face_class == 1))
    n_outside = int(np.sum(face_class == 0))
    if n_inside == 0:
        num_true = int(np.sum(vertex_mask))
        raise ValueError(
            f"No face has all three vertices masked in {mesh_mask_npz_path} "
            f"(true vertices={num_true}/{verts.shape[0]})"
        )

    if mesh_name is None:
        mesh_name = stem

    print(
        f"[polyscope] mesh_vertex_mask | mesh V={verts.shape[0]} F={faces.shape[0]} | "
        f"inside={n_inside}, boundary={n_boundary}, outside={n_outside} | "
        f"npz={mesh_mask_npz_path}",
        flush=True,
    )

    ps.init()
    ps.set_program_name("mesh_vertex_mask")
    ps.set_ground_plane_mode("none")
    ps_mesh = ps.register_surface_mesh(
        mesh_name,
        verts,
        faces,
        color=(0.85, 0.85, 0.85),
        edge_width=0.85 if show_edge else 0.0,
        material="clay",
        smooth_shade=False,
        back_face_policy="custom",
    )
    ps_mesh.add_color_quantity(
        "vertex_mask_region",
        face_rgb,
        defined_on="faces",
        enabled=True,
    )
    ps.show()


