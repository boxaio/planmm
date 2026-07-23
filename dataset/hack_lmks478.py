from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import mediapipe as mp
import numpy as np
import pyrender
import torch
import trimesh
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from trimesh.triangles import points_to_barycentric

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET_DIR = Path(__file__).resolve().parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from render.mesh_render import render_mesh
from utils.mediapipe_landmarks import (
    LEFT_EYE_LANDMARK_IDS,
    LEFT_IRIS_LANDMARK_IDS,
    RIGHT_EYE_LANDMARK_IDS,
    RIGHT_IRIS_LANDMARK_IDS,
)
from dataset.hack_seg_fids import load_region_info_pkl

N_LANDMARKS = 478

DEFAULT_MESH_FILES: tuple[str, ...] = (
    "hack_template.obj",
    "hack_01173.000016_fit_fine.obj",
    "hack_01200.000011_fit_fine.obj",
    "hack_01200.000025_fit_fine.obj",
)

DEFAULT_LANDMARKER_TASK = _DATASET_DIR / "face_landmarker.task"

# inner eye region and inner mouth region
region_info = load_region_info_pkl()


def run_mediapipe(
    face_detector: vision.FaceLandmarker,
    image_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """``image_rgb``: HWC uint8 RGB."""
    h, w = image_rgb.shape[:2]
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = face_detector.detect(image)
    if len(detection_result.face_landmarks) == 0:
        return None
    landmarks = detection_result.face_landmarks[0]
    n = len(landmarks)
    face_landmarks_imgs = np.zeros((n, 3), dtype=np.float64)
    face_landmarks = np.zeros((n, 3), dtype=np.float64)
    for i, lm in enumerate(landmarks):
        face_landmarks_imgs[i] = [lm.x * w, lm.y * h, lm.z]
        face_landmarks[i] = [lm.x, lm.y, lm.z]
    return face_landmarks_imgs, face_landmarks


def build_face_landmarker(model_path: Path) -> vision.FaceLandmarker:
    if not model_path.is_file():
        raise FileNotFoundError(f"Face landmarker task not found: {model_path}")
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def mediapipe_landmarks_pixels_batch(
    rgb_images: Sequence[np.ndarray],
    model_path: Path = DEFAULT_LANDMARKER_TASK,
    *,
    face_landmarker: vision.FaceLandmarker | None = None,
) -> np.ndarray:
    """``(K, 478, 2)`` 像素 (u,v)，未检出则为 ``nan``.

    若传入 ``face_landmarker``，则不再每次调用 ``build_face_landmarker``（适合大批量 PLY）。
    """
    det = (
        face_landmarker
        if face_landmarker is not None
        else build_face_landmarker(model_path)
    )
    K = len(rgb_images)
    out = np.full((K, N_LANDMARKS, 2), np.nan, dtype=np.float64)
    for k, rgb in enumerate(rgb_images):
        if rgb.dtype != np.uint8:
            rgb = (
                np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                if rgb.max() <= 1.0
                else rgb.astype(np.uint8)
            )
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected HWC RGB, got {rgb.shape}")
        res = run_mediapipe(det, rgb)
        if res is None:
            continue
        lm_img, _ = res
        if lm_img.shape[0] != N_LANDMARKS:
            raise RuntimeError(
                f"landmark count {lm_img.shape[0]} != expected {N_LANDMARKS}"
            )
        out[k, :, 0] = lm_img[:, 0]
        out[k, :, 1] = lm_img[:, 1]
    return out


def _load_obj_vf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from utils.mesh import read_obj

    obj = read_obj(str(path), tri=True)
    verts = np.asarray(obj.vs, dtype=np.float64)
    faces = np.asarray(obj.fvs, dtype=np.int64)
    return verts, faces


def _canonical_camera_bundle(
    R_max: float,
    *,
    yfov: float,
    fill_ratio: float,
) -> tuple[float, float, float, np.ndarray]:
    half = yfov * 0.5
    if R_max < 1e-8 or not math.isfinite(R_max):
        R_max = 1.0
    d = R_max / (fill_ratio * math.tan(half))
    znear = max(0.01, d - R_max * 3.0)
    zfar = max(znear + 0.1, d + R_max * 6.0, 50.0)
    cam_pose = np.eye(4, dtype=np.float64)
    cam_pose[2, 3] = float(d)
    return float(d), float(znear), float(zfar), cam_pose


def pyrender_projection_matrix(
    *,
    yfov: float,
    znear: float,
    zfar: float,
    width: int,
    height: int,
) -> np.ndarray:
    """与渲染时一致的 OpenGL/glTF 透视矩阵（参见 ``pyrender.PerspectiveCamera.get_projection_matrix``）。"""
    cam = pyrender.PerspectiveCamera(
        yfov=float(yfov), znear=float(znear), zfar=float(zfar)
    )
    return np.asarray(
        cam.get_projection_matrix(width=int(width), height=int(height)),
        dtype=np.float64,
    )


def world_points_to_pixel_uv_openvr_top_left_image(
    cam_pose_world: np.ndarray,
    P_gl: np.ndarray,
    xyz_w: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    世界坐标 ``(N,3)`` → 与 ``pixel_ray_opengl_top_left_origin`` **互逆** 的像素 (u,v)（左上角原点、向下为 +v）。
    ``cam_pose_world``：相机坐标系→世界坐标系（同 pyrender Scene ``get_pose``）；``V = inv(cam_pose_world)``.
    Returns
    -------
    uv : (N, 2)
    visible : (N,) bool — 点在相机前方（相机空间 ``z_cam < 0``）且在有限投影内。
    """
    xyz_w = np.asarray(xyz_w, dtype=np.float64).reshape(-1, 3)
    n = xyz_w.shape[0]
    V_w2c = np.linalg.inv(cam_pose_world)
    ph = np.ones((n, 4), dtype=np.float64)
    ph[:, :3] = xyz_w
    p_cam = (V_w2c @ ph.T).T
    zc = p_cam[:, 2]
    visible_cam = zc < -1e-9

    clip = (P_gl @ p_cam.T).T
    w = clip[:, 3]
    fin = visible_cam & (np.abs(w) > 1e-12)
    uv = np.full((n, 2), np.nan, dtype=np.float64)

    ndc_x = clip[:, 0] / w
    ndc_y = clip[:, 1] / w

    uv[:, 0] = np.where(fin, width * (ndc_x + 1.0) * 0.5 - 0.5, np.nan)
    uv[:, 1] = np.where(fin, height * (1.0 - ndc_y) * 0.5 - 0.5, np.nan)
    uv[~fin] = np.nan
    return uv, fin


def reprojection_residuals_px_vs_mediapipe(
    *,
    lm_uv_mp: np.ndarray,
    lm_uv_repr: np.ndarray,
) -> np.ndarray:
    """``(478,)`` 欧氏误差（像素），任一分量为 nan 则为 nan."""
    dv = lm_uv_repr - lm_uv_mp
    return np.sqrt(np.sum(np.square(dv), axis=-1))



def fuse_hits_prioritize_reference_view_then_median_reference_frames(
    hits_per_view: np.ndarray,
    ref_view_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ``hits_per_view`` (K, L, 3)。优先用 ``ref_view_index`` 视图上的击中点（与 ``hack_template`` 几何对齐）；
    若该视图无击中，再在 **其余视图**上对有效击中取中位数（避免因不同 OBJ 体型做全体中位数）。
    Returns
    -------
    P : (L, 3); source_view : (L,) — 若为 ref 则用 ``ref_view_index``, 若为备用中位数则 ``-1``。
    """
    K, L, _ = hits_per_view.shape
    P = np.zeros((L, 3), dtype=np.float64)
    src = np.full(L, -1, dtype=np.int64)
    for k in range(L):
        pref = hits_per_view[ref_view_index, k]
        if np.all(np.isfinite(pref)):
            P[k] = pref
            src[k] = ref_view_index
            continue
        alts = [hits_per_view[v, k] for v in range(K) if v != ref_view_index]
        stacked = []
        for p in alts:
            if np.all(np.isfinite(p)):
                stacked.append(p)
        if stacked:
            P[k] = np.median(np.stack(stacked, axis=0), axis=0)
            src[k] = -1
        else:
            P[k] = np.nan
    return P, src


def pixel_ray_opengl_top_left_origin(
    u: float,
    v: float,
    width: int,
    height: int,
    yfov: float,
    aspect: float,
    cam_pose_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    左上角为原点、``v`` 向下；与 pyrender flip 后的 numpy 图像/MediaPipe `x*w,y*h` 一致。
    相机局部视线沿 ``-Z``；``aspect = 宽度/高度``（与 ``PerspectiveCamera.get_projection_matrix(W,H)`` 一致）。
    """
    x_ndc = (2.0 * (float(u) + 0.5) / float(width)) - 1.0
    y_ndc = 1.0 - (2.0 * (float(v) + 0.5) / float(height))
    tan_h = math.tan(float(yfov) * 0.5)
    dir_c = np.array([x_ndc * tan_h * aspect, y_ndc * tan_h, -1.0], dtype=np.float64)
    dir_c /= max(np.linalg.norm(dir_c), 1e-12)
    R = cam_pose_world[:3, :3]
    origin = cam_pose_world[:3, 3].copy()
    direction = R @ dir_c
    direction /= max(np.linalg.norm(direction), 1e-12)
    return origin, direction


def ray_mesh_first_positive_hit(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    t_eps: float = 1e-5,
) -> tuple[np.ndarray, int] | None:
    o = np.asarray(origin, dtype=np.float64).reshape(1, 3)
    direc = np.asarray(direction, dtype=np.float64).reshape(3,)
    direc /= max(np.linalg.norm(direc), 1e-12)
    locs, _ray_i, tri_ids = mesh.ray.intersects_location(
        o, direc.reshape(1, 3), multiple_hits=True
    )
    if len(locs) == 0:
        return None
    tvals = np.einsum("ij,j->i", locs - o, direc)
    mask = tvals > t_eps
    if not np.any(mask):
        return None
    jj = int(np.argmin(tvals[mask]))
    idx = int(np.flatnonzero(mask)[jj])
    return locs[idx].astype(np.float64), int(tri_ids[idx])


def _eye_inner_fid_set(region_info_dict: dict) -> frozenset[int]:
    fids = np.concatenate(
        [
            np.asarray(region_info_dict["eye_0_inner"]["fids"], dtype=np.int64),
            np.asarray(region_info_dict["eye_1_inner"]["fids"], dtype=np.int64),
        ]
    )
    return frozenset(int(x) for x in np.unique(fids))


def _eye_shell_fid_sets(
    region_info_dict: dict, inner_fids: frozenset[int]
) -> tuple[frozenset[int], frozenset[int], frozenset[int]]:
    """``eye_0``/``eye_1`` 区域去掉 ``eye_inner`` 后的外侧眼眶三角面。"""
    e0 = frozenset(
        int(x)
        for x in np.asarray(region_info_dict["eye_0"]["fids"], dtype=np.int64)
        if int(x) not in inner_fids
    )
    e1 = frozenset(
        int(x)
        for x in np.asarray(region_info_dict["eye_1"]["fids"], dtype=np.int64)
        if int(x) not in inner_fids
    )
    return e0, e1, frozenset(e0 | e1)


def _iris_landmark_index_set() -> frozenset[int]:
    return frozenset(
        int(i)
        for i in np.concatenate([LEFT_IRIS_LANDMARK_IDS, RIGHT_IRIS_LANDMARK_IDS])
    )


def _left_eye_landmark_index_set() -> frozenset[int]:
    return frozenset(int(i) for i in LEFT_EYE_LANDMARK_IDS)


def _right_eye_landmark_index_set() -> frozenset[int]:
    return frozenset(int(i) for i in RIGHT_EYE_LANDMARK_IDS)


def _allowed_eye_shell_for_landmark(
    landmark_idx: int,
    eye0_shell: frozenset[int],
    eye1_shell: frozenset[int],
    eye_shell_both: frozenset[int],
) -> frozenset[int]:
    if landmark_idx in _left_eye_landmark_index_set():
        return eye0_shell
    if landmark_idx in _right_eye_landmark_index_set():
        return eye1_shell
    return eye_shell_both


def _triangle_normal(mesh: trimesh.Trimesh, face_id: int) -> np.ndarray:
    v = mesh.vertices[mesh.faces[int(face_id)]]
    e1 = v[1] - v[0]
    e2 = v[2] - v[0]
    n = np.cross(e1, e2)
    nrm = float(np.linalg.norm(n))
    if nrm < 1e-12:
        return n
    return n / nrm


def _hit_faces_camera(
    mesh: trimesh.Trimesh,
    face_id: int,
    hit_point: np.ndarray,
    ray_origin: np.ndarray,
) -> bool:
    """交点三角面朝向相机（避免落到后脑勺等背向面）。"""
    n = _triangle_normal(mesh, face_id)
    to_cam = np.asarray(ray_origin, dtype=np.float64).reshape(3) - np.asarray(
        hit_point, dtype=np.float64
    ).reshape(3)
    ntc = float(np.linalg.norm(to_cam))
    if ntc < 1e-12:
        return True
    return float(np.dot(n, to_cam / ntc)) > 0.05


def _ray_hits_along_direction(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    t_eps: float = 1e-5,
) -> list[tuple[np.ndarray, int, float]]:
    """沿射线返回 (交点, 三角面 id, t) 列表，按 t 升序。"""
    o = np.asarray(origin, dtype=np.float64).reshape(1, 3)
    direc = np.asarray(direction, dtype=np.float64).reshape(3,)
    nrm = float(np.linalg.norm(direc))
    if nrm < 1e-12:
        return []
    direc = direc / nrm
    locs, _, tri_ids = mesh.ray.intersects_location(
        o, direc.reshape(1, 3), multiple_hits=True
    )
    if len(locs) == 0:
        return []
    tvals = np.einsum("ij,j->i", locs - o.reshape(3), direc)
    out: list[tuple[np.ndarray, int, float]] = []
    for j in np.argsort(tvals):
        t = float(tvals[j])
        if t <= t_eps:
            continue
        out.append((locs[j].astype(np.float64), int(tri_ids[j]), t))
    return out


def ray_mesh_first_hit_on_eye_shell(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    direction: np.ndarray,
    allowed_fids: frozenset[int],
    inner_fids: frozenset[int],
    *,
    t_eps: float = 1e-5,
) -> tuple[np.ndarray, int] | None:
    """
    沿射线取距相机最近、落在指定眼眶外侧三角面且朝向相机的交点。
    """
    ro = np.asarray(origin, dtype=np.float64).reshape(3)
    for pt, fid, _t in _ray_hits_along_direction(mesh, origin, direction, t_eps=t_eps):
        if fid in inner_fids or fid not in allowed_fids:
            continue
        if not _hit_faces_camera(mesh, fid, pt, ro):
            continue
        return pt, fid
    return None


def ray_mesh_hit_for_landmark(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    landmark_idx: int,
    inner_fids: frozenset[int],
    iris_ids: frozenset[int],
    eye0_shell: frozenset[int],
    eye1_shell: frozenset[int],
    eye_shell_both: frozenset[int],
    t_eps: float = 1e-5,
) -> tuple[np.ndarray, int] | None:
    """
    虹膜关键点保留首个射线交点；其余关键点若落在 ``eye_inner`` 上，
    则在同一条射线上于对应眼眶外侧网格（``eye_0``/``eye_1`` 去掉 inner）取最近交点。
    """
    hit = ray_mesh_first_positive_hit(mesh, origin, direction, t_eps=t_eps)
    if hit is None:
        return None
    if landmark_idx in iris_ids:
        return hit
    pt, fid = hit
    if fid not in inner_fids:
        return hit
    allowed = _allowed_eye_shell_for_landmark(
        landmark_idx, eye0_shell, eye1_shell, eye_shell_both
    )
    shell_hit = ray_mesh_first_hit_on_eye_shell(
        mesh, origin, direction, allowed, inner_fids, t_eps=t_eps
    )
    if shell_hit is not None:
        return shell_hit
    return _closest_hit_on_eye_shell(mesh, pt, allowed)


def _closest_hit_on_eye_shell(
    mesh: trimesh.Trimesh,
    point: np.ndarray,
    allowed_fids: frozenset[int],
) -> tuple[np.ndarray, int] | None:
    """射线未击中眼眶外侧时，在眼眶外侧子网格上求最近点。"""
    if not allowed_fids:
        return None
    keep = np.array(sorted(allowed_fids), dtype=np.int64)
    sub = mesh.submesh([keep], append=True, only_watertight=False)
    cp, _, tri_local = trimesh.proximity.closest_point(
        sub, np.asarray(point, dtype=np.float64).reshape(1, 3)
    )
    fid = int(keep[int(tri_local[0])])
    return cp[0].astype(np.float64), fid


def _relocate_hit_off_eye_inner(
    mesh: trimesh.Trimesh,
    point: np.ndarray,
    *,
    inner_fids: frozenset[int],
    landmark_idx: int,
    iris_ids: frozenset[int],
    eye0_shell: frozenset[int],
    eye1_shell: frozenset[int],
    eye_shell_both: frozenset[int],
    cam_pose: np.ndarray,
    P_gl: np.ndarray,
    yfov: float,
    width: int,
    height: int,
    aspect: float,
) -> np.ndarray:
    """将已落在 eye_inner 上的 3D 点重定位到对应眼眶外侧网格。"""
    if landmark_idx in iris_ids:
        return np.asarray(point, dtype=np.float64).reshape(3)
    p = np.asarray(point, dtype=np.float64).reshape(3)
    allowed = _allowed_eye_shell_for_landmark(
        landmark_idx, eye0_shell, eye1_shell, eye_shell_both
    )
    uv, vis = world_points_to_pixel_uv_openvr_top_left_image(
        cam_pose, P_gl, p.reshape(1, 3), width=width, height=height
    )
    picked: tuple[np.ndarray, int] | None = None
    if bool(vis[0]) and np.all(np.isfinite(uv[0])):
        ro, rd = pixel_ray_opengl_top_left_origin(
            float(uv[0, 0]), float(uv[0, 1]), width, height, yfov, aspect, cam_pose
        )
        picked = ray_mesh_first_hit_on_eye_shell(
            mesh, ro, rd, allowed, inner_fids
        )
    if picked is not None:
        return picked[0]
    fb = _closest_hit_on_eye_shell(mesh, p, allowed)
    if fb is not None:
        return fb[0]
    return p


def relocate_fused_landmarks_off_eye_inner(
    P_world: np.ndarray,
    tmpl_mesh_c: trimesh.Trimesh,
    *,
    region_info_dict: dict,
    cam_pose: np.ndarray,
    P_gl: np.ndarray,
    yfov: float,
    res: int,
    eye0_shell: frozenset[int],
    eye1_shell: frozenset[int],
    eye_shell_both: frozenset[int],
) -> tuple[np.ndarray, int]:
    """融合后的模板空间 3D 点：将非虹膜且落在 eye_inner 上的点移到眼眶外侧网格。"""
    inner_fids = _eye_inner_fid_set(region_info_dict)
    iris_ids = _iris_landmark_index_set()
    aspect = 1.0
    P = np.asarray(P_world, dtype=np.float64).copy()
    n_reloc = 0
    for k in range(P.shape[0]):
        if k in iris_ids or not np.all(np.isfinite(P[k])):
            continue
        _cp, _d, tri = trimesh.proximity.closest_point(tmpl_mesh_c, P[k].reshape(1, 3))
        if int(tri[0]) not in inner_fids:
            continue
        P[k] = _relocate_hit_off_eye_inner(
            tmpl_mesh_c,
            P[k],
            inner_fids=inner_fids,
            landmark_idx=k,
            iris_ids=iris_ids,
            eye0_shell=eye0_shell,
            eye1_shell=eye1_shell,
            eye_shell_both=eye_shell_both,
            cam_pose=cam_pose,
            P_gl=P_gl,
            yfov=yfov,
            width=res,
            height=res,
            aspect=aspect,
        )
        n_reloc += 1
    return P, n_reloc


def _closest_on_ray_to_surface(
    mesh: trimesh.Trimesh,
    origin: np.ndarray,
    direction: np.ndarray,
    t_min: float,
    t_max: float,
    n_samples: int = 64,
) -> np.ndarray:
    direc = np.asarray(direction, dtype=np.float64).reshape(3,)
    direc /= max(np.linalg.norm(direc), 1e-12)
    o = np.asarray(origin, dtype=np.float64).reshape(3,)
    ts = np.linspace(t_min, t_max, int(n_samples), dtype=np.float64)
    pts = o + ts[:, None] * direc
    cp, _, _ = trimesh.proximity.closest_point(mesh, pts)
    dd = np.linalg.norm(pts - cp, axis=1)
    return cp[int(np.argmin(dd))]


def jitter_uv_offsets(radius_px: Iterable[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = [(0, 0)]
    for r in radius_px:
        for du in (-r, 0, r):
            for dv in (-r, 0, r):
                out.append((du, dv))
    return sorted(set(out), key=lambda t: abs(t[0]) + abs(t[1]))


def save_debug_lmks2d_overlays(
    rgb_images: Sequence[np.ndarray],
    lm_uv: np.ndarray,
    out_dir: Path,
    *,
    stems: Sequence[str],
    point_radius: int = 2,
) -> list[Path]:
    """
    在渲染得到的 RGB 图上用红色绘制 MediaPipe 的 2D 像素坐标，便于检查检测是否与渲染对齐。

    输出 ``{stem}_lmks2d.png``。``lm_uv`` 形状 ``(K, 478, 2)``，与 ``rgb_images`` 逐张对应。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    red_bgr = (0, 0, 255)
    written: list[Path] = []
    K = len(rgb_images)
    if lm_uv.shape[0] != K:
        raise ValueError(f"lm_uv K={lm_uv.shape[0]} vs rgb_images K={K}")
    if len(stems) != K:
        raise ValueError(f"stems len {len(stems)} != K={K}")
    for k in range(K):
        rgb = np.asarray(rgb_images[k])
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        H, W = vis_bgr.shape[:2]
        for i in range(lm_uv.shape[1]):
            u, vv = float(lm_uv[k, i, 0]), float(lm_uv[k, i, 1])
            if not (np.isfinite(u) and np.isfinite(vv)):
                continue
            ui = int(round(u))
            vi = int(round(vv))
            if ui < 0 or vi < 0 or ui >= W or vi >= H:
                continue
            cv2.circle(
                vis_bgr,
                (ui, vi),
                int(point_radius),
                red_bgr,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        out_p = out_dir / f"{stems[k]}_lmks2d.png"
        cv2.imwrite(str(out_p), vis_bgr)
        written.append(out_p)
    print(f"[hack_lmks478] wrote {len(written)} overlay images under {out_dir}")
    return written


def save_debug_reproj_compare_overlays(
    rgb_images: Sequence[np.ndarray],
    lm_uv_mp: np.ndarray,
    hits_xyz: np.ndarray,
    cam_pose_world: np.ndarray,
    P_gl: np.ndarray,
    out_dir: Path,
    *,
    stems: Sequence[str],
    pt_mp_radius: int = 3,
    pt_repr_radius: int = 2,
) -> list[Path]:
    """
    红点 = MediaPipe；绿点 = 该视角 ``hits_xyz`` 用 ``P·V`` 再投影到像素（与 pyrender 一致时应与红点重合）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    red_bgr, green_bgr = (0, 0, 255), (0, 255, 0)
    written: list[Path] = []
    for ki in range(len(rgb_images)):
        rgb = np.asarray(rgb_images[ki])
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        vis_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        H, W = vis_bgr.shape[:2]
        uv_r, _ = world_points_to_pixel_uv_openvr_top_left_image(
            cam_pose_world,
            P_gl,
            hits_xyz[ki],
            width=W,
            height=H,
        )
        for i in range(lm_uv_mp.shape[1]):
            gp = uv_r[i]
            if np.all(np.isfinite(gp)):
                uig = int(round(float(gp[0])))
                vig = int(round(float(gp[1])))
                if 0 <= uig < W and 0 <= vig < H:
                    cv2.circle(
                        vis_bgr,
                        (uig, vig),
                        int(pt_repr_radius),
                        green_bgr,
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )
            uu, vv = float(lm_uv_mp[ki, i, 0]), float(lm_uv_mp[ki, i, 1])
            if np.isfinite(uu) and np.isfinite(vv):
                ui, vi = int(round(uu)), int(round(vv))
                if 0 <= ui < W and 0 <= vi < H:
                    cv2.circle(
                        vis_bgr,
                        (ui, vi),
                        int(pt_mp_radius),
                        red_bgr,
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )
        out_p = out_dir / f"{stems[ki]}_lmks2d_mp_vs_reproj.png"
        cv2.imwrite(str(out_p), vis_bgr)
        written.append(out_p)
    print(f"[hack_lmks478] wrote {len(written)} reproj_compare images under {out_dir}")
    return written


def render_rgb_uint8(
    verts_centered: np.ndarray,
    faces: np.ndarray | torch.Tensor,
    *,
    res: int,
    camera_distance: float,
    znear: float,
    zfar: float,
    fill_ratio: float,
    yfov: float,
    flat_shading: bool,
    reuse_cached_renderer: bool = False,
) -> np.ndarray:
    V = torch.as_tensor(verts_centered, dtype=torch.float32)
    F = faces if isinstance(faces, torch.Tensor) else torch.as_tensor(faces, dtype=torch.long)
    chw = render_mesh(
        V,
        F,
        res=res,
        filename=None,
        flat_shading=flat_shading,
        center_mesh=False,
        camera_distance=camera_distance,
        fill_ratio=fill_ratio,
        yfov=yfov,
        znear_override=znear,
        zfar_override=zfar,
        reuse_cached_renderer=reuse_cached_renderer,
    )
    rgb = (chw.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def compute_lmks478_barycentric_template(
    *,
    dataset_dir: Path = _DATASET_DIR,
    template_name: str = "hack_template.obj",
    mesh_files: Sequence[str] = DEFAULT_MESH_FILES,
    out_pkl: Path | None = None,
    landmark_task: Path = DEFAULT_LANDMARKER_TASK,
    res: int = 512,
    yfov: float = np.pi / 3.5,
    fill_ratio: float = 0.98,
    flat_shading: bool = False,
    jitter_radii_px: Sequence[int] = (0, 1, 2, 4),
    debug_overlay_dir: Path | None = None,
    debug_overlay_pt_radius: int = 3,
) -> dict[str, Any]:
    """
    多视图：对每个关键点 **优先采用模板 OBJ 视图上的射线交点**，否则在非模板视图上对有效击中取中位数，
    （避免对不同体型网格做三维坐标直接中位数混合）。最后用 ``closest_point`` 贴回模板曲面并编码重心，
    并在各视图上将击中点再用 ``pyrender.PerspectiveCamera`` 的投影矩阵校验与 MediaPipe 的 2D 误差；
    ``debug_overlay_dir`` 下可同时生成 ``*_mp_vs_reproj.png``（红点=MediaPipe，绿点=再投影）。

    返回 ``face_indices`` ``(478,)``、``bary_coords`` ``(478,3)``，以及诊断字段。
    """
    mesh_paths = [dataset_dir / name for name in mesh_files]
    for p in mesh_paths:
        if not p.is_file():
            raise FileNotFoundError(p)

    template_path = dataset_dir / template_name
    V_tm, faces_t = _load_obj_vf(template_path)
    ctr_template = V_tm.mean(axis=0)

    variants: list[tuple[str, Path, np.ndarray]] = []
    radii: list[float] = []
    for p in mesh_paths:
        Vi, Fi = _load_obj_vf(p)
        if Fi.shape != faces_t.shape or not np.array_equal(Fi, faces_t):
            raise ValueError(f"topology mismatch with template: {p}")
        Vc = Vi - ctr_template
        variants.append((p.stem, p, Vc))
        Ri = float(np.linalg.norm(Vc, axis=1).max())
        if math.isfinite(Ri) and Ri > 0:
            radii.append(Ri)
    R_max = max(radii) if radii else 1.0
    cam_d, znear, zfar, cam_pose = _canonical_camera_bundle(
        R_max, yfov=float(yfov), fill_ratio=float(fill_ratio)
    )
    P_gl = pyrender_projection_matrix(
        yfov=float(yfov), znear=znear, zfar=zfar, width=int(res), height=int(res),
    )

    faces_torch = torch.as_tensor(faces_t, dtype=torch.long)
    rgbs: list[np.ndarray] = []
    meshes_c: list[trimesh.Trimesh] = []
    print(f"[hack_lmks478] R_max={R_max:.6f}, d={cam_d:.6f}, zn/zf={znear:.6f}/{zfar:.6f}")
    for _stem, _path, Vc in tqdm(variants, desc="render"):
        rgb = render_rgb_uint8(
            Vc,
            faces_torch,
            res=res,
            camera_distance=cam_d,
            znear=znear,
            zfar=zfar,
            fill_ratio=fill_ratio,
            yfov=float(yfov),
            flat_shading=flat_shading,
        )
        rgbs.append(rgb)
        meshes_c.append(trimesh.Trimesh(vertices=Vc, faces=faces_t, process=False))

    lm_uv = mediapipe_landmarks_pixels_batch(rgbs, model_path=landmark_task)

    tmpl_i = next(i for i, (_, p, _) in enumerate(variants) if p.name == template_name)
    tmpl_mesh_c = trimesh.Trimesh(vertices=variants[tmpl_i][2], faces=faces_t, process=False)

    hits_buf = np.full((lm_uv.shape[0], N_LANDMARKS, 3), np.nan, dtype=np.float64)
    jitter = jitter_uv_offsets(jitter_radii_px)
    inner_fids = _eye_inner_fid_set(region_info)
    eye0_shell, eye1_shell, eye_shell_both = _eye_shell_fid_sets(region_info, inner_fids)
    iris_ids = _iris_landmark_index_set()

    for view_i, mesh in enumerate(meshes_c):
        uv_k = lm_uv[view_i]
        H, W = rgbs[view_i].shape[0], rgbs[view_i].shape[1]
        aspect_wh = float(W) / float(max(H, 1))
        desc = f"rays {variants[view_i][0]}"
        for k in tqdm(range(N_LANDMARKS), desc=desc, leave=False):
            if not (np.isfinite(uv_k[k, 0]) and np.isfinite(uv_k[k, 1])):
                continue
            u0, v0 = float(uv_k[k, 0]), float(uv_k[k, 1])
            placed = False
            for du, dv in jitter:
                uu, vv = u0 + du, v0 + dv
                if uu < 0 or vv < 0 or uu >= W or vv >= H:
                    continue
                ro, rd = pixel_ray_opengl_top_left_origin(
                    uu, vv, W, H, float(yfov), aspect_wh, cam_pose
                )
                hit = ray_mesh_hit_for_landmark(
                    mesh,
                    ro,
                    rd,
                    landmark_idx=k,
                    inner_fids=inner_fids,
                    iris_ids=iris_ids,
                    eye0_shell=eye0_shell,
                    eye1_shell=eye1_shell,
                    eye_shell_both=eye_shell_both,
                )
                if hit is not None:
                    hits_buf[view_i, k] = hit[0]
                    placed = True
                    break
            if not placed:
                ro, rd = pixel_ray_opengl_top_left_origin(
                    u0, v0, W, H, float(yfov), aspect_wh, cam_pose
                )
                p_fb = _closest_on_ray_to_surface(
                    mesh, ro, rd, t_min=cam_d * 0.05, t_max=cam_d * 5.0
                )
                hits_buf[view_i, k] = _relocate_hit_off_eye_inner(
                    mesh,
                    p_fb,
                    inner_fids=inner_fids,
                    landmark_idx=k,
                    iris_ids=iris_ids,
                    eye0_shell=eye0_shell,
                    eye1_shell=eye1_shell,
                    eye_shell_both=eye_shell_both,
                    cam_pose=cam_pose,
                    P_gl=P_gl,
                    yfov=float(yfov),
                    width=W,
                    height=H,
                    aspect=aspect_wh,
                )

    K_views = hits_buf.shape[0]
    reproj_err_px = np.full((K_views, N_LANDMARKS), np.nan, dtype=np.float64)
    for vi in range(K_views):
        H, W = rgbs[vi].shape[0], rgbs[vi].shape[1]
        uv_repr, _ = world_points_to_pixel_uv_openvr_top_left_image(
            cam_pose, P_gl, hits_buf[vi], width=W, height=H
        )
        reproj_err_px[vi] = reprojection_residuals_px_vs_mediapipe(
            lm_uv_mp=lm_uv[vi], lm_uv_repr=uv_repr
        )
        e = reproj_err_px[vi]
        ok_mask = np.isfinite(e)
        if np.any(ok_mask):
            e_ok = np.where(ok_mask, e, np.nan)
            print(
                f"[hack_lmks478] reproj RMS view={variants[vi][0]}: "
                f"RMS={float(np.sqrt(np.nanmean(np.square(e_ok)))):.3f}px "
                f"median={float(np.nanmedian(e_ok)):.3f}px "
                f"max={float(np.nanmax(e_ok)):.3f}px"
            )
        else:
            print(f"[hack_lmks478] reproj RMS view={variants[vi][0]}: no finite errors")

    if debug_overlay_dir is not None:
        stems = [v[0] for v in variants]
        save_debug_lmks2d_overlays(
            rgbs,
            lm_uv,
            Path(debug_overlay_dir),
            stems=stems,
            point_radius=debug_overlay_pt_radius,
        )
        save_debug_reproj_compare_overlays(
            rgbs,
            lm_uv,
            hits_buf,
            cam_pose,
            P_gl,
            Path(debug_overlay_dir),
            stems=stems,
            pt_mp_radius=max(1, debug_overlay_pt_radius),
            pt_repr_radius=max(1, debug_overlay_pt_radius - 1),
        )

    P_world, fusion_src_view = fuse_hits_prioritize_reference_view_then_median_reference_frames(
        hits_buf, tmpl_i
    )
    for kk in range(N_LANDMARKS):
        if np.all(np.isfinite(P_world[kk])):
            continue
        vk = tmpl_mesh_c.vertices[int(kk % tmpl_mesh_c.vertices.shape[0])]
        cp, _, _ = trimesh.proximity.closest_point(tmpl_mesh_c, vk.reshape(1, 3))
        P_world[kk] = cp[0]

    P_world, n_reloc_fused = relocate_fused_landmarks_off_eye_inner(
        P_world,
        tmpl_mesh_c,
        region_info_dict=region_info,
        cam_pose=cam_pose,
        P_gl=P_gl,
        yfov=float(yfov),
        res=int(res),
        eye0_shell=eye0_shell,
        eye1_shell=eye1_shell,
        eye_shell_both=eye_shell_both,
    )
    if n_reloc_fused > 0:
        print(
            f"[hack_lmks478] relocated {n_reloc_fused} fused landmarks off eye_inner "
            f"(excluding iris)"
        )

    closest, _, tri_id = trimesh.proximity.closest_point(tmpl_mesh_c, P_world)
    verts_f = tmpl_mesh_c.vertices[faces_t[tri_id]]
    bc = points_to_barycentric(verts_f.reshape(-1, 3, 3), closest)
    face_indices = tri_id.astype(np.int64)
    bary_coords = np.clip(bc.astype(np.float64), 0.0, 1.0)
    s = bary_coords.sum(axis=1, keepdims=True)
    bary_coords /= np.maximum(s, 1e-12)

    H_t, W_t = rgbs[tmpl_i].shape[0], rgbs[tmpl_i].shape[1]
    aspect_t = float(W_t) / float(max(H_t, 1))
    n_reloc_bary = 0
    for k in range(N_LANDMARKS):
        if k in iris_ids or int(face_indices[k]) not in inner_fids:
            continue
        allowed_k = _allowed_eye_shell_for_landmark(
            k, eye0_shell, eye1_shell, eye_shell_both
        )
        p_new = _relocate_hit_off_eye_inner(
            tmpl_mesh_c,
            closest[k],
            inner_fids=inner_fids,
            landmark_idx=k,
            iris_ids=iris_ids,
            eye0_shell=eye0_shell,
            eye1_shell=eye1_shell,
            eye_shell_both=eye_shell_both,
            cam_pose=cam_pose,
            P_gl=P_gl,
            yfov=float(yfov),
            width=W_t,
            height=H_t,
            aspect=aspect_t,
        )
        shell_pt = _closest_hit_on_eye_shell(tmpl_mesh_c, p_new, allowed_k)
        if shell_pt is not None:
            p_new, fid = shell_pt
        else:
            cp, _, tri = trimesh.proximity.closest_point(
                tmpl_mesh_c, p_new.reshape(1, 3)
            )
            fid = int(tri[0])
            p_new = cp[0]
        if fid in inner_fids or fid not in allowed_k:
            shell_pt = _closest_hit_on_eye_shell(tmpl_mesh_c, p_new, allowed_k)
            if shell_pt is None:
                continue
            p_new, fid = shell_pt
        closest[k] = p_new
        face_indices[k] = fid
        tri_v = tmpl_mesh_c.vertices[faces_t[fid]].astype(np.float64)
        bc_k = points_to_barycentric(tri_v.reshape(1, 3, 3), p_new.reshape(1, 3))
        bc_k = np.clip(bc_k.reshape(3), 0.0, None)
        s_k = float(bc_k.sum())
        if s_k > 1e-12:
            bc_k /= s_k
        bary_coords[k] = bc_k
        P_world[k] = p_new
        n_reloc_bary += 1
    if n_reloc_bary > 0:
        print(
            f"[hack_lmks478] relocated {n_reloc_bary} bary landmarks off eye_inner "
            f"(excluding iris)"
        )

    tri_fb = tmpl_mesh_c.vertices[faces_t[face_indices]]
    lm_xyz_from_bary = np.einsum("ij,ijk->ik", bary_coords, tri_fb)
    uv_from_bary, _ = world_points_to_pixel_uv_openvr_top_left_image(
        cam_pose, P_gl, lm_xyz_from_bary, width=W_t, height=H_t
    )
    e_bary_repr = reprojection_residuals_px_vs_mediapipe(
        lm_uv_mp=lm_uv[tmpl_i], lm_uv_repr=uv_from_bary
    )
    if np.any(np.isfinite(e_bary_repr)):
        print(
            f"[hack_lmks478] after closest+bary, template-view reproj: "
            f"RMS={float(np.sqrt(np.nanmean(np.square(e_bary_repr)))):.3f}px "
            f"median={float(np.nanmedian(e_bary_repr)):.3f}px"
        )

    meta: dict[str, Any] = {
        "face_indices": face_indices,
        "bary_coords": bary_coords,
        "projection_matrix_gl": P_gl,
        "reproj_err_px_per_view": reproj_err_px,
        "reproj_err_px_bary_template_view": e_bary_repr,
        "fusion_prioritize_template_view_then_median_others": True,
        "fusion_src_view_per_lm": fusion_src_view,
        "template_center": ctr_template.astype(np.float64),
        "template_obj": str(template_path),
        "mesh_files": [str(p) for p in mesh_paths],
        "resolution": int(res),
        "yfov": float(yfov),
        "fill_ratio": float(fill_ratio),
        "camera_distance": float(cam_d),
        "znear": float(znear),
        "zfar": float(zfar),
        "cam_pose_world": cam_pose,
        "P_world_fused": P_world,
        "P_world_median": P_world,
        "hits_per_view": hits_buf,
    }
    if out_pkl is not None:
        op = Path(out_pkl)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[hack_lmks478] saved {op}")
    return meta


def extract_face_lmks_mediapipe(
    image_dir: Path | None = None,
    image_arrays: list[np.ndarray] | None = None,
    lmks_path: Path | None = None,
) -> list[np.ndarray]:
    assert image_dir is not None or image_arrays is not None, (
        "Either image_dir or image_arrays must be provided"
    )
    if image_dir is not None:
        images = []
        for img_path in sorted(image_dir.glob("*.jpg")):
            images.append(cv2.imread(str(img_path)))
        if lmks_path is None:
            _ = Path(str(image_dir).replace("rgb", "landmark2d/Mediapipe"))
    else:
        images = image_arrays or []

    det = build_face_landmarker(DEFAULT_LANDMARKER_TASK)
    landmarks: list[np.ndarray] = []
    for image_bgr in tqdm(images):
        if image_bgr is None:
            landmarks.append(np.full((N_LANDMARKS, 3), np.nan))
            continue
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        res = run_mediapipe(det, rgb)
        if res is None:
            landmarks.append(np.full((N_LANDMARKS, 3), np.nan))
            continue
        lm_img, _ = res
        landmarks.append(lm_img)
    return landmarks


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="MediaPipe478 → template barycentric")
    parser.add_argument("--out", type=Path, default=_DATASET_DIR / "hack_lmks478_bary.pkl")
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument(
        "--debug-overlay-dir",
        type=Path,
        default='/media/ubuntu/SSD/PHACK_code/dataset/hack_lmks478_debug_overlay',
        help="可选：写入红色 478 叠图 *_lmks2d.png 以及红绿对比 *_mp_vs_reproj.png",
    )
    parser.add_argument(
        "--debug-overlay-pt",
        type=int,
        default=1,
        help="叠图关键点半径（像素）",
    )
    args = parser.parse_args()
    compute_lmks478_barycentric_template(
        out_pkl=args.out,
        res=args.res,
        debug_overlay_dir=args.debug_overlay_dir,
        debug_overlay_pt_radius=args.debug_overlay_pt,
    )


if __name__ == "__main__":
    main_cli()





























