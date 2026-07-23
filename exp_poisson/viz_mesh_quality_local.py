"""局部操控网格质量对比图（LAMM vs PLANMM）。

两行：LAMM-MLPMixer / PLANMM-MLPMixer；
每行最左为 source 全局渲染，随后按 chin → eyes → mouth → nose。
每个区域：全局着色渲染 | 局部质量热图（略小），热图为邻接法线不连续性、``RdYlBu_r``。
相机角与裁剪对齐 ``viz_dARAP_local.py``。

网格路径：
  - LAMM: ``ablations/lamm_hack_manipulation_mlp/generated_local/{region}/``
  - PLANMM: ``results/planmm_hack_manipulation/<最新>/generated_local/{region}/``
四个区域由**同一 source** 网格分别操控生成（``--force-regenerate`` 可强制重生成）。
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import read_yaml
from dataset.hack_seg_fids import REGION_NAMES
from exp_poisson.LAMM_generate_local import (
    _generate_global_meshes as _lamm_generate_global_meshes,
    _resolve_device,
    get_random_displacements as lamm_get_random_displacements,
    save_mesh as lamm_save_mesh,
)
from exp_poisson.planmm_generate_local import (
    _generate_global_meshes as _planmm_generate_global_meshes,
    _resolve_gaussian_path,
    get_random_displacements as planmm_get_random_displacements,
    save_mesh as planmm_save_mesh,
)
from exp_poisson.planmm_generate_samples import build_planmm_net
from exp_poisson.planmm_prepare_inference import resolve_checkpoint_path, resolve_run_config
from networks.lamm import LAMM
from render.mesh_render import render_mesh
from utils.helpers import seed_everything
from utils.mesh import read_obj, vertex_normals
from utils.torch_utils import load_from_checkpoint

# ----- paths -----
LAMM_CONFIG = _REPO_ROOT / "exp_poisson" / "lamm_hack_manipulation.yaml"
LAMM_ROOT = _REPO_ROOT / "ablations" / "lamm_hack_manipulation_mlp"
LAMM_CKPT = LAMM_ROOT / "best_alpha_max.pth"
LAMM_LOCAL = LAMM_ROOT / "generated_local"

PLANMM_CONFIG = _REPO_ROOT / "exp_poisson" / "planmm_hack_manipulation.yaml"
PLANMM_ROOT = _REPO_ROOT / "results" / "planmm_hack_manipulation"
REGION_INFO_FILE = _REPO_ROOT / "dataset" / "hack_region_info.pkl"
DEFAULT_OUTPUT = _REPO_ROOT / "figures" / "mesh_quality_local.png"

# 列顺序：chin / eyes / mouth / nose
REGIONS: tuple[str, ...] = ("chin", "eyes", "mouth", "nose")
ROW_LABELS: tuple[str, ...] = ("LAMM-MLPMixer", "PLANMM-MLPMixer")

SAMPLE_ID_LAMM = 1
# 四个区域各自选取 sample idx：id_{id}_region_*_sample_{idx}.obj
SAMPLE_IDX_LAMM: dict[str, int] = {
    "chin": 1,
    "eyes": 1,
    "mouth": 3,
    "nose": 0,
}
SAMPLE_ID_PLANMM = 4
SAMPLE_IDX_PLANMM: dict[str, int] = {
    "chin": 3,
    "eyes": 2,
    "mouth": 2,
    "nose": 1,
}

# ----- render -----
RENDER_RES_GLOBAL = 640
RENDER_RES_LOCAL = 640
# 统一面板尺寸：全局与局部同高，局部为正方形
GLOBAL_PANEL_HW = (220, 190)  # (H, W)
LOCAL_PANEL_HW = (220, 220)   # 与全局同高的正方形
RENDER_FILL_RATIO = 0.97
RENDER_FILL_RATIO_LOCAL = 0.88  # 区域中心构图时区域更大地填满画面
RENDER_YFOV = np.pi / 3.0

MOSAIC_TRIM_BG = 1.0
MOSAIC_TRIM_TOL = 0.035
MOSAIC_TRIM_PAD = 4

PAIR_GAP_PX = 22
REGION_GAP_PX = 28
ROW_GAP_PX = 16
ROW_LABEL_FONT_SIZE = 28
ROW_LABEL_GAP = 8
ROW_LABEL_BG = (255, 255, 255, 255)
ROW_LABEL_FG = (0, 0, 0, 255)
COL_LABEL_FONT_SIZE = 20
COL_LABEL_PAD_Y = 8

HEATMAP_CMAP = "RdYlBu_r"
HEATMAP_PERCENTILE = 98.0
HEATMAP_GAMMA = 0.55

GEN_NUM_SAMPLES = 5
GEN_NUM_RANDOM = 5
GEN_K_STD_LAMM = 1.0
GEN_K_STD_PLANMM = 1.5
GEN_SEED = 3986

_FACE_REGION_NAMES = (
    "LeftFace", "RightFace", "Ears", "Chin", "Forehead", "Eyes", "Nose", "Mouth",
)
_REGION_NAME_TO_KEY = {name.lower(): key for key, name in REGION_NAMES.items()}


@dataclass(frozen=True)
class RegionView:
    """Camera / crop for one region column (aligned with viz_dARAP_local)."""

    region: str
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    # 全局：稍宽的脸部裁剪
    global_crop_w: float = 0.88
    global_crop_h: float = 0.78
    global_cx: float = 0.5
    global_cy: float = 0.36
    # 局部：对齐 dARAP 的紧裁剪
    local_crop_w: float = 0.9
    local_crop_h: float = 0.7
    local_cx: float = 0.5
    local_cy: float = 0.34


# Source 列：正面全局视角（与 mouth 全局裁剪接近，无局部 zoom）
SOURCE_VIEW = RegionView(
    region="source",
    yaw=0.0,
    pitch=0.0,
    global_crop_w=0.88,
    global_crop_h=0.78,
    global_cx=0.5,
    global_cy=0.36,
)

REGION_VIEWS: dict[str, RegionView] = {
    "chin": RegionView(
        region="chin",
        yaw=28.0,
        pitch=5.0,
        local_crop_w=0.40,
        local_crop_h=0.36,
        local_cx=0.7,
        local_cy=0.52,
        global_cx=0.58,
        global_cy=0.40,
    ),
    "eyes": RegionView(
        region="eyes",
        yaw=10.0,
        pitch=0.0,
        local_crop_w=0.50,
        local_crop_h=0.30,
        local_cx=0.55,
        local_cy=0.48,
        global_cy=0.32,
    ),
    "mouth": RegionView(
        region="mouth",
        yaw=0.0,
        pitch=0.0,
        local_crop_w=0.40,
        local_crop_h=0.30,
        local_cx=0.5,
        local_cy=0.50,
        global_cy=0.40,
    ),
    "nose": RegionView(
        region="nose",
        yaw=25.0,
        pitch=10.0,
        local_crop_w=0.34,
        local_crop_h=0.38,
        local_cx=0.63,
        local_cy=0.48,
        global_cy=0.36,
        global_cx=0.52,
    ),
}


def _parse_region_idxs(value: str, *, defaults: dict[str, int]) -> dict[str, int]:
    """Parse ``4`` / ``4,2,0,1`` (chin..nose) / ``chin:4,eyes:2,...`` into per-region idxs."""
    s = value.strip()
    if not s:
        return dict(defaults)
    if "," not in s and ":" not in s:
        v = int(s)
        return {r: v for r in REGIONS}
    if ":" in s:
        out = dict(defaults)
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            key, raw = part.split(":", 1)
            key = key.strip().lower()
            if key not in REGIONS:
                raise argparse.ArgumentTypeError(
                    f"unknown region {key!r}; expected one of {list(REGIONS)}"
                )
            out[key] = int(raw.strip())
        return out
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != len(REGIONS):
        raise argparse.ArgumentTypeError(
            f"expected {len(REGIONS)} idxs for {list(REGIONS)}, got {len(vals)}"
        )
    return dict(zip(REGIONS, vals))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local mesh-quality mosaic (LAMM vs PLANMM).")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--lamm-sample-id", type=int, default=SAMPLE_ID_LAMM)
    p.add_argument(
        "--lamm-sample-idx",
        type=str,
        default=None,
        help="Per-region idxs: single int, chin,eyes,mouth,nose ints, or chin:4,eyes:2,...",
    )
    p.add_argument("--planmm-sample-id", type=int, default=SAMPLE_ID_PLANMM)
    p.add_argument(
        "--planmm-sample-idx",
        type=str,
        default=None,
        help="Per-region idxs: single int, chin,eyes,mouth,nose ints, or chin:4,eyes:2,...",
    )
    p.add_argument("--device", type=str, default="0")
    p.add_argument(
        "--skip-generate",
        action="store_true",
        help="Do not generate missing region folders.",
    )
    p.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate chin/eyes/mouth/nose from one shared source for LAMM and PLANMM.",
    )
    p.add_argument("--planmm-run", type=Path, default=None)
    p.add_argument("--region-info", type=Path, default=REGION_INFO_FILE)
    args = p.parse_args()
    args.lamm_sample_idx = _parse_region_idxs(
        args.lamm_sample_idx or "", defaults=SAMPLE_IDX_LAMM
    )
    args.planmm_sample_idx = _parse_region_idxs(
        args.planmm_sample_idx or "", defaults=SAMPLE_IDX_PLANMM
    )
    return args


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------

def latest_planmm_local_root(root: Path = PLANMM_ROOT) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"missing {root}")
    cands = [
        p / "generated_local"
        for p in root.iterdir()
        if p.is_dir() and (p / "generated_local").is_dir()
    ]
    if not cands:
        raise FileNotFoundError(f"no generated_local under {root}")
    return max(cands, key=lambda p: p.parent.name)


def region_mesh_path(folder: Path, region: str, sample_id: int, sample_idx: int) -> Path:
    key = _REGION_NAME_TO_KEY[region]
    return folder / region / (
        f"id_{sample_id}_region_{key}_{region}_sample_{sample_idx}.obj"
    )


def load_obj_mesh(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(obj_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


def load_face_vertex_ids(region_info_file: Path) -> np.ndarray:
    with open(region_info_file, "rb") as f:
        region_info = pickle.load(f)
    vids: list[int] = []
    for name in _FACE_REGION_NAMES:
        if name in region_info:
            vids.extend(
                np.asarray(region_info[name]["vids"], dtype=np.int64).reshape(-1).tolist()
            )
    if not vids:
        raise RuntimeError(f"no face vids in {region_info_file}")
    return np.unique(np.asarray(vids, dtype=np.int64))


def load_region_vids(region_info_file: Path, region: str) -> np.ndarray:
    """Return vertex ids for a named region (e.g. chin / eyes / mouth / nose)."""
    canon = REGION_NAMES[_REGION_NAME_TO_KEY[region.lower()]]
    with open(region_info_file, "rb") as f:
        region_info = pickle.load(f)
    if canon not in region_info:
        raise KeyError(f"region {canon!r} missing in {region_info_file}")
    return np.asarray(region_info[canon]["vids"], dtype=np.int64).reshape(-1)


# ---------------------------------------------------------------------------
# generation: 同一 source → chin/eyes/mouth/nose 四个区域
# ---------------------------------------------------------------------------

def _clear_region_dirs(out_root: Path, regions: tuple[str, ...] = REGIONS) -> None:
    for region in regions:
        d = out_root / region
        if d.is_dir():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def generate_lamm_shared_source_regions(
    *,
    device: torch.device,
    out_root: Path = LAMM_LOCAL,
    checkpoint: Path = LAMM_CKPT,
    config_file: Path = LAMM_CONFIG,
    regions: tuple[str, ...] = REGIONS,
    num_samples: int = GEN_NUM_SAMPLES,
    num_random: int = GEN_NUM_RANDOM,
    k_std: float = GEN_K_STD_LAMM,
    seed: int = GEN_SEED,
) -> None:
    """One shared source identity → manipulate each of ``regions`` into subfolders."""
    config = read_yaml(config_file)
    config["MACHINE"] = "local"
    save_dir = Path(config["CHECKPOINT"]["save_dir"])
    if not save_dir.is_absolute():
        save_dir = _REPO_ROOT / save_dir

    control_lms = {int(k): v for k, v in config["MODEL"]["control_vertices"].items()}
    region_keys = {r: _REGION_NAME_TO_KEY[r] for r in regions}
    for r, k in region_keys.items():
        if k not in control_lms:
            raise KeyError(f"region {r} key={k} missing in control_vertices")

    seed_everything(seed)
    template_path = config["MODEL"].get(
        "reference_obj", str(_REPO_ROOT / "dataset" / "hack_template.obj")
    )
    faces = read_obj(str(template_path)).fvs
    mean, std, mm_mult = 0.0, 1.0, 1.0

    delta_stats_path = save_dir / "displacement_stats.pickle"
    if not delta_stats_path.is_file():
        raise FileNotFoundError(f"missing {delta_stats_path}")
    with open(delta_stats_path, "rb") as f:
        delta_stats = pickle.load(f)

    sources = _lamm_generate_global_meshes(
        config,
        str(checkpoint),
        num_samples,
        k_std,
        device,
        save_dir,
        save_meshes=False,
    )

    model_cfg = dict(config["MODEL"])
    model_cfg["manipulation"] = True
    net = LAMM(model_cfg).to(device)
    load_from_checkpoint(net, str(checkpoint), partial_restore=True, device=str(device))
    net.eval()

    _clear_region_dirs(out_root, regions)

    for sid, sample in enumerate(sources[:num_samples], start=1):
        x = sample["verts"].to(device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        source_np = mm_mult * (x[0].detach().cpu().numpy() * (std + 1e-7) + mean)
        print(f"[viz_mesh_quality_local] LAMM shared source id={sid}")
        for region, key in region_keys.items():
            out_dir = out_root / region
            lamm_save_mesh(source_np, faces, out_dir / f"id_{sid}_source.obj")
            for m in range(num_random):
                delta = lamm_get_random_displacements(
                    delta_stats, key, control_lms, k_std, device
                )
                out = net((x, delta))[-1]
                out_np = mm_mult * (out[0].detach().cpu().numpy() * (std + 1e-7) + mean)
                path = out_dir / f"id_{sid}_region_{key}_{region}_sample_{m}.obj"
                lamm_save_mesh(out_np, faces, path)
                print(f"[viz_mesh_quality_local] saved {path.relative_to(out_root)}")


def generate_planmm_shared_source_regions(
    *,
    device: torch.device,
    out_root: Path,
    config_file: Path = PLANMM_CONFIG,
    regions: tuple[str, ...] = REGIONS,
    num_samples: int = GEN_NUM_SAMPLES,
    num_random: int = GEN_NUM_RANDOM,
    k_std: float = GEN_K_STD_PLANMM,
    seed: int = GEN_SEED,
) -> None:
    """One shared PLANMM source → manipulate each of ``regions`` into subfolders."""
    run_dir = out_root.parent if out_root.name == "generated_local" else out_root
    checkpoint = resolve_checkpoint_path(config_file, str(run_dir))
    config = resolve_run_config(config_file, checkpoint)
    config["MACHINE"] = "local"

    control_lms = {int(k): v for k, v in config["MODEL"]["control_vertices"].items()}
    region_keys = {r: _REGION_NAME_TO_KEY[r] for r in regions}
    for r, k in region_keys.items():
        if k not in control_lms:
            raise KeyError(f"region {r} key={k} missing in control_vertices")

    seed_everything(seed)
    template_path = config["MODEL"].get(
        "reference_obj", str(_REPO_ROOT / "dataset" / "hack_template.obj")
    )
    faces = read_obj(str(template_path)).fvs

    delta_stats_path = run_dir / "displacement_stats.pickle"
    if not delta_stats_path.is_file():
        raise FileNotFoundError(f"missing {delta_stats_path}")
    with open(delta_stats_path, "rb") as f:
        delta_stats = pickle.load(f)

    gaussian_path = _resolve_gaussian_path(run_dir, None)
    sources = _planmm_generate_global_meshes(
        config,
        checkpoint,
        num_samples,
        k_std,
        device,
        run_dir,
        gaussian_path,
        save_meshes=False,
    )

    model_cfg = dict(config["MODEL"])
    model_cfg["manipulation"] = True
    net = build_planmm_net(model_cfg, device)
    load_from_checkpoint(net, str(checkpoint), partial_restore=True, device=str(device))
    net.eval()

    _clear_region_dirs(out_root, regions)

    with torch.no_grad():
        for sid, sample in enumerate(sources[:num_samples], start=1):
            x = sample["verts"].to(device)
            if x.ndim == 2:
                x = x.unsqueeze(0)
            source_np = x[0].detach().cpu().numpy()
            print(f"[viz_mesh_quality_local] PLANMM shared source id={sid}")
            for region, key in region_keys.items():
                out_dir = out_root / region
                planmm_save_mesh(source_np, faces, out_dir / f"id_{sid}_source.obj")
                for m in range(num_random):
                    delta = planmm_get_random_displacements(
                        delta_stats, key, control_lms, k_std, device
                    )
                    out = net((x, delta))[-1]
                    out_np = out[0].detach().cpu().numpy()
                    path = out_dir / f"id_{sid}_region_{key}_{region}_sample_{m}.obj"
                    planmm_save_mesh(out_np, faces, path)
                    print(f"[viz_mesh_quality_local] saved {path.relative_to(out_root)}")


def _sources_match_across_regions(out_root: Path, regions: tuple[str, ...] = REGIONS) -> bool:
    """True iff each region folder has the same id_*_source.obj bytes for shared ids."""
    ref_dir = out_root / regions[0]
    sources = sorted(ref_dir.glob("id_*_source.obj"))
    if not sources:
        return False
    for src in sources:
        data = src.read_bytes()
        for region in regions[1:]:
            other = out_root / region / src.name
            if not other.is_file() or other.read_bytes() != data:
                return False
            key = _REGION_NAME_TO_KEY[region]
            if not any((out_root / region).glob(f"id_*_region_{key}_{region}_sample_*.obj")):
                return False
    return True


def ensure_meshes(
    *,
    lamm_root: Path,
    planmm_root: Path,
    device: torch.device,
    skip_generate: bool,
    force_regenerate: bool,
) -> None:
    need_lamm = force_regenerate or not _sources_match_across_regions(lamm_root)
    need_planmm = force_regenerate or not _sources_match_across_regions(planmm_root)

    if need_lamm:
        if skip_generate:
            raise FileNotFoundError(
                f"LAMM region folders missing shared source under {lamm_root}; "
                "rerun without --skip-generate or pass --force-regenerate"
            )
        print("[viz_mesh_quality_local] regenerating LAMM chin/eyes/mouth/nose from one source …")
        generate_lamm_shared_source_regions(device=device, out_root=lamm_root)

    if need_planmm:
        if skip_generate:
            raise FileNotFoundError(
                f"PLANMM region folders missing shared source under {planmm_root}; "
                "rerun without --skip-generate or pass --force-regenerate"
            )
        print("[viz_mesh_quality_local] regenerating PLANMM chin/eyes/mouth/nose from one source …")
        generate_planmm_shared_source_regions(device=device, out_root=planmm_root)


# ---------------------------------------------------------------------------
# quality / render
# ---------------------------------------------------------------------------

def build_vertex_adjacency(faces: np.ndarray, n_verts: int) -> list[np.ndarray]:
    nbrs: list[set[int]] = [set() for _ in range(n_verts)]
    for a, b, c in faces.astype(np.int64):
        nbrs[a].update((int(b), int(c)))
        nbrs[b].update((int(a), int(c)))
        nbrs[c].update((int(a), int(b)))
    return [np.fromiter(s, dtype=np.int64) for s in nbrs]


def normal_smoothness_score(
    verts: np.ndarray,
    faces: np.ndarray,
    adjacency: list[np.ndarray] | None = None,
) -> np.ndarray:
    v = torch.from_numpy(np.asarray(verts, dtype=np.float32)).unsqueeze(0)
    f = torch.from_numpy(np.asarray(faces, dtype=np.int64)).unsqueeze(0)
    normals = vertex_normals(v, f)[0].numpy().astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    if adjacency is None:
        adjacency = build_vertex_adjacency(faces, len(verts))
    scores = np.zeros(len(verts), dtype=np.float64)
    for i, nbr in enumerate(adjacency):
        if nbr.size == 0:
            continue
        dots = np.clip((normals[i] * normals[nbr]).sum(axis=1), -1.0, 1.0)
        scores[i] = float(np.mean(1.0 - dots))
    return scores


def scores_to_vertex_colors(
    scores: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str = HEATMAP_CMAP,
    gamma: float = HEATMAP_GAMMA,
    region_vids: np.ndarray | None = None,
    base_rgb: tuple[float, float, float] = (0.86, 0.75, 0.69),
) -> np.ndarray:
    """Map scores to RGB; if ``region_vids`` given, only those verts get heatmap colors."""
    denom = max(float(vmax) - float(vmin), 1e-12)
    linear = np.clip((scores - float(vmin)) / denom, 0.0, 1.0)
    norm = np.power(linear, max(float(gamma), 1e-6))
    heat = plt.get_cmap(cmap)(norm)[:, :3].astype(np.float32)
    if region_vids is None:
        return heat
    colors = np.tile(
        np.asarray(base_rgb, dtype=np.float32).reshape(1, 3), (len(scores), 1)
    )
    vids = np.asarray(region_vids, dtype=np.int64).reshape(-1)
    colors[vids] = heat[vids]
    return colors


def rotation_matrix_from_view_deg(
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray | None:
    if abs(yaw_deg) < 1e-8 and abs(pitch_deg) < 1e-8 and abs(roll_deg) < 1e-8:
        return None
    ax = np.deg2rad(float(pitch_deg))
    ay = np.deg2rad(float(yaw_deg))
    az = np.deg2rad(float(roll_deg))
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def crop_chw_near_uniform_bg(
    img: torch.Tensor,
    *,
    bg: float = MOSAIC_TRIM_BG,
    tol: float = MOSAIC_TRIM_TOL,
    pad: int = MOSAIC_TRIM_PAD,
) -> torch.Tensor:
    _, h, w = img.shape
    fg = (torch.abs(img - bg) > tol).any(dim=0)
    ys, xs = torch.where(fg)
    if ys.numel() == 0:
        return img
    y0 = max(0, int(ys.min().item()) - pad)
    y1 = min(h, int(ys.max().item()) + 1 + pad)
    x0 = max(0, int(xs.min().item()) - pad)
    x1 = min(w, int(xs.max().item()) + 1 + pad)
    return img[:, y0:y1, x0:x1]


def crop_chw_to_face_region(
    img: torch.Tensor,
    *,
    width_ratio: float,
    height_ratio: float,
    center_x_frac: float,
    center_y_frac: float,
    pad: int = MOSAIC_TRIM_PAD,
    bg: float = MOSAIC_TRIM_BG,
    tol: float = MOSAIC_TRIM_TOL,
) -> torch.Tensor:
    _, h, w = img.shape
    fg = (torch.abs(img - bg) > tol).any(dim=0)
    ys, xs = torch.where(fg)
    if ys.numel() == 0:
        return img
    x0, y0 = int(xs.min().item()), int(ys.min().item())
    x1, y1 = int(xs.max().item()) + 1, int(ys.max().item()) + 1
    fg_w, fg_h = x1 - x0, y1 - y0
    crop_w = fg_w * width_ratio
    crop_h = fg_h * height_ratio
    cx = x0 + fg_w * center_x_frac
    cy = y0 + fg_h * center_y_frac
    px0 = max(0, int(np.floor(cx - crop_w * 0.5 - pad)))
    py0 = max(0, int(np.floor(cy - crop_h * 0.5 - pad)))
    px1 = min(w, max(int(np.ceil(cx + crop_w * 0.5 + pad)), px0 + 1))
    py1 = min(h, max(int(np.ceil(cy + crop_h * 0.5 + pad)), py0 + 1))
    return img[:, py0:py1, px0:px1]


def render_view(
    verts: np.ndarray,
    faces: np.ndarray,
    face_vids: np.ndarray,
    view: RegionView,
    *,
    res: int,
    local: bool,
    vertex_colors: np.ndarray | None = None,
    region_vids: np.ndarray | None = None,
) -> torch.Tensor:
    rot = rotation_matrix_from_view_deg(view.yaw, view.pitch, view.roll)
    half = RENDER_YFOV * 0.5
    # 局部：以区域质心为中心，并按区域尺寸拉近相机 → 局部放大
    if local and region_vids is not None and len(region_vids) > 0:
        rvids = np.asarray(region_vids, dtype=np.int64)
        ctr = verts[rvids].mean(axis=0)
        verts_c = verts - ctr
        if rot is not None:
            verts_rot = verts_c @ rot.T
            reg = verts_rot[rvids]
        else:
            reg = verts_c[rvids]
        radius = float(np.linalg.norm(reg, axis=1).max())
        radius = max(radius * 1.35, 1e-4)  # 稍留边
        cam_d = radius / (RENDER_FILL_RATIO_LOCAL * np.tan(half))
    else:
        ctr = verts[face_vids].mean(axis=0)
        verts_c = verts - ctr
        cam_d = None
    img = render_mesh(
        verts_c,
        faces,
        vertex_colors=vertex_colors,
        rot_mat=rot,
        res=res,
        filename=None,
        flat_shading=False,
        center_mesh=False,
        fill_ratio=RENDER_FILL_RATIO,
        camera_distance=cam_d,
        yfov=RENDER_YFOV,
        reuse_cached_renderer=True,
    )
    img = crop_chw_near_uniform_bg(img)
    if local:
        # 区域已在构图中心并拉近；再按 local_cx/cy 微调裁切中心
        return crop_chw_to_face_region(
            img,
            width_ratio=min(view.local_crop_w * 1.6, 0.95),
            height_ratio=min(view.local_crop_h * 1.6, 0.95),
            center_x_frac=view.local_cx,
            center_y_frac=view.local_cy,
        )
    return crop_chw_to_face_region(
        img,
        width_ratio=view.global_crop_w,
        height_ratio=view.global_crop_h,
        center_x_frac=view.global_cx,
        center_y_frac=view.global_cy,
    )


def resize_chw(img: torch.Tensor, scale: float) -> torch.Tensor:
    if abs(scale - 1.0) < 1e-6:
        return img
    x = img.unsqueeze(0)
    out = F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)
    return out.squeeze(0)


def fit_chw_letterbox(
    img: torch.Tensor,
    *,
    out_h: int,
    out_w: int,
    bg: float = MOSAIC_TRIM_BG,
) -> torch.Tensor:
    """Scale to fit inside ``(out_h, out_w)`` keeping aspect ratio, then center-pad."""
    _, h, w = img.shape
    if h < 1 or w < 1:
        return torch.ones(3, out_h, out_w, dtype=img.dtype) * bg
    scale = min(out_h / h, out_w / w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = F.interpolate(
        img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False
    ).squeeze(0)
    return pad_chw_to(resized, out_h=out_h, out_w=out_w, bg=bg)


def fit_chw_cover(
    img: torch.Tensor,
    *,
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    """Scale to cover ``(out_h, out_w)``, then center-crop — 面板填满、尺寸完全一致。"""
    _, h, w = img.shape
    if h < 1 or w < 1:
        return torch.ones(3, out_h, out_w, dtype=img.dtype)
    scale = max(out_h / h, out_w / w)
    nh = max(out_h, int(np.ceil(h * scale)))
    nw = max(out_w, int(np.ceil(w * scale)))
    resized = F.interpolate(
        img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False
    ).squeeze(0)
    y0 = max(0, (nh - out_h) // 2)
    x0 = max(0, (nw - out_w) // 2)
    return resized[:, y0 : y0 + out_h, x0 : x0 + out_w].contiguous()


def fit_fg_to_square_panel(
    img: torch.Tensor,
    *,
    out_h: int,
    out_w: int,
    bg: float = MOSAIC_TRIM_BG,
    tol: float = MOSAIC_TRIM_TOL,
    pad_frac: float = 0.08,
    skin_rgb: tuple[float, float, float] = (0.86, 0.75, 0.69),
    heat_tol: float = 0.12,
) -> torch.Tensor:
    """Crop ROI to a square then resize to exact ``out_h x out_w``.

    Prefer heatmap-colored pixels (偏离肤色) 作为 ROI，使各局部图内容占比一致。
    """
    _, h, w = img.shape
    skin = torch.tensor(skin_rgb, dtype=img.dtype, device=img.device).view(3, 1, 1)
    heat = (torch.abs(img - skin).amax(dim=0) > heat_tol) & (
        torch.abs(img - bg) > tol
    ).any(dim=0)
    ys, xs = torch.where(heat)
    if ys.numel() < 32:
        fg = (torch.abs(img - bg) > tol).any(dim=0)
        ys, xs = torch.where(fg)
    if ys.numel() == 0:
        return torch.ones(3, out_h, out_w, dtype=img.dtype) * bg

    y0 = int(ys.min().item())
    y1 = int(ys.max().item()) + 1
    x0 = int(xs.min().item())
    x1 = int(xs.max().item()) + 1
    bh, bw = y1 - y0, x1 - x0
    side = max(bh, bw, 1)
    side = int(np.ceil(side * (1.0 + pad_frac)))
    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)
    sy0 = int(np.floor(cy - 0.5 * side))
    sx0 = int(np.floor(cx - 0.5 * side))
    sy1 = sy0 + side
    sx1 = sx0 + side

    pad_top = max(0, -sy0)
    pad_left = max(0, -sx0)
    pad_bottom = max(0, sy1 - h)
    pad_right = max(0, sx1 - w)
    if pad_top or pad_left or pad_bottom or pad_right:
        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=bg)
        sy0 += pad_top
        sy1 += pad_top
        sx0 += pad_left
        sx1 += pad_left

    crop = img[:, sy0:sy1, sx0:sx1]
    return F.interpolate(
        crop.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False
    ).squeeze(0)


def pad_chw_to(
    img: torch.Tensor,
    *,
    out_h: int,
    out_w: int,
    bg: float = MOSAIC_TRIM_BG,
) -> torch.Tensor:
    _, h, w = img.shape
    if h > out_h or w > out_w:
        raise ValueError(f"cannot pad {h}x{w} into {out_h}x{out_w}")
    pad_h, pad_w = out_h - h, out_w - w
    return F.pad(
        img,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        value=bg,
    )


# ---------------------------------------------------------------------------
# mosaic assembly
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        import PIL

        font_path = os.path.join(os.path.dirname(PIL.__file__), "fonts", "DejaVuSans.ttf")
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def _pil_from_chw(img: torch.Tensor) -> Image.Image:
    arr = (
        img.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _chw_from_pil(img: Image.Image) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
        .permute(2, 0, 1)
        .contiguous()
    )


def _vertical_caption_layer(
    caption: str,
    *,
    row_h: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[Image.Image, int]:
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = measure.textbbox((0, 0), caption, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_img = Image.new("RGBA", (max(text_w, 1), max(text_h, 1)), (0, 0, 0, 0))
    ImageDraw.Draw(text_img).text((-bbox[0], -bbox[1]), caption, fill=ROW_LABEL_FG, font=font)
    rotated = text_img.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    max_h = max(row_h - 2 * ROW_LABEL_GAP, 1)
    if rotated.height > max_h:
        scale = max_h / rotated.height
        rotated = rotated.resize(
            (max(1, int(round(rotated.width * scale))), max(1, int(round(rotated.height * scale)))),
            Image.Resampling.LANCZOS,
        )
    cap_w = rotated.width + 2 * ROW_LABEL_GAP
    layer = Image.new("RGBA", (cap_w, row_h), (0, 0, 0, 0))
    layer.paste(rotated, ((cap_w - rotated.width) // 2, (row_h - rotated.height) // 2), rotated)
    return layer, cap_w


def prepend_row_labels(rows: list[torch.Tensor], labels: list[str]) -> list[torch.Tensor]:
    font = _load_font(ROW_LABEL_FONT_SIZE)
    layers, widths = [], []
    for row, label in zip(rows, labels):
        layer, w = _vertical_caption_layer(label, row_h=int(row.shape[1]), font=font)
        layers.append(layer)
        widths.append(w)
    label_w = max(widths)
    out = []
    for row, layer in zip(rows, layers):
        canvas = Image.new("RGBA", (label_w, int(row.shape[1])), ROW_LABEL_BG)
        canvas.paste(layer, ((label_w - layer.width) // 2, 0), layer)
        out.append(torch.cat([_chw_from_pil(canvas), row], dim=2))
    return out


def build_column_header(col_ws: list[int], labels: list[str], left_gutter: int) -> torch.Tensor:
    font = _load_font(COL_LABEL_FONT_SIZE)
    cells: list[Image.Image] = []
    max_h = 0
    for w, lab in zip(col_ws, labels):
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = measure.textbbox((0, 0), lab.capitalize(), font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        h = COL_LABEL_PAD_Y * 2 + th
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            ((w - tw) // 2 - bbox[0], COL_LABEL_PAD_Y - bbox[1]),
            lab.capitalize(),
            fill=(0, 0, 0),
            font=font,
        )
        cells.append(canvas)
        max_h = max(max_h, h)
    header = Image.new("RGB", (left_gutter + sum(col_ws), max_h), (255, 255, 255))
    x = left_gutter
    for cell, w in zip(cells, col_ws):
        header.paste(cell, (x, (max_h - cell.height) // 2))
        x += w
    return _chw_from_pil(header)


def assemble_mosaic(
    grid: list[list[tuple[torch.Tensor, torch.Tensor]]],
    *,
    sources: list[torch.Tensor],
    row_labels: list[str],
    col_labels: list[str],
) -> torch.Tensor:
    """``sources[r]`` = source shaded; ``grid[r][c] = (global_shaded, local_heatmap)``."""
    n_rows, n_cols = len(grid), len(grid[0])
    if len(sources) != n_rows:
        raise ValueError(f"sources length {len(sources)} != n_rows {n_rows}")
    g_h, g_w = GLOBAL_PANEL_HW
    l_h, l_w = LOCAL_PANEL_HW
    assert g_h == l_h, "global/local panel heights must match for aligned rows"

    src_panels: list[torch.Tensor] = [
        fit_chw_letterbox(src, out_h=g_h, out_w=g_w) for src in sources
    ]
    src_w = g_w

    pairs: list[list[torch.Tensor]] = []
    for r in range(n_rows):
        row_pairs = []
        for c in range(n_cols):
            g, loc = grid[r][c]
            g_f = fit_chw_letterbox(g, out_h=g_h, out_w=g_w)
            loc_f = fit_fg_to_square_panel(loc, out_h=l_h, out_w=l_w)
            if tuple(loc_f.shape[1:]) != (l_h, l_w):
                raise RuntimeError(f"local panel shape {tuple(loc_f.shape)} != {(3, l_h, l_w)}")
            if tuple(g_f.shape[1:]) != (g_h, g_w):
                raise RuntimeError(f"global panel shape {tuple(g_f.shape)} != {(3, g_h, g_w)}")
            row_pairs.append(torch.cat([g_f, loc_f], dim=2))
        pairs.append(row_pairs)

    col_ws = [max(int(pairs[r][c].shape[2]) for r in range(n_rows)) for c in range(n_cols)]
    row_hs = [
        max(
            int(src_panels[r].shape[1]),
            max(int(pairs[r][c].shape[1]) for c in range(n_cols)),
        )
        for r in range(n_rows)
    ]

    # full column widths including leftmost source
    all_col_ws = [src_w] + col_ws
    all_labels = ["source"] + list(col_labels)
    n_all = len(all_col_ws)

    row_tensors: list[torch.Tensor] = []
    for r in range(n_rows):
        cells: list[torch.Tensor] = [
            pad_chw_to(src_panels[r], out_h=row_hs[r], out_w=src_w),
            torch.ones(3, row_hs[r], REGION_GAP_PX, dtype=torch.float32) * MOSAIC_TRIM_BG,
        ]
        for c in range(n_cols):
            cells.append(pad_chw_to(pairs[r][c], out_h=row_hs[r], out_w=col_ws[c]))
            if c < n_cols - 1:
                cells.append(
                    torch.ones(3, row_hs[r], REGION_GAP_PX, dtype=torch.float32) * MOSAIC_TRIM_BG
                )
        row_tensors.append(torch.cat(cells, dim=2))

    row_tensors = prepend_row_labels(row_tensors, row_labels)
    label_w = int(
        row_tensors[0].shape[2] - sum(all_col_ws) - REGION_GAP_PX * (n_all - 1)
    )

    header_col_ws = [
        all_col_ws[c] + (REGION_GAP_PX if c < n_all - 1 else 0) for c in range(n_all)
    ]
    header = build_column_header(header_col_ws, all_labels, left_gutter=max(label_w, 0))

    max_w = max(int(header.shape[2]), max(int(r.shape[2]) for r in row_tensors))
    pieces: list[torch.Tensor] = [pad_chw_to(header, out_h=int(header.shape[1]), out_w=max_w)]
    for i, row in enumerate(row_tensors):
        pieces.append(pad_chw_to(row, out_h=int(row.shape[1]), out_w=max_w))
        if i < n_rows - 1:
            pieces.append(torch.ones(3, ROW_GAP_PX, max_w) * MOSAIC_TRIM_BG)
    return torch.cat(pieces, dim=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    if args.planmm_run is not None:
        planmm_local = Path(args.planmm_run)
        if planmm_local.name != "generated_local":
            planmm_local = planmm_local / "generated_local"
    else:
        planmm_local = latest_planmm_local_root()

    print(f"[viz_mesh_quality_local] LAMM  = {LAMM_LOCAL}")
    print(f"[viz_mesh_quality_local] PLANMM= {planmm_local}")

    ensure_meshes(
        lamm_root=LAMM_LOCAL,
        planmm_root=planmm_local,
        device=device,
        skip_generate=bool(args.skip_generate),
        force_regenerate=bool(args.force_regenerate),
    )

    face_vids = load_face_vertex_ids(Path(args.region_info))
    region_vids_map = {
        region: load_region_vids(Path(args.region_info), region) for region in REGIONS
    }
    roots = [LAMM_LOCAL, planmm_local]
    sample_specs: list[tuple[int, dict[str, int]]] = [
        (int(args.lamm_sample_id), dict(args.lamm_sample_idx)),
        (int(args.planmm_sample_id), dict(args.planmm_sample_idx)),
    ]
    print(
        f"[viz_mesh_quality_local] LAMM sample=id_{sample_specs[0][0]} "
        f"idx={sample_specs[0][1]}  "
        f"PLANMM sample=id_{sample_specs[1][0]} idx={sample_specs[1][1]}"
    )

    # load meshes + scores
    loaded: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    source_meshes: list[tuple[np.ndarray, np.ndarray]] = []
    all_scores: list[np.ndarray] = []
    adj_cache: dict[int, list[np.ndarray]] = {}

    for root, (sid, region_idxs) in zip(roots, sample_specs):
        # shared source for this row (any region folder works)
        src_path = root / REGIONS[0] / f"id_{sid}_source.obj"
        if not src_path.is_file():
            cands = sorted((root / REGIONS[0]).glob("id_*_source.obj"))
            if not cands:
                raise FileNotFoundError(src_path)
            src_path = cands[0]
        src_verts, src_faces = load_obj_mesh(src_path)
        source_meshes.append((src_verts, src_faces))
        print(f"  [{root.name}/source] {src_path.name}")

        row = []
        for region in REGIONS:
            sidx = int(region_idxs[region])
            path = region_mesh_path(root, region, sid, sidx)
            if not path.is_file():
                # fallback: first available sample in folder
                key = _REGION_NAME_TO_KEY[region]
                cands = sorted((root / region).glob(f"id_*_region_{key}_{region}_sample_*.obj"))
                if not cands:
                    raise FileNotFoundError(path)
                path = cands[0]
            verts, faces = load_obj_mesh(path)
            n = len(verts)
            if n not in adj_cache:
                adj_cache[n] = build_vertex_adjacency(faces, n)
            scores = normal_smoothness_score(verts, faces, adjacency=adj_cache[n])
            row.append((verts, faces, scores))
            all_scores.append(scores)
            print(f"  [{path.parent.parent.name}/{region}] {path.name}  mean={scores.mean():.4f}")
        loaded.append(row)

    stacked = np.concatenate(all_scores)
    vmin, vmax = 0.0, max(float(np.percentile(stacked, HEATMAP_PERCENTILE)), 1e-6)
    print(f"[viz_mesh_quality_local] heatmap scale: [{vmin:.4f}, {vmax:.4f}]")

    grid: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    source_imgs: list[torch.Tensor] = []
    for row_i, (row, (src_verts, src_faces)) in enumerate(zip(loaded, source_meshes)):
        source_imgs.append(
            render_view(
                src_verts,
                src_faces,
                face_vids,
                SOURCE_VIEW,
                res=RENDER_RES_GLOBAL,
                local=False,
            )
        )
        cells: list[tuple[torch.Tensor, torch.Tensor]] = []
        for region, (verts, faces, scores) in zip(REGIONS, row):
            view = REGION_VIEWS[region]
            colors = scores_to_vertex_colors(
                scores,
                vmin=vmin,
                vmax=vmax,
                region_vids=region_vids_map[region],
            )
            global_img = render_view(
                verts, faces, face_vids, view, res=RENDER_RES_GLOBAL, local=False
            )
            local_img = render_view(
                verts,
                faces,
                face_vids,
                view,
                res=RENDER_RES_LOCAL,
                local=True,
                vertex_colors=colors,
                region_vids=region_vids_map[region],
            )
            cells.append((global_img, local_img))
        grid.append(cells)
        print(f"[viz_mesh_quality_local] rendered row {ROW_LABELS[row_i]}")

    mosaic = assemble_mosaic(
        grid,
        sources=source_imgs,
        row_labels=list(ROW_LABELS),
        col_labels=list(REGIONS),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_image(mosaic, str(out))
    print(f"[viz_mesh_quality_local] wrote {out} ({mosaic.shape[2]}x{mosaic.shape[1]} px)")


if __name__ == "__main__":
    main()
