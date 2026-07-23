"""
Generate random local manipulations for a selected facial region, save OBJ meshes under
``planmm_hack_manipulation/<run>/generated_local/{region}``, then render a horizontal mosaic.

Generation logic follows ``planmm_generate_local.py``; rendering mirrors ``viz_lamm_generate_local.py``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torchvision.utils import save_image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import read_yaml
from dataset.hack_seg_fids import REGION_NAMES
from exp_poisson.planmm_generate_local import (
    _generate_global_meshes,
    _resolve_device,
    _resolve_gaussian_path,
    get_random_displacements,
    save_mesh,
)
from exp_poisson.planmm_generate_samples import build_planmm_net
from exp_poisson.planmm_prepare_inference import (
    resolve_checkpoint_path,
    resolve_run_config,
)
from render.mesh_render import render_mesh
from utils.helpers import seed_everything
from utils.mesh import read_obj
from utils.torch_utils import load_from_checkpoint

# ----- defaults (override via CLI) -----
CONFIG_FILE = _REPO_ROOT / "exp_poisson" / "planmm_hack_manipulation.yaml"
CHECKPOINT = None  # latest best_alpha_max under CHECKPOINT.save_dir
MACHINE = "local"
DEVICE_INDEX = "0"

NUM_SAMPLES = 1
NUM_RANDOM_GENERATIONS = 6
K_STD = 2.5

# RNG_SEED = 75996
RNG_SEED = 66666

# DEFAULT_REGION = "nose"
# DEFAULT_REGION = "eyes"
DEFAULT_REGION = "Chin"

RENDER_RES = 512
RENDER_FILL_RATIO = 0.97
RENDER_YFOV = np.pi / 3.0
MOSAIC_TRIM_BG = 1.0
MOSAIC_TRIM_TOL = 0.035
MOSAIC_TRIM_PAD = 4
# 图像空间二次裁剪：放大画面上方脸部区域（参考 viz_lamm_generate_local）
FACE_CROP_WIDTH_RATIO = 0.74
FACE_CROP_HEIGHT_RATIO = 0.65
FACE_CROP_CENTER_Y = 0.34

# 用于 3D 居中（排除 Neck / Skull，避免整张头拉远相机）
_FACE_REGION_NAMES = (
    "LeftFace", "RightFace", "Ears", "Chin", "Forehead", "Eyes", "Nose", "Mouth",
)

_REGION_NAME_TO_KEY = {name.lower(): key for key, name in REGION_NAMES.items()}


def parse_args():
    region_choices = sorted({name.lower() for name in REGION_NAMES.values()})
    parser = argparse.ArgumentParser(description="Viz local PLANMM region manipulations.")
    parser.add_argument(
        "--region",
        type=str,
        default=DEFAULT_REGION,
        help=(
            "Region name (e.g. nose, eyes, mouth) or patch id 0-9. "
            f"Available names: {', '.join(region_choices)}"
        ),
    )
    parser.add_argument(
        "--config_file",
        type=Path,
        default=CONFIG_FILE,
        help="Path to planmm_hack_manipulation.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT,
        help=(
            "Checkpoint .pth, run directory, or save_dir base. "
            "Default: latest timestamp under CHECKPOINT.save_dir "
            "(prefer best_alpha_max.pth)."
        ),
    )
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--num_random_generations", type=int, default=NUM_RANDOM_GENERATIONS)
    parser.add_argument("--k_std", type=float, default=K_STD)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--device", type=str, default=DEVICE_INDEX)
    parser.add_argument(
        "--gaussian_id",
        type=str,
        default=None,
        help="Optional gaussian_id.pickle (default: beside checkpoint).",
    )
    return parser.parse_args()


def resolve_region(spec: str) -> tuple[int, str]:
    """Parse region patch id (0-9) or name (e.g. nose, eyes). Returns (key, lowercase_name)."""
    spec = spec.strip()
    if spec.isdigit():
        key = int(spec)
        if key not in REGION_NAMES:
            raise ValueError(f"invalid region id {key}, expected 0-9")
        return key, REGION_NAMES[key].lower()

    name = spec.lower()
    if name in _REGION_NAME_TO_KEY:
        key = _REGION_NAME_TO_KEY[name]
        return key, name

    raise ValueError(
        f"unknown region {spec!r}; use patch id 0-9 or name: "
        f"{', '.join(sorted(_REGION_NAME_TO_KEY.keys()))}"
    )


def region_out_dir(run_dir: Path, region_name: str) -> Path:
    return run_dir / "generated_local" / region_name


def region_figures_out(region_name: str) -> Path:
    return _REPO_ROOT / "figures" / f"PLANMM_HACK_generate_local_{region_name}.png"


def load_face_vertex_ids(region_info_file: Path | None = None) -> np.ndarray:
    if region_info_file is None:
        cfg = read_yaml(CONFIG_FILE)
        region_info_file = Path(cfg["MODEL"]["region_info_file"])
    with open(region_info_file, "rb") as f:
        region_info = pickle.load(f)
    vids: list[int] = []
    for name in _FACE_REGION_NAMES:
        if name in region_info:
            vids.extend(np.asarray(region_info[name]["vids"], dtype=np.int64).reshape(-1).tolist())
    if not vids:
        raise RuntimeError(f"no face region vids found in {region_info_file}")
    return np.unique(np.asarray(vids, dtype=np.int64))


def crop_chw_near_uniform_bg(
    img: torch.Tensor,
    *,
    bg: float = MOSAIC_TRIM_BG,
    tol: float = MOSAIC_TRIM_TOL,
    pad: int = MOSAIC_TRIM_PAD,
) -> torch.Tensor:
    if img.ndim != 3:
        raise ValueError(f"expected CHW image, got shape {tuple(img.shape)}")
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
    width_ratio: float = FACE_CROP_WIDTH_RATIO,
    height_ratio: float = FACE_CROP_HEIGHT_RATIO,
    center_y_frac: float = FACE_CROP_CENTER_Y,
    pad: int = MOSAIC_TRIM_PAD,
    bg: float = MOSAIC_TRIM_BG,
    tol: float = MOSAIC_TRIM_TOL,
) -> torch.Tensor:
    """Crop trimmed render to upper-center face band so the face fills most of the frame."""
    if img.ndim != 3:
        raise ValueError(f"expected CHW image, got shape {tuple(img.shape)}")
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
    cx = x0 + fg_w * 0.5
    cy = y0 + fg_h * center_y_frac

    px0 = int(np.floor(cx - crop_w * 0.5 - pad))
    py0 = int(np.floor(cy - crop_h * 0.5 - pad))
    px1 = int(np.ceil(cx + crop_w * 0.5 + pad))
    py1 = int(np.ceil(cy + crop_h * 0.5 + pad))

    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(w, max(px1, px0 + 1))
    py1 = min(h, max(py1, py0 + 1))
    return img[:, py0:py1, px0:px1]


def align_stripe_heights_centervpad(
    stripes: list[torch.Tensor],
    *,
    bg: float = MOSAIC_TRIM_BG,
) -> list[torch.Tensor]:
    if not stripes:
        return stripes
    max_h = max(s.shape[1] for s in stripes)
    out: list[torch.Tensor] = []
    for t in stripes:
        _, h, _ = t.shape
        if h >= max_h:
            out.append(t)
            continue
        pad_total = max_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        out.append(F.pad(t, (0, 0, pad_top, pad_bottom), value=bg))
    return out


def load_obj_mesh(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(obj_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return verts, faces


def render_mesh_frame(
    verts: np.ndarray,
    faces: np.ndarray,
    face_vids: np.ndarray,
) -> torch.Tensor:
    """Render with face-centered framing and image-space face crop for a tight facial view."""
    face_ctr = verts[face_vids].mean(axis=0)
    verts_face = verts - face_ctr
    img_chw = render_mesh(
        verts_face,
        faces,
        res=RENDER_RES,
        filename=None,
        flat_shading=False,
        center_mesh=False,
        fill_ratio=RENDER_FILL_RATIO,
        camera_distance=None,
        yfov=RENDER_YFOV,
    )
    img_chw = crop_chw_near_uniform_bg(img_chw)
    return crop_chw_to_face_region(img_chw)


def generate_region_local(
    device: torch.device,
    out_dir: Path,
    region_key: int,
    region_name: str,
    num_samples: int,
    num_random_generations: int,
    k_std: float,
    seed: int,
    config_file: Path,
    checkpoint_arg: str | None,
    gaussian_id: str | None,
) -> list[Path]:
    """Generate source + random region manipulations; return OBJ paths in render order."""
    checkpoint_path = resolve_checkpoint_path(config_file, checkpoint_arg)
    config = resolve_run_config(config_file, checkpoint_path)
    config["MACHINE"] = MACHINE
    run_dir = checkpoint_path.parent

    control_lms = {int(k): v for k, v in config["MODEL"]["control_vertices"].items()}
    if region_key not in control_lms:
        raise KeyError(f"region key {region_key} ({region_name}) not in control_vertices")

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    template_path = config["MODEL"].get(
        "reference_obj",
        str(_REPO_ROOT / "dataset" / "hack_template.obj"),
    )
    mesh = read_obj(str(template_path))
    faces = mesh.fvs

    delta_stats_path = run_dir / "displacement_stats.pickle"
    if not delta_stats_path.is_file():
        raise FileNotFoundError(
            f"{delta_stats_path}: run planmm_prepare_inference.py first "
            f"(or planmm_generate_local.py --prepare_files)."
        )
    with open(delta_stats_path, "rb") as file:
        delta_stats = pickle.load(file)

    gaussian_path = _resolve_gaussian_path(run_dir, gaussian_id)
    data_loader = _generate_global_meshes(
        config,
        checkpoint_path,
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
    load_from_checkpoint(net, str(checkpoint_path), partial_restore=True, device=str(device))
    net.eval()

    obj_paths: list[Path] = []
    num_saved = 0

    with torch.no_grad():
        for sample in data_loader:
            remaining = num_samples - num_saved
            if remaining <= 0:
                break

            source = sample["verts"].to(device)
            if source.shape[0] > remaining:
                source = source[:remaining]

            for i in range(source.shape[0]):
                num_saved += 1
                sample_id = num_saved
                print(
                    f"[viz_planmm_generate_local] region={region_name} "
                    f"identity {sample_id}/{num_samples}"
                )

                x = source[i].unsqueeze(0)
                source_np = x[0].detach().cpu().numpy()
                source_path = out_dir / f"id_{sample_id}_source.obj"
                save_mesh(source_np, faces, source_path)
                obj_paths.append(source_path)

                for m in range(num_random_generations):
                    delta = get_random_displacements(
                        delta_stats, region_key, control_lms, k_std, device,
                    )
                    out = net((x, delta))[-1]
                    out_np = out[0].detach().cpu().numpy()
                    sample_path = out_dir / (
                        f"id_{sample_id}_region_{region_key}_{region_name}_sample_{m}.obj"
                    )
                    save_mesh(out_np, faces, sample_path)
                    obj_paths.append(sample_path)
                    print(f"[viz_planmm_generate_local] saved {sample_path.name}")

    return obj_paths


def render_obj_strip(
    obj_paths: list[Path],
    stripe_out: Path,
    face_vids: np.ndarray,
) -> None:
    stripes: list[torch.Tensor] = []
    for obj_path in obj_paths:
        if not obj_path.is_file():
            raise FileNotFoundError(f"missing mesh: {obj_path}")
        verts, faces = load_obj_mesh(obj_path)
        stripes.append(render_mesh_frame(verts, faces, face_vids))
        png_path = obj_path.with_suffix(".png")
        save_image(stripes[-1], str(png_path))
        print(f"[viz_planmm_generate_local] rendered {obj_path.name} -> {png_path.name}")

    mosaic = torch.cat(align_stripe_heights_centervpad(stripes), dim=2)
    stripe_out.parent.mkdir(parents=True, exist_ok=True)
    save_image(mosaic, str(stripe_out))
    print(
        f"[viz_planmm_generate_local] wrote horizontal strip "
        f"({mosaic.shape[2]}x{mosaic.shape[1]} px, {len(obj_paths)} frames) -> {stripe_out}"
    )


def main() -> None:
    args = parse_args()
    region_key, region_name = resolve_region(args.region)
    config_file = Path(args.config_file)
    checkpoint_path = resolve_checkpoint_path(config_file, args.checkpoint)
    run_dir = checkpoint_path.parent
    out_dir = region_out_dir(run_dir, region_name)
    figures_out = region_figures_out(region_name)
    device = _resolve_device(args.device)
    face_vids = load_face_vertex_ids()

    print(f"[viz_planmm_generate_local] checkpoint={checkpoint_path}")
    print(f"[viz_planmm_generate_local] out_dir={out_dir}")

    obj_paths = generate_region_local(
        device,
        out_dir,
        region_key,
        region_name,
        args.num_samples,
        args.num_random_generations,
        args.k_std,
        args.seed,
        config_file,
        args.checkpoint,
        args.gaussian_id,
    )
    render_obj_strip(obj_paths, figures_out, face_vids)


if __name__ == "__main__":
    main()
