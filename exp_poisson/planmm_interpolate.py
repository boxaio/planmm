"""
Linear interpolation between two random global latent codes (same MVN as
``planmm_generate_samples.py``), decode with PLANMM ``decode``, save OBJ meshes
along the path, render each frame, crop near-white margins, horizontally
concatenate, and save to ``figures/PLANMM_HACK_interpolate_global.png``.
"""

from __future__ import annotations

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

from exp_poisson.planmm_generate_samples import (
    build_planmm_net,
    resolve_checkpoint_path,
    resolve_config,
    resolve_gaussian_path,
)
from render.mesh_render import render_mesh
from utils.torch_utils import load_from_checkpoint

# ----- hard-coded run settings (edit here) -----
CONFIG_FILE = _REPO_ROOT / 'exp_poisson' / 'planmm_hack.yaml'
CHECKPOINT: str | None = None  # None -> newest timestamp run under CHECKPOINT.save_dir
MACHINE = 'local'
DEVICE_INDEX = '0'  # CUDA device id, or "cpu"

NUM_STEPS = 8
K_STD = 1.0  # multiplier for MVN(mu, K_STD * Sigma) when sampling endpoints
RNG_SEED = 1776519

# None -> use {checkpoint_dir}/interpolated_latent
OUT_DIR: Path | None = None

RENDER_RES = 512
FIGURES_STRIPE_OUT = _REPO_ROOT / 'figures' / 'PLANMM_HACK_interpolate_global.png'

MOSAIC_TRIM_BG = 1.0
MOSAIC_TRIM_TOL = 0.035
MOSAIC_TRIM_PAD = 4


def crop_chw_near_uniform_bg(
    img: torch.Tensor,
    *,
    bg: float = MOSAIC_TRIM_BG,
    tol: float = MOSAIC_TRIM_TOL,
    pad: int = MOSAIC_TRIM_PAD,
) -> torch.Tensor:
    """Crop near-uniform background borders from a ``[C,H,W]`` image."""
    if img.ndim != 3:
        raise ValueError(f'expected CHW image, got shape {tuple(img.shape)}')
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


def align_stripe_heights_centervpad(
    stripes: list[torch.Tensor],
    *,
    bg: float = MOSAIC_TRIM_BG,
) -> list[torch.Tensor]:
    """Center-pad stripes to a common height for horizontal concatenation."""
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


def save_mesh_vertices(verts: np.ndarray, faces: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=verts, faces=faces).export(path, file_type='obj')


def main() -> None:
    device = (
        torch.device('cpu')
        if DEVICE_INDEX == 'cpu'
        else torch.device(f'cuda:{DEVICE_INDEX}' if torch.cuda.is_available() else 'cpu')
    )

    config_path = Path(CONFIG_FILE)
    checkpoint_path = resolve_checkpoint_path(config_path, CHECKPOINT)
    print(f'[planmm_interpolate] checkpoint={checkpoint_path}')
    config = resolve_config(config_path, checkpoint_path)
    config['MACHINE'] = MACHINE

    checkpoint_dir = checkpoint_path.parent
    out_dir = OUT_DIR if OUT_DIR is not None else checkpoint_dir / 'interpolated_latent'
    out_dir.mkdir(parents=True, exist_ok=True)

    gaussian_path = resolve_gaussian_path(config, checkpoint_dir, gaussian_arg=None)
    if not gaussian_path.is_file():
        raise FileNotFoundError(
            f'{gaussian_path}: run planmm_generate_samples.py --fit_gaussian first '
            'or place gaussian_id.pickle beside the checkpoint.'
        )
    with open(gaussian_path, 'rb') as handle:
        gid = pickle.load(handle, encoding='latin1')
    mu = np.asarray(gid['mean'], dtype=np.float64)
    sigma = np.asarray(gid['sigma'], dtype=np.float64)

    rng = np.random.default_rng(seed=RNG_SEED)
    z0_np = rng.multivariate_normal(mu, K_STD * sigma).astype(np.float32)
    z1_np = rng.multivariate_normal(mu, K_STD * sigma).astype(np.float32)

    net = build_planmm_net(config['MODEL'], device)
    load_from_checkpoint(net, str(checkpoint_path), partial_restore=True, device=device)
    net.eval()

    faces = net.template_faces.detach().cpu().numpy()

    num_steps = max(int(NUM_STEPS), 2)
    alphas = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=torch.float32).view(-1, 1)
    z0 = torch.from_numpy(z0_np).to(device=device).view(1, -1)
    z1 = torch.from_numpy(z1_np).to(device=device).view(1, -1)
    z_batch = (1.0 - alphas) * z0 + alphas * z1
    z_id = z_batch.unsqueeze(1)

    with torch.no_grad():
        verts_batch = net.decode(z_id)[-1].detach().cpu().numpy()

    denom = max(num_steps - 1, 1)
    stripes: list[torch.Tensor] = []

    for i in range(num_steps):
        va = np.asarray(verts_batch[i], dtype=np.float64)
        t_val = float(i / denom)
        obj_path = out_dir / f'interp_{i:04d}_t{t_val:.4f}.obj'
        save_mesh_vertices(va, faces, obj_path)

        img_chw = render_mesh(
            va,
            faces,
            res=RENDER_RES,
            filename=None,
            flat_shading=False,
            center_mesh=True,
            fill_ratio=0.9,
            camera_distance=1.8,
            yfov=np.pi / 3.0,
        )
        stripes.append(crop_chw_near_uniform_bg(img_chw))
        print(f'[planmm_interpolate] rendered step {i + 1}/{num_steps} -> {obj_path.name}')

    mosaic = torch.cat(align_stripe_heights_centervpad(stripes), dim=2)
    FIGURES_STRIPE_OUT.parent.mkdir(parents=True, exist_ok=True)
    save_image(mosaic, str(FIGURES_STRIPE_OUT))

    print(
        f'[planmm_interpolate] checkpoint={checkpoint_path}'
    )
    print(f'[planmm_interpolate] endpoints from MVN (K_STD={K_STD}, seed={RNG_SEED})')
    print(f'[planmm_interpolate] saved {num_steps} meshes under {out_dir}')
    print(
        f'[planmm_interpolate] wrote horizontal strip '
        f'({mosaic.shape[2]}x{mosaic.shape[1]} px, {num_steps} frames) -> {FIGURES_STRIPE_OUT}'
    )


if __name__ == '__main__':
    main()
