"""Single-sample overfitting experiment for Poisson intrinsic deform encoder.

Trains PoissonDeformEncoder to gradually deform input -> HACKHead(neck_pose) neutral
using mesh-path + gradient-path supervision (not LAMM scale-to-zero).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import trimesh

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import read_yaml
from dataset.hack_dataset import HACKDataset, VAL_HACK_PT
from exp_poisson.train_planmm_ae import (
    build_hackhead_for_mean_shape,
    compute_mean_shape_from_neck_pose,
)
from networks.planmm import load_template_from_config
from networks.poisson_deform_encoder import (
    PoissonDeformEncoder,
    build_encoder_alphas,
    compute_deform_encoder_loss,
    grad_path_targets,
    mesh_path_targets,
)
from utils.helpers import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate Poisson deform encoder (single sample).')
    parser.add_argument(
        '--config',
        type=Path,
        default=_REPO_ROOT / 'exp_poisson' / 'validate_poisson_encoder.yaml',
    )
    parser.add_argument('--sample-idx', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--num-steps', type=int, default=None)
    return parser.parse_args()


def mesh_bbox_span(verts: torch.Tensor) -> torch.Tensor:
    return verts.max(dim=1).values - verts.min(dim=1).values


def save_obj(verts: np.ndarray, faces: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=verts, faces=faces).export(path, file_type='obj')


def log_layer_stats(
    layer_meshes: torch.Tensor,
    mesh_targets: torch.Tensor,
    x_in: torch.Tensor,
    step: int,
) -> None:
    n_layers = layer_meshes.shape[0]
    gt_span = mesh_bbox_span(x_in)[0]
    print(f'[step {step}] per-layer stats:')
    for i in range(n_layers):
        pred = layer_meshes[i, 0]
        tgt = mesh_targets[i, 0]
        l1 = (pred - tgt).abs().mean().item()
        span = mesh_bbox_span(pred.unsqueeze(0))[0]
        ratio = (span.max() / gt_span.max().clamp(min=1e-8)).item()
        print(
            f'  layer {i}: mesh_L1={l1:.6f} '
            f'span=({span[0]:.4f},{span[1]:.4f},{span[2]:.4f}) vs_GT={ratio:.4f}',
        )


def main() -> None:
    args = parse_args()
    config = read_yaml(str(args.config))

    model_cfg = config.get('MODEL', {})
    data_cfg = config.get('DATA', {})
    solver_cfg = config.get('SOLVER', {})
    out_cfg = config.get('OUTPUT', {})

    device_str = args.device or solver_cfg.get('device', '0')
    device = torch.device(
        'cpu' if device_str == 'cpu' or not torch.cuda.is_available()
        else f'cuda:{device_str}',
    )
    seed_everything(int(solver_cfg.get('seed', 42)))

    sample_idx = args.sample_idx if args.sample_idx is not None else int(data_cfg.get('sample_idx', 0))
    packed_pt = Path(data_cfg.get('packed_pt', VAL_HACK_PT))
    if not packed_pt.is_absolute():
        packed_pt = _REPO_ROOT / packed_pt

    dataset = HACKDataset(packed_pt=packed_pt)
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f'sample_idx {sample_idx} out of range [0, {len(dataset)})')

    sample = dataset[sample_idx]
    x_in = sample['verts'].unsqueeze(0).to(device)
    neck_pose = sample['neck_pose']
    if neck_pose.ndim == 1:
        neck_pose = neck_pose.unsqueeze(0)
    neck_pose = neck_pose.to(device)

    template_verts, template_faces = load_template_from_config(model_cfg, device=device)
    hackhead = build_hackhead_for_mean_shape(model_cfg, device)
    x_neutral = compute_mean_shape_from_neck_pose(
        hackhead, neck_pose, template_verts,
    ).to(device)
    if x_neutral.ndim == 2:
        x_neutral = x_neutral.unsqueeze(0)

    poisson_cfg = dict(model_cfg.get('poisson_cfg', {}))
    n_stages = int(model_cfg.get('n_stages', 4))
    enc = PoissonDeformEncoder(
        n_stages=n_stages,
        C_width=int(model_cfg.get('C_width', 128)),
        config=poisson_cfg,
    ).to(device)
    enc.register_template_mesh(template_verts, template_faces)

    alphas = build_encoder_alphas(n_stages, device=device)
    mesh_targets = mesh_path_targets(x_in, x_neutral, alphas)
    _, _, G, _, _ = enc._get_operators(1, device)
    grad_targets = grad_path_targets(x_in, x_neutral, G, alphas)

    num_steps = args.num_steps if args.num_steps is not None else int(solver_cfg.get('num_steps', 1500))
    lr = float(solver_cfg.get('lr', 1e-3))
    grad_weight = float(model_cfg.get('grad_weight', 0.5))
    log_every = int(solver_cfg.get('log_every', 100))

    optimizer = optim.Adam(enc.parameters(), lr=lr)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_base = Path(out_cfg.get('save_dir', 'results/poisson_encoder_feasibility'))
    if not save_base.is_absolute():
        save_base = _REPO_ROOT / save_base
    run_dir = save_base / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    faces_np = template_faces.detach().cpu().numpy()
    save_obj(x_in[0].detach().cpu().numpy(), faces_np, run_dir / 'input.obj')
    save_obj(x_neutral[0].detach().cpu().numpy(), faces_np, run_dir / 'neutral.obj')

    print(
        f'[validate_poisson_encoder] sample_idx={sample_idx} device={device} '
        f'arch={enc.ENCODER_ARCH} n_stages={n_stages} steps={num_steps} '
        f'params={sum(p.numel() for p in enc.parameters())/1e6:.2f}M',
    )
    print(f'run_dir={run_dir}')

    best_last_l1 = float('inf')
    best_state = None
    for step in range(1, num_steps + 1):
        enc.train()
        optimizer.zero_grad(set_to_none=True)

        layer_meshes, layer_grads = enc(x_in, x_neutral=x_neutral, alphas=alphas)
        loss, stats = compute_deform_encoder_loss(
            layer_meshes,
            layer_grads,
            mesh_targets,
            grad_targets,
            vertex_mass=enc.vertex_mass,
            grad_weight=grad_weight,
        )

        if not torch.isfinite(loss):
            print(f'[warn] step {step}: non-finite loss, stopping')
            break

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            last_l1 = (layer_meshes[-1, 0] - x_neutral[0]).abs().mean().item()
            if last_l1 < best_last_l1:
                best_last_l1 = last_l1
                best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}

        if step == 1 or step % log_every == 0 or step == num_steps:
            enc.eval()
            with torch.no_grad():
                layer_meshes, layer_grads = enc(x_in, x_neutral=x_neutral, alphas=alphas)
            print(
                f'step {step}/{num_steps} loss={loss.item():.6f} '
                f'mesh={stats["loss_mesh"].item():.6f} grad={stats["loss_grad"].item():.6f} '
                f'last_vs_neutral={last_l1:.6f}',
            )
            log_layer_stats(layer_meshes, mesh_targets, x_in, step)

    if best_state is not None:
        enc.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    enc.eval()
    with torch.no_grad():
        layer_meshes, _ = enc(x_in, x_neutral=x_neutral, alphas=alphas)

    for i in range(layer_meshes.shape[0]):
        save_obj(
            layer_meshes[i, 0].detach().cpu().numpy(),
            faces_np,
            run_dir / f'layer_{i:02d}.obj',
        )
    for i in range(mesh_targets.shape[0]):
        save_obj(
            mesh_targets[i, 0].detach().cpu().numpy(),
            faces_np,
            run_dir / f'target_{i:02d}.obj',
        )

    ckpt_path = run_dir / 'encoder.pth'
    torch.save({
        'model': enc.state_dict(),
        'config': config,
        'sample_idx': sample_idx,
        'best_last_l1': best_last_l1,
        'n_stages': n_stages,
        'encoder_arch': enc.ENCODER_ARCH,
    }, ckpt_path)

    final_l1 = (layer_meshes[-1, 0] - x_neutral[0]).abs().mean().item()
    mid_idx = layer_meshes.shape[0] // 2
    mid_span = mesh_bbox_span(layer_meshes[mid_idx, 0].unsqueeze(0))[0].max().item()
    gt_span = mesh_bbox_span(x_in)[0].max().item()

    print(f'[done] checkpoint={ckpt_path}')
    print(f'final L1 vs neutral={final_l1:.6f} (best={best_last_l1:.6f})')
    print(f'mid-layer max span={mid_span:.4f} (input max span={gt_span:.4f}, ratio={mid_span/gt_span:.4f})')
    if final_l1 < 0.02 or best_last_l1 < 0.02:
        print('[pass] L1 vs neutral < 0.02')
    else:
        print('[warn] L1 vs neutral >= 0.02')
    if mid_span > 0.1 * gt_span:
        print('[pass] mid-layer scale not collapsed')
    else:
        print('[warn] mid-layer scale may be collapsed')


if __name__ == '__main__':
    main()
