"""Compare per-layer NJF vs shared global NJF PoissonDeformEncoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

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
    mesh_path_targets,
)
from networks.PoissonNet import NJFHead, PoissonBlock


class PoissonDeformEncoderPerLayer(torch.nn.Module):
    """Legacy arch: one NJFHead per Poisson stage."""

    ENCODER_ARCH = 'per_layer_njf'

    def __init__(self, n_stages: int = 4, C_width: int = 128, config: dict | None = None):
        super().__init__()
        config = dict(config or {})
        self.n_stages = int(n_stages)
        self.n_layers = self.n_stages + 1
        self.C_width = C_width
        self.gradient_checkpoint = bool(config.get('gradient_checkpoint', False))
        self.proj_in = torch.nn.Linear(3, C_width)
        self.blocks = torch.nn.ModuleList([
            PoissonBlock(
                in_c=C_width, out_c=C_width, width=C_width, extra_feats=0, config=config,
            )
            for _ in range(self.n_stages)
        ])
        njf_cfg = dict(config)
        self.njf_heads = torch.nn.ModuleList([
            NJFHead(in_c=C_width, out_c=3, width=C_width, config=njf_cfg)
            for _ in range(self.n_stages)
        ])
        self.register_buffer('template_verts', torch.zeros(0, 3), persistent=False)
        self.register_buffer('template_faces', torch.zeros(0, 3, dtype=torch.long), persistent=False)
        self.register_buffer('vertex_mass', torch.zeros(0), persistent=False)
        self._ops_cache = {}

    def register_template_mesh(self, verts, faces):
        enc = PoissonDeformEncoder(self.n_stages, self.C_width)
        enc.register_template_mesh(verts, faces)
        self.template_verts = enc.template_verts
        self.template_faces = enc.template_faces
        self.vertex_mass = enc.vertex_mass
        self._ops_cache = {}

    def _get_operators(self, batch_size, device):
        enc = PoissonDeformEncoder(self.n_stages, self.C_width)
        enc.template_verts = self.template_verts
        enc.template_faces = self.template_faces
        enc.vertex_mass = self.vertex_mass
        enc._ops_cache = self._ops_cache
        return enc._get_operators(batch_size, device)

    def forward(self, x_in, x_neutral=None, alphas=None, **kwargs):
        B, device = x_in.shape[0], x_in.device
        vertex_mass, solver, G, M, faces = self._get_operators(B, device)
        if alphas is None:
            alphas = build_encoder_alphas(self.n_stages, device=device)
        mesh_targets = None if x_neutral is None else mesh_path_targets(x_in, x_neutral, alphas)

        layer_meshes = [x_in]
        layer_grads = []
        mesh = x_in
        for stage_idx, (block, njf) in enumerate(zip(self.blocks, self.njf_heads)):
            layer_idx = stage_idx + 1
            mesh_anchor = mesh_targets[stage_idx] if mesh_targets is not None else mesh
            feat = self.proj_in(mesh_anchor)
            feat, _ = block(feat, M, G, solver, faces, vertex_mass, extra_features=None)
            delta, grads = njf(feat, M, G, solver, faces, vertex_mass)
            mesh = mesh_targets[layer_idx] + delta if mesh_targets is not None else mesh + delta
            layer_meshes.append(mesh)
            layer_grads.append(grads)
        return torch.stack(layer_meshes, dim=0), torch.stack(layer_grads, dim=0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--per-layer-ckpt',
        type=Path,
        default=_REPO_ROOT / 'results/poisson_encoder_feasibility/20260707_112530/encoder.pth',
    )
    parser.add_argument(
        '--global-ckpt',
        type=Path,
        default=_REPO_ROOT / 'results/poisson_encoder_feasibility/20260707_131932/encoder.pth',
    )
    parser.add_argument('--device', type=str, default='0')
    return parser.parse_args()


def mesh_bbox_span(verts: torch.Tensor) -> torch.Tensor:
    return verts.max(dim=1).values - verts.min(dim=1).values


def eval_checkpoint(model, ckpt_path: Path, device: torch.device) -> dict:
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = payload.get('config') or read_yaml(
        str(_REPO_ROOT / 'exp_poisson' / 'validate_poisson_encoder.yaml'),
    )
    model_cfg = config.get('MODEL', {})
    data_cfg = config.get('DATA', {})

    sample_idx = int(payload.get('sample_idx', data_cfg.get('sample_idx', 0)))
    packed_pt = Path(data_cfg.get('packed_pt', VAL_HACK_PT))
    if not packed_pt.is_absolute():
        packed_pt = _REPO_ROOT / packed_pt

    dataset = HACKDataset(packed_pt=packed_pt)
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

    n_stages = int(payload.get('n_stages', model_cfg.get('n_stages', 4)))
    poisson_cfg = dict(model_cfg.get('poisson_cfg', {}))
    model = model(
        n_stages=n_stages,
        C_width=int(model_cfg.get('C_width', 128)),
        config=poisson_cfg,
    ).to(device)
    model.load_state_dict(payload['model'])
    model.register_template_mesh(template_verts, template_faces)
    model.eval()

    alphas = build_encoder_alphas(n_stages, device=device)
    mesh_targets = mesh_path_targets(x_in, x_neutral, alphas)
    with torch.no_grad():
        layer_meshes, _ = model(x_in, x_neutral=x_neutral, alphas=alphas)

    gt_span = mesh_bbox_span(x_in)[0].max().item()
    layer_l1 = []
    layer_vs_gt = []
    for i in range(layer_meshes.shape[0]):
        pred = layer_meshes[i, 0]
        tgt = mesh_targets[i, 0]
        layer_l1.append((pred - tgt).abs().mean().item())
        span = mesh_bbox_span(pred.unsqueeze(0))[0].max().item()
        layer_vs_gt.append(span / max(gt_span, 1e-8))

    final_l1 = (layer_meshes[-1, 0] - x_neutral[0]).abs().mean().item()
    mid_idx = layer_meshes.shape[0] // 2
    mid_vs_gt = layer_vs_gt[mid_idx]

    return {
        'arch': payload.get('encoder_arch', getattr(model, 'ENCODER_ARCH', 'unknown')),
        'ckpt': str(ckpt_path),
        'params_m': sum(p.numel() for p in model.parameters()) / 1e6,
        'best_last_l1_saved': payload.get('best_last_l1'),
        'final_l1_vs_neutral': final_l1,
        'layer_mesh_l1': layer_l1,
        'layer_vs_gt': layer_vs_gt,
        'mid_vs_gt': mid_vs_gt,
    }


def main():
    args = parse_args()
    device = torch.device(
        'cpu' if args.device == 'cpu' or not torch.cuda.is_available()
        else f'cuda:{args.device}',
    )

    rows = [
        eval_checkpoint(PoissonDeformEncoderPerLayer, args.per_layer_ckpt, device),
        eval_checkpoint(PoissonDeformEncoder, args.global_ckpt, device),
    ]

    print('=== PoissonDeformEncoder arch comparison ===')
    for row in rows:
        print(f"\n[{row['arch']}]")
        print(f"  checkpoint: {row['ckpt']}")
        print(f"  params: {row['params_m']:.2f}M")
        print(f"  saved best_last_l1: {row['best_last_l1_saved']:.6f}")
        print(f"  eval final_l1_vs_neutral: {row['final_l1_vs_neutral']:.6f}")
        print(f"  mid-layer vs_GT: {row['mid_vs_gt']:.4f}")
        print('  per-layer mesh_L1 vs target:')
        for i, (l1, ratio) in enumerate(zip(row['layer_mesh_l1'], row['layer_vs_gt'])):
            print(f'    L{i}: mesh_L1={l1:.6f} vs_GT={ratio:.4f}')


if __name__ == '__main__':
    main()
