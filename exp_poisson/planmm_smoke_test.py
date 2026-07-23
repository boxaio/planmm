"""Fast PLANMM smoke test: 1 epoch on subset + mesh quality metrics.

Usage:
    python exp_poisson/planmm_smoke_test.py
    python exp_poisson/planmm_smoke_test.py --skip-train --checkpoint results/planmm_smoke/best.pth
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import read_yaml
from exp_poisson.planmm_generate_samples import sample_latent_codes, sample_neck_poses
from exp_poisson.train_planmm_ae import build_planmm_net, get_dataloaders
from utils.torch_utils import load_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='PLANMM smoke test (1 epoch + mesh QA)')
    parser.add_argument(
        '--config',
        type=str,
        default=str(_REPO_ROOT / 'exp_poisson' / 'planmm_smoke.yaml'),
    )
    parser.add_argument('--skip-train', action='store_true', help='Only evaluate an existing checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--num-export', type=int, default=3, help='Number of eval meshes to export as OBJ')
    parser.add_argument('--num-random-gen', type=int, default=5, help='Random latent samples to export')
    parser.add_argument('--k_std', type=float, default=1.5, help='Latent sampling std for diversity test')
    return parser.parse_args()


def load_cross_patch_edges(region_info_path: Path) -> np.ndarray:
    with open(region_info_path, 'rb') as handle:
        region_info = pickle.load(handle)
    region_info = {k: v for k, v in region_info.items() if 'inner' not in k}
    v2p: dict[int, set[int]] = {}
    for pi, data in enumerate(region_info.values()):
        for v in data['vids']:
            v2p.setdefault(v, set()).add(pi)

    from utils.mesh import read_obj

    faces = read_obj(str(_REPO_ROOT / 'dataset' / 'hack_template.obj'), tri=True).fvs
    seam_edges = []
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            if v2p.get(a, {a}) != v2p.get(b, {b}):
                seam_edges.append((min(a, b), max(a, b)))
    seam_edges = np.array(sorted(set(seam_edges)), dtype=np.int64)
    return seam_edges


def mesh_quality_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    seam_edges: np.ndarray,
    coarse_mask: torch.Tensor,
    template_edge_len: dict[tuple[int, int], float] | None = None,
) -> dict[str, float]:
    """Per-sample metrics: overall L1, coarse-region L1, edge stretch."""
    per_v_l1 = (pred - gt).abs().mean(dim=-1)
    overall_l1 = float(per_v_l1.mean().cpu())
    coarse_l1 = float(per_v_l1[coarse_mask].mean().cpu()) if coarse_mask.any() else float('nan')
    max_l1 = float(per_v_l1.max().cpu())

    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    seam_stretch = []
    coarse_stretch = []
    coarse_ids = set(coarse_mask.nonzero(as_tuple=True)[0].tolist())
    for a, b in seam_edges:
        pred_len = np.linalg.norm(pred_np[a] - pred_np[b])
        gt_len = np.linalg.norm(gt_np[a] - gt_np[b]).clip(min=1e-8)
        seam_stretch.append(pred_len / gt_len)
        if a in coarse_ids or b in coarse_ids:
            coarse_stretch.append(pred_len / gt_len)

    return {
        'l1': overall_l1,
        'coarse_l1': coarse_l1,
        'max_l1': max_l1,
        'seam_stretch_mean': float(np.mean(seam_stretch)) if len(seam_stretch) else float('nan'),
        'seam_stretch_p95': float(np.percentile(seam_stretch, 95)) if len(seam_stretch) else float('nan'),
        'coarse_seam_stretch_max': float(max(coarse_stretch)) if coarse_stretch else float('nan'),
    }


def save_mesh(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces).export(str(path), file_type='obj')


@torch.no_grad()
def evaluate_and_export(net, eval_loader, device, out_dir: Path, seam_edges, num_export: int):
    net.eval()
    all_metrics = []
    all_latent_metrics = []
    faces_np = net.template_faces.detach().cpu().numpy()
    exported = 0

    for sample in eval_loader:
        vs = sample['verts'].to(device)
        neck_pose = sample['neck_pose'].to(device)
        recon = net(vs, neck_pose=neck_pose)[-1]
        id_token, _, _, _ = net.encode(vs, x_neutral=x_neutral)
        x_neutral = net.template_verts.unsqueeze(0).expand(vs.shape[0], -1, -1)
        latent_recon = net.generate_from_latent(
            id_token, neck_pose=neck_pose, x_neutral=x_neutral,
        )
        coarse_mask = net._local_edge_scale.to(device) > 1.25

        for i in range(vs.shape[0]):
            template = net.template_verts.to(device)
            m = mesh_quality_metrics(recon[i], vs[i], seam_edges, coarse_mask)
            m['recon_disp_from_template'] = (recon[i] - template).abs().mean().item()
            gt_disp = (vs[i] - template).abs().mean().item()
            m['gt_disp_from_template'] = gt_disp
            m['recon_disp_ratio'] = m['recon_disp_from_template'] / max(gt_disp, 1e-8)
            all_metrics.append(m)
            lm = mesh_quality_metrics(latent_recon[i], vs[i], seam_edges, coarse_mask)
            lm['recon_disp_from_template'] = (latent_recon[i] - template).abs().mean().item()
            lm['gt_disp_from_template'] = gt_disp
            lm['recon_disp_ratio'] = lm['recon_disp_from_template'] / max(gt_disp, 1e-8)
            all_latent_metrics.append(lm)

            if exported < num_export:
                stem = f'smoke_{exported:02d}'
                save_mesh(vs[i].cpu().numpy(), faces_np, out_dir / f'{stem}_gt.obj')
                save_mesh(recon[i].cpu().numpy(), faces_np, out_dir / f'{stem}_recon.obj')
                save_mesh(latent_recon[i].cpu().numpy(), faces_np, out_dir / f'{stem}_latent.obj')
                exported += 1

    keys = all_metrics[0].keys()
    summary = {k: float(np.mean([m[k] for m in all_metrics if np.isfinite(m[k])])) for k in keys}
    latent_summary = {
        f'latent_{k}': float(np.mean([m[k] for m in all_latent_metrics if np.isfinite(m[k])]))
        for k in keys
    }
    summary.update(latent_summary)
    return summary, exported


@torch.no_grad()
def evaluate_random_generation(
    net,
    eval_loader,
    device,
    out_dir: Path,
    seam_edges,
    num_samples: int,
    k_std: float,
    seed: int = 42,
) -> dict[str, float]:
    """Sample random latents from encoded eval batch; measure diversity + neck quality."""
    net.eval()
    sample = next(iter(eval_loader))
    vs = sample['verts'].to(device)
    neck_pose = sample['neck_pose'].to(device)
    batch_n = min(vs.shape[0], num_samples)
    vs = vs[:batch_n]
    neck_pose = neck_pose[:batch_n]

    id_token, _, _, _ = net.encode(vs)
    flat = id_token.reshape(batch_n, -1).detach().cpu().numpy()
    mu = flat.mean(axis=0)
    centered = flat - mu
    sigma = centered.T @ centered / max(batch_n - 1, 1)

    neck_np_arr = neck_pose.detach().cpu().numpy()
    mu_neck = neck_np_arr.mean(axis=0)
    if batch_n > 1:
        sigma_neck = np.cov(neck_np_arr, rowvar=False)
    else:
        sigma_neck = np.diag(np.full(mu_neck.shape[0], 1e-4, dtype=np.float64))

    rng = np.random.default_rng(seed)
    z = sample_latent_codes(mu, sigma, num_samples, k_std, 'diagonal', rng)
    neck_np = sample_neck_poses(
        {'mean': mu_neck.astype(np.float32), 'sigma': sigma_neck.astype(np.float32)},
        num_samples,
        k_std * 0.8,
        'diagonal',
        rng,
    )
    id_t = torch.from_numpy(
        z.reshape(num_samples, net.n_latent_tokens, net.bottleneck_dim).astype(np.float32),
    ).to(device)
    neck_t = torch.from_numpy(neck_np).to(device)
    meshes = net.reconstruct_from_latent(id_t, neck_pose=neck_t).detach().cpu()

    faces_np = net.template_faces.detach().cpu().numpy()
    coarse_mask = net._local_edge_scale > 1.25
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, verts in enumerate(meshes):
        save_mesh(verts.numpy(), faces_np, out_dir / f'random_{i:02d}.obj')

    pair_diffs = []
    coarse_stretches = []
    coarse_ids = set(coarse_mask.nonzero(as_tuple=True)[0].tolist())
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            pair_diffs.append((meshes[i] - meshes[j]).abs().mean().item())
        pred_np = meshes[i].numpy()
        for a, b in seam_edges:
            if a in coarse_ids or b in coarse_ids:
                pred_len = np.linalg.norm(pred_np[a] - pred_np[b])
                tmpl = net.template_verts.cpu().numpy()
                tmpl_len = np.linalg.norm(tmpl[a] - tmpl[b]).clip(min=1e-8)
                coarse_stretches.append(pred_len / tmpl_len)

    z_pair = float(np.linalg.norm(z[0] - z[1])) if num_samples >= 2 else 0.0
    return {
        'gen_latent_pair_dist': z_pair,
        'gen_mesh_pair_l1_mean': float(np.mean(pair_diffs)) if pair_diffs else 0.0,
        'gen_coarse_seam_stretch_max': float(max(coarse_stretches)) if coarse_stretches else float('nan'),
    }


def main():
    args = parse_args()
    config_path = Path(args.config)
    config = read_yaml(str(config_path))
    save_dir = Path(config['CHECKPOINT']['save_dir'])
    if not save_dir.is_absolute():
        save_dir = _REPO_ROOT / save_dir

    if not args.skip_train:
        cmd = [
            sys.executable,
            str(_REPO_ROOT / 'exp_poisson' / 'train_planmm_ae.py'),
            '--config', str(config_path),
            '--device', args.device,
        ]
        print('Running smoke training:', ' '.join(cmd))
        subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))

    checkpoint = Path(args.checkpoint) if args.checkpoint else save_dir / 'best.pth'
    if not checkpoint.is_file():
        checkpoint = save_dir / 'last.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'No checkpoint found under {save_dir}')

    device = torch.device(
        f'cuda:{args.device}' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu',
    )
    eval_loader = get_dataloaders(config, mode='dp', rank=0, world_size=1)[0]['eval']
    net = build_planmm_net(config['MODEL'], device)
    load_from_checkpoint(net, str(checkpoint), partial_restore=True, device=device)

    region_info_path = Path(config['MODEL']['region_info_file'])
    seam_edges = load_cross_patch_edges(region_info_path)
    out_dir = save_dir / 'smoke_meshes'
    summary, n_export = evaluate_and_export(
        net, eval_loader, device, out_dir, seam_edges, args.num_export,
    )
    gen_dir = save_dir / 'smoke_random_gen'
    gen_summary = evaluate_random_generation(
        net,
        eval_loader,
        device,
        gen_dir,
        seam_edges,
        num_samples=args.num_random_gen,
        k_std=args.k_std,
        seed=int(config.get('SOLVER', {}).get('seed', 42)),
    )
    summary.update(gen_summary)

    print('\n=== PLANMM Smoke Test Summary ===')
    print(f'encoder: Poisson(path_anchor), decoder: direct NJF, dim={config["MODEL"]["dim"]}')
    print(f'checkpoint: {checkpoint}')
    print(f'exported: {n_export} mesh pairs -> {out_dir}')
    print(f'random gen: {args.num_random_gen} meshes -> {gen_dir}')
    for k, v in summary.items():
        print(f'  {k}: {v:.6f}')

    # Heuristic pass/fail thresholds for quick iteration
    ok = (
        summary['max_l1'] < 0.15
        and summary['coarse_seam_stretch_max'] < 3.0
        and summary['seam_stretch_p95'] < 2.0
        and summary.get('latent_coarse_seam_stretch_max', 999.0) < 3.5
        and summary.get('gen_mesh_pair_l1_mean', 0.0) > 0.005
        and summary.get('recon_disp_from_template', 0.0) > 0.01
        and summary.get('recon_disp_ratio', 0.0) > 0.3
    )
    print(f'\nSmoke QA: {"PASS" if ok else "FAIL"} (heuristic thresholds)')
    if not ok:
        print(
            '  Tips: retrain from scratch after architecture changes; '
            'check poisson_n_blocks and multilayer loss weights.'
        )
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
