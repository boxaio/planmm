"""Train PLANMMSimp: simplified Poisson path-anchored VAE on HACKDataset.

All hyperparameters are inline (no yaml). Usage:
    PYTHONPATH=. python exp_poisson/train_planmm_simp.py
    PYTHONPATH=. python exp_poisson/train_planmm_simp.py --device 0 --epochs 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.hack_dataset import HACKDataset, TRAIN_HACK_PT, VAL_HACK_PT
from exp_poisson.lr_scheduler import build_scheduler
from networks.planmm_simp import PLANMMSimp, load_template_from_config
from networks.poisson_deform_encoder import (
    build_planmm_path_alphas,
    build_planmm_path_targets,
    compute_path_supervision_loss,
    grad_path_targets,
)
from utils.helpers import seed_everything, to_np

CONFIG = {
    'seed': 78986,
    'device': '0',
    'model': {
        'dim': 512,
        'C_width': 128,
        'encoder_depth': 3,
        'decoder_depth': 1,
        'Npatches': 10,
        'use_neck_pose': True,
        'seam_vertex_loss_boost': 1.5,
        'encoder_gradient_checkpoint': False,
        'decoder_gradient_checkpoint': False,
        'operator_high_precision': False,
        'poisson_cfg': {
            'cmlp_nlayers': 2,
            'cmlp_modulate': True,
            'mass_norm': True,
            'inner_prod_features': False,
            'drop_path': 0.0,
            'dropout_p': 0.0,
            'mlp_norm': False,
            'gradient_checkpoint': False,
            'multilayer_readout': True,
        },
    },
    'data': {
        'train_pt': str(TRAIN_HACK_PT),
        'val_pt': str(VAL_HACK_PT),
        'sources': ['FaceScape', 'ImHead'],
        'batch_size': 4,
        'eval_batch_size': 4,
        'num_workers': 4,
        'use_subset': True,
        'subset_ratio': 0.05,
        'eval_subset_ratio': 0.2,
    },
    'solver': {
        'num_epochs': 20,
        'lr': 5e-4,
        'lr_min': 1e-6,
        'lr_start': 1e-8,
        'lr_scheduler': 'cosine',
        'num_cycles': 1,
        'num_warmup_epochs': 1,
        'weight_decay': 1e-4,
        'clip_grad_norm': 0.3,
        'gradient_accumulation_steps': 4,
    },
    'loss': {
        'recon_weight': 15.0,
        'enc_path_weight': 0.3,
        'grad_weight': 0.02,
        'kl_weight': 1e-4,
        'gen_aux_weight': 0.25,
    },
    'checkpoint': {
        'save_dir': 'results/planmm_simp_vae',
        'save_optimizer_state': True,
        'viz_num_samples': 4,
    },
}


def _neck_pose_to_bone83(neck_pose: torch.Tensor) -> torch.Tensor:
    if neck_pose.ndim == 1:
        neck_pose = neck_pose.unsqueeze(0)
    b = neck_pose.shape[0]
    device, dtype = neck_pose.device, neck_pose.dtype
    if neck_pose.shape[-1] == 6:
        bone01 = neck_pose.view(b, 2, 3)
    elif neck_pose.shape[-2:] == (2, 3):
        bone01 = neck_pose
    elif neck_pose.shape[-2:] == (8, 3):
        return neck_pose
    else:
        raise ValueError(f'unsupported neck_pose shape {tuple(neck_pose.shape)}')
    out = torch.zeros(b, 8, 3, device=device, dtype=dtype)
    out[:, :2] = bone01
    return out


def build_hackhead_for_mean_shape(use_neck_pose: bool = True):
    if not use_neck_pose:
        return None
    from meshes.hackhead import HACKHead

    hackhead = HACKHead(use_teeth=False, device='cpu')
    hackhead.eval()
    for param in hackhead.parameters():
        param.requires_grad_(False)
    return hackhead


@torch.no_grad()
def compute_mean_shape_from_neck_pose(hackhead, neck_pose, fallback_template):
    if hackhead is None:
        b = neck_pose.shape[0] if neck_pose.ndim > 1 else 1
        return fallback_template.unsqueeze(0).expand(b, -1, -1)
    device = fallback_template.device
    neck_b83 = _neck_pose_to_bone83(neck_pose.to(device))
    b = neck_b83.shape[0]
    dtype = neck_b83.dtype
    shape = torch.zeros(b, hackhead.n_shape_params, device='cpu', dtype=dtype)
    expression = torch.zeros(b, hackhead.n_expr_params, device='cpu', dtype=dtype)
    tau = torch.zeros(b, 1, device='cpu', dtype=dtype)
    gamma = torch.ones(b, 1, device='cpu', dtype=dtype)
    return hackhead(
        shape=shape,
        expression=expression,
        neck_pose=neck_b83.cpu(),
        tau=tau,
        gamma=gamma,
        return_lmks_mp=False,
        normalize=True,
    )[0].to(device)


def build_planmm_simp_net(model_cfg: dict, device: torch.device) -> PLANMMSimp:
    net = PLANMMSimp(model_cfg).to(device)
    template_verts, template_faces = load_template_from_config(model_cfg, device=device)
    net.register_template_mesh(template_verts, template_faces)
    return net


def get_dataloaders(data_cfg: dict, seed: int):
    sources = data_cfg.get('sources')
    batch_size = int(data_cfg['batch_size'])
    eval_batch_size = int(data_cfg.get('eval_batch_size', batch_size))
    num_workers = int(data_cfg.get('num_workers', 4))

    loaders = {}
    for split_name, packed_pt, fraction_key, batch in (
        ('training', data_cfg['train_pt'], 'subset_ratio', batch_size),
        ('eval', data_cfg['val_pt'], 'eval_subset_ratio', eval_batch_size),
    ):
        dataset = HACKDataset(packed_pt=packed_pt, sources=sources)
        fraction = float(data_cfg.get(fraction_key, 1.0))
        if data_cfg.get('use_subset', False) and split_name == 'training':
            fraction = float(data_cfg.get('subset_ratio', fraction))
        if split_name == 'eval' and data_cfg.get('eval_subset_ratio') is not None:
            fraction = float(data_cfg['eval_subset_ratio'])

        if fraction < 1.0:
            n = len(dataset)
            n_subset = max(1, int(n * fraction))
            split_seed = seed if split_name == 'training' else seed + 1
            indices = np.random.default_rng(split_seed).choice(
                n, size=n_subset, replace=False,
            ).tolist()
            dataset = Subset(dataset, indices)
            print(f'[{split_name}] subset: {n_subset}/{n} ({fraction * 100:.0f}%)')

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch,
            shuffle=(split_name == 'training'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_name == 'training'),
            persistent_workers=num_workers > 0,
        )
    return loaders


def _faces_numpy(faces) -> np.ndarray:
    if isinstance(faces, torch.Tensor):
        f = faces[0] if faces.ndim == 3 else faces
        return f.detach().cpu().numpy()
    f = faces[0] if isinstance(faces, (list, tuple)) else faces
    return np.asarray(f)


def save_obj_mesh(vertices: np.ndarray, faces: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in faces:
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')


@torch.no_grad()
def save_eval_visuals(
    gt_verts: np.ndarray,
    pred_verts: np.ndarray,
    faces: np.ndarray,
    save_dir: str | Path,
    abs_step: int,
    stem: str = 'sample',
) -> Path:
    viz_dir = Path(save_dir) / 'eval_viz' / f'epoch_{abs_step:04d}'
    viz_dir.mkdir(parents=True, exist_ok=True)

    tag = str(stem).replace('/', '_')
    save_obj_mesh(gt_verts, faces, viz_dir / f'gt_{tag}.obj')
    save_obj_mesh(pred_verts, faces, viz_dir / f'recon_{tag}.obj')

    from render.mesh_render import add_text, render_mesh_on, render_vertex_error_map
    from torchvision.utils import save_image

    render_gt = add_text(render_mesh_on(gt_verts, faces), caption='GT')
    render_pred = add_text(render_mesh_on(pred_verts, faces), caption='Recon')
    render_err = add_text(
        render_vertex_error_map(pred_verts, gt_verts, faces),
        caption='Error',
    )
    compare_path = viz_dir / f'compare_{tag}.png'
    save_image(torch.cat([render_gt, render_pred, render_err], dim=-1), str(compare_path))
    return viz_dir


def _sample_stem(sample, idx: int) -> str:
    stem = sample.get('stem', 'sample')
    if isinstance(stem, (list, tuple)):
        stem = stem[idx]
    elif isinstance(stem, torch.Tensor):
        if stem.numel() == 1:
            stem = str(stem.item())
        else:
            stem = str(stem[idx].item())
    return str(stem)


def _optimizer_steps_per_epoch(num_batches: int, accum: int) -> int:
    return max(1, (num_batches + accum - 1) // accum)


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]['lr'])


@torch.no_grad()
def _reconstruct_batch(net, vs, neck_pose, hackhead):
    x_neutral = compute_mean_shape_from_neck_pose(hackhead, neck_pose, net.template_verts)
    id_token, _, _, patch_tokens, _, _ = net.encode(vs, x_neutral=x_neutral)
    recon = net.decode(
        id_token, patch_tokens=patch_tokens, x_neutral=x_neutral, neck_pose=neck_pose,
    )[-1]
    return recon, id_token, patch_tokens


@torch.no_grad()
def save_eval_viz_batch(
    net,
    eval_loader,
    device,
    save_dir: str | Path,
    epoch_idx: int,
    seed: int,
    hackhead=None,
    num_samples: int = 4,
) -> None:
    dataset = eval_loader.dataset
    dataset_len = len(dataset)
    if dataset_len <= 0:
        return

    net.eval()
    indices = [
        (int(seed) + i * max(1, dataset_len // num_samples)) % dataset_len
        for i in range(num_samples)
    ]

    for viz_idx in indices:
        sample = dataset[viz_idx]
        vs = sample['verts'].to(device)
        if vs.ndim == 2:
            vs = vs.unsqueeze(0)
        neck_pose = sample['neck_pose'].to(device)
        if neck_pose.ndim == 1:
            neck_pose = neck_pose.unsqueeze(0)

        recon, _, _ = _reconstruct_batch(net, vs, neck_pose, hackhead)
        faces_np = _faces_numpy(sample['faces'])
        stem = _sample_stem(sample, 0)
        out_dir = save_eval_visuals(
            to_np(vs[0]), to_np(recon[0]), faces_np, save_dir, epoch_idx, stem=stem,
        )
        l1 = _mass_weighted_vertex_l1(recon, vs, net.vertex_loss_weight).item()
        print(f'  [viz] idx={viz_idx} stem={stem} recon_l1={l1:.5f} -> {out_dir}')


@torch.no_grad()
def save_generation_viz(
    net,
    batch,
    device,
    save_dir: str | Path,
    epoch_idx: int,
    hackhead,
) -> None:
    net.eval()
    vs = batch['verts'].to(device)
    neck_pose = batch['neck_pose'].to(device)
    if vs.ndim == 2:
        vs = vs.unsqueeze(0)
    if neck_pose.ndim == 1:
        neck_pose = neck_pose.unsqueeze(0)
    b = vs.shape[0]
    x_neutral = compute_mean_shape_from_neck_pose(hackhead, neck_pose, net.template_verts)
    z = torch.randn(b, 1, net.dim, device=device)
    gen = net.generate_from_latent(z, neck_pose=neck_pose, x_neutral=x_neutral)

    viz_dir = Path(save_dir) / 'eval_viz' / f'epoch_{epoch_idx:04d}'
    viz_dir.mkdir(parents=True, exist_ok=True)
    faces_np = _faces_numpy(batch['faces'])

    from render.mesh_render import add_text, render_mesh_on
    from torchvision.utils import save_image

    for i in range(min(b, 2)):
        stem = _sample_stem(batch, i)
        tag = str(stem).replace('/', '_')
        save_obj_mesh(to_np(gen[i]), faces_np, viz_dir / f'random_{tag}.obj')
        render = add_text(render_mesh_on(to_np(gen[i]), faces_np), caption='Random z')
        save_image(render, str(viz_dir / f'random_{tag}.png'))


def save_checkpoint(
    path: Path,
    net,
    optimizer,
    cfg: dict,
    epoch: int,
    val_metrics: dict,
    *,
    include_optimizer: bool,
) -> None:
    payload = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'config': cfg,
        'val_metrics': val_metrics,
    }
    if include_optimizer:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(payload, path)


def save_model_weights(path: Path, net, cfg: dict, epoch: int, val_metrics: dict) -> None:
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'config': cfg,
            'val_metrics': val_metrics,
        },
        path,
    )


def configure_training_runtime(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')


def _mass_weighted_vertex_l1(pred, target, vertex_mass):
    mass_w = vertex_mass.to(pred.device).clamp(min=1e-8).view(1, -1, 1)
    return (torch.abs(pred - target) * mass_w).sum() / mass_w.sum()


@torch.no_grad()
def evaluate(net, loader, device, hackhead, loss_cfg: dict):
    net.eval()
    totals = {'loss': 0.0, 'recon_l1': 0.0, 'kl': 0.0, 'gen_aux': 0.0}
    n_batches = 0
    for sample in loader:
        vs = sample['verts'].to(device)
        neck_pose = sample['neck_pose'].to(device)
        x_neutral = compute_mean_shape_from_neck_pose(hackhead, neck_pose, net.template_verts)
        id_token, mu, logvar, patch_tokens, _, _ = net.encode(vs, x_neutral=x_neutral)
        dec_out = net.decode(
            id_token, patch_tokens=patch_tokens, x_neutral=x_neutral, neck_pose=neck_pose,
        )
        recon_l1 = _mass_weighted_vertex_l1(dec_out[-1], vs, net.vertex_loss_weight)
        kl = net.kl_divergence(mu, logvar)
        gen_patches = net.latent_to_patch_tokens(id_token)
        gen_recon = net.decode(
            id_token, patch_tokens=gen_patches, x_neutral=x_neutral, neck_pose=neck_pose,
        )[-1]
        gen_aux = _mass_weighted_vertex_l1(gen_recon, vs, net.vertex_loss_weight)
        loss = (
            float(loss_cfg['recon_weight']) * recon_l1
            + float(loss_cfg['kl_weight']) * kl
            + float(loss_cfg['gen_aux_weight']) * gen_aux
        )
        totals['loss'] += float(loss.item())
        totals['recon_l1'] += float(recon_l1.item())
        totals['kl'] += float(kl.item())
        totals['gen_aux'] += float(gen_aux.item())
        n_batches += 1
    if n_batches == 0:
        return totals
    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def generation_smoke(net, batch, device, hackhead):
    """Sample z ~ N(0,I) and decode; report finite check + template L1."""
    net.eval()
    vs = batch['verts'].to(device)
    neck_pose = batch['neck_pose'].to(device)
    b = vs.shape[0]
    x_neutral = compute_mean_shape_from_neck_pose(hackhead, neck_pose, net.template_verts)
    z = torch.randn(b, 1, net.dim, device=device)
    gen = net.generate_from_latent(z, neck_pose=neck_pose, x_neutral=x_neutral)
    template_l1 = (gen - x_neutral).abs().mean().item()
    gt_l1 = (vs - x_neutral).abs().mean().item()
    return {
        'finite': bool(torch.isfinite(gen).all().item()),
        'template_l1': template_l1,
        'gt_neutral_l1': gt_l1,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Train PLANMMSimp VAE')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--subset-ratio', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CONFIG
    if args.device is not None:
        cfg['device'] = args.device
    if args.epochs is not None:
        cfg['solver']['num_epochs'] = args.epochs
    if args.subset_ratio is not None:
        cfg['data']['subset_ratio'] = args.subset_ratio
        cfg['data']['use_subset'] = True
    if args.batch_size is not None:
        cfg['data']['batch_size'] = args.batch_size
    if args.save_dir is not None:
        cfg['checkpoint']['save_dir'] = args.save_dir

    seed_everything(cfg['seed'])
    device_str = cfg['device']
    if device_str == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device_str}')

    save_dir = Path(cfg['checkpoint']['save_dir'])
    if not save_dir.is_absolute():
        save_dir = _REPO_ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    configure_training_runtime(device)
    net = build_planmm_simp_net(cfg['model'], device)
    hackhead = build_hackhead_for_mean_shape(cfg['model'].get('use_neck_pose', True))
    loaders = get_dataloaders(cfg['data'], cfg['seed'])

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'PLANMMSimp trainable params: {n_params:,} ({n_params / 1e6:.2f}M)')

    accum = int(cfg['solver']['gradient_accumulation_steps'])
    opt_steps_per_epoch = _optimizer_steps_per_epoch(len(loaders['training']), accum)
    print(
        f'batches/epoch={len(loaders["training"])}, '
        f'optimizer_steps/epoch={opt_steps_per_epoch}, accum={accum}'
    )

    optimizer = optim.AdamW(
        net.parameters(),
        lr=float(cfg['solver']['lr']),
        weight_decay=float(cfg['solver']['weight_decay']),
    )
    scheduler_cfg = {
        'SOLVER': {
            **cfg['solver'],
            'lr_base': cfg['solver']['lr'],
        },
    }
    scheduler = build_scheduler(scheduler_cfg, optimizer, opt_steps_per_epoch)

    clip_grad = float(cfg['solver']['clip_grad_norm'])
    loss_cfg = cfg['loss']
    best_val = float('inf')
    save_optimizer = bool(cfg['checkpoint'].get('save_optimizer_state', True))
    viz_num_samples = int(cfg['checkpoint'].get('viz_num_samples', 4))

    global_step = 0
    for epoch in range(int(cfg['solver']['num_epochs'])):
        net.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_enc_path = 0.0
        epoch_gen_aux = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loaders['training'], desc=f'epoch {epoch + 1}/{cfg["solver"]["num_epochs"]}')
        for step_idx, sample in enumerate(pbar):
            is_accum_boundary = ((step_idx + 1) % accum == 0) or (step_idx + 1 == len(loaders['training']))
            step_out = _run_train_step(
                net, sample, device, hackhead, loss_cfg, optimizer,
                do_zero_grad=(step_idx % accum == 0),
                do_optimizer_step=is_accum_boundary,
                loss_scale=1.0 / accum,
                clip_grad=clip_grad if is_accum_boundary else None,
            )
            if scheduler is not None and is_accum_boundary:
                scheduler.step_update(global_step)
                global_step += 1

            epoch_loss += step_out['loss']
            epoch_recon += step_out['recon_l1']
            epoch_enc_path += step_out['enc_path']
            epoch_gen_aux += step_out['gen_aux']
            n_steps += 1
            pbar.set_postfix(
                loss=f"{step_out['loss']:.4f}",
                recon=f"{step_out['recon_l1']:.5f}",
                path=f"{step_out['enc_path']:.4f}",
                lr=f"{_current_lr(optimizer):.2e}",
            )

        avg_train_loss = epoch_loss / max(n_steps, 1)
        avg_train_recon = epoch_recon / max(n_steps, 1)
        avg_enc_path = epoch_enc_path / max(n_steps, 1)
        avg_gen_aux = epoch_gen_aux / max(n_steps, 1)
        val_metrics = evaluate(net, loaders['eval'], device, hackhead, loss_cfg)
        print(
            f'epoch {epoch + 1}: lr={_current_lr(optimizer):.2e} '
            f'train_loss={avg_train_loss:.4f} train_recon={avg_train_recon:.5f} '
            f'train_path={avg_enc_path:.4f} train_gen_aux={avg_gen_aux:.5f} '
            f'val_loss={val_metrics["loss"]:.4f} val_recon={val_metrics["recon_l1"]:.5f} '
            f'val_kl={val_metrics["kl"]:.4f}'
        )

        gen_smoke = generation_smoke(net, next(iter(loaders['eval'])), device, hackhead)
        print(
            f'  gen_smoke: finite={gen_smoke["finite"]} '
            f'template_l1={gen_smoke["template_l1"]:.5f} gt_neutral_l1={gen_smoke["gt_neutral_l1"]:.5f}'
        )

        save_eval_viz_batch(
            net, loaders['eval'], device, save_dir, epoch + 1, cfg['seed'],
            hackhead=hackhead, num_samples=viz_num_samples,
        )
        save_generation_viz(
            net, next(iter(loaders['eval'])), device, save_dir, epoch + 1, hackhead,
        )

        save_model_weights(save_dir / 'last_model.pth', net, cfg, epoch + 1, val_metrics)
        save_checkpoint(
            save_dir / 'last.pth', net, optimizer, cfg, epoch + 1, val_metrics,
            include_optimizer=save_optimizer,
        )
        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            save_model_weights(save_dir / 'best_model.pth', net, cfg, epoch + 1, val_metrics)
            save_checkpoint(
                save_dir / 'best.pth', net, optimizer, cfg, epoch + 1, val_metrics,
                include_optimizer=save_optimizer,
            )
            print(
                f'  saved best_model.pth (~{n_params * 4 / 1e6:.1f} MB weights only) '
                f'and best.pth (full state) val_loss={best_val:.4f}'
            )

    print(f'Training complete. Checkpoints in {save_dir}')
    print(
        'Note: best.pth is ~3x model size because AdamW stores momentum + variance. '
        'Use best_model.pth for inference (~19 MB for 4.8M params).'
    )


def _run_train_step(
    net,
    sample,
    device,
    hackhead,
    loss_cfg,
    optimizer,
    do_zero_grad,
    do_optimizer_step,
    loss_scale,
    clip_grad,
):
    vs = sample['verts'].to(device, non_blocking=True)
    neck_pose = sample['neck_pose'].to(device, non_blocking=True)
    x_neutral = compute_mean_shape_from_neck_pose(hackhead, neck_pose, net.template_verts)

    id_token, mu, logvar, patch_tokens, enc_out, enc_grads = net.encode(
        vs, x_neutral=x_neutral, return_grads=True,
    )
    dec_out, _ = net.decode(
        id_token, patch_tokens=patch_tokens, x_neutral=x_neutral,
        neck_pose=neck_pose, return_grads=True,
    )

    enc_tgt, _, _, _ = build_planmm_path_targets(
        vs, x_neutral, net.encoder_depth, net.decoder_depth, device=vs.device,
    )
    enc_alphas, _, _ = build_planmm_path_alphas(
        net.encoder_depth, net.decoder_depth, device=vs.device,
    )
    _, _, g_op, _, _ = net._get_batch_operators(vs.shape[0], vs.device)
    grad_targets = grad_path_targets(vs, x_neutral, g_op, enc_alphas)
    enc_layer_w = torch.ones(net.encoder_depth + 1, device=vs.device)
    enc_path_loss, _ = compute_path_supervision_loss(
        enc_out,
        enc_grads,
        enc_tgt,
        grad_targets[: net.encoder_depth + 1],
        vertex_mass=net.vertex_loss_weight,
        layer_weights=enc_layer_w,
        grad_weight=float(loss_cfg['grad_weight']),
        grad_weight_decoder=0.0,
        encoder_depth=net.encoder_depth,
    )

    recon_l1 = _mass_weighted_vertex_l1(dec_out[-1], vs, net.vertex_loss_weight)
    kl = net.kl_divergence(mu, logvar)
    loss = (
        float(loss_cfg['enc_path_weight']) * enc_path_loss
        + float(loss_cfg['recon_weight']) * recon_l1
        + float(loss_cfg['kl_weight']) * kl
    )

    gen_aux_val = 0.0
    if float(loss_cfg.get('gen_aux_weight', 0.0)) > 0.0:
        gen_patches = net.latent_to_patch_tokens(id_token)
        gen_recon = net.decode(
            id_token, patch_tokens=gen_patches, x_neutral=x_neutral, neck_pose=neck_pose,
        )[-1]
        gen_aux = _mass_weighted_vertex_l1(gen_recon, vs, net.vertex_loss_weight)
        loss = loss + float(loss_cfg['gen_aux_weight']) * gen_aux
        gen_aux_val = float(gen_aux.detach().item())

    if do_zero_grad:
        optimizer.zero_grad(set_to_none=True)
    (loss * loss_scale).backward()
    if do_optimizer_step:
        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
        optimizer.step()

    return {
        'loss': float(loss.detach().item()),
        'recon_l1': float(recon_l1.detach().item()),
        'enc_path': float(enc_path_loss.detach().item()),
        'kl': float(kl.detach().item()),
        'gen_aux': gen_aux_val,
    }


if __name__ == '__main__':
    main()
