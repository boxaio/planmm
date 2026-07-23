"""PLANMM manipulation training (LAMM-style handle deltas + hybrid mesh readout)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import copy_yaml, get_params_values, read_yaml
from dataset.hack_dataset import HACKDataset, TRAIN_HACK_PT, VAL_HACK_PT
from exp_poisson.loss import get_loss
from exp_poisson.lr_scheduler import build_scheduler
from exp_poisson.planmm_generate_samples import resolve_checkpoint_path
from exp_poisson.summaries import write_mean_summaries
from exp_poisson.train_planmm_ae import (
    atomic_torch_save,
    resolve_save_dir,
    setup_runtime,
    unwrap_net,
    wrap_model,
)
from networks.planmm import PLANMM, load_template_from_config
from render.mesh_render import add_text, render_mesh_on, render_vertex_error_map
from utils.helpers import seed_everything, to_np
from utils.torch_utils import get_net_trainable_params, load_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='PLANMM manipulation training')
    parser.add_argument(
        '--config',
        type=str,
        default=str(_REPO_ROOT / 'exp_poisson' / 'planmm_hack_manipulation.yaml'),
        help='configuration (.yaml) file to use',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='GPU ids for DataParallel, e.g. "0,1". Ignored when using --distributed.',
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Use DistributedDataParallel (launch with torchrun --nproc_per_node=2 ...).',
    )
    return parser.parse_args()


def _resolve_dataset_sources(datasets_cfg: dict) -> list[str] | None:
    sources = datasets_cfg.get('sources')
    if sources is None:
        return None
    if isinstance(sources, str):
        sources = [sources]
    return list(sources)


def get_dataloaders(config, mode, rank, world_size):
    dataloaders = {}
    seed = config.get('SOLVER', {}).get('seed', 42)
    datasets_cfg = config['DATASETS']
    train_cfg = datasets_cfg['train']
    eval_cfg = datasets_cfg['eval']
    sources = _resolve_dataset_sources(datasets_cfg)
    global_batch = train_cfg['batch_size']
    if mode == 'ddp':
        if global_batch % world_size != 0:
            raise ValueError(
                f"batch_size ({global_batch}) must be divisible by world_size ({world_size})"
            )
        per_gpu_batch = global_batch // world_size
    else:
        per_gpu_batch = global_batch

    train_fraction = train_cfg.get('data_fraction', 1.0)
    if train_cfg.get('use_subset', False):
        train_fraction = float(train_cfg.get('subset_ratio', 0.1))
    eval_fraction = eval_cfg.get('data_fraction', train_fraction)
    if eval_cfg.get('use_subset', train_cfg.get('use_subset', False)):
        eval_fraction = float(eval_cfg.get('subset_ratio', 0.1))

    for split_name, packed_pt, split_cfg, fraction, shuffle in (
        ('training', TRAIN_HACK_PT, train_cfg, train_fraction, True),
        ('eval', VAL_HACK_PT, eval_cfg, eval_fraction, False),
    ):
        dataset = HACKDataset(packed_pt=packed_pt, sources=sources)
        if fraction < 1.0:
            n = len(dataset)
            n_subset = max(1, int(n * fraction))
            split_seed = seed if split_name == 'training' else seed + 1
            indices = np.random.default_rng(split_seed).choice(
                n, size=n_subset, replace=False,
            ).tolist()
            dataset = Subset(dataset, indices)
            print(f"[get_dataloaders] {split_name} subset: {n_subset}/{n} ({fraction * 100:.0f}%)")

        sampler = None
        loader_shuffle = shuffle
        if mode == 'ddp':
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
            )
            loader_shuffle = False

        eval_batch = split_cfg['batch_size']
        if mode == 'ddp' and split_name == 'eval' and eval_batch % world_size != 0:
            eval_batch = max(world_size, (eval_batch // world_size) * world_size)

        batch_size = per_gpu_batch if split_name == 'training' else (
            eval_batch // world_size if mode == 'ddp' else eval_batch
        )

        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=loader_shuffle,
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split_name == 'training'),
        )

    return dataloaders, global_batch, per_gpu_batch


def build_planmm_net(model_cfg: dict, device: torch.device) -> PLANMM:
    net = PLANMM(model_cfg).to(device)
    template_verts, template_faces = load_template_from_config(model_cfg, device=device)
    net.register_template_mesh(template_verts, template_faces)
    return net


def _faces_numpy(faces):
    if isinstance(faces, torch.Tensor):
        f = faces[0] if faces.ndim == 3 else faces
        return f.detach().cpu().numpy()
    f = faces[0] if isinstance(faces, (list, tuple)) else faces
    return np.asarray(f)


def save_obj_mesh(vertices: np.ndarray, faces: np.ndarray, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for v in vertices:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for face in faces:
            f.write(f'f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n')


def _sample_stem(sample, idx: int):
    stem = sample.get('stem', 'sample')
    if isinstance(stem, (list, tuple)):
        stem = stem[idx]
    elif isinstance(stem, torch.Tensor):
        if stem.numel() == 1:
            stem = str(stem.item())
        else:
            stem = str(stem[idx].item())
    return str(stem)


@torch.no_grad()
def save_eval_visuals(
    gt_verts: np.ndarray,
    pred_verts: np.ndarray,
    faces: np.ndarray,
    save_dir: str | Path,
    abs_step: int,
    stem: str = 'sample',
    tag: str = 'manip',
) -> Path:
    viz_dir = Path(save_dir) / 'eval_viz' / f'step_{abs_step:06d}'
    viz_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = str(stem).replace('/', '_')
    save_obj_mesh(gt_verts, faces, viz_dir / f'target_{tag}_{safe_stem}.obj')
    save_obj_mesh(pred_verts, faces, viz_dir / f'recon_{tag}_{safe_stem}.obj')

    render_gt = add_text(render_mesh_on(gt_verts, faces), caption=f'Target ({tag})')
    render_pred = add_text(render_mesh_on(pred_verts, faces), caption=f'Recon ({tag})')
    render_err = add_text(
        render_vertex_error_map(pred_verts, gt_verts, faces),
        caption='Error',
    )
    compare_path = viz_dir / f'compare_{tag}_{safe_stem}.png'
    save_image(torch.cat([render_gt, render_pred, render_err], dim=-1), str(compare_path))
    return viz_dir


def train_step(net, sample, loss_fn, alpha_min_max, lambda_target, optimizer, device, clip_grad_norm=None):
    """Single training step — PLANMM manipulation with 2B batch split."""
    core = unwrap_net(net)
    vs = sample['verts'].to(device, non_blocking=True)
    vt = torch.roll(vs, 1, 0)
    b = vs.shape[0]
    control_vids = core.control_vid_tensors(device)

    x = torch.cat((vs, vs), dim=0)
    y = torch.cat((vs, vt), dim=0)

    alpha_min, alpha_max = alpha_min_max
    alpha = alpha_min + (alpha_max - alpha_min) * torch.rand(b, device=device).unsqueeze(-1)

    delta_vc = [
        alpha * (vt - vs)[:, control_vids[i]].reshape(b, -1)
        for i in range(len(core.control_region_keys))
    ]
    delta_vc = [torch.cat((torch.zeros_like(d), d), dim=0) for d in delta_vc]

    alpha = torch.cat((alpha, alpha)).unsqueeze(-1)
    y = alpha * y + (1 - alpha) * x

    lambda_target_encoder, lambda_target_decoder = lambda_target
    y_expanded = torch.cat((
        (1 - lambda_target_encoder) * torch.zeros_like(x) + lambda_target_encoder * x,
        (1 - lambda_target_decoder) * torch.zeros_like(y) + lambda_target_decoder * y,
    ), dim=0)

    outputs = net((x, delta_vc))
    loss = loss_fn(outputs, y_expanded)

    mask = torch.ones(loss.shape, device=device)
    mask[core.encoder_depth + 1, b:] = 0

    optimizer.zero_grad(set_to_none=True)
    total_loss = (mask * loss).sum() / mask.sum()
    total_loss.backward()
    if clip_grad_norm is not None and clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(unwrap_net(net).parameters(), clip_grad_norm)
    optimizer.step()

    return loss.mean(dim=[1, 2, 3])


def _viz_indices(n: int, n_viz: int, seed: int) -> list[int]:
    if n <= 0 or n_viz <= 0:
        return []
    n_viz = min(int(n_viz), n)
    rng = np.random.default_rng(int(seed))
    return sorted(rng.choice(n, size=n_viz, replace=False).tolist())


def evaluate(
    net,
    eval_loader,
    loss_fn,
    alpha_max,
    device,
    save_dir=None,
    abs_step=None,
    seed=42,
    n_viz: int=4,
):
    """Return (eval_ae, eval_alpha_max). Viz deferred until after metric sync."""
    core = unwrap_net(net)
    losses_ae = []
    losses_alpha_max = []
    sampler = getattr(eval_loader, 'sampler', None)
    if isinstance(sampler, DistributedSampler):
        n_local = int(sampler.num_samples)
    else:
        n_local = len(eval_loader.dataset)
    viz_idxs = set()
    if save_dir is not None and abs_step is not None:
        viz_idxs = set(_viz_indices(n_local, n_viz, seed))

    local_pos = 0
    viz_payloads: list[tuple] = []
    net.eval()
    with torch.no_grad():
        for sample in eval_loader:
            vs = sample['verts'].to(device, non_blocking=True)
            vt = torch.roll(vs, 1, 0)
            b = vs.shape[0]
            control_vids = core.control_vid_tensors(device)

            delta_vc_max = [
                alpha_max * (vt - vs)[:, control_vids[i]].reshape(b, -1)
                for i in range(len(core.control_region_keys))
            ]
            outputs_max = net((vs, delta_vc_max))[-1]
            losses_alpha_max.append(
                loss_fn(outputs_max, alpha_max * vt + (1 - alpha_max) * vs).mean().detach()
            )

            delta_vc_ae = [torch.zeros_like(d) for d in delta_vc_max]
            outputs_ae = net((vs, delta_vc_ae))[-1]
            losses_ae.append(loss_fn(outputs_ae, vs).mean().detach())

            if viz_idxs:
                faces_np = None
                for j in range(b):
                    gidx = local_pos + j
                    if gidx not in viz_idxs:
                        continue
                    if faces_np is None:
                        faces_np = _faces_numpy(sample['faces'])
                    stem = _sample_stem(sample, j)
                    viz_payloads.append((
                        to_np(alpha_max * vt[j] + (1 - alpha_max) * vs[j]),
                        to_np(outputs_max[j]),
                        to_np(vs[j]),
                        to_np(outputs_ae[j]),
                        faces_np,
                        stem,
                        gidx,
                    ))
                    viz_idxs.discard(gidx)
            local_pos += b

    if len(losses_ae) == 0:
        mean_ae = torch.zeros((), device=device)
        mean_alpha = torch.zeros((), device=device)
    else:
        mean_ae = torch.stack(losses_ae).mean()
        mean_alpha = torch.stack(losses_alpha_max).mean()

    if dist.is_initialized():
        for t in (mean_ae, mean_alpha):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        ws = dist.get_world_size()
        mean_ae = mean_ae / ws
        mean_alpha = mean_alpha / ws

    eval_ae = float(mean_ae.detach().cpu())
    eval_alpha = float(mean_alpha.detach().cpu())

    if save_dir is not None and abs_step is not None and viz_payloads:
        stems = []
        out_dir = None
        for gt_m, pred_m, gt_ae, pred_ae, faces_np, stem, gidx in viz_payloads:
            out_dir = save_eval_visuals(
                gt_m, pred_m, faces_np, save_dir, abs_step, stem=stem, tag='manip',
            )
            save_eval_visuals(
                gt_ae, pred_ae, faces_np, save_dir, abs_step, stem=stem, tag='ae',
            )
            stems.append(f'{stem}@{gidx}')
        print(
            f'[eval viz] seed={seed} n={len(viz_payloads)} '
            f'[{", ".join(stems)}] -> {out_dir}'
        )
    if dist.is_initialized():
        dist.barrier()

    return eval_ae, eval_alpha


def main():
    args = parse_args()
    config = read_yaml(args.config)
    mode, device, rank, world_size, device_ids, is_main = setup_runtime(args, config)

    model_cfg = dict(config['MODEL'])
    model_cfg['manipulation'] = True
    eval_viz_seed = int(config.get('SOLVER', {}).get('seed', 31415))
    seed_everything(eval_viz_seed + rank)

    dataloaders, global_batch, per_gpu_batch = get_dataloaders(config, mode, rank, world_size)
    net = build_planmm_net(model_cfg, device)

    core = unwrap_net(net)
    if is_main:
        print(f'mode={mode}, devices={device_ids}, global_batch={global_batch}, per_gpu={per_gpu_batch}')
        print(f'Decoder: mesh_readout={core.mesh_readout}')
        print(
            f"template: V={core.template_verts.shape[0]}, F={core.template_faces.shape[0]}, "
            f"C_width={core.C_width}"
        )
        print(f"Control regions ({len(core.control_region_keys)}): {core.control_region_keys}")
        for key in core.control_region_keys:
            n_v = len(core.control_vertices[key])
            print(f"  region {key}: {n_v} control verts")

    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    eval_viz_samples = int(config['CHECKPOINT'].get('eval_viz_samples', 4))
    save_dir = resolve_save_dir(config['CHECKPOINT'], _REPO_ROOT, is_main)
    if is_main and save_dir is not None:
        config['CHECKPOINT']['save_dir'] = save_dir
    num_steps_epoch = len(dataloaders['training'])
    weight_decay = get_params_values(config['SOLVER'], 'weight_decay', 0)
    clip_grad_norm = get_params_values(config['SOLVER'], 'clip_grad_norm', None)
    if clip_grad_norm is not None:
        clip_grad_norm = float(clip_grad_norm)

    encoder_depth = config['MODEL']['encoder_depth']
    decoder_depth = config['MODEL']['decoder_depth']
    lambda_target_encoder = torch.linspace(1, 0, encoder_depth + 1).view(
        encoder_depth + 1, 1, 1, 1,
    ).to(device)
    lambda_target_decoder = torch.linspace(0, 1, decoder_depth + 1).view(
        decoder_depth + 1, 1, 1, 1,
    ).to(device)
    lambda_target = (lambda_target_encoder, lambda_target_decoder)

    alpha_min = float(config['SOLVER']['alpha_min'])
    alpha_max = float(config['SOLVER']['alpha_max'])
    alpha_max_epoch = float(config['SOLVER'].get('alpha_max_epoch', 100))

    checkpoint_file = config['CHECKPOINT'].get('load_from_checkpoint')
    if checkpoint_file:
        ckpt_path = resolve_checkpoint_path(Path(args.config), str(checkpoint_file))
        load_from_checkpoint(
            net,
            str(ckpt_path),
            partial_restore=config['CHECKPOINT'].get('partial_restore', True),
            device=device,
        )
        if is_main:
            print(f'loaded pretrained: {ckpt_path}')

    if is_main and save_dir is not None:
        copy_yaml(config)
        print(f'save_dir: {save_dir}')

    net = wrap_model(net, mode, device, device_ids)
    if is_main:
        print(f"trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

    loss_fn = get_loss(config, reduction='none')
    optimizer = optim.AdamW(get_net_trainable_params(net), lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True)
    scheduler = build_scheduler(config, optimizer, num_steps_epoch)
    writer = SummaryWriter(save_dir) if is_main and save_dir else None

    best_eval_ae = 1e10
    best_eval_alpha = 1e10
    net.train()

    for epoch in range(1, num_epochs + 1):
        train_sampler = dataloaders['training'].sampler
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        alpha_max_curr = alpha_min + (alpha_max - alpha_min) * min(1.0, epoch / alpha_max_epoch)
        alpha_min_max = (alpha_min, alpha_max_curr)

        epoch_bar = tqdm(
            dataloaders['training'],
            desc=f'Epoch {epoch}/{num_epochs}',
            dynamic_ncols=True,
            disable=not is_main,
        )
        for step, sample in enumerate(epoch_bar):
            abs_step = (epoch - 1) * num_steps_epoch + step + 1
            loss = train_step(
                net, sample, loss_fn, alpha_min_max, lambda_target, optimizer, device,
                clip_grad_norm=clip_grad_norm,
            )
            scheduler.step_update(abs_step)
            if is_main:
                epoch_bar.set_postfix(
                    loss=f'{loss.mean().item():.5f}',
                    alpha=f'{alpha_max_curr:.3f}',
                    lr=f'{optimizer.param_groups[0]["lr"]:.2e}',
                )

            if is_main and abs_step % train_metrics_steps == 0:
                write_mean_summaries(
                    writer,
                    {f'train_loss_{i}': loss_.item() for i, loss_ in enumerate(loss)},
                    abs_step,
                    mode='training',
                    optimizer=optimizer,
                )
                print(
                    f"abs_step: {abs_step}, epoch: {epoch}, step: {step + 1}, "
                    f"loss: {loss.tolist()}, "
                    f"lr: {optimizer.param_groups[0]['lr']}, "
                    f"alpha_max_curr: {alpha_max_curr:.3f}"
                )

            if abs_step % eval_steps == 0:
                if is_main:
                    print('--------------------- EVAL ----------------------------')
                eval_ae, eval_alpha = evaluate(
                    net,
                    dataloaders['eval'],
                    loss_fn,
                    alpha_max,
                    device,
                    save_dir=save_dir if is_main else None,
                    abs_step=abs_step if is_main else None,
                    seed=eval_viz_seed,
                    n_viz=eval_viz_samples,
                )

                if is_main and save_dir is not None:
                    state = unwrap_net(net).state_dict()
                    if eval_ae < best_eval_ae:
                        atomic_torch_save(state, f'{save_dir}/best_ae.pth')
                        best_eval_ae = eval_ae
                    if eval_alpha < best_eval_alpha:
                        atomic_torch_save(state, f'{save_dir}/best_alpha_max.pth')
                        best_eval_alpha = eval_alpha
                    atomic_torch_save(state, f'{save_dir}/last.pth')

                if dist.is_initialized():
                    dist.barrier()

                if is_main:
                    write_mean_summaries(
                        writer,
                        {'eval_loss_ae': eval_ae, 'eval_loss_alpha_max': eval_alpha},
                        abs_step,
                        mode='eval',
                        optimizer=None,
                    )
                    print(
                        f"abs_step: {abs_step}, eval_ae: {eval_ae}, "
                        f"eval_alpha_max: {eval_alpha}"
                    )
                net.train()

        if is_main and save_dir is not None:
            atomic_torch_save(unwrap_net(net).state_dict(), f'{save_dir}/last.pth')
        if dist.is_initialized():
            dist.barrier()

    if is_main and save_dir is not None:
        atomic_torch_save(unwrap_net(net).state_dict(), f'{save_dir}/final.pth')
        print(f'saved final.pth -> {save_dir}/final.pth')

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
