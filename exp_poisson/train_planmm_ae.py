"""PLANMM AE training — LAMM multilayer L1, PoissonBlock tokenize only."""

import argparse
import os
import sys
from datetime import datetime
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
from exp_poisson.summaries import write_mean_summaries
from networks.planmm import (
    PLANMM,
    build_layer_blend_weights,
    build_layer_loss_weights,
    load_template_from_config,
)
from render.mesh_render import add_text, render_mesh_on, render_vertex_error_map
from utils.helpers import seed_everything, to_np
from utils.torch_utils import get_net_trainable_params, load_from_checkpoint


def atomic_torch_save(obj, path: str | Path) -> None:
    """Write via ``*.tmp`` then ``os.replace`` so a crash cannot truncate the previous file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def resolve_save_dir(checkpoint_cfg: dict, repo_root: Path, is_main: bool):
    """Resolve run output directory; optionally nest under a timestamp subfolder.

    Under DDP, rank0 creates the timestamped dir and broadcasts the path so all
    ranks share the same ``save_dir``.
    """
    base = checkpoint_cfg.get('save_dir')
    if not base:
        return None

    base_path = Path(base)
    if not base_path.is_absolute():
        base_path = repo_root / base_path

    if not checkpoint_cfg.get('use_timestamp_subdir', True):
        if is_main:
            base_path.mkdir(parents=True, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        return str(base_path)

    if is_main:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = base_path / ts
        save_path.mkdir(parents=True, exist_ok=True)
        resolved = str(save_path)
    else:
        resolved = str(base_path)

    if dist.is_initialized():
        obj = [resolved] if is_main else [None]
        dist.broadcast_object_list(obj, src=0)
        resolved = obj[0]
        dist.barrier()
    return resolved


def wrap_model(net, mode, device, device_ids):
    if mode == 'ddp':
        return DDP(
            net,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )
    if len(device_ids) > 1:
        return nn.DataParallel(net, device_ids=device_ids, output_device=device_ids[0])
    return net


def parse_args():
    parser = argparse.ArgumentParser(description='PLANMM (LAMM + Poisson tokenize) AE training')
    parser.add_argument(
        '--config',
        type=str,
        default=str(_REPO_ROOT / 'exp_poisson' / 'planmm_hack.yaml'),
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


def setup_runtime(args, config):
    """Return (mode, device, rank, world_size, device_ids, is_main)."""
    if args.distributed or os.environ.get('LOCAL_RANK') is not None:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        return 'ddp', device, rank, world_size, [local_rank], rank == 0

    device_ids = args.device or config.get('SOLVER', {}).get('devices', '0,1')
    device_ids = [int(d) for d in str(device_ids).split(',') if str(d).strip() != '']
    if not device_ids:
        device_ids = [0]
    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    return 'dp', device, 0, len(device_ids), device_ids, True


def unwrap_net(net):
    if isinstance(net, DDP):
        return net.module
    if isinstance(net, nn.DataParallel):
        return net.module
    return net


def get_dataloaders(config, mode, rank, world_size):
    dataloaders = {}
    seed = config.get('SOLVER', {}).get('seed', 42)
    train_cfg = config['DATASETS']['train']
    eval_cfg = config['DATASETS']['eval']
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
        train_fraction = float(train_cfg.get('subset_ratio', 0.2))
    eval_fraction = eval_cfg.get('data_fraction', train_fraction)
    if eval_cfg.get('use_full_eval', False):
        eval_fraction = 1.0
    elif eval_cfg.get('use_subset', train_cfg.get('use_subset', False)):
        eval_fraction = float(eval_cfg.get('subset_ratio', 0.2))

    for split_name, packed_pt, split_cfg, fraction, shuffle in (
        ('training', TRAIN_HACK_PT, train_cfg, train_fraction, True),
        ('eval', VAL_HACK_PT, eval_cfg, eval_fraction, False),
    ):
        dataset = HACKDataset(packed_pt=packed_pt)
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
    """Save GT / recon OBJ and a GT | Recon | Error render panel."""
    viz_dir = Path(save_dir) / 'eval_viz' / f'step_{abs_step:06d}'
    viz_dir.mkdir(parents=True, exist_ok=True)

    tag = str(stem).replace('/', '_')
    save_obj_mesh(gt_verts, faces, viz_dir / f'gt_{tag}.obj')
    save_obj_mesh(pred_verts, faces, viz_dir / f'recon_{tag}.obj')

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


def _viz_local_indices(n_local: int, n_viz: int, seed: int) -> list[int]:
    """Deterministic local sample indices for eval viz (sorted, unique)."""
    if n_local <= 0 or n_viz <= 0:
        return []
    n_viz = min(int(n_viz), n_local)
    rng = np.random.default_rng(int(seed))
    return sorted(rng.choice(n_local, size=n_viz, replace=False).tolist())


def evaluate(
    net,
    eval_loader,
    loss_fn,
    device,
    save_dir=None,
    abs_step=None,
    seed=42,
    n_viz: int = 4,
):
    """Return mean L1 at the final decoder layer.

    Under DDP, metrics are reduced **before** any rank0-only I/O / pyrender.
    Viz is deferred until after ``all_reduce`` so OpenGL cannot desync NCCL.
    ``n_viz`` meshes are stashed on rank0 and written after the collective.
    """
    losses_all = []
    id_token_stats = []
    sampler = getattr(eval_loader, 'sampler', None)
    if isinstance(sampler, DistributedSampler):
        n_local = int(sampler.num_samples)
    else:
        n_local = len(eval_loader.dataset)
    viz_idxs = set()
    if save_dir is not None and abs_step is not None:
        viz_idxs = set(_viz_local_indices(n_local, n_viz, seed))
    net.eval()
    core = unwrap_net(net)
    local_pos = 0
    viz_payloads: list[tuple] = []  # stash CPU arrays; render after collectives
    with torch.no_grad():
        for sample in eval_loader:
            vs = sample['verts'].to(device, non_blocking=True)
            outputs = core(vs)
            recon = outputs[-1]
            id_token, _ = core.encode(vs, return_layer_outputs=False)
            id_token_stats.append(id_token.abs().mean().detach())
            losses_all.append(loss_fn(recon, vs).mean().detach())

            B = vs.shape[0]
            if viz_idxs:
                for j in range(B):
                    gidx = local_pos + j
                    if gidx in viz_idxs:
                        viz_payloads.append((
                            to_np(vs[j]),
                            to_np(recon[j]),
                            _faces_numpy(sample['faces']),
                            _sample_stem(sample, j),
                            gidx,
                        ))
                        viz_idxs.discard(gidx)
            local_pos += B

    if len(losses_all) == 0:
        mean_loss = torch.zeros((), device=device)
        mean_id = torch.zeros((), device=device)
    else:
        mean_loss = torch.stack(losses_all).mean()
        mean_id = torch.stack(id_token_stats).mean()

    # Collectives first — every rank must reach here without rank0-only stalls.
    if dist.is_initialized():
        for t in (mean_loss, mean_id):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        ws = dist.get_world_size()
        mean_loss = mean_loss / ws
        mean_id = mean_id / ws

    eval_loss = float(mean_loss.detach().cpu())
    id_mean = float(mean_id.detach().cpu())

    # Rank0 I/O / pyrender only after NCCL sync; other ranks wait at barrier.
    if save_dir is not None and abs_step is not None and viz_payloads:
        out_dir = None
        stems = []
        for gt_np, pred_np, faces_np, stem, gidx in viz_payloads:
            out_dir = save_eval_visuals(
                gt_np, pred_np, faces_np, save_dir, abs_step, stem=stem,
            )
            stems.append(f'{stem}@{gidx}')
        print(
            f'[eval viz] seed={seed} n={len(viz_payloads)} '
            f'[{", ".join(stems)}] -> {out_dir}',
        )
    if dist.is_initialized():
        dist.barrier()

    return eval_loss, id_mean


def train_step(
    net, sample, loss_fn, optimizer, device,
    lambda_blend, layer_weights, clip_grad_norm=None,
    poisson_grad_weight: float = 0.0,
    seam_vertex_weight: float = 0.0,
    seam_grad_weight: float = 0.0,
):
    """Multilayer vertex L1 + optional full/seam gradient L1 on final mesh."""
    core = unwrap_net(net)
    vs = sample['verts'].to(device, non_blocking=True)
    id_token, encoder_outputs = core.encode(vs)
    decoder_outputs = core.decode(id_token)
    outputs = torch.cat((encoder_outputs, decoder_outputs), dim=0)
    vs_expanded = vs.unsqueeze(0).expand(outputs.shape[0], -1, -1, -1)
    vt_expanded = (1 - lambda_blend) * torch.zeros_like(vs_expanded) + lambda_blend * vs_expanded
    layer_loss = loss_fn(outputs, vt_expanded)
    loss_v_layers = (layer_weights.view(-1) * layer_loss.mean(dim=[1, 2, 3])).sum()

    pred_final = outputs[-1]
    loss_g = pred_final.new_zeros(())
    loss_seam_v = pred_final.new_zeros(())
    loss_seam_g = pred_final.new_zeros(())
    loss = loss_v_layers
    if poisson_grad_weight and poisson_grad_weight > 0:
        loss_g = core.poisson_grad_loss(pred_final, vs)
        loss = loss + float(poisson_grad_weight) * loss_g
    if seam_vertex_weight and seam_vertex_weight > 0:
        loss_seam_v = core.seam_vertex_loss(pred_final, vs)
        loss = loss + float(seam_vertex_weight) * loss_seam_v
    if seam_grad_weight and seam_grad_weight > 0:
        loss_seam_g = core.seam_grad_loss(pred_final, vs)
        loss = loss + float(seam_grad_weight) * loss_seam_g

    if not torch.isfinite(loss):
        print('[warn] : non-finite loss, skip step')
        optimizer.zero_grad(set_to_none=True)
        nan = torch.tensor(float('nan'), device=device)
        return (
            nan, nan, id_token.abs().mean().detach(), None, nan, nan, nan,
        )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if clip_grad_norm is not None and clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(unwrap_net(net).parameters(), clip_grad_norm)
    optimizer.step()

    final_layer_loss = layer_loss[-1].mean().detach()
    per_layer = layer_loss.mean(dim=[1, 2, 3]).detach()
    return (
        final_layer_loss,
        loss.detach(),
        id_token.abs().mean().detach(),
        per_layer,
        loss_g.detach(),
        loss_seam_v.detach(),
        loss_seam_g.detach(),
    )


def main():
    args = parse_args()
    config = read_yaml(args.config)
    mode, device, rank, world_size, device_ids, is_main = setup_runtime(args, config)
    eval_viz_seed = int(config.get('SOLVER', {}).get('seed', 78986))
    seed_everything(eval_viz_seed + rank)

    if config['MODEL'].get('manipulation', False):
        raise ValueError(
            'planmm_hack.yaml should set MODEL.manipulation=false for AE training.'
        )

    dataloaders, global_batch, per_gpu_batch = get_dataloaders(config, mode, rank, world_size)

    net = PLANMM(config['MODEL']).to(device)
    template_verts, template_faces = load_template_from_config(config['MODEL'], device=device)
    unwrap_net(net).register_template_mesh(template_verts, template_faces)

    if is_main:
        print(
            f"Multi-GPU mode={mode}, world_size={world_size}, "
            f"global_batch={global_batch}, per_gpu_batch={per_gpu_batch}"
        )
        core = unwrap_net(net)
        print(
            f"encoder_depth={core.encoder_depth}, decoder_depth={core.decoder_depth}, "
            f"bottleneck_dim={core.bottleneck_dim}, dim={core.dim}, "
            f"C_width={core.config['C_width']}, poisson_token_blocks={core.config['poisson_token_blocks']}, "
            f"mesh_readout={core.mesh_readout}, n_loss_layers={core.n_loss_layers}"
        )

    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    eval_viz_samples = int(config['CHECKPOINT'].get('eval_viz_samples', 4))
    save_dir = resolve_save_dir(config['CHECKPOINT'], _REPO_ROOT, is_main)
    if is_main and save_dir is not None:
        config['CHECKPOINT']['save_dir'] = save_dir
        print(f'save_dir: {save_dir}')
    num_steps_epoch = len(dataloaders['training'])
    weight_decay = get_params_values(config['SOLVER'], 'weight_decay', 0)
    clip_grad_norm = get_params_values(config['SOLVER'], 'clip_grad_norm', None)
    if clip_grad_norm is not None:
        clip_grad_norm = float(clip_grad_norm)
    poisson_grad_weight = float(get_params_values(config['SOLVER'], 'poisson_grad_weight', 0.0) or 0.0)
    seam_vertex_weight = float(get_params_values(config['SOLVER'], 'seam_vertex_weight', 0.0) or 0.0)
    seam_grad_weight = float(get_params_values(config['SOLVER'], 'seam_grad_weight', 0.0) or 0.0)

    layer_weights_cfg = config['SOLVER'].get('weights')
    if layer_weights_cfg is not None and len(layer_weights_cfg) == 0:
        layer_weights_cfg = None

    lambda_blend = build_layer_blend_weights(
        config['MODEL']['encoder_depth'],
        config['MODEL']['decoder_depth'],
        device=device,
    )
    layer_weights = build_layer_loss_weights(
        config['MODEL']['encoder_depth'],
        config['MODEL']['decoder_depth'],
        weights=layer_weights_cfg,
        device=device,
    )

    checkpoint_file = config['CHECKPOINT'].get('load_from_checkpoint')
    if checkpoint_file:
        load_from_checkpoint(
            net,
            checkpoint_file,
            partial_restore=config['CHECKPOINT'].get('partial_restore', False),
        )
    if is_main:
        print(
            f'clip_grad_norm={clip_grad_norm}, poisson_grad_weight={poisson_grad_weight}, '
            f'seam_vertex_weight={seam_vertex_weight}, seam_grad_weight={seam_grad_weight}, '
            f'n_loss_layers={lambda_blend.shape[0]}'
        )

    net = wrap_model(net, mode, device, device_ids)

    if is_main:
        print(f"trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")

    if is_main:
        copy_yaml(config)

    loss_fn = get_loss(config, reduction='none')
    optimizer = optim.AdamW(get_net_trainable_params(net), lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad(set_to_none=True)
    scheduler = build_scheduler(config, optimizer, num_steps_epoch)
    writer = SummaryWriter(save_dir) if is_main else None

    best_eval_loss = 1e10
    net.train()
    for epoch in range(1, num_epochs + 1):
        train_sampler = dataloaders['training'].sampler
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        epoch_bar = tqdm(
            dataloaders['training'],
            desc=f'Epoch {epoch}/{num_epochs}',
            dynamic_ncols=True,
            disable=not is_main,
        )
        for step, sample in enumerate(epoch_bar):
            abs_step = (epoch - 1) * num_steps_epoch + step + 1

            loss_v, loss, id_mean, per_layer, loss_g, loss_seam_v, loss_seam_g = train_step(
                net, sample, loss_fn, optimizer, device,
                lambda_blend=lambda_blend,
                layer_weights=layer_weights,
                clip_grad_norm=clip_grad_norm,
                poisson_grad_weight=poisson_grad_weight,
                seam_vertex_weight=seam_vertex_weight,
                seam_grad_weight=seam_grad_weight,
            )
            if not torch.isfinite(loss):
                if is_main:
                    print(f'[warn] abs_step {abs_step}: non-finite loss, skip step')
                continue
            scheduler.step_update(abs_step)
            if is_main:
                postfix = dict(
                    loss=f'{loss.item():.5f}',
                    loss_v=f'{loss_v.item():.5f}',
                    id=f'{id_mean.item():.3e}',
                    lr=f'{optimizer.param_groups[0]["lr"]:.2e}',
                )
                if poisson_grad_weight > 0:
                    postfix['loss_g'] = f'{loss_g.item():.5f}'
                if seam_vertex_weight > 0:
                    postfix['sv'] = f'{loss_seam_v.item():.5f}'
                if seam_grad_weight > 0:
                    postfix['sg'] = f'{loss_seam_g.item():.5f}'
                epoch_bar.set_postfix(**postfix)

            if is_main and abs_step % train_metrics_steps == 0:
                metrics = {
                    'train_loss': loss.item(),
                    'train_loss_v': loss_v.item(),
                    'id_token_abs_mean': id_mean.item(),
                }
                if poisson_grad_weight > 0:
                    metrics['train_loss_g'] = loss_g.item()
                if seam_vertex_weight > 0:
                    metrics['train_loss_seam_v'] = loss_seam_v.item()
                if seam_grad_weight > 0:
                    metrics['train_loss_seam_g'] = loss_seam_g.item()
                if per_layer is not None:
                    for i in range(per_layer.shape[0]):
                        metrics[f'train_loss_{i}'] = per_layer[i].item()
                write_mean_summaries(
                    writer,
                    metrics,
                    abs_step,
                    mode='training',
                    optimizer=optimizer,
                )
                layer_str = (
                    ', '.join(f'{v:.5f}' for v in per_layer.tolist())
                    if per_layer is not None else 'n/a'
                )
                extra = ''
                if poisson_grad_weight > 0:
                    extra += f', loss_g: {loss_g.item():.6f}'
                if seam_vertex_weight > 0:
                    extra += f', seam_v: {loss_seam_v.item():.6f}'
                if seam_grad_weight > 0:
                    extra += f', seam_g: {loss_seam_g.item():.6f}'
                print(
                    f"abs_step: {abs_step}, epoch: {epoch}, step: {step + 1}, "
                    f"loss: {loss.item():.6f}, loss_v: {loss_v.item():.6f}{extra}, "
                    f"layers: [{layer_str}], "
                    f"id_token: {id_mean.item():.6e}, "
                    f"lr: {optimizer.param_groups[0]['lr']}"
                )

            if abs_step % eval_steps == 0:
                if is_main:
                    print('--------------------- EVAL ----------------------------')
                eval_loss, id_token_mean = evaluate(
                    net,
                    dataloaders['eval'],
                    loss_fn,
                    device,
                    save_dir=save_dir if is_main else None,
                    abs_step=abs_step if is_main else None,
                    seed=eval_viz_seed,
                    n_viz=eval_viz_samples,
                )

                if is_main and save_dir is not None:
                    state = unwrap_net(net).state_dict()
                    # Always keep a recoverable snapshot (atomic).
                    atomic_torch_save(state, f'{save_dir}/last.pth')
                    if eval_loss < best_eval_loss:
                        atomic_torch_save(state, f'{save_dir}/best.pth')
                        best_eval_loss = eval_loss

                # Rank0 may still be writing ckpt while others would resume training.
                if dist.is_initialized():
                    dist.barrier()

                if is_main:
                    write_mean_summaries(
                        writer,
                        {
                            'eval_loss': eval_loss,
                            'id_token_abs_mean': id_token_mean,
                        },
                        abs_step,
                        mode='eval_micro',
                        optimizer=None,
                    )
                    print(
                        f"abs_step: {abs_step}, eval_loss: {eval_loss:.6f}, "
                        f"id_token_abs_mean: {id_token_mean:.6e}"
                    )
                net.train()

        # End-of-epoch snapshot (survives a late-epoch crash).
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
