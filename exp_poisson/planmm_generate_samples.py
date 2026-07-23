"""Generate random face meshes from a trained PLANMM (LAMM + Poisson tokenize) AE.

Matches ``LAMM_generate_samples.py``: fit / load a Gaussian on ``id_token``
``[B, 1, bottleneck_dim]``, sample, then ``decode(z)[-1]``.

Multi-GPU: pass ``--device 0,1`` to wrap encode/decode with DataParallel
(speeds up ``--fit_gaussian`` / ``--sample_mode train`` over large sets).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import trimesh
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import read_yaml
from dataset.hack_dataset import HACKDataset, TRAIN_HACK_PT
from networks.planmm import PLANMM, load_template_from_config
from render.mesh_render import add_text, render_mesh_on
from utils.torch_utils import load_from_checkpoint


def _repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def _latest_best_in_dir(base_dir: Path) -> Path:
    """Pick a checkpoint from the newest timestamp subfolder under ``base_dir``.

    Prefer non-empty ``best.pth``, then ``last.pth`` / ``final.pth``. Fall back
    to the same names directly under ``base_dir`` only if no timestamp run exists.
    """
    base_dir = _repo_path(base_dir)
    ckpt_names = ('best.pth', 'last.pth', 'final.pth')

    candidates: list[tuple[str, Path]] = []
    if base_dir.is_dir():
        for sub in base_dir.iterdir():
            if not sub.is_dir():
                continue
            for name in ckpt_names:
                ckpt = sub / name
                if ckpt.is_file() and ckpt.stat().st_size > 0:
                    candidates.append((sub.name, ckpt))
                    break

    if candidates:
        latest = max(candidates, key=lambda item: item[0])[1]
        print(f'Using latest checkpoint: {latest}')
        return latest

    for name in ckpt_names:
        direct = base_dir / name
        if direct.is_file() and direct.stat().st_size > 0:
            print(f'Using checkpoint: {direct}')
            return direct

    raise FileNotFoundError(
        f'No non-empty best.pth/last.pth/final.pth found under {base_dir}'
    )


def resolve_checkpoint_path(config_path: Path, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg is None:
        config = read_yaml(str(config_path))
        return _latest_best_in_dir(config['CHECKPOINT']['save_dir'])

    path = _repo_path(checkpoint_arg)
    if path.is_file():
        return path
    if path.is_dir():
        return _latest_best_in_dir(path)
    if path.name == 'best.pth':
        return _latest_best_in_dir(path.parent)
    raise FileNotFoundError(f'Checkpoint not found: {path}')


def resolve_config(config_path: Path, checkpoint_path: Path) -> dict:
    run_cfg = checkpoint_path.parent / 'config_file.yaml'
    if run_cfg.is_file():
        print(f'Using run config: {run_cfg}')
        return read_yaml(str(run_cfg))
    print(f'Using config: {config_path}')
    return read_yaml(str(config_path))


def resolve_gaussian_path(
    config: dict,
    checkpoint_dir: Path,
    gaussian_arg: str | None,
) -> Path:
    if gaussian_arg:
        path = _repo_path(gaussian_arg)
        if path.is_file():
            return path
        raise FileNotFoundError(f'gaussian_id not found: {path}')

    candidates = [
        checkpoint_dir / 'gaussian_id.pickle',
        _repo_path(config['CHECKPOINT']['save_dir']) / 'gaussian_id.pickle',
    ]
    for path in candidates:
        if path.is_file():
            print(f'Using gaussian_id: {path}')
            return path
    return candidates[0]


def parse_device_arg(device_arg: str) -> tuple[torch.device, list[int]]:
    """Parse ``--device``: ``cpu`` | ``0`` | ``0,1`` → (primary_device, cuda_ids)."""
    raw = str(device_arg).strip().lower()
    if raw == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu'), []
    ids = [int(x) for x in raw.split(',') if str(x).strip() != '']
    if not ids:
        ids = [0]
    return torch.device(f'cuda:{ids[0]}'), ids


def parse_args():
    parser = argparse.ArgumentParser(description='Generate meshes from PLANMM id_token')
    parser.add_argument(
        '--config',
        '--config_file',
        type=str,
        dest='config',
        default=str(_REPO_ROOT / 'exp_poisson' / 'planmm_hack.yaml'),
        help='Training configuration (.yaml)',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help=(
            'Path to checkpoint (.pth), run directory, or save_dir base. '
            'Default: latest timestamp run under CHECKPOINT.save_dir'
        ),
    )
    parser.add_argument(
        '--gaussian_id',
        type=str,
        default=None,
        help='Path to gaussian_id.pickle (default: beside checkpoint)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: <checkpoint_dir>/generated_global)',
    )
    parser.add_argument('--num_samples', type=int, default=30, help='Number of meshes to generate')
    parser.add_argument(
        '--k_std',
        type=float,
        default=1.0,
        help='Std multiplier for MVN(mu, k_std * Sigma) (LAMM default: 1.0)',
    )
    parser.add_argument(
        '--sample_mode',
        type=str,
        choices=('global', 'train'),
        default='global',
        help=(
            'global=sample id_token from fitted Gaussian; '
            'train=encode real training meshes then decode (recon sanity)'
        ),
    )
    parser.add_argument(
        '--fit_gaussian',
        action='store_true',
        help='Fit gaussian_id.pickle from training set before generation',
    )
    parser.add_argument(
        '--fit_epochs',
        type=int,
        default=1,
        help='Number of training-loader passes when fitting gaussian',
    )
    parser.add_argument(
        '--fit_batch_size',
        type=int,
        default=32,
        help=(
            'Global batch size when fitting gaussian_id '
            '(auto × n_gpu when --device 0,1 and default 32)'
        ),
    )
    parser.add_argument(
        '--fit_fraction',
        type=float,
        default=1.0,
        help='Fraction of training set to use when fitting gaussian (1.0 = all)',
    )
    parser.add_argument(
        '--fit_max_samples',
        type=int,
        default=0,
        help='Cap number of meshes for gaussian fit (0 = no cap)',
    )
    parser.add_argument(
        '--fit_num_workers',
        type=int,
        default=8,
        help='DataLoader workers when fitting gaussian_id',
    )
    parser.add_argument(
        '--decode_batch_size',
        type=int,
        default=0,
        help='Decode chunk size (0 = all samples in one batch)',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed for latent sampling')
    parser.add_argument(
        '--device',
        type=str,
        default='0,1',
        help='CUDA device id(s), e.g. "0" or "0,1" for DataParallel, or "cpu"',
    )
    parser.add_argument(
        '--no_render',
        action='store_true',
        help='Skip saving mesh preview PNGs beside each OBJ',
    )
    parser.add_argument(
        '--render_res',
        type=int,
        default=512,
        help='Preview image resolution (square)',
    )
    return parser.parse_args()


def save_mesh(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces).export(str(path), file_type='obj')


def save_mesh_render(
    vertices: np.ndarray,
    faces: np.ndarray,
    obj_path: Path,
    render_res: int = 512,
) -> Path:
    png_path = obj_path.with_suffix('.png')
    render = add_text(
        render_mesh_on(vertices, faces, res=render_res),
        caption=obj_path.stem,
    )
    save_image(render, str(png_path))
    return png_path


def build_planmm_net(model_cfg: dict, device: torch.device) -> PLANMM:
    net = PLANMM(model_cfg).to(device)
    tv, tf = load_template_from_config(model_cfg, device=device)
    net.register_template_mesh(tv, tf)
    return net


def _resolve_subset_fraction(split_cfg: dict, default_fraction: float) -> float:
    if 'subset_ratio' in split_cfg:
        return float(split_cfg['subset_ratio'])
    return float(default_fraction)


class _EncodeFlat(nn.Module):
    """Encode → flattened id_token [B, bd] (skip multilayer mesh merge)."""

    def __init__(self, core: PLANMM):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        id_token, _ = self.core.encode(x, return_layer_outputs=False)
        return id_token.reshape(id_token.shape[0], -1)


class _DecodeFinal(nn.Module):
    """Decode → final mesh [B, V, 3] (skip intermediate layer merges)."""

    def __init__(self, core: PLANMM):
        super().__init__()
        self.core = core

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z.unsqueeze(1)
        return self.core.decode(z, return_layer_outputs=False)


def wrap_for_dataparallel(module: nn.Module, device_ids: list[int]) -> nn.Module:
    if len(device_ids) <= 1:
        return module
    print(f'[gen] DataParallel on devices {device_ids}')
    return nn.DataParallel(module, device_ids=device_ids, output_device=device_ids[0])


def _run_maybe_dp(fn: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Call DP module; fall back to .module when batch cannot fill all GPUs."""
    if isinstance(fn, nn.DataParallel) and batch.shape[0] < len(fn.device_ids):
        return fn.module(batch)
    return fn(batch)


class _RunningGaussianStats:
    """Online mean / covariance without storing all latent codes."""

    def __init__(self, dim: int):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros((dim, dim), dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        if batch.size == 0:
            return
        batch = np.asarray(batch, dtype=np.float64)
        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        centered = batch - batch_mean

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = centered.T @ centered
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / total
        self.m2 = (
            self.m2
            + centered.T @ centered
            + np.outer(delta, delta) * self.count * batch_count / total
        )
        self.count = total

    def finalize(self) -> dict:
        if self.count < 2:
            raise RuntimeError(f'Need at least 2 samples to fit gaussian, got {self.count}')
        sigma = self.m2 / (self.count - 1)
        return {'mean': self.mean.astype(np.float32), 'sigma': sigma.astype(np.float32)}


def flatten_id_token(id_token: torch.Tensor) -> torch.Tensor:
    """[B, 1, bd] or [B, bd] -> [B, bd]."""
    return id_token.reshape(id_token.shape[0], -1)


def resolve_fit_indices(
    config: dict,
    fit_fraction: float = 1.0,
    fit_max_samples: int = 0,
) -> list[int]:
    """Resolve training-set indices used for gaussian fitting."""
    train_cfg = config['DATASETS']['train']
    seed = int(config.get('SOLVER', {}).get('seed', 42))
    dataset = HACKDataset(packed_pt=TRAIN_HACK_PT)
    n = len(dataset)
    train_fraction = float(train_cfg.get('data_fraction', 1.0))
    if train_cfg.get('use_subset', False):
        train_fraction = _resolve_subset_fraction(train_cfg, 0.2)
    fraction = min(1.0, max(fit_fraction, 0.0)) * train_fraction
    if fraction < 1.0:
        n_subset = max(1, int(n * fraction))
        indices = np.random.default_rng(seed).choice(n, size=n_subset, replace=False).tolist()
        print(f'[fit_gaussian] training subset: {n_subset}/{n} ({fraction * 100:.1f}%)')
    else:
        indices = list(range(n))
        print(f'[fit_gaussian] training full set: {n}')

    if fit_max_samples > 0 and len(indices) > fit_max_samples:
        indices = np.random.default_rng(seed + 17).choice(
            indices, size=fit_max_samples, replace=False,
        ).tolist()
        print(f'[fit_gaussian] capped to {fit_max_samples} samples')
    return indices


def build_fit_dataloader(
    config: dict,
    batch_size: int,
    num_workers: int,
    fit_fraction: float = 1.0,
    fit_max_samples: int = 0,
    indices: list[int] | None = None,
) -> DataLoader:
    dataset = HACKDataset(packed_pt=TRAIN_HACK_PT)
    if indices is None:
        indices = resolve_fit_indices(config, fit_fraction, fit_max_samples)
    if len(indices) < len(dataset) or indices != list(range(len(dataset))):
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    print(f'[fit_gaussian] loader: n={len(dataset)}, batch_size={batch_size}, workers={num_workers}')
    return loader


def _encode_shard_worker(payload: dict) -> np.ndarray:
    """Process worker: encode a shard of training meshes on one GPU."""
    device_id = int(payload['device_id'])
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')

    net = build_planmm_net(payload['model_cfg'], device)
    load_from_checkpoint(
        net, payload['checkpoint'], partial_restore=True, device=device,
    )
    net.eval()

    dataset = HACKDataset(packed_pt=TRAIN_HACK_PT)
    subset = Subset(dataset, payload['indices'])
    # Pool workers are daemons → cannot spawn DataLoader worker children.
    loader = DataLoader(
        subset,
        batch_size=max(1, int(payload['batch_size'])),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for sample in tqdm(
            loader,
            desc=f'encode gpu{device_id}',
            position=device_id,
            leave=True,
        ):
            vs = sample['verts'].to(device, non_blocking=True)
            id_token, _ = net.encode(vs, return_layer_outputs=False)
            chunks.append(id_token.reshape(id_token.shape[0], -1).float().cpu().numpy())
    if not chunks:
        return np.zeros((0, int(payload['bottleneck_dim'])), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


@torch.no_grad()
def fit_latent_distribution(
    encode_fn: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 1,
    bottleneck_dim: int | None = None,
) -> dict:
    """Fit multivariate Gaussian on flattened id_token (single-process path)."""
    encode_fn.eval()
    stats: _RunningGaussianStats | None = None
    use_pin = torch.cuda.is_available() and device.type == 'cuda'

    for epoch in range(num_epochs):
        print(f'Fitting latent distribution: epoch {epoch + 1}/{num_epochs}')
        for sample in tqdm(train_loader, desc='encode latents'):
            vs = sample['verts'].to(device, non_blocking=use_pin)
            id_flat = _run_maybe_dp(encode_fn, vs)
            batch = id_flat.detach().float().cpu().numpy()
            if stats is None:
                stats = _RunningGaussianStats(batch.shape[-1])
            stats.update(batch)

    if stats is None:
        raise RuntimeError('Training loader is empty; cannot fit latent distribution')
    print(f'[fit_gaussian] fitted id_token on {stats.count} samples')
    out = stats.finalize()
    out['bottleneck_dim'] = int(
        bottleneck_dim if bottleneck_dim is not None else out['mean'].shape[0]
    )
    return out


def fit_latent_distribution_multiprocess(
    model_cfg: dict,
    checkpoint: str,
    indices: list[int],
    device_ids: list[int],
    batch_size_per_gpu: int,
    num_workers: int,
    bottleneck_dim: int,
) -> dict:
    """Shard indices across GPUs; each process encodes independently."""
    import torch.multiprocessing as mp

    shards = [indices[i::len(device_ids)] for i in range(len(device_ids))]
    payloads = [
        {
            'device_id': int(dev),
            'indices': shard,
            'model_cfg': model_cfg,
            'checkpoint': checkpoint,
            'batch_size': int(batch_size_per_gpu),
            'num_workers': max(0, int(num_workers) // len(device_ids)),
            'bottleneck_dim': int(bottleneck_dim),
        }
        for dev, shard in zip(device_ids, shards)
    ]
    print(
        f'[fit_gaussian] multiprocess encode on {device_ids}, '
        f'shard sizes={[len(s) for s in shards]}, batch/gpu={batch_size_per_gpu}'
    )
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=len(device_ids)) as pool:
        parts = pool.map(_encode_shard_worker, payloads)

    codes = np.concatenate([p for p in parts if p.size > 0], axis=0)
    print(f'[fit_gaussian] gathered {codes.shape[0]} id_tokens')
    stats = _RunningGaussianStats(codes.shape[-1])
    # Update in chunks to limit peak RAM on the covariance path.
    step = 4096
    for i in range(0, codes.shape[0], step):
        stats.update(codes[i:i + step])
    out = stats.finalize()
    out['bottleneck_dim'] = int(bottleneck_dim)
    return out


def load_or_fit_gaussian(
    encode_fn: nn.Module,
    net: PLANMM,
    config: dict,
    device: torch.device,
    device_ids: list[int],
    checkpoint_path: Path,
    gaussian_path: Path,
    fit_gaussian: bool,
    fit_epochs: int,
    fit_batch_size: int,
    fit_fraction: float,
    fit_max_samples: int,
    fit_num_workers: int,
) -> dict:
    if gaussian_path.is_file() and not fit_gaussian:
        print(f'Loading gaussian_id from {gaussian_path}')
        with open(gaussian_path, 'rb') as handle:
            return pickle.load(handle, encoding='latin1')

    if gaussian_path.is_file() and fit_gaussian:
        print('--fit_gaussian set: refitting gaussian_id')
    else:
        print(f'gaussian_id not found at {gaussian_path}, fitting from training set')

    indices = resolve_fit_indices(config, fit_fraction, fit_max_samples)
    n_gpu = max(1, len(device_ids))
    # fit_batch_size is treated as per-GPU when multi-process.
    batch_per_gpu = max(1, int(fit_batch_size) // n_gpu) if n_gpu > 1 else int(fit_batch_size)

    if len(device_ids) > 1 and fit_epochs == 1:
        stats = fit_latent_distribution_multiprocess(
            model_cfg=config['MODEL'],
            checkpoint=str(checkpoint_path),
            indices=indices,
            device_ids=device_ids,
            batch_size_per_gpu=batch_per_gpu,
            num_workers=fit_num_workers,
            bottleneck_dim=int(net.bottleneck_dim),
        )
    else:
        if fit_epochs != 1 and len(device_ids) > 1:
            print('[fit_gaussian] fit_epochs>1: falling back to single-process / DataParallel')
        train_loader = build_fit_dataloader(
            config,
            batch_size=int(fit_batch_size),
            num_workers=fit_num_workers,
            indices=indices,
        )
        stats = fit_latent_distribution(
            encode_fn,
            train_loader,
            device,
            num_epochs=max(1, fit_epochs),
            bottleneck_dim=int(net.bottleneck_dim),
        )

    gaussian_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gaussian_path, 'wb') as handle:
        pickle.dump(stats, handle)
    print(f'Saved gaussian_id to {gaussian_path}')
    return stats


@torch.no_grad()
def sample_train_id_tokens(
    encode_fn: nn.Module,
    config: dict,
    device: torch.device,
    num_samples: int,
    rng: np.random.Generator,
    fit_batch_size: int = 32,
    fit_num_workers: int = 4,
) -> np.ndarray:
    """Encode random training meshes → id_token [N, bd]."""
    loader = build_fit_dataloader(
        config, batch_size=fit_batch_size, num_workers=fit_num_workers,
    )
    all_id: list[np.ndarray] = []
    for sample in loader:
        vs = sample['verts'].to(device, non_blocking=True)
        id_flat = _run_maybe_dp(encode_fn, vs)
        all_id.append(id_flat.detach().float().cpu().numpy())
    ids = np.concatenate(all_id, axis=0)
    if ids.shape[0] < num_samples:
        raise RuntimeError(f'Need at least {num_samples} training samples, got {ids.shape[0]}')
    idx = rng.choice(ids.shape[0], size=num_samples, replace=False)
    return ids[idx].astype(np.float32)


@torch.no_grad()
def decode_meshes(
    decode_fn: nn.Module,
    z_np: np.ndarray,
    device: torch.device,
    batch_size: int = 0,
) -> torch.Tensor:
    """Decode flattened / [B,1,bd] latents → [B, V, 3] on CPU."""
    z = torch.from_numpy(z_np.astype(np.float32))
    if z.ndim == 2:
        z = z.unsqueeze(1)
    n = z.shape[0]
    bs = n if batch_size is None or batch_size <= 0 else min(batch_size, n)
    outs: list[torch.Tensor] = []
    for start in range(0, n, bs):
        chunk = z[start:start + bs].to(device=device, dtype=torch.float32)
        outs.append(_run_maybe_dp(decode_fn, chunk).detach().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def generate_global(args):
    config_path = Path(args.config)
    checkpoint_path = resolve_checkpoint_path(config_path, args.checkpoint)
    config = resolve_config(config_path, checkpoint_path)

    device, device_ids = parse_device_arg(args.device)
    n_gpu = max(1, len(device_ids))

    checkpoint_dir = checkpoint_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / 'generated_global'
    gaussian_path = resolve_gaussian_path(config, checkpoint_dir, args.gaussian_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Checkpoint: {checkpoint_path}')
    print(f'Output: {output_dir}')
    print(f'[gen] device={device}, device_ids={device_ids or ["cpu"]}')

    net = build_planmm_net(config['MODEL'], device)
    load_from_checkpoint(net, str(checkpoint_path), partial_restore=True, device=device)
    net.eval()
    print(
        f'[gen] dim={net.dim}, bottleneck_dim={net.bottleneck_dim}, '
        f'C_width={net.config["C_width"]}, poisson_token_blocks={net.config["poisson_token_blocks"]}, '
        f'mesh_readout={net.mesh_readout}, Npatches={net.Npatches}'
    )

    encode_fn = _EncodeFlat(net)
    decode_fn = wrap_for_dataparallel(_DecodeFinal(net), device_ids)
    if len(device_ids) > 1:
        print(f'[gen] multi-GPU fit via process sharding on {device_ids}')

    fit_batch_size = int(args.fit_batch_size)
    if n_gpu > 1 and fit_batch_size == 32:
        fit_batch_size = 32 * n_gpu
        print(f'[gen] auto fit_batch_size={fit_batch_size} ({32}/gpu × {n_gpu})')

    rng = np.random.default_rng(args.seed)

    if args.sample_mode == 'train':
        z_np = sample_train_id_tokens(
            encode_fn, config, device, args.num_samples, rng,
            fit_batch_size=fit_batch_size,
            fit_num_workers=args.fit_num_workers,
        )
        print(f'[gen] sample_mode=train, picked {args.num_samples} encoded id_tokens')
    else:
        gaussian = load_or_fit_gaussian(
            encode_fn,
            net,
            config,
            device,
            device_ids,
            checkpoint_path,
            gaussian_path,
            fit_gaussian=args.fit_gaussian,
            fit_epochs=args.fit_epochs,
            fit_batch_size=fit_batch_size,
            fit_fraction=args.fit_fraction,
            fit_max_samples=args.fit_max_samples,
            fit_num_workers=args.fit_num_workers,
        )
        mu = np.asarray(gaussian['mean'], dtype=np.float64)
        sigma = np.asarray(gaussian['sigma'], dtype=np.float64)
        z_np = rng.multivariate_normal(mu, args.k_std * sigma, size=args.num_samples).astype(np.float32)
        print(
            f'[gen] sample_mode=global, k_std={args.k_std}, '
            f'latent_dim={z_np.shape[1]}'
        )

    decode_bs = int(args.decode_batch_size)
    if decode_bs <= 0:
        decode_bs = max(args.num_samples, n_gpu)
    meshes = decode_meshes(decode_fn, z_np, device, batch_size=decode_bs)

    if args.num_samples >= 2:
        pair = (meshes[0] - meshes[1]).abs()
        print(
            f'[gen sanity] mesh0 vs mesh1: max={pair.max().item():.6f}, '
            f'mean={pair.mean().item():.6f}'
        )
        template = net.template_verts.detach().cpu()
        disp = (meshes - template.unsqueeze(0)).abs().mean(dim=(1, 2))
        print(
            f'[gen sanity] ||mesh-template|| mean: '
            f'min={disp.min().item():.6f}, max={disp.max().item():.6f}, '
            f'avg={disp.mean().item():.6f}'
        )

    faces = net.template_faces.detach().cpu().numpy()
    save_render = not args.no_render
    for i, verts in enumerate(meshes):
        obj_path = output_dir / f'{i:04d}.obj'
        save_mesh(verts.numpy(), faces, obj_path)
        if save_render:
            save_mesh_render(verts.numpy(), faces, obj_path, render_res=args.render_res)

    suffix = ' (+ PNG previews)' if save_render else ''
    print(f'Generated {args.num_samples} meshes{suffix} -> {output_dir}')
    return meshes


if __name__ == '__main__':
    generate_global(parse_args())
