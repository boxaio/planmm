"""Evaluate PLANMM autoencoder on HACK eval split."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import read_yaml
from exp_poisson.planmm_generate_samples import build_planmm_net
from exp_poisson.train_planmm_ae import _faces_numpy, _sample_stem, get_dataloaders
from utils.torch_utils import load_from_checkpoint


def _repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def _latest_best_in_dir(base_dir: Path) -> Path:
    """Pick a checkpoint from the newest timestamp subfolder under ``base_dir``."""
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


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PLANMM autoencoder on HACK eval set')
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
        '--output_dir',
        type=str,
        default=None,
        help='Directory for eval outputs (default: <checkpoint_dir>/eval_ae)',
    )
    parser.add_argument('--device', type=str, default='0', help='CUDA device id or "cpu"')
    parser.add_argument(
        '--save_output',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Save reconstruction examples from the first batch',
    )
    return parser.parse_args()


def save_mesh(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces).export(str(path), file_type='obj')


def _neck_vids(net) -> torch.Tensor | None:
    if 'Neck' not in getattr(net, 'region_names', []):
        return None
    idx = net.region_names.index('Neck')
    return net.region_vert_ids[idx]


@torch.no_grad()
def evaluate_autoencoder(args):
    config_path = Path(args.config)
    checkpoint_path = resolve_checkpoint_path(config_path, args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    config = resolve_config(config_path, checkpoint_path)
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / 'eval_ae'
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_loader = get_dataloaders(config, mode='dp', rank=0, world_size=1)[0]['eval']

    net = build_planmm_net(config['MODEL'], device)
    load_from_checkpoint(net, str(checkpoint_path), partial_restore=True, device=device)
    net.eval()
    neck_vids = _neck_vids(net)

    losses_l1 = []
    losses_l2 = []
    distances = []
    neck_l1_all = []
    max_l1_all = []

    for step, sample in enumerate(tqdm(eval_loader, desc='eval', dynamic_ncols=True)):
        vs = sample['verts'].to(device)
        out = net(vs)[-1]

        l1 = torch.nn.functional.l1_loss(out, vs)
        l2 = torch.nn.functional.mse_loss(out, vs)
        dist = torch.sqrt(((out - vs) ** 2).sum(dim=-1)).mean()

        per_v = (out - vs).abs().mean(dim=-1)
        if neck_vids is not None:
            neck_l1 = per_v[:, neck_vids.to(device)].mean()
        else:
            neck_l1 = per_v.mean()
        max_l1 = per_v.max(dim=1).values.mean()

        losses_l1.append(float(l1.cpu()))
        losses_l2.append(float(l2.cpu()))
        distances.append(float(dist.cpu()))
        neck_l1_all.append(float(neck_l1.cpu()))
        max_l1_all.append(float(max_l1.cpu()))

        if args.save_output and step == 0:
            faces = _faces_numpy(sample['faces'])
            for i in range(vs.shape[0]):
                stem = _sample_stem(sample, i)
                save_mesh(
                    vs[i].detach().cpu().numpy(),
                    faces,
                    output_dir / f'{stem}_source.obj',
                )
                save_mesh(
                    out[i].detach().cpu().numpy(),
                    faces,
                    output_dir / f'{stem}_recon.obj',
                )

    mean_l1 = float(np.mean(losses_l1))
    mean_l2 = float(np.mean(losses_l2))
    mean_dist = float(np.mean(distances))
    mean_neck_l1 = float(np.mean(neck_l1_all))
    mean_max_l1 = float(np.mean(max_l1_all))

    print(
        f'L1: {mean_l1:.6f}, L2: {mean_l2:.6f}, '
        f'Euclidean: {mean_dist:.6f}, neck_L1: {mean_neck_l1:.6f}, max_L1: {mean_max_l1:.6f}'
    )

    csv_path = output_dir / 'eval_loss.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in (
            ['PLANMM Autoencoding Evaluation'],
            [str(config_path)],
            [str(checkpoint_path)],
            [config['DATASETS']['eval']['dataset'], 'eval'],
            ['L1', mean_l1],
            ['L2', mean_l2],
            ['Euclidean Distance', mean_dist],
            ['Neck L1', mean_neck_l1],
            ['Max L1 (batch mean)', mean_max_l1],
        ):
            writer.writerow(row)
    print(f'Saved metrics to {csv_path}')
    if args.save_output:
        print(f'Saved first-batch meshes to {output_dir}')


if __name__ == '__main__':
    evaluate_autoencoder(parse_args())
