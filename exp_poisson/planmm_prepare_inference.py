"""
Prepare PLANMM inference artifacts for local geometry control.

Mirrors ``LAMM_prepare_inference.py``: region boundaries, Gaussian prior over
identity latents, and per-region control-vertex displacement stats.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import lstsq
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config_utils import read_yaml
from dataset.hack_seg_fids import REGION_NAMES
from exp_poisson.planmm_generate_samples import build_planmm_net
from exp_poisson.train_planmm_manipulation import get_dataloaders
from utils.mesh import read_obj
from utils.torch_utils import load_from_checkpoint

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Prefer manip-specific best, then AE-style / fallbacks.
_CKPT_NAMES = (
    'best_alpha_max.pth',
    'best_ae.pth',
    'best.pth',
    'last.pth',
    'final.pth',
)


def _repo_path(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def _latest_checkpoint_in_dir(base_dir: Path) -> Path:
    """Pick checkpoint from the newest timestamp subfolder under ``base_dir``."""
    base_dir = _repo_path(base_dir)

    for name in _CKPT_NAMES:
        direct = base_dir / name
        if direct.is_file() and direct.stat().st_size > 0:
            print(f'Using checkpoint: {direct}')
            return direct

    candidates: list[tuple[str, Path]] = []
    if base_dir.is_dir():
        for sub in base_dir.iterdir():
            if not sub.is_dir():
                continue
            for name in _CKPT_NAMES:
                ckpt = sub / name
                if ckpt.is_file() and ckpt.stat().st_size > 0:
                    candidates.append((sub.name, ckpt))
                    break

    if not candidates:
        raise FileNotFoundError(
            f'No checkpoint found under {base_dir} '
            f'(looked for {", ".join(_CKPT_NAMES)})'
        )

    latest = max(candidates, key=lambda item: item[0])[1]
    print(f'Using latest checkpoint: {latest}')
    return latest


def resolve_checkpoint_path(config_file, checkpoint_arg) -> Path:
    config_file = Path(config_file)
    if checkpoint_arg is None:
        config = read_yaml(str(config_file))
        return _latest_checkpoint_in_dir(config['CHECKPOINT']['save_dir'])

    path = _repo_path(checkpoint_arg)
    if path.is_file():
        return path
    if path.is_dir():
        return _latest_checkpoint_in_dir(path)
    if path.suffix == '.pth':
        return _latest_checkpoint_in_dir(path.parent)
    raise FileNotFoundError(f'Checkpoint not found: {path}')


def resolve_run_config(config_file, checkpoint_path: Path) -> dict:
    run_cfg = checkpoint_path.parent / 'config_file.yaml'
    if run_cfg.is_file():
        print(f'Using run config: {run_cfg}')
        return read_yaml(str(run_cfg))
    print(f'Using config: {config_file}')
    return read_yaml(str(config_file))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare PLANMM local-control inference files.',
    )
    parser.add_argument(
        '--config_file',
        type=Path,
        default=str(_REPO_ROOT / 'exp_poisson' / 'planmm_hack_manipulation.yaml'),
        help='Path to the manipulation configuration file.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help=(
            'Checkpoint .pth, run directory, or save_dir base. '
            'Default: latest timestamp run under CHECKPOINT.save_dir'
        ),
    )
    parser.add_argument('--machine', type=str, default='local', help='Machine name')
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5,
        help='Epochs to gather displacement statistics (recommended >1).',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='CUDA device index or "cpu".',
    )
    return parser.parse_args()


def affine_transform(source_points, target_points):
    """Least-squares affine map source → target; returns augmented 4x4 matrix."""
    assert source_points.shape == target_points.shape
    num_points = source_points.shape[0]
    ones_column = np.ones((num_points, 1))
    source_points_augmented = np.hstack([source_points, ones_column])
    affine_matrix, _, _, _ = lstsq(source_points_augmented, target_points)
    return np.vstack([affine_matrix.T, [0, 0, 0, 1]])


def transform_points(points, affine_matrix):
    num_points = points.shape[0]
    ones_column = np.ones((num_points, 1))
    points_homogeneous = np.hstack([points, ones_column])
    transformed = np.dot(points_homogeneous, affine_matrix.T)
    return transformed[:, :3]


def get_region_boundaries(mesh, fpid):
    if isinstance(mesh, (str, Path)):
        mesh = read_obj(str(mesh))
    faces = np.asarray(mesh.fvs, dtype=np.int64)

    v2r = {}
    for reg, vals in fpid.items():
        for v in vals:
            if v not in v2r:
                v2r[v] = reg

    bounds_dict = {k: [] for k in fpid.keys()}
    for face in faces:
        if not all(f in v2r for f in face):
            continue
        regs = [v2r[f] for f in face]
        if len(set(regs)) > 1:
            for f in face:
                bounds_dict[v2r[f]].append(int(f))

    for k, v in bounds_dict.items():
        bounds_dict[k] = list(set(v))
    return bounds_dict


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg.startswith('cuda'):
        return torch.device(device_arg)
    return torch.device(f'cuda:{device_arg}')


def prepare_inference_files(args):
    config_file = Path(args.config_file)
    checkpoint_path = resolve_checkpoint_path(config_file, getattr(args, 'checkpoint', None))
    run_dir = checkpoint_path.parent
    savedir = str(run_dir)
    os.makedirs(savedir, exist_ok=True)
    print(f'Run directory: {savedir}')
    print(f'Checkpoint: {checkpoint_path}')

    num_epochs = int(getattr(args, 'num_epochs', 5))
    device = _resolve_device(str(getattr(args, 'device', '0')))

    config = resolve_run_config(config_file, checkpoint_path)
    config['MACHINE'] = getattr(args, 'machine', 'local')

    with open(config['MODEL']['region_info_file'], 'rb') as f:
        region_info = pickle.load(f)
    region_info = {k: v for k, v in region_info.items() if 'inner' not in k}

    if 'control_vertices' in config['MODEL']:
        control_lms = {int(k): v for k, v in config['MODEL']['control_vertices'].items()}
        print('control lms: ', list(control_lms.keys()))
    else:
        control_lms = None

    template_path = config['MODEL'].get(
        'reference_obj',
        str(_REPO_ROOT / 'dataset' / 'hack_template.obj'),
    )
    mesh = read_obj(str(template_path))

    region_ids = {
        i: [int(v) for v in region_info[REGION_NAMES[i]]['vids']]
        for i in range(10)
    }
    boundaries = {int(k): v for k, v in get_region_boundaries(mesh, region_ids).items()}
    boundaries_savename = f'{savedir}/region_boundaries.pickle'
    with open(boundaries_savename, 'wb') as file:
        pickle.dump(boundaries, file)
    print(f'Wrote {boundaries_savename}')

    dataloaders, _, _ = get_dataloaders(config, 'local', 0, 1)
    train_loader = dataloaders['training']

    model_cfg = dict(config['MODEL'])
    model_cfg['manipulation'] = False
    net = build_planmm_net(model_cfg, device)
    load_from_checkpoint(net, str(checkpoint_path), partial_restore=True, device=device)
    net.eval()

    Z = []
    Dlms = {key: [] for key in control_lms.keys()} if control_lms else {}

    with torch.no_grad():
        for epoch in range(num_epochs):
            print(f'epoch {epoch + 1} of {num_epochs}')
            for sample in tqdm(train_loader, leave=False):
                source = sample['verts']
                if epoch == 0:
                    id_token, _ = net.encode(source.to(device), return_layer_outputs=False)
                    Z.append(id_token[:, 0].detach().cpu())

                source_np = source.cpu().numpy()
                target_np = torch.roll(source, 1, 0).cpu().numpy()
                B = source_np.shape[0]

                if control_lms is None:
                    continue
                for k in range(B):
                    for name in control_lms.keys():
                        source_ = source_np[k]
                        modified_ = source_.copy()
                        target_ = target_np[k]

                        idx = boundaries[name]
                        trg_fp = target_[region_ids[name]]
                        src_bound = source_[idx]
                        trg_bound = target_[idx]
                        M = affine_transform(trg_bound, src_bound)

                        if (np.abs(np.diag(M)) > 1).any():
                            continue

                        target_aligned = transform_points(trg_fp, M)
                        modified_[region_ids[name]] = target_aligned
                        Dlms[name].append((modified_ - source_)[control_lms[name]])

    Z = torch.cat(Z).cpu().detach().numpy()
    mu = np.mean(Z, axis=0)
    sigma = np.cov(Z, rowvar=0)
    gaussian_id_savename = f'{savedir}/gaussian_id.pickle'
    with open(gaussian_id_savename, 'wb') as f:
        pickle.dump({'mean': mu, 'sigma': sigma}, f)
    print(f'Wrote {gaussian_id_savename} (N={len(Z)})')

    displacement_stats = {}
    for key in Dlms.keys():
        X = np.stack(Dlms[key])
        N = len(Dlms[key])
        X = X.reshape((N, -1))
        displacement_stats[key] = {
            'mean': np.mean(X, axis=0),
            'std': np.cov(X, rowvar=0),
        }
    displ_savename = f'{savedir}/displacement_stats.pickle'
    with open(displ_savename, 'wb') as file:
        pickle.dump(displacement_stats, file)
    print(f'Wrote {displ_savename}')

    return Path(boundaries_savename), Path(displ_savename), Path(gaussian_id_savename)


if __name__ == '__main__':
    prepare_inference_files(parse_args())
