"""Generate local PLANMM region manipulations.

Mirrors ``LAMM_generate_local.py``: per identity, per region, sample control Δ
from region MVN → ``net((x, delta))[-1]`` → save OBJs.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from copy import copy
from pathlib import Path

import numpy as np
import torch
import trimesh

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import read_yaml
from dataset.hack_seg_fids import REGION_NAMES
from exp_poisson.planmm_generate_samples import build_planmm_net
from exp_poisson.planmm_prepare_inference import (
    prepare_inference_files,
    resolve_checkpoint_path,
    resolve_run_config,
)
from exp_poisson.train_planmm_manipulation import get_dataloaders
from utils.helpers import seed_everything
from utils.mesh import read_obj
from utils.torch_utils import load_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Generate local PLANMM region manipulations.')
    parser.add_argument(
        '--config_file',
        type=Path,
        default=_REPO_ROOT / 'exp_poisson' / 'planmm_hack_manipulation.yaml',
        help='Path to the configuration file.',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help=(
            'Checkpoint .pth, run directory, or save_dir base. '
            'Default: latest timestamp under CHECKPOINT.save_dir '
            '(prefer best_alpha_max.pth).'
        ),
    )
    parser.add_argument(
        '--savedir_name',
        type=str,
        default='generated_local',
        help='Output subdirectory under the run directory. Pass "None" to skip saving.',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of source identities to manipulate.',
    )
    parser.add_argument(
        '--num_random_generations',
        type=int,
        default=5,
        help='Random local generations per region per identity.',
    )
    parser.add_argument(
        '--k_std',
        type=float,
        default=2.5,
        help='Std multiplier for sampled control-vertex displacements.',
    )
    parser.add_argument('--machine', type=str, default='local', help='Machine name.')
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='CUDA device index or "cpu".',
    )
    parser.add_argument(
        '--prepare_files',
        action='store_true',
        help='Run planmm_prepare_inference if displacement_stats.pickle is missing.',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5,
        help='Epochs for prepare_inference when --prepare_files is set.',
    )
    parser.add_argument(
        '--generate_random_source',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Sample random global identities; otherwise use eval dataloader meshes.',
    )
    parser.add_argument(
        '--gaussian_id',
        type=str,
        default=None,
        help='Optional gaussian_id.pickle for random sources (default: beside checkpoint).',
    )
    parser.add_argument('--seed', type=int, default=9926, help='Random seed.')
    return parser.parse_args()


def save_mesh(vertices: np.ndarray, faces: np.ndarray, path: str | Path) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type='obj')


def get_random_displacements(
    delta_stats: dict,
    key: int,
    control_lms: dict[int, list[int]],
    k_std: float = 1.0,
    device: str | torch.device = 'cpu',
) -> list[torch.Tensor]:
    """Sample control-vertex displacements for one region; others are zero.

    Each tensor is ``[1, 3 * n_control]`` to match PLANMM ``delta_control_net``.
    """
    delta = []
    for idx in control_lms.keys():
        n = 3 * len(control_lms[idx])
        if idx != key:
            delta.append(torch.zeros(1, n, device=device))
        else:
            cov = delta_stats[key].get('std', delta_stats[key].get('cov'))
            sample = np.random.multivariate_normal(
                delta_stats[key]['mean'],
                k_std * cov,
                1,
            )
            delta.append(torch.tensor(sample, dtype=torch.float32, device=device))
    return delta


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg.startswith('cuda'):
        return torch.device(device_arg)
    return torch.device(f'cuda:{device_arg}')


def _resolve_gaussian_path(
    run_dir: Path,
    gaussian_arg: str | None,
) -> Path:
    if gaussian_arg:
        path = Path(gaussian_arg)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f'gaussian_id not found: {path}')
        return path
    path = run_dir / 'gaussian_id.pickle'
    if not path.is_file():
        raise FileNotFoundError(
            f'gaussian_id.pickle not found at {path}. '
            'Run with --prepare_files or pass --gaussian_id.'
        )
    return path


def _generate_global_meshes(
    config: dict,
    checkpoint: str | Path,
    num_samples: int,
    k_std: float,
    device: torch.device,
    run_dir: Path,
    gaussian_path: Path,
    save_meshes: bool = True,
) -> list[dict]:
    """Sample random global identities from the latent Gaussian."""
    model_cfg = dict(config['MODEL'])
    model_cfg['manipulation'] = False
    net = build_planmm_net(model_cfg, device)
    load_from_checkpoint(net, str(checkpoint), partial_restore=True, device=device)
    net.eval()

    with open(gaussian_path, 'rb') as handle:
        gaussian_id = pickle.load(handle, encoding='latin1')
    mu = np.asarray(gaussian_id['mean'], dtype=np.float64)
    sigma = np.asarray(gaussian_id.get('sigma', gaussian_id.get('cov')), dtype=np.float64)

    z = torch.tensor(
        np.random.multivariate_normal(mu, k_std * sigma, num_samples),
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        out = net.decode(z.unsqueeze(1), return_layer_outputs=False).detach().cpu()

    template_path = model_cfg.get(
        'reference_obj',
        str(_REPO_ROOT / 'dataset' / 'hack_template.obj'),
    )
    mesh = read_obj(str(template_path))
    faces = mesh.fvs

    meshes = []
    out_dir = run_dir / 'generated_global'
    if save_meshes:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, verts in enumerate(out):
        meshes.append({'verts': verts.unsqueeze(0).to(torch.float32)})
        if save_meshes:
            save_mesh(verts.numpy(), faces, out_dir / f'{i}.obj')

    return meshes


def generate_local(args):
    config_file = Path(args.config_file)
    checkpoint_path = resolve_checkpoint_path(config_file, args.checkpoint)
    num_samples = args.num_samples
    num_random_generations = args.num_random_generations
    k_std = args.k_std
    device = _resolve_device(args.device)

    config = resolve_run_config(config_file, checkpoint_path)
    config['MACHINE'] = args.machine
    run_dir = checkpoint_path.parent

    control_lms = {int(k): v for k, v in config['MODEL']['control_vertices'].items()}

    if args.savedir_name != 'None':
        savedir = run_dir / args.savedir_name
        savedir.mkdir(parents=True, exist_ok=True)
    else:
        savedir = None

    seed_everything(args.seed)

    template_path = config['MODEL'].get(
        'reference_obj',
        str(_REPO_ROOT / 'dataset' / 'hack_template.obj'),
    )
    mesh = read_obj(str(template_path))
    faces = mesh.fvs

    delta_stats_path = run_dir / 'displacement_stats.pickle'
    if args.prepare_files or not delta_stats_path.is_file():
        prep_args = copy(args)
        prep_args.checkpoint = str(checkpoint_path)
        prep_args.num_epochs = getattr(args, 'num_epochs', 5)
        prepare_inference_files(prep_args)

    with open(delta_stats_path, 'rb') as file:
        delta_stats = pickle.load(file)

    if not args.generate_random_source:
        dataloaders, _, _ = get_dataloaders(config, 'local', 0, 1)
        data_loader = dataloaders['eval']
    else:
        gaussian_path = _resolve_gaussian_path(run_dir, args.gaussian_id)
        data_loader = _generate_global_meshes(
            config,
            checkpoint_path,
            num_samples,
            k_std,
            device,
            run_dir,
            gaussian_path,
            save_meshes=False,
        )

    model_cfg = dict(config['MODEL'])
    model_cfg['manipulation'] = True
    net = build_planmm_net(model_cfg, device)
    load_from_checkpoint(net, str(checkpoint_path), partial_restore=True, device=device)
    net.eval()

    num_saved_samples = 0
    source_meshes = []
    manipulated_meshes = []

    with torch.no_grad():
        for sample in data_loader:
            remaining_samples = num_samples - num_saved_samples
            if remaining_samples <= 0:
                break

            source = sample['verts'].to(device)
            batch_size = source.shape[0]
            if batch_size > remaining_samples:
                source = source[:remaining_samples]
                batch_size = remaining_samples

            for i in range(batch_size):
                num_saved_samples += 1
                sample_id = num_saved_samples
                print(f'generating identity {sample_id} of {num_samples}')

                x = source[i].unsqueeze(0)
                source_np = x[0].detach().cpu().numpy()
                source_meshes.append(source_np)
                if savedir is not None:
                    save_mesh(source_np, faces, savedir / f'id_{sample_id}_source.obj')

                sample_outputs = []
                for key in control_lms.keys():
                    for m in range(num_random_generations):
                        delta = get_random_displacements(
                            delta_stats, key, control_lms, k_std, device,
                        )
                        out = net((x, delta))[-1]
                        out_np = out[0].detach().cpu().numpy()

                        sample_outputs.append(
                            {'verts': torch.from_numpy(out_np).unsqueeze(0).float()}
                        )
                        if savedir is not None:
                            region_name = REGION_NAMES[key].lower()
                            save_mesh(
                                out_np,
                                faces,
                                savedir
                                / f'id_{sample_id}_region_{key}_{region_name}_sample_{m}.obj',
                            )

                manipulated_meshes.append(sample_outputs)

    return source_meshes, manipulated_meshes


if __name__ == '__main__':
    source_meshes, manipulated_meshes = generate_local(parse_args())
