"""Polyscope viewer: Poisson deform encoder along input -> neutral path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
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
from exp_poisson.viz_lamm_encoder import compute_cumulative_x_offsets
from networks.planmm import load_template_from_config
from networks.poisson_deform_encoder import (
    PoissonDeformEncoder,
    build_encoder_alphas,
    mesh_path_targets,
)


DEFAULT_SAVE_DIR = _REPO_ROOT / 'results' / 'poisson_encoder_feasibility'


def resolve_latest_checkpoint(save_dir: Path | None = None) -> Path:
    """Pick encoder.pth from the newest timestamp subfolder under save_dir."""
    base = Path(save_dir) if save_dir is not None else DEFAULT_SAVE_DIR
    if not base.is_absolute():
        base = _REPO_ROOT / base

    direct = base / 'encoder.pth'
    if direct.is_file():
        return direct

    candidates: list[tuple[str, Path]] = []
    if base.is_dir():
        for sub in base.iterdir():
            if not sub.is_dir():
                continue
            ckpt = sub / 'encoder.pth'
            if ckpt.is_file():
                candidates.append((sub.name, ckpt))

    if not candidates:
        raise FileNotFoundError(f'No encoder.pth found under {base}')

    latest = max(candidates, key=lambda item: item[0])[1]
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize Poisson deform encoder layers.')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=None,
        help=(
            'encoder.pth from validate_poisson_encoder.py run; '
            'default: latest timestamp under results/poisson_encoder_feasibility'
        ),
    )
    parser.add_argument(
        '--save-dir',
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help='Base directory to search for latest checkpoint when --checkpoint is omitted.',
    )
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--sample-idx', type=int, default=None)
    return parser.parse_args()


def layer_palette(n: int) -> list[tuple[float, float, float]]:
    base = np.linspace(0.2, 0.95, n)
    return [(0.35 + 0.4 * t, 0.55 + 0.2 * (1 - t), 0.75 - 0.3 * t) for t in base]


class PoissonEncoderViewer:
    def __init__(
        self,
        input_verts: np.ndarray,
        neutral_verts: np.ndarray,
        poisson_layers: np.ndarray,
        poisson_targets: np.ndarray,
        faces: np.ndarray,
        *,
        sample_idx: int,
    ) -> None:
        self.input_verts = input_verts
        self.neutral_verts = neutral_verts
        self.poisson_layers = poisson_layers
        self.poisson_targets = poisson_targets
        self.faces = faces
        self.sample_idx = sample_idx
        self.n_layers = int(poisson_layers.shape[0])

        self.layer_idx = 0
        self.show_input = True
        self.show_neutral = True
        self.show_poisson = True
        self.show_error_heatmap = False
        self.x_margin = 1.15
        self._palette = layer_palette(self.n_layers)
        self._mesh_names = [
            'input', 'neutral',
            *[f'poisson_{i}' for i in range(self.n_layers)],
        ]

        ps.init()
        ps.set_program_name('viz_poisson_encoder')
        ps.set_ground_plane_mode('none')
        self._refresh_offsets()
        self._register_meshes()
        ps.set_user_callback(self._ui_callback)

    def _mesh_list(self) -> list[np.ndarray]:
        items = []
        if self.show_input:
            items.append(self.input_verts)
        if self.show_neutral:
            items.append(self.neutral_verts)
        if self.show_poisson:
            items.extend(list(self.poisson_layers))
        return items if items else [self.input_verts]

    def _layer_error(self, layer_idx: int) -> float:
        return float(
            np.abs(self.poisson_layers[layer_idx] - self.poisson_targets[layer_idx]).mean(),
        )

    def _layer_vertex_error(self, layer_idx: int) -> np.ndarray:
        return np.linalg.norm(
            self.poisson_layers[layer_idx] - self.poisson_targets[layer_idx],
            axis=1,
        )

    def _refresh_offsets(self) -> None:
        base = [
            self.input_verts,
            self.neutral_verts,
            *list(self.poisson_layers),
        ]
        self._offsets = compute_cumulative_x_offsets(base, margin=self.x_margin)
        self._slot = {
            'input': 0,
            'neutral': 1,
            'poisson': 2,
        }

    def _off(self, slot: int) -> np.ndarray:
        return self._offsets[slot]

    def _mesh_bbox_span(self, verts: np.ndarray) -> np.ndarray:
        return verts.max(axis=0) - verts.min(axis=0)

    def _register_meshes(self) -> None:
        for name in self._mesh_names:
            if ps.has_surface_mesh(name):
                ps.remove_surface_mesh(name)

        ps.register_surface_mesh(
            'input',
            self.input_verts + self._off(self._slot['input']),
            self.faces,
            color=(0.72, 0.82, 0.95),
            enabled=self.show_input,
        )
        ps.register_surface_mesh(
            'neutral',
            self.neutral_verts + self._off(self._slot['neutral']),
            self.faces,
            color=(0.55, 0.85, 0.55),
            enabled=self.show_neutral,
        )

        for i in range(self.n_layers):
            self._register_poisson_mesh(i)

    def _register_poisson_mesh(self, layer_idx: int) -> None:
        name = f'poisson_{layer_idx}'
        verts = self.poisson_layers[layer_idx] + self._off(self._slot['poisson'] + layer_idx)
        mesh = ps.register_surface_mesh(
            name,
            verts,
            self.faces,
            color=self._palette[layer_idx],
            edge_width=0.8 if layer_idx == self.layer_idx else 0.3,
            enabled=self.show_poisson,
        )
        if self.show_error_heatmap:
            mesh.add_scalar_quantity(
                'pred_tgt_error',
                self._layer_vertex_error(layer_idx),
                enabled=True,
                cmap='reds',
            )

    def _ui_callback(self) -> None:
        psim.TextUnformatted(f'sample_idx = {self.sample_idx}')
        psim.TextUnformatted('Poisson path: input -> neutral (mesh + grad targets)')
        psim.Separator()

        changed = False
        m_changed, self.x_margin = psim.SliderFloat('x-margin', self.x_margin, 1.0, 2.5)
        changed = changed or m_changed

        _, self.layer_idx = psim.SliderInt('highlight layer', self.layer_idx, 0, self.n_layers - 1)

        psim.Separator()
        for label, attr in [
            ('show input', 'show_input'),
            ('show neutral', 'show_neutral'),
            ('show poisson pred', 'show_poisson'),
            ('show pred-target error heatmap', 'show_error_heatmap'),
        ]:
            c, val = psim.Checkbox(label, getattr(self, attr))
            if c:
                setattr(self, attr, val)
                changed = True

        psim.Separator()
        psim.TextUnformatted(
            f'highlight L{self.layer_idx} pred-target L1: '
            f'{self._layer_error(self.layer_idx):.6f}',
        )
        gt_span = self._mesh_bbox_span(self.input_verts)
        psim.TextUnformatted(
            f'input span: dx={gt_span[0]:.4f} dy={gt_span[1]:.4f} dz={gt_span[2]:.4f}',
        )
        for i in range(self.n_layers):
            p_span = self._mesh_bbox_span(self.poisson_layers[i])
            marker = '>' if i == self.layer_idx else ' '
            psim.TextUnformatted(
                f'{marker} L{i} vs_GT={p_span.max()/max(gt_span.max(),1e-8):.4f} '
                f'pred_tgt_L1={self._layer_error(i):.6f}',
            )

        if changed or m_changed:
            self._refresh_offsets()
            self._register_meshes()

    def show(self) -> None:
        ps.show()


def main() -> None:
    args = parse_args()
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = _REPO_ROOT / ckpt_path
    else:
        ckpt_path = resolve_latest_checkpoint(args.save_dir)
        print(f'[viz_poisson_encoder] using latest checkpoint: {ckpt_path}')

    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = payload.get('config') or read_yaml(
        str(_REPO_ROOT / 'exp_poisson' / 'validate_poisson_encoder.yaml'),
    )
    model_cfg = config.get('MODEL', {})
    data_cfg = config.get('DATA', {})

    device_str = args.device
    device = torch.device(
        'cpu' if device_str == 'cpu' or not torch.cuda.is_available()
        else f'cuda:{device_str}',
    )

    sample_idx = args.sample_idx if args.sample_idx is not None else int(payload.get('sample_idx', data_cfg.get('sample_idx', 0)))
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
    enc = PoissonDeformEncoder(
        n_stages=n_stages,
        C_width=int(model_cfg.get('C_width', 128)),
        config=dict(model_cfg.get('poisson_cfg', {})),
    ).to(device)
    enc.load_state_dict(payload['model'])
    enc.register_template_mesh(template_verts, template_faces)
    enc.eval()

    alphas = build_encoder_alphas(n_stages, device=device)
    mesh_tgt = mesh_path_targets(x_in, x_neutral, alphas)

    with torch.no_grad():
        layer_meshes, _ = enc(x_in, x_neutral=x_neutral, alphas=alphas)

    faces_np = template_faces.detach().cpu().numpy()
    viewer = PoissonEncoderViewer(
        x_in[0].cpu().numpy(),
        x_neutral[0].cpu().numpy(),
        layer_meshes[:, 0].cpu().numpy(),
        mesh_tgt[:, 0].cpu().numpy(),
        faces_np,
        sample_idx=sample_idx,
    )
    viewer.show()


if __name__ == '__main__':
    main()
