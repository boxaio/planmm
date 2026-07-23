"""Polyscope viewer for PLANMM encoder multilayer mesh outputs.

Loads a trained PLANMM checkpoint, runs ``encode()`` on a HACK mesh sample, and
visualizes each encoder layer mesh interactively.
"""

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
    build_planmm_net,
    compute_mean_shape_from_neck_pose,
)
from networks.planmm import PLANMM
from utils.mesh import read_obj
from utils.torch_utils import load_from_checkpoint

DEFAULT_CHECKPOINT = (
    _REPO_ROOT / "results" / "planmm_hack_ae" / "20260707_202137" / "best.pth"
)
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.parent / "config_file.yaml"
DEFAULT_REFERENCE_OBJ = _REPO_ROOT / "dataset" / "hack_template.obj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PLANMM encoder layer meshes in Polyscope.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "PLANMM training yaml (MODEL section used to build the network); "
            "defaults to config_file.yaml next to --checkpoint."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Trained PLANMM checkpoint (.pth).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Optional run directory containing config_file.yaml and best.pth; "
            "overrides --config/--checkpoint when set."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help='CUDA device id, e.g. "0", or "cpu".',
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=488,
        help="HACK val dataset sample index.",
    )
    parser.add_argument(
        "--packed-pt",
        type=Path,
        default=VAL_HACK_PT,
        help="Packed HACK dataset for loading GT meshes.",
    )
    parser.add_argument(
        "--reference-obj",
        type=Path,
        default=DEFAULT_REFERENCE_OBJ,
        help="Template OBJ for face topology (fallback if not in packed pt).",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = _REPO_ROOT / run_dir
        config_path = run_dir / "config_file.yaml"
        checkpoint_path = run_dir / "best.pth"
        if not config_path.is_file():
            raise FileNotFoundError(f"missing run config: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
        return config_path, checkpoint_path

    checkpoint_path = Path(args.checkpoint or DEFAULT_CHECKPOINT)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _REPO_ROOT / checkpoint_path

    if args.config is None:
        config_path = checkpoint_path.parent / "config_file.yaml"
    else:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = _REPO_ROOT / config_path

    if not config_path.is_file():
        raise FileNotFoundError(f"missing config: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    return config_path, checkpoint_path


def load_faces(reference_obj: Path, packed_pt: Path) -> np.ndarray:
    if packed_pt.is_file():
        data = torch.load(packed_pt, map_location="cpu", weights_only=False)
        faces = data.get("faces")
        if faces is not None:
            if isinstance(faces, torch.Tensor):
                return faces.detach().cpu().numpy().astype(np.int64)
            return np.asarray(faces, dtype=np.int64)
    obj = read_obj(str(reference_obj), tri=True)
    return np.asarray(obj.fvs, dtype=np.int64)


def build_net(config_path: Path, checkpoint_path: Path, device: torch.device) -> PLANMM:
    config = read_yaml(str(config_path))
    net = build_planmm_net(config["MODEL"], device)
    load_from_checkpoint(net, str(checkpoint_path), partial_restore=True, device=device)
    net.eval()
    return net


@torch.no_grad()
def run_encoder_layers(
    net: PLANMM,
    verts: torch.Tensor,
    neck_pose: torch.Tensor,
    hackhead,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (gt [V,3], encoder_layers [L,V,3])."""
    x = verts.unsqueeze(0).to(device)
    x_neutral = compute_mean_shape_from_neck_pose(
        hackhead, neck_pose, net.template_verts,
    ).to(device)
    if x_neutral.ndim == 2:
        x_neutral = x_neutral.unsqueeze(0)

    _, encoder_outputs, _, _ = net.encode(x, x_neutral=x_neutral)
    gt = x[0].detach().cpu().numpy().astype(np.float64)
    # encoder_outputs: [L_enc+1, B, V, 3]; layer 0 is the input mesh passthrough.
    layers = encoder_outputs[1:, 0].detach().cpu().numpy().astype(np.float64)
    return gt, layers


def layer_palette(n_layers: int) -> list[tuple[float, float, float]]:
    base = np.linspace(0.15, 0.95, n_layers)
    colors = []
    for i, t in enumerate(base):
        # cool encoder progression: teal -> blue -> purple
        colors.append(
            (
                0.25 + 0.15 * (1.0 - t),
                0.45 + 0.25 * t,
                0.55 + 0.35 * t + 0.05 * i / max(n_layers - 1, 1),
            ),
        )
    return colors


def compute_cumulative_x_offsets(
    verts_list: list[np.ndarray],
    margin: float = 1.15,
) -> list[np.ndarray]:
    """Place meshes left-to-right using each mesh's own X extent."""
    offsets: list[np.ndarray] = []
    cursor = 0.0
    for verts in verts_list:
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        span = vmax - vmin
        offset_x = cursor - float(vmin[0])
        offsets.append(np.array([offset_x, 0.0, 0.0], dtype=np.float64))
        cursor += float(span[0]) * margin
    return offsets


class PLANMMEncoderLayerViewer:
    def __init__(
        self,
        gt_verts: np.ndarray,
        encoder_layers: np.ndarray,
        faces: np.ndarray,
        *,
        sample_idx: int,
    ) -> None:
        self.gt_verts = gt_verts
        self.encoder_layers = encoder_layers
        self.faces = faces
        self.sample_idx = sample_idx
        self.n_layers = int(encoder_layers.shape[0])

        self.layer_idx = 0
        self.show_gt = True
        self.show_encoder_layers = True
        self.color_by_error = False
        self.x_margin = 1.15
        self._slot_offsets = self._compute_slot_offsets()

        ps.init()
        ps.set_program_name("viz_planmm_encoder")
        ps.set_ground_plane_mode("none")

        self._gt_name = "gt"
        self._enc_names = [f"encoder_layer_{i}" for i in range(self.n_layers)]
        self._palette = layer_palette(self.n_layers)

        self._register_all_meshes()
        ps.set_user_callback(self._ui_callback)

    def _mesh_list(self) -> list[np.ndarray]:
        return [self.gt_verts, *list(self.encoder_layers)]

    def _compute_slot_offsets(self) -> list[np.ndarray]:
        return compute_cumulative_x_offsets(self._mesh_list(), margin=self.x_margin)

    def _slot_offset(self, slot: int) -> np.ndarray:
        return self._slot_offsets[slot]

    def _mesh_bbox_span(self, slot: int) -> np.ndarray:
        """Axis-aligned bbox extent [dx, dy, dz] in original mesh coordinates."""
        verts = self._mesh_list()[slot]
        return verts.max(axis=0) - verts.min(axis=0)

    def _mesh_scale_summary(self, slot: int) -> tuple[np.ndarray, float, float]:
        span = self._mesh_bbox_span(slot)
        max_span = float(span.max())
        gt_max = float(self._mesh_bbox_span(0).max())
        ratio = max_span / gt_max if gt_max > 1e-12 else float("nan")
        return span, max_span, ratio

    def _layer_l1(self, layer_idx: int) -> float:
        return float(np.abs(self.encoder_layers[layer_idx] - self.gt_verts).mean())

    def _register_all_meshes(self) -> None:
        ps.register_surface_mesh(
            self._gt_name,
            self.gt_verts + self._slot_offset(0),
            self.faces,
            color=(0.72, 0.82, 0.95),
            edge_width=0.5,
            smooth_shade=True,
            material="clay",
            enabled=self.show_gt,
        )

        for i in range(self.n_layers):
            self._register_encoder_mesh(i)

    def _register_encoder_mesh(self, layer_idx: int) -> None:
        name = self._enc_names[layer_idx]
        if ps.has_surface_mesh(name):
            ps.remove_surface_mesh(name)

        verts = self.encoder_layers[layer_idx] + self._slot_offset(layer_idx + 1)
        mesh = ps.register_surface_mesh(
            name,
            verts,
            self.faces,
            color=self._palette[layer_idx],
            edge_width=0.8 if layer_idx == self.layer_idx else 0.4,
            smooth_shade=True,
            material="clay",
            enabled=self.show_encoder_layers,
        )
        if self.color_by_error:
            err = np.linalg.norm(self.encoder_layers[layer_idx] - self.gt_verts, axis=1)
            mesh.add_scalar_quantity("l1_error", err, enabled=True, cmap="reds")

    def _refresh_meshes(self) -> None:
        ps.get_surface_mesh(self._gt_name).update_vertex_positions(
            self.gt_verts + self._slot_offset(0),
        )
        ps.get_surface_mesh(self._gt_name).set_enabled(self.show_gt)

        for i in range(self.n_layers):
            name = self._enc_names[i]
            if ps.has_surface_mesh(name):
                ps.remove_surface_mesh(name)
            self._register_encoder_mesh(i)

    def _ui_callback(self) -> None:
        psim.TextUnformatted(f"sample_idx = {self.sample_idx}")
        psim.TextUnformatted(
            "layout: cumulative X offsets from per-mesh bbox width",
        )
        psim.Separator()

        changed = False
        margin_changed, self.x_margin = psim.SliderFloat(
            "x-margin (gap scale)",
            self.x_margin,
            1.0,
            2.5,
        )
        if margin_changed:
            self._slot_offsets = self._compute_slot_offsets()
        changed = changed or margin_changed

        gt_span, _, _ = self._mesh_scale_summary(0)
        psim.TextUnformatted(
            f"GT scale: dx={gt_span[0]:.4f}, dy={gt_span[1]:.4f}, dz={gt_span[2]:.4f}",
        )
        for i in range(self.n_layers):
            span, _, ratio = self._mesh_scale_summary(i + 1)
            psim.TextUnformatted(
                f"layer {i} scale: dx={span[0]:.4f}, dy={span[1]:.4f}, dz={span[2]:.4f}, "
                f"vs_GT={ratio:.4f}",
            )

        psim.Separator()
        changed_layer, self.layer_idx = psim.SliderInt(
            "highlight layer",
            self.layer_idx,
            0,
            self.n_layers - 1,
        )
        changed = changed or changed_layer
        psim.TextUnformatted(f"L1 vs GT: {self._layer_l1(self.layer_idx):.6f}")

        if psim.Button("prev layer") and self.layer_idx > 0:
            self.layer_idx -= 1
            changed = True
        psim.SameLine()
        if psim.Button("next layer") and self.layer_idx < self.n_layers - 1:
            self.layer_idx += 1
            changed = True

        psim.Separator()
        gt_changed, self.show_gt = psim.Checkbox("show GT", self.show_gt)
        enc_changed, self.show_encoder_layers = psim.Checkbox(
            "show encoder layers",
            self.show_encoder_layers,
        )
        err_changed, self.color_by_error = psim.Checkbox(
            "color encoder by L1 error",
            self.color_by_error,
        )
        changed = changed or gt_changed or enc_changed or err_changed

        psim.Separator()
        psim.TextUnformatted("per-layer L1:")
        for i in range(self.n_layers):
            marker = ">" if i == self.layer_idx else " "
            psim.TextUnformatted(f"{marker} layer {i}: {self._layer_l1(i):.6f}")

        if changed:
            self._refresh_meshes()

    def show(self) -> None:
        print(
            f"[viz_planmm_encoder] Polyscope: {self.n_layers} encoder layers, "
            f"sample_idx={self.sample_idx}",
        )
        print(f"  GT vertex mean: {self.gt_verts.mean(axis=0)}")
        for i in range(self.n_layers):
            print(
                f"  layer {i}: L1={self._layer_l1(i):.6f}, "
                f"vertex_mean={self.encoder_layers[i].mean(axis=0)}",
            )
        ps.show()


def main() -> None:
    args = parse_args()
    config_path, checkpoint_path = resolve_paths(args)

    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    if not args.packed_pt.is_file():
        raise FileNotFoundError(f"packed dataset not found: {args.packed_pt}")

    config = read_yaml(str(config_path))
    faces = load_faces(args.reference_obj, args.packed_pt)
    dataset = HACKDataset(packed_pt=args.packed_pt)
    if args.sample_idx < 0 or args.sample_idx >= len(dataset):
        raise IndexError(f"sample_idx {args.sample_idx} out of range [0, {len(dataset)})")

    sample = dataset[args.sample_idx]
    gt_verts = sample["verts"].detach().cpu().numpy().astype(np.float64)
    neck_pose = sample["neck_pose"]

    hackhead = build_hackhead_for_mean_shape(config["MODEL"], device)
    net = build_net(config_path, checkpoint_path, device)
    gt_np, encoder_layers = run_encoder_layers(
        net, sample["verts"], neck_pose, hackhead, device,
    )

    print(f"[viz_planmm_encoder] vertex mean (sample_idx={args.sample_idx}):")
    print(f"  GT: {gt_np.mean(axis=0)}")
    for i in range(encoder_layers.shape[0]):
        print(f"  encoder layer {i}: {encoder_layers[i].mean(axis=0)}")

    viewer = PLANMMEncoderLayerViewer(
        gt_np,
        encoder_layers,
        faces,
        sample_idx=args.sample_idx,
    )
    viewer.show()


if __name__ == "__main__":
    main()
