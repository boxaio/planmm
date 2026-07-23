"""Load PoissonAGC checkpoint, dARAP-refine predictions, and visualize all frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NICP_DIR = _REPO_ROOT / "meshes" / "nonrigid_icp"
for _path in (str(_REPO_ROOT), str(_NICP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from exp_poisson.train_poisson_track_phack import PHACKMeshDataset
from meshes.nonrigid_icp.deformations_MINIMAL import (
    ProcrustesPrecompute,
    SparseLaplaciansSolvers,
    calc_ARAP_global_solve,
    calc_rot_matrices_with_procrustes,
    vertex_procrustes_lambda_from_regions,
)
from meshes.operators import construct_mesh_operators
from networks.PoissonAGC import build_poisson_model
from utils.helpers import seed_everything, to_np
from utils.mesh import vertex_normals

DEFAULT_CKPT = (
    _REPO_ROOT / "results" / "poisson_agc_nersemble_phack" / "poisson_agc_nersemble_phack_final.pt"
)
CONFIG_PATH = _REPO_ROOT / "exp_poisson" / "nersemble_phack_config.json"

LAMBDA_DEFAULT = 3.0
LAMBDA_DEFORM = 6.5


def load_poisson_agc(checkpoint: Path, config: dict, device: torch.device):
    model = build_poisson_model(
        config,
        C_in=3,
        C_out=3,
        C_width=config["width"],
        n_blocks=config["nblocks"],
        head_type="njf",
        extra_features=1,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def predict_all_frames(
    model,
    dataset: PHACKMeshDataset,
    device: torch.device,
    *,
    center_outputs: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts_src, faces = dataset.get_source_cloth()
    verts_src = verts_src.to(device).unsqueeze(0)
    faces = faces.to(device).unsqueeze(0)
    mass, solver, G, M = construct_mesh_operators(verts_src, faces, high_precision=True)

    n_frames = len(dataset)
    preds_list: list[np.ndarray] = []
    for frame_idx in tqdm(range(n_frames), desc="predict frames", dynamic_ncols=True):
        t_val = dataset.t_vals[frame_idx].to(device).view(1, 1)
        preds, _ = model(
            x_in=verts_src,
            M=M,
            G=G,
            solver=solver,
            faces=faces,
            vertex_mass=mass,
            extra_features=t_val,
        )
        if center_outputs:
            preds = preds - preds.mean(dim=1, keepdim=True)
        preds_list.append(to_np(preds[0]))

    predictions = np.stack(preds_list, axis=0)
    targets = dataset.verts.numpy()
    faces_np = to_np(faces[0])
    return predictions, targets, faces_np


def deform_pred_to_target_normals(
    pred_verts: np.ndarray,
    faces: np.ndarray,
    tgt_verts: np.ndarray,
    *,
    lambda_default: float = LAMBDA_DEFAULT,
    lambda_deform: float = LAMBDA_DEFORM,
) -> np.ndarray:
    """dARAP: deform predicted mesh toward per-vertex normals of the target mesh."""
    tgt_normals = vertex_normals(
        torch.tensor(tgt_verts[None], dtype=torch.float32),
        torch.tensor(faces[None], dtype=torch.long),
    )[0].numpy()

    verts_list = [torch.tensor(pred_verts, dtype=torch.float32)]
    faces_list = [torch.tensor(faces, dtype=torch.long)]
    target_normals_list = [torch.tensor(tgt_normals, dtype=torch.float32)]

    y = pred_verts[:, 1]
    upper_half_vertex_indices = np.where(y >= np.median(y))[0].tolist()
    local_step_procrustes_lambda = vertex_procrustes_lambda_from_regions(
        len(pred_verts),
        default_lambda=lambda_default,
        region_vertex_indices=[upper_half_vertex_indices],
        region_lambdas=[lambda_deform],
    )

    solvers = SparseLaplaciansSolvers.from_meshes(
        verts_list,
        faces_list,
        pin_first_vertex=True,
        compute_poisson_rhs_lefts=False,
        compute_igl_arap_rhs_lefts=None,
    )
    procrustes_precompute = ProcrustesPrecompute.from_meshes(
        local_step_procrustes_lambda=local_step_procrustes_lambda,
        arap_energy_type="spokes_and_rims_mine",
        laplacians_solvers=solvers,
        verts_list=verts_list,
        faces_list=faces_list,
    )
    per_vertex_3x3matrices_packed = calc_rot_matrices_with_procrustes(
        procrustes_precompute,
        torch.cat(verts_list),
        torch.cat(target_normals_list),
    )
    per_vertex_3x3matrices_list = torch.split(
        per_vertex_3x3matrices_packed,
        [len(v) for v in verts_list],
    )
    deformed_verts_list = calc_ARAP_global_solve(
        verts_list,
        faces_list,
        solvers,
        per_vertex_3x3matrices_list,
        arap_energy_type="spokes_and_rims_mine",
        postprocess="recenter_only",
    )
    return deformed_verts_list[0].detach().cpu().numpy()


def deform_all_frames(
    predictions: np.ndarray,
    targets: np.ndarray,
    faces: np.ndarray,
    *,
    lambda_default: float = LAMBDA_DEFAULT,
    lambda_deform: float = LAMBDA_DEFORM,
) -> np.ndarray:
    n_frames = predictions.shape[0]
    deformed_list: list[np.ndarray] = []
    for frame_idx in tqdm(range(n_frames), desc="dARAP deform", dynamic_ncols=True):
        deformed = deform_pred_to_target_normals(
            predictions[frame_idx],
            faces,
            targets[frame_idx],
            lambda_default=lambda_default,
            lambda_deform=lambda_deform,
        )
        deformed_list.append(deformed)
    return np.stack(deformed_list, axis=0)


def _chain_x_offsets(
    meshes: list[np.ndarray],
    gap: float | None = None,
) -> list[float]:
    """Return x-offset for each mesh so they sit side by side left-to-right."""
    all_min = np.minimum.reduce([v.min(axis=0) for v in meshes])
    all_max = np.maximum.reduce([v.max(axis=0) for v in meshes])
    combined_diag = float(np.linalg.norm(all_max - all_min))
    if gap is None:
        gap = float(max(combined_diag * 0.05, 1e-6))

    offsets: list[float] = []
    cursor = 0.0
    for verts in meshes:
        x_min = float(verts[:, 0].min())
        x_max = float(verts[:, 0].max())
        offset = cursor - x_min
        offsets.append(offset)
        cursor = x_max + offset + gap
    return offsets


class PoissonAGCSequenceViewer:
    def __init__(
        self,
        dataset: PHACKMeshDataset,
        predictions: np.ndarray,
        targets: np.ndarray,
        deformed: np.ndarray,
        faces: np.ndarray,
        *,
        show_edge: bool = False,
        default_frame: int = 0,
    ) -> None:
        self.dataset = dataset
        self.predictions = np.asarray(predictions, dtype=np.float64)
        self.targets = np.asarray(targets, dtype=np.float64)
        self.deformed = np.asarray(deformed, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.show_edge = show_edge

        n_frames = self.predictions.shape[0]
        for name, arr in (("predictions", self.predictions), ("targets", self.targets), ("deformed", self.deformed)):
            if arr.shape != self.predictions.shape:
                raise ValueError(f"{name} shape mismatch: {arr.shape} vs {self.predictions.shape}")

        self.frame_idx = int(np.clip(default_frame, 0, n_frames - 1))
        self.play = False
        self.play_fps = 24.0
        self._play_accum = 0.0
        self._loaded_frame = -1

        pred0, tgt0, def0 = self.predictions[0], self.targets[0], self.deformed[0]
        self._x_offsets = _chain_x_offsets([pred0, tgt0, def0])

        ps.init()
        ps.set_program_name("demo_poissonagc_dARAP")
        ps.set_ground_plane_mode("none")
        edge_width = 0.9 if show_edge else 0.0
        self._pred_mesh = ps.register_surface_mesh(
            "prediction",
            self._shift_x(pred0, 0),
            self.faces,
            color=(0.72, 0.72, 0.78),
            edge_width=edge_width,
            smooth_shade=True,
            material="clay",
            back_face_policy="custom",
        )
        self._tgt_mesh = ps.register_surface_mesh(
            "target",
            self._shift_x(tgt0, 1),
            self.faces,
            color=(0.78, 0.72, 0.68),
            edge_width=edge_width,
            smooth_shade=True,
            material="clay",
            back_face_policy="custom",
        )
        self._def_mesh = ps.register_surface_mesh(
            "dARAP",
            self._shift_x(def0, 2),
            self.faces,
            color=(0.68, 0.78, 0.72),
            edge_width=edge_width,
            smooth_shade=True,
            material="clay",
            back_face_policy="custom",
        )
        self._set_frame(self.frame_idx)
        ps.set_user_callback(self._ui_callback)

    def _shift_x(self, verts: np.ndarray, mesh_idx: int) -> np.ndarray:
        out = verts.copy()
        out[:, 0] += self._x_offsets[mesh_idx]
        return out

    @property
    def n_frames(self) -> int:
        return int(self.predictions.shape[0])

    def _set_frame(self, frame_idx: int) -> None:
        frame_idx = int(np.clip(frame_idx, 0, self.n_frames - 1))
        if frame_idx == self._loaded_frame:
            return

        self._pred_mesh.update_vertex_positions(self._shift_x(self.predictions[frame_idx], 0))
        self._tgt_mesh.update_vertex_positions(self._shift_x(self.targets[frame_idx], 1))
        self._def_mesh.update_vertex_positions(self._shift_x(self.deformed[frame_idx], 2))
        self.frame_idx = frame_idx
        self._loaded_frame = frame_idx

    def _ui_callback(self) -> None:
        n_frames = self.n_frames

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("PoissonAGC + dARAP"):
            psim.Separator()
            changed, self.frame_idx = psim.SliderInt(
                "frame",
                self.frame_idx,
                0,
                max(n_frames - 1, 0),
            )
            if changed:
                self._set_frame(self.frame_idx)

            orig_id = self.dataset.orig_frame_ids[self.frame_idx]
            t_val = float(self.dataset.t_vals[self.frame_idx].item())
            psim.Text(f"orig_frame_id: {orig_id:03d}")
            psim.Text(f"t: {t_val:.4f}")
            psim.Text(f"frames: {self.frame_idx + 1}/{n_frames}")

            changed_play, self.play = psim.Checkbox("play", self.play)
            if changed_play:
                self._play_accum = 0.0
            _, self.play_fps = psim.SliderFloat("fps", self.play_fps, 1.0, 60.0)
            psim.TreePop()

        if self.play and n_frames > 1:
            io = psim.GetIO()
            self._play_accum += io.DeltaTime
            frame_dt = 1.0 / max(self.play_fps, 1e-3)
            if self._play_accum >= frame_dt:
                steps = int(self._play_accum / frame_dt)
                self._play_accum -= steps * frame_dt
                next_idx = (self.frame_idx + steps) % n_frames
                self._set_frame(next_idx)

    def show(self) -> None:
        orig_ids = self.dataset.orig_frame_ids
        print(
            f"[demo_poissonagc_dARAP] subject={self.dataset.subject_id} "
            f"frames={self.n_frames} "
            f"orig_range={orig_ids[0]:03d}..{orig_ids[-1]:03d} | "
            f"meshes: prediction | target | dARAP",
            flush=True,
        )
        ps.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize PoissonAGC predictions, targets, and dARAP-refined meshes.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--subject-id", type=str, default="224", choices=["224", "473"])
    parser.add_argument("--start-frame", type=int, default=0, help="initial frame index")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--no-center", action="store_true", help="disable mean-centering on predictions")
    parser.add_argument("--show-edge", action="store_true")
    parser.add_argument("--lambda-default", type=float, default=LAMBDA_DEFAULT)
    parser.add_argument("--lambda-deform", type=float, default=LAMBDA_DEFORM)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(31415)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["detail_residual"] = {**config.get("detail_residual", {}), "enabled": False}

    print(f"Loading dataset subject {args.subject_id}...")
    dataset = PHACKMeshDataset(subject_id=args.subject_id)

    print(f"Loading checkpoint: {checkpoint}")
    model = load_poisson_agc(checkpoint, config, device)

    verts_src, faces_src = dataset.get_source_cloth()
    if hasattr(model, "register_source_mesh"):
        model.register_source_mesh(verts_src.to(device), faces_src.to(device))

    predictions, targets, faces = predict_all_frames(
        model,
        dataset,
        device,
        center_outputs=not args.no_center,
    )

    deformed = deform_all_frames(
        predictions,
        targets,
        faces,
        lambda_default=args.lambda_default,
        lambda_deform=args.lambda_deform,
    )

    viewer = PoissonAGCSequenceViewer(
        dataset,
        predictions,
        targets,
        deformed,
        faces,
        show_edge=args.show_edge,
        default_frame=args.start_frame,
    )
    viewer.show()


if __name__ == "__main__":
    main()
