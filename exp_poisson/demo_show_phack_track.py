"""Polyscope viewer for phack_repair mesh sequences (per NeRSemble subject)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import trimesh

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_REPAIR_DIR = Path(__file__).resolve().parent / "phack_nersemble" / "phack_repair"
SUBJECT_CONFIGS: dict[str, str] = {
    "224": "224_cam_222200037",
    "473": "473_cam_222200037",
}
_FRAME_RE = re.compile(r"_(\d+)_repair$")


def _parse_frame_id(path: Path) -> int:
    match = _FRAME_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse frame id from {path.name}")
    return int(match.group(1))


def collect_repair_mesh_paths(
    repair_dir: Path,
    subject_id: str,
    *,
    camera_prefix: str | None = None,
) -> list[Path]:
    repair_dir = Path(repair_dir)
    if not repair_dir.is_dir():
        raise FileNotFoundError(f"repair dir not found: {repair_dir}")

    prefix = camera_prefix or SUBJECT_CONFIGS.get(subject_id, f"{subject_id}_cam_222200037")
    paths = sorted(
        repair_dir.glob(f"{prefix}_*_repair.obj"),
        key=_parse_frame_id,
    )
    if not paths:
        raise FileNotFoundError(
            f"No repair meshes matched prefix {prefix!r} under {repair_dir}"
        )
    return paths


def load_mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(path, process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular mesh in {path}")
    return verts, faces


class RepairSequenceViewer:
    def __init__(
        self,
        repair_dir: Path,
        subject_ids: list[str],
        *,
        default_subject: str,
    ) -> None:
        self.repair_dir = Path(repair_dir)
        self.subject_ids = subject_ids
        self.sequences: dict[str, list[Path]] = {}
        for subject_id in subject_ids:
            self.sequences[subject_id] = collect_repair_mesh_paths(
                self.repair_dir,
                subject_id,
            )

        self.subject_id = default_subject if default_subject in self.sequences else subject_ids[0]
        self.frame_idx = 0
        self.play = False
        self.play_fps = 24.0
        self._play_accum = 0.0

        first_path = self.sequences[self.subject_id][0]
        verts, faces = load_mesh_arrays(first_path)
        self._faces = faces
        self._mesh_name = "phack_repair_track"

        ps.init()
        ps.set_program_name("demo_show_phack_track")
        ps.set_ground_plane_mode("none")
        self._ps_mesh = ps.register_surface_mesh(
            self._mesh_name,
            verts,
            faces,
            color=(0.78, 0.72, 0.68),
            edge_width=0.8,
            smooth_shade=True,
            material="clay",
        )
        ps.set_user_callback(self._ui_callback)

    @property
    def current_paths(self) -> list[Path]:
        return self.sequences[self.subject_id]

    @property
    def current_path(self) -> Path:
        return self.current_paths[self.frame_idx]

    def _set_frame(self, frame_idx: int) -> None:
        frame_idx = int(np.clip(frame_idx, 0, len(self.current_paths) - 1))
        if frame_idx == self.frame_idx and hasattr(self, "_loaded_path"):
            if self._loaded_path == self.current_paths[frame_idx]:
                return

        path = self.current_paths[frame_idx]
        verts, faces = load_mesh_arrays(path)
        if faces.shape != self._faces.shape or not np.array_equal(faces, self._faces):
            ps.remove_surface_mesh(self._mesh_name)
            self._faces = faces
            self._ps_mesh = ps.register_surface_mesh(
                self._mesh_name,
                verts,
                faces,
                color=(0.78, 0.72, 0.68),
                edge_width=0.8,
                smooth_shade=True,
                material="clay",
            )
        else:
            self._ps_mesh.update_vertex_positions(verts)

        self.frame_idx = frame_idx
        self._loaded_path = path

    def _set_subject(self, subject_id: str) -> None:
        if subject_id == self.subject_id:
            return
        self.subject_id = subject_id
        self.frame_idx = 0
        self._loaded_path = None
        self._set_frame(0)

    def _ui_callback(self) -> None:
        n_frames = len(self.current_paths)

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("PHACK repair track"):
            psim.Text(f"repair_dir: {self.repair_dir}")
            psim.Separator()

            for sid in self.subject_ids:
                label = f"{sid} ({SUBJECT_CONFIGS.get(sid, sid)})"
                if psim.RadioButton(label, self.subject_id == sid):
                    self._set_subject(sid)

            psim.Separator()
            changed, self.frame_idx = psim.SliderInt(
                "frame",
                self.frame_idx,
                0,
                max(n_frames - 1, 0),
            )
            if changed:
                self._set_frame(self.frame_idx)

            psim.Text(f"file: {self.current_path.name}")
            psim.Text(f"frame_id: {_parse_frame_id(self.current_path)}")
            psim.Text(f"frames: {self.frame_idx + 1}/{n_frames}")

            changed_play, self.play = psim.Checkbox("play", self.play)
            if changed_play:
                self._play_accum = 0.0
            _, self.play_fps = psim.SliderFloat(
                "fps",
                self.play_fps,
                1.0,
                60.0,
            )

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
        for sid, paths in self.sequences.items():
            frame_ids = [_parse_frame_id(p) for p in paths]
            print(
                f"[demo_show_phack_track] subject={sid} "
                f"frames={len(paths)} "
                f"range={frame_ids[0]}..{frame_ids[-1]}",
                flush=True,
            )
        print(f"[demo_show_phack_track] default_subject={self.subject_id}", flush=True)
        ps.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show phack_repair mesh sequences in polyscope.",
    )
    parser.add_argument(
        "--repair-dir",
        type=Path,
        default=DEFAULT_REPAIR_DIR,
        help="directory containing *_repair.obj meshes",
    )
    parser.add_argument(
        "--subject-ids",
        nargs="+",
        default=list(SUBJECT_CONFIGS.keys()),
        choices=list(SUBJECT_CONFIGS.keys()),
        help="subjects to load; switch between them in the UI",
    )
    parser.add_argument(
        "--default-subject",
        # default="224",
        default="473",
        choices=list(SUBJECT_CONFIGS.keys()),
        help="subject shown first when the viewer opens",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    viewer = RepairSequenceViewer(
        args.repair_dir,
        args.subject_ids,
        default_subject=args.default_subject,
    )
    viewer.show()


if __name__ == "__main__":
    main()
