"""按 region_info 勾选局部区域，将源网格按目标法线做 dARAP 形变，并排显示。

左→右：source | target | 指定区域按 target 形变后的整体网格。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NICP_DIR = _REPO_ROOT / "meshes" / "nonrigid_icp"
for _path in (str(_REPO_ROOT), str(_NICP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.mesh import read_obj, vertex_normals
from meshes.nonrigid_icp.deformations_MINIMAL import (
    ProcrustesPrecompute,
    SparseLaplaciansSolvers,
    calc_ARAP_global_solve,
    calc_rot_matrices_with_procrustes,
    vertex_procrustes_lambda_from_regions,
)

# REPAIR_DIR = Path(__file__).resolve().parent / "phack_nersemble" / "phack_repair"
# SRC_MESH_PATH = REPAIR_DIR / "224_cam_222200037_000_repair.obj"
# TGT_MESH_PATH = REPAIR_DIR / "224_cam_222200037_040_repair.obj"

ROOT_DIR = Path('/media/ubuntu/xb/ImHead_dataset/HACK_fit_repaired')
SRC_MESH_PATH = ROOT_DIR / "00001.000001_repair.obj"
TGT_MESH_PATH = ROOT_DIR / "00022.000001_repair.obj"

REGION_INFO_FILE = _REPO_ROOT / "dataset" / "hack_region_info.pkl"

REGION_NAME_ORDER = (
    "LeftFace",
    "RightFace",
    "Ears",
    "Neck",
    "Chin",
    "Skull",
    "Forehead",
    "Eyes",
    "Nose",
    "Mouth",
)

LAMBDA_DEFAULT = 3.0
LAMBDA_DEFORM = 8.5

COLORS = (
    (0.72, 0.72, 0.78),
    (0.78, 0.72, 0.68),
    (0.68, 0.78, 0.72),
)
NAMES = ("source", "target", "deformed")


def load_region_vids(region_info_file: Path) -> dict[str, np.ndarray]:
    with open(region_info_file, "rb") as f:
        region_info = pickle.load(f)
    region_info = {k: v for k, v in region_info.items() if "inner" not in k}

    ordered = [n for n in REGION_NAME_ORDER if n in region_info]
    ordered.extend(sorted(set(region_info.keys()) - set(ordered)))

    out: dict[str, np.ndarray] = {}
    for name in ordered:
        vids = np.asarray(region_info[name]["vids"], dtype=np.int64).reshape(-1)
        out[name] = vids
    return out


def deform_source_to_target_normals(
    src_verts: np.ndarray,
    faces: np.ndarray,
    tgt_normals: np.ndarray,
    region_vertex_indices: list[list[int]],
    *,
    default_lambda: float = LAMBDA_DEFAULT,
    region_lambda: float = LAMBDA_DEFORM,
) -> np.ndarray:
    verts_list = [torch.tensor(src_verts, dtype=torch.float32)]
    faces_list = [torch.tensor(faces, dtype=torch.long)]
    target_normals_list = [torch.tensor(tgt_normals, dtype=torch.float32)]

    local_step_procrustes_lambda = vertex_procrustes_lambda_from_regions(
        len(src_verts),
        default_lambda=default_lambda,
        region_vertex_indices=region_vertex_indices,
        region_lambdas=[region_lambda] * len(region_vertex_indices),
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


def _chain_x_offsets(
    verts_list: list[np.ndarray],
    gap: float | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    all_min = np.minimum.reduce([v.min(axis=0) for v in verts_list])
    all_max = np.maximum.reduce([v.max(axis=0) for v in verts_list])
    combined_diag = float(np.linalg.norm(all_max - all_min))
    if gap is None:
        gap = float(max(combined_diag * 0.18, 1e-6))

    shifted: list[np.ndarray] = []
    offsets: list[float] = []
    offset_x = 0.0
    for verts in verts_list:
        offsets.append(offset_x)
        v = verts.copy()
        v[:, 0] += offset_x
        shifted.append(v)
        offset_x = float(v[:, 0].max()) + gap
    return shifted, offsets


class LocalSwitchViewer:
    def __init__(
        self,
        src_verts: np.ndarray,
        tgt_verts: np.ndarray,
        faces: np.ndarray,
        tgt_normals: np.ndarray,
        region_vids: dict[str, np.ndarray],
        *,
        show_edge: bool = False,
        smooth_shade: bool = False,
    ) -> None:
        self.src_verts = np.asarray(src_verts, dtype=np.float64)
        self.tgt_verts = np.asarray(tgt_verts, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.tgt_normals = np.asarray(tgt_normals, dtype=np.float32)
        self.region_vids = region_vids
        self.region_name_list = list(region_vids.keys())

        self.check_boxes = {name: False for name in self.region_name_list}
        if self.region_name_list:
            # 默认勾选 Mouth，便于立刻看到局部效果
            default_name = "Mouth" if "Mouth" in self.check_boxes else self.region_name_list[0]
            self.check_boxes[default_name] = True

        self.status = ""
        self._deformed = self.src_verts.copy()
        self._applied_key: frozenset[str] | None = None

        shifted, self._x_offsets = _chain_x_offsets(
            [self.src_verts, self.tgt_verts, self._deformed]
        )

        ps.init()
        ps.set_program_name("demo_switch_local")
        ps.set_ground_plane_mode("none")
        edge_width = 0.9 if show_edge else 0.0

        self._src_mesh = ps.register_surface_mesh(
            NAMES[0],
            shifted[0],
            self.faces,
            enabled=True,
            color=COLORS[0],
            edge_width=edge_width,
            material="clay",
            smooth_shade=smooth_shade,
            back_face_policy="custom",
        )
        self._tgt_mesh = ps.register_surface_mesh(
            NAMES[1],
            shifted[1],
            self.faces,
            enabled=True,
            color=COLORS[1],
            edge_width=edge_width,
            material="clay",
            smooth_shade=smooth_shade,
            back_face_policy="custom",
        )
        self._def_mesh = ps.register_surface_mesh(
            NAMES[2],
            shifted[2],
            self.faces,
            enabled=True,
            color=COLORS[2],
            edge_width=edge_width,
            material="clay",
            smooth_shade=smooth_shade,
            back_face_policy="custom",
        )

        self._apply_deform()
        ps.set_user_callback(self._ui_callback)

    def _selected_names(self) -> list[str]:
        return [n for n in self.region_name_list if self.check_boxes[n]]

    def _selected_vids(self) -> list[list[int]]:
        return [self.region_vids[n].tolist() for n in self._selected_names()]

    def _selection_key(self) -> frozenset[str]:
        return frozenset(self._selected_names())

    def _shift_x(self, verts: np.ndarray, mesh_idx: int) -> np.ndarray:
        out = verts.copy()
        out[:, 0] += self._x_offsets[mesh_idx]
        return out

    def _update_region_color(self) -> None:
        """在 deformed 网格上高亮当前勾选区域顶点。"""
        n = self.src_verts.shape[0]
        colors = np.tile(np.asarray(COLORS[2], dtype=np.float64), (n, 1))
        highlight = np.array([0.95, 0.35, 0.25], dtype=np.float64)
        for name in self._selected_names():
            vids = self.region_vids[name]
            colors[vids] = highlight
        self._def_mesh.add_color_quantity(
            "selected_region",
            colors,
            defined_on="vertices",
            enabled=True,
        )

    def _apply_deform(self) -> None:
        selected = self._selected_names()
        key = self._selection_key()
        if key == self._applied_key:
            return

        if not selected:
            self._deformed = self.src_verts.copy()
            self.status = "未勾选区域：deformed = source"
        else:
            region_vids = self._selected_vids()
            n_sel = sum(len(v) for v in region_vids)
            self.status = f"形变中… regions={selected} ({n_sel} verts)"
            print(self.status, flush=True)
            self._deformed = deform_source_to_target_normals(
                self.src_verts,
                self.faces,
                self.tgt_normals,
                region_vids,
            )
            max_disp = float(
                np.max(np.linalg.norm(self._deformed - self.src_verts, axis=1))
            )
            self.status = (
                f"done: {', '.join(selected)} | max disp={max_disp:.6f}"
            )
            print(self.status, flush=True)

        self._def_mesh.update_vertex_positions(self._shift_x(self._deformed, 2))
        self._update_region_color()
        self._applied_key = key

    def _ui_callback(self) -> None:
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Local dARAP regions"):
            psim.TextUnformatted("勾选区域后点 Apply，更新右侧 deformed 网格")
            psim.Separator()

            for name in self.region_name_list:
                n_v = int(self.region_vids[name].shape[0])
                label = f"{name} ({n_v})"
                changed, checked = psim.Checkbox(label, self.check_boxes[name])
                if changed:
                    self.check_boxes[name] = checked

            psim.Separator()
            if psim.Button("Select all"):
                for name in self.region_name_list:
                    self.check_boxes[name] = True
            psim.SameLine()
            if psim.Button("Clear all"):
                for name in self.region_name_list:
                    self.check_boxes[name] = False

            psim.Separator()
            dirty = self._selection_key() != self._applied_key
            if dirty:
                psim.TextUnformatted("(selection changed — click Apply)")
            if psim.Button("Apply deform"):
                self._apply_deform()

            if self.status:
                psim.TextUnformatted(self.status)
            psim.TreePop()

    def show(self) -> None:
        print(
            f"source: {SRC_MESH_PATH.name}\n"
            f"target: {TGT_MESH_PATH.name}\n"
            f"regions ({len(self.region_name_list)}): {self.region_name_list}\n"
            f"layout: source | target | deformed (local)",
            flush=True,
        )
        ps.show()


def main() -> None:
    if not REGION_INFO_FILE.is_file():
        raise FileNotFoundError(f"region_info not found: {REGION_INFO_FILE}")

    src_mesh = read_obj(str(SRC_MESH_PATH))
    tgt_mesh = read_obj(str(TGT_MESH_PATH))

    src_verts = np.asarray(src_mesh.vs, dtype=np.float64)
    tgt_verts = np.asarray(tgt_mesh.vs, dtype=np.float64)
    faces = np.asarray(src_mesh.fvs, dtype=np.int64)

    if src_verts.shape != tgt_verts.shape:
        raise ValueError(
            f"源/目标顶点数不一致: {src_verts.shape[0]} vs {tgt_verts.shape[0]}"
        )

    region_vids = load_region_vids(REGION_INFO_FILE)
    if len(region_vids) != 10:
        print(
            f"warning: expected 10 regions, got {len(region_vids)}: "
            f"{list(region_vids.keys())}",
            flush=True,
        )

    max_vid = max(int(v.max()) for v in region_vids.values())
    if max_vid >= src_verts.shape[0]:
        raise ValueError(
            f"region vid {max_vid} out of range for mesh with "
            f"{src_verts.shape[0]} verts"
        )

    tgt_normals = vertex_normals(
        torch.tensor(tgt_verts[None], dtype=torch.float32),
        torch.tensor(faces[None], dtype=torch.long),
    )[0].numpy()

    viewer = LocalSwitchViewer(
        src_verts,
        tgt_verts,
        faces,
        tgt_normals,
        region_vids,
        show_edge=False,
        smooth_shade=False,
    )
    viewer.show()


if __name__ == "__main__":
    main()
