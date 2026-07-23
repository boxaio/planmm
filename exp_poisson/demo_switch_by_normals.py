"""将源网格按目标网格法线做 dARAP 形变，并用 Polyscope 并排显示源 / 目标 / 形变结果。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polyscope as ps
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

REPAIR_DIR = Path(__file__).resolve().parent / "phack_nersemble" / "phack_repair"
SRC_MESH_PATH = REPAIR_DIR / "224_cam_222200037_000_repair.obj"
TGT_MESH_PATH = REPAIR_DIR / "224_cam_222200037_040_repair.obj"
# TGT_MESH_PATH = REPAIR_DIR / "224_cam_222200037_088_repair.obj"

LAMBDA_DEFAULT = 3.0
LAMBDA_DEFORM = 9.5


def _to_numpy_xyz(
    verts: np.ndarray | torch.Tensor,
    faces: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def show_mesh_triple(
    verts_list: list[np.ndarray | torch.Tensor],
    faces_list: list[np.ndarray | torch.Tensor],
    *,
    show_edge: bool = False,
    smooth_shade: bool = False,
    gap: float | None = None,
    colors: tuple[tuple[float, float, float], ...] = (
        (0.72, 0.72, 0.78),
        (0.78, 0.72, 0.68),
        (0.68, 0.78, 0.72),
    ),
    names: tuple[str, str, str] = ("source", "target", "deformed"),
) -> None:
    """沿 +x 并排显示三个三角网格。"""
    assert len(verts_list) == 3 and len(faces_list) == 3

    arrays = [_to_numpy_xyz(v, f) for v, f in zip(verts_list, faces_list)]
    shifted: list[np.ndarray] = []
    offset_x = 0.0

    all_min = np.minimum.reduce([v.min(axis=0) for v, _ in arrays])
    all_max = np.maximum.reduce([v.max(axis=0) for v, _ in arrays])
    combined_diag = float(np.linalg.norm(all_max - all_min))
    if gap is None:
        gap = float(max(combined_diag * 0.05, 1e-6))

    for verts, _ in arrays:
        v = verts.copy()
        v[:, 0] += offset_x
        shifted.append(v)
        offset_x = float(v[:, 0].max()) + gap

    ps.init()
    for idx, ((_, faces), verts, name) in enumerate(zip(arrays, shifted, names)):
        ps.register_surface_mesh(
            name,
            vertices=verts,
            faces=faces,
            enabled=True,
            color=colors[idx],
            edge_width=0.9 if show_edge else 0,
            material="clay",
            smooth_shade=smooth_shade,
            back_face_policy="custom",
        )
    ps.show()


def deform_source_to_target_normals(
    src_verts: np.ndarray,
    faces: np.ndarray,
    tgt_normals: np.ndarray,
) -> np.ndarray:
    verts_list = [torch.tensor(src_verts, dtype=torch.float32)]
    faces_list = [torch.tensor(faces, dtype=torch.long)]
    target_normals_list = [torch.tensor(tgt_normals, dtype=torch.float32)]

    y = src_verts[:, 1]
    upper_half_vertex_indices = np.where(y >= np.median(y))[0].tolist()
    local_step_procrustes_lambda = vertex_procrustes_lambda_from_regions(
        len(src_verts),
        default_lambda=LAMBDA_DEFAULT,
        region_vertex_indices=[upper_half_vertex_indices],
        region_lambdas=[LAMBDA_DEFORM],
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


def main() -> None:
    src_mesh = read_obj(str(SRC_MESH_PATH))
    tgt_mesh = read_obj(str(TGT_MESH_PATH))

    src_verts = np.asarray(src_mesh.vs, dtype=np.float64)
    tgt_verts = np.asarray(tgt_mesh.vs, dtype=np.float64)
    faces = np.asarray(src_mesh.fvs, dtype=np.int64)

    if src_verts.shape != tgt_verts.shape:
        raise ValueError(
            f"源/目标顶点数不一致: {src_verts.shape[0]} vs {tgt_verts.shape[0]}"
        )

    tgt_normals = vertex_normals(
        torch.tensor(tgt_verts[None], dtype=torch.float32),
        torch.tensor(faces[None], dtype=torch.long),
    )[0].numpy()

    deformed_verts = deform_source_to_target_normals(src_verts, faces, tgt_normals)
    max_disp = float(np.max(np.linalg.norm(deformed_verts - src_verts, axis=1)))
    print(f"source: {SRC_MESH_PATH.name}")
    print(f"target: {TGT_MESH_PATH.name}")
    print(f"max vertex displacement: {max_disp:.6f}")

    show_mesh_triple(
        [src_verts, tgt_verts, deformed_verts],
        [faces, faces, faces],
        show_edge=False,
        smooth_shade=False,
        names=("source", "target", "deformed"),
    )


if __name__ == "__main__":
    main()
