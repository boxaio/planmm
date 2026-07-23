"""
Batch-refine LAMM generated_global OBJ meshes with regional dARAP normal deformation.

Uses the same deformations_MINIMAL pipeline as meshes/nonrigid_icp/example.py and
demo_deform_dARAP.py: per-vertex Procrustes rotations toward target normals, then
ARAP global solve to update vertex positions while preserving the normal field.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NICP_DIR = _REPO_ROOT / "meshes" / "nonrigid_icp"
for _path in (str(_REPO_ROOT), str(_NICP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.mesh import read_obj
from meshes.nonrigid_icp.geometry import vertex_normals
from deformations_MINIMAL import (
    ProcrustesPrecompute,
    SparseLaplaciansSolvers,
    calc_ARAP_global_solve,
    calc_rot_matrices_with_procrustes,
    vertex_procrustes_lambda_from_regions,
)

DEFAULT_INPUT_DIR = _REPO_ROOT / "results" / "lamm_ae_bfm" / "generated_global"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results" / "lamm_ae_bfm" / "generated_global_refine"
DEFAULT_REGION_INFO = _REPO_ROOT / "dataset" / "bfm_region_info.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine generated_global OBJ meshes with regional dARAP normal deformation.",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing source OBJ meshes.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write refined OBJ meshes.",
    )
    parser.add_argument(
        "--region_info",
        type=Path,
        default=DEFAULT_REGION_INFO,
        help="BFM region_info pickle used for per-region Procrustes lambda.",
    )
    parser.add_argument(
        "--lambda_default",
        type=float,
        default=3.0,
        help="Default Procrustes lambda (example.py: regional_darap_lambda_default).",
    )
    parser.add_argument(
        "--lambda_face_interior",
        type=float,
        default=6.5,
        help="Lambda on face interior vertices.",
    )
    parser.add_argument(
        "--lambda_forehead",
        type=float,
        default=1.0,
        help="Lambda on forehead vertices.",
    )
    parser.add_argument(
        "--lambda_frown",
        type=float,
        default=8.5,
        help="Lambda on frown vertices.",
    )
    parser.add_argument(
        "--lambda_eye",
        type=float,
        default=1.0,
        help="Lambda on eye region vertices.",
    )
    parser.add_argument(
        "--pin_first_vertex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin the first vertex during the Laplacian solve.",
    )
    parser.add_argument(
        "--arap_energy_type",
        type=str,
        default="spokes_and_rims_mine",
        choices=("spokes_mine", "spokes_and_rims_mine", "spokes_igl", "spokes_and_rims_igl"),
    )
    parser.add_argument(
        "--postprocess",
        type=str,
        default="recenter_only",
        choices=("recenter_only", "recenter_rescale"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for dARAP solve (cpu recommended).",
    )
    return parser.parse_args()


def _region_vids(region_info: dict, *names: str) -> np.ndarray:
    out: list[np.ndarray] = []
    for name in names:
        if name not in region_info:
            continue
        out.append(np.asarray(region_info[name]["vids"], dtype=np.int64).reshape(-1))
    if not out:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(out))


def build_bfm_regional_lambda(
    n_vertices: int,
    region_info: dict,
    *,
    lambda_default: float,
    lambda_face_interior: float,
    lambda_forehead: float,
    lambda_frown: float,
    lambda_eye: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Regional Procrustes lambda aligned with example.py defaults, using BFM region masks."""
    region_vertex_indices: list[list[int]] = []
    region_lambdas: list[float] = []

    face_indices = _region_vids(region_info, "face_0", "face_1")
    if face_indices.size > 0:
        region_vertex_indices.append(face_indices.tolist())
        region_lambdas.append(float(lambda_face_interior))

    forehead_indices = _region_vids(region_info, "forehead")
    if forehead_indices.size > 0:
        region_vertex_indices.append(forehead_indices.tolist())
        region_lambdas.append(float(lambda_forehead))

    eye_indices = _region_vids(region_info, "eye_0", "eye_1")
    if eye_indices.size > 0:
        region_vertex_indices.append(eye_indices.tolist())
        region_lambdas.append(float(lambda_eye))

    frown_indices = _region_vids(region_info, "frown")
    if frown_indices.size > 0:
        region_vertex_indices.append(frown_indices.tolist())
        region_lambdas.append(float(lambda_frown))

    return vertex_procrustes_lambda_from_regions(
        n_vertices,
        default_lambda=lambda_default,
        region_vertex_indices=region_vertex_indices,
        region_lambdas=region_lambdas,
        device=device,
    )


def darap_normal_deform(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_normals: np.ndarray,
    local_lambda: torch.Tensor,
    *,
    pin_first_vertex: bool,
    arap_energy_type: str,
    postprocess: str,
    device: torch.device,
) -> np.ndarray:
    """Regional dARAP normal deformation (same core steps as example.apply_regional_darap_normal_deform)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    target_normals = np.asarray(target_normals, dtype=np.float64)
    if vertices.shape[0] != target_normals.shape[0]:
        raise ValueError(
            f"vertices ({vertices.shape[0]}) and target_normals ({target_normals.shape[0]}) "
            "must have the same length"
        )

    verts_t = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces_t = torch.tensor(faces, dtype=torch.long, device=device)
    target_normals_t = torch.tensor(target_normals, dtype=torch.float32, device=device)
    local_lambda = local_lambda.to(device=device, dtype=torch.float32)

    verts_list = [verts_t]
    faces_list = [faces_t]
    target_normals_list = [target_normals_t]

    solvers = SparseLaplaciansSolvers.from_meshes(
        verts_list,
        faces_list,
        pin_first_vertex=pin_first_vertex,
        compute_poisson_rhs_lefts=False,
        compute_igl_arap_rhs_lefts=None,
    )
    procrustes_precompute = ProcrustesPrecompute.from_meshes(
        local_step_procrustes_lambda=local_lambda,
        arap_energy_type=arap_energy_type,
        laplacians_solvers=solvers,
        verts_list=verts_list,
        faces_list=faces_list,
    )
    per_vertex_3x3_packed = calc_rot_matrices_with_procrustes(
        procrustes_precompute,
        torch.cat(verts_list),
        torch.cat(target_normals_list),
    )
    per_vertex_3x3_list = torch.split(
        per_vertex_3x3_packed, [int(v.shape[0]) for v in verts_list]
    )
    deformed_list = calc_ARAP_global_solve(
        verts_list,
        faces_list,
        solvers,
        per_vertex_3x3_list,
        arap_energy_type=arap_energy_type,
        postprocess=postprocess,
    )
    return deformed_list[0].detach().cpu().numpy().astype(np.float64, copy=False)


def save_obj_mesh(vertices: np.ndarray, faces: np.ndarray, output_path: os.PathLike[str]) -> None:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.export(output_path, file_type="obj")


def list_obj_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.obj"), key=lambda p: (len(p.stem), p.stem))


def refine_mesh(
    obj_path: Path,
    faces: np.ndarray,
    n_vertices: int,
    region_info: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    mesh = read_obj(str(obj_path), tri=True)
    vertices = np.asarray(mesh.vs, dtype=np.float64)
    if vertices.shape[0] != n_vertices:
        raise ValueError(
            f"{obj_path.name}: expected {n_vertices} vertices, got {vertices.shape[0]}"
        )
    target_normals = vertex_normals(vertices, faces)

    local_lambda = build_bfm_regional_lambda(
        vertices.shape[0],
        region_info,
        lambda_default=args.lambda_default,
        lambda_face_interior=args.lambda_face_interior,
        lambda_forehead=args.lambda_forehead,
        lambda_frown=args.lambda_frown,
        lambda_eye=args.lambda_eye,
        device=device,
    )
    deformed = darap_normal_deform(
        vertices,
        faces,
        target_normals,
        local_lambda,
        pin_first_vertex=args.pin_first_vertex,
        arap_energy_type=args.arap_energy_type,
        postprocess=args.postprocess,
        device=device,
    )
    max_disp = float(np.max(np.linalg.norm(deformed - vertices, axis=1)))
    return deformed, max_disp


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    with open(args.region_info, "rb") as handle:
        region_info = pickle.load(handle)

    obj_files = list_obj_files(input_dir)
    if not obj_files:
        raise FileNotFoundError(f"No OBJ files found in {input_dir}")

    # All generated_global meshes share BFM template topology; read faces once.
    template_mesh = read_obj(str(obj_files[0]), tri=True)
    faces = np.asarray(template_mesh.fvs, dtype=np.int64)
    n_vertices = int(len(template_mesh.vs))

    device = torch.device(args.device)
    print(f"[smooth_by_normals] input={input_dir}")
    print(f"[smooth_by_normals] output={output_dir}")
    print(f"[smooth_by_normals] meshes={len(obj_files)}  device={device}")

    for obj_path in obj_files:
        deformed, max_disp = refine_mesh(
            obj_path, faces, n_vertices, region_info, args, device
        )
        out_path = output_dir / obj_path.name
        save_obj_mesh(deformed, faces, out_path)
        print(f"  {obj_path.name}: max_disp={max_disp:.6f} -> {out_path.name}")

    print(f"[smooth_by_normals] done, wrote {len(obj_files)} meshes to {output_dir}")


if __name__ == "__main__":
    main()
