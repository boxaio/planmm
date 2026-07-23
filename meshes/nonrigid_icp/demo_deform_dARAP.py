import numpy as np
import os
import sys
import torch
import trimesh
import polyscope as ps
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from utils.mesh import read_obj, extract_submesh, vertex_normals
from utils.ps_tools import show_mesh, show_mesh_pair
from meshes.nonrigid_icp.deformations_MINIMAL import *




# _id = '000063_52'
_id = '100881_479'
# _id = '101001_73'
# _id = '101053_136'
# _id = '101067_79'
# _id = '101292_284'
# _id = '101375_65'
# _id = '101198_391'
# _id = 'Extropic_CTO'
# _id = 'BenjaminHoover'
# _id = 'LeiJun'
# _id = 'LennartEing'
# _id = 'Huang'
# _id = 'YaoShunyu'
# _id = 'Zheng'

hack_mesh_path = f'/media/ubuntu/SSD/hack-3dgs/test/phack_raw/{_id}_phack_raw.obj'
mesh = read_obj(hack_mesh_path)
verts = np.array(mesh.vs)
faces = np.array(mesh.fvs)

# tgt_normals_path = f'/media/ubuntu/SSD/hack-3dgs/test/phack_raw/{_id}_phack_raw_target_normals.npz'
# tgt_normals = np.load(tgt_normals_path)['normals']

tgt_mesh_path = f'/media/ubuntu/SSD/hack-3dgs/test_nicp/{_id}_nicp.obj'
tgt_mesh = read_obj(tgt_mesh_path)
tgt_verts = np.array(tgt_mesh.vs)
tgt_faces = np.array(tgt_mesh.fvs)
tgt_normals = vertex_normals(torch.tensor(tgt_verts[None]), torch.tensor(tgt_faces[None]))[0]

# tgt_normals_path = f'/media/ubuntu/SSD/hack-3dgs/test_nicp/{_id}_nicp_vertex_normals.npz'
# tgt_normals = np.load(tgt_normals_path)['normals']

# L, cholespy_solver, maybe_removed_first_L_column = calc_cot_laplacian_and_cholespy_solver_until_it_works(
#     torch.tensor(verts, dtype=torch.float32),
#     torch.tensor(faces, dtype=torch.long),
#     # pin_first_vertex=True,
#     pin_first_vertex=False,
# )
# print(L)


verts_list = [torch.tensor(verts, dtype=torch.float32)]
faces_list = [torch.tensor(faces, dtype=torch.long)]
target_normals_list = [torch.tensor(tgt_normals, dtype=torch.float32)]

# Per-region Procrustes lambda (higher => more deformation toward target normals).
# Replace region_vertex_indices with your own vertex lists (e.g. from extract_submesh).
LAMBDA_DEFAULT = 3.0
LAMBDA_DEFORM = 6.5
y = verts[:, 1]
upper_half_vertex_indices = np.where(y >= np.median(y))[0].tolist()
local_step_procrustes_lambda = vertex_procrustes_lambda_from_regions(
    len(verts),
    default_lambda=LAMBDA_DEFAULT,
    region_vertex_indices=[upper_half_vertex_indices],
    region_lambdas=[LAMBDA_DEFORM],
)

# prepare the solver
solvers = SparseLaplaciansSolvers.from_meshes(
  verts_list, faces_list,
  pin_first_vertex=True,
  compute_poisson_rhs_lefts=False,
  compute_igl_arap_rhs_lefts=None,
)

# find per-vertex matrices with local step
procrustes_precompute = ProcrustesPrecompute.from_meshes(
    local_step_procrustes_lambda=local_step_procrustes_lambda,
    arap_energy_type="spokes_and_rims_mine", 
    laplacians_solvers=solvers, 
    verts_list=verts_list, 
    faces_list=faces_list
)
per_vertex_3x3matrices_packed = calc_rot_matrices_with_procrustes(
    procrustes_precompute, 
    torch.cat(verts_list), 
    torch.cat(target_normals_list)
)
# (assuming target_normals_list is a list of tensors each (n_verts, 3), each tensor is the target per-vertex normals of a single mesh in the batch)

per_vertex_3x3matrices_list = torch.split(per_vertex_3x3matrices_packed, [len(v) for v in verts_list])

# solve for deformations given per-vertex 3x3 matrices
deformed_verts_list = calc_ARAP_global_solve(
    verts_list, faces_list, solvers, per_vertex_3x3matrices_list,
    arap_energy_type="spokes_and_rims_mine",
    postprocess="recenter_only"
)

deformed_verts = deformed_verts_list[0]



show_mesh_pair(
    verts, faces,
    deformed_verts, faces,
    show_edge=False,
    smooth_shade=False,
    names=('raw', 'deformed'),
)
