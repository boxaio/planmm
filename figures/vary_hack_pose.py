import numpy as np
import torch
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import polyscope as ps
from meshes.hackhead import HACKHead
from utils.mesh import write_mesh_obj



device = 'cuda'

hack_params_path = '/media/ubuntu/xb/ImHead_dataset/HACK_fit/00006.000001_neutral_parms.npz'
hack_params = np.load(hack_params_path)
shape_params = hack_params['shape_params']
expression_params = hack_params['expression_params']

neck_poses = torch.zeros((1,8,3)).to(device)

# neck turn right
# neck_poses[0, 1, 1] = -0.4

# neck up
neck_poses[0, 1, 0] = -0.4
neck_poses[0, 1, 1] = 0.37


hackhead = HACKHead(use_teeth=False).to(device)
results = hackhead(
    shape=torch.tensor(shape_params)[None].float().to(device), 
    expression=torch.tensor(expression_params)[None].float().to(device), 
    neck_pose=torch.tensor(neck_poses).to(device), 
    tau=torch.zeros((1,1)).to(device), 
    gamma=torch.zeros((1,1)).to(device),
    return_lmks_mp=False,
)

verts = results[0].detach().cpu().squeeze().numpy()
faces = hackhead.temp_mesh.fvs

mesh_info = {
    'v': verts,
    'vt': hackhead.temp_mesh.vts,
    'fv': faces,
    'fvt': hackhead.temp_mesh.fvts,
}

# write_mesh_obj(mesh_info, 'figures/HACK_neck_turn_right.obj')
write_mesh_obj(mesh_info, 'figures/HACK_neck_up.obj')


ps.init()
ps.register_surface_mesh(
    "hackhead", 
    vertices=verts, 
    faces=faces,
    color=(0.75, 0.75, 0.75),
    edge_width=0.9,
    material='clay',
    back_face_policy='custom',
)
ps.show()









