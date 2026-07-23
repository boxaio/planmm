import numpy as np
import os
import trimesh
import torch
import igl
import pickle

from copy import deepcopy
from numpy.linalg import eigvals
from scipy.spatial.transform import Rotation
from pytorch3d import transforms
from torch_sparse import coalesce, transpose
from cupyx.scipy.sparse.linalg import SuperLU
from torch.utils.dlpack import from_dlpack

from utils.deformation_transfer import Mesh, Transfer
from utils.mesh import calc_center, calc_norm

from external.diffusion_net.src import diffusion_net



def get_mesh_operators(mesh):
    N_FACE = mesh.faces.shape[0]
    N_VERTEX = mesh.vertices.shape[0]
    transf = Transfer(mesh, deepcopy(mesh))
    lu_solver = SuperLU(transf.lu)
    idxs, vals = coalesce(
        from_dlpack(transf.idxs.toDlpack()).long(), 
        from_dlpack(transf.vals.toDlpack()), 
        m=N_FACE *3, n=N_VERTEX,
    )
    idxs, vals = transpose(idxs, vals, m=N_FACE *3, n=N_VERTEX)
    rhs = transf.cupy_A.T
    return lu_solver, idxs, vals, rhs



class Normalizer(object):
    def __init__(self, std_path, device, zero_mean=True):
        if zero_mean:
            self.gradients_std = np.load(os.path.join(std_path, 'gradients_std.npy'))
            self.gradients_std = torch.from_numpy(self.gradients_std).to(device).float()
            self.gradients_mean = torch.eye(3).view(-1,).unsqueeze(0).unsqueeze(0).to(device).float()
        else:
            self.gradients_mean, self.gradients_std = np.load(
                os.path.join(std_path, 'wks_mean_std.npz'), 
                allow_pickle=True,
            )['arr_0'].item().values()
            self.gradients_std = torch.from_numpy(self.gradients_std).to(device).float()
            self.gradients_mean = torch.from_numpy(self.gradients_mean).to(device).float()

    def normalize(self, tensor):
        return (tensor - self.gradients_mean.to(tensor.device)) / self.gradients_std.to(tensor.device)

    def inv_normalize(self, tensor):
        return tensor * self.gradients_std.to(tensor.device) + self.gradients_mean.to(tensor.device)
    

class Normalizer_img(Normalizer):
    def __init__(self, std_path, device):
        import warnings
        warnings.filterwarnings('ignore')
        with open(os.path.join(std_path, 'img_stat.pkl'), 'rb') as f:
            self.gradients_mean, self.gradients_std = pickle.load(f).values()
        self.gradients_std = torch.tensor(self.gradients_std).to(device).float()
        self.gradients_mean = torch.tensor(self.gradients_mean).to(device).float()



def calc_jacobian_stat(jacobians):
    """
    Args:
        jacobians: (N, 3, 3) N matrices of the jacobians
    
        # Using the Polar Decomposition J = UP
         - U: Unitary, rotation
         - P: Positive semi-definite Hermitian (Symmetric) Matrix
    Return:
        theta: Rotation angle
        omega: Rotation axis
        A: scaling factor
        
    """
    W, S, V = np.linalg.svd(jacobians, full_matrices=True)
    P = (V.transpose(0, 2, 1) * S[:, np.newaxis]) @ V # Scaling matrix
    U = W @ V # Rotation matrix
    A = eigvals(P)
    rotvec = Rotation.from_matrix(U).as_rotvec()
    theta = np.linalg.norm(rotvec, axis=-1)
    omega = rotvec / (theta[:, np.newaxis] + 1e-5)

    return theta, omega, A

def reconstruct_jacobians(inputs, repr='matrix', eps=1e-9):
    # inputs: (BS, N, ?) inputs for reconstruct the jacobians
    # ? = 9, repr == 'matrix'
    # ? = 12, repr == '6dof'
    # ? = 10, repr == 'quat'
    # ? = 9,  repr == 'expmap'
    BS = inputs.shape[0]
    N = inputs.shape[1]
    if repr == 'matrix':
        return inputs.reshape(BS, N, 3, 3)
        
    rots = inputs[:, :, :-6]
    scals = inputs[:, :, -6:]
    # rots = rots.reshape(BS, N, 2, 3)
    rot_matrix = torch.empty(BS, N, 3, 3)

    if repr == '6dof':
        rot_matrix = transforms.rotation_6d_to_matrix(rots)
    elif repr == 'quat':
        rot_matrix = transforms.quaternion_to_matrix(rots)
    elif repr == 'expmap':
        rot_matrix = transforms.axis_angle_to_matrix(rots)

    scal_matrix = torch.empty((BS, N, 9)).to(inputs.device)
    scal_matrix[:, :, [0, 1, 2, 4, 5, 8]] = scals
    scal_matrix[:, :, [3, 6, 7]] = scals[:, :, [1, 2, 4]]
    scal_matrix = scal_matrix.reshape(BS, N, 3, 3)

    return torch.matmul(rot_matrix, scal_matrix)


def calc_new_mesh(
    args, normalizer, model, myfunc, mesh, z, operators, dfn_info, img=None,
):
    """
    z: latent code
    
    """
    lu_solver, idxs, vals, rhs = operators
    # calc center of model
    cents = calc_center(mesh)
    cents = torch.from_numpy(cents).float().unsqueeze(0)

    # normals
    _, norms = calc_norm(mesh)
    norms = torch.from_numpy(norms).float().unsqueeze(0)

    # set inputs
    inputs = torch.cat([cents, norms], dim=-1)
    inputs = inputs.to('cuda')
    norms_v = torch.from_numpy(igl.per_vertex_normals(mesh.vertices, mesh.faces)).float()

    if torch.isnan(norms_v).any():
        # If something wrong with the igl computation
        norms_v, _ = calc_norm(mesh)
        norms_v = torch.from_numpy(norms_v).float()
    # set source vertex
    input_source_v = torch.cat([torch.from_numpy(mesh.vertices), norms_v], dim=-1).float().unsqueeze(0)
    input_source_v = input_source_v.to('cuda')

    # get image feture
    img_feat = model.img_feat(img) 

    # set inputs
    inputs = torch.cat([inputs, img_feat.unsqueeze(1).expand(-1, inputs.shape[1], -1)], dim=-1)

    # if not args.img_only_mlp:
    input_source_v = torch.cat([input_source_v, img_feat.unsqueeze(1).expand(-1, input_source_v.shape[1], -1)], dim=-1)

    with torch.no_grad():
        model.update_precomputes(dfn_info)
        
        if z.shape[0] > 1:
            z = z.reshape(-1, z.shape[-1]) 
            inputs = inputs.repeat(z.shape[0], 1, 1)
            input_source_v = input_source_v.repeat(z.shape[0], 1, 1)
        
        # decode to get mesh
        g_pred, z_iden = model.decode([inputs.float(), input_source_v.float()], z.float())

        # solve for the mesh
        g_pred  = normalizer.inv_normalize(g_pred)
        g_pred  = reconstruct_jacobians(g_pred, repr='matrix')
        out_pred = myfunc(g_pred, lu_solver, idxs, vals, rhs.shape)
        out_pred = out_pred - out_pred.mean(axis=[0, 1], keepdim=True)

        # get the mesh
        mesh_out = Mesh(out_pred[0].detach().cpu().numpy(), mesh.faces)
        theta, omega, A = calc_jacobian_stat(g_pred[0].detach().cpu().numpy())

    if z_iden is not None:
        z_iden = z_iden.detach().cpu().numpy()
    return mesh_out, theta, omega, A, z_iden


###### Reference for codes below: https://github.com/dafei-qin/NFR_pytorch/blob/master/myutils.py
def get_dfn_info(mesh, cache_dir=None, device='cuda'):
    """
    Args:
        mesh (trimesh.Trimesh)
    """
    verts_list = torch.from_numpy(mesh.vertices).unsqueeze(0).float()
    face_list = torch.from_numpy(mesh.faces).unsqueeze(0).long()
    frames_list, mass_list, L_list, evals_list, evecs_list, gradX_list, gradY_list = \
        diffusion_net.geometry.get_all_operators(
            verts_list, face_list, k_eig=128, op_cache_dir=cache_dir,
        )

    dfn_info = [
        mass_list[0], L_list[0], evals_list[0], evecs_list[0], 
        gradX_list[0], gradY_list[0], torch.from_numpy(mesh.faces),
    ]
    dfn_info = [_.to(device).float() if type(_) is not torch.Size else _  for _ in dfn_info]
    return dfn_info