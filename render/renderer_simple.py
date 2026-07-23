import numpy as np
import torch
import torch.nn.functional as F
import kornia
import trimesh
import cv2
from kornia.geometry.camera import pixel2cam
from typing import List
import nvdiffrast.torch as dr
from scipy.io import loadmat
from torch import nn
import matplotlib.pyplot as plt
import polyscope as ps
from typing import Tuple, Literal, Optional, Union
import torch.nn.functional as tfunc


specified_light = torch.tensor([
    # [ 2.7162,  2.6870,  2.7218],
    [ 1.7162,  1.6870,  1.7218],
    [-0.0768, -0.0500, -0.0616],
    [ 0.0215,  0.0332,  0.0437],
    [-0.0145, -0.0119, -0.0158],
    [-0.0195,  0.0284,  0.0110],
    [-0.2319, -0.2813, -0.2629],
    [ 0.1273,  0.1927,  0.2105],
    [ 0.2398,  0.1616,  0.0848],
    [ 0.5114,  0.4712,  0.5472]],
)



def dot(x: torch.Tensor, y: torch.Tensor):
    return torch.sum(x*y, -1, keepdim=True)

def safe_normalize(x: torch.Tensor, eps: float =1e-20):
    # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
    return x / torch.sqrt(torch.clamp(dot(x, x), min=eps))

def compute_v_normals(verts: torch.Tensor, faces: torch.Tensor)->torch.Tensor:
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    v_normals = torch.zeros_like(verts)
    N = verts.shape[0]
    v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

    v_normals = torch.where(dot(v_normals, v_normals) > 1e-20, 
                            v_normals, 
                            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals


def _translation(x, y, z, device):
    return torch.tensor([[1., 0, 0, x],
                         [0, 1, 0, y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]], device=device) 

def _projection(r, device, l=None, t=None, b=None, near=1.7, far=20.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = torch.zeros([4, 4], device=device)
    p[0,0] = 2 * near / (r - l)
    p[0,2] = (r + l) / (r - l)
    p[1,1] = 2 * near / (t - b) * (-1 if flip_y else 1)
    p[1,2] = (t + b) / (t - b)
    p[2,2] = -(far + near) / (far - near)
    p[2,3] = -2 * far * near / (far - near)
    p[3,2] = -1
    return p    # (4, 4)


def calc_face_normals(vertices:torch.Tensor, faces:torch.Tensor, normalize:bool=False)->torch.Tensor: 
    '''
    vertices: torch.Tensor, (V, 3), first vertex may be unreferenced
    faces: torch.Tensor, (F, 3) long, first face may be all zero
         n
         |
         c0     corners ordered counterclockwise when
        / \     looking onto surface (in neg normal direction)
      c1---c2
    '''
    full_vertices = vertices[faces]   # (F, 3, 3)
    v0, v1, v2 = full_vertices.unbind(dim=1)   # (F, 3)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)   # (F, 3)
    if normalize:
        face_normals = tfunc.normalize(face_normals, eps=1e-6, dim=1) #TODO inplace?
    return face_normals   # (F, 3)


def calc_vertex_normals(vertices:torch.Tensor, faces:torch.Tensor, face_normals:torch.Tensor=None)->torch.Tensor:
    '''
    vertices: torch.Tensor, (V, 3), first vertex may be unreferenced
    faces: torch.Tensor, (F, 3) long, first face may be all zero
    face_normals: torch.Tensor, (F, 3) not normalized
    returns vertex_normals: torch.Tensor, (V, 3)
    '''
    F = faces.shape[0]
    if face_normals is None:
        face_normals = calc_face_normals(vertices, faces)  # (F, 3)
        
    vertex_normals = torch.zeros((vertices.shape[0],3,3), dtype=vertices.dtype, device=vertices.device)   # (V, 3, 3)
    vertex_normals.scatter_add_(dim=0, index=faces[:,:,None].expand(F,3,3), src=face_normals[:,None,:].expand(F,3,3))
    vertex_normals = vertex_normals.sum(dim=1)   # (V, 3)
    return tfunc.normalize(vertex_normals, eps=1e-6, dim=1)


def get_SH_shading(normals, sh_coefficients, sh_const):
    """
    :param normals: shape [B, H, W, K, 3]
    :param sh_coefficients: shape [B, 9, 3]
    :return:
    """
    N = normals
    # compute sh basis function values of shape [N, H, W, K, 9]
    sh = torch.stack([
            N[..., 0] * 0.0 + 1.0,
            N[..., 0],
            N[..., 1],
            N[..., 2],
            N[..., 0] * N[..., 1],
            N[..., 0] * N[..., 2],
            N[..., 1] * N[..., 2],
            N[..., 0] ** 2 - N[..., 1]**2,
            3 * (N[..., 2]**2) - 1,
    ], dim=-1)
    sh = sh * sh_const[None, None, None, :].to(sh.device)

    # shape [N, H, W, K, 9, 1]
    sh = sh[..., None]

    # shape [N, H, W, K, 9, 3]
    sh_coefficients = sh_coefficients[:, None, None, :, :]

    # shape after linear combination [N, H, W, K, 3]
    shading = torch.sum(sh_coefficients * sh, dim=3)
    return shading


def _warmup(glctx):
    # windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device='cuda', **kwargs)
    pos = tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]], dtype=torch.float32)
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx=glctx, pos=pos, tri=tri, resolution=[256, 256])


class MyRenderer(nn.Module):
    _glctx: dr.RasterizeGLContext = None
    def __init__(self, 
        mv: torch.Tensor,  # (C,4,4)
        proj: torch.Tensor,  # (4,4)
        image_size: tuple[int,int],
        device='cuda',
    ):
        super().__init__()

        self._mvp = proj @ mv  # (C,4,4)
        self._image_size = image_size
        self._glctx = dr.RasterizeGLContext(device=device)
        # self._glctx = dr.RasterizeCUDAContext(device=device)
        _warmup(self._glctx)

        # constant factor of first three bands of spherical harmonics
        pi = np.pi
        sh_const = torch.tensor([
            1 / np.sqrt(4 * pi),
            ((2 * pi) / 3) * np.sqrt(3 / (4 * pi)),
            ((2 * pi) / 3) * np.sqrt(3 / (4 * pi)),
            ((2 * pi) / 3) * np.sqrt(3 / (4 * pi)),
            (pi / 4) * (3) * np.sqrt(5 / (12 * pi)),
            (pi / 4) * (3) * np.sqrt(5 / (12 * pi)),
            (pi / 4) * (3) * np.sqrt(5 / (12 * pi)),
            (pi / 4) * (3 / 2) * np.sqrt(5 / (12 * pi)),
            (pi / 4) * (1 / 2) * np.sqrt(5 / (4 * pi)),
            ], dtype=torch.float32,
        )
        self.register_buffer("sh_const", sh_const, persistent=False)
        self.lighting_type = 'SH'

    def render_normals(self, vertices: torch.Tensor, normals: torch.Tensor, faces: torch.Tensor) ->torch.Tensor:
        '''
        vertices: (B, V, 3)
        normals: (B, V, 3)
        faces: (F, 3)
        return: (C, H, W, 4)
        '''

        B, V, _ = vertices.shape
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(B, V, 1, device=vertices.device)), axis=-1) # (B, V, 3) -> (B, V, 4)
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) # (B, V, 4)
        rast_out,_ = dr.rasterize(glctx=self._glctx, pos=vertices_clip, tri=faces, resolution=self._image_size, grad_db=False) # (B, H, W, 4)
        vert_col = (normals + 1) / 2     # (B, V, 3)
        col, _ = dr.interpolate(attr=vert_col, rast=rast_out, tri=faces)   # (B, H, W, 3)
        alpha = torch.clamp(rast_out[..., -1:], max=1)   # (B, H, W, 1)
        col = torch.concat((col, alpha), dim=-1)   # (B, H, W, 4)
        col = dr.antialias(color=col, rast=rast_out, pos=vertices_clip, tri=faces)   # (B, H, W, 4)
        return col   # (B, H, W, 4)

    def render_shape(self, vertices: torch.Tensor, faces: torch.Tensor) ->torch.Tensor:
        '''
        vertices: (B, V, 3)
        faces: (F, 3)
        return: (B, H, W, 4)
        '''

        B, V, _ = vertices.shape
        faces = faces.type(torch.int32)
        vert_hom = torch.cat((vertices, torch.ones(B, V, 1, device=vertices.device)), axis=-1) # (B, V, 3) -> (B, V, 4)
        vertices_clip = vert_hom @ self._mvp.transpose(-2,-1) # (B, V, 4)
        rast_out,_ = dr.rasterize(glctx=self._glctx, pos=vertices_clip, tri=faces, resolution=self._image_size, grad_db=False) # (B, H, W, 4)
        
        v_normal = compute_v_normals(vertices, faces)
        normals, _ = dr.interpolate(v_normal, rast_out, faces)
        normals = safe_normalize(normals)  # [B, H, W, 3]

        shading_images = self.shade(normals, specified_light[None].to(vertices.device))
        shading_images = shading_images.permute(0, 3, 1, 2)  # [B, 3, H, W]

        return shading_images   
    
    def shade(self, normals, lighting_coeff=None):
        if self.lighting_type == 'constant':
            shaded_images = torch.ones_like(normals[..., :3])
        elif self.lighting_type == 'front':
            # diffuse = torch.clamp(V.dot(normal, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')), 0.0, 1.0)
            shaded_images = dot(normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
            mask_backface = shaded_images < 0
            shaded_images[mask_backface] = shaded_images[mask_backface].abs()*0.3
        elif self.lighting_type == 'front-range':
            bias = 0.75
            shaded_images = torch.clamp(
                dot(normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')) + bias, 
                0.0, 1.0,
            )
        elif self.lighting_type == 'SH':
            shaded_images = get_SH_shading(normals, lighting_coeff, self.sh_const)
        else:
            raise NotImplementedError(f"Unknown lighting type: {self.lighting_type}")
        return shaded_images
    