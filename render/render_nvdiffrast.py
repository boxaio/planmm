import numpy as np
import torch
import nvdiffrast.torch as dr
import torch.nn.functional as F
from typing import Tuple, Literal, Optional, Union
from pytorch3d.structures.meshes import Meshes
from skimage.io import imread
import matplotlib.pyplot as plt

from configs.env_paths import HACK_TEMPLATE_TRI_PATH, EYE_MASK
from models.hackhead import HACKMask
import utils
from utils.mesh import dot, safe_normalize, compute_v_normals, compute_face_normals


def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    # print('--------')
    # print(camera[0, 1:].abs())
    # print(X[0].abs().mean(0))

    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    #print(X_trans[0].abs().mean(0))
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn

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

def get_intrinsics(focal_length, principal_point, use_hack: bool=True, size: list=[512, 512]):
    '''
    :param focal_length: [B, 1]
    :param principal_point: [B, 2]
    '''
    h, w = size
    intrinsics = torch.eye(3)[None, ...].float().cuda().repeat(focal_length.shape[0],1,1)
    intrinsics[:, 0, 0] = focal_length.squeeze() * w
    intrinsics[:, 1, 1] = focal_length.squeeze() * h
    intrinsics[:, 0, 2] = w/2+0.5 + principal_point[:, 0] * (w/2+0.5) 
    intrinsics[:, 1, 2] = h/2+0.5 + principal_point[:, 1] * (h/2+0.5) 

    if use_hack:
        intrinsics[:, 0:1, 2:3] = w - intrinsics[:, 0:1, 2:3]  

    return intrinsics

def get_extrinsics(R, t):
    '''
    :param R: [B, 3, 3]
    :param t: [B, 3]
    '''
    B = R.shape[0]
    w2c_openGL = torch.eye(4)[None, ...].float().repeat(B, 1, 1)
    w2c_openGL[:, :3, :3] = R
    w2c_openGL[:, :3, 3] = t
    return w2c_openGL.to(R.device)

def project_points_screen_space(points3d, focal_length, principal_point, R, t, size: int=512):
    # construct camera matrices
    intrinsics = get_intrinsics(focal_length, principal_point, size=size)
    w2c_openGL = get_extrinsics(R, t).repeat(focal_length.shape[0], 1, 1)

    B = points3d.shape[0]
    reps_extr = B if w2c_openGL.shape[0] == 1 else 1
    reps_intr = B if intrinsics.shape[0] == 1 else 1
    # apply w2c transformation
    pts_cam_space = torch.bmm(
        torch.cat([points3d, torch.ones_like(points3d[..., :1])], dim=-1),
        w2c_openGL.permute(0, 2, 1).repeat(reps_extr, 1, 1))

    # project from cam_space to screen_space
    pts_cam_space_prime = pts_cam_space[...,:3] / -pts_cam_space[...,[2]]
    pts_screen_space = (-1) * torch.bmm(
        pts_cam_space_prime, 
        intrinsics.permute(0, 2, 1).repeat(reps_intr, 1, 1),
    )[..., :2]

    h, w = size
    pts_screen_space = torch.stack([
        w - 1 - pts_screen_space[..., 0],  # x坐标翻转（宽度方向）
        pts_screen_space[..., 1], # y坐标翻转（高度方向）
        pts_cam_space[..., 2],                
    ], dim=-1)

    # pts_screen_space = torch.stack([
    #     size - 1 - pts_screen_space[..., 0], 
    #     pts_screen_space[..., 1], 
    #     pts_cam_space[..., 2],
    # ], dim=-1)
    return pts_screen_space


def intrinsics2projection(K, znear, zfar, width, height):
    x0 = 0
    y0 = 0
    if len(K.shape) == 2:
        proj = torch.zeros([4, 4], device = K.device)
        proj[0, 0] = 2 * K[0, 0] / width
        proj[0, 1] = -2 * K[0, 1] / width
        proj[0, 2] = (width - 2 * K[0, 2] + 2 * x0) / width
        proj[1, 1] = -2 * K[1, 1] / height
        proj[1, 2] = (height - 2 * K[1, 2] + 2 * y0) / height
        proj[2, 2] = (-zfar - znear) / (zfar - znear)
        proj[2, 3] = -2 * zfar * znear / (zfar - znear)
        proj[3, 2] = -1
    else:
        proj = torch.zeros([K.shape[0], 4, 4], device=K.device)
        proj[:, 0, 0] = 2 * K[:, 0, 0] / width
        proj[:, 0, 1] = -2 * K[:, 0, 1] / width
        proj[:, 0, 2] = (width - 2 * K[:, 0, 2] + 2 * x0) / width
        proj[:, 1, 1] = -2 * K[:, 1, 1] / height
        proj[:, 1, 2] = (height - 2 * K[:, 1, 2] + 2 * y0) / height
        proj[:, 2, 2] = (-zfar - znear) / (zfar - znear)
        proj[:, 2, 3] = -2 * zfar * znear / (zfar - znear)
        proj[:, 3, 2] = -1
    return proj

def projection_from_intrinsics(
    K: torch.Tensor, image_size: Tuple[int], near: float=0.1, far:float=10,
):
    """
    Transform points from camera space (x: right, y: up, z: out) to 
    clip space (x: right, y: down, z: in)
    Args:
        K: Intrinsic matrix, (N, 3, 3)
            K = [[
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1],
                ]
            ]
        image_size: (height, width)
    Output:
        proj = [[
                [2*fx/w, 0.0,     (w - 2*cx)/w,             0.0                     ],
                [0.0,    2*fy/h,  (h - 2*cy)/h,             0.0                     ],
                [0.0,    0.0,     -(far+near) / (far-near), -2*far*near / (far-near)],
                [0.0,    0.0,     -1.0,                     0.0                     ]
            ]
        ]
    """

    B = K.shape[0]
    h, w = image_size

    if K.shape[-2:] == (3, 3):
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
    elif K.shape[-1] == 4:
        fx, fy, cx, cy = K[..., [0, 1, 2, 3]].split(1, dim=-1)
    else:
        raise ValueError(f"Expected K to be (N, 3, 3) or (N, 4) but got: {K.shape}")

    proj = torch.zeros([B, 4, 4], device=K.device)
    proj[:, 0, 0]  = fx * 2 / w 
    proj[:, 1, 1]  = fy * 2 / h
    proj[:, 0, 2]  = (w - 2 * cx) / w
    proj[:, 1, 2]  = (h - 2 * cy) / h
    proj[:, 2, 2]  = -(far+near) / (far-near)
    proj[:, 2, 3]  = -2*far*near / (far-near)
    proj[:, 3, 2]  = -1
    return proj

def mvp_from_camera_param(RT, K, image_size):
    # projection matrix
    proj = projection_from_intrinsics(K, image_size)

    # Modelview and modelview + projection matrices.
    if RT.shape[-2] == 3:
        mv = torch.nn.functional.pad(RT, [0, 0, 0, 1])
        mv[..., 3, 3] = 1
    elif RT.shape[-2] == 4:
        mv = RT
    mvp = torch.bmm(proj, mv)
    return mvp

def world_to_camera(vtx, RT):
    """Transform vertex positions from the world space to the camera space"""
    RT = torch.from_numpy(RT).cuda() if isinstance(RT, np.ndarray) else RT
    if RT.shape[-2] == 3:
        mv = torch.nn.functional.pad(RT, [0, 0, 0, 1])
        mv[..., 3, 3] = 1
    elif RT.shape[-2] == 4:
        mv = RT

    # (x,y,z) -> (x',y',z',w)
    assert vtx.shape[-1] in [3, 4]
    if vtx.shape[-1] == 3:
        posw = torch.cat([vtx, torch.ones([*vtx.shape[:2], 1]).cuda()], axis=-1)
    elif vtx.shape[-1] == 4:
        posw = vtx
    else:
        raise ValueError(f"Expected 3D or 4D points but got: {vtx.shape[-1]}")
    return torch.bmm(posw, RT.transpose(-1, -2))

def camera_to_clip(vtx, K, image_size):
    """Transform vertex positions from the camera space to the clip space"""
    K = torch.from_numpy(K).cuda() if isinstance(K, np.ndarray) else K
    proj = projection_from_intrinsics(K, image_size)
    
    # (x,y,z) -> (x',y',z',w)
    assert vtx.shape[-1] in [3, 4]
    if vtx.shape[-1] == 3:
        posw = torch.cat([vtx, torch.ones([*vtx.shape[:2], 1]).cuda()], axis=-1)
    elif vtx.shape[-1] == 4:
        posw = vtx
    else:
        raise ValueError(f"Expected 3D or 4D points but got: {vtx.shape[-1]}")
    return torch.bmm(posw, proj.transpose(-1, -2))

def world_to_clip(vtx, RT, K, image_size):
    """Transform vertex positions from the world space to the clip space"""
    mvp = mvp_from_camera_param(RT, K, image_size)

    mvp = torch.from_numpy(mvp).cuda() if isinstance(mvp, np.ndarray) else mvp
    # (x,y,z) -> (x',y',z',w)
    posw = torch.cat([vtx, torch.ones([*vtx.shape[:2], 1]).cuda()], axis=-1)
    return torch.bmm(posw, mvp.transpose(-1, -2))

def world_to_ndc(vtx, RT, K, image_size, flip_y=False):
    """Transform vertex positions from the world space to the NDC space"""
    verts_clip = world_to_clip(vtx, RT, K, image_size)
    verts_ndc = verts_clip[:, :, :3] / verts_clip[:, :, 3:]
    if flip_y:
        verts_ndc[:, :, 1] *= -1
    return verts_ndc
    


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



class NVDiffRenderer(torch.nn.Module):
    def __init__(self,
        mesh_path: str,
        # hack_mask: HACKMask,
        use_opengl: bool=False, 
        lighting_type: Literal['constant', 'front', 'front-range', 'SH']='front',
        lighting_space: Literal['camera', 'world']='world',
        disturb_rate_fg: Optional[float]=0.5,
        disturb_rate_bg: Optional[float]=0.5,
        fid2cid: Optional[torch.Tensor]=None,
        shade_smooth: bool=True,
        device: Union[str, torch.device]='cuda'
    ):
        super().__init__()

        verts, uv_coords, colors, faces, uv_faces = utils.mesh.load_obj(f'{mesh_path}')
        self.pos_idx = torch.from_numpy(np.array(faces)).int()
        self.uv_idx = torch.from_numpy(np.array(uv_faces)).int()

        self.uv = torch.from_numpy(np.array(uv_coords)).float()
        self.uv[:, 1] = (self.uv[:, 1] * -1) + 1
        self.uv[:, 0] = (self.uv[:, 0] * -1) + 1

        # self.hack_mask = hack_mask
        # self.render_mask = self.hack_mask.get_mask_face().cuda()
        # self.face_mask = self.hack_mask.get_mask_face().cuda()
        # self.eyes_mask = self.hack_mask.get_mask_eyes().cuda()

        mask = torch.from_numpy(imread(f'{EYE_MASK}')/255.).permute(2,0,1)[0:3, :, :]
        mask = mask > 0.
        mask = F.interpolate(mask[None].float(), [2048, 2048], mode='bilinear')
        self.register_buffer('mask', mask)

        self.lighting_type = lighting_type
        self.lighting_space = lighting_space
        self.shade_smooth = shade_smooth
        self.glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext(device=device)
        self.fragment_cache = None

        self.disturb_rate_fg = disturb_rate_fg
        self.disturb_rate_bg = disturb_rate_bg
        # self.fid2cid = fid2cid
        if fid2cid is not None:
            fid2cid = F.pad(fid2cid, [1, 0], value=0)  # for nvdiffrast, fid==0 means background pixels
            self.register_buffer("fid2cid", fid2cid, persistent=False)

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

    def clear_cache(self):
        self.fragment_cache = None
    
    def rasterize(self, verts, faces, RT, K, image_size, use_cache=False, require_grad=False):
        """
        Rasterizes meshes using a standard rasterization approach
        :param meshes:
        :param cameras:
        :param image_size:
        :return: fragments:
                 screen_coords: N x H x W x 2  with x, y values following pytorch3ds NDC-coord system convention
                                top left = +1, +1 ; bottom_right = -1, -1
        """
        # v_normals = self.compute_v_normals(verts, faces)
        # vertices and faces
        verts_camera = world_to_camera(verts, RT)
        verts_clip = camera_to_clip(verts_camera, K, image_size)
        tri = faces.int()
        rast_out, rast_out_db = self.rasterize_fragments(
            verts_clip, tri, image_size, use_cache, require_grad,
        )
        rast_dict = {
            "rast_out": rast_out,
            "rast_out_db": rast_out_db,
            "verts": verts,
            "verts_camera": verts_camera[..., :3],
            "verts_clip": verts_clip,
        }
        
        # if not require_grad:
        #     verts_ndc = verts_clip[:, :, :3] / verts_clip[:, :, 3:]
        #     screen_coords = self.compute_screen_coords(rast_out, verts_ndc, faces, image_size)
        #     rast_dict["screen_coords"] = screen_coords

        return rast_dict

    def rasterize_fragments(self, verts_clip, tri, image_size, use_cache, require_grad=False):
        """ Either rasterizes meshes or returns cached result
        """
        if not use_cache or self.fragment_cache is None:
            if require_grad:
                rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)
            else:
                with torch.no_grad():
                    rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)
            self.fragment_cache = (rast_out, rast_out_db)

        return self.fragment_cache

    def compute_screen_coords(self, 
        rast_out: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor, image_size: Tuple[int],
    ):
        """ Compute screen coords for visible pixels
        Args:
            verts: (N, V, 3), the verts should lie in the ndc space 
            faces: (F, 3)
        """
        N = verts.shape[0]
        F = faces.shape[0]
        meshes = Meshes(verts, faces[None, ...].expand(N, -1, -1))
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        face_verts = verts_packed[faces_packed]

        # NOTE: nvdiffrast shifts face index by +1, and use 0 to flag empty pixel
        pix2face = rast_out[..., -1:].long() - 1  # (N, H, W, 1)
        is_visible = pix2face > -1  # (N, H, W, 1)
        # NOTE: is_visible is computed before packing pix2face to ensure correctness
        pix2face_packed = pix2face + torch.arange(0, N)[:, None, None, None].to(pix2face) * F

        bary_coords = rast_out[..., :2]  # (N, H, W, 2)
        bary_coords = torch.cat([bary_coords, 1 - bary_coords.sum(dim=-1, keepdim=True)], dim =-1)  # (N, H, W, 3)

        visible_faces = pix2face_packed[is_visible]  # (sum(is_visible), 3, 3)
        visible_face_verts = face_verts[visible_faces]
        visible_bary_coords = bary_coords[is_visible[..., 0]]  # (sum(is_visible), 3, 1)
        # visible_bary_coords = torch.cat([visible_bary_coords, 1 - visible_bary_coords.sum(dim=-1, keepdim=True)], dim =-1)

        visible_surface_point = visible_face_verts * visible_bary_coords[..., None]
        visible_surface_point = visible_surface_point.sum(dim=1)

        screen_coords = torch.zeros(*pix2face_packed.shape[:3], 2, device=meshes.device)
        screen_coords[is_visible[..., 0]] = visible_surface_point[:, :2]  # now have gradient

        return screen_coords
    
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
    
    def detach_by_indices(self, x, indices):
        x = x.clone()
        x[:, indices] = x[:, indices].detach()
        return x
    
    def forward(self, 
        verts, faces, verts_clip, RT, K, image_size=[512, 512], tex=None, background_color=[1., 1., 1.], 
        light=None, 
        neck_mask=None, face_mask=None, ears_mask=None,
        eyes_mask=None, mouth_mask=None, lipread_mask=None, 
        enable_disturbance=False, 
        align_texture_except_fid=None, align_boundary_except_vid=None,    
        use_cache=False, require_grad=True,       
    ):
        B = verts.shape[0]
        tri = faces.int()
        if verts_clip is None:
            verts_camera_ = world_to_camera(verts, RT)
            verts_camera = verts_camera_[..., :3]
            verts_clip = camera_to_clip(verts_camera_, K, image_size)

        if not use_cache or self.fragment_cache is None:
            if require_grad:
                rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)
            else:
                with torch.no_grad():
                    rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)
            self.fragment_cache = (rast_out, rast_out_db)

        # rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)

        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1).bool()
        face_id = torch.clamp(rast_out[..., -1:].long() - 1, 0)  # (B, H, W, 1)
        H, W = face_id.shape[1:3]

        if  self.lighting_space == 'world':
            v_normal = compute_v_normals(verts, tri)
        elif  self.lighting_space == 'camera':
            assert verts_clip is None
            v_normal = compute_v_normals(verts_camera, tri)
        else:
            raise NotImplementedError(f"Unknown lighting space: {self.lighting_space}")

        normals, _ = dr.interpolate(v_normal, rast_out, tri)
        normals = safe_normalize(normals)  # [B, H, W, 3]
        
        texc, texd = dr.interpolate(
            attr=self.uv.to(verts.device), rast=rast_out, tri=self.uv_idx.to(verts.device), 
            rast_db=rast_out_db, diff_attrs='all',
        )

        if align_texture_except_fid is not None:  # TODO: rethink when shading with normal
            fid = rast_out[..., -1:].long()  # the face index is shifted by +1
            mask_ = torch.zeros(faces.shape[0]+1, dtype=torch.bool, device=fid.device)
            mask_[align_texture_except_fid + 1] = True
            # b, h, w = rast_out.shape[:3]
            rast_mask = torch.gather(mask_.reshape(1, 1, 1, -1).expand(B, H, W, -1), 3, fid)
            texc = torch.where(rast_mask, texc.detach(), texc)

        if tex is None:
            albedo = dr.texture(
                tex=self.mask.repeat(B,1,1,1).permute(0,2,3,1).contiguous(), 
                uv=texc, uv_da=texd, filter_mode='linear-mipmap-linear', max_mip_level=None,
            )
        else:
            tex = tex.permute(0, 2, 3, 1).contiguous()  # (N, T, T, 3)
            albedo = dr.texture(
                tex=tex, uv=texc, uv_da=texd, filter_mode='linear-mipmap-linear', max_mip_level=None,
            )  # (B, H, W, 3)

        # mask_images = dr.texture(
        #     self.mask.repeat(B, 1, 1, 1).permute(0,2,3,1).contiguous(), texc, filter_mode='linear') #.permute(0, 3, 1, 2)
        # mask_images = dr.antialias(
        #     mask_images, rast_out, verts_clip, self.pos_idx.to(verts.device),
        # )

        if face_mask is not None:
            assert face_mask.shape[0] == B and len(face_mask.shape) == 3
            rendered_face_mask = dr.interpolate(face_mask, rast_out, self.pos_idx.to(verts.device))[0]
        
        if neck_mask is not None:
            assert neck_mask.shape[0] == B and len(neck_mask.shape) == 3
            rendered_neck_mask = dr.interpolate(neck_mask, rast_out, self.pos_idx.to(verts.device))[0]
        
        if ears_mask is not None:
            assert ears_mask.shape[0] == B and len(ears_mask.shape) == 3
            rendered_ears_mask = dr.interpolate(ears_mask, rast_out, self.pos_idx.to(verts.device))[0]

        if eyes_mask is not None:
            assert eyes_mask.shape[0] == B and len(eyes_mask.shape) == 3
            # eyes_mask = self.eyes_mask.repeat(B, 1, 1)
            rendered_eyes_mask = dr.interpolate(eyes_mask, rast_out, self.pos_idx.to(verts.device))[0]

        if mouth_mask is not None:
            assert mouth_mask.shape[0] == B and len(mouth_mask.shape) == 3
            rendered_mouth_mask = dr.interpolate(mouth_mask, rast_out, self.pos_idx.to(verts.device))[0]

        if lipread_mask is not None:
            assert lipread_mask.shape[0] == B and len(lipread_mask.shape) == 3
            rendered_lipread_mask = dr.interpolate(lipread_mask, rast_out, self.pos_idx.to(verts.device))[0]

        uv_images = torch.cat([1 - texc[..., :1], texc[..., 1:]], dim=-1)
        
        # ---- shading ----
        if light is None:
            light = specified_light[None].to(verts.device) # [1, 9, 3]

        shading_images = self.shade(normals, light)       

        rgb = albedo * shading_images
        alpha = fg_mask.float()
        rgba = torch.cat([rgb, alpha], dim=-1)

        # ---- background ----
        if isinstance(background_color, list) or isinstance(background_color, tuple):
            """Background as a constant color"""
            rgba_bg = torch.tensor(list(background_color) + [0]).to(rgba).expand_as(rgba)  # RGBA
        elif isinstance(background_color, torch.Tensor):
            """Background as a image"""
            rgba_bg = background_color  # [1, H, W, 3]
            rgba_bg = torch.cat([rgba_bg, torch.zeros_like(rgba_bg[..., :1])], dim=-1)  # RGBA
        else:
            raise ValueError(f"Unknown background type: {type(background_color)}")
        rgba_bg = rgba_bg.flip(1)  # opengl camera has y-axis up, needs flipping
        
        # normals = torch.where(fg_mask, normals, rgba_bg[...,:3])
        # diffuse = torch.where(fg_mask, diffuse, rgba_bg[...,:3])
        rgba = torch.where(fg_mask, rgba, rgba_bg)
        
        # uv_images = torch.where(fg_mask, uv_images, rgba_bg[...,:2])
        uv_images = torch.concat([uv_images, torch.zeros_like(uv_images)[...,:1]], dim=-1)

        if enable_disturbance and self.fid2cid is not None:
            # ------- color disturbance -------
            B, H, W, _ = rgba.shape
            # compute random blending weights based on the disturbance rate
            if self.disturb_rate_fg is not None:
                w_fg = (torch.rand_like(rgba[..., :1]) < self.disturb_rate_fg).int()
            else:
                w_fg = torch.zeros_like(rgba[..., :1]).int()
            if self.disturb_rate_bg is not None:
                w_bg = (torch.rand_like(rgba[..., :1]) < self.disturb_rate_bg).int()
            else:
                w_bg = torch.zeros_like(rgba[..., :1]).int()
            
            # sample pixles from clusters
            fid = rast_out[..., -1:].long()  # the face index is shifted by +1
            num_clusters = self.fid2cid.max() + 1

            fid2cid = self.fid2cid[None, None, None, :].expand(B, H, W, -1)
            cid = torch.gather(fid2cid, -1, fid)

            rgba_ = torch.zeros_like(rgba)
            for i in range(num_clusters):
                c_rgba = rgba_bg if i == 0 else rgba
                w = w_bg if i == 0 else w_fg

                c_mask = cid == i
                c_pixels = c_rgba[c_mask.repeat_interleave(4, dim=-1)].reshape(-1, 4).detach()  # NOTE: detach to avoid gradient flow

                if i != 1:  # skip #1 indicate faces that are not in any cluster
                    if len(c_pixels) > 0:
                        c_idx = torch.randint(0, len(c_pixels), (B * H * W, ), device=c_pixels.device)
                        c_sample = c_pixels[c_idx].reshape(B, H, W, 4)
                        rgba_ += c_mask * (c_sample * w + c_rgba * (1 - w))
                else:
                    rgba_ += c_mask * c_rgba
            rgba = rgba_
        else:
            cid = None
        
        # -------- AA on both RGB and alpha channels --------
        if align_boundary_except_vid is not None:
            verts_clip = self.detach_by_indices(verts_clip, align_boundary_except_vid)
        rgba_aa = dr.antialias(rgba, rast_out, verts_clip, tri)
        aa = ((rgba - rgba_aa) != 0).any(dim=-1, keepdim=True).repeat_interleave(4, dim=-1)

        outputs = {
            'albedo': albedo.flip(1).permute(0,3,1,2),  # channel first, [B, C, H, W]
            'normals': normals.flip(1).permute(0,3,1,2),
            'rgb': rgba.flip(1).permute(0,3,1,2),
            'rgba': rgba_aa.flip(1).permute(0,3,1,2),
            'aa': aa[..., :3].float().flip(1).permute(0,3,1,2),
            'verts_clip': verts_clip,
            # 'fg_images': mask_images.flip(1).permute(0,3,1,2),
            'uv_images': uv_images.flip(1).permute(0,3,1,2),
        }

        if cid is not None:
            outputs.update({
                'cid': cid.flip(1),
            })
        if face_mask is not None:
            outputs.update({
                'mask_images_face': (rendered_face_mask > 0).float().flip(1).permute(0,3,1,2),
            })
        if neck_mask is not None:
            outputs.update({
                'mask_images_neck': (rendered_neck_mask > 0).float().flip(1).permute(0,3,1,2),
            })
        if ears_mask is not None:
            outputs.update({
                'mask_images_ears': (rendered_ears_mask > 0).float().flip(1).permute(0,3,1,2),
            })
        if eyes_mask is not None:
            outputs.update({
                'mask_images_eyes': (rendered_eyes_mask > 0).float().flip(1).permute(0,3,1,2),
            })
        if mouth_mask is not None:
            outputs.update({
                'mask_images_mouth': (rendered_mouth_mask > 0).float().flip(1).permute(0,3,1,2),
            })
        if lipread_mask is not None:
            outputs.update({
                'mask_images_lipread': (rendered_lipread_mask > 0).float().flip(1).permute(0,3,1,2),
            })

        return outputs

    def add_directionlight(self, normals, lights):
        '''
        normals: [B, V, 3]
        lights: [B, nlight, 6]
        returns:
            shading: [B, V, 3]
        '''
        light_direction = lights[...,:3] 
        light_intensities = lights[...,3:]
        directions_to_lights = F.normalize(
            light_direction[:,:,None,:].expand(-1, -1, normals.shape[1], -1), dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None] * light_intensities[:,:,None,:]
        return shading.mean(1)
    
# plt.imshow(outputs['albedo'].detach().cpu().numpy()[0].transpose((1, 2, 0)))
# plt.imshow(outputs['normals'].detach().cpu().numpy()[0].transpose((1, 2, 0)))
# plt.imshow(outputs['diffuse'].detach().cpu().numpy()[0].transpose((1, 2, 0)))
# plt.imshow(outputs['rgba'].detach().cpu().numpy()[0].transpose((1, 2, 0)))
# plt.imshow(outputs['fg_images'].detach().cpu().numpy()[0].transpose((1, 2, 0)))
# plt.imshow(outputs['uv_images'].detach().cpu().numpy()[0].transpose((1, 2, 0)))
# plt.imshow(outputs['mask_images_eyes'].detach().cpu().numpy()[0].transpose((1, 2, 0)))

    def render_shape(self, 
        vertices, camera, faces, image_size=[224, 224], light=None,
        background_color=[0., 0., 0.],
        neck_mask=None, face_mask=None, ears_mask=None,
        eyes_mask=None, mouth_mask=None, lipread_mask=None, 
        align_texture_except_fid=None, align_boundary_except_vid=None,    
        use_cache=False, require_grad=True,       
    ):
        B = vertices.shape[0]
        tri = faces.int()

        # transformation
        transformed_vertices = batch_orth_proj(vertices, camera)
        verts_clip = torch.cat([
            transformed_vertices, 
            torch.ones([*transformed_vertices.shape[:2], 1]).cuda()], axis=-1)
        rot_mat = torch.eye(4, 4).unsqueeze(0).expand(B, -1, -1).to(faces.device)
        rot_mat[:, 1:3, 1:3] = torch.tensor([[-1.0, 0.0],[0.0, -1.0]])[None].repeat(B, 1, 1)
        verts_clip = torch.bmm(verts_clip, rot_mat)

        if light is None:
            light = specified_light[None].to(faces.device) # [1, 9, 3]
        
        if not use_cache or self.fragment_cache is None:
            if require_grad:
                rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)
            else:
                with torch.no_grad():
                    rast_out, rast_out_db = dr.rasterize(self.glctx, verts_clip, tri, image_size)
            self.fragment_cache = (rast_out, rast_out_db)
        
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1).bool()
        face_id = torch.clamp(rast_out[..., -1:].long() - 1, 0)  # (B, H, W, 1)
        H, W = face_id.shape[1:3]

        if self.shade_smooth:
            if  self.lighting_space == 'world':
                v_normal = compute_v_normals(vertices, tri)
            elif  self.lighting_space == 'camera':
                v_normal = compute_v_normals(transformed_vertices, tri)
            else:
                raise NotImplementedError(f"Unknown lighting space: {self.lighting_space}")

            normals, _ = dr.interpolate(v_normal, rast_out, tri)
            normals = safe_normalize(normals)  # [B, H, W, 3]
        else:
            face_normals = compute_face_normals(transformed_vertices, tri)  # (B, F, 3)
            face_normals_ = face_normals[:, None, None, :, :].expand(-1, W, H, -1, -1)  # (B, 1, 1, F, 3)
            face_id_ = face_id[:, :, :, None].expand(-1, -1, -1, -1, 3)  # (B, W, H, 1, 1)
            normals = torch.gather(face_normals_, -2, face_id_).squeeze(-2) # (B, W, H, 3)
        
        shading_images = self.shade(normals, light)  

        texc, texd = dr.interpolate(
            attr=self.uv.to(faces.device), rast=rast_out, tri=self.uv_idx.to(faces.device), 
            rast_db=rast_out_db, diff_attrs='all',
        )

        if align_texture_except_fid is not None:  # TODO: rethink when shading with normal
            fid = rast_out[..., -1:].long()  # the face index is shifted by +1
            mask_ = torch.zeros(faces.shape[0]+1, dtype=torch.bool, device=fid.device)
            mask_[align_texture_except_fid + 1] = True
            # b, h, w = rast_out.shape[:3]
            rast_mask = torch.gather(mask_.reshape(1, 1, 1, -1).expand(B, H, W, -1), 3, fid)
            texc = torch.where(rast_mask, texc.detach(), texc)

        albedo = dr.texture(
            tex=self.mask.repeat(B,1,1,1).permute(0,2,3,1).contiguous(), 
            uv=texc, uv_da=texd, filter_mode='linear-mipmap-linear', max_mip_level=None,
        )

        if face_mask is not None:
            assert face_mask.shape[0] == B and len(face_mask.shape) == 3
            rendered_face_mask = dr.interpolate(face_mask, rast_out, self.pos_idx.to(vertices.device))[0]
        
        if neck_mask is not None:
            assert neck_mask.shape[0] == B and len(neck_mask.shape) == 3
            rendered_neck_mask = dr.interpolate(neck_mask, rast_out, self.pos_idx.to(vertices.device))[0]
        
        if ears_mask is not None:
            assert ears_mask.shape[0] == B and len(ears_mask.shape) == 3
            rendered_ears_mask = dr.interpolate(ears_mask, rast_out, self.pos_idx.to(vertices.device))[0]

        if eyes_mask is not None:
            assert eyes_mask.shape[0] == B and len(eyes_mask.shape) == 3
            # eyes_mask = self.eyes_mask.repeat(B, 1, 1)
            rendered_eyes_mask = dr.interpolate(eyes_mask, rast_out, self.pos_idx.to(vertices.device))[0]

        if mouth_mask is not None:
            assert mouth_mask.shape[0] == B and len(mouth_mask.shape) == 3
            rendered_mouth_mask = dr.interpolate(mouth_mask, rast_out, self.pos_idx.to(vertices.device))[0]

        if lipread_mask is not None:
            assert lipread_mask.shape[0] == B and len(lipread_mask.shape) == 3
            rendered_lipread_mask = dr.interpolate(lipread_mask, rast_out, self.pos_idx.to(vertices.device))[0]

        # rgb = albedo * shading_images
        rgb = shading_images
        alpha = fg_mask.float()
        rgba = torch.cat([rgb, alpha], dim=-1)

        # ---- background ----
        if isinstance(background_color, list) or isinstance(background_color, tuple):
            """Background as a constant color"""
            rgba_bg = torch.tensor(list(background_color) + [0]).to(rgba).expand_as(rgba)  # RGBA
        elif isinstance(background_color, torch.Tensor):
            """Background as a image"""
            rgba_bg = background_color  # [1, H, W, 3]
            rgba_bg = torch.cat([rgba_bg, torch.zeros_like(rgba_bg[..., :1])], dim=-1)  # RGBA
        else:
            raise ValueError(f"Unknown background type: {type(background_color)}")
        # rgba_bg = rgba_bg.flip(1)  # opengl camera has y-axis up, needs flipping
        
        rgba = torch.where(fg_mask, rgba, rgba_bg)
        
        if align_boundary_except_vid is not None:
            verts_clip = self.detach_by_indices(verts_clip, align_boundary_except_vid)
        rgba_aa = dr.antialias(rgba, rast_out, verts_clip, tri)

        outputs = {
            'albedo': albedo.permute(0,3,1,2),  # channel first, [B, C, H, W]
            'normals': normals.permute(0,3,1,2),
            'fg_mask': fg_mask.permute(0,3,1,2),  # [B, 1, H, W]
            'rgb': rgb.permute(0,3,1,2),
            'rgba': rgba.permute(0,3,1,2),
            'rgba_aa': rgba_aa.permute(0,3,1,2),
        }

        if face_mask is not None:
            outputs.update({
                'mask_images_face': (rendered_face_mask > 0).float().permute(0,3,1,2),
            })
        if neck_mask is not None:
            outputs.update({
                'mask_images_neck': (rendered_neck_mask > 0).float().permute(0,3,1,2),
            })
        if ears_mask is not None:
            outputs.update({
                'mask_images_ears': (rendered_ears_mask > 0).float().permute(0,3,1,2),
            })
        if eyes_mask is not None:
            outputs.update({
                'mask_images_eyes': (rendered_eyes_mask > 0).float().permute(0,3,1,2),
            })
        if mouth_mask is not None:
            outputs.update({
                'mask_images_mouth': (rendered_mouth_mask > 0).float().permute(0,3,1,2),
            })
        if lipread_mask is not None:
            outputs.update({
                'mask_images_lipread': (rendered_lipread_mask > 0).float().permute(0,3,1,2),
            })

        return outputs






