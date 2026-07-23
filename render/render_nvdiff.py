import numpy as np
import torch
import torch.nn.functional as F
import kornia
import matplotlib.pyplot as plt

from kornia.geometry.camera import pixel2cam
from typing import List
from scipy.io import loadmat
from torch import nn
from typing import Tuple, Literal, Optional, Union
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
import nvdiffrast.torch as dr

from utils.mesh import dict2obj, read_obj, face_vertices, expand_mesh_by_uv, generate_triangles


def dot(x: torch.Tensor, y: torch.Tensor):
    return torch.sum(x*y, -1, keepdim=True)

def safe_normalize(x: torch.Tensor, eps: float=1e-6):
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

    v_normals = torch.where(
        dot(v_normals, v_normals) > 1e-20, 
        v_normals, 
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'),
    )
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals


def compute_face_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_normals = safe_normalize(face_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(face_normals))
    return face_normals


def ndc_projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0, n/-x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

def perspective_projection(focal, center):
    # return p.T (N, 3) @ (3, 3) 
    return np.array([
                focal,  0,    center,
                  0,   focal, center,
                  0,    0,     1
            ]).reshape([3, 3]).astype(np.float32).transpose()

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
        N[..., 0]**2 - N[..., 1]**2,
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


class Pytorch3dRasterizer(nn.Module):
    ## TODO: add support for rendering non-squared images, since pytorc3d supports this now
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """

    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin':  None,
            'perspective_correct': False,
        }
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[...,:2] = -fixed_vertices[...,:2]
        raster_settings = self.raster_settings
        if h is None and w is None:
            image_size = raster_settings.image_size
        else:
            image_size = [h, w]
            if h > w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h
            
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
        pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
        # print(image_size)
        # import ipdb; ipdb.set_trace()
        return pixel_vals   # [B, 4, H, W]


class MeshRenderer(nn.Module):
    def __init__(self,
        rasterize_fov,
        znear=5.0,
        zfar=15.0, 
        rasterize_size=224,
        use_opengl=True,
        lighting_type: Literal['constant', 'front', 'front-range', 'SH']='SH',
        align_normals=True,
        uv_size=256,
        template_path='/home/ubuntu/MyExp/TRACK/data/hack_data/normalized_hack_template_uv.obj',
    ):
        """
        
        fov = 2 * np.arctan(center / focal) * 180 / np.pi
        """

        super(MeshRenderer, self).__init__()

        x = np.tan(np.deg2rad(rasterize_fov * 0.5)) * znear
        self.ndc_proj = torch.tensor(
            ndc_projection(x=x, n=znear, f=zfar),
        ).matmul(torch.diag(torch.tensor([1., -1, -1, 1])))

        self.rasterize_size = rasterize_size
        self.use_opengl = use_opengl
        self.ctx = None
        
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
        self.align_normals = align_normals

        self.uv_rasterizer = Pytorch3dRasterizer(uv_size)
        self.uv_size = uv_size
        mesh_info = read_obj(template_path)
        verts = torch.from_numpy(mesh_info.vs)
        faces = torch.from_numpy(mesh_info.fvs[None, ...])

        if len(mesh_info.vts) == 0:
            _mesh_info = read_obj('/home/ubuntu/MyExp/TRACK/data/hack_data/normalized_hack_template_uv.obj')
            uvcoords = torch.from_numpy(_mesh_info.vts[None, ...])
            uvfaces = torch.from_numpy(_mesh_info.fvts[None, ...])
        else:
            uvcoords = torch.from_numpy(mesh_info.vts[None, ...])
            uvfaces = torch.from_numpy(mesh_info.fvts[None, ...])
    
        dense_triangles = generate_triangles(uv_size, uv_size)
        self.register_buffer('dense_faces', torch.from_numpy(dense_triangles).long()[None,:,:])
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat([uvcoords, uvcoords[:,:,0:1]*0.+1.], -1)   # [B, V, 3]
        uvcoords = uvcoords * 2 - 1
        uvcoords[...,1] = -uvcoords[...,1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)  # [B, F, 3, 3]
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)


    def forward(self, vertex, tri, feat=None, face_mask=None, eyes_region_mask=None, nose_mask=None):
        """
        Return:
            mask               -- torch.tensor, size (B, 1, H, W)
            depth              -- torch.tensor, size (B, 1, H, W)
            features(optional) -- torch.tensor, size (B, C, H, W) if feat is not None
            mask_images_face(optional) -- (B, C, H, W) if face_mask is not None
            mask_images_eyes(optional) -- (B, C, H, W) if eyes_mask is not None

        Parameters:
            vertex          -- torch.tensor, size (B, N, 3)
            tri             -- torch.tensor, size (B, M, 3) or (M, 3), triangles
            feat(optional)  -- torch.tensor, size (B, C), features
            face_mask(optional) -- per-vertex mask, size (B, N, C_face), rasterized to screen
            eyes_mask(optional) -- per-vertex eyes region mask, size (B, N, C_eyes), same convention as face_mask
        """
        device = vertex.device
        B = vertex.shape[0]
        rsize = int(self.rasterize_size)
        ndc_proj = self.ndc_proj.to(device)
        # trans to homogeneous coordinates of 3d vertices, the direction of y is the same as v
        if vertex.shape[-1] == 3:
            vertex = torch.cat([vertex, torch.ones([*vertex.shape[:2], 1]).to(device)], dim=-1)
            vertex[..., 1] = -vertex[..., 1] 

        vertex_ndc = vertex @ ndc_proj.t()
        if self.ctx is None:
            if self.use_opengl:
                self.ctx = dr.RasterizeGLContext(device=device)
                ctx_str = "opengl"
            else:
                self.ctx = dr.RasterizeCudaContext(device=device)
                ctx_str = "cuda"
            print("create %s ctx on device cuda:%d"%(ctx_str, device.index))
        
        ranges = None
        if isinstance(tri, List) or len(tri.shape) == 3:
            vum = vertex_ndc.shape[1]
            fnum = torch.tensor([f.shape[0] for f in tri]).unsqueeze(1).to(device) 
            fstartidx = torch.cumsum(fnum, dim=0) - fnum 
            ranges = torch.cat([fstartidx, fnum], axis=1).type(torch.int32).cpu()
            for i in range(tri.shape[0]):
                tri[i] = tri[i] + i*vum
            vertex_ndc = torch.cat(vertex_ndc, dim=0)
            tri = torch.cat(tri, dim=0)

        # for range_mode vertex: [B*N, 4], tri: [B*M, 3], for instance_mode vetex: [B, N, 4], tri: [M, 3]
        tri = tri.type(torch.int32).contiguous()
        # 必须保留 rast_out_db 并传入 interpolate，否则对 clip 顶点 / 屏幕投影的反传无有效导数，易出现 NaN 梯度
        rast_out, rast_out_db = dr.rasterize(
            self.ctx, vertex_ndc.contiguous(), tri, resolution=[rsize, rsize], ranges=ranges,
        )
        
        fg_mask = torch.clamp(rast_out[..., -1:], 0, 1).bool()
        face_id = torch.clamp(rast_out[..., -1:].long() - 1, 0)  # (B, H, W, 1)
        
        v_normal = compute_v_normals(vertex[...,:3], tri)
        normals, _ = dr.interpolate(v_normal, rast_out, tri, rast_db=rast_out_db)
        if self.align_normals:
            normals[...,[1,2]] *= -1.0
        normals = safe_normalize(normals)  # [B, H, W, 3]
        shading_images = self.shade(normals, specified_light[None].to(device))
        shading_images = shading_images.permute(0,3,1,2)  # [B, 3, H, W]

        depth, _ = dr.interpolate(
            vertex.reshape([-1, 4])[..., 2].unsqueeze(1).contiguous(),
            rast_out, tri, rast_db=rast_out_db,
        )
        depth = depth.permute(0,3,1,2)
        mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
        depth = mask * depth

        image = None
        if feat is not None:
            image, _ = dr.interpolate(feat, rast_out, tri, rast_db=rast_out_db)
            image = image.permute(0,3,1,2)
            image = mask * image

        render_out = {
            'depth': depth,    # [B, 1, H, W]
            'mask': mask,
            'image': image,
            'rgb': shading_images,
            'normals': normals.permute(0,3,1,2),  # [B, 3, H, W]
        }

        if face_mask is not None:
            assert face_mask.shape[0] == B and len(face_mask.shape) == 3
            rendered_face_mask, _ = dr.interpolate(
                face_mask, rast_out, tri.to(device), rast_db=rast_out_db,
            )
            render_out.update({
                'mask_images_face': (rendered_face_mask > 0).float().permute(0,3,1,2),
            })

        if eyes_region_mask is not None:
            assert eyes_region_mask.shape[0] == B and len(eyes_region_mask.shape) == 3
            rendered_eyes_mask, _ = dr.interpolate(
                eyes_region_mask, rast_out, tri.to(device), rast_db=rast_out_db,
            )
            render_out.update({
                'mask_images_eyes_region': (rendered_eyes_mask > 0).float().permute(0,3,1,2),
            })
        if nose_mask is not None:
            assert nose_mask.shape[0] == B and len(nose_mask.shape) == 3
            rendered_nose_mask, _ = dr.interpolate(
                nose_mask, rast_out, tri.to(device), rast_db=rast_out_db,
            )
            render_out.update({
                'mask_images_nose': (rendered_nose_mask > 0).float().permute(0,3,1,2),
            })

        return render_out

    def world2uv(self, vertices: torch.Tensor) -> torch.Tensor:
        '''
            warp vertices from world space to uv space
            vertices: [B, V, 3]
            face_vertices: [B, F, 3, 3]
            uv_vertices: [B, 3, H, W]
        '''
        device = vertices.device
        B = vertices.shape[0]
        face_verts = face_vertices(vertices, self.faces.expand(B, -1, -1).to(device))
        uv_vertices = self.uv_rasterizer(
            self.uvcoords.expand(B, -1, -1).to(device), 
            self.uvfaces.expand(B, -1, -1).to(device), 
            face_verts,
        )[:, :3]

        return uv_vertices
    

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



if __name__ == '__main__':

    import trimesh
    import polyscope as ps
    from utils.mesh import read_obj
        
    device = 'cuda'

    # hack_mesh_path = '/home/ubuntu/MyExp/TRACK/data/hack_data/normalized_hack_template_uv.obj'
    hack_mesh_path = '/home/ubuntu/MyExp/TRACK/data/hack_data/hack_template_turn_left_open_mouth.obj'

    mesh_info = read_obj(hack_mesh_path)
    verts_full = mesh_info.vs
    faces_full = mesh_info.fvs

    uv_coords_full = mesh_info.vts   # [14434, 2]
    uv_faces_full = mesh_info.fvts   # [28068, 3]

    # hack_mesh_path = f'/home/ubuntu/MyExp/Deep3DFaceRecon_pytorch/checkpoints/face_recon' \
    #                     + f'/results/Deep3DFaceRecon_results/epoch_20_000000/zheng_224.obj'
    
    # mesh = trimesh.load(hack_mesh_path)
    # verts_full = np.array(mesh.vertices).astype(np.float32)
    # faces_full = np.array(mesh.faces)

    center = 112.0
    focal = 1015.0
    z_near = 5.0
    z_far = 15.0

    # center = 112.0
    # focal = 1015.0
    # z_near = 5.0
    # z_far = 15.0

    fov = 2 * np.arctan(center / focal) * 180 / np.pi

    camera_distance = 10.0

    renderer = MeshRenderer(
        rasterize_fov=fov, znear=z_near, zfar=z_far, 
        rasterize_size=int(2 * center), 
        use_opengl=True,
    )

    face_vertex = verts_full
    face_vertex[..., -1] = camera_distance - face_vertex[..., -1]

    render_out = renderer(
        torch.from_numpy(face_vertex)[None].to(device),
        torch.from_numpy(faces_full).to(device),
    )

    # world2uv test (pass faces when V!=Vt for expand_mesh_by_uv)
    verts_t = torch.from_numpy(verts_full)[None].float().to(device)

    v_normals = compute_v_normals(torch.tensor(verts_full)[None].to(device), torch.tensor(faces_full).to(device))


    uvcoords_t = torch.from_numpy(uv_coords_full).float().to(device)
    uvfaces_t = torch.from_numpy(uv_faces_full).long().to(device)
    faces_t = torch.from_numpy(faces_full).long().to(device)

    uv_verts = renderer.world2uv(verts_t)
    # uv_verts = renderer.world2uv(v_normals)

    assert uv_verts.shape == (1, 3, 256, 256)
    assert uv_verts.min() >= -1.0 and uv_verts.max() <= 1.0, 'uv_verts is out of range (-1.0, 1.0)'

    plt.imshow(uv_verts[0].detach().cpu().permute(1,2,0) * 0.5 + 0.5)

    # plt.imshow(render_out['rgb'][0].detach().cpu().permute(1,2,0))

    # plt.imshow(render_out['mask'][0].detach().cpu().permute(1,2,0))

    # RGB TO BGR for normals
    normals_show = render_out['normals'][0].detach().cpu().permute(1,2,0) * 0.5 + 0.5
    plt.imshow(normals_show)



    ps.init()
    ps.register_surface_mesh(
        "mesh",
        vertices=verts_full,
        faces=faces_full,
    )
    ps.show()

