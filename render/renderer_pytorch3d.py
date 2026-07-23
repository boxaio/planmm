import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import rasterize_meshes

from utils.mesh import dict2obj, read_obj, write_mesh_obj, face_vertices, vertex_normals, generate_triangles


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



class SRenderY(nn.Module):
    def __init__(self, image_size, 
        obj_filename='/home/ubuntu/MyExp/TRACK/data/hack_data/normalized_hack_template_uv.obj', 
        uv_size=256, rasterizer_type='pytorch3d',
    ):
        super(SRenderY, self).__init__()
        self.image_size = image_size
        self.uv_size = uv_size

        if rasterizer_type == 'pytorch3d':

            self.rasterizer = Pytorch3dRasterizer(image_size)
            self.uv_rasterizer = Pytorch3dRasterizer(uv_size)
            # verts, faces, aux = load_obj(obj_filename)
            # uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
            # uvfaces = faces.textures_idx[None, ...] # (N, F, 3)
            # faces = faces.verts_idx[None,...]

            mesh_info = read_obj(obj_filename)
            verts = torch.from_numpy(mesh_info.vs)
            faces = torch.from_numpy(mesh_info.fvs[None, ...])

            if len(mesh_info.vts) == 0:
                _mesh_info = read_obj('/home/ubuntu/MyExp/TRACK/data/hack_data/normalized_hack_template_uv.obj')
                uvcoords = torch.from_numpy(_mesh_info.vts[None, ...])
                uvfaces = torch.from_numpy(_mesh_info.fvts[None, ...])
            else:
                uvcoords = torch.from_numpy(mesh_info.vts[None, ...])
                uvfaces = torch.from_numpy(mesh_info.fvts[None, ...])
            
            # uvfaces = faces
        else:
            NotImplementedError

        # faces
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

        # shape colors, for rendering shape overlay
        colors = torch.tensor([180, 180, 180])[None, None, :].repeat(1, faces.max()+1, 1).float()/255.
        face_colors = face_vertices(colors, faces)  # [B, F, 3, 3]
        self.register_buffer('face_colors', face_colors)

        ## SH factors for lighting
        pi = np.pi
        constant_factor = torch.tensor([
            1/np.sqrt(4*pi), ((2*pi)/3)*(np.sqrt(3/(4*pi))), ((2*pi)/3)*(np.sqrt(3/(4*pi))),\
            ((2*pi)/3)*(np.sqrt(3/(4*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3)*(np.sqrt(5/(12*pi))),\
            (pi/4)*(3)*(np.sqrt(5/(12*pi))), (pi/4)*(3/2)*(np.sqrt(5/(12*pi))), (pi/4)*(1/2)*(np.sqrt(5/(4*pi))),
        ]).float()
        self.register_buffer('constant_factor', constant_factor)
    

    def forward(self, vertices, transformed_vertices, albedos, lights=None, light_type='point'):
        '''
        -- Texture Rendering
        vertices: [batch_size, V, 3], vertices in world space, for calculating normals, then shading
        transformed_vertices: [batch_size, V, 3], range:normalized to [-1,1], projected vertices in image space (that is aligned to the iamge pixel), for rasterization
        albedos: [batch_size, 3, h, w], uv map
        lights: 
            spherical homarnic: [N, 9(shcoeff), 3(rgb)]
            points/directional lighting: [N, n_lights, 6(xyzrgb)]
        light_type:
            point or directional
        '''
        batch_size = vertices.shape[0]
        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        # attributes
        face_verts = face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        
        attributes = torch.cat([
            self.face_uvcoords.expand(batch_size, -1, -1, -1), 
            transformed_face_normals.detach(), 
            face_verts.detach(), 
            face_normals], -1,
        )
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        
        ####
        # vis mask
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        uvcoords_images = rendering[:, :3, :, :]
        grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
        albedo_images = F.grid_sample(albedos, grid, align_corners=False)

        # visible mask for pixels with positive normal direction
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < -0.05).float()

        # shading
        normal_images = rendering[:, 9:12, :, :]
        if lights is not None:
            if lights.shape[1] == 9:
                shading_images = self.add_SHlight(normal_images, lights)
            else:
                if light_type=='point':
                    vertice_images = rendering[:, 6:9, :, :].detach()
                    shading = self.add_pointlight(
                        vertice_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), 
                        normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), 
                        lights,
                    )
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
                else:
                    shading = self.add_directionlight(
                        normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), 
                        lights,
                    )
                    shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2)
            images = albedo_images*shading_images
        else:
            images = albedo_images
            shading_images = images.detach()*0.

        outputs = {
            'images': images*alpha_images,
            'albedo_images': albedo_images*alpha_images,
            'alpha_images': alpha_images,
            'pos_mask': pos_mask,
            'shading_images': shading_images,
            'grid': grid,
            'normals': normals,
            'normal_images': normal_images*alpha_images,
            'transformed_normals': transformed_normals,
        }
        
        return outputs

    def add_SHlight(self, normal_images, gamma, init_lit):
        '''
            sh_coeff: [bz, 9, 3]
        '''
        batch_size = gamma.shape[0]
        gamma = gamma.reshape([batch_size, 3, 9])
        gamma = gamma + init_lit
        sh_coeff = gamma.permute(0, 2, 1)

        N = normal_images
        sh = torch.stack([
                N[:,0]*0.+1., N[:,0], N[:,1], \
                N[:,2], N[:,0]*N[:,1], N[:,0]*N[:,2], 
                N[:,1]*N[:,2], N[:,0]**2 - N[:,1]**2, 3*(N[:,2]**2) - 1
                ], 
                1) # [bz, 9, h, w]
        sh = sh*self.constant_factor[None,:,None,None]
        shading = torch.sum(sh_coeff[:,:,:,None,None]*sh[:,:,None,:,:], 1) # [bz, 9, 3, h, w]  
        return shading

    def add_pointlight(self, vertices, normals, lights):
        '''
            vertices: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_positions = lights[:,:,:3]
        light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_positions[:,:,None,:] - vertices[:,None,:,:], dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def add_directionlight(self, normals, lights):
        '''
            normals: [bz, nv, 3]
            lights: [bz, nlight, 6]
        returns:
            shading: [bz, nv, 3]
        '''
        light_direction = lights[:,:,:3]
        light_intensities = lights[:,:,3:]
        directions_to_lights = F.normalize(light_direction[:,:,None,:].expand(-1,-1,normals.shape[1],-1), dim=3)
        # normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        # normals_dot_lights = (normals[:,None,:,:]*directions_to_lights).sum(dim=3)
        normals_dot_lights = torch.clamp((normals[:,None,:,:]*directions_to_lights).sum(dim=3), 0., 1.)
        shading = normals_dot_lights[:,:,:,None]*light_intensities[:,:,None,:]
        return shading.mean(1)

    def render_shape(self, 
        vertices, transformed_vertices, colors=None, images=None, detail_normal_images=None, 
        lights=None, return_grid=False, uv_detail_normals=None, h=None, w=None,
    ):
        '''
        -- rendering shape with detail normal map
        '''
        batch_size = vertices.shape[0]
        # set lighting
        if lights is None:
            light_positions = torch.tensor(
                [
                [-1,1,1],
                [1,1,1],
                [-1,-1,1],
                [1,-1,1],
                [0,0,1]
                ]
            )[None,:,:].expand(batch_size, -1, -1).float()
            light_intensities = torch.ones_like(light_positions).float()*1.7
            lights = torch.cat((light_positions, light_intensities), 2).to(vertices.device)
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10

        # Attributes
        face_verts = face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        normals = vertex_normals(vertices, self.faces.expand(batch_size, -1, -1))
        face_normals = face_vertices(normals, self.faces.expand(batch_size, -1, -1))
        transformed_normals = vertex_normals(transformed_vertices, self.faces.expand(batch_size, -1, -1))
        transformed_face_normals = face_vertices(transformed_normals, self.faces.expand(batch_size, -1, -1))
        if colors is None:
            colors = self.face_colors.expand(batch_size, -1, -1, -1)
        attributes = torch.cat([
            colors, 
            transformed_face_normals.detach(), 
            face_verts.detach(), 
            face_normals,
            self.face_uvcoords.expand(batch_size, -1, -1, -1)], -1,
        )
        # rasterize
        # import ipdb; ipdb.set_trace()
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes, h, w)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()

        # albedo
        albedo_images = rendering[:, :3, :, :]
        # mask
        transformed_normal_map = rendering[:, 3:6, :, :].detach()
        pos_mask = (transformed_normal_map[:, 2:, :, :] < 0.15).float()

        # shading
        normal_images = rendering[:, 9:12, :, :].detach()
        vertice_images = rendering[:, 6:9, :, :].detach()
        if detail_normal_images is not None:
            normal_images = detail_normal_images

        shading = self.add_directionlight(normal_images.permute(0,2,3,1).reshape([batch_size, -1, 3]), lights)
        shading_images = shading.reshape([batch_size, albedo_images.shape[2], albedo_images.shape[3], 3]).permute(0,3,1,2).contiguous()        
        shaded_images = albedo_images*shading_images

        alpha_images = alpha_images*pos_mask
        if images is None:
            shape_images = shaded_images*alpha_images + torch.zeros_like(shaded_images).to(vertices.device)*(1-alpha_images)
        else:
            shape_images = shaded_images*alpha_images + images*(1-alpha_images)
        if return_grid:
            uvcoords_images = rendering[:, 12:15, :, :]
            grid = (uvcoords_images).permute(0, 2, 3, 1)[:, :, :, :2]
            return shape_images, normal_images, grid, alpha_images
        else:
            return shape_images
    
    def render_depth(self, transformed_vertices):
        '''
        -- rendering depth
        '''
        batch_size = transformed_vertices.shape[0]

        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] - transformed_vertices[:,:,2].min()
        z = -transformed_vertices[:,:,2:].repeat(1,1,3).clone()
        z = z-z.min()
        z = z/z.max()
        # Attributes
        attributes = face_vertices(z, self.faces.expand(batch_size, -1, -1))
        # rasterize
        transformed_vertices[:,:,2] = transformed_vertices[:,:,2] + 10
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)

        ####
        alpha_images = rendering[:, -1, :, :][:, None, :, :].detach()
        depth_images = rendering[:, :1, :, :]
        return depth_images
    
    def render_colors(self, transformed_vertices, colors):
        '''
        -- rendering colors: could be rgb color/ normals, etc
            colors: [bz, num of vertices, 3]
        '''
        batch_size = colors.shape[0]

        # Attributes
        attributes = face_vertices(colors, self.faces.expand(batch_size, -1, -1))
        # rasterize
        rendering = self.rasterizer(transformed_vertices, self.faces.expand(batch_size, -1, -1), attributes)
        ####
        alpha_images = rendering[:, [-1], :, :].detach()
        images = rendering[:, :3, :, :]* alpha_images
        return images

    def world2uv(self, vertices):
        '''
            warp vertices from world space to uv space
            vertices: [B, V, 3]
            face_vertices: [B, F, 3, 3]
            uv_vertices: [B, 3, H, W]
        '''
        batch_size = vertices.shape[0]
        face_verts = face_vertices(vertices, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(
            self.uvcoords.expand(batch_size, -1, -1), 
            self.uvfaces.expand(batch_size, -1, -1), 
            face_verts,
        )[:, :3]

        return uv_vertices
    


if __name__ == '__main__':

    import os
    import sys
    import os.path as osp
    import numpy as np
    import torch
    import trimesh
    import matplotlib.pyplot as plt
    import polyscope as ps

    from models.hackhead import *



    device = 'cuda'

    base_mesh_path = '/home/ubuntu/MyExp/TRACK/data/hack_data/normalized_hack_template_uv.obj'

    expr_mesh_path = '/media/ubuntu/SSD/Hallo3_preprocessed/010968_255/hack_mesh/frame_000_hack_stage1.obj'

    # hack_params_path = '/media/ubuntu/xb/nersemble_data/PHACK_output/030_FREE/cam_222200037_021_hack_stage1.npz'
    # hack_params = np.load(hack_params_path)

    # shape_params = hack_params['shape_params']  # [200,]
    # expression_params = hack_params['expression_params']  # [55,]
    # neck_poses = hack_params['neck_poses']  # [8, 3]

    # hackhead = HACKHead(use_teeth=False, device=device).to(device)

    # result = hackhead(
    #     shape=torch.tensor(shape_params)[None].to(device), 
    #     expression=torch.tensor(expression_params)[None].to(device), 
    #     neck_pose=torch.tensor(neck_poses)[None].to(device), 
    #     tau=torch.zeros(1, 1).to(device),
    #     gamma=torch.ones(1, 1).to(device),
    #     return_lmks_mp=True, 
    #     return_lmks68=True,
    # )
    # verts = result[0].detach().cpu().squeeze().numpy()
    # faces = hackhead.temp_mesh.fvs

    # uv_coords = hackhead.temp_mesh.vts          # [V_uv, 2]
    # uv_faces = hackhead.temp_mesh.fvts          # [F_uv, 3]
    # mtl_name = None
    # mesh_info = {
    #     'v': verts,
    #     'vt': uv_coords,
    #     'fv': faces,
    #     'fvt': uv_faces,
    #     'mtl_name': mtl_name,
    # }

    # save_hack_mesh_path = hack_params_path.replace('.npz', '.obj')

    # write_mesh_obj(mesh_info=mesh_info, file_path=save_hack_mesh_path)


    # render = SRenderY(
    #     image_size=224, 
    #     obj_filename=base_mesh_path, 
    #     # obj_filename=save_hack_mesh_path, 
    #     uv_size=256, rasterizer_type='pytorch3d',
    # ).to(device)

    # mesh_info = read_obj(base_mesh_path)
    # # mesh_info = read_obj(save_hack_mesh_path)
    # verts = mesh_info.vs
    # faces = mesh_info.fvs

    # # v_normals = compute_v_normals(torch.tensor(verts)[None].to(device), torch.tensor(faces).to(device))
    # v_normals = vertex_normals(torch.tensor(verts)[None].to(device), torch.tensor(faces)[None].to(device))

    # position_map = render.world2uv(
    #     # torch.tensor(verts)[None].to(device),
    #     v_normals,
    # )

    # print(position_map.shape)  # [1, 3, 256, 256], float (-1.0, 1.0)

    # p_map = position_map * 0.5 + 0.5

    # plt.imshow(p_map[0].permute(1,2,0).detach().cpu().numpy())


    def get_position_map(image_size, obj_file, uv_size, device):
        render = SRenderY(
            image_size=224, 
            obj_filename=base_mesh_path, 
            # obj_filename=save_hack_mesh_path, 
            uv_size=256, rasterizer_type='pytorch3d',
        ).to(device)

        mesh_info = read_obj(obj_file)
        verts = mesh_info.vs
        faces = mesh_info.fvs

        v_normals = vertex_normals(torch.tensor(verts)[None].to(device), torch.tensor(faces)[None].to(device))

        position_map = render.world2uv(
            # torch.tensor(verts)[None].to(device),
            v_normals,
        )

        return position_map


    p_map_0 = get_position_map(image_size=224, obj_file=base_mesh_path, uv_size=256, device=device)
    p_map_0 = p_map_0 * 0.5 + 0.5

    p_map_1 = get_position_map(image_size=224, obj_file=expr_mesh_path, uv_size=256, device=device)
    p_map_1 = p_map_1 * 0.5 + 0.5

    err_pmap = torch.abs(p_map_0 - p_map_1)

    err_pmap = (err_pmap - err_pmap.min()) / (err_pmap.max() - err_pmap.min())

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 3, 1)
    plt.title('original')
    plt.imshow(p_map_0[0].permute(1,2,0).detach().cpu().numpy())
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('hacked')
    plt.imshow(p_map_1[0].permute(1,2,0).detach().cpu().numpy())
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('error')
    plt.imshow(err_pmap[0].permute(1,2,0).detach().cpu().numpy())
    plt.axis('off')

    plt.show()


