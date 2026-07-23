import numpy as np
import pickle
import trimesh
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import roma
import cv2

from pathlib import Path
from pytorch3d.io import load_obj
from pytorch3d.structures.meshes import Meshes
from matplotlib import cm
from collections import defaultdict
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))


from meshes.common import to_np, to_tensor, BufferContainer
from meshes.hack_masks import *
from utils.log import get_logger
from utils.mesh import read_obj, Obj
from configs.env_paths import (
    HACK_DATA_PATH,
    HACK_TEMPLATE_TRI_PATH,
    HACK_TEMPLATE_QUAD_PATH,
    HACK_template_larynx_PATH,
    HACK_template_bones_PATH,
    HACK_SHAPE_PATH,
    HACK_BLENDSHAPE_PATH,
    HACK_EXPRESSION_PATH,
    HACK_Lc_PATH,
    HACK_POSE_PATH,
    HACK_WEIGHT_PATH,
    HACK_LMKS3D_PATH,
    HACK_LMKS68_PATH,
    HACK_SCALE_INFO,
)

eyeball_lmks_ids = [473, 474, 476, 468, 469, 471]
valid_lmks_ids = [i for i in range(478) if i not in eyeball_lmks_ids]

bones = json.load(open(HACK_template_bones_PATH))
bone_names = list(bones.keys())

for name in bones:
    bone = bones[name]
    parent = bone["parent"]

    L2P_rotation = np.array(bone["matrix"])
    head_in_p = (np.array(bone["head"]) + ([0, bones[parent]["length"], 0] if parent is not None else 0))
    L2P_transformation = np.identity(4)
    L2P_transformation[:3, :3] = L2P_rotation
    L2P_transformation[:3, 3] = head_in_p

    bone["L2P_transformation"] = torch.tensor(L2P_transformation, dtype=torch.float32)


def update_L2W_transformation(name):
    bone = bones[name]

    transformation_name = "L2W_transformation"
    local_pose_transformation = torch.eye(4)[None]

    if transformation_name in bone:
        return bone[transformation_name]

    parent = bone["parent"]

    L2W_transformation = bone["L2P_transformation"] @ local_pose_transformation

    if parent is not None:
        L2W_transformation = update_L2W_transformation(parent) @ L2W_transformation

    bone[transformation_name] = L2W_transformation

    return L2W_transformation


def update_L2W_transformation_pose(L2W_transformation_pose, L2P_transformation, ith_bone, pose_matrix):
    """
    L2W_transformation_pose: list of N_bones
    """
    name = bone_names[ith_bone]
    bone = bones[name]

    if L2W_transformation_pose[ith_bone] is not None:
        return L2W_transformation_pose[ith_bone]

    local_pose_transformation = torch.eye(
        4, device=pose_matrix.device)[None].repeat(pose_matrix.shape[0], 1, 1)
    local_pose_transformation[:, :3, :3] = pose_matrix[:, bone_names.index(name)]

    L2W_transformation = L2P_transformation[ith_bone] @ local_pose_transformation

    parent = bone["parent"]
    if parent is not None:
        ith_parent = bone_names.index(parent)
        L2W_transformation = update_L2W_transformation_pose(
            L2W_transformation_pose, L2P_transformation, ith_parent, pose_matrix,
        ) @ L2W_transformation

    L2W_transformation_pose[ith_bone] = L2W_transformation
    return L2W_transformation_pose[ith_bone]


N_bones = 8

# [N_bones, 4, 4]
L2P_transformation = torch.stack([bones[bone_names[i]]["L2P_transformation"] for i in range(len(bone_names))])

L2W_transformation = torch.zeros(1, N_bones, 4, 4)  # [1, Nb, 4, 4]
for name in bones:
    L2W_transformation[:, bone_names.index(name)] = update_L2W_transformation(name)

W2L_transformation = torch.linalg.inv(L2W_transformation)




class PCA(nn.Module):
    def __init__(self, mean, diff):
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean[None]).to(torch.float32))
        self.register_buffer("diff", torch.tensor(diff[None]).to(torch.float32))

    def forward(self, a=None, clip=999):
        if a is None:
            return self.mean
        m = len(self.diff.shape)
        return self.mean + (a.reshape([a.shape[0], a.shape[1]]+[1]*(m-2)) * self.diff)[:, :clip].sum(dim=1)


def load_pca(path):
    pca = np.load(path, allow_pickle=True).item()
    mean = pca["mean"]    # [V, 3] or [..., V, 3]
    VT_std = pca["VT_std"]   # [..., V, 3]
    pca = PCA(mean, VT_std)
    return pca




def uv1d_construct_delta(obj_template: Obj, uv1d, tau):
    """
    uv1d: [1, 1, 256, 256]
    tau: [B, 1]
    return: [B, 14062, 1]

    Cached ``grid`` is stored on ``uv1d_construct_delta`` for speed. If the same
    process later uses another CUDA device (e.g. multiprocessing worker running
    successive jobs on cuda:0 then cuda:1), move the cache to ``uv1d.device``
    instead of mixing devices.
    """

    grid = getattr(uv1d_construct_delta, "grid", None)
    if grid is None:
        obj = obj_template
        uv = obj.vts
        uv[:, 1] = 1 - uv[:, 1]
        uv = uv * 2 - 1
        fv = obj.fvs
        fvt = obj.fvts
        assert fv.shape[-1] == 4, "must be quad mesh"
        grid = np.ones((1, 1, 14062, 2)) * 2
        for i in range(len(fv)):
            for j in range(4):
                if grid[0][0][fv[i][j]][0] == 2:
                    grid[0][0][fv[i][j]] = uv[fvt[i][j]]
                else:
                    continue

        grid = torch.tensor(grid).to(uv1d)
        setattr(uv1d_construct_delta, "grid", grid)
    else:
        grid = grid.to(device=uv1d.device, dtype=uv1d.dtype)
        setattr(uv1d_construct_delta, "grid", grid)

    grid = grid + F.pad(tau * 2, [1, 0])[:, None, None, :]

    output = torch.nn.functional.grid_sample(
        uv1d.expand(grid.shape[0], -1, -1, -1), grid, 
        mode='bilinear', padding_mode="border", align_corners=True,
    )
    return output[:, 0, 0, :, None]


def vertices2landmarks(vertices, faces, lmk_faces_indices, lmk_bary_coords):
    """Calculates landmarks by barycentric interpolation

    Parameters
    ----------
    vertices: torch.tensor [B,V,3], dtype = torch.float32
        The tensor of input vertices
    faces: torch.tensor [F,3], dtype = torch.long
        The faces of the mesh
    lmk_faces_idx: torch.tensor [L], dtype = torch.long
        The tensor with the indices of the faces used to calculate the
        landmarks.
    lmk_bary_coords: torch.tensor [L,3], dtype = torch.float32
        The tensor of barycentric coordinates that are used to interpolate
        the landmarks

    Returns
    -------
    landmarks: torch.tensor [B,L,3], dtype = torch.float32
        The coordinates of the landmarks for each mesh in the batch
    """
    # Extract the indices of the vertices for each face
    # [B,L,3]
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    # lmk_faces_verts_indices = torch.index_select(
    #     faces, 0, lmk_faces_indices.view(-1),
    # ).view(1, -1, 3).repeat(batch_size, 1, 1)

    # lmk_vertices = vertices.view(-1, 3)[lmk_faces_verts_indices].view(batch_size, -1, 3, 3)
    
    lmk_faces = torch.index_select(faces, 0, lmk_faces_indices.view(-1)).view(batch_size, -1, 3)

    lmk_faces += torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(batch_size, -1, 3, 3)
    landmarks = torch.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])

    return landmarks



class HACKHead(nn.Module):
    """
    Given HACK parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks

    几何与 ``faces``/UV 以三角模板 ``hack_template_tri_path`` 为准；四边面模板 ``hack_template_quad_path`` 的 UV
    与三角模板可不一致。将 ``register_quad_face_indices=True`` 时，额外注册与三角网格**同一顶点表**对齐的
    四边面面索引 ``faces_quad``、四边面 ``vt`` 表 ``uv_coords_quad``、以及每面每角在 ``uv_coords_quad`` 中的
    索引 ``uv_face_indices_quad``（形状 ``[F_quad, 4]``）。
    """
    def __init__(self,
        n_shape_params=200, 
        n_expr_params=55,
        hack_template_tri_path=HACK_TEMPLATE_TRI_PATH,
        hack_template_quad_path=HACK_TEMPLATE_QUAD_PATH,
        hack_shape_path=HACK_SHAPE_PATH,
        hack_expr_path=HACK_EXPRESSION_PATH,
        hack_pose_path=HACK_POSE_PATH,
        hack_weight_path=HACK_WEIGHT_PATH,
        hack_blend_path=HACK_BLENDSHAPE_PATH,
        hack_template_larynx_path=HACK_template_larynx_PATH,
        hack_lmks3d_path=HACK_LMKS3D_PATH,
        hack_lmks68_path=HACK_LMKS68_PATH,
        hack_lc_path=HACK_Lc_PATH,
        hack_scale_info=HACK_SCALE_INFO,
        include_mask=True,
        include_lbs_color=False,
        use_teeth=False,
        connect_lip_inside=False,
        remove_lip_inside=False,
        disable_deformation_on_torso=False,
        remove_torso=False,
        face_clusters=[],
        register_quad_face_indices=False,
        device="cuda",
    ):
        super().__init__()

        # logger.info("Initializing HACK mesh model...")
        
        self.n_shape_params = n_shape_params
        self.n_expr_params = n_expr_params

        self.dtype = torch.float32
        self.device = device

        self.N_bones = 8

        # template mesh
        # self.temp_mesh = trimesh.load(hack_template_path)
        self.temp_mesh = read_obj(hack_template_tri_path)
        self.temp_quad_mesh = read_obj(hack_template_quad_path)
        self.register_buffer("base_vertices", to_tensor(self.temp_mesh.vs, dtype=self.dtype))
        self.register_buffer("faces", to_tensor(self.temp_mesh.fvs, dtype=torch.long))

        self.register_buffer("uv_coords", to_tensor(self.temp_mesh.vts), persistent=False)
        self.register_buffer(
            "uv_indices", to_tensor(self.temp_mesh.fvts, dtype=torch.int), persistent=False,
        )

        # 四边面：与三角相同几何顶点下，来自四边面 OBJ 的每面 4 顶点索引（及每角在 quad vt 表中的 UV 角点索引）
        self._register_quad_face_indices = bool(register_quad_face_indices)
        if self._register_quad_face_indices:
            fqv = self.temp_quad_mesh.fvs
            fqvt = self.temp_quad_mesh.fvts
            n_v = int(self.base_vertices.shape[0])
            if fqv is None or len(fqv) == 0:
                raise ValueError("register_quad_face_indices=True 但四边面模板中无面 f")
            fqv = np.asarray(fqv, dtype=np.int64)
            if fqv.ndim != 2 or fqv.shape[1] != 4:
                raise ValueError(
                    f"四边面模板须为四边形面 [F,4]，当前 fvs 形状 {getattr(fqv, 'shape', None)}；"
                    "若 OBJ 为三角或混合，请使用仅含 f 1/1 2/2 3/3 4/4 的网格。"
                )
            if int(fqv.min()) < 0 or int(fqv.max()) >= n_v:
                raise ValueError(
                    f"faces_quad 顶点下标越界: 需落在 [0, {n_v})（与三角模板 v 行数一致）。"
                )
            if not np.array_equal(
                self.temp_mesh.vs.shape, self.temp_quad_mesh.vs.shape,
            ):
                raise ValueError(
                    f"三角与四边面模板须含相同行数的 v 顶点，当前 tri vs {self.temp_mesh.vs.shape} vs quad vs {self.temp_quad_mesh.vs.shape}。"
                )
            fqvt = np.asarray(fqvt, dtype=np.int64)
            if fqvt.shape != fqv.shape:
                raise ValueError(
                    f"四边面 f 与 fvt/角点数不一致: fvs {fqv.shape} fvts {fqvt.shape}"
                )
            self.register_buffer("faces_quad", to_tensor(fqv, dtype=torch.long))
            self.register_buffer(
                "uv_coords_quad", to_tensor(self.temp_quad_mesh.vts), persistent=False,
            )
            self.register_buffer(
                "uv_face_indices_quad", to_tensor(fqvt, dtype=torch.int), persistent=False,
            )

        scale_info = np.load(hack_scale_info)
        self.scale_info = {
            'center': torch.tensor(scale_info['center_tri'], dtype=self.dtype).to(self.device),
            # 'center': torch.zeros(3, dtype=self.dtype).to(self.device),
            'scale': torch.tensor(scale_info['scale_tri'], dtype=self.dtype).to(self.device),
        }

        self.S = load_pca(hack_shape_path).to(self.device)
        self.E = load_pca(hack_expr_path).to(self.device)
        self.P = load_pca(hack_pose_path).to(self.device)

        # pose parameters
        # P = torch.zeros(self.N_bones, 3, 3, len(self.base_vertices), 3, dtype=self.dtype)
        # self.register_buffer("P", P)  # [8, 3, 3, V, 3]

        # skinning weights
        weights = to_tensor(np.load(hack_weight_path), dtype=self.dtype)
        weights = weights / weights.sum(axis=0, keepdims=True)   # [8, V]
        self.register_buffer("weights", weights, persistent=False)

        # blendshapes, [56, V, 3]
        blendshapes = torch.tensor(np.load(hack_blend_path), dtype=self.dtype)  
        neutral = blendshapes[:1]
        blendshapes = blendshapes[1:] - neutral  # [55, V, 3]
        self.register_buffer("blendshapes", blendshapes, persistent=False)

        L = torch.tensor(
            cv2.imread(hack_lc_path, cv2.IMREAD_GRAYSCALE)/255, 
            dtype=torch.float32)[None, None]  # [1, 1, 256, 256]
        self.register_buffer("L", L, persistent=False)

        # larynx
        ts = torch.tensor(np.load(hack_template_larynx_path), dtype=self.dtype)  # [3]
        self.register_buffer("ts", ts, persistent=False)

        # landmarks
        self.hack_lmks3d = np.load(hack_lmks3d_path)
        self.hack_lmks68 = np.load(hack_lmks68_path)

        self.register_buffer("L2P_transformation", L2P_transformation, persistent=False)
        self.register_buffer("W2L_transformation", W2L_transformation, persistent=False)

        if include_mask:
            self.mask = HACKMask(vertices=self.base_vertices, faces=self.faces)

        if use_teeth:
            self.use_teeth()

        # laplacian matrix
        laplacian_matrix = Meshes(
            verts=[self.base_vertices], faces=[self.faces],
        ).laplacian_packed().to_dense()  # [V, V]
        self.register_buffer("laplacian_matrix", laplacian_matrix, persistent=False)

        D = torch.diag(laplacian_matrix)
        laplacian_matrix_negate_diag = laplacian_matrix - torch.diag(D) * 2
        self.register_buffer("laplacian_matrix_negate_diag", laplacian_matrix_negate_diag, persistent=False)

    def get_L_tau(self, tau):
        """
        tau > 0 means upper
        """
        dist = uv1d_construct_delta(self.temp_quad_mesh, self.L, tau)
        L_tau = dist * self.ts
        return L_tau
    
    def use_teeth(self,):
        self.mask.update(self.faces, self.textures_idx)
    
    def normalize_head(self, vertices):
        vertices = (vertices - self.scale_info['center']) * self.scale_info['scale']
        return vertices

    def forward(self, 
        shape, expression, neck_pose, tau, gamma, return_lmks_mp=True, normalize=True,
    ):
        """
        input:
            shape parameters: [B, 200], torch.tensor
            expression parameters: [B, 55], torch.tensor
            neck pose parameters: [B, 8, 3], torch.tensor
        return:
            vertices: [B, V, 3]], torch.tensor
            landmarks: [N, #landmarks, 3]], torch.tensor

        """
        batch_size = shape.shape[0]
        theta_matrix = roma.rotvec_to_rotmat(neck_pose)  # [B, 8, 3, 3]
        theta_matrix_zero = theta_matrix - torch.cat([
            theta_matrix[:, :1], 
            (torch.eye(3).to(neck_pose)[None, None]).expand(batch_size, N_bones-1, 3, 3),
        ], dim=1)

        P_theta = (theta_matrix_zero[:, :, :, :, None, None] * self.P()).sum(dim=(1, 2, 3))  # [B, V, 3]
        L2W_transformation_pose = [None] * N_bones
        for ith_bone in range(len(bone_names)):
            update_L2W_transformation_pose(
                L2W_transformation_pose, self.L2P_transformation, ith_bone, theta_matrix,
            )
        L2W_transformation_pose = torch.stack(L2W_transformation_pose, dim=1)  # [B, Nb, 4, 4]

        W2L2pWs = L2W_transformation_pose @ self.W2L_transformation  # [B, Nb, 4, 4]
        W2L2pW_weighted = (W2L2pWs[:, :, None, :, :] * self.weights[None, :, :, None, None]).sum(axis=1)  # [B, 14062, 4, 4]


        # shape_pca = np.load(HACK_SHAPE_PATH, allow_pickle=True).item()
        # shape_pca_mean = shape_pca['mean']
        # shape_pca_std = shape_pca['VT_std']
        # T = torch.tensor(shape_pca_mean, dtype=self.dtype, device=self.device)[None].repeat(batch_size,1,1) \
        #     + (torch.tensor(shape_pca_std, dtype=self.dtype, device=self.device) * shape[:,:,None,None]).sum(dim=1)

        T = self.S(shape)  # [B, V, 3]
        T_theta = T + P_theta + (self.blendshapes * expression[:, :, None, None]).sum(dim=1) \
                    + self.get_L_tau(tau) * gamma[:, :, None]   # [B, V, 3]
        vertices = (W2L2pW_weighted @ F.pad(T_theta, [0, 1], value=1)[:, :, :, None])[:, :, :3, 0]  # [B, 14062, 3]

        # vertices = T

        if normalize:
            vertices = self.normalize_head(vertices)
            
        ret_vals = [vertices]
        if return_lmks_mp:
            lmk_faces_indices = torch.tensor(self.hack_lmks3d['hack_indices']).to(self.device)
            lmk_bary_coords = torch.tensor(self.hack_lmks3d['hack_bary_coords'], dtype=self.dtype).to(self.device)
            landmarks = vertices2landmarks(
                vertices=vertices,
                faces=self.faces,
                lmk_faces_indices=lmk_faces_indices.repeat(batch_size, 1),
                lmk_bary_coords=lmk_bary_coords.repeat(batch_size, 1, 1),
            )
            # Match landmarks dtype: vertices may be fp32 after normalize_head while
            # einsum in vertices2landmarks can still be fp16 under autocast.
            all_landmarks = -torch.ones(
                (batch_size, 478, 3), device=self.device, dtype=landmarks.dtype,
            )
            all_landmarks[:, valid_lmks_ids, :] = landmarks
            ret_vals.append(all_landmarks)

        return ret_vals
    

class HACKMask(nn.Module):
    def __init__(self,
        vertices, faces, faces_t=None, face_clusters=[], device="cuda",
    ):
        super().__init__()
        self.device = device
        self.vertices = vertices.to(device)
        self.faces = faces.to(device)
        
        self.faces_t = faces_t
        self.face_clusters = face_clusters

        self.f = BufferContainer()
        self.v = BufferContainer()
        self.create_custom_mask()

        # if faces is not None:
        #     self.num_faces = faces.shape[0]
        # else:
        #     self.num_faces = num_faces
        
        # """ Available part masks from the FLAME model: 
        #     face, neck, scalp, boundary, right_eyeball, left_eyeball, 
        #     right_ear, left_ear, forehead, eye_region, nose, lips,
        #     right_eye_region, left_eye_region.
        # """
        # patch_masks = np.load(flame_masks_path, allow_pickle=True, encoding='latin1') 

        # self.v = BufferContainer()
        # for k, v_mask in patch_masks.items():
        #     self.v.register_buffer(k, torch.tensor(v_mask, dtype=torch.long))
        
        # self.create_custom_mask()

        # if self.faces is not None:
        #     self.construct_vid_table()
        #     self.process_face_mask(self.faces)
        #     self.process_face_clusters(self.face_clusters)
        #     if self.faces_t is not None:
        #         self.process_vt_mask(self.faces, self.faces_t)


    def create_custom_mask(self):
        """ Create some custom masks.
        """
        eye_mask = torch.cat([torch.tensor(left_eye_mask), torch.tensor(right_eye_mask)])
        self.f.register_buffer("boundary", torch.tensor(boundary, dtype=torch.long))
        self.f.register_buffer("chin", torch.tensor(chin, dtype=torch.long))
        self.f.register_buffer("left_ear", torch.tensor(leftear, dtype=torch.long))
        self.f.register_buffer("right_ear", torch.tensor(rightear, dtype=torch.long))
        self.f.register_buffer("eye_region", torch.tensor(eyeregion, dtype=torch.long))
        self.f.register_buffer("left_eyelid", torch.tensor(left_eyelid, dtype=torch.long))
        self.f.register_buffer("right_eyelid", torch.tensor(right_eyelid, dtype=torch.long))
        self.f.register_buffer("eye_mask", torch.tensor(eye_mask, dtype=torch.long))
        self.f.register_buffer("mouth_mask", torch.tensor(mouth_mask, dtype=torch.long))
        self.f.register_buffer("forehead", torch.tensor(forehead, dtype=torch.long))
        self.f.register_buffer("hair", torch.tensor(hair, dtype=torch.long))
        self.f.register_buffer("face", torch.tensor(face, dtype=torch.long))
        self.f.register_buffer("deform", torch.tensor(deform_fids, dtype=torch.long))
        self.f.register_buffer("face_dfn", torch.tensor(face_dfn, dtype=torch.long))
        self.f.register_buffer("neck", torch.tensor(neck, dtype=torch.long))
        self.f.register_buffer("nose", torch.tensor(nose, dtype=torch.long))
        self.f.register_buffer("mouth", torch.tensor(mouth, dtype=torch.long))
        self.f.register_buffer("lips", torch.tensor(lips, dtype=torch.long))
        self.f.register_buffer("lips_tight", torch.tensor(lips_tight, dtype=torch.long))
        self.f.register_buffer("lipread", torch.tensor(lipread, dtype=torch.long))
        self.f.register_buffer("skin", torch.tensor(skin, dtype=torch.long))

        # remove the intersection with neck from scalp and get the region for hair
        face_and_neck = torch.cat([self.f.face, self.f.neck]).unique()

        # unions
        self.f.register_buffer("ears", torch.cat([self.f.right_ear, self.f.left_ear]))
        self.f.register_buffer("eyelids", torch.cat([self.f.left_eyelid, self.f.right_eyelid]))

        # # skin
        # skin_except = ["hair", "lips_tight", "lips"]
        # if self.num_verts == 14062:
        #     skin_except.append("teeth")
        # skin = self.get_fid_except_region(skin_except)
        # self.f.register_buffer("skin", skin)
        
        self.update_v()

        self.f.to(self.device)
        self.v.to(self.device)

    def update_v(self):
        for region_name, face_ids in self.f.items():
            vert_ids = torch.unique(self.faces[face_ids].flatten())
            self.v.register_buffer(region_name, vert_ids)

    
    def get_vid_by_region(self, regions, keep_order=False):
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            vid = torch.cat([self.v.get_buffer(k) for k in regions])
            if keep_order:
                return vid
            else:
                return vid.unique()
        else:
            return torch.tensor([], dtype=torch.long)
    
    def get_vid_except_region(self, regions):
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            indices = torch.cat([self.v.get_buffer(k) for k in regions]).unique()
        else:
            indices = torch.tensor([], dtype=torch.long)

        # get the vertex indicies that are not included by regions
        vert_idx = torch.arange(0, self.num_verts, device=indices.device)
        combined = torch.cat((indices, vert_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]

    def get_fid_by_region(self, regions):
        """Get face indicies by regions"""
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            return torch.cat([self.f.get_buffer(k) for k in regions]).unique()
        else:
            return torch.tensor([], dtype=torch.long)
    
    def get_fid_except_region(self, regions):
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            indices = torch.cat([self.f.get_buffer(k) for k in regions]).unique()
        else:
            indices = torch.tensor([], dtype=torch.long)

        # get the face indicies that are not included by regions
        face_idx = torch.arange(0, self.num_faces, device=indices.device)
        combined = torch.cat((indices, face_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]
    
    def get_fid_except_fids(self, fids):
        # get the face indicies that are not included
        face_idx = torch.arange(0, self.num_faces, device=fids.device)
        combined = torch.cat((fids, face_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]
    
    def construct_vid_table(self):
        self.vid_to_region = defaultdict(list)  # vertex id -> region name
        for region_name, v_mask in self.v:
            for v_id in v_mask:
                self.vid_to_region[v_id.item()].append(region_name)
        self.vid_to_region = defaultdict(list)  # vertex id -> region name
        for region_name, v_mask in self.v:
            for v_id in v_mask:
                self.vid_to_region[v_id.item()].append(region_name)
    
    def process_face_mask(self, faces):
        face_masks = defaultdict(list)  # region name -> face id
        for f_id, f in enumerate(faces):
            counters = defaultdict(int)
            for v_id in f:
                for region_name in self.vid_to_region[v_id.item()]:
                    counters[region_name] += 1
            
            for region_name, count in counters.items():
                if count >= 3:  # create straight boundaries, with seams
                # if count > 1:  # create zigzag boundaries, no seams
                    face_masks[region_name].append(f_id)

        self.f = BufferContainer()
        for region_name, f_mask in face_masks.items():
            self.f.register_buffer(region_name, torch.tensor(f_mask, dtype=torch.long))
    
    def process_face_clusters(self, face_clusters):
        """ Construct a lookup table from face id to cluster id.
            
            cluster #0: background
            cluster #1: foreground
            cluster #2: faces in face_clusters[0]
            cluster #3: faces in face_clusters[1]
            ...
        """

        fid2cid = torch.ones(self.num_faces+1, dtype=torch.long)  # faces are always treated as foreground
        for cid, cluster in enumerate(face_clusters):
            try:
                fids = self.get_fid_by_region([cluster])
            except Exception as e:
                print(f"Ignoring unknown cluster {cluster}.")
                continue
            fid2cid[fids] = cid + 2  # reserve cluster #0 for the background and #1 for faces that do not belong to any cluster
        self.register_buffer("fid2cid", fid2cid)
    
    def process_vt_mask(self, faces, faces_t):
        vt_masks = defaultdict(list)  # region name -> vt id
        for f_id, (face, face_t) in enumerate(zip(faces, faces_t)):
            for v_id, vt_id in zip(face, face_t):
                for region_name in self.vid_to_region[v_id.item()]:
                    vt_masks[region_name].append(vt_id.item())

        self.vt = BufferContainer()
        for region_name, vt_mask in vt_masks.items():
            self.vt.register_buffer(region_name, torch.tensor(vt_mask, dtype=torch.long))
    
    def update(self, faces=None, faces_t=None, face_clusters=None):
        """Update the faces properties when vertex masks are changed"""
        if faces is not None:
            self.faces = faces
            self.num_faces = faces.shape[0]
        if faces_t is not None:
            self.faces_t = faces_t
        if face_clusters is not None:
            self.face_clusters = face_clusters
        
        self.construct_vid_table()
        self.process_face_mask(self.faces)
        self.process_face_clusters(self.face_clusters)
        if self.faces_t is not None:
            self.process_vt_mask(self.faces, self.faces_t)

    def get_mask_face(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('face'), :] = 1.0
        return face_mask.to(self.device)
    
    def get_mask_deform(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('deform'), :] = 1.0
        return face_mask.to(self.device)
    
    def get_mask_ears(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('left_ear'), :] = 1.0
        face_mask[:, self.v.get_buffer('right_ear'), :] = 1.0
        return face_mask.to(self.device)
    
    def get_mask_neck(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('neck'), :] = 1.0
        face_mask[:, self.v.get_buffer('boundary'), :] = 1.0
        return face_mask.to(self.device)
    
    def get_mask_render_vid(self):
        vids = torch.hstack([
            self.v.get_buffer('face'), 
            # self.v.get_buffer('neck'), 
            # self.v.get_buffer('boundary'), 
            self.v.get_buffer('left_ear'), 
            self.v.get_buffer('right_ear'), 
        ]).cpu()
        return vids
    
    def get_mask_eyes(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('eye_mask'), :] = 1.0
        return face_mask.to(self.device)

    def get_mask_eyes_region(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('eye_region'), :] = 1.0
        return face_mask.to(self.device)

    def get_mask_mouth(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('mouth_mask'), :] = 1.0
        return face_mask.to(self.device)
    
    def get_mask_lips(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('lips'), :] = 1.0
        return face_mask.to(self.device)

    def get_mask_lipread(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('lipread'), :] = 1.0
        return face_mask.to(self.device)
    
    def get_mask_nose(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('nose'), :] = 1.0
        return face_mask.to(self.device)
    
    def get_mask_chin(self):
        face_mask = torch.zeros_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('chin'), :] = 1.0
        return face_mask.to(self.device)
    
    def get_mask_depth(self):
        face_mask = torch.ones_like(self.vertices)[None]
        face_mask[:, self.v.get_buffer('boundary'), :] = 0.0
        face_mask[:, self.v.get_buffer('left_ear'), :] = 0.0
        face_mask[:, self.v.get_buffer('right_ear'), :] = 0.0
        return face_mask.to(self.device)






if __name__=="__main__":


    # temp_mesh = read_obj(HACK_TEMPLATE_PATH)
    # print(temp_mesh.vs.shape)  # [V, 3] or [14062, 3]

    # verts, faces, aux = load_obj(HACK_TEMPLATE_PATH)
    # print(verts.shape)   # [V, 3] or [14062, 3]
    # print(faces.verts_idx.shape)   # [F, 3] or [28068, 3]
    # print(faces.textures_idx.shape)   # [F, 3] or [28068, 3]
    # print(faces.normals_idx.shape)   # [F, 3] or [28068, 3]
    # print(faces.textures_idx.shape)   # [F, 3] or [28068, 3]


    # assert np.allclose(temp_mesh.vs, verts.numpy())
    # assert np.allclose(temp_mesh.fvs, faces.verts_idx.numpy())
    

    hack = HACKHead()

    