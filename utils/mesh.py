import torch
import polyscope as ps
import numpy as np
import trimesh
import igl
import torch.nn.functional as F
from torch_scatter import scatter_add



def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d
    class C(object):
        pass
    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def generate_triangles(h, w, margin_x=2, margin_y=5, mask=None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    #.
    # w*h
    triangles = []
    for x in range(margin_x, w-1-margin_x):
        for y in range(margin_y, h-1-margin_y):
            triangle0 = [y*w + x, y*w + x + 1, (y+1)*w + x]
            triangle1 = [y*w + x + 1, (y+1)*w + x + 1, (y+1)*w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:,[0,2,1]]
    return triangles

def expand_mesh_by_uv(verts: np.ndarray, faces: np.ndarray, uv_coords: np.ndarray, uv_faces: np.ndarray,
                     return_v_indices: bool = False):
    """
    Expand (v, vt) to 1:1 vertex-UV correspondence. OBJ allows one geometric vertex
    to bind different UVs at seams, so vts count can differ from vs count.

    Args:
        verts: (V, 3)
        faces: (F, 3) vertex indices
        uv_coords: (Vt, 2)
        uv_faces: (F, 3) UV indices (same face layout as faces)
        return_v_indices: if True, also return v_indices s.t. new_verts[i] = verts[v_indices[i]]

    Returns:
        new_verts: (Vnew, 3)
        new_uv_coords: (Vnew, 2)
        new_faces: (F, 3)
        [v_indices]: (Vnew,) optional, original vertex index per new vertex
    """
    faces = np.asarray(faces, dtype=np.int64)
    uv_faces = np.asarray(uv_faces, dtype=np.int64)
    assert faces.shape == uv_faces.shape, f"faces {faces.shape} vs uv_faces {uv_faces.shape} mismatch"

    pair2new = {}
    new_verts = []
    new_uvs = []
    v_indices = [] if return_v_indices else None
    new_faces = np.empty_like(faces, dtype=np.int64)

    next_idx = 0
    for fi in range(faces.shape[0]):
        for j in range(faces.shape[1]):
            v_idx = int(faces[fi, j])
            vt_idx = int(uv_faces[fi, j])
            key = (v_idx, vt_idx)
            if key not in pair2new:
                pair2new[key] = next_idx
                new_verts.append(verts[v_idx])
                new_uvs.append(uv_coords[vt_idx])
                if return_v_indices:
                    v_indices.append(v_idx)
                next_idx += 1
            new_faces[fi, j] = pair2new[key]

    out = (np.asarray(new_verts, dtype=np.float32), np.asarray(new_uvs, dtype=np.float32), new_faces)
    if return_v_indices:
        out = out + (np.array(v_indices, dtype=np.int64),)
    return out


def calc_center(vertices, faces, mode='np'):
    """
    Args:
        vertices (torch.tensor): [B, V, 3] vertices 
        faces (torch.tensor): [F, 3] vertex index for each face
    """
    if len(vertices.size()) < 3:
        vertices = vertices.unsqueeze(0)
    fv = vertices[:, faces]
    
    if mode=='np':
        return np.mean(fv, axis=-2)
    else:
        return torch.mean(fv, dim=-2)
    
def calc_norm(mesh):
    """
        mesh(trimesh.Trimesh)
    """
    cross1 = lambda x,y:np.cross(x,y)
    fv = mesh.vertices[mesh.faces]

    span = fv[ :, 1:, :] - fv[ :, :1, :]
    norm = cross1(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[ :, np.newaxis] + 1e-8)
    norm_v = trimesh.geometry.mean_vertex_normals(mesh.vertices.shape[0], mesh.faces, norm)
    return norm_v, norm

def calc_normals(batch_v, face, at='face'):
    """
    Args:
        batch_v (torch.tensor): [B*T, V, 3] vertices for current batch
        face (torch.tensor): [F, 3] vertex index for each face
        at (str): mode 'face', 'vertex' (default: 'face')
        
    Returns:
        norm | face_norm (torch.tensor): corresponding normal vector
    """
    B_S = batch_v.shape[0]
    N_V = batch_v.shape[1]

    batch_vf = batch_v[:, face] # --> [B, F, 3, 3]
    span = batch_vf[..., 1:, :] - batch_vf[..., :1, :] # --> [B, V, 2, 3]
    cross = torch.linalg.cross(span[..., 0, :], span[..., 1, :], dim=-1) # --> [B, F, 3]
    face_norm = torch.nn.functional.normalize(cross, p=2, dim=-1)  # --> [B, F, 3]
    
    if at == 'face':
        return face_norm
    else: # at == 'vertex'
        idx = torch.cat([face[:, 0], face[:, 1], face[:, 2]], dim=0)
        face_norm = face_norm.repeat(1, 3, 1)

        norm = scatter_add(face_norm, idx, dim=1, dim_size=N_V)
        norm = torch.nn.functional.normalize(norm, p=2, dim=-1)  # [N, 3]
        return norm


def calc_span_matrix(batch_vertices, faces):
    """
    Args
        batch_vertices (torch.tensor): [B, V, 3]
        faces (torch.tensor): [F, 3]
    Return
        span (torch.tensor): [B, F, 3, 3] (v2-v1, v3-v1, v4-v1)
    """
    B_v = batch_vertices

    faces  = faces[None].repeat(B_v.shape[0], 1 ,1)
    B, num_faces  = faces.shape[:2]
    batch_indices = torch.arange(B)[:, None, None]
    batch_indices = torch.tile(batch_indices, (1, num_faces, 1))

    B_vf = B_v[batch_indices, faces].permute(0, 1, 3, 2)
    v1, v2, v3 = B_vf[..., 0], B_vf[..., 1], B_vf[..., 2]
    cross = torch.linalg.cross(v2 - v1, v3 - v1, dim=-1)
    vn = torch.nn.functional.normalize(cross, p=2, dim=-1)  # [F, 3]
    v4 = v1 + vn

    span = torch.stack((v2 - v1, v3 - v1, v4 - v1), dim=-1)
    return span

def calc_jacobian_matrix(verts, faces, template, return_torch=False):
    """Reference from Deformation Transfer for Triangle Meshes [Sumner and Popovic, 2004]
    Args
        verts (torch.tensor): [B*T, V, 3] target vertices (deformed)
        faces (torch.tensor): [F, 3]
        template (torch.tensor): [B, V, 3] source vertices (undeformed)
    Return
        Q (torch.tensor): [B, F, 3, 3] transformations (v2-v1, v3-v1, v4-v1)
    """
    B, V, _ = verts.shape

    span_matrix = calc_span_matrix(verts, faces)
    neutral_span_matrix = calc_span_matrix(template, faces)

    # https://pytorch.org/docs/stable/generated/torch.linalg.inv.html -> [Solving A @ X = B (X = A^-1 @ B)]
    # Consider using torch.linalg.solve() if possible for multiplying a matrix on the left by the inverse, as:
    # ``` linalg.solve(A, B) == linalg.inv(A) @ B ```
    # When B is a matrix

    # It is always preferred to use `solve()` when possible, 
    # as it is faster and more numerically stable than computing the inverse explicitly.

    # It is possible to compute the solution of the system X @ A = B (X = B @ A^-1) 
    # by passing the inputs A and B transposed and transposing the output returned by this function.
    # ``` linalg.solve(A.T, B.T).T == B @ linalg.inv(A) ```

    # neutral_span_inv_matrix = torch.linalg.inv(neutral_span_matrix)
    # Q = (span_matrix @ neutral_span_inv_matrix).permute(0, 1, 3, 2)
    Q = torch.linalg.solve(neutral_span_matrix.permute(0, 1, 3, 2), span_matrix.permute(0, 1, 3, 2))
    return Q




def get_mtl_content(tex_fname):
    return f'newmtl Material\nmap_Kd {tex_fname}\n'

def get_obj_content(vertices, faces, uv_coordinates=None, uv_indices=None, mtl_fname=None):
    obj = ('# \n')

    if mtl_fname is not None:
        obj += f'mtllib {mtl_fname}\n'
        obj += 'usemtl Material\n'

    # Write the vertices
    for vertex in vertices:
        obj += f"v {vertex[0]} {vertex[1]} {vertex[2]}\n"

    # Write the UV coordinates
    if uv_coordinates is not None:
        for uv in uv_coordinates:
            obj += f"vt {uv[0]} {uv[1]}\n"

    # Write the faces with UV indices
    if uv_indices is not None:
        for face, uv_indices in zip(faces, uv_indices):
            obj += f"f {face[0]+1}/{uv_indices[0]+1} {face[1]+1}/{uv_indices[1]+1} {face[2]+1}/{uv_indices[2]+1}\n"
    else:
        for face in faces:
            obj += f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
    return obj

def normalize_image_points(pts, resolution):
    """
    normalizes u, v coordinates from [0, image_size] to [-1, 1]
    :param u:
    :param v:
    :param resolution:
    :return:
    """
    pts[...,0] = 2 * (pts[...,0] - resolution[1] / 2.0) / resolution[1]
    pts[...,1] = 2 * (pts[...,1] - resolution[0] / 2.0) / resolution[0]
    return pts

def lmks_pixels(lmks, size):
    """
    lmks: [B, 68, 2], float32, range [0, 1]
    """
    lmks = lmks.clone()
    lmks[...,0] = lmks[...,0] * size[1]
    lmks[...,1] = lmks[...,1] * size[0]
    return lmks


def unnormalize_image_points(pts, image_size):
    """
    pts: [B, N, 2], float32, range [0, 1] or [-1, 1]

    return:
        [B, N, 2] 张量，表示原始图像尺寸下的坐标点。
    """
    if not isinstance(pts, torch.Tensor):
        pts = torch.tensor(pts, dtype=torch.float32)
    
    if isinstance(image_size, int):
        height = width = image_size
    else:
        height, width = image_size
    
    min_val = pts.min()
    max_val = pts.max()
    
    if min_val >= 0 and max_val <= 1:
        pixels = pts.clone()
        pixels[..., 0] *= width  
        pixels[..., 1] *= height 
    elif min_val >= -1 and max_val <= 1:
        pixels = pts.clone()
        pixels[..., 0] = (pts[..., 0] + 1) * width / 2  
        pixels[..., 1] = (pts[..., 1] + 1) * height / 2 
    else:
        pixels = pts.clone()
        pixels = pixels.clamp(-1, 1)
        pixels[..., 0] = (pts[..., 0] + 1) * width / 2  
        pixels[..., 1] = (pts[..., 1] + 1) * height / 2 
        # raise ValueError(f"Invalid range [{min_val},{max_val}]. Valid range: [0, 1] or [-1, 1]")
    
    return pixels    


def normalize(vertices, use_NDC: bool=False, return_scale: bool=False):
    """ center and normalize vertices in [-1, 1]^3 """
    if isinstance(vertices, np.ndarray):
        mu = np.mean(vertices, axis=0)
        vertices = vertices - mu
        if use_NDC:
            vmin = vertices.min(axis=0)
            vmax = vertices.max(axis=0)
            scale = np.sqrt(((vmax - vmin) ** 2).sum(-1))
        else:
            scale = np.linalg.norm(vertices, axis=-1).max()
    elif isinstance(vertices, torch.Tensor):
        mu = torch.mean(vertices, dim=0)
        vertices = vertices - mu
        if use_NDC:
            vmin = vertices.min(dim=0)[0]
            vmax = vertices.max(dim=0)[0]
            scale = torch.sqrt(((vmax - vmin) ** 2).sum(-1))
        else:
            scale = torch.norm(vertices, dim=-1).max()
    else:
        raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")
    
    vertices_normalized = vertices / scale
    if return_scale:
        return vertices_normalized, mu, scale
    return vertices_normalized


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def normalize_vertices(obj):
    if obj.vs is None or len(obj.vs) == 0:
        return obj
    
    # bounding box
    min_coords = np.min(obj.vs, axis=0)
    max_coords = np.max(obj.vs, axis=0)
    center = (min_coords + max_coords) / 2.0
    scale = 2.0 / np.max(max_coords - min_coords)
    
    # normalize
    obj.vs = (obj.vs - center) * scale
    return obj, center, scale

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

def compute_face_normals(verts: torch.Tensor, faces: torch.Tensor)->torch.Tensor:
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

def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(), 
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(), 
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

def extract_submesh(vertices: np.ndarray, faces: np.ndarray, indices: np.ndarray, mode: str='face'):
    """
    input:
        vertices (np.ndarray):  (V, 3)
        faces (np.ndarray):  (F, 3)
        indices (list): 
        
    return:
        sub_vertices (np.ndarray): 
        sub_faces (np.ndarray): 
        vertex_map (dict): mapping from old vertex indices to new vertex indices
    """
    assert mode in ['face', 'vertex']
    assert indices is not None and len(indices) > 0

    if mode == 'face':
        all_vertex_indices = np.unique(faces[indices].flatten())
        selected_faces = faces[indices]
    elif mode == 'vertex':
        vertex_set = set(indices)
        mask = np.array([all(v in vertex_set for v in face) for face in faces])
        selected_faces = faces[mask]
        
        face_vertices = np.unique(selected_faces.flatten())
        all_vertex_indices = np.unique(np.concatenate([indices, face_vertices]))
    
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(all_vertex_indices)}
    sub_vertices = vertices[all_vertex_indices]
    sub_faces = np.vectorize(vertex_map.get)(selected_faces)
    
    return sub_vertices, sub_faces, vertex_map


def submesh_cotangent_laplacian_dense(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_indices,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 ``face_indices`` 指定的三角形子集所诱导的子网格，计算与 ``HACKHead`` 中相同的
    PyTorch3D 余切 Laplacian 稠密矩阵。

    参数
    ----
    verts: ``[V, 3]`` 全网格顶点
    faces: ``[F, 3]`` 全网格三角面，``long``，与 ``verts`` 一致
    face_indices: 面下标（``list`` / 一维 ``LongTensor``），指向 ``faces`` 的行

    返回
    ----
    L: ``[V_sub, V_sub]`` 子网格上稠密 Laplacian
    v_global: ``[V_sub]`` 长整型，子网格第 ``i`` 个顶点对应全网格 ``verts[v_global[i]]``
    """
    from pytorch3d.structures.meshes import Meshes

    if verts.dim() != 2 or verts.shape[1] != 3:
        raise ValueError("verts 应为 [V, 3]")
    if faces.dim() != 2 or faces.shape[1] != 3:
        raise ValueError("faces 应为 [F, 3]")

    device = verts.device
    dtype = verts.dtype
    faces = faces.to(device=device, dtype=torch.long)
    fi = torch.as_tensor(face_indices, device=device, dtype=torch.long)
    if fi.numel() == 0:
        raise ValueError("face_indices 为空")
    sub_faces = faces[fi]
    v_unique = torch.unique(sub_faces.reshape(-1))
    v_sub = v_unique.numel()
    v_full = int(verts.shape[0])
    mapper = -torch.ones(v_full, device=device, dtype=torch.long)
    mapper[v_unique] = torch.arange(v_sub, device=device, dtype=torch.long)
    new_faces = mapper[sub_faces]
    if (new_faces < 0).any():
        raise RuntimeError("子网格面重映射出现非法顶点下标")
    sub_verts = verts[v_unique]
    L = (
        Meshes(verts=[sub_verts], faces=[new_faces])
        .laplacian_packed()
        .to_dense()
    )
    return L.to(dtype=dtype), v_unique


def normalize_vertices(vertices):
    """
    Normalize vertices to [0, 1] range per axis.

    Args:
        vertices: Tensor of shape (N, 3) or (B, N, 3)

    Returns:
        normalized vertices of the same shape
    """
    if vertices.dim() == 2:
        # Single mesh: (N, 3)
        x_min = vertices[:, 0].min()
        x_max = vertices[:, 0].max()
        y_min = vertices[:, 1].min()
        y_max = vertices[:, 1].max()
        z_min = vertices[:, 2].min()
        z_max = vertices[:, 2].max()

        vertices[:, 0] = (vertices[:, 0] - x_min) / (x_max - x_min) * 2 -1
        vertices[:, 1] = (vertices[:, 1] - y_min) / (y_max - y_min) * 2 -1
        vertices[:, 2] = (vertices[:, 2] - z_min) / (z_max - z_min) * 2 -1

    elif vertices.dim() == 3:
        # Batch of meshes: (B, N, 3)
        x_min = vertices[:, :, 0].min(dim=1, keepdim=True).values
        x_max = vertices[:, :, 0].max(dim=1, keepdim=True).values
        y_min = vertices[:, :, 1].min(dim=1, keepdim=True).values
        y_max = vertices[:, :, 1].max(dim=1, keepdim=True).values
        z_min = vertices[:, :, 2].min(dim=1, keepdim=True).values
        z_max = vertices[:, :, 2].max(dim=1, keepdim=True).values

        vertices[:, :, 0] = (vertices[:, :, 0] - x_min) / (x_max - x_min) * 2 -1
        vertices[:, :, 1] = (vertices[:, :, 1] - y_min) / (y_max - y_min) * 2 -1
        vertices[:, :, 2] = (vertices[:, :, 2] - z_min) / (z_max - z_min) * 2 -1

    else:
        raise ValueError("Input must be of shape (N, 3) or (B, N, 3)")

    return vertices


def normalize_vertices_pair_shared(verts_a: torch.Tensor, verts_b: torch.Tensor) -> tuple:
    """
    Apply one per-batch, per-axis affine map to both vertex tensors so each axis
    lies in ``[-1, 1]`` with comparable scale (helps when e.g. depth dominates x/y).
    Min/max are taken over the **union** of ``verts_a`` and ``verts_b`` along the
    vertex dimension so base vs. refined relative geometry is preserved.

    Args:
        verts_a: ``[B, N1, 3]``
        verts_b: ``[B, N2, 3]``

    Returns:
        ``(verts_a_norm, verts_b_norm)`` — new tensors; inputs are not modified.
    """
    if verts_a.dim() != 3 or verts_b.dim() != 3:
        raise ValueError("normalize_vertices_pair_shared expects (B, N, 3) tensors")
    if verts_a.shape[0] != verts_b.shape[0]:
        raise ValueError("batch sizes must match")
    v = torch.cat([verts_a, verts_b], dim=1)
    x_min = v[:, :, 0].min(dim=1, keepdim=True).values
    x_max = v[:, :, 0].max(dim=1, keepdim=True).values
    y_min = v[:, :, 1].min(dim=1, keepdim=True).values
    y_max = v[:, :, 1].max(dim=1, keepdim=True).values
    z_min = v[:, :, 2].min(dim=1, keepdim=True).values
    z_max = v[:, :, 2].max(dim=1, keepdim=True).values
    eps = 1e-8
    rx = (x_max - x_min).clamp(min=eps)
    ry = (y_max - y_min).clamp(min=eps)
    rz = (z_max - z_min).clamp(min=eps)

    a_out = verts_a.clone()
    b_out = verts_b.clone()
    a_out[:, :, 0] = (verts_a[:, :, 0] - x_min) / rx * 2 - 1
    a_out[:, :, 1] = (verts_a[:, :, 1] - y_min) / ry * 2 - 1
    a_out[:, :, 2] = (verts_a[:, :, 2] - z_min) / rz * 2 - 1
    b_out[:, :, 0] = (verts_b[:, :, 0] - x_min) / rx * 2 - 1
    b_out[:, :, 1] = (verts_b[:, :, 1] - y_min) / ry * 2 - 1
    b_out[:, :, 2] = (verts_b[:, :, 2] - z_min) / rz * 2 - 1
    return a_out, b_out

def normalize_mesh(v, f, mode='unit_sphere', discretization_aware=True, return_scale_shift=False):
    '''
    Normalizes mesh vertices using one of the following modes:
    - 'unit_sphere': centers the mesh and scales it to fit in unit sphere
    - 'surface_area': centers the mesh and scales it to have unit surface area

    `discretization_aware=True` optionally shifts the mesh using proper area-weighted centroid of the mesh
    '''
    eps = 1e-8

    use_torch = torch.is_tensor(v)
    if use_torch:
        device, dtype = v.device, v.dtype
        v_np = v.detach().cpu().numpy()
        f_np = f.detach().cpu().numpy()
    else:
        v_np, f_np = v, f

    # compute (area-weighted) centroid in NumPy
    if discretization_aware or mode == 'surface_area':
        areas = igl.doublearea(v_np, f_np) * 0.5
        total_area = areas.sum()
        tri_centers = v_np[f_np].mean(axis=1)
        centroid_np = (areas[:,None] * tri_centers).sum(axis=0) / total_area
    else:
        centroid_np = v_np.mean(axis=0)

    v_centered_np = v_np - centroid_np

    if mode == 'unit_sphere':
        # max distance from origin
        dists = np.linalg.norm(v_centered_np, axis=1)
        scale_np = dists.max()
    elif mode == 'surface_area':
        scale_np = np.sqrt(total_area) + eps
    else:
        raise ValueError(f"Unknown mode '{mode}'")

    v_out = v_centered_np / scale_np

    if use_torch:
        v_out = torch.from_numpy(v_out).to(device=device, dtype=dtype)
    
    if return_scale_shift:
        return v_out, scale_np, centroid_np
    
    return v_out

    
## load obj,  similar to load_obj from pytorch3d
def load_obj(obj_filename):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        lines = [line.strip() for line in f]

    verts, uvcoords = [], []
    colors = []
    faces, uv_faces = [], []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)

            if len(tokens) > 4:
                if '.' in tokens[4]:
                    color = [int(float(x)*255) for x in tokens[4:7]]
                else:
                    color = [int(x) for x in tokens[4:7]]
                if len(color) != 3:
                    msg = "Color %s does not have 3 values. Line: %s"
                    raise ValueError(msg % (str(vert), str(line)))
                colors.append(color)
        elif line.startswith("vt "):  # Line is a texture.
            tx = [float(x) for x in tokens[1:3]]
            if len(tx) != 2:
                raise ValueError(
                    "Texture %s does not have 2 values. Line: %s" % (str(tx), str(line))
                )
            uvcoords.append(tx)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            for vert_props in face_list:
                # Vertex index.
                faces.append(int(vert_props[0]))
                if len(vert_props) > 1:
                    if vert_props[1] != "":
                        # Texture index is present e.g. f 4/1/1.
                        uv_faces.append(int(vert_props[1]))

    verts = np.array(verts).astype(np.float32)
    uvcoords  = np.array(uvcoords ).astype(np.float32)
    colors = np.array(colors).astype(int)
    faces  = np.array(faces).astype(np.int64)
    faces = faces.reshape(-1, 3) - 1
    uv_faces = np.array(uv_faces).astype(np.int64)
    uv_faces = uv_faces.reshape(-1, 3) - 1

    return verts, uvcoords, colors, faces, uv_faces



class Obj:
    vs = None
    vts = None
    fvs = None
    fvts = None
    vns = None
    headers = []
    usemtls = []


def read_obj(obj_path, only_vs=False, tri=False, normalize=False):
    objfile = open(obj_path, encoding="utf8").read().strip().split("\n")
    if only_vs:
        obj = Obj()
        obj.vs = np.array([[float(j) for j in i[2:].strip().split()] for i in filter(lambda x:x.startswith("v "), objfile)], np.float32)
        return obj

    vs = []    # vertices coordinates, [#V, 3]
    vts = []   # uv coordinates, [#F, 2]
    fvs = []   # face vertices indices, [#F, 3]
    fvts = []  # face uv indices, [#F, 3]

    headers = []
    usemtls = []

    for line in objfile:
        if line.startswith("v "):
            vs.append(list(map(float, line[2:].strip().split())))
        elif line.startswith("vt "):
            vts.append(list(map(float, line[2:].strip().split())))
        elif line.startswith("f "):
            fv = []
            fvt = []
            for i in line[2:].strip().split():
                component = i.split("/")[:2]
                vth = int(component[0]) - 1
                fv.append(vth)

                if len(component) > 1 and component[1] != "":
                    vtth = int(component[1]) - 1
                    fvt.append(vtth)

            if tri:
                for i in range(2, len(fv)):
                    fvs.append(fv[:1] + fv[i - 1:i + 1])
                    fvts.append(fvt[:1] + fvt[i - 1:i + 1])

            else:
                fvs.append(fv)
                fvts.append(fvt)

        elif line.startswith("usemtl "):
            usemtls.append((len(fvs), line))

        elif line.startswith("mtllib "):
            headers.append(line)

    obj = Obj()
    obj.vs = np.array(vs, np.float32)
    obj.vts = np.array(vts, np.float32)

    obj.headers = headers
    obj.usemtls = usemtls

    try:
        obj.fvs = np.array(fvs, int)
        obj.fvts = np.array(fvts, int)
    except ValueError:
        obj.fvs = fvs
        obj.fvts = fvts
    
    if normalize:
        obj, center, scale = normalize_vertices(obj)
        return obj, center, scale
    else:
        return obj


def write_obj(obj, output_path):
    with open(output_path, 'w', encoding="utf8") as f:
        for header in obj.headers:
            f.write(header + "\n")
        
        for _, usemtl in obj.usemtls:
            f.write(usemtl + "\n")
        
        # write vertices
        if obj.vs is not None:
            for v in obj.vs:
                f.write(f"v {' '.join(map(str, v))}\n")
        
        # write texture coordinates
        if obj.vts is not None:
            for vt in obj.vts:
                f.write(f"vt {' '.join(map(str, vt))}\n")
        
        # write normals
        if obj.vns is not None:
            for vn in obj.vns:
                f.write(f"vn {' '.join(map(str, vn))}\n")
        
        # write faces
        if obj.fvs is not None:
            for i, face in enumerate(obj.fvs):
                mtl_line = ""
                for idx, line in obj.usemtls:
                    if i == idx:
                        mtl_line = line + "\n"
                        break
                if mtl_line:
                    f.write(mtl_line)
                
                face_data = []
                for j in range(len(face)):
                    v_idx = face[j] + 1
                    vt_idx = obj.fvts[i][j] + 1 if i < len(obj.fvts) and j < len(obj.fvts[i]) else ""
                    vn_idx = ""  # 
                    component = f"{v_idx}"
                    if vt_idx or vn_idx:
                        component += f"/{vt_idx}"
                        if vn_idx:
                            component += f"/{vn_idx}"
                    face_data.append(component)
                
                f.write("f " + " ".join(face_data) + "\n")



def read_mesh_obj(file_path):
    vertices = []  # v
    vertices_texture = []  # vt
    vertices_normal = []  # vn

    face_v = []  # f 1 2 3
    face_vt = []  # f 1/1 2/2 3/3
    face_vn = []  # f 1/1/1 2/2/2 3/3/3

    lines = open(file_path, 'r').readlines()
    for line in lines:
        line = re.sub(' +', ' ', line)
        if line.startswith('v '):
            toks = line.strip().split(' ')[1:]
            try:
                vertices.append([float(toks[0]), float(toks[1]), float(toks[2])])
            except Exception:
                print(toks)
        elif line.startswith('vt '):
            toks = line.strip().split(' ')[1:]
            vertices_texture.append([float(toks[0]), float(toks[1])])
        elif line.startswith('vn '):
            toks = line.strip().split(' ')[1:]
            vertices_normal.append([float(toks[0]), float(toks[1]), float(toks[2])])
        elif line.startswith('f '):
            toks = line.strip().split(' ')[1:]
            if len(toks) == 3:  # tri faces
                faces1 = toks[0].split('/')
                faces2 = toks[1].split('/')
                faces3 = toks[2].split('/')

                face_v.append(np.array([faces1[0], faces2[0], faces3[0]], np.int32) - 1)
                if len(faces1) >= 2:
                    face_vt.append(np.array([faces1[1], faces2[1], faces3[1]], np.int32) - 1)
                if len(faces1) >= 3:
                    if len(faces1[2]) == 0:
                        continue
                    face_vn.append(np.array([faces1[2], faces2[2], faces3[2]], np.int32) - 1)

            if len(toks) == 4:  # quad faces
                faces1 = toks[0].split('/')
                faces2 = toks[1].split('/')
                faces3 = toks[2].split('/')
                faces4 = toks[3].split('/')

                face_v.append(np.array([faces1[0], faces2[0], faces3[0], faces4[0]], np.int32) - 1)
                if len(faces1) >= 2:
                    face_vt.append(np.array([faces1[1], faces2[1], faces3[1], faces4[1]], np.int32) - 1)
                if len(faces1) >= 3:
                    if len(faces1[2]) == 0:
                        continue
                    face_vn.append(np.array([faces1[2], faces2[2], faces3[2], faces4[2]], np.int32) - 1)

    results = {}
    results['v'] = np.array(vertices, np.float32)
    if len(vertices_texture) > 0:
        results['vt'] = np.array(vertices_texture, np.float32)
    if len(vertices_normal) > 0:
        results['vn'] = np.array(vertices_normal, np.float32)

    if len(face_v) > 0:
        results['fv'] = face_v
    if len(face_vt) > 0:
        results['fvt'] = face_vt
    if len(face_vn) > 0:
        results['fvn'] = face_vn

    return results


def write_mesh_obj(mesh_info, file_path):
    v = mesh_info['v']
    vt = mesh_info['vt'] if 'vt' in mesh_info else None
    vn = mesh_info['vn'] if 'vn' in mesh_info else None
    fv = mesh_info['fv'] if 'fv' in mesh_info else None
    fvt = mesh_info['fvt'] if 'fvt' in mesh_info else None
    fvn = mesh_info['fvn'] if 'fvn' in mesh_info else None
    mtl_name = mesh_info['mtl_name'] if 'mtl_name' in mesh_info else None

    if vt is None:
        rgb_tex = False
    elif vt.shape[1] == 2:
        rgb_tex = False
    elif vt.shape[1] == 3:
        rgb_tex = True

    with open(file_path, 'w') as fp:
        # write mtl info
        if mtl_name is not None:
            fp.write(f'mtllib {mtl_name}\n')

        # write vertices
        if rgb_tex:
            for (x, y, z), (r, g, b) in zip(v, vt):
                fp.write('v %f %f %f %f %f %f\n' % (x, y, z, r, g, b))
        else:
            for x, y, z in v:
                fp.write('v %f %f %f\n' % (x, y, z))

        # write vertex textures (UV coordinates)
        if vt is not None and not rgb_tex:
            for u, v in vt:
                fp.write('vt %f %f\n' % (u, v))

        # write vertex normal
        if vn is not None:
            for x, y, z in vn:
                fp.write('vn %f %f %f\n' % (x, y, z))

        # write faces
        if fv is not None:  # have face
            if rgb_tex or (fvt is None and fvn is None):  # fv only
                for v_list in fv:
                    v_list = v_list + 1
                    if len(v_list) == 3:
                        v1, v2, v3 = v_list
                        fp.write('f %d %d %d\n' % (v1, v2, v3))
                    else:
                        v1, v2, v3, v4 = v_list
                        fp.write('f %d %d %d %d\n' % (v1, v2, v3, v4))
            elif fvn is None:  # fv/fvt
                for v_list, vt_list in zip(fv, fvt):
                    v_list = v_list + 1
                    vt_list = vt_list + 1
                    if len(v_list) == 3:
                        v1, v2, v3 = v_list
                        t1, t2, t3 = vt_list
                        fp.write('f %d/%d %d/%d %d/%d\n' % (v1, t1, v2, t2, v3, t3))
                    else:
                        v1, v2, v3, v4 = v_list
                        t1, t2, t3, t4 = vt_list
                        fp.write('f %d/%d %d/%d %d/%d %d/%d\n' % (v1, t1, v2, t2, v3, t3, v4, t4))
            else:  # fv/fvt/fvn
                for v_list, vt_list, vn_list in zip(fv, fvt, fvn):
                    v_list = v_list + 1
                    vt_list = vt_list + 1
                    vn_list = vn_list + 1
                    if len(v_list) == 3:
                        v1, v2, v3 = v_list
                        t1, t2, t3 = vt_list
                        n1, n2, n3 = vn_list
                        fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (v1, t1, n1, v2, t2, n2, v3, t3, n3))
                    else:
                        v1, v2, v3, v4 = v_list
                        t1, t2, t3, t4 = vt_list
                        n1, n2, n3, n4 = vn_list
                        fp.write('f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d\n' %
                                (v1, t1, n1, v2, t2, n2, v3, t3, n3, v4, t4, n4))


if __name__ == "__main__":

    normalized_hack_quad_mesh_with_texture, center_quad, scale_quad = read_obj(
        # "/media/box/Elements/MyExp/AHACK/data/hack_data/hack_template_uv.obj",
        "/media/box/Elements/MyExp/AHACK/data/hack_data/000_generic_neutral_mesh_newuv.obj",
        normalize=True,
    )
    
    write_obj(
        normalized_hack_quad_mesh_with_texture, 
        "/media/box/Elements/MyExp/AHACK/data/hack_data/normalzied_neutral_mesh_newuv.obj",
    )

    normalized_hack_tri_mesh_with_texture, center_tri, scale_tri = read_obj(
        "/media/box/Elements/MyExp/AHACK/data/hack_data/hack_template_uv.obj",
        normalize=True,
    )
    
    
    write_obj(
        normalized_hack_tri_mesh_with_texture, 
        "/media/box/Elements/MyExp/AHACK/data/hack_data/normalzied_hack_template_uv.obj",
    )

    scale_info = {
        "scale_quad": scale_quad,
        "center_quad": center_quad,
        "scale_tri": scale_tri,
        "center_tri": center_tri,
    }
    np.savez(
        "/media/box/Elements/MyExp/AHACK/data/hack_data/scale_info.npz", **scale_info
    )