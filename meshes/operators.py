import torch
import cholespy
import torch.linalg as LA

import torch_mesh_ops as TMO
from torch_geometric.nn import knn



def norm(v):
    """Computes the norm of a vector field."""
    if len(v.shape) == 2:
        _, C = v.size()
        return LA.norm(v.view(-1, 2, C), dim=1)
    elif len(v.shape) == 3:
        B, _, C = v.size()
        return LA.norm(v.view(B, -1, 2, C), dim=2)

def J(v):
    """Rotates a vector field by 90-degrees counter-clockwise."""
    if len(v.shape) == 2:
        N, C = v.size()
        v = v.view(-1, 2, C)
        J_v = torch.zeros_like(v)
        J_v[:, 0] = -v[:, 1]
        J_v[:, 1] = v[:, 0]
        J_v = J_v.view(N, C)
    elif len(v.shape) == 3:
        B, N, C = v.size()
        v = v.view(B, -1, 2, C)
        J_v = torch.zeros_like(v)
        J_v[:, :, 0] = -v[:, :, 1]
        J_v[:, :, 1] = v[:, :, 0]
        J_v = J_v.view(B, N, C)
    return J_v

def I_J(v):
    """Concatenates a vector field and its 90-degree rotated counterpart."""
    return torch.cat([v, J(v)], dim=1)

def curl(v, div):
    """Computes the curl of a vector field using divergence:
    curl = - div J V.
    """
    assert len(v.shape) == len(div.shape)
    if len(v.shape) == 2:
        return - (div @ J(v))
    elif len(v.shape) == 3:
        return -torch.bmm(div, J(v))

def laplacian(x, grad, div):
    """Computes the laplacian of a function using gradient and divergence:
    laplacian = - div grad X.
    """
    assert len(x.shape) == len(grad.shape) == len(div.shape)
    if len(x.shape) == 2:
        return - (div @ (grad @ x))
    elif len(x.shape) == 3:
        return -torch.bmm(div, torch.bmm(grad, x))


def hodge_laplacian(v, grad, div):
    """Computes the Hodge-Laplacian of a vector field using gradient and divergence:
    hodge-laplacian = - (grad div + J grad curl) V.
    """
    assert len(v.shape) == len(grad.shape) == len(div.shape)
    if len(v.shape) == 2:
        # Compute - G G.T v (grad div)
        grad_div_v = grad @ (div @ v)
        # Compute J G G.T J v (J grad curl)
        J_grad_curl_v = J(grad @ curl(v, div))

    elif len(v.shape) == 3:
        grad_div_v = torch.bmm(grad, torch.bmm(div, v))
        J_grad_curl_v = J(torch.bmm(grad, curl(v, div)))

    # Combine
    return - (grad_div_v + J_grad_curl_v)

def vertices_to_faces(x, faces):
    # https://github.com/nmwsharp/diffusion-net
    x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
    faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
    xf = torch.gather(x_gather, 1, faces_gather)
    x_out = torch.mean(xf, dim=-1)
    return x_out


def faces_to_vertices(face_data, faces, face_areas, num_vertices):
    """
    Maps face data [B, F, C] to vertex data [B, V, C] in a discretization-invariant manner.

    face_data: Tensor of shape [B, F, C], signal on each face.
    faces: Tensor of shape [B, F, 3], each row contains vertex indices for a face.
    face_areas: Tensor of shape [B, F], computed face areas.
    num_vertices: total number of vertices, V.
    
    Returns:
        vertex_data: Tensor of shape [B, V, C] containing area-weighted averages.
    """
    B, F, C = face_data.shape

    # Expand face data to each of the 3 vertices per face.
    # (B, F, 3, C) where each face's data is repeated for each vertex.
    face_data_expanded = face_data.unsqueeze(2).expand(B, F, 3, C)
    face_data_flat = face_data_expanded.reshape(B, F*3, C)
    
    # Similarly, flatten face indices: (B, F*3)
    faces_flat = faces.reshape(B, F*3)
    
    # Each face contributes with weight (face_area / 3) to each vertex.
    weights = (face_areas.unsqueeze(-1) / 3).expand(B, F, 3)
    weights_flat = weights.reshape(B, F*3)
    
    # Allocate accumulation tensors for weighted signals and weights.
    vertex_data = torch.zeros(B, num_vertices, C, device=face_data.device, dtype=face_data.dtype)
    vertex_mass = torch.zeros(B, num_vertices, device=face_data.device, dtype=face_data.dtype)
    
    # Scatter-add the weighted face signals to the corresponding vertices.
    vertex_data = vertex_data.scatter_add(
        dim=1,
        index=faces_flat.unsqueeze(-1).expand(B, F*3, C),
        src=face_data_flat * weights_flat.unsqueeze(-1)
    )
    
    # Scatter-add the weights to compute total mass per vertex.
    vertex_mass = vertex_mass.scatter_add(dim=1, index=faces_flat, src=weights_flat)
    
    # Normalize by the accumulated weights at each vertex.
    vertex_data = vertex_data / vertex_mass.unsqueeze(-1)
    
    return vertex_data


def output_at(x, faces, mass, domain='verts'):
    ''' Remaps input vertex signal `x: [B, V, C]` from vertices to faces or global mean '''

    assert x.ndim == 3, "Input signal must have shape (B, V, C)"
    
    if domain == 'vertices' or domain == 'verts':
        return x
    
    if domain == 'faces': 
        return vertices_to_faces(x, faces)
    
    if domain == 'global_mean': 
        return torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)

    raise ValueError(f"Unknown domain: {domain}")


def knn_graph_mesh(vertices, k, batch=None, loop=True, flow='target_to_source'):
    """
    Args:
        vertices (torch.Tensor): mesh vertices coordinates, shape [N, 3]
        k (int): number of neighbor nodes
        batch (torch.Tensor, optional): batch indices, shape [N,], 区分不同网格的顶点。
                default None.
        loop (bool): whether include the query node itself. default True
        flow (str): 'target_to_source' or 'source_to_target'. default to 'target_to_source'
    
    Returns:
        edge_index (torch.Tensor): shape [2, E], edge indices, E = N * k
    """
    if batch is None:
        batch = torch.zeros(vertices.size(0), dtype=torch.long, device=vertices.device)
    
    k_eff = k if loop else k + 1
    
    # knn function in PyG, row=query point index, col=neighbor point index
    row, col = knn(vertices, vertices, k_eff, batch_x=batch, batch_y=batch)
    
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    
    N = vertices.size(0)
    row_new, col_new = [], []
    for i in range(N):
        idx = (row == i).nonzero().squeeze()
        if idx.numel() == 0:
            row_new.extend([i] * k)
            col_new.extend([i] * k)
        else:
            idx = idx[:k] if idx.numel() > k else idx
            row_new.extend(row[idx].tolist())
            col_new.extend(col[idx].tolist())
    
    row = torch.tensor(row_new, device=vertices.device)
    col = torch.tensor(col_new, device=vertices.device)
    
    if flow == 'target_to_source':
        edge_index = torch.stack([row, col], dim=0)  # target(row) → source(col)
    elif flow == 'source_to_target':
        edge_index = torch.stack([col, row], dim=0)  # source(col) → target(row)
    else:
        raise ValueError(f"flow must be either 'target_to_source' or 'source_to_target' but got {flow}")
    
    return edge_index


def repeat_sparse_batch(sparse_tensor: torch.sparse.Tensor, B: int) -> torch.sparse.Tensor:
    """
    将形状为 (1, M, K) 的稀疏张量在batch维度重复B次，得到 (B, M, K) 的稀疏张量
    若输入是2维 (M, K)，自动升维为 (1, M, K)
    """
    if sparse_tensor.dim() == 2:
        sparse_tensor = sparse_tensor.unsqueeze(0).coalesce()
    assert sparse_tensor.dim() == 3 and sparse_tensor.size(0) == 1
    
    orig_indices = sparse_tensor.indices()  # [3, N]，N是非零元素数
    orig_values = sparse_tensor.values()    # [N,]
    device = sparse_tensor.device
    dtype = sparse_tensor.dtype

    # 构造新的batch维度索引（0~B-1）
    batch_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, orig_indices.size(1))  # [B, N]
    # 扩展M/K维度的索引（重复B次）
    m_idx = orig_indices[1:2, :].repeat(1, B)  # [1, B*N]
    k_idx = orig_indices[2:3, :].repeat(1, B)  # [1, B*N]

    # new indices [3, B*N]
    new_indices = torch.cat([
        batch_idx.reshape(1, -1),  # [1, B*N]
        m_idx,                     # M维度 [1, B*N]
        k_idx                      # K维度 [1, B*N]
    ], dim=0)

    new_values = orig_values.repeat(B)  # [B*N,]
    new_size = (B, sparse_tensor.size(1), sparse_tensor.size(2))
    return torch.sparse_coo_tensor(
        indices=new_indices,
        values=new_values,
        size=new_size,
        device=device,
        dtype=dtype
    )


def edge_to_face_map(faces, return_inverse=False):
    """
     Compute unique edge to face mapping from faces indices
    """
    with torch.no_grad():
        # create all edges, packed by triangle
        all_edges = torch.cat((
            torch.stack((faces[:, 0], faces[:, 1]), dim=-1),
            torch.stack((faces[:, 1], faces[:, 2]), dim=-1),
            torch.stack((faces[:, 2], faces[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # elliminate duplicates and return inverse mapping
        unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

        tris = torch.arange(faces.shape[0]).repeat_interleave(3).cuda()

        tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

        # compute edge to face table
        mask0 = order[:,0] == 0
        mask1 = order[:,0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge

@torch.no_grad()
def construct_mesh_operators(V, F, high_precision=False, poisson_solve=True):
    '''
    Creates the following operators for a mesh. Uses PyTorch CUDA extension.
    - vertex_mass:  [B, V]       lumped vertex masses
    - solver:       [B,]         list of Cholesky solvers for mesh's cotangent Laplacian
    - G:            [B, 2*F, V]  face-based intrinsic gradient operator
    - M:            [B, 2*F]     interleaved face areas

    Optionally set high_precision to True to use double precision in intermediate
    computations. Note the operators will be returned in float precision regardless.
    '''
    # ensure contiguous memory layout before calling CUDA extension
    V = V.contiguous()
    F = F.contiguous()
    if V.ndim == 2:
        V = V.unsqueeze(0)
        F = F.unsqueeze(0)
            
    if high_precision:
        V = V.double()

    nB, nV, _ = V.shape
    device = V.device

    vert_mass = TMO.vertex_mass_batched(V, F, 1e-8)       # [B, V]
    face_areas = TMO.face_areas_batched(V, F)             # [B, F]
    edge_lengths = TMO.edge_lengths_batched(V, F)         # [B, F, 3]
    G = TMO.intrinsic_gradient_batched(edge_lengths, F)   # [B, 2*F, V]

    vert_mass = vert_mass.float()
    face_areas = face_areas.float()
    G = G.float()
    M = torch.repeat_interleave(face_areas, 2, dim=-1) # grad operator is interleaved convention, so M is interleaved too

    if not poisson_solve:
        return vert_mass, None, G, M
    
    solvers = []
    Ls = TMO.cotangent_laplacian_batched(V, F, 1e-10).float() # [B, V, V]

    # Create Cholesky solver for each Laplacian in batch
    # This could be block-diagonalized, but speedup seems marginal
    for bi in range(nB):
        L = Ls[bi]
        eps = 1e-6
        sparse_eps_diag = torch.sparse.spdiags(eps * torch.ones(nV), torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
        L = L + sparse_eps_diag   # [B, V, V]
        
        nretry = 0
        while nretry < 5:
            try:
                nrows = L.shape[-1]
                ii = L._indices()[0]
                jj = L._indices()[1]
                x = L._values()
                solver = cholespy.CholeskySolverF(
                    n_rows=nrows, 
                    ii=ii, 
                    jj=jj, 
                    x=x, 
                    type=cholespy.MatrixType.COO,
                    deviceID=device.index,   # ADD multi-gpu support
                )
                solvers.append(solver)
                break
            except Exception as e:
                eps = eps * 10.0
                sparse_eps_diag = torch.sparse.spdiags(eps * torch.ones(nV), torch.zeros(1, dtype=torch.long), (nV, nV)).to(device)
                L = L + sparse_eps_diag
                nretry += 1
        if nretry >= 5:
            raise RuntimeError(f"Failed to create Cholesky solver for mesh {bi} in batch")
    return vert_mass, solvers, G, M