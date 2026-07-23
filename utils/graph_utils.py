import numpy as np
import scipy.sparse as sp
import torch
import os.path as osp
import matplotlib.pyplot as plt

from scipy.sparse import random, csr_matrix

def laplacian(W, normalized=True):
    """Return graph Laplacian"""

    # Degree matrix.
    d = W.sum(axis=0)

    # Laplacian matrix.
    if not normalized:
        D = sp.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = sp.diags(d.A.squeeze(), 0)
        I = sp.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is sp.csr.csr_matrix
    return L

def rescale_L(L, lmax=2):
    """Rescale Laplacian eigenvalues to [-1,1]"""
    M, M = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L /= lmax * 2  # L = 2.0*L / lmax
    L -= I
    return L

def lmax_L(L):
    """Compute largest Laplacian eigenvalue"""
    return sp.linalg.eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]

def coarsen(A, levels):
    graphs, parents = HEM(A, levels)
    perms = compute_perm(parents)

    adjacencies = []
    laplacians = []
    for i, A in enumerate(graphs):
        M, M = A.shape

        if i < levels:
            A = perm_adjacency(A, perms[i])

        A = A.tocsr()
        A.eliminate_zeros()
        adjacencies.append(A)
        # Mnew, Mnew = A.shape
        # print('Layer {0}: M_{0} = |V| = {1} nodes ({2} added), |E| = {3} edges'.format(i, Mnew, Mnew - M, A.nnz // 2))

        L = laplacian(A, normalized=True)
        laplacians.append(L)

    return adjacencies, laplacians, perms if len(perms) > 0 else None

def HEM(W, levels, rid=None):
    """
    Coarsen a graph multiple times using the Heavy Edge Matching (HEM).

    Input
    W: symmetric sparse weight (adjacency) matrix
    levels: the number of coarsened graphs

    Output
    graph[0]: original graph of size N_1
    graph[2]: coarser graph of size N_2 < N_1
    graph[levels]: coarsest graph of Size N_levels < ... < N_2 < N_1
    parents[i] is a vector of size N_i with entries ranging from 1 to N_{i+1}
        which indicate the parents in the coarser graph[i+1]
    nd_sz{i} is a vector of size N_i that contains the size of the supernode in the graph{i}

    Note
    if "graph" is a list of length k, then "parents" will be a list of length k-1
    """

    N, N = W.shape

    if rid is None:
        rid = np.random.permutation(range(N))

    ss = np.array(W.sum(axis=0)).squeeze()
    rid = np.argsort(ss)

    parents = []
    degree = W.sum(axis=0) - W.diagonal()
    graphs = []
    graphs.append(W)

    print('Heavy Edge Matching coarsening with Xavier version')

    for _ in range(levels):

        # CHOOSE THE WEIGHTS FOR THE PAIRING
        # weights = ones(N,1)       # metis weights
        weights = degree  # graclus weights
        # weights = supernode_size  # other possibility
        weights = np.array(weights).squeeze()

        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = sp.find(W)
        cc = idx_row
        rr = idx_col
        vv = val

        # TO BE SPEEDUP
        if not (list(cc) == list(np.sort(cc))):
            tmp = cc
            cc = rr
            rr = tmp

        cluster_id = HEM_one_level(cc, rr, vv, rid, weights)  # cc is ordered
        parents.append(cluster_id)

        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = sp.csr_matrix((nvv, (nrr, ncc)), shape=(Nnew, Nnew))
        W.eliminate_zeros()

        # Add new graph to the list of all coarsened graphs
        graphs.append(W)
        N, N = W.shape

        # COMPUTE THE DEGREE (OMIT OR NOT SELF LOOPS)
        degree = W.sum(axis=0)
        # degree = W.sum(axis=0) - W.diagonal()

        # CHOOSE THE ORDER IN WHICH VERTICES WILL BE VISTED AT THE NEXT PASS
        # [~, rid]=sort(ss);     # arthur strategy
        # [~, rid]=sort(supernode_size);    #  thomas strategy
        # rid=randperm(N);                  #  metis/graclus strategy
        ss = np.array(W.sum(axis=0)).squeeze()
        rid = np.argsort(ss)

    return graphs, parents


# Coarsen a graph given by rr,cc,vv.  rr is assumed to be ordered
def HEM_one_level(rr, cc, vv, rid, weights):
    nnz = rr.shape[0]
    N = rr[nnz - 1] + 1

    marked = np.zeros(N, np.bool)
    rowstart = np.zeros(N, np.int32)
    rowlength = np.zeros(N, np.int32)
    cluster_id = np.zeros(N, np.int32)

    oldval = rr[0]
    count = 0
    clustercount = 0

    for ii in range(nnz):
        rowlength[count] = rowlength[count] + 1
        if rr[ii] > oldval:
            oldval = rr[ii]
            rowstart[count + 1] = ii
            count = count + 1

    for ii in range(N):
        tid = rid[ii]
        if not marked[tid]:
            wmax = 0.0
            rs = rowstart[tid]
            marked[tid] = True
            bestneighbor = -1
            for jj in range(rowlength[tid]):
                nid = cc[rs + jj]
                if marked[nid]:
                    tval = 0.0
                else:

                    # First approach
                    if 2 == 1:
                        tval = vv[rs + jj] * (1.0 / weights[tid] + 1.0 / weights[nid])

                    # Second approach
                    if 1 == 1:
                        Wij = vv[rs + jj]
                        Wii = vv[rowstart[tid]]
                        Wjj = vv[rowstart[nid]]
                        di = weights[tid]
                        dj = weights[nid]
                        tval = (2. * Wij + Wii + Wjj) * 1. / (di + dj + 1e-9)

                if tval > wmax:
                    wmax = tval
                    bestneighbor = nid

            cluster_id[tid] = clustercount

            if bestneighbor > -1:
                cluster_id[bestneighbor] = clustercount
                marked[bestneighbor] = True

            clustercount += 1

    return cluster_id



def compute_perm(parents):
    """
    Return a list of indices to reorder the adjacency and data matrices so
    that the union of two neighbors from layer to layer forms a binary tree.
    """

    # Order of last layer is random (chosen by the clustering algorithm).
    indices = []
    if len(parents) > 0:
        M_last = max(parents[-1]) + 1
        indices.append(list(range(M_last)))

    for parent in parents[::-1]:

        # Fake nodes go after real ones.
        pool_singeltons = len(parent)

        indices_layer = []
        for i in indices[-1]:
            indices_node = list(np.where(parent == i)[0])
            assert 0 <= len(indices_node) <= 2

            # Add a node to go with a singelton.
            if len(indices_node) == 1:
                indices_node.append(pool_singeltons)
                pool_singeltons += 1

            # Add two nodes as children of a singelton in the parent.
            elif len(indices_node) == 0:
                indices_node.append(pool_singeltons + 0)
                indices_node.append(pool_singeltons + 1)
                pool_singeltons += 2

            indices_layer.extend(indices_node)
        indices.append(indices_layer)

    # Sanity checks.
    for i, indices_layer in enumerate(indices):
        M = M_last * 2 ** i
        # Reduction by 2 at each layer (binary tree).
        assert len(indices[0] == M)
        # The new ordering does not omit an indice.
        assert sorted(indices_layer) == list(range(M))

    return indices[::-1]


def perm_adjacency(A, indices):
    """
    Permute adjacency matrix, i.e. exchange node ids,
    so that binary unions form the clustering tree.
    """
    if indices is None:
        return A

    M, M = A.shape
    Mnew = len(indices)
    A = A.tocoo()

    # Add Mnew - M isolated vertices.
    rows = sp.coo_matrix((Mnew - M, M), dtype=np.float32)
    cols = sp.coo_matrix((Mnew, Mnew - M), dtype=np.float32)
    A = sp.vstack([A, rows])
    A = sp.hstack([A, cols])

    # Permute the rows and the columns.
    perm = np.argsort(indices)
    A.row = np.array(perm)[A.row]
    A.col = np.array(perm)[A.col]

    assert np.abs(A - A.T).mean() < 1e-8  # 1e-9
    assert type(A) is sp.coo.coo_matrix
    return A

def perm_index_reverse(indices):
    indices_reverse = np.copy(indices)

    for i, j in enumerate(indices):
        indices_reverse[j] = i

    return indices_reverse

def perm_tri(tri, indices):
    """
    tri: T x 3
    """
    indices_reverse = perm_index_reverse(indices)
    tri_new = np.copy(tri)
    for i in range(len(tri)):
        try:
            tri_new[i, 0] = indices_reverse[tri[i, 0]]
            tri_new[i, 1] = indices_reverse[tri[i, 1]]
            tri_new[i, 2] = indices_reverse[tri[i, 2]]
        except:
            print("i: ", i)
            print("tri[i, 0]: ", tri[i, 0])
            print("len indices_reverse: ", len(indices_reverse))
            raise
    return tri_new



def normalize_sparse_mx(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def build_graph(mesh_tri, num_vertex):
    """
    param: mesh_tri: [F, 3]
    return: adj: sparse matrix, [V, V] (torch.sparse.FloatTensor)
    """
    num_tri = mesh_tri.shape[0]
    edges = np.empty((num_tri * 3, 2))
    for i_tri in range(num_tri):
        edges[i_tri * 3] = mesh_tri[i_tri, :2]
        edges[i_tri * 3 + 1] = mesh_tri[i_tri, 1:]
        edges[i_tri * 3 + 2] = mesh_tri[i_tri, [0, 2]]

    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(num_vertex, num_vertex), dtype=np.float32,
    )

    adj = adj - (adj > 1) * 1.0
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = normalize_sparse_mx(adj + sp.eye(adj.shape[0]))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj


def build_adj(num_kpts, skeleton, flip_pairs):
    adj_matrix = np.zeros((num_kpts, num_kpts))
    for line in skeleton:
        adj_matrix[line[0], line[1]] = 1
        adj_matrix[line[1], line[0]] = 1
    for lr in flip_pairs:
        adj_matrix[lr[0], lr[1]] = 1
        adj_matrix[lr[1], lr[0]] = 1

    return adj_matrix + np.eye(num_kpts)


def build_coarse_graphs(mesh_face, num_kpts, skeleton, flip_pairs, levels=9):
    kpts_adj = build_adj(num_kpts, skeleton, flip_pairs)
    # Build graph
    mesh_adj = build_graph(mesh_face, mesh_face.max() + 1)
    graph_Adj, graph_L, graph_perm = coarsen(mesh_adj, levels=levels)
    input_Adj = sp.csr_matrix(kpts_adj)
    input_Adj.eliminate_zeros()
    input_L = laplacian(input_Adj, normalized=True)

    graph_L[-1] = input_L
    graph_Adj[-1] = input_Adj

    # Compute max eigenvalue of graph Laplacians, rescale Laplacian
    graph_lmax = []
    renewed_lmax = []
    for i in range(levels):
        graph_lmax.append(lmax_L(graph_L[i]))
        graph_L[i] = rescale_L(graph_L[i], graph_lmax[i])
    #     renewed_lmax.append(lmax_L(graph_L[i]))

    return graph_Adj, graph_L, graph_perm, perm_index_reverse(graph_perm[0])


def sparse_python_to_torch(sp_python):
    L = sp_python.tocoo()
    indices = np.column_stack((L.row, L.col)).T
    indices = indices.astype(np.int64)
    indices = torch.from_numpy(indices)
    indices = indices.type(torch.LongTensor)
    L_data = L.data.astype(np.float32)
    L_data = torch.from_numpy(L_data)
    L_data = L_data.type(torch.FloatTensor)
    L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))

    return L



def visualize_sparse_matrix(
    matrix, cmap='viridis', zero_color='white',
):
    """
        matrix: numpy.ndarray or scipy.sparse矩阵 (CSR/CSC)
        cmap: ['viridis'、'plasma'、'coolwarm']
        zero_color: 零元素的颜色（默认白色）
    """
    # 若为稀疏矩阵，转换为稠密矩阵（小矩阵适用，超大矩阵建议用spy函数，见下文补充）
    if hasattr(matrix, 'toarray'):
        dense_matrix = matrix.toarray()
    else:
        dense_matrix = matrix.astype(np.float64) 
    
    fig, ax = plt.subplots(figsize=(8, 8))  

    mask = (dense_matrix == 0)
    im = ax.imshow(
        dense_matrix,
        cmap=cmap,
        aspect='auto',  
        vmin=np.nanmin(dense_matrix[dense_matrix != 0]), 
        vmax=np.nanmax(dense_matrix[dense_matrix != 0]) 
    )
    
    im.set_extent([-0.5, dense_matrix.shape[1]-0.5, dense_matrix.shape[0]-0.5, -0.5])
    # ax.set_facecolor(zero_color)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Non-zero Element Value', rotation=270, labelpad=20)
    ax.set_xlabel('Column Index', fontsize=12)
    ax.set_ylabel('Row Index', fontsize=12)
    
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks(range(dense_matrix.shape[1]))
    ax.set_yticks(range(dense_matrix.shape[0]))
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    from src.hackhead import HACKHead
    from utils.mediapipe_landmarks import *
    
    
    device = torch.device("cuda")
    hackhead = HACKHead(use_teeth=False).to(device)
    faces = hackhead.faces.cpu().numpy()

    shape_params = torch.zeros(1, 200).to(device)
    expression_params = torch.zeros(1, 55).to(device)

    base_results = hackhead(
        shape=shape_params, 
        expression=expression_params, 
        neck_pose=torch.zeros(1, 8, 3).to(device),
        tau=torch.zeros(1, 1).to(device),
        gamma=torch.ones(1, 1).to(device),
        return_lmks_mp=True, 
        return_lmks68=True,
    )[0]
    base_verts = base_results[0].squeeze().cpu().numpy()
    lmks478 = base_results[1].squeeze().cpu().numpy()
    lmks105 = lmks478[:, mediapipe_indices, :]
    lmks68 = base_results[2].squeeze().cpu().numpy()

    skeleton68 = [
        [], [], [], [], [], 
        [], [], [], [], [], 
        [], [], [], [], [], 
        [], [], [], [], [], 
    ]
    flip_pairs68 = [
        [], [], [], [], [], 
        [], [], [], [], [], 
        
    ]

    graph_Adj, graph_L, graph_perm, graph_perm_reverse = build_coarse_graphs(
        mesh_face=faces, 
        num_kpts=len(lmks68), 
        skeleton=skeleton68, 
        flip_pairs=flip_pairs68, 
        levels=9,
    )


    visualize_sparse_matrix(
        matrix=graph_L[-1],  # [68, 68]
        cmap='viridis',  # 'coolwarm'、'YlOrRd'、'viridis'
    )