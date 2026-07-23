import numpy as np
import os
import scipy
import logging
import trimesh
import math
import torch
import scipy.sparse.linalg as sla
import robust_laplacian


def build_mass_matrix(mesh: trimesh.Trimesh):
    """Build the sparse diagonal mass matrix of size (V, V) for a given mesh
    """
    areas = np.zeros(shape=(len(mesh.vertices)))
    for face, area in zip(mesh.faces, mesh.area_faces):
        areas[face] += area / 3.0

    return scipy.sparse.diags(areas)

def approx_methods() -> list[str]:
    """Available laplace approximation types."""
    try:
        import lapy
        return [ 'beltrami', 'cotangens', 'mesh', 'fem' ]
    except ImportError:
        # logging.warn(
        #     "fem appxoimation only works if lapy is installed. "
        #     "You can find lapy on github: https://github.com/Deep-MI/LaPy.\n"
        #     "Install it with pip:\n"
        #      "pip3 install --user git+https://github.com/Deep-MI/LaPy.git#egg=lapy")
        return [ 'beltrami', 'contangens', 'mesh' ]        
    
def build_laplace_betrami_matrix(mesh : trimesh.Trimesh):
    """Build the sparse laplace beltrami matrix of the given mesh M=(V, E).
    This is a positive semidefinite matrix C:

           -1         if (i, j) in E
    C_ij =  deg(V(i)) if i == j
            0         otherwise
    
    Args:
        mesh (trimesh.Trimesh): Mesh used to compute the matrix C
    """
    n = len(mesh.vertices)
    IJ = np.concatenate([
        mesh.edges,
        [[i, i] for i in range(n)]
    ], axis=0)
    V  = np.concatenate([
        [-1 for _ in range(len(mesh.edges))],
        mesh.vertex_degree
    ], axis= 0)

    A = scipy.sparse.coo_matrix((V, (IJ[..., 0], IJ[..., 1])), shape=(n, n), dtype=np.float64)
    return A

def build_cotangens_matrix(mesh: trimesh.Trimesh):
    """Build the sparse cotangens weight matrix of the given mesh.
    This is a positive semidefinite matrix C:

           -0.5 * (tan(a) + tan(b))  if (i, j) in E
    C_ij = -sum_{j in N(i)} (C_ij)   if i == j
            0                        otherwise
    
    Args:
        mesh (trimesh.Trimesh): Mesh used to compute the matrix C

    Returns:
        A sparse matrix of size (#vertices, #vertices) representing the discrete Laplace operator.
    """
    n = len(mesh.vertices)
    ij = mesh.face_adjacency_edges
    ab = mesh.face_adjacency_unshared

    uv = mesh.vertices[ij]
    lr = mesh.vertices[ab]

    def cotan(v1, v2):
        return np.sum(v1*v2) / np.linalg.norm(np.cross(v1, v2), axis=-1)

    ca = cotan(lr[:, 0] - uv[:, 0], lr[:, 0] - uv[:, 1])
    cb = cotan(lr[:, 1] - uv[:, 0], lr[:, 1] - uv[:, 1])

    wij = np.maximum(0.5 * (ca + cb), 0.0)

    I = []
    J = []
    V = []
    for idx, (i, j) in enumerate(ij):
        I += [i, j, i, j]
        J += [j, i, i, j]
        V += [-wij[idx], -wij[idx], wij[idx], wij[idx]]
    
    A = scipy.sparse.coo_matrix((V, (I, J)), shape=(n, n), dtype=np.float64)
    return A

def build_mesh_laplace_matrix(mesh: trimesh.Trimesh):
    """Build the sparse mesh laplacian matrix of the given mesh.
    This is a positive semidefinite matrix C:

           -1/(4pi*h^2) * e^(-||vi-vj||^2/(4h)) if (i, j) in E
    C_ij = -sum_{j in N(i)} (C_ij)              if i == j
            0                                   otherwise
    here h is the average edge length

    Args:
        mesh (trimesh.Trimesh): Mesh used to compute the matrix C

    Returns:
        A sparse matrix of size (#vertices, #vertices) representing the discrete Laplace operator.
    """
    n = len(mesh.vertices)
    h = np.mean(mesh.edges_unique_length)
    a = 1.0 / (4 * math.pi * h*h)
    wij = a * np.exp(-mesh.edges_unique_length**2/(4.0*h))
    I = []
    J = []
    V = []
    for idx, (i, j) in enumerate(mesh.edges_unique):
        I += [i, j, i, j]
        J += [j, i, i, j]
        V += [-wij[idx], -wij[idx], wij[idx], wij[idx]]
    
    A = scipy.sparse.coo_matrix((V, (I, J)), shape=(n, n), dtype=np.float64)
    return A

def build_laplace_approximation_matrix(mesh: trimesh.Trimesh, approx='beltrami'):
    """Build the sparse mesh laplacian matrix of the given mesh.
    This is a positive semidefinite matrix C:

           w_ij                    if (i, j) in E
    C_ij = -sum_{j in N(i)} (w_ij) if i == j
            0                      otherwise
    here h is the average edge length

    Args:
        mesh (trimesh.Trimesh): Mesh used to compute the matrix C
        approx (str): Approximation type to use, must be in ['beltrami', 'cotangens', 'mesh']. Defaults to 'beltrami'.

    Returns:
        A sparse matrix of size (#vertices, #vertices) representing the discrete Laplace operator.
    """
    
    assert approx in approx_methods(), f"Invalid method '{approx}', must be in {approx_methods()}"

    if approx == 'beltrami':
        return build_laplace_betrami_matrix(mesh)
    elif approx == 'cotangens':
        return build_cotangens_matrix(mesh)
    else:
        return build_mesh_laplace_matrix(mesh)

def get_laplace_operator_approximation(
    mesh: trimesh.Trimesh, approx='cotangens',
) -> tuple[np.ndarray, np.ndarray]:
    """Computes a discrete approximation of the laplace-beltrami operator on
    a given mesh. The approximation is given by a Mass matrix A and a weight or stiffness matrix W

    Args:
        mesh (trimesh.Trimesh): Input mesh
        approx (str, optional): Laplace approximation to use See laplace.approx_methods() for possible values. Defaults to 'cotangens'.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of sparse matrices (Stiffness, Mass)
    """
    if approx not in approx_methods():
        raise ValueError(
            f"Invalid approximation method must be one of {approx_methods}."
            f"Got {approx}")
    
    if approx == 'fem':
        import lapy
        T = lapy.TriaMesh(mesh.vertices, mesh.faces)
        solver = lapy.Solver(T)
        return solver.stiffness, solver.mass
    else:
        W = build_laplace_approximation_matrix(mesh, approx)
        M = build_mass_matrix(mesh)
        return W, M




class Signature():
    def __init__(self, mesh: trimesh.Trimesh=None, points: np.ndarray=None, n_basis: int=1, approx='cotangens'):
        assert mesh is not None or points is not None, "Either mesh or points must be provided"

        try:
            from sksparse.cholmod import cholesky
            use_cholmod = True
        except ImportError as e:
            # logging.warn(
            #     "Package scikit-sparse not found (Cholesky decomp). "
            #     "This leads to less efficient eigen decomposition.")
            use_cholmod = False

        if mesh is not None:
            self.L, self.M = get_laplace_operator_approximation(mesh, approx)
            self.n_basis = min(len(mesh.vertices) - 1, n_basis)

        elif points is not None:
            self.L, self.M = robust_laplacian.point_cloud_laplacian(points.astype(np.float64))
            self.n_basis = min(len(points) - 1, n_basis)
            massvec_np = self.M.diagonal()

            if(np.isnan(self.L.data).any()):
                raise RuntimeError("NaN Laplace matrix")
            if(np.isnan(massvec_np).any()):
                raise RuntimeError("NaN mass matrix")
            
        sigma = -0.01
        if use_cholmod:
            chol = cholesky((self.L - sigma * self.M).tocsc())
            op_inv = sla.LinearOperator(
                matvec=chol, shape=self.L.shape, dtype=self.L.dtype,
            )
        else:
            lu = sla.splu((self.L - sigma * self.M).tocsc())
            op_inv = sla.LinearOperator(
                matvec=lu.solve, shape=self.L.shape, dtype=self.L.dtype,
            )
        
        self.evals, self.evecs = sla.eigsh(
            self.L, self.n_basis, self.M, sigma=sigma, OPinv=op_inv,
        )

    def heat_signatures(self, dim: int, return_times=False, times=None):
        """Compute the heat signature for all vertices/points
        Args:
            dim (int): Dimensionality (timesteps) of the signature.
            return_times (bool, optional): If True the function returns a tuple (signature, timesteps) 
                                           otherwise only the signature is returned. Defaults to False.
            times (arraylike, optional): Timesteps used for signature computation.
                                         If None the times are spaced logarithmically. Defaults to None.

        Note:
            This signature is based on 'A Concise and Provably Informative Multi-Scale Signature Based on Heat Diffusion'
            by Jian Sun et al (http://www.lix.polytechnique.fr/~maks/papers/hks.pdf)

        Returns:
            Returns an array of shape (#vertices, dim) containing the heat signatures of every vertex.
            If return_times is True this function returns a tuple (Signature, timesteps).
        """
        if times is None:
            tmax_idx = np.argmax(self.evals > 0)
            tmin = 4 * math.log(10) / self.evals[-1]
            tmax = 4 * math.log(10) / self.evals[tmax_idx]
            times = np.geomspace(tmin, tmax, dim)
        else:
            times = np.array(times).flatten()
            assert len(times) == dim, f"Requested feature dimension and time steps array do not match: {dim} and {len(times)}"

        phi2 = np.square(self.evecs[:, tmax_idx:])
        exp = np.exp(-self.evals[tmax_idx:, None]*times[None])
        s = np.sum(phi2[..., None]*exp[None], axis=1)
        heat_trace = np.sum(exp, axis=0)
        try:
            s = s / heat_trace[None] 
        except RuntimeWarning:
            raise ValueError

        if return_times:
            return s, times
        else:
            return s

    def wave_signatures(self, dim: int, return_energies=False, energies=None):
        """Compute the wave signature for all vertices
        Args:
            dim (int): Dimensionality (energy spectra) of the signature.
            return_energies (bool, optional): If True the function returns a tuple (signature, energies) 
                                              otherwise only the signature is returned. Defaults to False.
            energies (arraylike, optional): Energie spectra used for signature computation.
                                            If None the energy is linearly spaced. Defaults to None.

        Note:
            This signature is based on 'The Wave Kernel Signature: A Quantum Mechanical Approach to Shape Analysis'
            by Mathieu Aubry et al (https://vision.informatik.tu-muenchen.de/_media/spezial/bib/aubry-et-al-4dmod11.pdf)

        Returns:
            Returns an array of shape (#vertices, dim) containing the heat signatures of every vertex.
            If return_times is True this function returns a tuple (Signature, timesteps).
        """
        emin_idx = np.argmax(self.evals > 0)

        if energies is None:
            emin = math.log(self.evals[emin_idx])
            emax = math.log(self.evals[-1]) / 1.02
            energies = np.linspace(emin, emax, dim)
        else:
            energies = np.array(energies).flatten()
            assert len(energies) == dim, f"Requested featrue dimension and energies array do not match: {dim} and {len(energies)}"

        sigma = 7.0 * (energies[-1] - energies[0]) / dim
        # phi2 = np.square(self.evecs[:, 1:])
        phi2 = np.square(self.evecs[:, emin_idx:])
        exp = np.exp(-np.square(energies[None] - np.log(self.evals[emin_idx:, None])) / (2.0 * sigma * sigma))
        s = np.sum(phi2[..., None]*exp[None], axis=1)
        energy_trace = np.sum(exp, axis=0)
        s = s / energy_trace[None] 
            
        if return_energies:
            return s, energies
        else:
            return s

    def signatures(self, dim: int, kernel: str, return_x_ticks=False, x_ticks=None):
        """Computes a signature for each vertex
        Args:
            dim (int): Dimensionality (energy spectra) of the signature.
            kernel (str): Feature kernel used must be in ['heat', 'wave'].
            return_x_ticks (bool, optional): If True the function returns a tuple (signature, x_ticks) 
                                            otherwise only the signature is returned. Defaults to False.
            x_ticks (arraylike, optional): Variable used for the feature dimension. Defaults to None.

        Returns:
            Returns an array of shape (#vertices, dim) containing the mesh signatures of every vertex.
            If return_x_ticks is True this function returns a tuple (signature, x_ticks).
        """
        assert kernel in ['heat', 'wave'], f"Invalid kernel type '{kernel}'. Must be in ['heat', 'wave']"

        if kernel == 'heat':
            return self.heat_signatures(dim, return_x_ticks, x_ticks)
        else:
            return self.wave_signatures(dim, return_x_ticks, x_ticks)

    def heat_distances(self, query, dim: int, return_signature=False, times=None, cutoff=1.0):
        """Compute distances of all vertices to vertices in query based on heat signature

        Note:
            We use the L2 norm of the feature vectors.

        Args:
            query (int, arrylike): queried indices
            dim (int): target index
            return_signature (bool, optional): If True the function returns a tuple (distances, signatures). Defaults to False
            times (None, arraylike, optional): Time steps used for signature computation. Defaults to None 
            cutoff (float, optional): Fraction of dimensionality to use for distance computation. Defaults to 1.0

        Returns:
            An array of shape (#vertices, len(query)) holding the heat signature distance 
            of each vertex to the queried vertices. If return_signature is True this function returns a
            tuple (distances, signature).  
        """

        e = max(0, int(cutoff*dim)) + 1
        a = self.heat_signatures(dim, times=times)[..., :e]
        
        b = np.atleast_2d(a[np.array(query, dtype=np.int64)])
        a_dim = a.ndim
        b_dim = b.ndim
        if a_dim == 1:
            a = a.reshape(1, 1, a.shape[0])
        if a_dim >= 2:
            a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
        if b_dim > 2:
            b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
        diff = a - b
        dist_arr = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        dist_arr = np.squeeze(dist_arr)
        if return_signature:
            return dist_arr, a[:, 0, :]
        else:
            return dist_arr

    def wave_distance(self, query, dim: int, return_signatures=False, energies=None, cutoff=1.0):
        """Compute distances of all vertices to vertices in query based on the wave signature

        Note:
            We use the L1 norm of the feature vectors.

        Args:
            query (int, arrylike): queried indices
            dim (int): target index
            return_signature (bool): If True the function returns a tuple (distances, signatures). Defaults to False
            times (None, arraylike): Time steps used for signature computation. Defaults to None 
            cutoff (float, optional): Fraction of dimensionality to use for distance computation. Defaults to 1.0

        Returns:
            An array of shape (#vertices, len(query)) holding the wave signature distance 
            of each vertex to the queried vertices. If return_signature is True this function returns a
            tuple (distances, signature).  
        """
        e = max(0, int(dim * cutoff)) + 1
        a = self.wave_signatures(dim, energies=energies)[..., :e]
        b = np.atleast_2d(a[np.array(query, dtype=np.int64)])
        a_dim = a.ndim
        b_dim = b.ndim
        if a_dim == 1:
            a = a.reshape(1, 1, a.shape[0])
        if a_dim >= 2:
            a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
        if b_dim > 2:
            b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
        diff = a - b
        dist_arr = np.sum(np.abs(diff), axis=-1)
        dist_arr = np.squeeze(dist_arr)
        if return_signatures:
            return dist_arr, a[:, 0, :]
        else:
            return dist_arr
            
    def feature_distance(self, query, dim: int, kernel: str, return_signatures=False, x_ticks=None, cutoff=1.0):
        assert kernel in ['heat', 'wave'], f"Invalid kernel type '{kernel}'. Must be in ['heat', 'wave']"
        if kernel == 'heat':
            return self.heat_distances(query, dim, return_signatures, x_ticks, cutoff)
        else:
            return self.wave_distance(query, dim, return_signatures, x_ticks, cutoff)




