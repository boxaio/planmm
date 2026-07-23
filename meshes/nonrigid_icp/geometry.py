import heapq

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def validate_vertices_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return contiguous float64 vertices and int64 triangular faces.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (n, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (m, 3), got {faces.shape}")
    if faces.size > 0:
        if faces.min() < 0 or faces.max() >= vertices.shape[0]:
            raise ValueError("faces contain indices outside the vertex array")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(faces)


def face_region_boundary_seed(faces: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    seed = np.zeros(region_mask.shape[0], dtype=bool)
    for tri in np.asarray(faces, dtype=np.int64):
        tri_region = region_mask[tri]
        if np.any(tri_region) and not np.all(tri_region):
            seed[tri] = True
    return seed if np.any(seed) else np.asarray(region_mask, dtype=bool).copy()


def to_homogeneous(vertices: np.ndarray) -> np.ndarray:
    """
    vertices: (n, 3)
    return: (n, 4)
    """
    ones = np.ones((vertices.shape[0], 1), dtype=vertices.dtype)
    return np.hstack([vertices, ones])

def extract_edges(faces: np.ndarray) -> np.ndarray:
    """
    从三角面提取无向边。
    return: (m, 2)
    """
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = set()
    for tri in faces:
        i, j, k = tri
        for a, b in [(i, j), (j, k), (k, i)]:
            if a > b:
                a, b = b, a
            edges.add((a, b))
    return np.array(sorted(edges), dtype=np.int64)


def edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Euclidean lengths for undirected mesh edges.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)


def median_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Robust local length scale for a triangular mesh.
    """
    vertices, faces = validate_vertices_faces(vertices, faces)
    lengths = edge_lengths(vertices, extract_edges(faces))
    lengths = lengths[np.isfinite(lengths) & (lengths > 1e-12)]
    if lengths.size == 0:
        return 0.0
    return float(np.median(lengths))


def weighted_vertex_adjacency(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> list[list[tuple[int, float]]]:
    """
    Build an undirected vertex adjacency list with Euclidean edge lengths.
    """
    vertices, faces = validate_vertices_faces(vertices, faces)
    edges = extract_edges(faces)
    lengths = edge_lengths(vertices, edges)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(vertices.shape[0])]
    for (i, j), length in zip(edges, lengths):
        if not np.isfinite(length) or length <= 0.0:
            continue
        i = int(i)
        j = int(j)
        length = float(length)
        adjacency[i].append((j, length))
        adjacency[j].append((i, length))
    return adjacency


def geodesic_distance_from_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    seed_mask: np.ndarray,
    max_distance: float | None = None,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Approximate mesh geodesic distance from a vertex mask using edge-length Dijkstra.
    """
    vertices, faces = validate_vertices_faces(vertices, faces)
    seed_mask = np.asarray(seed_mask, dtype=bool).reshape(-1)
    if seed_mask.shape[0] != vertices.shape[0]:
        raise ValueError(f"seed_mask must have length {vertices.shape[0]}, got {seed_mask.shape[0]}")

    n_vertices = vertices.shape[0]
    if allowed_mask is None:
        allowed = np.ones(n_vertices, dtype=bool)
    else:
        allowed = np.asarray(allowed_mask, dtype=bool).reshape(-1)
        if allowed.shape[0] != n_vertices:
            raise ValueError(f"allowed_mask must have length {n_vertices}, got {allowed.shape[0]}")

    distances = np.full(n_vertices, np.inf, dtype=np.float64)
    seeds = np.flatnonzero(seed_mask & allowed)
    if seeds.size == 0:
        return distances

    if max_distance is None:
        max_distance = np.inf
    max_distance = float(max_distance)
    if max_distance < 0.0:
        return distances

    adjacency = weighted_vertex_adjacency(vertices, faces)
    heap: list[tuple[float, int]] = []
    for vid in seeds:
        vid = int(vid)
        distances[vid] = 0.0
        heapq.heappush(heap, (0.0, vid))

    while heap:
        dist, vid = heapq.heappop(heap)
        if dist != distances[vid]:
            continue
        if dist > max_distance:
            continue
        for nb, edge_length in adjacency[vid]:
            if not allowed[nb]:
                continue
            new_dist = dist + edge_length
            if new_dist < distances[nb] and new_dist <= max_distance:
                distances[nb] = new_dist
                heapq.heappush(heap, (new_dist, nb))
    return distances


def smooth_vertex_scalar_field(
    values: np.ndarray,
    faces: np.ndarray,
    iterations: int = 0,
    step: float = 0.5,
    fixed_mask: np.ndarray | None = None,
    fixed_values: np.ndarray | None = None,
) -> np.ndarray:
    """
    Uniform Laplacian smoothing for a per-vertex scalar field.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = values.shape[0]
    iterations = int(iterations)
    if iterations <= 0 or faces.size == 0:
        return values.copy()

    edges = extract_edges(faces)
    if edges.size == 0:
        return values.copy()

    step = float(np.clip(step, 0.0, 1.0))
    if step <= 0.0:
        return values.copy()

    row = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64, copy=False)
    col = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64, copy=False)
    data = np.ones(row.shape[0], dtype=np.float64)
    degree = np.bincount(row, minlength=n_vertices).astype(np.float64)
    ok = degree[row] > 0.0
    neighbor_average = sp.coo_matrix(
        (data[ok] / degree[row[ok]], (row[ok], col[ok])),
        shape=(n_vertices, n_vertices),
        dtype=np.float64,
    ).tocsr()

    if fixed_mask is None:
        fixed_mask = np.zeros(n_vertices, dtype=bool)
    else:
        fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
        if fixed_mask.shape[0] != n_vertices:
            raise ValueError(f"fixed_mask must have length {n_vertices}, got {fixed_mask.shape[0]}")

    if fixed_values is None:
        fixed_values = values
    else:
        fixed_values = np.asarray(fixed_values, dtype=np.float64).reshape(-1)
        if fixed_values.shape[0] != n_vertices:
            raise ValueError(f"fixed_values must have length {n_vertices}, got {fixed_values.shape[0]}")

    out = values.copy()
    free = ~fixed_mask
    for _ in range(iterations):
        smoothed = (1.0 - step) * out + step * (neighbor_average @ out)
        out[free] = smoothed[free]
        out[fixed_mask] = fixed_values[fixed_mask]
    return out


def _uniform_graph_laplacian_matrix(
    faces: np.ndarray,
    n_vertices: int,
    allowed_vertex_mask: np.ndarray | None = None,
) -> sp.csr_matrix:
    """Uniform graph Laplacian L = I - A with A the one-ring neighbor average."""
    if allowed_vertex_mask is not None:
        neighbor_average = _mesh_neighbor_average_matrix_restricted(
            faces,
            n_vertices,
            np.asarray(allowed_vertex_mask, dtype=bool).reshape(-1),
        )
    else:
        neighbor_average = _mesh_neighbor_average_matrix(faces, n_vertices)
    return sp.eye(n_vertices, dtype=np.float64, format="csr") - neighbor_average


def compute_vertex_laplacian_defect_scores(
    vertices: np.ndarray,
    faces: np.ndarray,
    allowed_vertex_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-vertex smoothness defect: ||Lv||, ||L^2 v||, and normal jitter vs neighbors.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = vertices.shape[0]
    if allowed_vertex_mask is None:
        neighbor_average = _mesh_neighbor_average_matrix(faces, n_vertices)
    else:
        neighbor_average = _mesh_neighbor_average_matrix_restricted(
            faces,
            n_vertices,
            np.asarray(allowed_vertex_mask, dtype=bool).reshape(-1),
        )
    lap_coord = vertices - neighbor_average @ vertices
    lap_mag = np.linalg.norm(lap_coord, axis=1)
    bi_coord = lap_coord - neighbor_average @ lap_coord
    bi_mag = np.linalg.norm(bi_coord, axis=1)
    normals = vertex_normals(vertices, faces)
    target_normals = normalize_vectors(neighbor_average @ normals)
    normal_jitter = 1.0 - np.clip(np.sum(normals * target_normals, axis=1), -1.0, 1.0)
    return lap_mag, bi_mag, normal_jitter


def bilaplacian_smooth_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_weights: np.ndarray,
    *,
    fixed_mask: np.ndarray | None = None,
    fixed_vertices: np.ndarray | None = None,
    allowed_vertex_mask: np.ndarray | None = None,
    bilaplacian_weight: float = 1.0,
    anchor_weight: float = 8.0,
    anchor_vertex_weights: np.ndarray | None = None,
    damping_scale: float = 1e-4,
    edge_scale: float = 0.0,
    max_displacement: float | np.ndarray | None = None,
    reference_vertices: np.ndarray | None = None,
) -> np.ndarray:
    """
    Minimize weighted Bi-Laplacian energy ||W^{1/2} L^2 (v - v_ref)||^2 plus anchor.

    L is the uniform graph Laplacian on the mesh (optionally restricted to
    allowed_vertex_mask). Free vertices are those with positive weight and not fixed.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    weights = np.clip(np.asarray(vertex_weights, dtype=np.float64).reshape(-1), 0.0, 1.0)
    n_vertices = vertices.shape[0]
    if weights.shape[0] != n_vertices:
        raise ValueError(f"vertex_weights must have length {n_vertices}, got {weights.shape[0]}")
    if faces.size == 0 or not np.any(weights > 1e-6):
        return vertices.copy()

    if fixed_mask is None:
        fixed_mask = np.zeros(n_vertices, dtype=bool)
    else:
        fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
    if fixed_vertices is None:
        fixed_vertices = vertices
    else:
        fixed_vertices = np.asarray(fixed_vertices, dtype=np.float64)
    if reference_vertices is None:
        reference_vertices = vertices
    else:
        reference_vertices = np.asarray(reference_vertices, dtype=np.float64)

    if edge_scale <= 0.0:
        edge_scale = float(median_edge_length(vertices, faces))

    free_mask = (~fixed_mask) & (weights > 1e-6)
    free_ids = np.flatnonzero(free_mask)
    if free_ids.size == 0:
        return vertices.copy()
    fixed_ids = np.flatnonzero(fixed_mask)
    if fixed_ids.size == 0:
        return vertices.copy()

    lap = _uniform_graph_laplacian_matrix(faces, n_vertices, allowed_vertex_mask)
    bi_lap = lap @ lap
    bi_ff = bi_lap[np.ix_(free_ids, free_ids)].tocsr()
    bi_fc = bi_lap[np.ix_(free_ids, fixed_ids)].tocsr()

    w_free = weights[free_ids] * float(max(bilaplacian_weight, 0.0))
    if not np.any(w_free > 1e-12):
        return vertices.copy()
    w_mat = sp.diags(w_free, dtype=np.float64, format="csr")

    fixed_pos = fixed_vertices[fixed_ids]
    rhs_bi = -(bi_fc @ fixed_pos)
    anchor_base = float(max(anchor_weight, 0.0))
    anchor_per = np.full(free_ids.size, anchor_base, dtype=np.float64)
    if anchor_vertex_weights is not None:
        anchor_per = anchor_base * np.clip(
            np.asarray(anchor_vertex_weights, dtype=np.float64).reshape(-1)[free_ids],
            0.0,
            None,
        )
    damp = float(max(damping_scale, 0.0)) * float(edge_scale) ** 4

    system = bi_ff.T @ w_mat @ bi_ff
    if np.any(anchor_per > 0.0):
        system = system + sp.diags(anchor_per, dtype=np.float64, format="csr")
    if damp > 0.0:
        system = system + damp * sp.eye(free_ids.size, dtype=np.float64, format="csr")

    out = vertices.copy()
    for dim in range(3):
        b = bi_ff.T @ w_mat @ rhs_bi[:, dim]
        if np.any(anchor_per > 0.0):
            b = b + anchor_per * reference_vertices[free_ids, dim]
        out[free_ids, dim] = spla.spsolve(system, b)

    out[fixed_mask] = fixed_vertices[fixed_mask]
    if max_displacement is not None:
        out = clamp_relative_vertex_displacement(reference_vertices, out, max_displacement)
        out[fixed_mask] = fixed_vertices[fixed_mask]
    return out


def taubin_smooth_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_weights: np.ndarray,
    iterations: int = 0,
    lamb: float = 0.33,
    mu: float = -0.34,
) -> np.ndarray:
    """
    Weighted Taubin smoothing for vertex positions.

    vertex_weights controls where smoothing is applied. A weight of 0 fixes a
    vertex, while 1 applies the full smoothing step.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    weights = np.asarray(vertex_weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != vertices.shape[0]:
        raise ValueError(f"vertex_weights must have length {vertices.shape[0]}, got {weights.shape[0]}")
    iterations = int(iterations)
    if iterations <= 0 or faces.size == 0 or not np.any(weights > 0.0):
        return vertices.copy()

    n_vertices = vertices.shape[0]
    neighbor_average = _mesh_neighbor_average_matrix(faces, n_vertices)

    weights = np.clip(weights, 0.0, 1.0)[:, None]
    out = vertices.copy()
    for _ in range(iterations):
        delta = neighbor_average @ out - out
        out = out + float(lamb) * weights * delta
        delta = neighbor_average @ out - out
        out = out + float(mu) * weights * delta
    return out


def _mesh_neighbor_average_matrix(faces: np.ndarray, n_vertices: int) -> sp.csr_matrix:
    edges = extract_edges(faces)
    if edges.size == 0:
        return sp.csr_matrix((n_vertices, n_vertices), dtype=np.float64)
    row = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64, copy=False)
    col = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64, copy=False)
    degree = np.bincount(row, minlength=n_vertices).astype(np.float64)
    ok = degree[row] > 0.0
    return sp.coo_matrix(
        (np.ones(row.shape[0], dtype=np.float64)[ok] / degree[row[ok]], (row[ok], col[ok])),
        shape=(n_vertices, n_vertices),
        dtype=np.float64,
    ).tocsr()


def _mesh_neighbor_average_matrix_restricted(
    faces: np.ndarray,
    n_vertices: int,
    allowed_mask: np.ndarray,
) -> sp.csr_matrix:
    """Uniform neighbor average using only edges with both endpoints in allowed_mask."""
    allowed_mask = np.asarray(allowed_mask, dtype=bool).reshape(-1)
    edges = extract_edges(faces)
    if edges.size == 0:
        return sp.csr_matrix((n_vertices, n_vertices), dtype=np.float64)
    keep = allowed_mask[edges[:, 0]] & allowed_mask[edges[:, 1]]
    if not np.any(keep):
        return sp.csr_matrix((n_vertices, n_vertices), dtype=np.float64)
    edges = edges[keep]
    row = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64, copy=False)
    col = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64, copy=False)
    degree = np.bincount(row, minlength=n_vertices).astype(np.float64)
    ok = degree[row] > 0.0
    return sp.coo_matrix(
        (np.ones(row.shape[0], dtype=np.float64)[ok] / degree[row[ok]], (row[ok], col[ok])),
        shape=(n_vertices, n_vertices),
        dtype=np.float64,
    ).tocsr()


def optimize_face_nonface_junction_normal_smoothness(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    smooth_vertex_weights: np.ndarray,
    *,
    iterations: int = 12,
    lamb: float = 0.42,
    mu: float = -0.46,
    tangential_only: bool = True,
    normal_alignment_step: float = 0.22,
    max_displacement: np.ndarray | None = None,
    max_iteration_displacement_scale: float = 0.35,
    pin_vertex_mask: np.ndarray | None = None,
    neighbor_allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Smooth junction normals by moving weighted vertices.

    Taubin-style position updates (optionally tangential-only) plus optional
    normal alignment toward a neighbor-averaged normal field.

    By default, vertices in face_mask are hard-pinned. Pass pin_vertex_mask to
    override which vertices stay fixed (e.g. relax a narrow forehead band at a
    skull seam while keeping the rest of the face frozen).

    neighbor_allowed_mask restricts the one-ring used for Laplacian and normal
    diffusion (e.g. forehead|skull only, so neck vertices are not pulled in).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    if pin_vertex_mask is None:
        pin_mask = face_mask.copy()
    else:
        pin_mask = np.asarray(pin_vertex_mask, dtype=bool).reshape(-1)
        if pin_mask.shape[0] != face_mask.shape[0]:
            raise ValueError(
                f"pin_vertex_mask must have length {face_mask.shape[0]}, got {pin_mask.shape[0]}"
            )
    weights = np.clip(np.asarray(smooth_vertex_weights, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if weights.shape[0] != vertices.shape[0]:
        raise ValueError(f"smooth_vertex_weights must have length {vertices.shape[0]}, got {weights.shape[0]}")
    iterations = int(iterations)
    if iterations <= 0 or faces.size == 0 or not np.any(weights > 0.0):
        return vertices.copy()

    reference_vertices = vertices.copy()
    if neighbor_allowed_mask is not None:
        neighbor_allowed_mask = np.asarray(neighbor_allowed_mask, dtype=bool).reshape(-1)
        neighbor_average = _mesh_neighbor_average_matrix_restricted(
            faces,
            vertices.shape[0],
            neighbor_allowed_mask,
        )
    else:
        neighbor_average = _mesh_neighbor_average_matrix(faces, vertices.shape[0])
    pinned_positions = vertices[pin_mask].copy()
    weight_col = weights[:, None]
    edge_scale = float(median_edge_length(vertices, faces))
    step_scale = float(normal_alignment_step) * edge_scale
    per_step_limit = None
    if float(max_iteration_displacement_scale) > 0.0:
        per_step_limit = float(max_iteration_displacement_scale) * edge_scale * weights
    cumulative_limit = None
    if max_displacement is not None:
        cumulative_limit = np.asarray(max_displacement, dtype=np.float64).reshape(-1)
        if cumulative_limit.shape[0] != vertices.shape[0]:
            raise ValueError(
                f"max_displacement must have length {vertices.shape[0]}, got {cumulative_limit.shape[0]}"
            )
    out = vertices.copy()
    out[pin_mask] = pinned_positions

    def _apply_displacement_limits(prev: np.ndarray) -> None:
        nonlocal out
        out[pin_mask] = pinned_positions
        if per_step_limit is not None:
            out = clamp_relative_vertex_displacement(prev, out, per_step_limit)
        if cumulative_limit is not None:
            out = clamp_relative_vertex_displacement(reference_vertices, out, cumulative_limit)
        out[pin_mask] = pinned_positions

    for _ in range(iterations):
        prev = out.copy()
        out[pin_mask] = pinned_positions
        delta = neighbor_average @ out - out
        if tangential_only:
            normals = vertex_normals(out, faces)
            delta = delta - np.sum(delta * normals, axis=1, keepdims=True) * normals
        out = out + float(lamb) * weight_col * delta
        _apply_displacement_limits(prev)

        prev = out.copy()
        out[pin_mask] = pinned_positions
        delta = neighbor_average @ out - out
        if tangential_only:
            normals = vertex_normals(out, faces)
            delta = delta - np.sum(delta * normals, axis=1, keepdims=True) * normals
        out = out + float(mu) * weight_col * delta
        _apply_displacement_limits(prev)

        if step_scale > 0.0:
            prev = out.copy()
            out[pin_mask] = pinned_positions
            normals = vertex_normals(out, faces)
            target_normals = normalize_vectors(neighbor_average @ normals)
            dots = np.clip(np.sum(normals * target_normals, axis=1), -1.0, 1.0)
            angles = np.arccos(dots)
            rot_axes = normalize_vectors(np.cross(normals, target_normals))
            move_dir = np.cross(rot_axes, normals)
            out = out + step_scale * weight_col * angles[:, None] * move_dir
            _apply_displacement_limits(prev)

    out[pin_mask] = pinned_positions
    if cumulative_limit is not None:
        out = clamp_relative_vertex_displacement(reference_vertices, out, cumulative_limit)
        out[pin_mask] = pinned_positions
    return out


def harmonize_region_seam_displacement(
    reference_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    active_mask: np.ndarray,
    fixed_mask: np.ndarray,
    displacement_ramp: np.ndarray,
    *,
    iterations: int = 12,
    inner_ramp_threshold: float = 0.92,
) -> np.ndarray:
    """
    Diffuse displacement only in the transition shell near a region seam.

    Neighbor averaging is restricted to active_mask so lateral non-face regions
    are not pulled toward zero displacement.
    """
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    active_mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    fixed_mask = np.asarray(fixed_mask, dtype=bool).reshape(-1)
    displacement_ramp = np.clip(np.asarray(displacement_ramp, dtype=np.float64).reshape(-1), 0.0, 1.0)
    iterations = int(iterations)
    inner_ramp_threshold = float(np.clip(inner_ramp_threshold, 0.5, 1.0))
    if iterations <= 0 or faces.size == 0 or not np.any(active_mask):
        return vertices.copy()

    delta = vertices - reference_vertices
    delta[fixed_mask] = 0.0
    shell = active_mask & ~fixed_mask & (displacement_ramp > 1e-6) & (displacement_ramp < inner_ramp_threshold)
    inner_anchor = active_mask & (displacement_ramp >= inner_ramp_threshold)
    if not np.any(shell):
        return vertices.copy()

    anchor = delta.copy()
    neighbor_average = _mesh_neighbor_average_matrix_restricted(
        faces,
        vertices.shape[0],
        active_mask,
    )
    for _ in range(iterations):
        for axis in range(3):
            smoothed = neighbor_average @ delta[:, axis]
            delta[shell, axis] = smoothed[shell]
        delta[fixed_mask] = 0.0
        if np.any(inner_anchor):
            delta[inner_anchor] = anchor[inner_anchor]

    out = reference_vertices + delta
    out[fixed_mask] = reference_vertices[fixed_mask]
    return out


def relax_region_pair_cross_edge_displacement(
    reference_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    region_a_mask: np.ndarray,
    region_b_mask: np.ndarray,
    *,
    iterations: int = 10,
    blend: float = 0.85,
    move_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Reduce displacement jumps on edges between two labeled regions by blending
    each movable endpoint toward the average displacement of its cross-edge pair.
    """
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    region_a_mask = np.asarray(region_a_mask, dtype=bool).reshape(-1)
    region_b_mask = np.asarray(region_b_mask, dtype=bool).reshape(-1)
    iterations = int(iterations)
    blend = float(np.clip(blend, 0.0, 1.0))
    if iterations <= 0 or faces.size == 0 or blend <= 0.0:
        return vertices.copy()

    edges = extract_edges(faces)
    if edges.size == 0:
        return vertices.copy()
    cross = region_a_mask[edges[:, 0]] & region_b_mask[edges[:, 1]]
    cross |= region_b_mask[edges[:, 0]] & region_a_mask[edges[:, 1]]
    if not np.any(cross):
        return vertices.copy()

    cross_edges = edges[cross]
    if move_mask is not None:
        move_mask = np.asarray(move_mask, dtype=bool).reshape(-1)

    out = vertices.copy()
    for _ in range(iterations):
        delta = out - reference_vertices
        accum = np.zeros_like(delta)
        counts = np.zeros(out.shape[0], dtype=np.float64)
        for v0, v1 in cross_edges:
            pair_mean = 0.5 * (delta[v0] + delta[v1])
            accum[v0] += pair_mean
            accum[v1] += pair_mean
            counts[v0] += 1.0
            counts[v1] += 1.0
        movable = counts > 0.0
        if move_mask is not None:
            movable &= move_mask
        if not np.any(movable):
            break
        target = np.zeros_like(delta)
        target[movable] = accum[movable] / counts[movable, None]
        delta[movable] = (1.0 - blend) * delta[movable] + blend * target[movable]
        out = reference_vertices + delta
    return out


def weld_region_pair_cross_edge_positions(
    vertices: np.ndarray,
    faces: np.ndarray,
    region_a_mask: np.ndarray,
    region_b_mask: np.ndarray,
    *,
    iterations: int = 16,
    blend: float = 1.0,
    move_mask: np.ndarray | None = None,
    pin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Collapse crease steps by pulling cross-region edge endpoints toward their
    shared midpoint in world space (not relative to a reference mesh).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    region_a_mask = np.asarray(region_a_mask, dtype=bool).reshape(-1)
    region_b_mask = np.asarray(region_b_mask, dtype=bool).reshape(-1)
    iterations = int(iterations)
    blend = float(np.clip(blend, 0.0, 1.0))
    if iterations <= 0 or faces.size == 0 or blend <= 0.0:
        return vertices.copy()

    edges = extract_edges(faces)
    if edges.size == 0:
        return vertices.copy()
    cross = region_a_mask[edges[:, 0]] & region_b_mask[edges[:, 1]]
    cross |= region_b_mask[edges[:, 0]] & region_a_mask[edges[:, 1]]
    if not np.any(cross):
        return vertices.copy()

    cross_edges = edges[cross]
    if move_mask is not None:
        move_mask = np.asarray(move_mask, dtype=bool).reshape(-1)
    if pin_mask is not None:
        pin_mask = np.asarray(pin_mask, dtype=bool).reshape(-1)

    pinned = vertices.copy()
    if pin_mask is not None:
        pinned[pin_mask] = vertices[pin_mask]
    out = vertices.copy()
    if pin_mask is not None:
        out[pin_mask] = pinned[pin_mask]

    for _ in range(iterations):
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]
        accum = np.zeros_like(out)
        counts = np.zeros(out.shape[0], dtype=np.float64)
        for v0, v1 in cross_edges:
            midpoint = 0.5 * (out[v0] + out[v1])
            accum[v0] += midpoint
            accum[v1] += midpoint
            counts[v0] += 1.0
            counts[v1] += 1.0
        movable = counts > 0.0
        if move_mask is not None:
            movable &= move_mask
        if pin_mask is not None:
            movable &= ~pin_mask
        if np.any(movable):
            out[movable] = (1.0 - blend) * out[movable] + blend * (
                accum[movable] / counts[movable, None]
            )
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]
    if pin_mask is not None:
        out[pin_mask] = pinned[pin_mask]
    return out


def align_region_pair_cross_edges_to_face_tangent(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    region_a_mask: np.ndarray,
    region_b_mask: np.ndarray,
    *,
    iterations: int = 12,
    blend: float = 1.0,
    move_nonface_mask: np.ndarray | None = None,
    pin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Remove normal-direction gaps on cross edges between two regions by pulling
    the non-face endpoint onto the face neighbor's tangent plane.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    region_a_mask = np.asarray(region_a_mask, dtype=bool).reshape(-1)
    region_b_mask = np.asarray(region_b_mask, dtype=bool).reshape(-1)
    iterations = int(iterations)
    blend = float(np.clip(blend, 0.0, 1.0))
    if iterations <= 0 or faces.size == 0 or blend <= 0.0:
        return vertices.copy()

    edges = extract_edges(faces)
    if edges.size == 0:
        return vertices.copy()
    cross = region_a_mask[edges[:, 0]] & region_b_mask[edges[:, 1]]
    cross |= region_b_mask[edges[:, 0]] & region_a_mask[edges[:, 1]]
    if not np.any(cross):
        return vertices.copy()

    cross_edges = edges[cross]
    if move_nonface_mask is not None:
        move_nonface_mask = np.asarray(move_nonface_mask, dtype=bool).reshape(-1)
    if pin_mask is not None:
        pin_mask = np.asarray(pin_mask, dtype=bool).reshape(-1)

    pinned = vertices.copy()
    if pin_mask is not None:
        pinned[pin_mask] = vertices[pin_mask]
    out = vertices.copy()
    if pin_mask is not None:
        out[pin_mask] = pinned[pin_mask]

    for _ in range(iterations):
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]
        normals = vertex_normals(out, faces)
        accum = np.zeros_like(out)
        counts = np.zeros(out.shape[0], dtype=np.float64)
        for v0, v1 in cross_edges:
            if face_mask[v0] and not face_mask[v1]:
                face_id, nonface_id = int(v0), int(v1)
            elif face_mask[v1] and not face_mask[v0]:
                face_id, nonface_id = int(v1), int(v0)
            else:
                continue
            vec = out[nonface_id] - out[face_id]
            normal = normals[face_id]
            normal_comp = np.sum(vec * normal) * normal
            target = out[nonface_id] - blend * normal_comp
            accum[nonface_id] += target
            counts[nonface_id] += 1.0
        active = counts > 0.0
        if move_nonface_mask is not None:
            active &= move_nonface_mask
        if pin_mask is not None:
            active &= ~pin_mask
        if np.any(active):
            out[active] = accum[active] / counts[active, None]
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]
    if pin_mask is not None:
        out[pin_mask] = pinned[pin_mask]
    return out


def relax_region_pair_cross_edge_stretch(
    template_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    region_a_mask: np.ndarray,
    region_b_mask: np.ndarray,
    *,
    max_stretch_ratio: float = 1.5,
    min_stretch_ratio: float = 0.55,
    iterations: int = 24,
    blend: float = 0.9,
    move_mask_a: np.ndarray | None = None,
    move_mask_b: np.ndarray | None = None,
    pin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Clamp edge stretch on cross-region edges (e.g. forehead/skull).

    Either endpoint may move when included in move_mask_a / move_mask_b.
    """
    template_vertices = np.asarray(template_vertices, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    region_a_mask = np.asarray(region_a_mask, dtype=bool).reshape(-1)
    region_b_mask = np.asarray(region_b_mask, dtype=bool).reshape(-1)
    max_stretch_ratio = max(float(max_stretch_ratio), 1.0)
    min_stretch_ratio = max(float(min_stretch_ratio), 1e-3)
    iterations = int(iterations)
    blend = float(np.clip(blend, 0.0, 1.0))
    if iterations <= 0 or faces.size == 0 or blend <= 0.0:
        return vertices.copy()

    edges = extract_edges(faces)
    if edges.size == 0:
        return vertices.copy()
    cross = region_a_mask[edges[:, 0]] & region_b_mask[edges[:, 1]]
    cross |= region_b_mask[edges[:, 0]] & region_a_mask[edges[:, 1]]
    if not np.any(cross):
        return vertices.copy()

    cross_edges = edges[cross]
    a_ids = np.where(region_a_mask[cross_edges[:, 0]], cross_edges[:, 0], cross_edges[:, 1])
    b_ids = np.where(region_a_mask[cross_edges[:, 0]], cross_edges[:, 1], cross_edges[:, 0])
    if move_mask_a is not None:
        move_mask_a = np.asarray(move_mask_a, dtype=bool).reshape(-1)
    if move_mask_b is not None:
        move_mask_b = np.asarray(move_mask_b, dtype=bool).reshape(-1)
    if pin_mask is not None:
        pin_mask = np.asarray(pin_mask, dtype=bool).reshape(-1)

    pinned = vertices.copy()
    if pin_mask is not None:
        pinned[pin_mask] = vertices[pin_mask]
    out = vertices.copy()
    if pin_mask is not None:
        out[pin_mask] = pinned[pin_mask]

    for _ in range(iterations):
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]
        for a_id, b_id in zip(a_ids, b_ids):
            move_a = move_mask_a is None or move_mask_a[a_id]
            move_b = move_mask_b is None or move_mask_b[b_id]
            if pin_mask is not None:
                move_a &= not pin_mask[a_id]
                move_b &= not pin_mask[b_id]
            if not move_a and not move_b:
                continue
            ref_vec = template_vertices[b_id] - template_vertices[a_id]
            ref_len = float(np.linalg.norm(ref_vec))
            if ref_len <= 1e-12:
                continue
            cur_vec = out[b_id] - out[a_id]
            cur_len = float(np.linalg.norm(cur_vec))
            if cur_len <= 1e-12:
                unit = ref_vec / ref_len
            else:
                unit = cur_vec / cur_len
            target_len = float(
                np.clip(
                    cur_len if cur_len > 1e-12 else ref_len,
                    min_stretch_ratio * ref_len,
                    max_stretch_ratio * ref_len,
                )
            )
            if abs(cur_len - target_len) <= 1e-12 * max(ref_len, 1.0):
                continue
            if move_a and move_b:
                mid = 0.5 * (out[a_id] + out[b_id])
                new_a = mid - 0.5 * unit * target_len
                new_b = mid + 0.5 * unit * target_len
                out[a_id] = (1.0 - blend) * out[a_id] + blend * new_a
                out[b_id] = (1.0 - blend) * out[b_id] + blend * new_b
            elif move_b:
                target = out[a_id] + unit * target_len
                out[b_id] = (1.0 - blend) * out[b_id] + blend * target
            else:
                target = out[b_id] - unit * target_len
                out[a_id] = (1.0 - blend) * out[a_id] + blend * target
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]
    if pin_mask is not None:
        out[pin_mask] = pinned[pin_mask]
    return out


def relax_edges_in_vertex_mask_stretch(
    template_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_mask: np.ndarray,
    *,
    max_stretch_ratio: float = 1.5,
    min_stretch_ratio: float = 0.55,
    iterations: int = 20,
    blend: float = 0.88,
    pin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Clamp stretch on edges whose endpoints both lie in vertex_mask."""
    template_vertices = np.asarray(template_vertices, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    vertex_mask = np.asarray(vertex_mask, dtype=bool).reshape(-1)
    max_stretch_ratio = max(float(max_stretch_ratio), 1.0)
    min_stretch_ratio = max(float(min_stretch_ratio), 1e-3)
    iterations = int(iterations)
    blend = float(np.clip(blend, 0.0, 1.0))
    if iterations <= 0 or faces.size == 0 or blend <= 0.0 or not np.any(vertex_mask):
        return vertices.copy()

    edges = extract_edges(faces)
    keep = vertex_mask[edges[:, 0]] & vertex_mask[edges[:, 1]]
    if not np.any(keep):
        return vertices.copy()
    edge_list = edges[keep]

    pinned = vertices.copy()
    if pin_mask is not None:
        pin_mask = np.asarray(pin_mask, dtype=bool).reshape(-1)
        pinned[pin_mask] = vertices[pin_mask]
    out = vertices.copy()
    if pin_mask is not None:
        out[pin_mask] = pinned[pin_mask]

    for _ in range(iterations):
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]
        for a_id, b_id in edge_list:
            if pin_mask is not None and (pin_mask[a_id] or pin_mask[b_id]):
                continue
            ref_vec = template_vertices[b_id] - template_vertices[a_id]
            ref_len = float(np.linalg.norm(ref_vec))
            if ref_len <= 1e-12:
                continue
            cur_vec = out[b_id] - out[a_id]
            cur_len = float(np.linalg.norm(cur_vec))
            if cur_len <= 1e-12:
                unit = ref_vec / ref_len
            else:
                unit = cur_vec / cur_len
            target_len = float(
                np.clip(
                    cur_len if cur_len > 1e-12 else ref_len,
                    min_stretch_ratio * ref_len,
                    max_stretch_ratio * ref_len,
                )
            )
            if abs(cur_len - target_len) <= 1e-12 * max(ref_len, 1.0):
                continue
            mid = 0.5 * (out[a_id] + out[b_id])
            new_a = mid - 0.5 * unit * target_len
            new_b = mid + 0.5 * unit * target_len
            out[a_id] = (1.0 - blend) * out[a_id] + blend * new_a
            out[b_id] = (1.0 - blend) * out[b_id] + blend * new_b
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]
    if pin_mask is not None:
        out[pin_mask] = pinned[pin_mask]
    return out


def compress_region_edges_toward_reference_median_length(
    vertices: np.ndarray,
    faces: np.ndarray,
    region_mask: np.ndarray,
    reference_vertex_mask: np.ndarray,
    *,
    target_ratio: float = 0.98,
    iterations: int = 32,
    blend: float = 0.88,
    pin_mask: np.ndarray | None = None,
    refresh_reference_every: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Shorten (or lengthen) edges fully inside ``region_mask`` toward
    ``target_ratio`` times the median edge length on edges in ``reference_vertex_mask``.

    Used to shrink template-heavy regions (e.g. HACK ``frown_0``) toward surrounding face scale.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    region_mask = np.asarray(region_mask, dtype=bool).reshape(-1)
    reference_vertex_mask = np.asarray(reference_vertex_mask, dtype=bool).reshape(-1)
    target_ratio = float(max(target_ratio, 1e-3))
    iterations = int(iterations)
    blend = float(np.clip(blend, 0.0, 1.0))
    stats = {
        "enabled": 0.0,
        "iterations": float(iterations),
        "target_ratio": target_ratio,
        "reference_median_edge_length": float("nan"),
        "region_edges": 0.0,
        "mean_stretch_before": float("nan"),
        "mean_stretch_after": float("nan"),
    }
    if iterations <= 0 or blend <= 0.0 or faces.size == 0:
        return vertices.copy(), stats
    if not np.any(region_mask) or not np.any(reference_vertex_mask):
        return vertices.copy(), stats

    edges = extract_edges(faces)
    if edges.size == 0:
        return vertices.copy(), stats

    ref_keep = reference_vertex_mask[edges[:, 0]] & reference_vertex_mask[edges[:, 1]]
    region_keep = region_mask[edges[:, 0]] & region_mask[edges[:, 1]]
    if not np.any(region_keep):
        return vertices.copy(), stats

    def _edge_lengths(verts: np.ndarray, edge_idx: np.ndarray) -> np.ndarray:
        return np.linalg.norm(
            verts[edge_idx[:, 1]] - verts[edge_idx[:, 0]],
            axis=1,
        )

    ref_edges = edges[ref_keep]
    ref_lens = _edge_lengths(vertices, ref_edges)
    ref_lens = ref_lens[ref_lens > 1e-12]
    if ref_lens.size == 0:
        return vertices.copy(), stats
    region_edges = edges[region_keep]
    before = _edge_lengths(vertices, region_edges)

    def _reference_median_len(verts: np.ndarray) -> float:
        lens = _edge_lengths(verts, ref_edges)
        lens = lens[lens > 1e-12]
        if lens.size == 0:
            return float("nan")
        return float(np.median(lens))

    median_len = _reference_median_len(vertices)
    if not np.isfinite(median_len):
        return vertices.copy(), stats
    target_len = target_ratio * median_len
    stats["reference_median_edge_length"] = median_len
    stats["region_edges"] = float(np.count_nonzero(region_keep))
    stats["mean_stretch_before"] = float(np.mean(before / max(median_len, 1e-12)))

    refresh_every = max(int(refresh_reference_every), 0)
    pinned = vertices.copy()
    if pin_mask is not None:
        pin_mask = np.asarray(pin_mask, dtype=bool).reshape(-1)
        pinned[pin_mask] = vertices[pin_mask]
    out = vertices.copy()
    if pin_mask is not None:
        out[pin_mask] = pinned[pin_mask]

    for it in range(iterations):
        if refresh_every > 0 and it > 0 and it % refresh_every == 0:
            median_len = _reference_median_len(out)
            if np.isfinite(median_len):
                target_len = target_ratio * median_len
                stats["reference_median_edge_length"] = median_len
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]
        for a_id, b_id in region_edges:
            if pin_mask is not None and (pin_mask[a_id] or pin_mask[b_id]):
                continue
            cur_vec = out[b_id] - out[a_id]
            cur_len = float(np.linalg.norm(cur_vec))
            if cur_len <= 1e-12:
                continue
            unit = cur_vec / cur_len
            if abs(cur_len - target_len) <= 1e-12 * max(median_len, 1.0):
                continue
            mid = 0.5 * (out[a_id] + out[b_id])
            new_a = mid - 0.5 * unit * target_len
            new_b = mid + 0.5 * unit * target_len
            out[a_id] = (1.0 - blend) * out[a_id] + blend * new_a
            out[b_id] = (1.0 - blend) * out[b_id] + blend * new_b
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]

    after = _edge_lengths(out, region_edges)
    stats["mean_stretch_after"] = float(np.mean(after / max(median_len, 1e-12)))
    stats["enabled"] = 1.0
    if pin_mask is not None:
        out[pin_mask] = pinned[pin_mask]
    return out, stats


def shrink_vertex_region_toward_centroid(
    vertices: np.ndarray,
    region_mask: np.ndarray,
    *,
    scale: float = 0.97,
    iterations: int = 16,
    blend: float = 0.85,
    pin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """In-plane-ish shrink of free vertices in ``region_mask`` toward their centroid."""
    vertices = np.asarray(vertices, dtype=np.float64)
    region_mask = np.asarray(region_mask, dtype=bool).reshape(-1)
    scale = float(np.clip(scale, 0.5, 1.0))
    iterations = int(iterations)
    blend = float(np.clip(blend, 0.0, 1.0))
    if iterations <= 0 or blend <= 0.0 or scale >= 1.0 - 1e-9:
        return vertices.copy()

    move_mask = region_mask.copy()
    if pin_mask is not None:
        move_mask &= ~np.asarray(pin_mask, dtype=bool).reshape(-1)
    if not np.any(move_mask):
        return vertices.copy()

    pinned = vertices.copy()
    if pin_mask is not None:
        pinned[pin_mask] = vertices[pin_mask]
    out = vertices.copy()
    move_ids = np.flatnonzero(move_mask)
    for _ in range(iterations):
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]
        centroid = np.mean(out[move_ids], axis=0)
        for vid in move_ids:
            delta = out[vid] - centroid
            target = centroid + scale * delta
            out[vid] = (1.0 - blend) * out[vid] + blend * target
        if pin_mask is not None:
            out[pin_mask] = pinned[pin_mask]
    if pin_mask is not None:
        out[pin_mask] = pinned[pin_mask]
    return out


def relax_face_nonface_cross_edge_stretch(
    template_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    nonface_mask: np.ndarray,
    *,
    max_stretch_ratio: float = 1.85,
    min_stretch_ratio: float = 0.55,
    iterations: int = 16,
    blend: float = 0.82,
    move_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pull movable non-face seam vertices so cross face/non-face edges stay near
    template length (relative to the HACK source mesh).
    """
    template_vertices = np.asarray(template_vertices, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    nonface_mask = np.asarray(nonface_mask, dtype=bool).reshape(-1)
    max_stretch_ratio = max(float(max_stretch_ratio), 1.0)
    min_stretch_ratio = max(float(min_stretch_ratio), 1e-3)
    iterations = int(iterations)
    blend = float(np.clip(blend, 0.0, 1.0))
    if iterations <= 0 or faces.size == 0 or blend <= 0.0:
        return vertices.copy()

    edges = extract_edges(faces)
    if edges.size == 0:
        return vertices.copy()
    cross = face_mask[edges[:, 0]] != face_mask[edges[:, 1]]
    if not np.any(cross):
        return vertices.copy()

    cross_edges = edges[cross]
    face_ids = np.where(face_mask[cross_edges[:, 0]], cross_edges[:, 0], cross_edges[:, 1])
    nonface_ids = np.where(face_mask[cross_edges[:, 0]], cross_edges[:, 1], cross_edges[:, 0])
    if move_mask is not None:
        move_mask = np.asarray(move_mask, dtype=bool).reshape(-1)

    face_positions = vertices[face_mask].copy()
    out = vertices.copy()
    out[face_mask] = face_positions
    for _ in range(iterations):
        out[face_mask] = face_positions
        for face_id, nonface_id in zip(face_ids, nonface_ids):
            if move_mask is not None and not move_mask[nonface_id]:
                continue
            ref_vec = template_vertices[nonface_id] - template_vertices[face_id]
            ref_len = float(np.linalg.norm(ref_vec))
            if ref_len <= 1e-12:
                continue
            cur_vec = out[nonface_id] - out[face_id]
            cur_len = float(np.linalg.norm(cur_vec))
            if cur_len <= 1e-12:
                unit = ref_vec / ref_len
            else:
                unit = cur_vec / cur_len
            target_len = float(
                np.clip(
                    cur_len if cur_len > 1e-12 else ref_len,
                    min_stretch_ratio * ref_len,
                    max_stretch_ratio * ref_len,
                )
            )
            target = out[face_id] + unit * target_len
            out[nonface_id] = (1.0 - blend) * out[nonface_id] + blend * target
        out[face_mask] = face_positions
    out[face_mask] = face_positions
    return out


def project_nonface_seam_onto_face_tangent_planes(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    nonface_mask: np.ndarray,
    *,
    iterations: int = 6,
    blend: float = 0.72,
    max_normal_offset_scale: float = 0.28,
    reference_vertices: np.ndarray | None = None,
    free_face_mask: np.ndarray | None = None,
    movable_nonface_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Remove the normal-direction step across face/non-face edges.

    Each non-face seam vertex is pulled toward the tangent plane of its fixed
    face neighbors. Motion is purely along the local face normal.

    Pass free_face_mask for face vertices that should keep their current
    positions (e.g. a relaxed forehead band at the skull seam). By default all
    face vertices are pinned to reference_vertices.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    nonface_mask = np.asarray(nonface_mask, dtype=bool).reshape(-1)
    iterations = int(iterations)
    if iterations <= 0 or faces.size == 0:
        return vertices.copy()

    reference_vertices = np.asarray(
        reference_vertices if reference_vertices is not None else vertices,
        dtype=np.float64,
    ).copy()
    if free_face_mask is not None:
        free_face_mask = np.asarray(free_face_mask, dtype=bool).reshape(-1)
        if free_face_mask.shape[0] != face_mask.shape[0]:
            raise ValueError(
                f"free_face_mask must have length {face_mask.shape[0]}, got {free_face_mask.shape[0]}"
            )
        free_face_mask &= face_mask
    else:
        free_face_mask = np.zeros(face_mask.shape[0], dtype=bool)
    pinned_face_mask = face_mask & ~free_face_mask
    pinned_face_positions = reference_vertices[pinned_face_mask].copy()

    if movable_nonface_mask is not None:
        movable_nonface_mask = np.asarray(movable_nonface_mask, dtype=bool).reshape(-1)
        movable_nonface_mask &= nonface_mask
    else:
        movable_nonface_mask = nonface_mask.copy()

    edge_scale = float(median_edge_length(reference_vertices, faces))
    max_offset = max(float(max_normal_offset_scale), 0.0) * edge_scale
    blend = float(np.clip(blend, 0.0, 1.0))
    edges = extract_edges(faces)
    cross = face_mask[edges[:, 0]] != face_mask[edges[:, 1]]
    if not np.any(cross):
        return vertices.copy()

    cross_edges = edges[cross]
    face_ids = np.where(face_mask[cross_edges[:, 0]], cross_edges[:, 0], cross_edges[:, 1])
    nonface_ids = np.where(face_mask[cross_edges[:, 0]], cross_edges[:, 1], cross_edges[:, 0])
    out = np.asarray(vertices, dtype=np.float64).copy()

    def _restore_pinned_face() -> None:
        if np.any(pinned_face_mask):
            out[pinned_face_mask] = pinned_face_positions

    for _ in range(iterations):
        _restore_pinned_face()
        normals = vertex_normals(out, faces)
        accum = np.zeros_like(out)
        counts = np.zeros(out.shape[0], dtype=np.float64)
        vec = out[nonface_ids] - out[face_ids]
        face_normals = normals[face_ids]
        normal_comp = np.sum(vec * face_normals, axis=1, keepdims=True) * face_normals
        adjustment = -blend * normal_comp
        np.add.at(accum, nonface_ids, adjustment)
        np.add.at(counts, nonface_ids, 1.0)
        active = (counts > 0.0) & movable_nonface_mask
        step = accum[active] / counts[active, None]
        candidate = out.copy()
        candidate[active] = out[active] + step
        if max_offset > 0.0:
            candidate[active] = clamp_relative_vertex_displacement(
                reference_vertices[active],
                candidate[active],
                max_offset,
            )
        out[active] = candidate[active]
        _restore_pinned_face()

    _restore_pinned_face()
    return out


def optimize_nonface_region_smoothness(
    vertices: np.ndarray,
    faces: np.ndarray,
    smooth_vertex_weights: np.ndarray,
    *,
    iterations: int = 12,
    lamb: float = 0.5,
    mu: float = -0.53,
) -> np.ndarray:
    """
    Post-ICP smoothness pass for non-face vertices only.

    Vertices with smooth_vertex_weights <= 0 stay fixed (typically the face and
    any pinned landmarks). Only smoothness is optimized; this does not refit
    to the target surface.
    """
    return taubin_smooth_vertices(
        vertices,
        faces,
        smooth_vertex_weights,
        iterations=int(iterations),
        lamb=float(lamb),
        mu=float(mu),
    )


def face_normals(vertices: np.ndarray, faces: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Area-weighted triangle normals. With normalize=True, each non-degenerate
    face normal is unit length.
    """
    vertices, faces = validate_vertices_faces(vertices, faces)
    if faces.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    if normalize:
        normals = normalize_vectors(normals)
    return normals


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Area-weighted per-vertex normals.
    """
    vertices, faces = validate_vertices_faces(vertices, faces)
    normals = np.zeros_like(vertices, dtype=np.float64)
    if faces.shape[0] == 0:
        return normals

    fn = face_normals(vertices, faces, normalize=False)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], fn)
    return normalize_vectors(normals)


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, np.maximum(norms, eps), out=np.zeros_like(vectors), where=norms > eps)


def identity_affine_field(n_vertices: int) -> np.ndarray:
    """
    Return the row-vector affine field X with shape (4*n, 3).

    For vertex homogeneous row v_h = [x, y, z, 1], the deformed point is
    v_h @ X_i, where X_i is the 4x3 block stored at rows 4*i:4*i+4.
    """
    block = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    return np.tile(block, (int(n_vertices), 1))


def apply_affine_field(vertices: np.ndarray, affine_field: np.ndarray) -> np.ndarray:
    """
    Apply a per-vertex affine field stored as (4*n, 3).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    affine_field = np.asarray(affine_field, dtype=np.float64)
    n_vertices = vertices.shape[0]
    if affine_field.shape != (4 * n_vertices, 3):
        raise ValueError(
            f"affine_field must have shape {(4 * n_vertices, 3)}, got {affine_field.shape}"
        )

    vh = to_homogeneous(vertices)
    blocks = affine_field.reshape(n_vertices, 4, 3)
    return np.einsum("ni,nij->nj", vh, blocks)


def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.sum((a - b) ** 2, axis=-1)


def evaluate_mesh_edge_stretch(
    reference_vertices: np.ndarray,
    deformed_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    face_vertex_mask: np.ndarray | None = None,
    cavity_vertex_mask: np.ndarray | None = None,
    face_boundary_vertex_mask: np.ndarray | None = None,
    exclude_nonface_vertex_mask: np.ndarray | None = None,
    region_a_vertex_mask: np.ndarray | None = None,
    region_b_vertex_mask: np.ndarray | None = None,
    min_edge_length_ratio: float = 0.01,
    stretch_warn_ratio: float = 1.5,
    stretch_bad_ratio: float = 2.0,
) -> dict[str, object]:
    """
    Summarize edge-length stretch between a reference mesh and a deformed mesh.

    Regions are defined on valid (non-degenerate) edges only. ``face_deep_interior``
    uses vertices at least four median edge lengths away from the supplied face
    boundary seed mask; it excludes cavity vertices when a cavity mask is given.
    """
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    deformed_vertices = np.asarray(deformed_vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    edges = extract_edges(faces)
    if edges.size == 0:
        return {"regions": {}, "num_edges": 0}

    ref_len = np.linalg.norm(
        reference_vertices[edges[:, 1]] - reference_vertices[edges[:, 0]],
        axis=1,
    )
    def_len = np.linalg.norm(
        deformed_vertices[edges[:, 1]] - deformed_vertices[edges[:, 0]],
        axis=1,
    )
    length_scale = float(median_edge_length(reference_vertices, faces))
    valid = ref_len > float(min_edge_length_ratio) * max(length_scale, 1e-12)
    ratio = def_len / np.maximum(ref_len, 1e-12)
    log_dev = np.abs(np.log(np.clip(ratio, 1e-3, 1e3)))

    face_vertex_mask = (
        np.asarray(face_vertex_mask, dtype=bool).reshape(-1)
        if face_vertex_mask is not None
        else np.zeros(reference_vertices.shape[0], dtype=bool)
    )
    cavity_vertex_mask = (
        np.asarray(cavity_vertex_mask, dtype=bool).reshape(-1)
        if cavity_vertex_mask is not None
        else np.zeros(reference_vertices.shape[0], dtype=bool)
    )
    if face_boundary_vertex_mask is not None:
        boundary_seed = np.asarray(face_boundary_vertex_mask, dtype=bool).reshape(-1)
    else:
        boundary_seed = face_region_boundary_seed(faces, face_vertex_mask)

    dist_from_boundary = geodesic_distance_from_mask(
        reference_vertices,
        faces,
        boundary_seed,
        max_distance=24.0 * max(length_scale, 1e-12),
    )
    deep_face_vertices = (
        face_vertex_mask
        & ~cavity_vertex_mask
        & np.isfinite(dist_from_boundary)
        & (dist_from_boundary > 4.0 * max(length_scale, 1e-12))
    )
    boundary_ring_vertices = (
        face_vertex_mask
        & ~cavity_vertex_mask
        & np.isfinite(dist_from_boundary)
        & (dist_from_boundary <= 2.0 * max(length_scale, 1e-12))
    )

    def edge_mask(both: np.ndarray) -> np.ndarray:
        return valid & both[edges[:, 0]] & both[edges[:, 1]]

    cross_face = face_vertex_mask[edges[:, 0]] != face_vertex_mask[edges[:, 1]]
    cross_face_no_exclude = cross_face.copy()
    if exclude_nonface_vertex_mask is not None:
        exclude_nonface_vertex_mask = np.asarray(
            exclude_nonface_vertex_mask, dtype=bool
        ).reshape(-1)
        exclude_edge = np.zeros(edges.shape[0], dtype=bool)
        exclude_edge |= (~face_vertex_mask[edges[:, 0]]) & exclude_nonface_vertex_mask[edges[:, 0]]
        exclude_edge |= (~face_vertex_mask[edges[:, 1]]) & exclude_nonface_vertex_mask[edges[:, 1]]
        cross_face_no_exclude &= cross_face & ~exclude_edge
    cross_cavity = cavity_vertex_mask[edges[:, 0]] != cavity_vertex_mask[edges[:, 1]]
    cross_region_pair = np.zeros(edges.shape[0], dtype=bool)
    if region_a_vertex_mask is not None and region_b_vertex_mask is not None:
        region_a_vertex_mask = np.asarray(region_a_vertex_mask, dtype=bool).reshape(-1)
        region_b_vertex_mask = np.asarray(region_b_vertex_mask, dtype=bool).reshape(-1)
        cross_region_pair = region_a_vertex_mask[edges[:, 0]] & region_b_vertex_mask[edges[:, 1]]
        cross_region_pair |= region_b_vertex_mask[edges[:, 0]] & region_a_vertex_mask[edges[:, 1]]
    region_masks = {
        "face_deep_interior": edge_mask(deep_face_vertices),
        "face_boundary_ring": edge_mask(boundary_ring_vertices),
        "face_all": edge_mask(face_vertex_mask),
        "face_nonface_seam": valid & cross_face,
        "face_nonface_seam_no_ear": valid & cross_face_no_exclude,
        "forehead_skull_cross": valid & cross_region_pair,
        "cavity_rim": valid & cross_cavity & face_vertex_mask[edges].any(axis=1),
        "all": valid,
    }

    def summarize(mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {
                "count": 0.0,
                "ratio_median": float("nan"),
                "ratio_p90": float("nan"),
                "ratio_p99": float("nan"),
                "ratio_max": float("nan"),
                "log_dev_median": float("nan"),
                "fraction_above_warn": float("nan"),
                "fraction_above_bad": float("nan"),
            }
        values = ratio[mask]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {
                "count": float(mask.sum()),
                "ratio_median": float("nan"),
                "ratio_p90": float("nan"),
                "ratio_p99": float("nan"),
                "ratio_max": float("nan"),
                "log_dev_median": float("nan"),
                "fraction_above_warn": float("nan"),
                "fraction_above_bad": float("nan"),
            }
        log_finite = log_dev[mask][np.isfinite(values)]
        return {
            "count": float(mask.sum()),
            "ratio_median": float(np.median(finite)),
            "ratio_p90": float(np.percentile(finite, 90)),
            "ratio_p99": float(np.percentile(finite, 99)),
            "ratio_max": float(np.max(finite)),
            "log_dev_median": float(np.median(log_finite)) if log_finite.size else float("nan"),
            "fraction_above_warn": float(np.mean(finite > float(stretch_warn_ratio))),
            "fraction_above_bad": float(np.mean(finite > float(stretch_bad_ratio))),
        }

    regions = {name: summarize(mask) for name, mask in region_masks.items()}
    return {
        "num_edges": int(edges.shape[0]),
        "num_valid_edges": int(valid.sum()),
        "median_edge_length": length_scale,
        "stretch_warn_ratio": float(stretch_warn_ratio),
        "stretch_bad_ratio": float(stretch_bad_ratio),
        "regions": regions,
    }


def count_inverted_faces(
    reference_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    threshold: float = 0.0,
) -> int:
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return 0
    ref_normals = face_normals(reference_vertices, faces, normalize=True)
    cur_normals = face_normals(vertices, faces, normalize=True)
    dots = np.sum(ref_normals * cur_normals, axis=1)
    return int(np.count_nonzero(dots < float(threshold)))


def inverted_face_vertex_weights(
    reference_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    threshold: float = 0.0,
) -> np.ndarray:
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = vertices.shape[0]
    weights = np.zeros(n_vertices, dtype=np.float64)
    if faces.size == 0:
        return weights
    ref_normals = face_normals(reference_vertices, faces, normalize=True)
    cur_normals = face_normals(vertices, faces, normalize=True)
    dots = np.sum(ref_normals * cur_normals, axis=1)
    inverted = dots < float(threshold)
    if not np.any(inverted):
        return weights
    inv_faces = faces[inverted]
    for tri in inv_faces:
        weights[tri] = 1.0
    return weights


def clamp_relative_vertex_displacement(
    reference_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    max_displacement: float | np.ndarray,
) -> np.ndarray:
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    candidate_vertices = np.asarray(candidate_vertices, dtype=np.float64)
    delta = candidate_vertices - reference_vertices
    delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
    if np.ndim(max_displacement) == 0:
        limit = np.full((reference_vertices.shape[0], 1), float(max_displacement), dtype=np.float64)
    else:
        limit = np.asarray(max_displacement, dtype=np.float64).reshape(-1, 1)
    scale = np.ones_like(delta_norm)
    move = delta_norm > np.maximum(limit, 1e-12)
    scale[move] = limit[move] / delta_norm[move]
    return reference_vertices + delta * scale


def retarget_affine_field_vertices(
    source_vertices: np.ndarray,
    affine_field: np.ndarray,
    target_vertices: np.ndarray,
) -> np.ndarray:
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    affine_field = np.asarray(affine_field, dtype=np.float64)
    target_vertices = np.asarray(target_vertices, dtype=np.float64)
    n_vertices = source_vertices.shape[0]
    blocks = affine_field.reshape(n_vertices, 4, 3).copy()
    current = apply_affine_field(source_vertices, affine_field)
    blocks[:, 3, :] += target_vertices - current
    return blocks.reshape(-1, 3)


def restore_region_pair_hairline_template_layout(
    template_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    region_a_mask: np.ndarray,
    region_b_mask: np.ndarray,
    skull_anchor_mask: np.ndarray,
    forehead_adjust_mask: np.ndarray,
    *,
    forehead_blend: float = 0.72,
) -> np.ndarray:
    """
    Re-anchor the skull side to the template and pull movable forehead seam
    vertices toward the template edge offset. The HACK template hairline is
    smooth (~5deg); ICP folds at this seam show up as visible steps.
    """
    template_vertices = np.asarray(template_vertices, dtype=np.float64)
    out = np.asarray(vertices, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    region_a_mask = np.asarray(region_a_mask, dtype=bool).reshape(-1)
    region_b_mask = np.asarray(region_b_mask, dtype=bool).reshape(-1)
    skull_anchor_mask = np.asarray(skull_anchor_mask, dtype=bool).reshape(-1)
    forehead_adjust_mask = np.asarray(forehead_adjust_mask, dtype=bool).reshape(-1)
    forehead_blend = float(np.clip(forehead_blend, 0.0, 1.0))
    if faces.size == 0 or forehead_blend <= 0.0:
        return out

    if np.any(skull_anchor_mask):
        out[skull_anchor_mask] = template_vertices[skull_anchor_mask]

    edges = extract_edges(faces)
    cross = region_a_mask[edges[:, 0]] & region_b_mask[edges[:, 1]]
    cross |= region_b_mask[edges[:, 0]] & region_a_mask[edges[:, 1]]
    if not np.any(cross):
        return out

    accum = np.zeros_like(out)
    counts = np.zeros(out.shape[0], dtype=np.float64)
    for v0, v1 in edges[cross]:
        if face_mask[v0] and not face_mask[v1]:
            face_id, skull_id = int(v0), int(v1)
        elif face_mask[v1] and not face_mask[v0]:
            face_id, skull_id = int(v1), int(v0)
        else:
            continue
        template_offset = template_vertices[face_id] - template_vertices[skull_id]
        target = out[skull_id] + template_offset
        accum[face_id] += target
        counts[face_id] += 1.0

    active = (counts > 0.0) & forehead_adjust_mask
    if np.any(active):
        target = accum[active] / counts[active, None]
        out[active] = (1.0 - forehead_blend) * out[active] + forehead_blend * target
    return out


def repair_inverted_faces(
    reference_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 6,
    blend_step: float = 0.55,
    smooth_iterations: int = 2,
    smooth_step: float = 0.35,
    max_restore_displacement_scale: float = 1.5,
    protect_vertex_band: np.ndarray | None = None,
    protect_strength: float = 0.85,
) -> np.ndarray:
    """
    Iteratively pull inverted regions back toward the reference mesh while
    preserving the rest of the deformation.

    ``protect_vertex_band`` suppresses reference pull on detail regions so
    wrinkles and local shape are not flattened during repair.
    """
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0 or int(iterations) <= 0:
        return vertices

    length_scale = float(median_edge_length(reference_vertices, faces))
    max_restore = max(float(max_restore_displacement_scale) * max(length_scale, 1e-12), 1e-12)
    for _ in range(int(iterations)):
        inv_band = inverted_face_vertex_weights(reference_vertices, vertices, faces)
        if not np.any(inv_band > 1e-6):
            break
        inv_band = smooth_vertex_scalar_field(
            inv_band,
            faces,
            iterations=int(smooth_iterations),
            step=float(smooth_step),
        )
        inv_band = np.clip(inv_band, 0.0, 1.0)
        if protect_vertex_band is not None:
            protect = np.clip(np.asarray(protect_vertex_band, dtype=np.float64).reshape(-1), 0.0, 1.0)
            inv_band = inv_band * (1.0 - float(protect_strength) * protect)
        restored = (
            vertices * (1.0 - float(blend_step) * inv_band[:, None])
            + reference_vertices * (float(blend_step) * inv_band[:, None])
        )
        vertices = clamp_relative_vertex_displacement(reference_vertices, restored, max_restore)
    return vertices


def evaluate_mesh_validity(
    reference_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, float]:
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return {
            "num_inverted_faces": 0.0,
            "fraction_inverted_faces": 0.0,
            "num_degenerate_faces": 0.0,
            "fraction_degenerate_faces": 0.0,
        }
    ref_normals_unit = face_normals(reference_vertices, faces, normalize=True)
    cur_normals_unit = face_normals(vertices, faces, normalize=True)
    ref_area = np.linalg.norm(face_normals(reference_vertices, faces, normalize=False), axis=1)
    cur_area = np.linalg.norm(face_normals(vertices, faces, normalize=False), axis=1)
    area_eps = np.maximum(1e-10 * np.maximum(ref_area, 1e-12), 1e-12)
    inverted = (np.sum(ref_normals_unit * cur_normals_unit, axis=1) < 0.0) | (cur_area < area_eps)
    degenerate = cur_area < area_eps
    n_faces = float(faces.shape[0])
    return {
        "num_inverted_faces": float(np.count_nonzero(inverted)),
        "fraction_inverted_faces": float(np.count_nonzero(inverted) / n_faces),
        "num_degenerate_faces": float(np.count_nonzero(degenerate)),
        "fraction_degenerate_faces": float(np.count_nonzero(degenerate) / n_faces),
    }
