from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import scipy.sparse as sp

from geometry import apply_affine_field, squared_distances, to_homogeneous


@dataclass
class LinearSystem:
    """
    Sparse least-squares system A X ~= B.

    X is the stacked row-vector affine field with shape (4*n_vertices, 3).
    """

    A: sp.csr_matrix
    B: np.ndarray
    row_counts: dict[str, int]
    axis_extra_A: Optional[tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]] = None
    axis_extra_B: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None


def _as_vertex_weights(
    weights: Optional[np.ndarray],
    n_vertices: int,
    default: float = 1.0,
) -> np.ndarray:
    if weights is None:
        return np.full(n_vertices, float(default), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.shape[0] != n_vertices:
        raise ValueError(f"weights must have length {n_vertices}, got {weights.shape[0]}")
    return weights


def build_smoothness_system(
    edges: np.ndarray,
    n_vertices: int,
    stiffness: float,
    gamma: float = 1.0,
    edge_weights: Optional[np.ndarray] = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Build rows for sum_edges ||G (X_i - X_j)||_F^2.

    The per-vertex affine block X_i is 4x3. Rows 0:3 are the linear part and
    row 3 is translation. gamma controls the translation regularization weight.
    """
    edges = np.asarray(edges, dtype=np.int64)
    n_vertices = int(n_vertices)
    if edges.size == 0 or stiffness <= 0.0:
        return sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    if edge_weights is None:
        edge_weights = np.ones(edges.shape[0], dtype=np.float64)
    else:
        edge_weights = np.asarray(edge_weights, dtype=np.float64).reshape(-1)
        if edge_weights.shape[0] != edges.shape[0]:
            raise ValueError(f"edge_weights must have length {edges.shape[0]}, got {edge_weights.shape[0]}")

    row = []
    col = []
    data = []
    base_g = np.array([1.0, 1.0, 1.0, float(gamma)], dtype=np.float64) * float(stiffness)
    out_edge_idx = 0
    for e_idx, (i, j) in enumerate(edges):
        edge_scale = float(np.sqrt(max(edge_weights[e_idx], 0.0)))
        if edge_scale <= 0.0:
            continue
        g = base_g * edge_scale
        for k in range(4):
            r = 4 * out_edge_idx + k
            row.extend([r, r])
            col.extend([4 * int(i) + k, 4 * int(j) + k])
            data.extend([g[k], -g[k]])
        out_edge_idx += 1

    if out_edge_idx == 0:
        return sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    A = sp.coo_matrix(
        (data, (row, col)),
        shape=(4 * out_edge_idx, 4 * n_vertices),
        dtype=np.float64,
    ).tocsr()
    B = np.zeros((A.shape[0], 3), dtype=np.float64)
    return A, B


def build_vertex_fit_system(
    source_vertices: np.ndarray,
    target_points: np.ndarray,
    correspondence_weights: Optional[np.ndarray] = None,
    data_weight: float = 1.0,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Build rows for fixed correspondences: sum_i w_i ||v_i^h X_i - u_i||^2.

    Returns A, B, and the valid source vertex indices represented by the rows.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if source_vertices.ndim != 2 or source_vertices.shape[1] != 3:
        raise ValueError(f"source_vertices must have shape (n, 3), got {source_vertices.shape}")
    if target_points.shape != source_vertices.shape:
        raise ValueError(
            f"target_points must have shape {source_vertices.shape}, got {target_points.shape}"
        )

    n_vertices = source_vertices.shape[0]
    weights = _as_vertex_weights(correspondence_weights, n_vertices)
    finite = np.isfinite(target_points).all(axis=1)
    valid = np.flatnonzero((weights > 0.0) & finite)
    if valid.size == 0 or data_weight <= 0.0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            valid,
        )

    vh = to_homogeneous(source_vertices[valid])
    row = np.repeat(np.arange(valid.size), 4)
    col = (4 * valid[:, None] + np.arange(4, dtype=np.int64)[None, :]).reshape(-1)
    multipliers = np.sqrt(float(data_weight) * weights[valid])
    data = (vh * multipliers[:, None]).reshape(-1)
    A = sp.coo_matrix(
        (data, (row, col)),
        shape=(valid.size, 4 * n_vertices),
        dtype=np.float64,
    ).tocsr()
    B = target_points[valid] * multipliers[:, None]
    return A, B, valid


def build_position_prior_system(
    source_vertices: np.ndarray,
    reference_vertices: Optional[np.ndarray] = None,
    vertex_weights: Optional[np.ndarray] = None,
    axis_weights: Optional[np.ndarray] = None,
    prior_weight: float = 0.0,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Penalize deformed vertex positions drifting from reference vertices.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    n_vertices = source_vertices.shape[0]
    if reference_vertices is None or prior_weight <= 0.0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    if reference_vertices.shape != source_vertices.shape:
        raise ValueError(
            f"reference_vertices must have shape {source_vertices.shape}, got {reference_vertices.shape}"
    )
    weights = _as_vertex_weights(vertex_weights, n_vertices) if vertex_weights is not None else np.ones(n_vertices)
    if axis_weights is not None:
        axis_weights = np.asarray(axis_weights, dtype=np.float64)
        if axis_weights.shape != (n_vertices, 3):
            raise ValueError(f"axis_weights must have shape {(n_vertices, 3)}, got {axis_weights.shape}")
        if not np.allclose(axis_weights, axis_weights[:, :1]):
            raise ValueError("directional axis_weights require build_position_prior_axis_systems")
        weights = weights * np.clip(axis_weights[:, 0], 0.0, None)
    valid = np.flatnonzero((weights > 0.0) & np.isfinite(reference_vertices).all(axis=1))
    if valid.size == 0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    vh = to_homogeneous(source_vertices[valid])
    row = np.repeat(np.arange(valid.size), 4)
    col = (4 * valid[:, None] + np.arange(4, dtype=np.int64)[None, :]).reshape(-1)
    multipliers = np.sqrt(float(prior_weight) * weights[valid])
    data = (vh * multipliers[:, None]).reshape(-1)
    A = sp.coo_matrix(
        (data, (row, col)),
        shape=(valid.size, 4 * n_vertices),
        dtype=np.float64,
    ).tocsr()
    B = reference_vertices[valid] * multipliers[:, None]
    return A, B


def build_position_prior_axis_systems(
    source_vertices: np.ndarray,
    reference_vertices: Optional[np.ndarray] = None,
    vertex_weights: Optional[np.ndarray] = None,
    axis_weights: Optional[np.ndarray] = None,
    prior_weight: float = 0.0,
) -> tuple[
    tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Build per-output-axis raw-position prior rows.

    The main least-squares system shares A for x/y/z outputs. Directional
    position weights therefore need per-axis extra rows that are appended only
    when solving the corresponding RHS column.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    n_vertices = source_vertices.shape[0]
    empty_A = sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64)
    empty_b = np.zeros(0, dtype=np.float64)
    empty = ((empty_A, empty_A, empty_A), (empty_b, empty_b, empty_b))
    if reference_vertices is None or prior_weight <= 0.0:
        return empty

    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    if reference_vertices.shape != source_vertices.shape:
        raise ValueError(
            f"reference_vertices must have shape {source_vertices.shape}, got {reference_vertices.shape}"
        )
    weights = _as_vertex_weights(vertex_weights, n_vertices) if vertex_weights is not None else np.ones(n_vertices)
    if axis_weights is None:
        axis_weights = np.ones((n_vertices, 3), dtype=np.float64)
    else:
        axis_weights = np.asarray(axis_weights, dtype=np.float64)
        if axis_weights.shape != (n_vertices, 3):
            raise ValueError(f"axis_weights must have shape {(n_vertices, 3)}, got {axis_weights.shape}")

    systems_A = []
    systems_b = []
    finite = np.isfinite(reference_vertices).all(axis=1)
    for axis in range(3):
        axis_vertex_weights = weights * np.clip(axis_weights[:, axis], 0.0, None)
        valid = np.flatnonzero((axis_vertex_weights > 0.0) & finite)
        if valid.size == 0:
            systems_A.append(empty_A)
            systems_b.append(empty_b)
            continue
        vh = to_homogeneous(source_vertices[valid])
        row = np.repeat(np.arange(valid.size), 4)
        col = (4 * valid[:, None] + np.arange(4, dtype=np.int64)[None, :]).reshape(-1)
        multipliers = np.sqrt(float(prior_weight) * axis_vertex_weights[valid])
        data = (vh * multipliers[:, None]).reshape(-1)
        systems_A.append(
            sp.coo_matrix(
                (data, (row, col)),
                shape=(valid.size, 4 * n_vertices),
                dtype=np.float64,
            ).tocsr()
        )
        systems_b.append(reference_vertices[valid, axis] * multipliers)

    return (
        (systems_A[0], systems_A[1], systems_A[2]),
        (systems_b[0], systems_b[1], systems_b[2]),
    )


def build_landmark_system(
    source_vertices: np.ndarray,
    landmark_vertex_indices: Optional[np.ndarray],
    landmark_targets: Optional[np.ndarray],
    landmark_weights: Optional[np.ndarray] = None,
    landmark_weight: float = 1.0,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Build optional landmark rows using source vertex indices and target positions.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    n_vertices = source_vertices.shape[0]
    if landmark_vertex_indices is None or landmark_targets is None or landmark_weight <= 0.0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    ids = np.asarray(landmark_vertex_indices, dtype=np.int64).reshape(-1)
    targets = np.asarray(landmark_targets, dtype=np.float64).reshape((-1, 3))
    if ids.shape[0] != targets.shape[0]:
        raise ValueError(
            f"landmark ids and targets must have equal length, got {ids.shape[0]} and {targets.shape[0]}"
        )
    if ids.size == 0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            ids,
        )
    if ids.min() < 0 or ids.max() >= n_vertices:
        raise ValueError("landmark_vertex_indices contain indices outside source_vertices")

    if landmark_weights is None:
        weights = np.ones(ids.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(landmark_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != ids.shape[0]:
            raise ValueError(
                f"landmark_weights must have length {ids.shape[0]}, got {weights.shape[0]}"
            )

    finite = np.isfinite(targets).all(axis=1)
    valid_rows = np.flatnonzero((weights > 0.0) & finite)
    ids = ids[valid_rows]
    targets = targets[valid_rows]
    weights = weights[valid_rows]
    if ids.size == 0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            ids,
        )

    vh = to_homogeneous(source_vertices[ids])
    row = np.repeat(np.arange(ids.size), 4)
    col = (4 * ids[:, None] + np.arange(4, dtype=np.int64)[None, :]).reshape(-1)
    multipliers = np.sqrt(float(landmark_weight) * weights)
    data = (vh * multipliers[:, None]).reshape(-1)
    A = sp.coo_matrix(
        (data, (row, col)),
        shape=(ids.size, 4 * n_vertices),
        dtype=np.float64,
    ).tocsr()
    B = targets * multipliers[:, None]
    return A, B, ids


def build_barycentric_landmark_system(
    source_vertices: np.ndarray,
    landmark_face_vertices: Optional[np.ndarray],
    landmark_barycentric: Optional[np.ndarray],
    landmark_targets: Optional[np.ndarray],
    landmark_weights: Optional[np.ndarray] = None,
    landmark_weight: float = 1.0,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Build landmark rows for points defined by triangle vertices and barycentric weights.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    n_vertices = source_vertices.shape[0]
    if (
        landmark_face_vertices is None
        or landmark_barycentric is None
        or landmark_targets is None
        or landmark_weight <= 0.0
    ):
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    face_vertices = np.asarray(landmark_face_vertices, dtype=np.int64).reshape((-1, 3))
    bary = np.asarray(landmark_barycentric, dtype=np.float64).reshape((-1, 3))
    targets = np.asarray(landmark_targets, dtype=np.float64).reshape((-1, 3))
    if face_vertices.shape[0] != bary.shape[0] or face_vertices.shape[0] != targets.shape[0]:
        raise ValueError(
            "landmark face vertices, barycentric weights, and targets must have equal length, "
            f"got {face_vertices.shape[0]}, {bary.shape[0]}, {targets.shape[0]}"
        )
    if face_vertices.size == 0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    if face_vertices.min() < 0 or face_vertices.max() >= n_vertices:
        raise ValueError("landmark_face_vertices contain indices outside source_vertices")

    if landmark_weights is None:
        weights = np.ones(face_vertices.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(landmark_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != face_vertices.shape[0]:
            raise ValueError(
                f"landmark_weights must have length {face_vertices.shape[0]}, got {weights.shape[0]}"
            )

    finite = np.isfinite(targets).all(axis=1) & np.isfinite(bary).all(axis=1)
    valid_rows = np.flatnonzero((weights > 0.0) & finite)
    face_vertices = face_vertices[valid_rows]
    bary = bary[valid_rows]
    targets = targets[valid_rows]
    weights = weights[valid_rows]
    if face_vertices.size == 0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            valid_rows.astype(np.int64),
        )

    multipliers = np.sqrt(float(landmark_weight) * weights)
    row = []
    col = []
    data = []
    vh = to_homogeneous(source_vertices)
    for out_r, (tri, bc, multiplier) in enumerate(zip(face_vertices, bary, multipliers)):
        for corner in range(3):
            vid = int(tri[corner])
            coeff = float(multiplier) * float(bc[corner])
            for k in range(4):
                row.append(out_r)
                col.append(4 * vid + k)
                data.append(coeff * vh[vid, k])

    A = sp.coo_matrix(
        (data, (row, col)),
        shape=(face_vertices.shape[0], 4 * n_vertices),
        dtype=np.float64,
    ).tocsr()
    B = targets * multipliers[:, None]
    return A, B, valid_rows.astype(np.int64)


def build_barycentric_delta_landmark_system(
    source_vertices: np.ndarray,
    face_vertices_a: Optional[np.ndarray],
    barycentric_a: Optional[np.ndarray],
    face_vertices_b: Optional[np.ndarray],
    barycentric_b: Optional[np.ndarray],
    target_deltas: Optional[np.ndarray],
    weights: Optional[np.ndarray] = None,
    landmark_weight: float = 1.0,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Build rows for landmark-pair deltas: point_a - point_b ~= target_delta.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    n_vertices = source_vertices.shape[0]
    if (
        face_vertices_a is None
        or barycentric_a is None
        or face_vertices_b is None
        or barycentric_b is None
        or target_deltas is None
        or landmark_weight <= 0.0
    ):
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    fva = np.asarray(face_vertices_a, dtype=np.int64).reshape((-1, 3))
    fvb = np.asarray(face_vertices_b, dtype=np.int64).reshape((-1, 3))
    bca = np.asarray(barycentric_a, dtype=np.float64).reshape((-1, 3))
    bcb = np.asarray(barycentric_b, dtype=np.float64).reshape((-1, 3))
    deltas = np.asarray(target_deltas, dtype=np.float64).reshape((-1, 3))
    n_rows = fva.shape[0]
    if any(arr.shape[0] != n_rows for arr in (fvb, bca, bcb, deltas)):
        raise ValueError("delta landmark arrays must have equal length")
    if n_rows == 0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    if min(fva.min(), fvb.min()) < 0 or max(fva.max(), fvb.max()) >= n_vertices:
        raise ValueError("delta landmark face vertices contain indices outside source_vertices")

    if weights is None:
        weights = np.ones(n_rows, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != n_rows:
            raise ValueError(f"delta landmark weights must have length {n_rows}, got {weights.shape[0]}")

    finite = (
        np.isfinite(bca).all(axis=1)
        & np.isfinite(bcb).all(axis=1)
        & np.isfinite(deltas).all(axis=1)
        & (weights > 0.0)
    )
    valid_rows = np.flatnonzero(finite)
    fva = fva[valid_rows]
    fvb = fvb[valid_rows]
    bca = bca[valid_rows]
    bcb = bcb[valid_rows]
    deltas = deltas[valid_rows]
    weights = weights[valid_rows]
    if fva.shape[0] == 0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            valid_rows.astype(np.int64),
        )

    vh = to_homogeneous(source_vertices)
    multipliers = np.sqrt(float(landmark_weight) * weights)
    row = []
    col = []
    data = []
    for out_r, (tri_a, bc_a, tri_b, bc_b, multiplier) in enumerate(zip(fva, bca, fvb, bcb, multipliers)):
        for corner in range(3):
            vid = int(tri_a[corner])
            coeff = float(multiplier) * float(bc_a[corner])
            for k in range(4):
                row.append(out_r)
                col.append(4 * vid + k)
                data.append(coeff * vh[vid, k])
        for corner in range(3):
            vid = int(tri_b[corner])
            coeff = -float(multiplier) * float(bc_b[corner])
            for k in range(4):
                row.append(out_r)
                col.append(4 * vid + k)
                data.append(coeff * vh[vid, k])

    A = sp.coo_matrix(
        (data, (row, col)),
        shape=(fva.shape[0], 4 * n_vertices),
        dtype=np.float64,
    ).tocsr()
    B = deltas * multipliers[:, None]
    return A, B, valid_rows.astype(np.int64)


def build_affine_prior_system(
    reference_affine_field: Optional[np.ndarray],
    n_vertices: int,
    prior_weight: float = 0.0,
    vertex_weights: Optional[np.ndarray] = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Penalize per-vertex affine blocks drifting too far from a reference field.

    Smoothness alone has zero cost for any constant affine transform, including
    a depth-flattening transform. This prior anchors that global mode.
    """
    if reference_affine_field is None or prior_weight <= 0.0:
        return (
            sp.csr_matrix((0, 4 * int(n_vertices)), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    reference_affine_field = np.asarray(reference_affine_field, dtype=np.float64)
    expected_shape = (4 * int(n_vertices), 3)
    if reference_affine_field.shape != expected_shape:
        raise ValueError(
            f"reference_affine_field must have shape {expected_shape}, "
            f"got {reference_affine_field.shape}"
        )

    if vertex_weights is None:
        row_weights = np.full(4 * int(n_vertices), float(prior_weight), dtype=np.float64)
    else:
        vertex_weights = np.asarray(vertex_weights, dtype=np.float64).reshape(-1)
        if vertex_weights.shape[0] != int(n_vertices):
            raise ValueError(
                f"affine prior vertex weights must have length {int(n_vertices)}, "
                f"got {vertex_weights.shape[0]}"
            )
        row_weights = np.repeat(float(prior_weight) * vertex_weights, 4)

    active = row_weights > 0.0
    if not np.any(active):
        return (
            sp.csr_matrix((0, 4 * int(n_vertices)), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    row_ids = np.flatnonzero(active)
    multipliers = np.sqrt(row_weights[active])
    A = sp.coo_matrix(
        (multipliers, (np.arange(row_ids.size), row_ids)),
        shape=(row_ids.size, 4 * int(n_vertices)),
        dtype=np.float64,
    ).tocsr()
    B = reference_affine_field[row_ids] * multipliers[:, None]
    return A, B


def build_laplacian_coordinate_system(
    source_vertices: np.ndarray,
    edges: np.ndarray,
    weight: float = 0.0,
    vertex_weights: Optional[np.ndarray] = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Preserve uniform Laplacian coordinates of deformed vertices.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    n_vertices = source_vertices.shape[0]
    if edges.size == 0 or weight <= 0.0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )
    weights = _as_vertex_weights(vertex_weights, n_vertices) if vertex_weights is not None else np.ones(n_vertices)

    neighbors: list[list[int]] = [[] for _ in range(n_vertices)]
    for i, j in edges:
        i = int(i)
        j = int(j)
        neighbors[i].append(j)
        neighbors[j].append(i)

    active = np.array([(len(nbs) > 0) and (weights[i] > 0.0) for i, nbs in enumerate(neighbors)], dtype=bool)
    active_ids = np.flatnonzero(active)
    vh = to_homogeneous(source_vertices)
    row = []
    col = []
    data = []
    B = np.zeros((active_ids.size, 3), dtype=np.float64)
    for out_r, i in enumerate(active_ids):
        nbs = neighbors[int(i)]
        inv_deg = 1.0 / float(len(nbs))
        multiplier = float(np.sqrt(weight * weights[int(i)]))
        coeffs = [(int(i), 1.0)] + [(int(nb), -inv_deg) for nb in nbs]
        for vid, coeff in coeffs:
            for k in range(4):
                row.append(out_r)
                col.append(4 * vid + k)
                data.append(multiplier * coeff * vh[vid, k])
        B[out_r] = multiplier * (
            source_vertices[int(i)] - np.mean(source_vertices[np.asarray(nbs, dtype=np.int64)], axis=0)
        )

    A = sp.coo_matrix(
        (data, (row, col)),
        shape=(active_ids.size, 4 * n_vertices),
        dtype=np.float64,
    ).tocsr()
    return A, B


def build_vertex_mass_system(
    source_vertices: np.ndarray,
    edges: np.ndarray,
    weight: float = 0.0,
    vertex_weights: Optional[np.ndarray] = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Linearized mass_loss-style regularizer based on mean incident edge length.

    The torch mass_loss in eval2.py uses per-vertex mean incident edge length as
    a mass proxy. Exact edge lengths are nonlinear in this least-squares form,
    so this term preserves edge vectors normalized by the source per-vertex
    mass. It discourages local mean edge length from drifting unevenly.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    n_vertices = source_vertices.shape[0]
    if edges.size == 0 or weight <= 0.0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )
    weights = _as_vertex_weights(vertex_weights, n_vertices) if vertex_weights is not None else np.ones(n_vertices)

    incident_lengths: list[list[float]] = [[] for _ in range(n_vertices)]
    for i, j in edges:
        i = int(i)
        j = int(j)
        length = float(np.linalg.norm(source_vertices[i] - source_vertices[j]))
        if length <= 1e-12:
            continue
        incident_lengths[i].append(length)
        incident_lengths[j].append(length)

    vertex_mass = np.ones(n_vertices, dtype=np.float64)
    nonempty = np.array([len(lengths) > 0 for lengths in incident_lengths], dtype=bool)
    for i in np.flatnonzero(nonempty):
        vertex_mass[int(i)] = float(np.mean(incident_lengths[int(i)]))
    global_mass = float(np.mean(vertex_mass[nonempty])) if np.any(nonempty) else 1.0
    vertex_mass = np.maximum(vertex_mass, max(global_mass * 1e-3, 1e-12))

    multiplier = float(np.sqrt(weight))
    vh = to_homogeneous(source_vertices)
    row = []
    col = []
    data = []
    B_rows = []
    out_r = 0
    for i, j in edges:
        i = int(i)
        j = int(j)
        edge_weight = 0.5 * (weights[i] + weights[j])
        if edge_weight <= 0.0:
            continue
        local_mass = 0.5 * (vertex_mass[i] + vertex_mass[j])
        local_multiplier = multiplier * float(np.sqrt(edge_weight)) * (global_mass / local_mass)
        for k in range(4):
            row.extend([out_r, out_r])
            col.extend([4 * i + k, 4 * j + k])
            data.extend([local_multiplier * vh[i, k], -local_multiplier * vh[j, k]])
        B_rows.append(local_multiplier * (source_vertices[i] - source_vertices[j]))
        out_r += 1

    if out_r == 0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    A = sp.coo_matrix(
        (data, (row, col)),
        shape=(out_r, 4 * n_vertices),
        dtype=np.float64,
    ).tocsr()
    B = np.asarray(B_rows, dtype=np.float64).reshape(out_r, 3)
    return A, B


def build_edge_vector_system(
    source_vertices: np.ndarray,
    edges: np.ndarray,
    weight: float = 0.0,
    edge_weights: Optional[np.ndarray] = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """
    Preserve selected source edge vectors after deformation.

    This is a linear proxy for local edge-length preservation:
    (v_i^h X_i - v_j^h X_j) ~= (v_i - v_j).
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    n_vertices = source_vertices.shape[0]
    if edges.size == 0 or weight <= 0.0 or edge_weights is None:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    edge_weights = np.asarray(edge_weights, dtype=np.float64).reshape(-1)
    if edge_weights.shape[0] != edges.shape[0]:
        raise ValueError(f"edge_weights must have length {edges.shape[0]}, got {edge_weights.shape[0]}")

    vh = to_homogeneous(source_vertices)
    base_multiplier = float(np.sqrt(weight))
    row = []
    col = []
    data = []
    B_rows = []
    out_r = 0
    for e_idx, (i, j) in enumerate(edges):
        local_weight = float(edge_weights[e_idx])
        if local_weight <= 0.0:
            continue
        i = int(i)
        j = int(j)
        multiplier = base_multiplier * float(np.sqrt(local_weight))
        for k in range(4):
            row.extend([out_r, out_r])
            col.extend([4 * i + k, 4 * j + k])
            data.extend([multiplier * vh[i, k], -multiplier * vh[j, k]])
        B_rows.append(multiplier * (source_vertices[i] - source_vertices[j]))
        out_r += 1

    if out_r == 0:
        return (
            sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
        )

    A = sp.coo_matrix(
        (data, (row, col)),
        shape=(out_r, 4 * n_vertices),
        dtype=np.float64,
    ).tocsr()
    B = np.asarray(B_rows, dtype=np.float64).reshape(out_r, 3)
    return A, B


def assemble_linear_system(
    source_vertices: np.ndarray,
    edges: np.ndarray,
    target_points: np.ndarray,
    correspondence_weights: Optional[np.ndarray],
    stiffness: float,
    gamma: float = 1.0,
    smoothness_edge_weights: Optional[np.ndarray] = None,
    data_weight: float = 1.0,
    landmarks: Optional[Mapping[str, np.ndarray]] = None,
    landmark_weight: float = 0.0,
    affine_prior: Optional[np.ndarray] = None,
    affine_prior_weight: float = 0.0,
    affine_prior_vertex_weights: Optional[np.ndarray] = None,
    position_prior_vertices: Optional[np.ndarray] = None,
    position_prior_weight: float = 0.0,
    position_prior_vertex_weights: Optional[np.ndarray] = None,
    position_prior_axis_weights: Optional[np.ndarray] = None,
    laplacian_weight: float = 0.0,
    laplacian_vertex_weights: Optional[np.ndarray] = None,
    vertex_mass_weight: float = 0.0,
    vertex_mass_vertex_weights: Optional[np.ndarray] = None,
    edge_vector_weight: float = 0.0,
    edge_vector_edge_weights: Optional[np.ndarray] = None,
) -> LinearSystem:
    """
    Assemble the fixed-correspondence nonrigid ICP least-squares system.

    landmarks, when provided, may contain:
      - vertex_indices: (k,)
      - target_points: (k, 3)
      - weights: optional (k,)
    """
    n_vertices = np.asarray(source_vertices).shape[0]
    smooth_A, smooth_B = build_smoothness_system(
        edges,
        n_vertices,
        stiffness,
        gamma,
        edge_weights=smoothness_edge_weights,
    )
    data_A, data_B, _ = build_vertex_fit_system(
        source_vertices,
        target_points,
        correspondence_weights,
        data_weight=data_weight,
    )

    if landmarks is None:
        lmk_A, lmk_B, _ = build_landmark_system(
            source_vertices,
            None,
            None,
            landmark_weight=0.0,
        )
    elif "face_vertices" in landmarks and "barycentric" in landmarks:
        lmk_A, lmk_B, _ = build_barycentric_landmark_system(
            source_vertices,
            landmarks.get("face_vertices"),
            landmarks.get("barycentric"),
            landmarks.get("target_points"),
            landmarks.get("weights"),
            landmark_weight=landmark_weight,
        )
    else:
        lmk_A, lmk_B, _ = build_landmark_system(
            source_vertices,
            landmarks.get("vertex_indices"),
            landmarks.get("target_points"),
            landmarks.get("weights"),
            landmark_weight=landmark_weight,
        )
    if landmarks is not None and "delta_face_vertices_a" in landmarks:
        delta_A, delta_B, _ = build_barycentric_delta_landmark_system(
            source_vertices,
            landmarks.get("delta_face_vertices_a"),
            landmarks.get("delta_barycentric_a"),
            landmarks.get("delta_face_vertices_b"),
            landmarks.get("delta_barycentric_b"),
            landmarks.get("delta_target_deltas"),
            landmarks.get("delta_weights"),
            landmark_weight=landmark_weight,
        )
    else:
        delta_A, delta_B, _ = build_barycentric_delta_landmark_system(
            source_vertices,
            None,
            None,
            None,
            None,
            None,
            landmark_weight=0.0,
        )

    prior_A, prior_B = build_affine_prior_system(
        affine_prior,
        n_vertices,
        prior_weight=affine_prior_weight,
        vertex_weights=affine_prior_vertex_weights,
    )
    if position_prior_axis_weights is None:
        pos_A, pos_B = build_position_prior_system(
            source_vertices,
            reference_vertices=position_prior_vertices,
            vertex_weights=position_prior_vertex_weights,
            prior_weight=position_prior_weight,
        )
        axis_extra_A = None
        axis_extra_B = None
    else:
        pos_A = sp.csr_matrix((0, 4 * n_vertices), dtype=np.float64)
        pos_B = np.zeros((0, 3), dtype=np.float64)
        axis_extra_A, axis_extra_B = build_position_prior_axis_systems(
            source_vertices,
            reference_vertices=position_prior_vertices,
            vertex_weights=position_prior_vertex_weights,
            axis_weights=position_prior_axis_weights,
            prior_weight=position_prior_weight,
        )
    lap_A, lap_B = build_laplacian_coordinate_system(
        source_vertices,
        edges,
        weight=laplacian_weight,
        vertex_weights=laplacian_vertex_weights,
    )
    mass_A, mass_B = build_vertex_mass_system(
        source_vertices,
        edges,
        weight=vertex_mass_weight,
        vertex_weights=vertex_mass_vertex_weights,
    )
    edge_vec_A, edge_vec_B = build_edge_vector_system(
        source_vertices,
        edges,
        weight=edge_vector_weight,
        edge_weights=edge_vector_edge_weights,
    )

    blocks_A = [smooth_A, data_A, lmk_A, delta_A, prior_A, pos_A, lap_A, mass_A, edge_vec_A]
    blocks_B = [smooth_B, data_B, lmk_B, delta_B, prior_B, pos_B, lap_B, mass_B, edge_vec_B]
    A = sp.vstack(blocks_A, format="csr")
    B = np.vstack(blocks_B).astype(np.float64, copy=False)
    return LinearSystem(
        A=A,
        B=B,
        row_counts={
            "smoothness": smooth_A.shape[0],
            "data": data_A.shape[0],
            "landmark": lmk_A.shape[0],
            "landmark_delta": delta_A.shape[0],
            "affine_prior": prior_A.shape[0],
            "position_prior": pos_A.shape[0]
            if axis_extra_A is None
            else int(sum(axis_A.shape[0] for axis_A in axis_extra_A)),
            "laplacian": lap_A.shape[0],
            "vertex_mass": mass_A.shape[0],
            "edge_vector": edge_vec_A.shape[0],
        },
        axis_extra_A=axis_extra_A,
        axis_extra_B=axis_extra_B,
    )


def data_residuals(
    source_vertices: np.ndarray,
    affine_field: np.ndarray,
    target_points: np.ndarray,
    correspondence_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    deformed = apply_affine_field(source_vertices, affine_field)
    target_points = np.asarray(target_points, dtype=np.float64)
    residual = np.linalg.norm(deformed - target_points, axis=1)
    weights = _as_vertex_weights(correspondence_weights, deformed.shape[0])
    valid = (weights > 0.0) & np.isfinite(target_points).all(axis=1)
    out = np.full(deformed.shape[0], np.nan, dtype=np.float64)
    out[valid] = residual[valid]
    return out


def smoothness_residuals(
    affine_field: np.ndarray,
    edges: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    affine_field = np.asarray(affine_field, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return np.empty(0, dtype=np.float64)
    blocks = affine_field.reshape((-1, 4, 3))
    g = np.array([1.0, 1.0, 1.0, float(gamma)], dtype=np.float64).reshape(1, 4, 1)
    diff = (blocks[edges[:, 0]] - blocks[edges[:, 1]]) * g
    return np.linalg.norm(diff.reshape(edges.shape[0], -1), axis=1)


def landmark_residuals(
    source_vertices: np.ndarray,
    affine_field: np.ndarray,
    landmark_vertex_indices: Optional[np.ndarray],
    landmark_targets: Optional[np.ndarray],
    landmark_face_vertices: Optional[np.ndarray] = None,
    landmark_barycentric: Optional[np.ndarray] = None,
) -> np.ndarray:
    if landmark_targets is None:
        return np.empty(0, dtype=np.float64)
    targets = np.asarray(landmark_targets, dtype=np.float64).reshape((-1, 3))
    deformed = apply_affine_field(source_vertices, affine_field)
    if landmark_face_vertices is not None and landmark_barycentric is not None:
        face_vertices = np.asarray(landmark_face_vertices, dtype=np.int64).reshape((-1, 3))
        bary = np.asarray(landmark_barycentric, dtype=np.float64).reshape((-1, 3))
        if face_vertices.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        lmk_points = (deformed[face_vertices] * bary[:, :, None]).sum(axis=1)
        finite = np.isfinite(targets).all(axis=1) & np.isfinite(lmk_points).all(axis=1)
        return np.linalg.norm(lmk_points[finite] - targets[finite], axis=1)

    if landmark_vertex_indices is None:
        return np.empty(0, dtype=np.float64)
    ids = np.asarray(landmark_vertex_indices, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return np.empty(0, dtype=np.float64)
    finite = np.isfinite(targets).all(axis=1)
    return np.linalg.norm(deformed[ids][finite] - targets[finite], axis=1)


def summarize_registration_energy(
    source_vertices: np.ndarray,
    affine_field: np.ndarray,
    edges: np.ndarray,
    target_points: np.ndarray,
    correspondence_weights: Optional[np.ndarray],
    gamma: float = 1.0,
    landmarks: Optional[Mapping[str, np.ndarray]] = None,
) -> dict[str, float]:
    """
    Lightweight diagnostics for logs and tests.
    """
    data_r = data_residuals(source_vertices, affine_field, target_points, correspondence_weights)
    finite_data = data_r[np.isfinite(data_r)]
    smooth_r = smoothness_residuals(affine_field, edges, gamma)
    summary = {
        "data_rmse": float(np.sqrt(np.mean(finite_data**2))) if finite_data.size else float("nan"),
        "data_mean": float(np.mean(finite_data)) if finite_data.size else float("nan"),
        "data_max": float(np.max(finite_data)) if finite_data.size else float("nan"),
        "valid_correspondences": int(finite_data.size),
        "smoothness_mean": float(np.mean(smooth_r)) if smooth_r.size else 0.0,
    }
    if landmarks is not None:
        lmk_r = landmark_residuals(
            source_vertices,
            affine_field,
            landmarks.get("vertex_indices"),
            landmarks.get("target_points"),
            landmarks.get("face_vertices"),
            landmarks.get("barycentric"),
        )
        summary["landmark_rmse"] = float(np.sqrt(np.mean(lmk_r**2))) if lmk_r.size else float("nan")
    return summary


def mean_squared_data_error(
    deformed_vertices: np.ndarray,
    target_points: np.ndarray,
    correspondence_weights: Optional[np.ndarray] = None,
) -> float:
    """
    Convenience p2p data loss for already deformed vertices.
    """
    deformed_vertices = np.asarray(deformed_vertices, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    weights = _as_vertex_weights(correspondence_weights, deformed_vertices.shape[0])
    valid = (weights > 0.0) & np.isfinite(target_points).all(axis=1)
    if not np.any(valid):
        return float("nan")
    sq = squared_distances(deformed_vertices[valid], target_points[valid])
    return float(np.average(sq, weights=weights[valid]))
