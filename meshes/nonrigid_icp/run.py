from __future__ import annotations

import os
import os.path as osp
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence
from tqdm import trange, tqdm
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import trimesh
from scipy.spatial import cKDTree

from energy import assemble_linear_system, mean_squared_data_error, summarize_registration_energy
from geometry import (
    apply_affine_field,
    clamp_relative_vertex_displacement,
    count_inverted_faces,
    extract_edges,
    face_normals,
    geodesic_distance_from_mask,
    identity_affine_field,
    median_edge_length,
    retarget_affine_field_vertices,
    validate_vertices_faces,
    vertex_normals,
)


@dataclass
class NonRigidICPConfig:
    """
    Parameters for optimal-step nonrigid ICP.

    The defaults favor a robust first run on the HACK raw -> BNI meshes. For a
    quicker smoke test, use fewer stiffness values or set max_iterations to 1.
    """

    stiffness_schedule: Sequence[float] = field(
        default_factory=lambda: (5.0, 3.0, 1.6)
    )
    max_iterations: int = 3
    gamma: float = 1.0
    data_weight: float = 3.0
    landmark_weight: float = 2.0
    correspondence_mode: str = "vertex"
    robust_correspondence: bool = False
    robust_tukey_scale: float = 4.685
    robust_min_scale_fraction: float = 1e-4
    normal_angle_threshold_degrees: Optional[float] = None
    distance_threshold: Optional[float] = None
    distance_threshold_scale: Optional[float] = None
    trim_percentile: Optional[float] = 98.0
    trim_percentile_end: Optional[float] = None
    stage_trim_percentiles: Optional[Sequence[float]] = None
    query_fraction_start: float = 1.0
    query_fraction_end: float = 1.0
    stage_query_fractions: Optional[Sequence[float]] = None
    min_stage_query_vertices: int = 280
    reject_target_boundary_correspondences: bool = True
    target_boundary_width_scale: float = 2.0
    target_boundary_weight: float = 0.0
    damping: float = 1e-8
    solver: str = "lsmr"
    solver_tolerance: float = 1e-5
    solver_max_iterations: int = 100
    affine_prior_weight: float = 0.45
    affine_prior_decay: float = 0.5
    position_prior_weight: float = 1.2
    position_prior_decay: float = 0.8
    laplacian_weight: float = 1.2
    vertex_mass_weight: float = 0.7
    edge_vector_weight: float = 3.5
    geometry_weight_decay: float = 0.8
    adaptive_relaxation: bool = True
    adaptive_relaxation_start_stage: int = 1
    adaptive_relaxation_end_stage: Optional[int] = None
    adaptive_residual_percentile: float = 85.0
    adaptive_residual_smooth_iterations: int = 3
    adaptive_residual_smooth_step: float = 0.5
    adaptive_data_boost: float = 0.5
    adaptive_data_boost_end: Optional[float] = None
    adaptive_regularization_multiplier: float = 1.0
    adaptive_min_edge_reg: float = 0.0
    adaptive_vertex_mass_multiplier: float = 0.75
    max_iteration_displacement_scale: float = 0.0
    flip_guard_enabled: bool = False
    flip_guard_blend: float = 0.55
    flip_guard_tolerance: int = 5
    final_projection_blend: float = 1.0
    final_projection_max_displacement_scale: float = 0.0
    landmark_weight_decay: float = 0.5
    robust_correspondence_start_stage: int = 1
    initial_trim_percentile: Optional[float] = None
    final_normal_projection: bool = True
    convergence_tol: float = 1e-5
    min_valid_correspondence_ratio: float = 0.05
    visualize_correspondences: bool = False
    visualize_correspondence_limit: int = 2000
    verbose: bool = True
    camera_depth_attenuation_enabled: bool = False
    camera_depth_axis: int = 2
    camera_depth_min_weight: float = 0.35
    camera_depth_power: float = 2.0
    camera_depth_far_percentile: float = 95.0


@dataclass
class Correspondences:
    points: np.ndarray
    distances: np.ndarray
    weights: np.ndarray
    face_indices: np.ndarray
    target_normals: np.ndarray
    source_normals: np.ndarray
    valid_mask: np.ndarray
    method: str
    stats: dict[str, float]


@dataclass
class RegistrationResult:
    vertices: np.ndarray
    faces: np.ndarray
    affine_field: np.ndarray
    history: list[dict[str, float]]
    correspondences: Correspondences
    config: NonRigidICPConfig


class TargetSurface:
    """
    Thin wrapper around a target triangle mesh for nearest-surface queries.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, face_mask: Optional[np.ndarray] = None):
        vertices, faces = validate_vertices_faces(vertices, faces)
        if face_mask is not None:
            face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
            if face_mask.shape[0] != faces.shape[0]:
                raise ValueError(f"target face_mask must have length {faces.shape[0]}, got {face_mask.shape[0]}")
            faces = faces[face_mask]
            if faces.size == 0:
                raise ValueError("target face_mask selected zero faces")
        self.vertices = vertices
        self.faces = faces
        self.face_normals = face_normals(vertices, faces)
        self.vertex_normals = vertex_normals(vertices, faces)
        self.num_total_vertices = int(vertices.shape[0])
        self.num_query_vertices = int(np.unique(faces.reshape(-1)).shape[0]) if faces.size else 0
        self.num_total_faces = int(face_mask.shape[0]) if face_mask is not None else int(faces.shape[0])
        self.num_query_faces = int(faces.shape[0])
        self.boundary_vertex_mask = self._compute_boundary_vertex_mask()
        self.boundary_face_mask = np.any(self.boundary_vertex_mask[self.faces], axis=1)
        self._boundary_vertex_tree = None
        boundary_vertex_ids = np.flatnonzero(self.boundary_vertex_mask)
        if cKDTree is not None and boundary_vertex_ids.size:
            self._boundary_vertex_tree = cKDTree(self.vertices[boundary_vertex_ids])
        self._boundary_band_cache: dict[float, np.ndarray] = {}
        self._mesh = None
        if trimesh is not None:
            self._mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if cKDTree is None:
            self._vertex_tree = None
            self._query_vertices = None
            self._query_vertex_ids = None
        else:
            query_vertex_ids = np.unique(faces.reshape(-1)).astype(np.int64)
            self._query_vertex_ids = query_vertex_ids
            self._query_vertices = vertices[query_vertex_ids]
            self._vertex_tree = cKDTree(self._query_vertices)

    def _compute_boundary_vertex_mask(self) -> np.ndarray:
        edge_counts: dict[tuple[int, int], int] = {}
        for tri in np.asarray(self.faces, dtype=np.int64):
            for a, b in ((int(tri[0]), int(tri[1])), (int(tri[1]), int(tri[2])), (int(tri[2]), int(tri[0]))):
                if a > b:
                    a, b = b, a
                edge_counts[(a, b)] = edge_counts.get((a, b), 0) + 1
        mask = np.zeros(self.vertices.shape[0], dtype=bool)
        for (a, b), count in edge_counts.items():
            if count == 1:
                mask[a] = True
                mask[b] = True
        return mask

    def boundary_band_vertex_mask(self, width_scale: float) -> np.ndarray:
        width_scale = float(max(width_scale, 0.0))
        if width_scale in self._boundary_band_cache:
            return self._boundary_band_cache[width_scale]
        if not np.any(self.boundary_vertex_mask):
            mask = np.zeros(self.vertices.shape[0], dtype=bool)
        elif width_scale <= 0.0:
            mask = self.boundary_vertex_mask.copy()
        else:
            length_scale = median_edge_length(self.vertices, self.faces)
            max_distance = float(width_scale) * max(float(length_scale), 1e-12)
            distances = geodesic_distance_from_mask(
                self.vertices,
                self.faces,
                self.boundary_vertex_mask,
                max_distance=max_distance,
            )
            mask = np.isfinite(distances) & (distances <= max_distance)
        self._boundary_band_cache[width_scale] = mask
        return mask

    def boundary_correspondence_mask(
        self,
        closest_points: np.ndarray,
        face_indices: np.ndarray,
        width_scale: float,
    ) -> np.ndarray:
        closest_points = np.asarray(closest_points, dtype=np.float64)
        face_indices = np.asarray(face_indices, dtype=np.int64).reshape(-1)
        boundary = np.zeros(face_indices.shape[0], dtype=bool)
        ok_faces = (face_indices >= 0) & (face_indices < self.boundary_face_mask.shape[0])
        if np.any(ok_faces):
            boundary[ok_faces] = self.boundary_face_mask[face_indices[ok_faces]]

        band = self.boundary_band_vertex_mask(width_scale)
        if np.any(band) and self._vertex_tree is not None:
            finite = np.isfinite(closest_points).all(axis=1)
            if np.any(finite):
                _, local_vids = self._vertex_tree.query(closest_points[finite], k=1)
                global_vids = self._query_vertex_ids[np.asarray(local_vids, dtype=np.int64)]
                boundary[finite] |= band[global_vids]

        if self._boundary_vertex_tree is not None:
            length_scale = median_edge_length(self.vertices, self.faces)
            max_distance = float(max(width_scale, 0.0)) * max(float(length_scale), 1e-12)
            if max_distance > 0.0:
                finite = np.isfinite(closest_points).all(axis=1)
                if np.any(finite):
                    distances, _ = self._boundary_vertex_tree.query(closest_points[finite], k=1)
                    boundary[finite] |= np.asarray(distances, dtype=np.float64) <= max_distance
        return boundary

    def closest_points(
        self,
        points: np.ndarray,
        mode: str = "vertex",
        query_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        points = np.asarray(points, dtype=np.float64)
        n_points = int(points.shape[0])
        closest = np.full((n_points, 3), np.nan, dtype=np.float64)
        distances = np.full(n_points, np.inf, dtype=np.float64)
        face_indices = np.full(n_points, -1, dtype=np.int64)
        if query_mask is None:
            active_idx = np.arange(n_points, dtype=np.int64)
        else:
            query_mask = np.asarray(query_mask, dtype=bool).reshape(-1)
            if query_mask.shape[0] != n_points:
                raise ValueError(
                    f"query_mask must have length {n_points}, got {query_mask.shape[0]}"
                )
            active_idx = np.flatnonzero(query_mask)
        if active_idx.size == 0:
            return closest, distances, face_indices, "none"

        active_points = points[active_idx]
        method = "none"
        if mode == "surface" and self._mesh is not None:
            try:
                active_closest, active_distances, active_face_indices = trimesh.proximity.closest_point(
                    self._mesh,
                    active_points,
                )
                closest[active_idx] = np.asarray(active_closest, dtype=np.float64)
                distances[active_idx] = np.asarray(active_distances, dtype=np.float64)
                face_indices[active_idx] = np.asarray(active_face_indices, dtype=np.int64)
                return closest, distances, face_indices, "surface"
            except Exception:
                # Fall back below. This commonly handles optional proximity backend issues.
                pass

        if mode not in {"vertex", "surface"}:
            raise ValueError(f"Unknown correspondence_mode {mode!r}")
        if self._vertex_tree is None:
            raise RuntimeError(
                "No target proximity backend available. Install trimesh proximity deps or scipy.spatial."
            )
        active_distances, vertex_ids = self._vertex_tree.query(active_points, k=1)
        closest[active_idx] = self._query_vertices[np.asarray(vertex_ids, dtype=np.int64)]
        distances[active_idx] = np.asarray(active_distances, dtype=np.float64)
        return closest, distances, face_indices, "vertex"

    def normals_for_faces(self, face_indices: np.ndarray, closest_points: np.ndarray) -> np.ndarray:
        face_indices = np.asarray(face_indices, dtype=np.int64)
        closest_points = np.asarray(closest_points, dtype=np.float64)
        normals = np.zeros((face_indices.shape[0], 3), dtype=np.float64)
        ok = (face_indices >= 0) & (face_indices < self.face_normals.shape[0])
        normals[ok] = self.face_normals[face_indices[ok]]
        if np.any(~ok) and self._vertex_tree is not None:
            fallback = ~ok & np.isfinite(closest_points).all(axis=1)
            if np.any(fallback):
                _, vids = self._vertex_tree.query(closest_points[fallback], k=1)
                global_vids = self._query_vertex_ids[np.asarray(vids, dtype=np.int64)]
                normals[fallback] = self.vertex_normals[global_vids]
        return normals


def visualize_correspondences_polyscope(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target: TargetSurface,
    closest_points: np.ndarray,
    valid_mask: np.ndarray,
    weights: np.ndarray,
    limit: int = 500,
) -> None:
    try:
        import polyscope as ps
    except ImportError as exc:
        raise RuntimeError("polyscope is required for correspondence visualization") from exc

    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    closest_points = np.asarray(closest_points, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.flatnonzero(valid_mask & np.isfinite(closest_points).all(axis=1) & (weights > 0.0))
    if int(limit) > 0 and valid.size > int(limit):
        order = np.argsort(weights[valid])[::-1]
        valid = valid[order[: int(limit)]]

    ps.init()
    ps.remove_all_structures()
    ps.register_surface_mesh(
        "source_deformed",
        source_vertices,
        source_faces,
        color=(0.85, 0.85, 0.85),
        transparency=0.75,
    )
    ps.register_surface_mesh(
        "target",
        target.vertices,
        target.faces,
        color=(0.95, 0.55, 0.95),
        transparency=1.0,
        edge_width=0.96,
    )
    ps.register_point_cloud(
        "source_correspondence_vertices",
        source_vertices[valid],
        radius=0.0015,
        color=(0.95, 0.25, 0.15),
    )
    ps.register_point_cloud(
        "target_correspondence_points",
        closest_points[valid],
        radius=0.0015,
        color=(0.1, 0.85, 0.35),
    )
    if valid.size:
        nodes = np.vstack([source_vertices[valid], closest_points[valid]])
        edges = np.column_stack(
            [
                np.arange(valid.size, dtype=np.int64),
                np.arange(valid.size, dtype=np.int64) + valid.size,
            ]
        )
        ps.register_curve_network(
            "correspondence_links",
            nodes,
            edges,
            radius=0.00035,
            color=(1.0, 0.7, 0.05),
        )
    ps.show()


def build_correspondence_query_mask(
    n_vertices: int,
    data_vertex_weights: Optional[np.ndarray] = None,
    exclude_vertex_mask: Optional[np.ndarray] = None,
    min_data_weight: float = 0.15,
) -> np.ndarray:
    mask = np.ones(int(n_vertices), dtype=bool)
    if data_vertex_weights is not None:
        weights = np.asarray(data_vertex_weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != int(n_vertices):
            raise ValueError(
                f"data_vertex_weights must have length {n_vertices}, got {weights.shape[0]}"
            )
        mask &= weights > float(max(min_data_weight, 0.0))
    if exclude_vertex_mask is not None:
        exclude = np.asarray(exclude_vertex_mask, dtype=bool).reshape(-1)
        if exclude.shape[0] != int(n_vertices):
            raise ValueError(
                f"exclude_vertex_mask must have length {n_vertices}, got {exclude.shape[0]}"
            )
        mask &= ~exclude
    return mask


def _stage_schedule_value(
    stage_index: int,
    n_stages: int,
    start_value: float,
    end_value: float,
    explicit_schedule: Optional[Sequence[float]] = None,
) -> float:
    if explicit_schedule is not None and len(explicit_schedule) > 0:
        idx = min(max(int(stage_index), 0), len(explicit_schedule) - 1)
        return float(explicit_schedule[idx])
    if n_stages <= 1:
        return float(start_value)
    t = float(stage_index) / float(max(n_stages - 1, 1))
    return float(start_value) + t * (float(end_value) - float(start_value))


def merge_correspondence_query_pin_vertex_ids(
    landmark_pin_ids: Optional[np.ndarray],
    extra_pin_vertex_ids: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Union of vertex ids always kept in stage correspondence queries."""
    parts: list[np.ndarray] = []
    if landmark_pin_ids is not None:
        parts.append(np.asarray(landmark_pin_ids, dtype=np.int64).reshape(-1))
    if extra_pin_vertex_ids is not None:
        parts.append(np.asarray(extra_pin_vertex_ids, dtype=np.int64).reshape(-1))
    if not parts:
        return None
    return np.unique(np.concatenate(parts))


def build_stage_correspondence_query_mask(
    base_query_mask: np.ndarray,
    data_vertex_weights: Optional[np.ndarray],
    fraction: float,
    *,
    pin_vertex_ids: Optional[np.ndarray] = None,
    min_vertices: int = 280,
) -> np.ndarray:
    """
    Keep a high-data-weight subset of correspondence queries for later stages.

    Later stages use fewer but more reliable matching points, which reduces
    conflicting pulls and local overfitting bumps.
    """
    base_query_mask = np.asarray(base_query_mask, dtype=bool).reshape(-1)
    fraction = float(np.clip(fraction, 0.05, 1.0))
    candidates = np.flatnonzero(base_query_mask)
    if candidates.size == 0 or fraction >= 1.0 - 1e-6:
        return base_query_mask.copy()

    pinned = np.zeros(base_query_mask.shape[0], dtype=bool)
    if pin_vertex_ids is not None:
        pin_vertex_ids = np.asarray(pin_vertex_ids, dtype=np.int64).reshape(-1)
        pin_vertex_ids = pin_vertex_ids[(pin_vertex_ids >= 0) & (pin_vertex_ids < base_query_mask.shape[0])]
        pinned[pin_vertex_ids] = True
        pinned &= base_query_mask

    n_keep = int(round(fraction * candidates.size))
    n_keep = max(n_keep, int(min_vertices), int(np.count_nonzero(pinned)))
    n_keep = min(n_keep, candidates.size)

    if data_vertex_weights is None:
        order = np.arange(candidates.size, dtype=np.int64)
    else:
        weights = np.asarray(data_vertex_weights, dtype=np.float64).reshape(-1)
        order = np.argsort(-weights[candidates])

    keep_ids = candidates[order[:n_keep]]
    if np.any(pinned):
        keep_ids = np.unique(np.concatenate([keep_ids, np.flatnonzero(pinned)]))

    out = np.zeros(base_query_mask.shape[0], dtype=bool)
    out[keep_ids] = True
    return out


def compute_correspondences(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    deformed_vertices: np.ndarray,
    target: TargetSurface,
    config: NonRigidICPConfig,
    exclude_vertex_mask: Optional[np.ndarray] = None,
    query_vertex_mask: Optional[np.ndarray] = None,
) -> Correspondences:
    if exclude_vertex_mask is not None:
        exclude_vertex_mask = np.asarray(exclude_vertex_mask, dtype=bool).reshape(-1)
        if exclude_vertex_mask.shape[0] != deformed_vertices.shape[0]:
            raise ValueError(
                f"exclude_vertex_mask must have length {deformed_vertices.shape[0]}, "
                f"got {exclude_vertex_mask.shape[0]}"
            )
    if query_vertex_mask is not None:
        query_vertex_mask = np.asarray(query_vertex_mask, dtype=bool).reshape(-1)
        if query_vertex_mask.shape[0] != deformed_vertices.shape[0]:
            raise ValueError(
                f"query_vertex_mask must have length {deformed_vertices.shape[0]}, "
                f"got {query_vertex_mask.shape[0]}"
            )
    closest, distances, face_indices, method = target.closest_points(
        deformed_vertices,
        mode=config.correspondence_mode,
        query_mask=query_vertex_mask,
    )
    need_normals = bool(config.robust_correspondence) or config.normal_angle_threshold_degrees is not None
    if need_normals and (query_vertex_mask is None or np.any(query_vertex_mask)):
        source_normals = vertex_normals(deformed_vertices, source_faces)
        target_normals = target.normals_for_faces(face_indices, closest)
    else:
        source_normals = np.zeros_like(deformed_vertices)
        target_normals = np.zeros_like(deformed_vertices)

    valid = np.zeros(deformed_vertices.shape[0], dtype=bool)
    if query_vertex_mask is None:
        valid = np.isfinite(closest).all(axis=1) & np.isfinite(distances)
    else:
        valid[query_vertex_mask] = (
            np.isfinite(closest[query_vertex_mask]).all(axis=1)
            & np.isfinite(distances[query_vertex_mask])
        )
    if exclude_vertex_mask is not None:
        valid &= ~exclude_vertex_mask
    stats: dict[str, float] = {
        "trim_cutoff": float("nan"),
        "robust_cutoff": float("nan"),
        "robust_scale": float("nan"),
        "num_excluded_vertices": float(np.count_nonzero(exclude_vertex_mask)) if exclude_vertex_mask is not None else 0.0,
        "num_query_vertices": float(np.count_nonzero(query_vertex_mask)) if query_vertex_mask is not None else float(deformed_vertices.shape[0]),
        "num_target_boundary_correspondences": 0.0,
        "target_boundary_weight": float(config.target_boundary_weight),
    }
    target_boundary_mask = np.zeros(deformed_vertices.shape[0], dtype=bool)
    if bool(config.reject_target_boundary_correspondences) and np.any(valid):
        target_boundary_mask = target.boundary_correspondence_mask(
            closest,
            face_indices,
            width_scale=float(config.target_boundary_width_scale),
        )
        target_boundary_mask &= valid
        stats["num_target_boundary_correspondences"] = float(np.count_nonzero(target_boundary_mask))
        if float(config.target_boundary_weight) <= 0.0:
            valid &= ~target_boundary_mask

    normal_dot = None
    if need_normals:
        normal_dot = np.einsum("ij,ij->i", source_normals, target_normals)
        # Meshes may have opposite global orientation; compare by absolute alignment.
        normal_alignment = np.abs(normal_dot)
    else:
        normal_alignment = None

    if config.normal_angle_threshold_degrees is not None:
        cos_limit = float(np.cos(np.deg2rad(config.normal_angle_threshold_degrees)))
        if normal_alignment is None:
            normal_dot = np.einsum("ij,ij->i", source_normals, target_normals)
            normal_alignment = np.abs(normal_dot)
        valid &= normal_alignment >= cos_limit

    if config.distance_threshold is not None:
        valid &= distances <= float(config.distance_threshold)
    elif config.distance_threshold_scale is not None and config.distance_threshold_scale > 0.0:
        src_diag = np.linalg.norm(source_vertices.max(axis=0) - source_vertices.min(axis=0))
        valid &= distances <= float(config.distance_threshold_scale) * max(src_diag, 1e-12)

    if config.trim_percentile is not None and 0.0 < config.trim_percentile < 100.0:
        good_dist = distances[valid]
        if good_dist.size > 8:
            cutoff = float(np.percentile(good_dist, config.trim_percentile))
            stats["trim_cutoff"] = cutoff
            valid &= distances <= cutoff

    weights = valid.astype(np.float64)
    if bool(config.reject_target_boundary_correspondences) and float(config.target_boundary_weight) > 0.0:
        weights[target_boundary_mask] *= float(config.target_boundary_weight)
    if bool(config.robust_correspondence):
        good_dist = distances[valid]
        if good_dist.size > 8:
            median = float(np.median(good_dist))
            mad = float(np.median(np.abs(good_dist - median)))
            robust_scale = 1.4826 * mad
            src_diag = np.linalg.norm(source_vertices.max(axis=0) - source_vertices.min(axis=0))
            min_scale = float(config.robust_min_scale_fraction) * max(float(src_diag), 1e-12)
            robust_scale = max(robust_scale, min_scale, 1e-12)
            cutoff = max(float(config.robust_tukey_scale) * robust_scale, 1e-12)
            stats["robust_scale"] = float(robust_scale)
            stats["robust_cutoff"] = float(cutoff)
            u = distances / cutoff
            tukey = np.square(1.0 - np.square(u))
            tukey[(u >= 1.0) | ~valid] = 0.0
            weights *= np.clip(tukey, 0.0, 1.0)
        if normal_alignment is not None:
            normal_weights = np.clip(normal_alignment, 0.0, 1.0)
            weights *= normal_weights * normal_weights
        valid &= weights > 0.0
    if bool(config.visualize_correspondences):
        visualize_correspondences_polyscope(
            deformed_vertices,
            source_faces,
            target,
            closest,
            valid,
            weights,
            limit=int(config.visualize_correspondence_limit),
        )
    return Correspondences(
        points=closest,
        distances=distances,
        weights=weights,
        face_indices=face_indices,
        target_normals=target_normals,
        source_normals=source_normals,
        valid_mask=valid,
        method=method,
        stats=stats,
    )


def camera_depth_correspondence_weights(
    vertices: np.ndarray,
    *,
    depth_axis: int = 2,
    reference_depth: float | None = None,
    reference_vertex_mask: np.ndarray | None = None,
    min_weight: float = 0.35,
    power: float = 2.0,
    far_percentile: float = 95.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Down-weight vertices farther from the camera reference plane (profile-side relief).

    Uses |depth - reference| along ``depth_axis``; reference defaults to the median
    depth of ``reference_vertex_mask`` (typically the face region).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    depth = vertices[:, int(depth_axis)]
    stats: dict[str, float] = {
        "camera_depth_axis": float(depth_axis),
        "camera_depth_reference": float("nan"),
        "camera_depth_far_span": float("nan"),
    }
    if reference_depth is None:
        if reference_vertex_mask is not None:
            ref_mask = np.asarray(reference_vertex_mask, dtype=bool).reshape(-1)
            ref_vals = depth[ref_mask & np.isfinite(depth)]
        else:
            ref_vals = depth[np.isfinite(depth)]
        reference_depth = float(np.median(ref_vals)) if ref_vals.size else 0.0
    stats["camera_depth_reference"] = float(reference_depth)
    distance = np.abs(depth - float(reference_depth))
    finite_dist = distance[np.isfinite(distance)]
    if finite_dist.size == 0:
        return np.ones(vertices.shape[0], dtype=np.float64), stats
    far = float(np.percentile(finite_dist, float(far_percentile)))
    stats["camera_depth_far_span"] = far
    if not np.isfinite(far) or far < 1e-8:
        return np.ones(vertices.shape[0], dtype=np.float64), stats
    confidence = 1.0 - np.clip(distance / far, 0.0, 1.0)
    min_w = float(np.clip(min_weight, 0.0, 1.0))
    weights = min_w + (1.0 - min_w) * np.power(confidence, float(max(power, 0.0)))
    weights[~np.isfinite(depth)] = min_w
    return weights.astype(np.float64), stats


def apply_camera_depth_correspondence_attenuation(
    correspondences: Correspondences,
    deformed_vertices: np.ndarray,
    config: NonRigidICPConfig,
    reference_vertex_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Scale correspondence weights down for source points far from the camera plane."""
    if not bool(config.camera_depth_attenuation_enabled):
        return {"camera_depth_enabled": 0.0}
    depth_w, depth_stats = camera_depth_correspondence_weights(
        deformed_vertices,
        depth_axis=int(config.camera_depth_axis),
        reference_vertex_mask=reference_vertex_mask,
        min_weight=float(config.camera_depth_min_weight),
        power=float(config.camera_depth_power),
        far_percentile=float(config.camera_depth_far_percentile),
    )
    correspondences.weights = correspondences.weights * depth_w
    correspondences.valid_mask = correspondences.valid_mask & (
        correspondences.weights > 1e-12
    )
    valid = correspondences.valid_mask
    out = {
        "camera_depth_enabled": 1.0,
        **depth_stats,
        "camera_depth_weight_mean": float(np.mean(depth_w[valid])) if np.any(valid) else 1.0,
        "camera_depth_weight_min": float(np.min(depth_w[valid])) if np.any(valid) else 1.0,
    }
    return out


def apply_correspondence_vertex_controls(
    correspondences: Correspondences,
    data_vertex_weights: Optional[np.ndarray] = None,
    exclude_vertex_mask: Optional[np.ndarray] = None,
) -> None:
    if exclude_vertex_mask is not None:
        exclude_vertex_mask = np.asarray(exclude_vertex_mask, dtype=bool).reshape(-1)
        correspondences.weights = correspondences.weights.copy()
        correspondences.valid_mask = correspondences.valid_mask & ~exclude_vertex_mask
        correspondences.weights[exclude_vertex_mask] = 0.0
    if data_vertex_weights is not None:
        correspondences.weights = correspondences.weights * data_vertex_weights
        correspondences.valid_mask = correspondences.valid_mask & (correspondences.weights > 0.0)


def _smooth_vertex_field_by_edges(
    values: np.ndarray,
    edges: np.ndarray,
    iterations: int,
    step: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    edges = np.asarray(edges, dtype=np.int64)
    if values.size == 0 or edges.size == 0 or int(iterations) <= 0:
        return np.clip(values, 0.0, 1.0)
    step = float(np.clip(step, 0.0, 1.0))
    accum_edges = edges.reshape(-1)
    counts = np.bincount(accum_edges, minlength=values.shape[0]).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    out = values.copy()
    for _ in range(int(iterations)):
        accum = np.zeros_like(out)
        np.add.at(accum, edges[:, 0], out[edges[:, 1]])
        np.add.at(accum, edges[:, 1], out[edges[:, 0]])
        neighbor_mean = accum / counts
        out = (1.0 - step) * out + step * neighbor_mean
        out = np.clip(out, 0.0, 1.0)
    return out


def _edge_band_from_vertex_band(edges: np.ndarray, vertex_band: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return np.empty(0, dtype=np.float64)
    vertex_band = np.asarray(vertex_band, dtype=np.float64).reshape(-1)
    return 0.5 * (vertex_band[edges[:, 0]] + vertex_band[edges[:, 1]])


def build_adaptive_relaxation_weights(
    distances: np.ndarray,
    controlled_mask: np.ndarray,
    edges: np.ndarray,
    config: NonRigidICPConfig,
    exclude_vertex_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[str, float]]:
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    controlled = np.asarray(controlled_mask, dtype=bool).reshape(-1) & np.isfinite(distances)
    if exclude_vertex_mask is not None:
        controlled &= ~np.asarray(exclude_vertex_mask, dtype=bool).reshape(-1)

    band = np.zeros(distances.shape[0], dtype=np.float64)
    stats = {
        "adaptive_enabled": float(bool(config.adaptive_relaxation)),
        "adaptive_cutoff": float("nan"),
        "adaptive_vertices": 0.0,
        "adaptive_controlled_ratio": float(np.mean(controlled)) if controlled.size else 0.0,
        "adaptive_band_mean": 0.0,
        "adaptive_data_weight_mean": 1.0,
        "adaptive_regularization_weight_mean": 1.0,
        "adaptive_vertex_mass_weight_mean": 1.0,
    }
    good = distances[controlled]
    if good.size <= 8:
        return band, stats

    percentile = float(np.clip(config.adaptive_residual_percentile, 0.0, 100.0))
    cutoff = float(np.percentile(good, percentile))
    stats["adaptive_cutoff"] = cutoff
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        return band, stats

    high = controlled & (distances >= cutoff)
    band[high] = np.clip((distances[high] - cutoff) / max(cutoff, 1e-12), 0.0, 1.0)
    band = _smooth_vertex_field_by_edges(
        band,
        edges,
        int(config.adaptive_residual_smooth_iterations),
        float(config.adaptive_residual_smooth_step),
    )
    if exclude_vertex_mask is not None:
        band[np.asarray(exclude_vertex_mask, dtype=bool).reshape(-1)] = 0.0

    stats["adaptive_vertices"] = float(np.count_nonzero(band > 1e-6))
    stats["adaptive_band_mean"] = float(np.mean(band)) if band.size else 0.0
    data_weights = 1.0 + max(float(config.adaptive_data_boost), 0.0) * band
    reg_weights = 1.0 - (1.0 - float(np.clip(config.adaptive_regularization_multiplier, 0.0, 1.0))) * band
    mass_weights = 1.0 - (1.0 - float(np.clip(config.adaptive_vertex_mass_multiplier, 0.0, 1.0))) * band
    stats["adaptive_data_weight_mean"] = float(np.mean(data_weights)) if data_weights.size else 1.0
    stats["adaptive_regularization_weight_mean"] = float(np.mean(reg_weights)) if reg_weights.size else 1.0
    stats["adaptive_vertex_mass_weight_mean"] = float(np.mean(mass_weights)) if mass_weights.size else 1.0
    return band, stats


def controlled_correspondence_mask(
    distances: np.ndarray,
    closest_points: np.ndarray,
    face_indices: np.ndarray,
    target: TargetSurface,
    config: NonRigidICPConfig,
    exclude_vertex_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    closest_points = np.asarray(closest_points, dtype=np.float64)
    face_indices = np.asarray(face_indices, dtype=np.int64).reshape(-1)
    controlled = np.isfinite(distances) & np.isfinite(closest_points).all(axis=1)
    if exclude_vertex_mask is not None:
        controlled &= ~np.asarray(exclude_vertex_mask, dtype=bool).reshape(-1)
    if bool(config.reject_target_boundary_correspondences):
        boundary = target.boundary_correspondence_mask(
            closest_points,
            face_indices,
            width_scale=float(config.target_boundary_width_scale),
        )
        controlled &= ~boundary
    return controlled


def summarize_controlled_correspondence_distances(
    distances: np.ndarray,
    closest_points: np.ndarray,
    face_indices: np.ndarray,
    target: TargetSurface,
    config: NonRigidICPConfig,
    exclude_vertex_mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    controlled = controlled_correspondence_mask(
        distances,
        closest_points,
        face_indices,
        target,
        config,
        exclude_vertex_mask=exclude_vertex_mask,
    )
    vals = distances[controlled]
    return {
        "controlled_rmse": float(np.sqrt(np.mean(vals**2))) if vals.size else float("nan"),
        "controlled_mean_distance": float(np.mean(vals)) if vals.size else float("nan"),
        "controlled_ratio": float(np.mean(controlled)) if controlled.size else 0.0,
    }


def solve_affine_field(
    A: sp.spmatrix,
    B: np.ndarray,
    damping: float = 1e-8,
    solver: str = "lsmr",
    tolerance: float = 1e-5,
    max_iterations: int = 200,
    axis_extra_A: Optional[tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]] = None,
    axis_extra_B: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """
    Solve min_X ||A X - B||_F^2 for three coordinate RHS columns.
    """
    A = A.tocsr()
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] == 0:
        raise ValueError("Cannot solve an empty least-squares system")
    if B.shape != (A.shape[0], 3):
        raise ValueError(f"B must have shape {(A.shape[0], 3)}, got {B.shape}")

    def axis_system(axis: int) -> tuple[sp.csr_matrix, np.ndarray]:
        rhs = B[:, axis]
        if axis_extra_A is None:
            return A, rhs
        if axis_extra_B is None:
            raise ValueError("axis_extra_B is required when axis_extra_A is provided")
        extra_A = axis_extra_A[axis].tocsr()
        extra_b = np.asarray(axis_extra_B[axis], dtype=np.float64).reshape(-1)
        if extra_A.shape[0] == 0:
            return A, rhs
        if extra_A.shape[1] != A.shape[1] or extra_b.shape[0] != extra_A.shape[0]:
            raise ValueError("axis extra system has incompatible shape")
        return sp.vstack([A, extra_A], format="csr"), np.concatenate([rhs, extra_b])

    if solver == "lsmr":
        cols = []
        for axis in range(3):
            axis_A, axis_b = axis_system(axis)
            cols.append(
                spla.lsmr(
                    axis_A,
                    axis_b,
                    damp=float(np.sqrt(max(damping, 0.0))),
                    atol=float(tolerance),
                    btol=float(tolerance),
                    maxiter=int(max_iterations),
                )[0]
            )
        return np.column_stack(cols)

    if solver not in {"normal_cholesky", "spsolve"}:
        raise ValueError(f"Unknown solver {solver!r}")

    cols = []
    for axis in range(3):
        axis_A, axis_b = axis_system(axis)
        normal = (axis_A.T @ axis_A).tocsc()
        rhs = axis_A.T @ axis_b
        if damping > 0.0:
            normal = normal + float(damping) * sp.eye(normal.shape[0], format="csc")
        if solver == "spsolve":
            cols.append(spla.spsolve(normal, rhs))
            continue
        try:
            cols.append(spla.factorized(normal)(rhs))
        except Exception:
            cols.append(spla.spsolve(normal, rhs))
    return np.column_stack(cols)


def _landmark_weight_for_stage(
    landmark_weight: float,
    stage_index: int,
    n_stages: int,
    decay: float = 0.5,
) -> float:
    if landmark_weight <= 0.0:
        return 0.0
    return float(landmark_weight) * (float(decay) ** int(stage_index))


def _correspondence_config_for_stage(
    config: NonRigidICPConfig,
    stage_index: int,
    n_stages: int = 1,
) -> NonRigidICPConfig:
    use_robust = bool(config.robust_correspondence) and stage_index >= int(config.robust_correspondence_start_stage)
    if config.stage_trim_percentiles is not None and len(config.stage_trim_percentiles) > 0:
        trim = _stage_schedule_value(
            stage_index,
            n_stages,
            float(config.trim_percentile if config.trim_percentile is not None else 100.0),
            float(config.trim_percentile_end if config.trim_percentile_end is not None else config.trim_percentile or 100.0),
            explicit_schedule=config.stage_trim_percentiles,
        )
    elif not use_robust and config.initial_trim_percentile is not None:
        trim = float(config.initial_trim_percentile)
    elif config.trim_percentile_end is not None and config.trim_percentile is not None:
        trim = _stage_schedule_value(
            stage_index,
            n_stages,
            float(config.trim_percentile),
            float(config.trim_percentile_end),
        )
    else:
        trim = config.trim_percentile
    return replace(config, robust_correspondence=use_robust, trim_percentile=trim)


def _stage_adaptive_data_boost(
    config: NonRigidICPConfig,
    stage_index: int,
    n_stages: int,
) -> float:
    end_boost = (
        float(config.adaptive_data_boost)
        if config.adaptive_data_boost_end is None
        else float(config.adaptive_data_boost_end)
    )
    return _stage_schedule_value(
        stage_index,
        n_stages,
        float(config.adaptive_data_boost),
        end_boost,
    )


def _adaptive_relaxation_enabled(config: NonRigidICPConfig, stage_index: int) -> bool:
    if not bool(config.adaptive_relaxation):
        return False
    if stage_index < int(config.adaptive_relaxation_start_stage):
        return False
    if config.adaptive_relaxation_end_stage is not None:
        return stage_index < int(config.adaptive_relaxation_end_stage)
    return True


def project_vertices_along_normals_to_target(
    vertices: np.ndarray,
    faces: np.ndarray,
    target: TargetSurface,
    query_vertex_mask: Optional[np.ndarray] = None,
    *,
    blend: float | np.ndarray = 1.0,
    max_displacement: Optional[float] = None,
    reference_vertices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Final Amberg-style projection: move vertices toward the target along source normals,
    then snap to the nearest target surface point.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    out = vertices.copy()
    if query_vertex_mask is None:
        active = np.ones(vertices.shape[0], dtype=bool)
    else:
        active = np.asarray(query_vertex_mask, dtype=bool).reshape(-1)
    if not np.any(active):
        return out

    active_vertices = vertices[active]
    normals = vertex_normals(vertices, faces)[active]
    closest, _, _, method = target.closest_points(active_vertices, mode="surface", query_mask=None)
    if method != "surface" or not np.isfinite(closest).all():
        closest, _, _, _ = target.closest_points(active_vertices, mode="vertex", query_mask=None)
    delta = closest - active_vertices
    normal_step = np.einsum("ij,ij->i", delta, normals)[:, None] * normals
    projected = active_vertices + normal_step
    snapped, _, _, _ = target.closest_points(projected, mode="surface", query_mask=None)
    if not np.isfinite(snapped).all():
        snapped, _, _, _ = target.closest_points(projected, mode="vertex", query_mask=None)
    ok = np.isfinite(snapped).all(axis=1)
    moved = np.where(ok[:, None], snapped, active_vertices)
    if np.ndim(blend) == 0:
        blend_active = np.full((active_vertices.shape[0], 1), float(np.clip(blend, 0.0, 1.0)), dtype=np.float64)
    else:
        blend_all = np.asarray(blend, dtype=np.float64).reshape(-1)
        if blend_all.shape[0] != vertices.shape[0]:
            raise ValueError(f"blend must have length {vertices.shape[0]}, got {blend_all.shape[0]}")
        blend_active = np.clip(blend_all[active], 0.0, 1.0)[:, None]
    moved = active_vertices + blend_active * (moved - active_vertices)
    if max_displacement is not None and reference_vertices is not None:
        reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
        moved = clamp_relative_vertex_displacement(reference_vertices[active], moved, float(max_displacement))
    out[active] = moved
    return out


def configure_robust_auto(config: NonRigidICPConfig) -> NonRigidICPConfig:
    """
    Tune a NonRigidICPConfig for robust_auto fitting on HACK raw -> BNI meshes.

    Prioritizes surface correspondences, stage-wise robust filtering, lighter
    geometry priors, and stronger adaptive relaxation on high-residual regions.
    """
    config.correspondence_mode = "surface"
    config.robust_correspondence = True
    config.robust_correspondence_start_stage = 1
    config.initial_trim_percentile = 100.0
    config.normal_angle_threshold_degrees = 55.0
    config.trim_percentile = 99.5
    config.trim_percentile_end = None
    config.stage_trim_percentiles = None
    config.query_fraction_start = 1.0
    config.query_fraction_end = 1.0
    config.stage_query_fractions = None
    config.stiffness_schedule = (3.0, 1.7, 0.9)
    if int(config.max_iterations) != 1:
        config.max_iterations = 3
    config.data_weight = 3.0
    config.affine_prior_weight = 0.2
    config.affine_prior_decay = 0.5
    config.position_prior_weight = 0.3
    config.position_prior_decay = 0.7
    config.laplacian_weight = 0.5
    config.vertex_mass_weight = 0.3
    config.edge_vector_weight = 1.0
    config.geometry_weight_decay = 0.8
    config.landmark_weight_decay = 0.5
    config.adaptive_relaxation = True
    config.adaptive_relaxation_start_stage = 1
    config.adaptive_relaxation_end_stage = None
    config.adaptive_data_boost = 1.5
    config.adaptive_data_boost_end = None
    config.adaptive_regularization_multiplier = 0.4
    config.adaptive_min_edge_reg = 0.0
    config.adaptive_vertex_mass_multiplier = 0.6
    config.max_iteration_displacement_scale = 0.0
    config.flip_guard_enabled = True
    config.flip_guard_tolerance = 0
    config.flip_guard_blend = 0.85
    config.final_projection_blend = 1.0
    config.final_projection_max_displacement_scale = 0.0
    config.final_normal_projection = True
    config.camera_depth_attenuation_enabled = True
    config.camera_depth_axis = 2
    config.camera_depth_min_weight = 0.35
    config.camera_depth_power = 2.0
    config.camera_depth_far_percentile = 95.0
    return config


def nonrigid_icp(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    config: Optional[NonRigidICPConfig] = None,
    target_face_mask: Optional[np.ndarray] = None,
    landmarks: Optional[dict[str, np.ndarray]] = None,
    data_vertex_weights: Optional[np.ndarray] = None,
    correspondence_exclude_vertex_mask: Optional[np.ndarray] = None,
    smoothness_edge_weights: Optional[np.ndarray] = None,
    edge_vector_edge_weights: Optional[np.ndarray] = None,
    affine_prior_vertex_weights: Optional[np.ndarray] = None,
    position_prior_vertex_weights: Optional[np.ndarray] = None,
    position_prior_axis_weights: Optional[np.ndarray] = None,
    shape_prior_vertex_weights: Optional[np.ndarray] = None,
    vertex_mass_vertex_weights: Optional[np.ndarray] = None,
    max_displacement_vertex_scales: Optional[np.ndarray] = None,
    flip_guard_vertex_scales: Optional[np.ndarray] = None,
    final_projection_vertex_blend: Optional[np.ndarray] = None,
    camera_depth_reference_vertex_mask: Optional[np.ndarray] = None,
    correspondence_query_pin_vertex_ids: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[dict[str, float]], None]] = None,
) -> RegistrationResult:
    """
    Register source mesh vertices to a target triangle mesh.
    """
    if config is None:
        config = NonRigidICPConfig()
    source_vertices, source_faces = validate_vertices_faces(source_vertices, source_faces)
    target_vertices, target_faces = validate_vertices_faces(target_vertices, target_faces)
    if target_face_mask is not None:
        target_face_mask = np.asarray(target_face_mask, dtype=bool).reshape(-1)
        if target_face_mask.shape[0] != target_faces.shape[0]:
            raise ValueError(
                f"target_face_mask must have length {target_faces.shape[0]}, "
                f"got {target_face_mask.shape[0]}"
            )
    if data_vertex_weights is not None:
        data_vertex_weights = np.asarray(data_vertex_weights, dtype=np.float64).reshape(-1)
        if data_vertex_weights.shape[0] != source_vertices.shape[0]:
            raise ValueError(
                f"data_vertex_weights must have length {source_vertices.shape[0]}, "
                f"got {data_vertex_weights.shape[0]}"
            )
    if correspondence_exclude_vertex_mask is not None:
        correspondence_exclude_vertex_mask = np.asarray(correspondence_exclude_vertex_mask, dtype=bool).reshape(-1)
        if correspondence_exclude_vertex_mask.shape[0] != source_vertices.shape[0]:
            raise ValueError(
                f"correspondence_exclude_vertex_mask must have length {source_vertices.shape[0]}, "
                f"got {correspondence_exclude_vertex_mask.shape[0]}"
            )
    if affine_prior_vertex_weights is not None:
        affine_prior_vertex_weights = np.asarray(affine_prior_vertex_weights, dtype=np.float64).reshape(-1)
        if affine_prior_vertex_weights.shape[0] != source_vertices.shape[0]:
            raise ValueError(
                f"affine_prior_vertex_weights must have length {source_vertices.shape[0]}, "
                f"got {affine_prior_vertex_weights.shape[0]}"
            )
    if position_prior_vertex_weights is not None:
        position_prior_vertex_weights = np.asarray(position_prior_vertex_weights, dtype=np.float64).reshape(-1)
        if position_prior_vertex_weights.shape[0] != source_vertices.shape[0]:
            raise ValueError(
                f"position_prior_vertex_weights must have length {source_vertices.shape[0]}, "
                f"got {position_prior_vertex_weights.shape[0]}"
            )
    if position_prior_axis_weights is not None:
        position_prior_axis_weights = np.asarray(position_prior_axis_weights, dtype=np.float64)
        if position_prior_axis_weights.shape != (source_vertices.shape[0], 3):
            raise ValueError(
                f"position_prior_axis_weights must have shape {(source_vertices.shape[0], 3)}, "
                f"got {position_prior_axis_weights.shape}"
            )
    if shape_prior_vertex_weights is not None:
        shape_prior_vertex_weights = np.asarray(shape_prior_vertex_weights, dtype=np.float64).reshape(-1)
        if shape_prior_vertex_weights.shape[0] != source_vertices.shape[0]:
            raise ValueError(
                f"shape_prior_vertex_weights must have length {source_vertices.shape[0]}, "
                f"got {shape_prior_vertex_weights.shape[0]}"
            )
    if vertex_mass_vertex_weights is not None:
        vertex_mass_vertex_weights = np.asarray(vertex_mass_vertex_weights, dtype=np.float64).reshape(-1)
        if vertex_mass_vertex_weights.shape[0] != source_vertices.shape[0]:
            raise ValueError(
                f"vertex_mass_vertex_weights must have length {source_vertices.shape[0]}, "
                f"got {vertex_mass_vertex_weights.shape[0]}"
            )
    if max_displacement_vertex_scales is not None:
        max_displacement_vertex_scales = np.asarray(max_displacement_vertex_scales, dtype=np.float64).reshape(-1)
        if max_displacement_vertex_scales.shape[0] != source_vertices.shape[0]:
            raise ValueError(
                f"max_displacement_vertex_scales must have length {source_vertices.shape[0]}, "
                f"got {max_displacement_vertex_scales.shape[0]}"
            )
    if flip_guard_vertex_scales is not None:
        flip_guard_vertex_scales = np.asarray(flip_guard_vertex_scales, dtype=np.float64).reshape(-1)
        if flip_guard_vertex_scales.shape[0] != source_vertices.shape[0]:
            raise ValueError(
                f"flip_guard_vertex_scales must have length {source_vertices.shape[0]}, "
                f"got {flip_guard_vertex_scales.shape[0]}"
            )
    if final_projection_vertex_blend is not None:
        final_projection_vertex_blend = np.asarray(final_projection_vertex_blend, dtype=np.float64).reshape(-1)
        if final_projection_vertex_blend.shape[0] != source_vertices.shape[0]:
            raise ValueError(
                f"final_projection_vertex_blend must have length {source_vertices.shape[0]}, "
                f"got {final_projection_vertex_blend.shape[0]}"
            )
    edges = extract_edges(source_faces)
    mesh_length_scale = float(median_edge_length(source_vertices, source_faces))
    if smoothness_edge_weights is not None:
        smoothness_edge_weights = np.asarray(smoothness_edge_weights, dtype=np.float64).reshape(-1)
        if smoothness_edge_weights.shape[0] != edges.shape[0]:
            raise ValueError(
                f"smoothness_edge_weights must have length {edges.shape[0]}, "
                f"got {smoothness_edge_weights.shape[0]}"
            )
    if edge_vector_edge_weights is not None:
        edge_vector_edge_weights = np.asarray(edge_vector_edge_weights, dtype=np.float64).reshape(-1)
        if edge_vector_edge_weights.shape[0] != edges.shape[0]:
            raise ValueError(
                f"edge_vector_edge_weights must have length {edges.shape[0]}, "
                f"got {edge_vector_edge_weights.shape[0]}"
            )
    target = TargetSurface(target_vertices, target_faces, face_mask=target_face_mask)
    correspondence_query_mask = build_correspondence_query_mask(
        source_vertices.shape[0],
        data_vertex_weights=data_vertex_weights,
        exclude_vertex_mask=correspondence_exclude_vertex_mask,
        min_data_weight=0.0,
    )
    affine_field = identity_affine_field(source_vertices.shape[0])
    affine_prior = affine_field.copy()

    history: list[dict[str, float]] = []
    n_stages = len(config.stiffness_schedule)
    landmark_pin_ids = None
    if landmarks is not None and "face_vertices" in landmarks:
        landmark_pin_ids = np.unique(np.asarray(landmarks["face_vertices"], dtype=np.int64).reshape(-1))
    query_pin_ids = merge_correspondence_query_pin_vertex_ids(
        landmark_pin_ids,
        correspondence_query_pin_vertex_ids,
    )
    stage0_corr_config = _correspondence_config_for_stage(config, 0, n_stages)
    stage0_query_fraction = _stage_schedule_value(
        0,
        n_stages,
        float(config.query_fraction_start),
        float(config.query_fraction_end),
        explicit_schedule=config.stage_query_fractions,
    )
    stage0_query_mask = build_stage_correspondence_query_mask(
        correspondence_query_mask,
        data_vertex_weights,
        stage0_query_fraction,
        pin_vertex_ids=query_pin_ids,
        min_vertices=int(config.min_stage_query_vertices),
    )
    corr = compute_correspondences(
        source_vertices,
        source_faces,
        apply_affine_field(source_vertices, affine_field),
        target,
        stage0_corr_config,
        exclude_vertex_mask=correspondence_exclude_vertex_mask,
        query_vertex_mask=stage0_query_mask,
    )
    apply_camera_depth_correspondence_attenuation(
        corr,
        apply_affine_field(source_vertices, affine_field),
        config,
        reference_vertex_mask=camera_depth_reference_vertex_mask,
    )
    apply_correspondence_vertex_controls(corr, data_vertex_weights, correspondence_exclude_vertex_mask)

    for stage_idx, stiffness in enumerate(tqdm(config.stiffness_schedule)):
        last_mse = None
        stage_corr_config = _correspondence_config_for_stage(config, stage_idx, n_stages)
        stage_query_fraction = _stage_schedule_value(
            stage_idx,
            n_stages,
            float(config.query_fraction_start),
            float(config.query_fraction_end),
            explicit_schedule=config.stage_query_fractions,
        )
        stage_query_mask = build_stage_correspondence_query_mask(
            correspondence_query_mask,
            data_vertex_weights,
            stage_query_fraction,
            pin_vertex_ids=query_pin_ids,
            min_vertices=int(config.min_stage_query_vertices),
        )
        stage_adaptive_boost = _stage_adaptive_data_boost(config, stage_idx, n_stages)
        lmk_weight = _landmark_weight_for_stage(
            config.landmark_weight,
            stage_idx,
            n_stages,
            decay=config.landmark_weight_decay,
        )
        prior_weight = float(config.affine_prior_weight) * (float(config.affine_prior_decay) ** stage_idx)
        pos_weight = float(config.position_prior_weight) * (float(config.position_prior_decay) ** stage_idx)
        lap_weight = float(config.laplacian_weight) * (float(config.geometry_weight_decay) ** stage_idx)
        mass_weight = float(config.vertex_mass_weight) * (float(config.geometry_weight_decay) ** stage_idx)
        edge_vec_weight = float(config.edge_vector_weight) * (float(config.geometry_weight_decay) ** stage_idx)
        for iter_idx in trange(int(config.max_iterations)):
            iter_t0 = time.perf_counter()
            deformed = apply_affine_field(source_vertices, affine_field)
            corr_t0 = time.perf_counter()
            corr = compute_correspondences(
                source_vertices,
                source_faces,
                deformed,
                target,
                stage_corr_config,
                exclude_vertex_mask=correspondence_exclude_vertex_mask,
                query_vertex_mask=stage_query_mask,
            )
            if np.any(correspondence_query_mask):
                valid_ratio = float(np.mean(corr.valid_mask[correspondence_query_mask]))
                valid_distances = corr.distances[correspondence_query_mask & corr.valid_mask]
            else:
                valid_ratio = float(np.mean(corr.valid_mask)) if corr.valid_mask.size else 0.0
                valid_distances = corr.distances[corr.valid_mask]
            depth_stats = apply_camera_depth_correspondence_attenuation(
                corr,
                deformed,
                config,
                reference_vertex_mask=camera_depth_reference_vertex_mask,
            )
            apply_correspondence_vertex_controls(corr, data_vertex_weights, correspondence_exclude_vertex_mask)
            corr_seconds = time.perf_counter() - corr_t0
            if valid_ratio < float(config.min_valid_correspondence_ratio):
                raise RuntimeError(
                    f"Too few valid correspondences: {valid_ratio:.3f}. "
                    "Relax rejection thresholds or improve initialization."
                )

            adaptive_band = np.zeros(source_vertices.shape[0], dtype=np.float64)
            adaptive_stats = {
                "adaptive_enabled": 0.0,
                "adaptive_cutoff": float("nan"),
                "adaptive_vertices": 0.0,
                "adaptive_controlled_ratio": 0.0,
                "adaptive_band_mean": 0.0,
                "adaptive_data_weight_mean": 1.0,
                "adaptive_regularization_weight_mean": 1.0,
                "adaptive_vertex_mass_weight_mean": 1.0,
            }
            if _adaptive_relaxation_enabled(config, stage_idx):
                adaptive_controlled_mask = controlled_correspondence_mask(
                    corr.distances,
                    corr.points,
                    corr.face_indices,
                    target,
                    config,
                    exclude_vertex_mask=correspondence_exclude_vertex_mask,
                )
                adaptive_band, adaptive_stats = build_adaptive_relaxation_weights(
                    corr.distances,
                    adaptive_controlled_mask,
                    edges,
                    config,
                    exclude_vertex_mask=correspondence_exclude_vertex_mask,
                )
            adaptive_data_weights = 1.0 + max(float(stage_adaptive_boost), 0.0) * adaptive_band
            adaptive_vertex_reg_weights = 1.0 - (
                1.0 - float(np.clip(config.adaptive_regularization_multiplier, 0.0, 1.0))
            ) * adaptive_band
            adaptive_mass_weights = 1.0 - (
                1.0 - float(np.clip(config.adaptive_vertex_mass_multiplier, 0.0, 1.0))
            ) * adaptive_band
            effective_corr_weights = corr.weights * adaptive_data_weights
            effective_shape_weights = (
                adaptive_vertex_reg_weights
                if shape_prior_vertex_weights is None
                else np.asarray(shape_prior_vertex_weights, dtype=np.float64) * adaptive_vertex_reg_weights
            )
            if vertex_mass_vertex_weights is not None:
                base_mass_vertex_weights = np.asarray(vertex_mass_vertex_weights, dtype=np.float64)
            elif shape_prior_vertex_weights is not None:
                base_mass_vertex_weights = np.asarray(shape_prior_vertex_weights, dtype=np.float64)
            else:
                base_mass_vertex_weights = np.ones(source_vertices.shape[0], dtype=np.float64)
            effective_mass_vertex_weights = base_mass_vertex_weights * adaptive_mass_weights
            effective_smoothness_edge_weights = smoothness_edge_weights
            if smoothness_edge_weights is not None and np.any(adaptive_band > 1e-6):
                edge_reg = _edge_band_from_vertex_band(edges, adaptive_vertex_reg_weights)
                min_edge_reg = float(np.clip(config.adaptive_min_edge_reg, 0.0, 1.0))
                if min_edge_reg > 0.0:
                    edge_reg = np.maximum(edge_reg, min_edge_reg)
                effective_smoothness_edge_weights = np.asarray(smoothness_edge_weights, dtype=np.float64) * edge_reg
            effective_edge_vector_edge_weights = edge_vector_edge_weights
            if edge_vector_edge_weights is not None and np.any(adaptive_band > 1e-6):
                edge_reg = _edge_band_from_vertex_band(edges, adaptive_vertex_reg_weights)
                min_edge_reg = float(np.clip(config.adaptive_min_edge_reg, 0.0, 1.0))
                if min_edge_reg > 0.0:
                    edge_reg = np.maximum(edge_reg, min_edge_reg)
                effective_edge_vector_edge_weights = np.asarray(edge_vector_edge_weights, dtype=np.float64) * edge_reg

            system = assemble_linear_system(
                source_vertices=source_vertices,
                edges=edges,
                target_points=corr.points,
                correspondence_weights=effective_corr_weights,
                stiffness=float(stiffness),
                gamma=config.gamma,
                smoothness_edge_weights=effective_smoothness_edge_weights,
                data_weight=config.data_weight,
                landmarks=landmarks,
                landmark_weight=lmk_weight,
                affine_prior=affine_prior,
                affine_prior_weight=prior_weight,
                affine_prior_vertex_weights=affine_prior_vertex_weights,
                position_prior_vertices=source_vertices,
                position_prior_weight=pos_weight,
                position_prior_vertex_weights=position_prior_vertex_weights,
                position_prior_axis_weights=position_prior_axis_weights,
                laplacian_weight=lap_weight,
                laplacian_vertex_weights=effective_shape_weights,
                vertex_mass_weight=mass_weight,
                vertex_mass_vertex_weights=effective_mass_vertex_weights,
                edge_vector_weight=edge_vec_weight,
                edge_vector_edge_weights=effective_edge_vector_edge_weights,
            )
            solve_t0 = time.perf_counter()
            affine_field = solve_affine_field(
                system.A,
                system.B,
                damping=config.damping,
                solver=config.solver,
                tolerance=config.solver_tolerance,
                max_iterations=config.solver_max_iterations,
                axis_extra_A=system.axis_extra_A,
                axis_extra_B=system.axis_extra_B,
            )
            deformed_new = apply_affine_field(source_vertices, affine_field)
            if float(config.max_iteration_displacement_scale) > 0.0:
                max_step = float(config.max_iteration_displacement_scale) * mesh_length_scale
                if max_displacement_vertex_scales is not None:
                    max_step = max_step * np.clip(max_displacement_vertex_scales, 0.25, 4.0)
                deformed_new = clamp_relative_vertex_displacement(deformed, deformed_new, max_step)
            if bool(config.flip_guard_enabled):
                prev_inverted = count_inverted_faces(source_vertices, deformed, source_faces)
                new_inverted = count_inverted_faces(source_vertices, deformed_new, source_faces)
                if new_inverted > prev_inverted + int(config.flip_guard_tolerance):
                    flip_blend = float(np.clip(config.flip_guard_blend, 0.0, 1.0))
                    if flip_guard_vertex_scales is not None:
                        guard_weight = flip_blend * np.clip(flip_guard_vertex_scales, 0.0, 1.0)[:, None]
                    else:
                        guard_weight = flip_blend
                    deformed_new = (1.0 - guard_weight) * deformed_new + guard_weight * deformed
            affine_field = retarget_affine_field_vertices(source_vertices, affine_field, deformed_new)
            solve_seconds = time.perf_counter() - solve_t0

            mse = mean_squared_data_error(deformed_new, corr.points, effective_corr_weights)
            all_closest, all_distances, all_face_indices, _ = target.closest_points(
                deformed_new,
                mode=config.correspondence_mode,
                query_mask=stage_query_mask,
            )
            queried_distances = all_distances[stage_query_mask] if np.any(stage_query_mask) else all_distances
            queried_distances = queried_distances[np.isfinite(queried_distances)]
            all_rmse = float(np.sqrt(np.mean(queried_distances**2))) if queried_distances.size else float("nan")
            all_mean = float(np.mean(queried_distances)) if queried_distances.size else float("nan")
            controlled_summary = summarize_controlled_correspondence_distances(
                all_distances,
                all_closest,
                all_face_indices,
                target,
                config,
                exclude_vertex_mask=correspondence_exclude_vertex_mask,
            )
            summary = summarize_registration_energy(
                source_vertices,
                affine_field,
                edges,
                corr.points,
                effective_corr_weights,
                gamma=config.gamma,
                landmarks=landmarks,
            )
            record = {
                "stage_type": "main",
                "stage": float(stage_idx),
                "iteration": float(iter_idx),
                "stiffness": float(stiffness),
                "landmark_weight": float(lmk_weight),
                "mse": float(mse),
                "mean_distance": float(np.mean(valid_distances)) if valid_distances.size else float("nan"),
                "max_distance": float(np.max(valid_distances)) if valid_distances.size else float("nan"),
                "all_rmse": all_rmse,
                "all_mean_distance": all_mean,
                **controlled_summary,
                "valid_ratio": valid_ratio,
                "correspondence_weight_mean": float(np.mean(corr.weights)) if corr.weights.size else 0.0,
                "camera_depth_enabled": float(depth_stats.get("camera_depth_enabled", 0.0)),
                "camera_depth_weight_mean": float(
                    depth_stats.get("camera_depth_weight_mean", 1.0)
                ),
                "camera_depth_weight_min": float(
                    depth_stats.get("camera_depth_weight_min", 1.0)
                ),
                "camera_depth_reference": float(
                    depth_stats.get("camera_depth_reference", float("nan"))
                ),
                "effective_correspondence_weight_mean": float(np.mean(effective_corr_weights)) if effective_corr_weights.size else 0.0,
                "adaptive_enabled": float(adaptive_stats.get("adaptive_enabled", 0.0)),
                "adaptive_cutoff": float(adaptive_stats.get("adaptive_cutoff", float("nan"))),
                "adaptive_vertices": float(adaptive_stats.get("adaptive_vertices", 0.0)),
                "adaptive_controlled_ratio": float(adaptive_stats.get("adaptive_controlled_ratio", 0.0)),
                "adaptive_band_mean": float(adaptive_stats.get("adaptive_band_mean", 0.0)),
                "adaptive_data_weight_mean": float(adaptive_stats.get("adaptive_data_weight_mean", 1.0)),
                "adaptive_regularization_weight_mean": float(adaptive_stats.get("adaptive_regularization_weight_mean", 1.0)),
                "adaptive_vertex_mass_weight_mean": float(adaptive_stats.get("adaptive_vertex_mass_weight_mean", 1.0)),
                "robust_correspondence": float(bool(config.robust_correspondence)),
                "correspondence_method": corr.method,
                "source_excluded_vertices": float(corr.stats.get("num_excluded_vertices", 0.0)),
                "source_query_vertices": float(corr.stats.get("num_query_vertices", float(np.count_nonzero(correspondence_query_mask)))),
                "target_boundary_correspondences": float(corr.stats.get("num_target_boundary_correspondences", 0.0)),
                "target_boundary_weight": float(corr.stats.get("target_boundary_weight", 0.0)),
                "target_query_faces": float(target.num_query_faces),
                "target_total_faces": float(target.num_total_faces),
                "target_query_vertices": float(target.num_query_vertices),
                "target_total_vertices": float(target.num_total_vertices),
                "trim_cutoff": float(corr.stats.get("trim_cutoff", float("nan"))),
                "robust_cutoff": float(corr.stats.get("robust_cutoff", float("nan"))),
                "system_rows": float(system.A.shape[0]),
                "system_cols": float(system.A.shape[1]),
                "data_rows": float(system.row_counts["data"]),
                "smoothness_rows": float(system.row_counts["smoothness"]),
                "landmark_rows": float(system.row_counts["landmark"]),
                "landmark_delta_rows": float(system.row_counts["landmark_delta"]),
                "affine_prior_rows": float(system.row_counts["affine_prior"]),
                "position_prior_rows": float(system.row_counts["position_prior"]),
                "laplacian_rows": float(system.row_counts["laplacian"]),
                "vertex_mass_rows": float(system.row_counts["vertex_mass"]),
                "edge_vector_rows": float(system.row_counts["edge_vector"]),
                "affine_prior_weight": float(prior_weight),
                "position_prior_weight": float(pos_weight),
                "laplacian_weight": float(lap_weight),
                "vertex_mass_weight": float(mass_weight),
                "edge_vector_weight": float(edge_vec_weight),
                "affine_prior_vertex_weight_mean": (
                    float(np.mean(affine_prior_vertex_weights))
                    if affine_prior_vertex_weights is not None
                    else 1.0
                ),
                "data_vertex_weight_mean": (
                    float(np.mean(data_vertex_weights * adaptive_data_weights))
                    if data_vertex_weights is not None
                    else float(np.mean(adaptive_data_weights))
                ),
                "position_prior_vertex_weight_mean": (
                    float(np.mean(position_prior_vertex_weights))
                    if position_prior_vertex_weights is not None
                    else (1.0 if pos_weight > 0.0 else 0.0)
                ),
                "shape_prior_vertex_weight_mean": (
                    float(np.mean(np.asarray(shape_prior_vertex_weights, dtype=np.float64) * adaptive_vertex_reg_weights))
                    if shape_prior_vertex_weights is not None
                    else float(np.mean(adaptive_vertex_reg_weights))
                ),
                "smoothness_edge_weight_mean": (
                    float(np.mean(smoothness_edge_weights))
                    if smoothness_edge_weights is not None
                    else 1.0
                ),
                "smoothness_edge_weight_max": (
                    float(np.max(smoothness_edge_weights))
                    if smoothness_edge_weights is not None and smoothness_edge_weights.size
                    else 1.0
                ),
                "vertex_mass_vertex_weight_mean": (
                    float(np.mean(np.asarray(shape_prior_vertex_weights, dtype=np.float64) * adaptive_mass_weights))
                    if shape_prior_vertex_weights is not None
                    else float(np.mean(adaptive_mass_weights))
                ),
                "edge_vector_edge_weight_mean": (
                    float(np.mean(edge_vector_edge_weights))
                    if edge_vector_edge_weights is not None
                    else 0.0
                ),
                "edge_vector_edge_weight_max": (
                    float(np.max(edge_vector_edge_weights))
                    if edge_vector_edge_weights is not None and edge_vector_edge_weights.size
                    else 0.0
                ),
                "correspondence_seconds": float(corr_seconds),
                "solve_seconds": float(solve_seconds),
                "iteration_seconds": float(time.perf_counter() - iter_t0),
                "stage_query_fraction": float(stage_query_fraction),
                "num_stage_query_vertices": float(np.count_nonzero(stage_query_mask)),
                "stage_trim_percentile": float(stage_corr_config.trim_percentile or float("nan")),
                "stage_adaptive_data_boost": float(stage_adaptive_boost),
                "num_inverted_faces": float(count_inverted_faces(source_vertices, deformed_new, source_faces)),
                **summary,
            }
            history.append(record)
            if progress_callback is not None:
                progress_callback(record)
            elif config.verbose:
                print(
                    "[nonrigid_icp] "
                    f"stage={stage_idx + 1}/{n_stages} "
                    f"iter={iter_idx + 1}/{config.max_iterations} "
                    f"alpha={float(stiffness):.4g} "
                    f"valid={valid_ratio:.3f} "
                    f"valid_rmse={record['data_rmse']:.6f} "
                    f"lmk_rmse={record.get('landmark_rmse', float('nan')):.6f} "
                    f"controlled_rmse={record['controlled_rmse']:.6f} "
                    f"all_rmse={record['all_rmse']:.6f} "
                    f"adapt={int(record['adaptive_vertices'])} "
                    f"q={int(record['num_stage_query_vertices'])} "
                    f"trim={record['stage_trim_percentile']:.1f} "
                    f"corr={record['correspondence_seconds']:.2f}s "
                    f"solve={record['solve_seconds']:.2f}s",
                    flush=True,
                )

            if last_mse is not None and np.isfinite(mse):
                denom = max(abs(last_mse), 1e-12)
                if abs(last_mse - mse) / denom < float(config.convergence_tol):
                    break
            last_mse = mse

    final_vertices = apply_affine_field(source_vertices, affine_field)
    if bool(config.final_normal_projection):
        pre_projection_vertices = final_vertices.copy()
        max_projection_disp = None
        if float(config.final_projection_max_displacement_scale) > 0.0:
            max_projection_disp = float(config.final_projection_max_displacement_scale) * mesh_length_scale
        final_vertices = project_vertices_along_normals_to_target(
            final_vertices,
            source_faces,
            target,
            query_vertex_mask=correspondence_query_mask,
            blend=(
                final_projection_vertex_blend
                if final_projection_vertex_blend is not None
                else float(config.final_projection_blend)
            ),
            max_displacement=max_projection_disp,
            reference_vertices=pre_projection_vertices if max_projection_disp is not None else None,
        )
    final_corr_config = _correspondence_config_for_stage(config, max(n_stages - 1, 0), n_stages)
    final_query_fraction = _stage_schedule_value(
        max(n_stages - 1, 0),
        n_stages,
        float(config.query_fraction_start),
        float(config.query_fraction_end),
        explicit_schedule=config.stage_query_fractions,
    )
    final_query_mask = build_stage_correspondence_query_mask(
        correspondence_query_mask,
        data_vertex_weights,
        final_query_fraction,
        pin_vertex_ids=query_pin_ids,
        min_vertices=int(config.min_stage_query_vertices),
    )
    corr = compute_correspondences(
        source_vertices,
        source_faces,
        final_vertices,
        target,
        final_corr_config,
        exclude_vertex_mask=correspondence_exclude_vertex_mask,
        query_vertex_mask=final_query_mask,
    )
    apply_camera_depth_correspondence_attenuation(
        corr,
        final_vertices,
        config,
        reference_vertex_mask=camera_depth_reference_vertex_mask,
    )
    apply_correspondence_vertex_controls(corr, data_vertex_weights, correspondence_exclude_vertex_mask)
    return RegistrationResult(
        vertices=final_vertices,
        faces=source_faces.copy(),
        affine_field=affine_field,
        history=history,
        correspondences=corr,
        config=config,
    )


def make_default_config(
    quick: bool = False,
    verbose: bool = True,
) -> NonRigidICPConfig:
    if quick:
        return NonRigidICPConfig(
            stiffness_schedule=(5.0, 2.0, 1.0),
            max_iterations=1,
            verbose=verbose,
        )
    return NonRigidICPConfig(verbose=verbose)


def stem_from_raw_obj(path: str | os.PathLike[str]) -> str:
    stem = Path(path).stem
    if stem.endswith("_phack_raw"):
        return stem[: -len("_phack_raw")]
    return stem


def find_pairs(
    raw_mesh_dir: str | os.PathLike[str],
    target_mesh_dir: str | os.PathLike[str],
) -> list[tuple[str, str, str]]:
    raw_mesh_dir = osp.abspath(os.fspath(raw_mesh_dir))
    target_mesh_dir = osp.abspath(os.fspath(target_mesh_dir))
    pairs: list[tuple[str, str, str]] = []
    for raw_path in sorted(Path(raw_mesh_dir).glob("*_phack_raw.obj")):
        stem = stem_from_raw_obj(raw_path)
        target_path = Path(target_mesh_dir) / f"{stem}.obj"
        if target_path.is_file():
            pairs.append((stem, str(raw_path), str(target_path)))
    return pairs


