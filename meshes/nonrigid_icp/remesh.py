"""
Anisotropic vertex relaxation on a fixed triangle mesh (topology unchanged).

Redistribute vertices in a region (e.g. HACK ``frown_0``) so edge lengths approach a
spatially varying target field (short in the glabella, long elsewhere), while each
vertex stays on the original surface via closest-point projection.

Energy minimized iteratively (gradient-style updates):

    E = w_edge * E_edge + w_surf * E_surf + w_reg * E_reg

- E_edge: spring penalty sum_e (|e| - L_target(e))^2
- E_surf: anchor toward initial positions on the reference surface
- E_reg: uniform Laplacian smoothing in the tangent plane (reduces fold-over)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _path in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import trimesh

from geometry import (
    clamp_relative_vertex_displacement,
    count_inverted_faces,
    extract_edges,
    geodesic_distance_from_mask,
    median_edge_length,
    normalize_vectors,
    validate_vertices_faces,
    vertex_normals,
)
from seg28_masks import (
    HACK_FROWN_REGION_NAMES,
    build_region_boundary_ring_vertex_mask,
    build_seg28_face_vertex_mask,
    face_boundary_seed,
)


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class ReferenceSurfaceProjector:
    """Closest-point projection and face normals on a fixed reference mesh."""

    def __init__(self, reference_vertices: np.ndarray, reference_faces: np.ndarray) -> None:
        reference_vertices, reference_faces = validate_vertices_faces(
            reference_vertices, reference_faces
        )
        self.reference_vertices = reference_vertices
        self.reference_faces = reference_faces
        self._mesh = trimesh.Trimesh(
            vertices=reference_vertices,
            faces=reference_faces,
            process=False,
        )
        self._face_normals = normalize_vectors(
            np.cross(
                reference_vertices[reference_faces[:, 1]]
                - reference_vertices[reference_faces[:, 0]],
                reference_vertices[reference_faces[:, 2]]
                - reference_vertices[reference_faces[:, 0]],
            )
        )

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=np.float64)
        closest, _distances, face_indices = trimesh.proximity.closest_point(
            self._mesh, points
        )
        closest = np.asarray(closest, dtype=np.float64)
        face_indices = np.asarray(face_indices, dtype=np.int64)
        normals = np.zeros_like(closest)
        ok = (face_indices >= 0) & (face_indices < self.reference_faces.shape[0])
        if np.any(ok):
            normals[ok] = self._face_normals[face_indices[ok]]
        fallback = normalize_vectors(vertex_normals(self.reference_vertices, self.reference_faces))
        bad = ~ok | ~np.isfinite(normals).all(axis=1)
        if np.any(bad):
            from scipy.spatial import cKDTree

            tree = cKDTree(self.reference_vertices)
            _, vids = tree.query(points[bad], k=1)
            normals[bad] = fallback[np.asarray(vids, dtype=np.int64)]
        return closest, normalize_vectors(normals)


def build_vertex_target_length_scale(
    vertices: np.ndarray,
    faces: np.ndarray,
    region_mask: np.ndarray,
    *,
    region_target_ratio: float = 0.62,
    transition_band_scale: float = 3.0,
    allowed_mask: np.ndarray | None = None,
    short_at_center: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Per-vertex multiplicative scale for target edge length (1 outside ``region_mask``).

    Default (``short_at_center=True``): **shortest at the region interior / glabella**,
    easing to reference scale (1.0) toward the region seam so the border does not
    spuriously densify.

    Legacy (``short_at_center=False``): short targets hug the region boundary.
    """
    vertices, faces = validate_vertices_faces(vertices, faces)
    region_mask = np.asarray(region_mask, dtype=bool).reshape(-1)
    region_target_ratio = float(np.clip(region_target_ratio, 0.15, 1.0))
    if not np.any(region_mask):
        scale = np.ones(vertices.shape[0], dtype=np.float64)
        return scale, {"enabled": 0.0}

    boundary_seed = face_boundary_seed(faces, region_mask)
    edge_scale = float(median_edge_length(vertices, faces))
    band_dist = float(transition_band_scale) * max(edge_scale, 1e-12)
    dist = geodesic_distance_from_mask(
        vertices,
        faces,
        boundary_seed,
        max_distance=None,
        allowed_mask=allowed_mask if allowed_mask is not None else region_mask,
    )
    scale = np.ones(vertices.shape[0], dtype=np.float64)
    inside = region_mask & np.isfinite(dist)
    inside_ids = np.flatnonzero(inside)
    if inside_ids.size == 0:
        return scale, {"enabled": 0.0}

    depth = dist[inside_ids]
    depth_max = float(np.max(depth))
    if depth_max <= 1e-12:
        scale[inside_ids] = region_target_ratio if short_at_center else 1.0
    else:
        # Normalized depth: 0 on seam, 1 at deepest interior point.
        t = _smoothstep01(np.clip(depth / depth_max, 0.0, 1.0))
        if short_at_center:
            scale[inside_ids] = 1.0 + t * (region_target_ratio - 1.0)
        else:
            scale[inside_ids] = (1.0 - t) * region_target_ratio + t * 1.0

    # Optional outer transition: ease from seam scale 1.0 toward region_target_ratio
    # is only defined inside; outside stays 1.0.
    return scale, {
        "enabled": 1.0,
        "short_at_center": float(short_at_center),
        "region_target_ratio": region_target_ratio,
        "transition_band_scale": float(transition_band_scale),
        "band_geodesic_distance": band_dist,
        "region_interior_depth_max": depth_max,
        "scale_at_seam_mean": float(np.mean(scale[boundary_seed & region_mask]))
        if np.any(boundary_seed & region_mask)
        else 1.0,
        "scale_at_center_mean": float(np.mean(scale[inside_ids][depth >= 0.75 * depth_max]))
        if depth_max > 1e-12
        else float(region_target_ratio),
    }


def build_edge_shortening_weights(
    edges: np.ndarray,
    vertex_length_scale: np.ndarray,
    region_target_ratio: float,
    *,
    active_mask: np.ndarray | None = None,
    power: float = 1.5,
) -> np.ndarray:
    """
    Per-edge weights in [0, 1]: strongest when both endpoints target short edges (center),
    weak on the region seam where scale ~ 1.
    """
    edges = np.asarray(edges, dtype=np.int64)
    vertex_length_scale = np.asarray(vertex_length_scale, dtype=np.float64).reshape(-1)
    ratio = float(np.clip(region_target_ratio, 0.15, 1.0))
    edge_sc = 0.5 * (vertex_length_scale[edges[:, 0]] + vertex_length_scale[edges[:, 1]])
    denom = max(1.0 - ratio, 1e-6)
    weights = np.clip((1.0 - edge_sc) / denom, 0.0, 1.0)
    if float(power) != 1.0:
        weights = np.power(weights, float(power))
    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=bool)
        weights[~active_mask] = 0.0
    return weights


def build_edge_target_lengths(
    reference_vertices: np.ndarray,
    faces: np.ndarray,
    vertex_length_scale: np.ndarray,
    edges: np.ndarray | None = None,
) -> np.ndarray:
    """Target length per undirected edge from reference lengths and vertex scales."""
    reference_vertices, faces = validate_vertices_faces(reference_vertices, faces)
    vertex_length_scale = np.asarray(vertex_length_scale, dtype=np.float64).reshape(-1)
    if edges is None:
        edges = extract_edges(faces)
    else:
        edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return np.empty(0, dtype=np.float64)
    ref_len = np.linalg.norm(
        reference_vertices[edges[:, 1]] - reference_vertices[edges[:, 0]],
        axis=1,
    )
    edge_scale = 0.5 * (vertex_length_scale[edges[:, 0]] + vertex_length_scale[edges[:, 1]])
    return np.maximum(ref_len * edge_scale, 1e-12)


def _relax_edges_toward_targets(
    vertices: np.ndarray,
    edges: np.ndarray,
    target_lengths: np.ndarray,
    edge_weights: np.ndarray,
    move_mask: np.ndarray,
    *,
    blend: float,
) -> np.ndarray:
    """Midpoint edge relaxation (same spirit as geometry.compress_region_edges)."""
    out = vertices.copy()
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0.0:
        return out
    for idx, (i, j) in enumerate(edges):
        move_i = move_mask[i]
        move_j = move_mask[j]
        if not move_i and not move_j:
            continue
        w = float(edge_weights[idx])
        if w <= 0.0:
            continue
        vec = out[j] - out[i]
        cur_len = float(np.linalg.norm(vec))
        if cur_len <= 1e-12:
            continue
        unit = vec / cur_len
        target_len = float(target_lengths[idx])
        if abs(cur_len - target_len) <= 1e-12 * max(target_len, 1e-12):
            continue
        if move_i and move_j:
            mid = 0.5 * (out[i] + out[j])
            new_i = mid - 0.5 * unit * target_len
            new_j = mid + 0.5 * unit * target_len
            out[i] = (1.0 - blend) * out[i] + blend * new_i
            out[j] = (1.0 - blend) * out[j] + blend * new_j
        elif move_i:
            new_i = out[j] - unit * target_len
            out[i] = (1.0 - blend) * out[i] + blend * new_i
        else:
            new_j = out[i] + unit * target_len
            out[j] = (1.0 - blend) * out[j] + blend * new_j
    return out


def _accumulate_tangent_regularization(
    vertices: np.ndarray,
    faces: np.ndarray,
    move_mask: np.ndarray,
    *,
    allowed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Uniform neighbor Laplacian residual (E_reg gradient direction)."""
    from geometry import _mesh_neighbor_average_matrix_restricted, _mesh_neighbor_average_matrix

    n_vertices = vertices.shape[0]
    if allowed_mask is not None:
        neighbor_average = _mesh_neighbor_average_matrix_restricted(
            faces, n_vertices, allowed_mask
        )
    else:
        neighbor_average = _mesh_neighbor_average_matrix(faces, n_vertices)
    delta = neighbor_average @ vertices - vertices
    delta[~move_mask] = 0.0
    return delta


def _tangential_component(delta: np.ndarray, normals: np.ndarray) -> np.ndarray:
    return delta - np.sum(delta * normals, axis=1, keepdims=True) * normals


def anisotropic_surface_vertex_relaxation(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    reference_vertices: np.ndarray | None = None,
    vertex_length_scale: np.ndarray | None = None,
    region_mask: np.ndarray | None = None,
    move_mask: np.ndarray | None = None,
    pin_mask: np.ndarray | None = None,
    edge_weight_mask: np.ndarray | None = None,
    edge_spring_weights: np.ndarray | None = None,
    iterations: int = 80,
    step_size: float = 0.35,
    edge_blend: float | None = None,
    w_edge: float = 1.0,
    w_surf: float = 0.15,
    w_reg: float = 0.08,
    max_step_scale: float = 0.45,
    max_cumulative_displacement_scale: float = 2.5,
    project_each_step: bool = False,
    surface_project_interval: int = 0,
    edge_passes_per_iteration: int = 3,
    reg_allowed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Relax vertex positions on a fixed surface toward an edge-length target field.

    Parameters
    ----------
    vertices
        Initial/deformed vertex positions (same topology as ``faces``).
    reference_vertices
        Fixed surface for projection and reference edge lengths. Defaults to ``vertices``.
    vertex_length_scale
        Per-vertex multiplier for target edge length. If None and ``region_mask`` is set,
        built via :func:`build_vertex_target_length_scale`.
    move_mask
        Vertices allowed to move (tangential + surface projection). If None, uses
        ``region_mask`` interior excluding ``pin_mask``.
    pin_mask
        Vertices held at their reference positions (after projection).
    """
    vertices, faces = validate_vertices_faces(vertices, faces)
    reference_vertices = (
        np.asarray(reference_vertices, dtype=np.float64)
        if reference_vertices is not None
        else vertices.copy()
    )
    reference_vertices, _ = validate_vertices_faces(reference_vertices, faces)

    if vertex_length_scale is None:
        if region_mask is None:
            vertex_length_scale = np.ones(vertices.shape[0], dtype=np.float64)
        else:
            vertex_length_scale, _ = build_vertex_target_length_scale(
                reference_vertices,
                faces,
                region_mask,
            )
    vertex_length_scale = np.asarray(vertex_length_scale, dtype=np.float64).reshape(-1)

    if region_mask is not None:
        region_mask = np.asarray(region_mask, dtype=bool).reshape(-1)
    else:
        region_mask = vertex_length_scale < (1.0 - 1e-6)

    if move_mask is None:
        move_mask = region_mask.copy()
        boundary = face_boundary_seed(faces, move_mask)
        move_mask &= ~boundary
    move_mask = np.asarray(move_mask, dtype=bool).reshape(-1)

    if pin_mask is None:
        pin_mask = region_mask & ~move_mask
    else:
        pin_mask = np.asarray(pin_mask, dtype=bool).reshape(-1)
    move_mask &= ~pin_mask

    projector = ReferenceSurfaceProjector(reference_vertices, faces)
    edges = extract_edges(faces)
    target_lengths = build_edge_target_lengths(
        reference_vertices, faces, vertex_length_scale, edges=edges
    )
    edge_weights = np.ones(edges.shape[0], dtype=np.float64)
    if edge_spring_weights is not None:
        edge_weights *= np.asarray(edge_spring_weights, dtype=np.float64).reshape(-1)
    if edge_weight_mask is not None:
        edge_weight_mask = np.asarray(edge_weight_mask, dtype=bool)
        edge_weights[~edge_weight_mask] = 0.0

    edge_scale = float(median_edge_length(reference_vertices, faces))
    per_step_limit = max(float(max_step_scale), 0.0) * edge_scale
    cumulative_limit = max(float(max_cumulative_displacement_scale), 0.0) * edge_scale

    anchor, _ = projector.project(vertices)
    pinned_positions = reference_vertices[pin_mask].copy()
    out = vertices.copy()
    out[pin_mask] = pinned_positions

    move_ids = np.flatnonzero(move_mask)
    region_edge_keep = region_mask[edges[:, 0]] | region_mask[edges[:, 1]]
    region_edges = edges[region_edge_keep]

    def _mean_region_stretch(verts: np.ndarray) -> float:
        if region_edges.size == 0:
            return float("nan")
        cur = np.linalg.norm(verts[region_edges[:, 1]] - verts[region_edges[:, 0]], axis=1)
        tgt = target_lengths[region_edge_keep]
        return float(np.mean(cur / np.maximum(tgt, 1e-12)))

    def _region_displacement_stats(verts: np.ndarray) -> tuple[float, float]:
        if not np.any(region_mask):
            return float("nan"), float("nan")
        disp = np.linalg.norm(verts[region_mask] - reference_vertices[region_mask], axis=1)
        return float(np.max(disp)), float(np.mean(disp))

    max_disp_before, mean_disp_before = _region_displacement_stats(out)
    stats: dict[str, object] = {
        "iterations": int(iterations),
        "num_move_vertices": int(move_ids.size),
        "num_pin_vertices": int(np.count_nonzero(pin_mask)),
        "num_edges": int(edges.shape[0]),
        "median_reference_edge_length": edge_scale,
        "mean_region_stretch_before": _mean_region_stretch(out),
        "max_region_displacement_before": max_disp_before,
        "mean_region_displacement_before": mean_disp_before,
        "w_edge": float(w_edge),
        "w_surf": float(w_surf),
        "w_reg": float(w_reg),
        "project_each_step": float(project_each_step),
        "edge_passes_per_iteration": int(edge_passes_per_iteration),
        "inverted_faces_before": count_inverted_faces(reference_vertices, out, faces),
    }

    w_edge = float(max(w_edge, 0.0))
    w_surf = float(max(w_surf, 0.0))
    w_reg = float(max(w_reg, 0.0))
    step_size = float(np.clip(step_size, 0.0, 1.0))
    edge_blend = float(np.clip(edge_blend if edge_blend is not None else step_size, 0.0, 1.0))
    edge_blend *= w_edge
    reg_step = step_size * w_reg
    surf_step = step_size * w_surf

    active_edge_idx = np.arange(edges.shape[0], dtype=np.int64)
    if edge_weight_mask is not None:
        active_edge_idx = active_edge_idx[edge_weights > 0.0]
    active_edges = edges[active_edge_idx]
    active_targets = target_lengths[active_edge_idx]
    active_weights = edge_weights[active_edge_idx]
    edge_passes = max(int(edge_passes_per_iteration), 1)
    project_interval = max(int(surface_project_interval), 0)

    for it in range(int(iterations)):
        prev = out.copy()
        out[pin_mask] = pinned_positions

        if edge_blend > 0.0:
            for _ in range(edge_passes):
                out[pin_mask] = pinned_positions
                out = _relax_edges_toward_targets(
                    out,
                    active_edges,
                    active_targets,
                    active_weights,
                    move_mask,
                    blend=edge_blend,
                )

        if reg_step > 0.0:
            out[pin_mask] = pinned_positions
            normals = vertex_normals(out, faces)
            reg_delta = _accumulate_tangent_regularization(
                out,
                faces,
                move_mask,
                allowed_mask=reg_allowed_mask,
            )
            reg_delta = _tangential_component(reg_delta, normals)
            out[move_mask] = out[move_mask] + reg_step * reg_delta[move_mask]

        if surf_step > 0.0:
            out[pin_mask] = pinned_positions
            normals = vertex_normals(out, faces)
            surf_delta = anchor - out
            surf_delta = _tangential_component(surf_delta, normals)
            out[move_mask] = out[move_mask] + surf_step * surf_delta[move_mask]

        if per_step_limit > 0.0:
            out = clamp_relative_vertex_displacement(prev, out, per_step_limit)
        if cumulative_limit > 0.0:
            out = clamp_relative_vertex_displacement(
                reference_vertices,
                out,
                cumulative_limit,
            )

        should_project = project_each_step or (
            project_interval > 0 and (it + 1) % project_interval == 0
        )
        if should_project:
            out, _ = projector.project(out)

        out[pin_mask] = pinned_positions

    if not project_each_step:
        out, _ = projector.project(out)
        out[pin_mask] = pinned_positions

    out[pin_mask] = pinned_positions
    max_disp_after, mean_disp_after = _region_displacement_stats(out)
    stats["max_region_displacement_after"] = max_disp_after
    stats["mean_region_displacement_after"] = mean_disp_after
    stats["mean_region_stretch_after"] = _mean_region_stretch(out)
    stats["inverted_faces_after"] = count_inverted_faces(reference_vertices, out, faces)
    stats["mean_edge_length_ratio_in_region"] = stats["mean_region_stretch_after"]
    if region_edges.size > 0:
        cur = np.linalg.norm(out[region_edges[:, 1]] - out[region_edges[:, 0]], axis=1)
        stats["mean_region_edge_length"] = float(np.mean(cur))
        stats["mean_reference_region_edge_length"] = float(
            np.mean(
                np.linalg.norm(
                    reference_vertices[region_edges[:, 1]]
                    - reference_vertices[region_edges[:, 0]],
                    axis=1,
                )
            )
        )
    return out, stats


def remesh_frown_0_region(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    seg28_pkl_path: str | os.PathLike[str],
    frown_vertex_mask: np.ndarray | None = None,
    region_target_ratio: float = 0.62,
    transition_band_scale: float = 3.0,
    include_transition_ring: bool = True,
    transition_ring_band_scale: float = 1.5,
    pin_boundary_ring: bool = False,
    pin_region_seam_only: bool = True,
    boundary_ring_band_scale: float = 0.0,
    **relax_kwargs: object,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Convenience entry: anisotropic relaxation for HACK ``frown_0`` on a face mesh.

    Topology is unchanged; only vertex positions are updated on the original surface.
    """
    vertices, faces = validate_vertices_faces(vertices, faces)
    reference_vertices = vertices.copy()

    if frown_vertex_mask is None:
        frown_mask, frown_info = build_seg28_face_vertex_mask(
            vertices.shape[0],
            faces,
            seg28_pkl_path,
            region_names=HACK_FROWN_REGION_NAMES,
        )
    else:
        frown_mask = np.asarray(frown_vertex_mask, dtype=bool).reshape(-1)
        frown_info = {"region_names": list(HACK_FROWN_REGION_NAMES)}

    vertex_scale, scale_stats = build_vertex_target_length_scale(
        reference_vertices,
        faces,
        frown_mask,
        region_target_ratio=region_target_ratio,
        transition_band_scale=transition_band_scale,
    )

    move_mask = frown_mask.copy()
    pin_mask = np.zeros(vertices.shape[0], dtype=bool)
    if pin_boundary_ring or pin_region_seam_only:
        if pin_region_seam_only:
            pin_mask |= face_boundary_seed(faces, frown_mask) & frown_mask
        if pin_boundary_ring and float(boundary_ring_band_scale) > 0.0:
            pin_mask |= build_region_boundary_ring_vertex_mask(
                reference_vertices,
                faces,
                frown_mask,
                band_width_scale=boundary_ring_band_scale,
            )
        move_mask &= ~pin_mask

    reg_allowed_mask = frown_mask.copy()
    if include_transition_ring:
        transition_ring = build_region_boundary_ring_vertex_mask(
            reference_vertices,
            faces,
            frown_mask,
            band_width_scale=transition_ring_band_scale,
        )
        # Outer shell of the transition band may move slightly for smooth length field.
        move_mask |= transition_ring & ~pin_mask
        reg_allowed_mask |= transition_ring

    edges = extract_edges(faces)
    touch_region = frown_mask[edges[:, 0]] | frown_mask[edges[:, 1]]
    edge_spring_weights = build_edge_shortening_weights(
        edges,
        vertex_scale,
        region_target_ratio,
        active_mask=touch_region,
    )
    if include_transition_ring:
        edge_spring_weights *= (reg_allowed_mask[edges[:, 0]] | reg_allowed_mask[edges[:, 1]]).astype(
            np.float64
        )

    relax_kwargs_clean = dict(relax_kwargs)
    out, relax_stats = anisotropic_surface_vertex_relaxation(
        vertices,
        faces,
        reference_vertices=reference_vertices,
        vertex_length_scale=vertex_scale,
        region_mask=frown_mask,
        move_mask=move_mask,
        pin_mask=pin_mask,
        edge_spring_weights=edge_spring_weights,
        reg_allowed_mask=reg_allowed_mask,
        **relax_kwargs_clean,
    )
    stats = {
        "frown": frown_info,
        "length_scale": scale_stats,
        "relax": relax_stats,
        "num_frown_vertices": int(np.count_nonzero(frown_mask)),
        "num_move_vertices": int(np.count_nonzero(move_mask)),
    }
    return out, stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Anisotropic frown_0 vertex relaxation (topology-preserving remesh)."
    )
    parser.add_argument(
        "--input_obj", type=str, 
        default="/media/ubuntu/SSD/hack-3dgs/test_nicp/101001_73_repair.obj",
        help="Input OBJ mesh path")
    parser.add_argument(
        "--output_obj", type=str, 
        default="/media/ubuntu/SSD/hack-3dgs/test_nicp/101001_73_remesh.obj",
        help="Output OBJ mesh path")
    parser.add_argument(
        "--seg28-pkl",
        type=str,
        default="/media/ubuntu/SSD/hack-3dgs/data/hack_data/HACK_seg_28.pkl",
        help="HACK_seg_28 pickle for frown_0 region mask",
    )
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--region-target-ratio", type=float, default=0.25)
    parser.add_argument("--transition-band-scale", type=float, default=3.0)
    parser.add_argument("--step-size", type=float, default=0.35)
    parser.add_argument("--edge-blend", type=float, default=0.65)
    parser.add_argument("--edge-passes", type=int, default=3, help="Edge relax sweeps per iteration")
    parser.add_argument("--w-edge", type=float, default=1.0)
    parser.add_argument(
        "--w-surf",
        type=float,
        default=0.0,
        help="Pull toward initial positions (0 = rely on end projection only)",
    )
    parser.add_argument("--w-reg", type=float, default=0.06)
    parser.add_argument(
        "--project-each-step",
        action="store_true",
        help="Project every iteration (often cancels tangential motion)",
    )
    parser.add_argument(
        "--pin-boundary-ring",
        action="store_true",
        help="Pin geodesic band around frown_0 (fewer movable vertices)",
    )
    parser.add_argument("--boundary-ring-scale", type=float, default=1.25)
    parser.add_argument(
        "--pin-region-seam",
        action="store_true",
        default=True,
        help="Pin frown_0 seam vertices so only interior densifies (default: on)",
    )
    parser.add_argument(
        "--no-pin-region-seam",
        action="store_false",
        dest="pin_region_seam",
        help="Allow seam vertices to move",
    )
    return parser.parse_args()


def _obj_has_valid_uv(obj) -> bool:
    """True when ``read_obj`` UV tables are present and aligned with faces."""
    vts = getattr(obj, "vts", None)
    fvts = getattr(obj, "fvts", None)
    fvs = getattr(obj, "fvs", None)
    if vts is None or fvts is None or fvs is None:
        return False
    vts = np.asarray(vts)
    if vts.size == 0 or vts.ndim < 2:
        return False
    if len(fvts) != len(fvs):
        return False
    for fv_row, fvt_row in zip(fvs, fvts):
        if np.asarray(fvt_row).reshape(-1).size != np.asarray(fv_row).reshape(-1).size:
            return False
    return True


def _obj_to_mesh_info(obj, vertices: np.ndarray) -> dict:
    """Build ``write_mesh_obj`` payload from :func:`read_obj` result."""
    mesh_info: dict = {
        "v": np.asarray(vertices, dtype=np.float32),
        "fv": obj.fvs,
    }
    if _obj_has_valid_uv(obj):
        mesh_info["vt"] = obj.vts
        mesh_info["fvt"] = obj.fvts
    if getattr(obj, "vns", None) is not None and len(obj.vns) > 0:
        mesh_info["vn"] = obj.vns
    return mesh_info


def main() -> None:
    args = _parse_args()
    from utils.mesh import read_obj, write_mesh_obj

    obj = read_obj(args.input_obj)
    vertices = np.asarray(obj.vs, dtype=np.float64)
    faces = np.asarray(obj.fvs, dtype=np.int64)

    out, stats = remesh_frown_0_region(
        vertices,
        faces,
        seg28_pkl_path=args.seg28_pkl,
        region_target_ratio=args.region_target_ratio,
        transition_band_scale=args.transition_band_scale,
        iterations=args.iterations,
        step_size=args.step_size,
        edge_blend=args.edge_blend,
        edge_passes_per_iteration=args.edge_passes,
        w_edge=args.w_edge,
        w_surf=args.w_surf,
        w_reg=args.w_reg,
        project_each_step=bool(args.project_each_step),
        pin_boundary_ring=bool(args.pin_boundary_ring),
        pin_region_seam_only=bool(args.pin_region_seam),
        boundary_ring_band_scale=(
            float(args.boundary_ring_scale) if args.pin_boundary_ring else 0.0
        ),
    )
    write_mesh_obj(_obj_to_mesh_info(obj, out), args.output_obj)
    relax = stats["relax"]
    print(
        f"[remesh] wrote {args.output_obj} "
        f"frown_0 verts={stats['num_frown_vertices']} "
        f"move={stats['num_move_vertices']} "
        f"disp max {relax.get('max_region_displacement_before', 0):.6f} -> "
        f"{relax.get('max_region_displacement_after', 0):.6f} "
        f"stretch {relax['mean_region_stretch_before']:.4f} -> "
        f"{relax['mean_region_stretch_after']:.4f} "
        f"edge_len {relax.get('mean_reference_region_edge_length', 0):.6f} -> "
        f"{relax.get('mean_region_edge_length', 0):.6f} "
        f"inverted {relax['inverted_faces_before']} -> {relax['inverted_faces_after']}"
    )


if __name__ == "__main__":
    main()
