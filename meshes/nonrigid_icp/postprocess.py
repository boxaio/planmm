from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np

from seg28_masks import (
    HACK_CHIN_REGION_NAMES,
    HACK_EAR_REGION_NAMES,
    HACK_FOREHEAD_REGION_NAMES,
    HACK_FROWN_REGION_NAMES,
    HACK_NECK_FRONT_REGION_NAMES,
    HACK_SEG28_FACE_REGION_NAMES,
    HACK_SKULL_REGION_NAMES,
    build_neck_protect_vertex_mask,
    build_seg28_face_vertex_mask,
    face_boundary_seed,
    region_cross_boundary_seed,
)
from geometry import (
    align_region_pair_cross_edges_to_face_tangent,
    clamp_relative_vertex_displacement,
    extract_edges,
    geodesic_distance_from_mask,
    harmonize_region_seam_displacement,
    bilaplacian_smooth_vertices,
    compute_vertex_laplacian_defect_scores,
    median_edge_length,
    vertex_normals,
    optimize_face_nonface_junction_normal_smoothness,
    optimize_nonface_region_smoothness,
    project_nonface_seam_onto_face_tangent_planes,
    relax_edges_in_vertex_mask_stretch,
    relax_face_nonface_cross_edge_stretch,
    relax_region_pair_cross_edge_displacement,
    relax_region_pair_cross_edge_stretch,
    repair_inverted_faces,
    restore_region_pair_hairline_template_layout,
    smooth_vertex_scalar_field,
    taubin_smooth_vertices,
    weld_region_pair_cross_edge_positions,
)


def _postprocess_lite_with_darap(args: argparse.Namespace) -> bool:
    """Regional dARAP runs after postprocess; skip redundant heavy seam/bump passes."""
    if bool(getattr(args, "no_postprocess_lite_with_darap", False)):
        return False
    if bool(getattr(args, "no_regional_darap_normal_deform", False)):
        return False
    return bool(getattr(args, "postprocess_lite_with_darap", True))


def _example_fast_enabled(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "no_example_fast", False)):
        return False
    return bool(getattr(args, "example_fast", False))


def _postprocess_ultra_lite(args: argparse.Namespace) -> bool:
    """example_fast + regional dARAP: skip remaining post-ICP smooth/restore (dARAP handles face)."""
    return _postprocess_lite_with_darap(args) and _example_fast_enabled(args)


def build_region_junction_motion_ramp(
    vertices: np.ndarray,
    faces: np.ndarray,
    anchor_region_mask: np.ndarray,
    deform_region_mask: np.ndarray,
    *,
    ramp_width_scale: float = 5.5,
    bidirectional: bool = False,
    ease_mode: str = "softerstep",
    shared_geodesic_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, np.ndarray, float]:
    """
    Ramp smooth weights on a non-face region near its boundary with a face region.

    Returns 1 far from the seam on the deform region and 0 on the seam ring.
    When bidirectional=True, also returns an anchor-side ramp that fades to 0
    on the anchor region near the same seam.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    anchor_region_mask = np.asarray(anchor_region_mask, dtype=bool).reshape(-1)
    deform_region_mask = np.asarray(deform_region_mask, dtype=bool).reshape(-1)
    edge_scale = float(median_edge_length(vertices, faces))
    ramp_width = max(float(ramp_width_scale) * edge_scale, edge_scale)
    seed = region_cross_boundary_seed(faces, anchor_region_mask, deform_region_mask)
    deform_geodesic_mask = (
        np.asarray(shared_geodesic_mask, dtype=bool).reshape(-1)
        if shared_geodesic_mask is not None
        else deform_region_mask
    )
    dist = geodesic_distance_from_mask(
        vertices,
        faces,
        seed,
        max_distance=ramp_width,
        allowed_mask=deform_geodesic_mask,
    )
    ramp = np.ones(anchor_region_mask.shape[0], dtype=np.float64)
    shell = deform_region_mask & np.isfinite(dist)
    ramp[shell] = _ease_region_ramp(dist[shell] / ramp_width, ease_mode)
    if not bidirectional:
        return ramp, float(ramp_width)

    anchor_geodesic_mask = (
        np.asarray(shared_geodesic_mask, dtype=bool).reshape(-1)
        if shared_geodesic_mask is not None
        else anchor_region_mask
    )
    anchor_dist = geodesic_distance_from_mask(
        vertices,
        faces,
        seed,
        max_distance=ramp_width,
        allowed_mask=anchor_geodesic_mask,
    )
    anchor_ramp = np.ones(anchor_region_mask.shape[0], dtype=np.float64)
    anchor_shell = anchor_region_mask & np.isfinite(anchor_dist)
    anchor_ramp[anchor_shell] = _ease_region_ramp(anchor_dist[anchor_shell] / ramp_width, ease_mode)
    return ramp, anchor_ramp, float(ramp_width)


def build_region_seam_band_soften_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    anchor_region_mask: np.ndarray,
    deform_region_mask: np.ndarray,
    *,
    band_width_scale: float = 6.0,
    peak_weight: float = 0.62,
    ease_mode: str = "softerstep",
    bidirectional: bool = False,
    shared_geodesic_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Weights for geometry-only softening in a narrow band near a region seam.

    Operates on the current mesh state (not an ICP reference), so it does not
    undo non-face deformation from the main junction smooth pass.
    """
    ramp_kwargs = {}
    if shared_geodesic_mask is not None:
        ramp_kwargs["shared_geodesic_mask"] = shared_geodesic_mask
    if bidirectional:
        deform_ramp, anchor_ramp, band_width = build_region_junction_motion_ramp(
            vertices,
            faces,
            anchor_region_mask,
            deform_region_mask,
            ramp_width_scale=band_width_scale,
            bidirectional=True,
            ease_mode=ease_mode,
            **ramp_kwargs,
        )
    else:
        deform_ramp, band_width = build_region_junction_motion_ramp(
            vertices,
            faces,
            anchor_region_mask,
            deform_region_mask,
            ramp_width_scale=band_width_scale,
            bidirectional=False,
            ease_mode=ease_mode,
            **ramp_kwargs,
        )
        anchor_ramp = None

    weights = np.zeros(deform_ramp.shape[0], dtype=np.float64)
    deform_region_mask = np.asarray(deform_region_mask, dtype=bool).reshape(-1)
    anchor_region_mask = np.asarray(anchor_region_mask, dtype=bool).reshape(-1)
    peak_weight = float(np.clip(peak_weight, 0.0, 1.0))
    deform_shell = deform_region_mask & (deform_ramp < 0.995)
    weights[deform_shell] = peak_weight * (1.0 - deform_ramp[deform_shell]) ** 1.15
    if bidirectional and anchor_ramp is not None:
        anchor_shell = anchor_region_mask & (anchor_ramp < 0.995)
        weights[anchor_shell] = np.maximum(
            weights[anchor_shell],
            peak_weight * (1.0 - anchor_ramp[anchor_shell]) ** 1.15,
        )
    return weights, float(band_width)


def build_forehead_skull_bump_smooth_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    forehead_mask: np.ndarray,
    skull_mask: np.ndarray,
    *,
    detect_band_scale: float = 6.0,
    detect_threshold_scale: float = 0.032,
    detect_percentile: float = 68.0,
    peak_weight: float = 1.0,
    support_weight_scale: float = 0.42,
    cross_edge_score_boost_scale: float = 0.06,
    min_bump_vertices: int = 12,
    max_bump_fraction: float = 0.22,
    score_p90_blend: float = 0.55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """
    Detect vertices that actually need smoothing on the forehead/skull hairline.

    Scores combine Laplacian / Bi-Laplacian defect and normal jitter inside a
    cross-seam geodesic shell; bumps get full weight, their one-ring support
    gets a reduced weight for connectivity.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    forehead_mask = np.asarray(forehead_mask, dtype=bool).reshape(-1)
    skull_mask = np.asarray(skull_mask, dtype=bool).reshape(-1)
    n_vertices = vertices.shape[0]
    stats: dict[str, float] = {
        "candidate": 0.0,
        "bumps": 0.0,
        "support": 0.0,
        "score_p90": 0.0,
        "score_p50": 0.0,
        "score_max": 0.0,
        "thresh": 0.0,
    }
    weights = np.zeros(n_vertices, dtype=np.float64)
    bump_mask = np.zeros(n_vertices, dtype=bool)
    candidate_mask = np.zeros(n_vertices, dtype=bool)
    if faces.size == 0 or not np.any(forehead_mask) or not np.any(skull_mask):
        return weights, bump_mask, candidate_mask, stats

    fs_allowed = forehead_mask | skull_mask
    edge_scale = float(median_edge_length(vertices, faces))
    cross_seed = region_cross_boundary_seed(faces, forehead_mask, skull_mask)
    cross_edges = extract_edges(faces)
    cross_pair = np.zeros(cross_edges.shape[0], dtype=bool)
    cross_verts = np.array([], dtype=np.int64)
    if cross_edges.size > 0:
        cross_pair = (forehead_mask[cross_edges[:, 0]] & skull_mask[cross_edges[:, 1]]) | (
            skull_mask[cross_edges[:, 0]] & forehead_mask[cross_edges[:, 1]]
        )
        if np.any(cross_pair):
            cross_verts = np.unique(cross_edges[cross_pair].reshape(-1))

    detect_dist = float(detect_band_scale) * edge_scale
    if detect_dist > 0.0:
        candidate_mask = fs_allowed & np.isfinite(
            geodesic_distance_from_mask(
                vertices,
                faces,
                cross_seed,
                max_distance=detect_dist,
                allowed_mask=fs_allowed,
            )
        )
    if cross_verts.size:
        candidate_mask[cross_verts] = True

    if not np.any(candidate_mask):
        return weights, bump_mask, candidate_mask, stats

    lap_mag, bi_mag, normal_jitter = compute_vertex_laplacian_defect_scores(
        vertices,
        faces,
        allowed_vertex_mask=fs_allowed,
    )
    score = lap_mag + 0.65 * bi_mag + normal_jitter * edge_scale * 0.55
    if cross_verts.size:
        boost = float(cross_edge_score_boost_scale) * edge_scale
        score[cross_verts] += boost

    active = candidate_mask & np.isfinite(score)
    if not np.any(active):
        return weights, bump_mask, candidate_mask, stats

    active_scores = score[active]
    score_p90 = float(np.percentile(active_scores, 90))
    score_p50 = float(np.percentile(active_scores, 50))
    score_max = float(np.max(active_scores))
    pct_thresh = float(np.percentile(active_scores, float(np.clip(detect_percentile, 75.0, 99.5))))
    abs_thresh = float(detect_threshold_scale) * edge_scale
    rel_thresh = float(np.clip(score_p90_blend, 0.2, 0.95)) * score_p90
    thresh = max(abs_thresh, pct_thresh, rel_thresh)
    bump_mask = active & (score >= thresh)

    max_frac = float(np.clip(max_bump_fraction, 0.02, 1.0))
    max_bumps = max(int(min_bump_vertices), int(np.ceil(max_frac * np.count_nonzero(candidate_mask))))
    bump_count = int(np.count_nonzero(bump_mask))
    if bump_count > max_bumps:
        order = np.argsort(score[active])[::-1]
        active_ids = np.flatnonzero(active)
        bump_mask[:] = False
        bump_mask[active_ids[order[:max_bumps]]] = True
    elif bump_count < int(min_bump_vertices):
        order = np.argsort(score[active])[::-1]
        active_ids = np.flatnonzero(active)
        take = min(int(min_bump_vertices), active_ids.size)
        bump_mask[:] = False
        bump_mask[active_ids[order[:take]]] = True

    peak = float(np.clip(peak_weight, 0.0, 1.0))
    support_scale = float(np.clip(support_weight_scale, 0.0, 1.0))
    weights[bump_mask] = peak

    if support_scale > 0.0 and np.any(bump_mask) and cross_edges.size > 0:
        support = np.zeros(n_vertices, dtype=bool)
        for i, j in cross_edges:
            if bump_mask[i] and fs_allowed[j]:
                support[j] = True
            if bump_mask[j] and fs_allowed[i]:
                support[i] = True
        support &= ~bump_mask
        support &= candidate_mask
        weights[support] = np.maximum(weights[support], peak * support_scale)

    stats["candidate"] = float(np.count_nonzero(candidate_mask))
    stats["bumps"] = float(np.count_nonzero(bump_mask))
    stats["support"] = float(np.count_nonzero((weights > 1e-6) & ~bump_mask))
    stats["score_p90"] = score_p90
    stats["score_p50"] = score_p50
    stats["score_max"] = score_max
    stats["thresh"] = float(thresh)
    return weights, bump_mask, candidate_mask, stats


def build_face_interior_preserve_mask(
    face_mask: np.ndarray,
    seam_deform_mask: np.ndarray,
    vertices: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    forehead_mask: np.ndarray | None = None,
    skull_mask: np.ndarray | None = None,
    *,
    forehead_seam_exclusion_band_scale: float = 0.0,
) -> np.ndarray:
    """
    Face vertices to roll back to the post-ICP shape.

    Excludes the explicit seam deform band and, optionally, a wider forehead
    ring around the forehead/skull cross edges so interior restore does not
    re-open the hairline step.
    """
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    seam_deform_mask = np.asarray(seam_deform_mask, dtype=bool).reshape(-1)
    interior = face_mask & ~seam_deform_mask
    band_scale = float(forehead_seam_exclusion_band_scale)
    if (
        band_scale > 0.0
        and vertices is not None
        and faces is not None
        and forehead_mask is not None
        and skull_mask is not None
    ):
        forehead_mask = np.asarray(forehead_mask, dtype=bool).reshape(-1)
        skull_mask = np.asarray(skull_mask, dtype=bool).reshape(-1)
        if np.any(forehead_mask) and np.any(skull_mask):
            vertices = np.asarray(vertices, dtype=np.float64)
            faces = np.asarray(faces, dtype=np.int64)
            edge_scale = float(median_edge_length(vertices, faces))
            cross_seed = region_cross_boundary_seed(faces, forehead_mask, skull_mask)
            cross_seed &= forehead_mask
            near_forehead = forehead_mask & np.isfinite(
                geodesic_distance_from_mask(
                    vertices,
                    faces,
                    cross_seed,
                    max_distance=band_scale * edge_scale,
                    allowed_mask=face_mask,
                )
            )
            interior &= ~near_forehead
    return interior


def build_forehead_skull_seam_influence_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    forehead_mask: np.ndarray,
    skull_mask: np.ndarray,
    *,
    band_width_scale: float = 10.0,
    include_cross_edge_face_one_ring: bool = True,
) -> np.ndarray:
    """
    Vertices that must stay at the seam-pass positions after face-interior restore.

    Includes a geodesic ball plus every vertex on triangles incident to
    forehead/skull cross edges (so cheek motion does not alter hairline normals).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    forehead_mask = np.asarray(forehead_mask, dtype=bool).reshape(-1)
    skull_mask = np.asarray(skull_mask, dtype=bool).reshape(-1)
    n_vertices = vertices.shape[0]
    influence = np.zeros(n_vertices, dtype=bool)
    if faces.size == 0 or not np.any(forehead_mask) or not np.any(skull_mask):
        return influence

    if include_cross_edge_face_one_ring:
        mesh_edges = extract_edges(faces)
        cross_edge = (forehead_mask[mesh_edges[:, 0]] & skull_mask[mesh_edges[:, 1]]) | (
            skull_mask[mesh_edges[:, 0]] & forehead_mask[mesh_edges[:, 1]]
        )
        if np.any(cross_edge):
            cross_pairs = {
                tuple(sorted((int(a), int(b))))
                for a, b in mesh_edges[cross_edge]
            }
            for tri in np.asarray(faces, dtype=np.int64):
                tri_edges = (
                    tuple(sorted((int(tri[0]), int(tri[1])))),
                    tuple(sorted((int(tri[1]), int(tri[2])))),
                    tuple(sorted((int(tri[2]), int(tri[0])))),
                )
                if any(edge in cross_pairs for edge in tri_edges):
                    influence[tri] = True

    band_width_scale = float(band_width_scale)
    if band_width_scale > 0.0:
        edge_scale = float(median_edge_length(vertices, faces))
        cross_seed = region_cross_boundary_seed(faces, forehead_mask, skull_mask)
        cross_seed |= forehead_mask & skull_mask
        distance = geodesic_distance_from_mask(
            vertices,
            faces,
            cross_seed,
            max_distance=band_width_scale * edge_scale,
        )
        influence |= np.isfinite(distance)
    return influence


def apply_forehead_skull_seam_finalize(
    template_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    forehead_mask: np.ndarray,
    skull_mask: np.ndarray,
    forehead_face_relax_mask: np.ndarray,
    skull_relax_mask: np.ndarray,
    *,
    weld_iterations: int = 20,
    weld_blend: float = 1.0,
    tangent_iterations: int = 12,
    tangent_blend: float = 1.0,
    template_restore_blend: float = 0.9,
    pin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Re-apply seam welding / tangent / template layout after face-interior restore."""
    move_mask = np.asarray(forehead_face_relax_mask, dtype=bool) | np.asarray(
        skull_relax_mask, dtype=bool
    )
    out = np.asarray(vertices, dtype=np.float64)
    if weld_iterations > 0 and np.any(forehead_mask) and np.any(skull_mask):
        out = weld_region_pair_cross_edge_positions(
            out,
            faces,
            forehead_mask,
            skull_mask,
            iterations=int(weld_iterations),
            blend=float(weld_blend),
            move_mask=move_mask,
            pin_mask=pin_mask,
        )
    if tangent_iterations > 0 and np.any(forehead_mask) and np.any(skull_mask):
        out = align_region_pair_cross_edges_to_face_tangent(
            out,
            faces,
            face_mask,
            forehead_mask,
            skull_mask,
            iterations=int(tangent_iterations),
            blend=float(tangent_blend),
            move_nonface_mask=skull_relax_mask,
            pin_mask=pin_mask,
        )
    if template_restore_blend > 0.0 and np.any(forehead_mask) and np.any(skull_mask):
        out = restore_region_pair_hairline_template_layout(
            template_vertices,
            out,
            faces,
            face_mask,
            forehead_mask,
            skull_mask,
            skull_anchor_mask=skull_relax_mask,
            forehead_adjust_mask=forehead_face_relax_mask,
            forehead_blend=float(template_restore_blend),
        )
    return out


def resolve_mandatory_local_smooth_iters(
    args: argparse.Namespace,
    *,
    force: bool = True,
) -> tuple[int, int, int, int, int]:
    """Enforce non-zero local smooth iteration counts (robust_auto mandatory pass)."""
    junction = int(getattr(args, "final_local_smooth_iterations", 0))
    taubin = int(getattr(args, "final_local_smooth_taubin_iterations", 0))
    harmonize = int(getattr(args, "final_local_smooth_harmonize_iterations", 0))
    weld = int(getattr(args, "final_local_smooth_post_weld_iterations", 0))
    tangent = int(getattr(args, "final_local_smooth_post_tangent_iterations", 0))
    if force and not bool(getattr(args, "no_mandatory_final_smooth", False)):
        junction = max(junction, int(getattr(args, "mandatory_local_smooth_min_junction_iters", 36)))
        taubin = max(taubin, int(getattr(args, "mandatory_local_smooth_min_taubin_iters", 24)))
        harmonize = max(
            harmonize, int(getattr(args, "mandatory_local_smooth_min_harmonize_iters", 14))
        )
        weld = max(weld, int(getattr(args, "mandatory_local_smooth_min_post_weld_iters", 18)))
        tangent = max(
            tangent, int(getattr(args, "mandatory_local_smooth_min_post_tangent_iters", 14))
        )
    return junction, taubin, harmonize, weld, tangent


def build_local_seam_smooth_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    args: argparse.Namespace,
    *,
    exclude_vertex_mask: np.ndarray | None = None,
    mandatory: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Vertex weights and displacement caps for a final localized seam smooth pass."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = vertices.shape[0]
    weights = np.zeros(n_vertices, dtype=np.float64)
    limits = np.full(n_vertices, np.inf, dtype=np.float64)
    edge_scale = float(median_edge_length(vertices, faces))
    if edge_scale <= 0.0:
        return weights, limits, edge_scale

    peak = float(np.clip(getattr(args, "final_local_smooth_peak_weight", 1.0), 0.0, 1.0))
    if mandatory:
        peak = max(
            peak,
            float(getattr(args, "mandatory_local_smooth_min_peak_weight", 1.0)),
        )
    max_scale = float(getattr(args, "final_local_smooth_max_displacement_scale", 0.28))
    band_boost = (
        float(getattr(args, "mandatory_local_smooth_band_boost", 1.2)) if mandatory else 1.0
    )

    face_mask, _ = build_seg28_face_vertex_mask(
        n_vertices,
        faces,
        seg28_pkl_path,
        region_names=HACK_SEG28_FACE_REGION_NAMES,
    )
    nonface_mask = ~np.asarray(face_mask, dtype=bool)
    forehead_mask, _ = build_seg28_face_vertex_mask(
        n_vertices,
        faces,
        seg28_pkl_path,
        region_names=HACK_FOREHEAD_REGION_NAMES,
    )
    skull_mask, _ = build_seg28_face_vertex_mask(
        n_vertices,
        faces,
        seg28_pkl_path,
        region_names=HACK_SKULL_REGION_NAMES,
    )
    if np.any(forehead_mask) and np.any(skull_mask):
        fs_weights, _ = build_region_seam_band_soften_weights(
            vertices,
            faces,
            forehead_mask,
            skull_mask,
            band_width_scale=float(
                getattr(args, "final_local_smooth_forehead_skull_band_scale", 6.5)
            )
            * band_boost,
            peak_weight=peak,
            bidirectional=True,
            shared_geodesic_mask=forehead_mask | skull_mask,
        )
        weights = np.maximum(weights, fs_weights)

    junction_weights, junction_width, junction_seam_distance = (
        build_face_nonface_junction_normal_smooth_weights(
            vertices,
            faces,
            face_mask,
            nonface_mask,
            junction_width_scale=float(
                getattr(args, "final_local_smooth_face_nonface_band_scale", 6.0)
            )
            * band_boost,
        )
    )
    fnf_peak = float(
        np.clip(getattr(args, "final_local_smooth_face_nonface_peak_weight", peak), 0.0, 1.0)
    )
    if mandatory:
        fnf_peak = max(fnf_peak, peak)
    weights[nonface_mask] = np.maximum(
        weights[nonface_mask],
        fnf_peak * junction_weights[nonface_mask],
    )
    face_boundary = face_boundary_seed(faces, face_mask)
    face_dist = geodesic_distance_from_mask(
        vertices,
        faces,
        face_boundary,
        max_distance=max(float(junction_width), edge_scale),
        allowed_mask=face_mask,
    )
    face_shell = face_mask & np.isfinite(face_dist)
    if np.any(face_shell):
        ramp = _smoothstep(
            np.clip(1.0 - face_dist[face_shell] / max(float(junction_width), 1e-12), 0.0, 1.0)
        )
        weights[face_shell] = np.maximum(weights[face_shell], fnf_peak * ramp)

    if exclude_vertex_mask is not None:
        weights[np.asarray(exclude_vertex_mask, dtype=bool).reshape(-1)] = 0.0

    active = weights > 1e-6
    if np.any(active):
        seam_cap = max_scale * edge_scale
        limits[active] = seam_cap * weights[active]
        shell = nonface_mask & active & np.isfinite(junction_seam_distance)
        if np.any(shell):
            t = _smoothstep(
                np.clip(
                    junction_seam_distance[shell] / max(float(junction_width), 1e-12),
                    0.0,
                    1.0,
                )
            )
            tight = float(getattr(args, "final_local_smooth_seam_displacement_scale", 0.10))
            limits[shell] = (tight + (max_scale - tight) * t) * edge_scale

    return np.clip(weights, 0.0, 1.0), limits, edge_scale


def apply_local_seam_band_smoothness(
    vertices: np.ndarray,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    args: argparse.Namespace,
    *,
    template_vertices: np.ndarray | None = None,
    landmark_vertex_ids: np.ndarray | None = None,
    correspondence_exclude_vertex_mask: np.ndarray | None = None,
    force_mandatory: bool = False,
) -> np.ndarray:
    """
    Localized normal + Taubin smoothing on the forehead/skull and face/non-face
    transition bands after the main pipeline finishes. Displacement is capped to
    the pre-smooth geometry so seams stay continuous.
    """
    if seg28_pkl_path is None:
        return np.asarray(vertices, dtype=np.float64)
    mandatory = bool(force_mandatory) or not bool(
        getattr(args, "no_mandatory_final_smooth", False)
    )
    junction_iters, taubin_iters, harmonize_iters, weld_iters, tangent_iters = (
        resolve_mandatory_local_smooth_iters(args, force=mandatory)
    )

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    anchor_vertices = vertices.copy()
    pin_mask = np.zeros(vertices.shape[0], dtype=bool)
    if correspondence_exclude_vertex_mask is not None:
        pin_mask |= np.asarray(correspondence_exclude_vertex_mask, dtype=bool)
    if landmark_vertex_ids is not None and landmark_vertex_ids.size:
        pin_mask[landmark_vertex_ids] = True
    ear_mask, _ = build_seg28_face_vertex_mask(
        vertices.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_EAR_REGION_NAMES,
    )
    pin_mask |= np.asarray(ear_mask, dtype=bool)

    weights, displacement_limits, edge_scale = build_local_seam_smooth_weights(
        vertices,
        faces,
        seg28_pkl_path,
        args,
        exclude_vertex_mask=pin_mask,
        mandatory=mandatory,
    )
    if not np.any(weights > 1e-6):
        return vertices

    pin_vertex_mask = (weights <= 1e-6) | pin_mask
    face_mask, _ = build_seg28_face_vertex_mask(
        vertices.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_SEG28_FACE_REGION_NAMES,
    )
    forehead_mask, _ = build_seg28_face_vertex_mask(
        vertices.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_FOREHEAD_REGION_NAMES,
    )
    skull_mask, _ = build_seg28_face_vertex_mask(
        vertices.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_SKULL_REGION_NAMES,
    )
    (
        forehead_face_relax,
        skull_relax,
        _,
        _,
        _,
        _,
        _,
    ) = build_forehead_skull_seam_flex_masks(
        vertices,
        faces,
        seg28_pkl_path,
        band_width_scale=float(args.forehead_skull_face_flex_band_scale),
        inner_ramp_threshold=float(
            args.nonface_smooth_forehead_skull_harmonize_inner_threshold
        ),
    )

    if junction_iters > 0:
        vertices = optimize_face_nonface_junction_normal_smoothness(
            vertices,
            faces,
            face_mask,
            weights,
            iterations=junction_iters,
            lamb=float(getattr(args, "final_local_smooth_lambda", 0.44)),
            mu=float(getattr(args, "final_local_smooth_mu", -0.48)),
            tangential_only=not bool(
                getattr(args, "final_local_smooth_normal_on", True)
            ),
            normal_alignment_step=float(
                getattr(args, "final_local_smooth_normal_alignment_step", 0.48)
            ),
            max_displacement=displacement_limits,
            max_iteration_displacement_scale=float(
                getattr(args, "final_local_smooth_max_iteration_step_scale", 0.18)
            ),
            pin_vertex_mask=pin_vertex_mask,
        )
    if taubin_iters > 0:
        vertices = taubin_smooth_vertices(
            vertices,
            faces,
            weights,
            iterations=taubin_iters,
            lamb=float(getattr(args, "final_local_smooth_taubin_lambda", 0.42)),
            mu=float(getattr(args, "final_local_smooth_taubin_mu", -0.46)),
        )
        cap = displacement_limits.copy()
        cap[~np.isfinite(cap)] = np.inf
        vertices = clamp_relative_vertex_displacement(anchor_vertices, vertices, cap)
    if harmonize_iters > 0 and np.any(forehead_mask) and np.any(skull_mask):
        harmonize_fixed = pin_vertex_mask.copy()
        vertices = harmonize_region_seam_displacement(
            anchor_vertices,
            vertices,
            faces,
            active_mask=(forehead_mask | skull_mask) & (weights > 1e-6),
            fixed_mask=harmonize_fixed,
            displacement_ramp=weights,
            iterations=harmonize_iters,
            inner_ramp_threshold=float(
                getattr(args, "final_local_smooth_harmonize_inner_threshold", 0.82)
            ),
        )
        cap = displacement_limits.copy()
        cap[~np.isfinite(cap)] = np.inf
        vertices = clamp_relative_vertex_displacement(anchor_vertices, vertices, cap)

    vertices[pin_vertex_mask] = anchor_vertices[pin_vertex_mask]

    refine_iters = int(getattr(args, "mandatory_local_smooth_refine_taubin_iters", 0))
    if mandatory and refine_iters > 0 and np.any(weights > 1e-6):
        refine_anchor = vertices.copy()
        refine_weights = np.clip(weights * 1.08, 0.0, 1.0)
        vertices = taubin_smooth_vertices(
            vertices,
            faces,
            refine_weights,
            iterations=refine_iters,
            lamb=float(getattr(args, "final_local_smooth_taubin_lambda", 0.46)),
            mu=float(getattr(args, "final_local_smooth_taubin_mu", -0.50)),
        )
        cap = displacement_limits.copy()
        cap[~np.isfinite(cap)] = np.inf
        if mandatory:
            cap_scale = float(
                getattr(args, "mandatory_local_smooth_refine_displacement_scale", 0.32)
            )
            cap = np.minimum(cap, cap_scale * edge_scale)
        vertices = clamp_relative_vertex_displacement(refine_anchor, vertices, cap)
        vertices[pin_vertex_mask] = refine_anchor[pin_vertex_mask]

    if np.any(forehead_mask) and np.any(skull_mask):
        if weld_iters > 0:
            vertices = weld_region_pair_cross_edge_positions(
                vertices,
                faces,
                forehead_mask,
                skull_mask,
                iterations=weld_iters,
                blend=float(getattr(args, "final_local_smooth_post_weld_blend", 1.0)),
                move_mask=forehead_face_relax | skull_relax,
                pin_mask=pin_mask,
            )
        if tangent_iters > 0:
            vertices = align_region_pair_cross_edges_to_face_tangent(
                vertices,
                faces,
                face_mask,
                forehead_mask,
                skull_mask,
                iterations=tangent_iters,
                blend=float(getattr(args, "final_local_smooth_post_tangent_blend", 0.85)),
                move_nonface_mask=skull_relax,
                pin_mask=pin_mask,
            )
        tpl_blend = float(getattr(args, "final_local_smooth_post_template_blend", 0.0))
        if tpl_blend > 0.0 and template_vertices is not None:
            vertices = restore_region_pair_hairline_template_layout(
                template_vertices,
                vertices,
                faces,
                face_mask,
                forehead_mask,
                skull_mask,
                skull_anchor_mask=skull_relax,
                forehead_adjust_mask=forehead_face_relax,
                forehead_blend=tpl_blend,
            )
    return vertices


def apply_mandatory_final_seam_smoothness(
    vertices: np.ndarray,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    args: argparse.Namespace,
    *,
    template_vertices: np.ndarray | None = None,
    landmark_vertex_ids: np.ndarray | None = None,
    correspondence_exclude_vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    """robust_auto: always run localized seam smooth (cannot be turned off with 0 iters)."""
    if bool(getattr(args, "no_mandatory_final_smooth", False)):
        return np.asarray(vertices, dtype=np.float64)
    return apply_local_seam_band_smoothness(
        vertices,
        faces,
        seg28_pkl_path,
        args,
        template_vertices=template_vertices,
        landmark_vertex_ids=landmark_vertex_ids,
        correspondence_exclude_vertex_mask=correspondence_exclude_vertex_mask,
        force_mandatory=True,
    )


def apply_robust_auto_final_seam_repair(
    template_vertices: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    args: argparse.Namespace,
    *,
    landmark_vertex_ids: np.ndarray | None = None,
    correspondence_exclude_vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Last-pass repair before writing the mesh: forehead/skull weld+tangent+template,
    face/non-face cross-edge stretch, tangent projection, and local invert repair.
    """
    if seg28_pkl_path is None:
        return np.asarray(vertices, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask, _ = build_seg28_face_vertex_mask(
        vertices.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_SEG28_FACE_REGION_NAMES,
    )
    nonface_mask = ~np.asarray(face_mask, dtype=bool)
    forehead_mask, _ = build_seg28_face_vertex_mask(
        vertices.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_FOREHEAD_REGION_NAMES,
    )
    skull_mask, _ = build_seg28_face_vertex_mask(
        vertices.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_SKULL_REGION_NAMES,
    )
    ear_mask, _ = build_seg28_face_vertex_mask(
        vertices.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_EAR_REGION_NAMES,
    )
    (
        forehead_face_relax,
        skull_relax,
        _,
        _,
        _,
        _,
        _,
    ) = build_forehead_skull_seam_flex_masks(
        vertices,
        faces,
        seg28_pkl_path,
        band_width_scale=float(args.forehead_skull_face_flex_band_scale),
        inner_ramp_threshold=float(
            args.nonface_smooth_forehead_skull_harmonize_inner_threshold
        ),
    )
    pin_mask = np.asarray(ear_mask, dtype=bool)
    if landmark_vertex_ids is not None and landmark_vertex_ids.size:
        pin_mask = pin_mask.copy()
        pin_mask[landmark_vertex_ids] = True
    if correspondence_exclude_vertex_mask is not None:
        pin_mask |= np.asarray(correspondence_exclude_vertex_mask, dtype=bool)
    fnf_iters = int(getattr(args, "final_face_nonface_stretch_iterations", 0))
    if fnf_iters > 0 and np.any(face_mask) and np.any(nonface_mask):
        stretch_move = build_face_nonface_cross_seam_nonface_mask(
            faces, face_mask, nonface_mask
        )
        stretch_move &= ~pin_mask
        edge_scale = float(median_edge_length(vertices, faces))
        band_scale = float(getattr(args, "final_face_nonface_stretch_band_scale", 8.0))
        _, _, jsd = build_face_nonface_junction_normal_smooth_weights(
            vertices,
            faces,
            face_mask,
            nonface_mask,
            junction_width_scale=float(args.nonface_junction_width_scale),
        )
        if band_scale > 0.0 and edge_scale > 0.0:
            stretch_move &= np.isfinite(jsd)
            stretch_move &= jsd <= band_scale * edge_scale
        if np.any(stretch_move):
            vertices = relax_face_nonface_cross_edge_stretch(
                template_vertices,
                vertices,
                faces,
                face_mask,
                nonface_mask,
                max_stretch_ratio=float(args.nonface_smooth_face_nonface_max_stretch_ratio),
                min_stretch_ratio=float(args.nonface_smooth_face_nonface_min_stretch_ratio),
                iterations=fnf_iters,
                blend=float(getattr(args, "final_face_nonface_stretch_blend", 0.92)),
                move_mask=stretch_move,
            )
    if np.any(forehead_mask) and np.any(skull_mask):
        vertices = apply_forehead_skull_seam_finalize(
            template_vertices,
            vertices,
            faces,
            face_mask,
            forehead_mask,
            skull_mask,
            forehead_face_relax,
            skull_relax,
            weld_iterations=int(getattr(args, "final_forehead_skull_weld_iterations", 24)),
            weld_blend=float(getattr(args, "final_forehead_skull_weld_blend", 1.0)),
            tangent_iterations=int(getattr(args, "final_forehead_skull_tangent_iterations", 16)),
            tangent_blend=float(getattr(args, "final_forehead_skull_tangent_blend", 1.0)),
            template_restore_blend=float(
                getattr(args, "final_forehead_skull_template_restore_blend", 0.88)
            ),
            pin_mask=pin_mask,
        )
    proj_iters = int(getattr(args, "final_face_nonface_projection_iterations", 0))
    if proj_iters > 0 and np.any(face_mask) and np.any(nonface_mask):
        vertices = project_nonface_seam_onto_face_tangent_planes(
            vertices,
            faces,
            face_mask,
            nonface_mask,
            iterations=proj_iters,
            blend=float(getattr(args, "final_face_nonface_projection_blend", 0.85)),
            max_normal_offset_scale=float(
                getattr(args, "final_face_nonface_projection_max_offset_scale", 0.35)
            ),
            free_face_mask=forehead_face_relax,
        )
    invert_iters = int(getattr(args, "final_seam_invert_repair_iterations", 0))
    if invert_iters > 0:
        invert_protect = np.ones(vertices.shape[0], dtype=np.float64)
        invert_protect[~face_mask] = 0.0
        invert_protect[pin_mask] = 1.0
        vertices = repair_inverted_faces(
            template_vertices,
            vertices,
            faces,
            iterations=invert_iters,
            blend_step=float(getattr(args, "final_seam_invert_repair_blend", 0.7)),
            max_restore_displacement_scale=float(
                getattr(args, "final_seam_invert_repair_max_restore_scale", 2.0)
            ),
            protect_vertex_band=invert_protect,
            protect_strength=float(getattr(args, "final_seam_invert_repair_protect_strength", 0.92)),
        )
        if np.any(forehead_mask) and np.any(skull_mask):
            vertices = weld_region_pair_cross_edge_positions(
                vertices,
                faces,
                forehead_mask,
                skull_mask,
                iterations=max(8, int(getattr(args, "final_forehead_skull_weld_iterations", 24)) // 2),
                blend=float(getattr(args, "final_forehead_skull_weld_blend", 1.0)),
                move_mask=forehead_face_relax | skull_relax,
                pin_mask=pin_mask,
            )
    return vertices


def build_forehead_skull_seam_flex_masks(
    vertices: np.ndarray,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    *,
    band_width_scale: float = 8.0,
    inner_ramp_threshold: float = 0.88,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Forehead/skull seam flex masks and ramps for ICP prior relaxation.

    Returns forehead_relax, skull_relax, forehead_flex_band, skull_flex_band,
    band_width, anchor_ramp, deform_ramp.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = vertices.shape[0]
    forehead_mask, _ = build_seg28_face_vertex_mask(
        n_vertices,
        faces,
        seg28_pkl_path,
        region_names=HACK_FOREHEAD_REGION_NAMES,
    )
    skull_mask, _ = build_seg28_face_vertex_mask(
        n_vertices,
        faces,
        seg28_pkl_path,
        region_names=HACK_SKULL_REGION_NAMES,
    )
    union_mask = forehead_mask | skull_mask
    deform_ramp, anchor_ramp, band_width = build_region_junction_motion_ramp(
        vertices,
        faces,
        forehead_mask,
        skull_mask,
        ramp_width_scale=float(band_width_scale),
        bidirectional=True,
        shared_geodesic_mask=union_mask,
    )
    inner_ramp_threshold = float(np.clip(inner_ramp_threshold, 0.5, 1.0))
    forehead_flex_band = np.zeros(n_vertices, dtype=np.float64)
    skull_flex_band = np.zeros(n_vertices, dtype=np.float64)
    forehead_shell = forehead_mask & np.isfinite(anchor_ramp)
    skull_shell = skull_mask & np.isfinite(deform_ramp)
    forehead_flex_band[forehead_shell] = 1.0 - anchor_ramp[forehead_shell]
    skull_flex_band[skull_shell] = 1.0 - deform_ramp[skull_shell]
    forehead_flex_band = np.clip(forehead_flex_band, 0.0, 1.0)
    skull_flex_band = np.clip(skull_flex_band, 0.0, 1.0)
    forehead_relax_mask = forehead_mask & (anchor_ramp < inner_ramp_threshold)
    skull_relax_mask = skull_mask & (deform_ramp < inner_ramp_threshold)
    return (
        forehead_relax_mask,
        skull_relax_mask,
        forehead_flex_band,
        skull_flex_band,
        float(band_width),
        anchor_ramp,
        deform_ramp,
    )


def apply_forehead_skull_seam_registration_flex(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    position_prior_vertex_weights: np.ndarray,
    data_vertex_weights: np.ndarray,
    shape_prior_vertex_weights: np.ndarray,
    vertex_mass_vertex_weights: np.ndarray,
    smoothness_edge_weights: np.ndarray,
    edge_vector_edge_weights: np.ndarray,
    affine_prior_vertex_weights: np.ndarray,
    *,
    enable_face_flex: bool = True,
    enable_skull_flex: bool = True,
    enable_vertex_mass_boost: bool = True,
    band_width_scale: float = 8.0,
    seam_vertex_mass_multiplier: float = 4.0,
    face_position_weight: float = 0.01,
    face_data_weight_boost: float = 0.88,
    face_shape_weight: float = 0.38,
    face_smoothness_multiplier: float = 0.52,
    face_edge_vector_multiplier: float = 0.42,
    skull_position_weight: float = 0.08,
    skull_affine_prior_scale: float = 0.22,
    skull_data_weight_boost: float = 0.35,
    skull_shape_weight: float = 0.42,
    skull_smoothness_multiplier: float = 0.48,
    skull_edge_vector_multiplier: float = 0.38,
    protect_ear_vertices: bool = True,
) -> dict[str, object]:
    """
    Relax ICP priors on both sides of the forehead/skull seam.

    The skull side is critical: non-face vertices default to a strong raw-mesh
    position prior (~2.5) and high affine prior (~12), which pins the skull and
    creates a step against the forehead.
    """
    if not enable_face_flex and not enable_skull_flex and not enable_vertex_mass_boost:
        return {"enabled": False}

    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    (
        forehead_relax_mask,
        skull_relax_mask,
        forehead_flex_band,
        skull_flex_band,
        band_width,
        _,
        _,
    ) = build_forehead_skull_seam_flex_masks(
        source_vertices,
        source_faces,
        seg28_pkl_path,
        band_width_scale=band_width_scale,
    )

    if protect_ear_vertices:
        ear_mask, _ = build_seg28_face_vertex_mask(
            source_vertices.shape[0],
            source_faces,
            seg28_pkl_path,
            region_names=HACK_EAR_REGION_NAMES,
        )
        forehead_relax_mask &= ~ear_mask
        skull_relax_mask &= ~ear_mask
        forehead_flex_band[ear_mask] = 0.0
        skull_flex_band[ear_mask] = 0.0

    position_prior_vertex_weights = np.asarray(position_prior_vertex_weights, dtype=np.float64)
    data_vertex_weights = np.asarray(data_vertex_weights, dtype=np.float64)
    shape_prior_vertex_weights = np.asarray(shape_prior_vertex_weights, dtype=np.float64)
    vertex_mass_vertex_weights = np.asarray(vertex_mass_vertex_weights, dtype=np.float64)
    affine_prior_vertex_weights = np.asarray(affine_prior_vertex_weights, dtype=np.float64)
    edges = extract_edges(source_faces)
    combined_flex_band = np.maximum(forehead_flex_band, skull_flex_band)
    seam_mass_mult = float(max(seam_vertex_mass_multiplier, 1.0))

    if enable_face_flex and np.any(forehead_relax_mask):
        face_pos = float(max(face_position_weight, 0.0))
        position_prior_vertex_weights[forehead_relax_mask] = (
            (1.0 - forehead_flex_band[forehead_relax_mask])
            * position_prior_vertex_weights[forehead_relax_mask]
            + forehead_flex_band[forehead_relax_mask] * face_pos
        )
        data_vertex_weights[:] = np.maximum(
            data_vertex_weights,
            float(face_data_weight_boost) * forehead_flex_band,
        )
        shape_prior_vertex_weights[:] *= 1.0 - (
            1.0 - float(np.clip(face_shape_weight, 0.0, 1.0))
        ) * forehead_flex_band

    if enable_skull_flex and np.any(skull_relax_mask):
        skull_pos = float(max(skull_position_weight, 0.0))
        position_prior_vertex_weights[skull_relax_mask] = (
            (1.0 - skull_flex_band[skull_relax_mask])
            * position_prior_vertex_weights[skull_relax_mask]
            + skull_flex_band[skull_relax_mask] * skull_pos
        )
        skull_affine_scale = float(np.clip(skull_affine_prior_scale, 0.0, 1.0))
        affine_prior_vertex_weights[skull_relax_mask] *= (
            (1.0 - skull_flex_band[skull_relax_mask])
            + skull_flex_band[skull_relax_mask] * skull_affine_scale
        )
        data_vertex_weights[:] = np.maximum(
            data_vertex_weights,
            float(skull_data_weight_boost) * skull_flex_band,
        )
        shape_prior_vertex_weights[:] *= 1.0 - (
            1.0 - float(np.clip(skull_shape_weight, 0.0, 1.0))
        ) * skull_flex_band

    if np.any(combined_flex_band > 1e-6):
        smoothness_edge_weights *= _edge_relax_weights_from_vertex_band(
            source_faces,
            combined_flex_band,
            float(min(face_smoothness_multiplier, skull_smoothness_multiplier)),
            edges=edges,
        )
        edge_vector_edge_weights *= _edge_relax_weights_from_vertex_band(
            source_faces,
            combined_flex_band,
            float(min(face_edge_vector_multiplier, skull_edge_vector_multiplier)),
            edges=edges,
        )

    if enable_vertex_mass_boost and seam_mass_mult > 1.0 and np.any(combined_flex_band > 1e-6):
        vertex_mass_vertex_weights *= 1.0 + (seam_mass_mult - 1.0) * combined_flex_band

    return {
        "enabled": True,
        "num_forehead_relax_vertices": int(np.count_nonzero(forehead_relax_mask)),
        "num_skull_relax_vertices": int(np.count_nonzero(skull_relax_mask)),
        "seam_band_width": float(band_width),
        "face_position_weight": float(face_position_weight),
        "skull_position_weight": float(skull_position_weight),
        "skull_affine_prior_scale": float(skull_affine_prior_scale),
        "seam_vertex_mass_multiplier": float(seam_mass_mult),
    }






def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _smootherstep(values: np.ndarray) -> np.ndarray:
    """Perlin smootherstep — zero 1st/2nd derivatives at both ends."""
    values = np.clip(values, 0.0, 1.0)
    return values * values * values * (values * (values * 6.0 - 15.0) + 10.0)


def _ease_region_ramp(values: np.ndarray, mode: str = "softerstep") -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    if mode == "gentle":
        return values ** 2.5
    if mode == "smoothstep":
        return _smoothstep(values)
    return _smootherstep(values)


def _boundary_transition_band(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    width: float,
) -> np.ndarray:
    seed = face_boundary_seed(faces, face_mask)
    dist = geodesic_distance_from_mask(
        vertices,
        faces,
        seed,
        max_distance=width,
    )
    band = np.zeros(vertices.shape[0], dtype=np.float64)
    ok = np.isfinite(dist)
    band[ok] = 1.0 - dist[ok] / max(float(width), 1e-12)
    return _smoothstep(band)


def _smooth_weight_field(
    values: np.ndarray,
    faces: np.ndarray,
    iterations: int,
    step: float,
    fixed_mask: np.ndarray,
) -> np.ndarray:
    if int(iterations) <= 0:
        return np.clip(values, 0.0, 1.0)
    smoothed = smooth_vertex_scalar_field(
        values,
        faces,
        iterations=int(iterations),
        step=float(step),
        fixed_mask=fixed_mask,
        fixed_values=values,
    )
    return np.clip(smoothed, 0.0, 1.0)


def build_face_nonface_junction_normal_smooth_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    nonface_mask: np.ndarray,
    *,
    junction_width_scale: float = 6.0,
    exclude_vertex_mask: Optional[np.ndarray] = None,
    landmark_vertex_ids: Optional[np.ndarray] = None,
    weight_smooth_iterations: int = 3,
    weight_smooth_step: float = 0.5,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Weights for junction normal smoothing on the non-face side of the seam only.

    Face vertices always receive weight 0. Distance is measured from the
    face/non-face seam along non-face vertices only.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    nonface_mask = np.asarray(nonface_mask, dtype=bool).reshape(-1)
    if face_mask.shape[0] != nonface_mask.shape[0]:
        raise ValueError("face_mask and nonface_mask must have the same length")

    junction_width = max(float(junction_width_scale) * float(median_edge_length(vertices, faces)), 1e-12)
    seam_seed = face_boundary_seed(faces, face_mask)
    dist = geodesic_distance_from_mask(
        vertices,
        faces,
        seam_seed,
        max_distance=junction_width,
        allowed_mask=nonface_mask,
    )
    weights = np.zeros(face_mask.shape[0], dtype=np.float64)
    shell = nonface_mask & np.isfinite(dist)
    weights[shell] = _smoothstep(1.0 - dist[shell] / junction_width)

    pin_mask = face_mask.copy()
    if exclude_vertex_mask is not None:
        exclude_vertex_mask = np.asarray(exclude_vertex_mask, dtype=bool).reshape(-1)
        weights[exclude_vertex_mask] = 0.0
        pin_mask |= exclude_vertex_mask
    if int(weight_smooth_iterations) > 0 and np.any(weights > 1e-6):
        weights = _smooth_weight_field(
            weights,
            faces,
            int(weight_smooth_iterations),
            float(weight_smooth_step),
            pin_mask,
        )
    weights[face_mask] = 0.0
    if exclude_vertex_mask is not None:
        weights[exclude_vertex_mask] = 0.0
    if landmark_vertex_ids is not None:
        landmark_vertex_ids = np.asarray(landmark_vertex_ids, dtype=np.int64).reshape(-1)
        landmark_vertex_ids = landmark_vertex_ids[(landmark_vertex_ids >= 0) & (landmark_vertex_ids < weights.shape[0])]
        if landmark_vertex_ids.size:
            weights[landmark_vertex_ids] = 0.0
    return np.clip(weights, 0.0, 1.0), float(junction_width), dist




def measure_cross_seam_nonface_displacement(
    vertices: np.ndarray,
    reference_vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    nonface_mask: np.ndarray,
) -> dict[str, float]:
    """Displacement magnitudes on non-face endpoints of face/non-face edges."""
    vertices = np.asarray(vertices, dtype=np.float64)
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    nonface_mask = np.asarray(nonface_mask, dtype=bool).reshape(-1)
    edges = extract_edges(faces)
    if edges.size == 0:
        return {"count": 0.0, "max": 0.0, "p90": 0.0, "mean": 0.0}
    cross = face_mask[edges[:, 0]] != face_mask[edges[:, 1]]
    if not np.any(cross):
        return {"count": 0.0, "max": 0.0, "p90": 0.0, "mean": 0.0}
    cross_edges = edges[cross]
    nf_ids = np.where(nonface_mask[cross_edges[:, 0]], cross_edges[:, 0], cross_edges[:, 1])
    delta = np.linalg.norm(vertices[nf_ids] - reference_vertices[nf_ids], axis=1)
    return {
        "count": float(delta.size),
        "max": float(np.max(delta)),
        "p90": float(np.percentile(delta, 90)),
        "mean": float(np.mean(delta)),
    }


def build_face_nonface_cross_seam_nonface_mask(
    faces: np.ndarray,
    face_mask: np.ndarray,
    nonface_mask: np.ndarray,
) -> np.ndarray:
    """Non-face vertices incident to at least one face/non-face edge."""
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    nonface_mask = np.asarray(nonface_mask, dtype=bool).reshape(-1)
    mask = np.zeros(face_mask.shape[0], dtype=bool)
    edges = extract_edges(faces)
    if edges.size == 0:
        return mask
    cross = face_mask[edges[:, 0]] != face_mask[edges[:, 1]]
    if not np.any(cross):
        return mask
    cross_edges = edges[cross]
    mask[cross_edges.reshape(-1)] = True
    return mask & nonface_mask


def build_face_nonface_junction_displacement_limits(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    nonface_mask: np.ndarray,
    *,
    junction_width: float,
    seam_displacement_scale: float = 0.12,
    outer_displacement_scale: float = 0.65,
    exclude_vertex_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Per-vertex cumulative displacement limits for junction normal smoothing.

    Limits are tightest on the non-face seam ring and grow toward the outer
    shell so normal fairing cannot introduce large position jumps at the boundary.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    nonface_mask = np.asarray(nonface_mask, dtype=bool).reshape(-1)
    edge_scale = float(median_edge_length(vertices, faces))
    junction_width = max(float(junction_width), edge_scale * 1e-3)
    seam_seed = face_boundary_seed(faces, face_mask)
    dist = geodesic_distance_from_mask(
        vertices,
        faces,
        seam_seed,
        max_distance=junction_width,
        allowed_mask=nonface_mask,
    )
    limits = np.full(face_mask.shape[0], np.inf, dtype=np.float64)
    shell = nonface_mask & np.isfinite(dist)
    if np.any(shell):
        t = _smoothstep(np.clip(dist[shell] / junction_width, 0.0, 1.0))
        seam_limit = float(seam_displacement_scale) * edge_scale
        outer_limit = float(outer_displacement_scale) * edge_scale
        limits[shell] = seam_limit + (outer_limit - seam_limit) * t
    limits[face_mask] = 0.0
    if exclude_vertex_mask is not None:
        exclude_vertex_mask = np.asarray(exclude_vertex_mask, dtype=bool).reshape(-1)
        limits[exclude_vertex_mask] = 0.0
    return limits


def build_nonface_interior_smooth_weights(
    nonface_mask: np.ndarray,
    face_mask: np.ndarray,
    junction_seam_distance: np.ndarray,
    junction_width: float,
    *,
    deep_weight: float = 1.0,
    seam_floor_weight: float = 0.30,
    forehead_skull_cap_mask: Optional[np.ndarray] = None,
    forehead_skull_cap_weight: float = 0.42,
    exclude_vertex_mask: Optional[np.ndarray] = None,
    min_seam_distance_scale: float = 0.0,
    edge_scale: float = 0.0,
) -> np.ndarray:
    """
    Weights for a full non-face Taubin pass: strongest in the skull/neck interior,
    attenuated toward the face/non-face junction and optional forehead/skull hairline.
    """
    nonface_mask = np.asarray(nonface_mask, dtype=bool).reshape(-1)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    junction_seam_distance = np.asarray(junction_seam_distance, dtype=np.float64).reshape(-1)
    junction_width = float(np.asarray(junction_width, dtype=np.float64).reshape(()))
    deep = float(np.clip(deep_weight, 0.0, 1.0))
    floor = float(np.clip(seam_floor_weight, 0.0, deep))
    weights = np.zeros(nonface_mask.shape[0], dtype=np.float64)
    weights[nonface_mask] = deep
    shell = nonface_mask & np.isfinite(junction_seam_distance)
    junction_width = max(junction_width, 1e-12)
    if np.any(shell):
        ramp = _smoothstep(np.clip(junction_seam_distance[shell] / junction_width, 0.0, 1.0))
        weights[shell] = floor + (deep - floor) * ramp
        min_scale = float(min_seam_distance_scale)
        if min_scale > 0.0 and float(edge_scale) > 0.0:
            shell_ids = np.where(shell)[0]
            near = junction_seam_distance[shell_ids] <= min_scale * float(edge_scale)
            weights[shell_ids[near]] = 0.0
    if forehead_skull_cap_mask is not None:
        cap_mask = np.asarray(forehead_skull_cap_mask, dtype=bool).reshape(-1) & nonface_mask
        if float(forehead_skull_cap_weight) <= 0.0 and np.any(cap_mask):
            weights[cap_mask] = 0.0
        elif float(forehead_skull_cap_weight) < 1.0:
            cap = float(np.clip(forehead_skull_cap_weight, 0.0, 1.0))
            if np.any(cap_mask):
                weights[cap_mask] = np.minimum(weights[cap_mask], cap)
    weights[face_mask] = 0.0
    if exclude_vertex_mask is not None:
        weights[np.asarray(exclude_vertex_mask, dtype=bool).reshape(-1)] = 0.0
    return np.clip(weights, 0.0, 1.0)


def reapply_post_nonface_smooth_vertex_pins(
    vertices: np.ndarray,
    *,
    face_reference_vertices: np.ndarray,
    face_pin_mask: np.ndarray,
    ear_vertex_mask: np.ndarray,
    ear_vertices_frozen: np.ndarray,
    region_seam_pin_mask: np.ndarray,
    region_seam_vertices_frozen: np.ndarray,
) -> np.ndarray:
    """Restore face/ear/region-seam pins after a non-face-only smooth pass."""
    vertices = np.asarray(vertices, dtype=np.float64)
    face_reference_vertices = np.asarray(face_reference_vertices, dtype=np.float64)
    face_pin_mask = np.asarray(face_pin_mask, dtype=bool).reshape(-1)
    if np.any(face_pin_mask):
        vertices[face_pin_mask] = face_reference_vertices[face_pin_mask]
    ear_vertex_mask = np.asarray(ear_vertex_mask, dtype=bool).reshape(-1)
    if np.any(ear_vertex_mask):
        vertices[ear_vertex_mask] = ear_vertices_frozen
    region_seam_pin_mask = np.asarray(region_seam_pin_mask, dtype=bool).reshape(-1)
    if np.any(region_seam_pin_mask):
        vertices[region_seam_pin_mask] = region_seam_vertices_frozen
    return vertices


def build_face_interior_smooth_exclusions(
    vertices: np.ndarray,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    face_mask: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Vertices that must not move during face-interior smoothing."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    exclude = np.zeros(face_mask.shape[0], dtype=bool)
    if seg28_pkl_path is None:
        return exclude

    forehead_mask, _ = build_seg28_face_vertex_mask(
        face_mask.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_FOREHEAD_REGION_NAMES,
    )
    skull_mask, _ = build_seg28_face_vertex_mask(
        face_mask.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_SKULL_REGION_NAMES,
    )
    if bool(getattr(args, "restore_face_interior_exclude_forehead", True)):
        exclude |= np.asarray(forehead_mask, dtype=bool)
    else:
        forehead_relax, skull_relax, _, _, _, _, _ = build_forehead_skull_seam_flex_masks(
            vertices,
            faces,
            seg28_pkl_path,
            band_width_scale=float(args.forehead_skull_face_flex_band_scale),
            inner_ramp_threshold=float(
                args.nonface_smooth_forehead_skull_harmonize_inner_threshold
            ),
        )
        exclude |= np.asarray(forehead_relax | skull_relax, dtype=bool)

    if np.any(forehead_mask) and np.any(skull_mask):
        seam_influence = build_forehead_skull_seam_influence_mask(
            vertices,
            faces,
            forehead_mask,
            skull_mask,
            band_width_scale=float(args.restore_face_interior_seam_influence_band_scale),
        )
        exclude |= seam_influence
        cheek_band = float(args.restore_face_interior_cheek_exclusion_band_scale)
        if cheek_band > 0.0:
            cross_seed = region_cross_boundary_seed(faces, forehead_mask, skull_mask)
            cheek_near = face_mask & np.isfinite(
                geodesic_distance_from_mask(
                    vertices,
                    faces,
                    cross_seed,
                    max_distance=cheek_band * float(median_edge_length(vertices, faces)),
                    allowed_mask=face_mask,
                )
            )
            exclude |= cheek_near

    ear_mask, _ = build_seg28_face_vertex_mask(
        face_mask.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_EAR_REGION_NAMES,
    )
    exclude |= np.asarray(ear_mask, dtype=bool)
    return exclude


def build_face_interior_smooth_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
    *,
    boundary_clear_scale: float = 2.5,
    full_weight_scale: float = 5.0,
    exclude_vertex_mask: np.ndarray | None = None,
    max_displacement_scale: float = 0.32,
    mandatory_min_peak_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Smooth weights on the face interior: zero near the face/non-face boundary,
    ramping to full weight in the deep interior (cheeks, nose bridge, etc.).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    edge_scale = float(median_edge_length(vertices, faces))
    weights = np.zeros(face_mask.shape[0], dtype=np.float64)
    limits = np.full(face_mask.shape[0], np.inf, dtype=np.float64)
    if edge_scale <= 0.0 or not np.any(face_mask):
        return weights, limits, edge_scale

    clear_dist = max(float(boundary_clear_scale), 0.0) * edge_scale
    full_dist = max(float(full_weight_scale) * edge_scale, clear_dist + edge_scale * 0.5)
    boundary_seed = face_boundary_seed(faces, face_mask)
    dist = geodesic_distance_from_mask(
        vertices,
        faces,
        boundary_seed,
        max_distance=full_dist + edge_scale,
        allowed_mask=face_mask,
    )
    interior = face_mask & np.isfinite(dist)
    if np.any(interior):
        ramp_span = max(full_dist - clear_dist, edge_scale * 0.25)
        t = np.clip((dist[interior] - clear_dist) / ramp_span, 0.0, 1.0)
        peak = float(np.clip(mandatory_min_peak_weight, 0.0, 1.0))
        weights[interior] = peak * _smoothstep(t)
    if exclude_vertex_mask is not None:
        weights[np.asarray(exclude_vertex_mask, dtype=bool).reshape(-1)] = 0.0

    active = weights > 1e-6
    if np.any(active):
        cap = float(max_displacement_scale) * edge_scale
        limits[active] = cap * weights[active]
    return weights, limits, edge_scale


def resolve_mandatory_face_interior_smooth_iters(args: argparse.Namespace) -> tuple[int, int]:
    taubin = int(getattr(args, "face_interior_smooth_taubin_iterations", 0))
    junction = int(getattr(args, "face_interior_smooth_junction_iterations", 0))
    if not bool(getattr(args, "no_mandatory_face_interior_smooth", False)):
        taubin = max(taubin, int(getattr(args, "mandatory_face_interior_smooth_min_taubin_iters", 28)))
        junction = max(
            junction, int(getattr(args, "mandatory_face_interior_smooth_min_junction_iters", 12))
        )
    return junction, taubin


def apply_mandatory_face_interior_smoothness(
    vertices: np.ndarray,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    args: argparse.Namespace,
    *,
    landmark_vertex_ids: np.ndarray | None = None,
    correspondence_exclude_vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Mandatory Taubin + junction smooth on the deep face interior (robust_auto)."""
    if seg28_pkl_path is None or bool(getattr(args, "no_mandatory_face_interior_smooth", False)):
        return np.asarray(vertices, dtype=np.float64)

    junction_iters, taubin_iters = resolve_mandatory_face_interior_smooth_iters(args)
    refine_iters = int(getattr(args, "face_interior_smooth_refine_taubin_iterations", 0))
    if junction_iters <= 0 and taubin_iters <= 0 and refine_iters <= 0:
        return np.asarray(vertices, dtype=np.float64)

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_mask, _ = build_seg28_face_vertex_mask(
        vertices.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_SEG28_FACE_REGION_NAMES,
    )
    exclude = build_face_interior_smooth_exclusions(
        vertices, faces, seg28_pkl_path, face_mask, args
    )
    if landmark_vertex_ids is not None and landmark_vertex_ids.size:
        exclude = exclude.copy()
        exclude[landmark_vertex_ids] = True
    if correspondence_exclude_vertex_mask is not None:
        exclude |= np.asarray(correspondence_exclude_vertex_mask, dtype=bool)

    weights, displacement_limits, edge_scale = build_face_interior_smooth_weights(
        vertices,
        faces,
        face_mask,
        boundary_clear_scale=float(args.face_interior_smooth_boundary_clear_scale),
        full_weight_scale=float(args.face_interior_smooth_full_weight_scale),
        exclude_vertex_mask=exclude,
        max_displacement_scale=float(args.face_interior_smooth_max_displacement_scale),
        mandatory_min_peak_weight=float(args.mandatory_face_interior_smooth_min_peak_weight),
    )
    if not np.any(weights > 1e-6):
        return vertices

    anchor_vertices = vertices.copy()
    pin_vertex_mask = (weights <= 1e-6) | exclude
    nonface_mask = ~np.asarray(face_mask, dtype=bool)

    if junction_iters > 0:
        face_junction_weights = weights.copy()
        face_junction_weights[nonface_mask] = 0.0
        vertices = optimize_face_nonface_junction_normal_smoothness(
            vertices,
            faces,
            face_mask,
            face_junction_weights,
            iterations=junction_iters,
            lamb=float(args.face_interior_smooth_junction_lambda),
            mu=float(args.face_interior_smooth_junction_mu),
            tangential_only=True,
            normal_alignment_step=float(args.face_interior_smooth_normal_alignment_step),
            max_displacement=displacement_limits,
            max_iteration_displacement_scale=float(
                args.face_interior_smooth_max_iteration_step_scale
            ),
            pin_vertex_mask=pin_vertex_mask,
        )
    if taubin_iters > 0:
        vertices = taubin_smooth_vertices(
            vertices,
            faces,
            weights,
            iterations=taubin_iters,
            lamb=float(args.face_interior_smooth_taubin_lambda),
            mu=float(args.face_interior_smooth_taubin_mu),
        )
        cap = displacement_limits.copy()
        cap[~np.isfinite(cap)] = np.inf
        vertices = clamp_relative_vertex_displacement(anchor_vertices, vertices, cap)
    if refine_iters > 0:
        refine_anchor = vertices.copy()
        refine_weights = np.clip(weights * 1.05, 0.0, 1.0)
        vertices = taubin_smooth_vertices(
            vertices,
            faces,
            refine_weights,
            iterations=refine_iters,
            lamb=float(args.face_interior_smooth_taubin_lambda),
            mu=float(args.face_interior_smooth_taubin_mu),
        )
        cap = displacement_limits.copy()
        cap[~np.isfinite(cap)] = np.inf
        refine_cap = float(args.face_interior_smooth_refine_max_displacement_scale) * edge_scale
        cap = np.minimum(cap, refine_cap)
        vertices = clamp_relative_vertex_displacement(refine_anchor, vertices, cap)

    vertices[pin_vertex_mask] = anchor_vertices[pin_vertex_mask]
    return vertices

def _region_seam_stats(
    template_vertices: np.ndarray,
    vertices: np.ndarray,
    reference_vertices: np.ndarray,
    faces: np.ndarray,
    region_a_mask: np.ndarray,
    region_b_mask: np.ndarray,
    *,
    track_mask: np.ndarray | None = None,
    max_stretch_ratio: float = 1.5,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Cross displacement, seam jump, and edge stretch for a region pair."""
    track = track_mask if track_mask is not None else (region_a_mask | region_b_mask)
    edges = extract_edges(faces)
    cross = region_a_mask[edges[:, 0]] & region_b_mask[edges[:, 1]]
    cross |= region_b_mask[edges[:, 0]] & region_a_mask[edges[:, 1]]
    empty = {"count": 0.0, "max": 0.0, "p90": 0.0, "mean": 0.0}
    stretch_empty = {
        "count": 0.0,
        "median": float("nan"),
        "p90": float("nan"),
        "max": float("nan"),
        "fraction_above_max": float("nan"),
    }
    if edges.size == 0 or not np.any(cross):
        return empty, empty, stretch_empty
    cross_edges = edges[cross]
    delta = vertices - reference_vertices
    jump = np.linalg.norm(delta[cross_edges[:, 0]] - delta[cross_edges[:, 1]], axis=1)
    jump_stats = {
        "count": float(jump.size),
        "max": float(np.max(jump)),
        "p90": float(np.percentile(jump, 90)),
        "mean": float(np.mean(jump)),
    }
    track_ids = np.where(track[cross_edges[:, 0]], cross_edges[:, 0], cross_edges[:, 1])
    cross_delta = np.linalg.norm(vertices[track_ids] - reference_vertices[track_ids], axis=1)
    cross_stats = {
        "count": float(cross_delta.size),
        "max": float(np.max(cross_delta)),
        "p90": float(np.percentile(cross_delta, 90)),
        "mean": float(np.mean(cross_delta)),
    }
    ref_len = np.linalg.norm(
        template_vertices[cross_edges[:, 1]] - template_vertices[cross_edges[:, 0]], axis=1
    )
    def_len = np.linalg.norm(vertices[cross_edges[:, 1]] - vertices[cross_edges[:, 0]], axis=1)
    ratio = def_len / np.maximum(ref_len, 1e-12)
    stretch_stats = {
        "count": float(ratio.size),
        "median": float(np.median(ratio)),
        "p90": float(np.percentile(ratio, 90)),
        "max": float(np.max(ratio)),
        "fraction_above_max": float(np.mean(ratio > float(max_stretch_ratio))),
    }
    return cross_stats, jump_stats, stretch_stats



def _edge_relax_weights_from_vertex_band(
    faces: np.ndarray,
    vertex_band: np.ndarray,
    multiplier: float,
    edges: Optional[np.ndarray] = None,
) -> np.ndarray:
    edges = extract_edges(faces) if edges is None else np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return np.empty(0, dtype=np.float64)
    edge_band = 0.5 * (vertex_band[edges[:, 0]] + vertex_band[edges[:, 1]])
    multiplier = float(np.clip(multiplier, 0.0, 1.0))
    return 1.0 - (1.0 - multiplier) * np.clip(edge_band, 0.0, 1.0)



def _forehead_skull_bilap_pass(
    vertices: np.ndarray,
    faces: np.ndarray,
    args: argparse.Namespace,
    *,
    bilap_weights: np.ndarray,
    bump_mask: np.ndarray,
    fixed: np.ndarray,
    fs_allowed: np.ndarray,
    edge_scale: float,
    reference_vertices: np.ndarray,
    fixed_vertices: np.ndarray,
    max_disp: np.ndarray,
    bump_anchor_scale: float = 1.0,
) -> np.ndarray:
    n_vertices = vertices.shape[0]
    anchor_w = np.zeros(n_vertices, dtype=np.float64)
    bump_anchor = float(
        getattr(args, "forehead_skull_bump_bilaplacian_anchor_weight_bump", 0.15)
    )
    support_anchor = float(
        getattr(args, "forehead_skull_bump_bilaplacian_anchor_weight_support", 12.0)
    )
    anchor_w[bump_mask] = bump_anchor * float(bump_anchor_scale)
    anchor_w[(bilap_weights > 1e-6) & ~bump_mask] = support_anchor
    return bilaplacian_smooth_vertices(
        vertices,
        faces,
        bilap_weights,
        fixed_mask=fixed,
        fixed_vertices=fixed_vertices,
        allowed_vertex_mask=fs_allowed,
        bilaplacian_weight=float(getattr(args, "forehead_skull_bump_bilaplacian_weight", 1.0)),
        anchor_weight=1.0,
        anchor_vertex_weights=anchor_w,
        damping_scale=float(getattr(args, "forehead_skull_bump_bilaplacian_damping_scale", 1e-4)),
        edge_scale=edge_scale,
        max_displacement=max_disp,
        reference_vertices=reference_vertices,
    )


def apply_forehead_skull_seam_bump_flatten(
    vertices: np.ndarray,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    args: argparse.Namespace,
    *,
    landmark_vertex_ids: np.ndarray | None = None,
    correspondence_exclude_vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Detect hairline bumps, Bi-Laplacian fairing on those vertices, then seam weld.
    """
    if bool(getattr(args, "no_forehead_skull_bump_flatten", False)) or seg28_pkl_path is None:
        return np.asarray(vertices, dtype=np.float64)

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n_vertices = vertices.shape[0]
    forehead_mask, _ = build_seg28_face_vertex_mask(
        n_vertices, faces, seg28_pkl_path, region_names=HACK_FOREHEAD_REGION_NAMES
    )
    skull_mask, _ = build_seg28_face_vertex_mask(
        n_vertices, faces, seg28_pkl_path, region_names=HACK_SKULL_REGION_NAMES
    )
    ear_mask, _ = build_seg28_face_vertex_mask(
        n_vertices, faces, seg28_pkl_path, region_names=HACK_EAR_REGION_NAMES
    )
    edge_scale = float(median_edge_length(vertices, faces))
    fs_allowed = np.asarray(forehead_mask | skull_mask, dtype=bool)
    detect_scale = float(getattr(args, "forehead_skull_bump_detect_band_scale", 6.0))
    bump_mask = np.zeros(n_vertices, dtype=bool)
    bump_stats: dict[str, float] = {}
    weights, bump_mask, candidate_mask, bump_stats = build_forehead_skull_bump_smooth_weights(
        vertices,
        faces,
        forehead_mask,
        skull_mask,
        detect_band_scale=detect_scale,
        detect_threshold_scale=float(
            getattr(args, "forehead_skull_bump_detect_threshold_scale", 0.032)
        ),
        detect_percentile=float(
            getattr(args, "forehead_skull_bump_detect_percentile", 68.0)
        ),
        peak_weight=float(getattr(args, "forehead_skull_bump_bilaplacian_peak_weight", 0.92)),
        support_weight_scale=float(
            getattr(args, "forehead_skull_bump_support_weight_scale", 0.42)
        ),
        cross_edge_score_boost_scale=float(
            getattr(args, "forehead_skull_bump_cross_edge_score_boost_scale", 0.06)
        ),
        min_bump_vertices=int(getattr(args, "forehead_skull_bump_min_vertices", 12)),
        max_bump_fraction=float(getattr(args, "forehead_skull_bump_max_fraction", 0.22)),
        score_p90_blend=float(getattr(args, "forehead_skull_bump_score_p90_blend", 0.55)),
    )
    move_mask = weights > 1e-6
    if not np.any(move_mask):
        print(
            "[example] forehead/skull bump flatten: no bumps detected in candidate shell, skip",
            flush=True,
        )
        return vertices

    neck_protect = build_neck_protect_vertex_mask(
        n_vertices,
        faces,
        seg28_pkl_path,
        vertices,
        chin_neck_band_scale=float(getattr(args, "neck_protect_chin_band_scale", 16.0)),
    )
    fixed = np.ones(n_vertices, dtype=bool)
    active = move_mask & fs_allowed
    fixed[active] = False
    fixed |= np.asarray(ear_mask, dtype=bool)
    fixed |= np.asarray(neck_protect, dtype=bool)
    if correspondence_exclude_vertex_mask is not None:
        fixed |= np.asarray(correspondence_exclude_vertex_mask, dtype=bool)
    if landmark_vertex_ids is not None and landmark_vertex_ids.size:
        fixed[np.asarray(landmark_vertex_ids, dtype=np.int64).reshape(-1)] = True
    deep_fs = fs_allowed & ~candidate_mask
    fixed[deep_fs] = True
    weights[fixed] = 0.0

    reference = vertices.copy()
    bilap_weights = weights.copy()
    free_mask = (bilap_weights > 1e-6) & ~fixed
    max_disp = np.full(n_vertices, np.inf, dtype=np.float64)
    max_disp_scale = float(
        getattr(args, "forehead_skull_bump_bilaplacian_max_displacement_scale", 0.28)
    )
    if max_disp_scale > 0.0:
        max_disp[free_mask] = max_disp_scale * edge_scale
        bump_cap = float(
            getattr(args, "forehead_skull_bump_bilaplacian_bump_max_displacement_scale", 0.36)
        )
        if bump_cap > 0.0 and np.any(bump_mask):
            max_disp[bump_mask & free_mask] = bump_cap * edge_scale

    out = vertices.copy()
    if not bool(getattr(args, "no_forehead_skull_bump_bilaplacian", False)):
        out = _forehead_skull_bilap_pass(
            vertices,
            faces,
            args,
            bilap_weights=bilap_weights,
            bump_mask=bump_mask,
            fixed=fixed,
            fs_allowed=fs_allowed,
            edge_scale=edge_scale,
            reference_vertices=reference,
            fixed_vertices=reference,
            max_disp=max_disp,
        )
        if int(getattr(args, "forehead_skull_bump_bilaplacian_post_weld_passes", 1)) > 0:
            post_ref = out.copy()
            out = _forehead_skull_bilap_pass(
                out,
                faces,
                args,
                bilap_weights=bilap_weights,
                bump_mask=bump_mask,
                fixed=fixed,
                fs_allowed=fs_allowed,
                edge_scale=edge_scale,
                reference_vertices=post_ref,
                fixed_vertices=post_ref,
                max_disp=max_disp,
                bump_anchor_scale=0.5,
            )

    seam_move = np.asarray(move_mask, dtype=bool)
    ce_iters = int(getattr(args, "forehead_skull_bump_flatten_cross_edge_iterations", 24))
    ce_blend = float(getattr(args, "forehead_skull_bump_flatten_cross_edge_blend", 0.92))
    weld_iters = int(getattr(args, "forehead_skull_bump_flatten_weld_iterations", 16))
    pin = np.asarray(ear_mask, dtype=bool)
    if landmark_vertex_ids is not None and landmark_vertex_ids.size:
        pin[landmark_vertex_ids] = True
    if np.any(forehead_mask) and np.any(skull_mask):
        if ce_iters > 0:
            out = weld_region_pair_cross_edge_positions(
                out,
                faces,
                forehead_mask,
                skull_mask,
                iterations=ce_iters,
                blend=ce_blend,
                move_mask=seam_move,
                pin_mask=pin,
            )
        if weld_iters > 0:
            out = weld_region_pair_cross_edge_positions(
                out,
                faces,
                forehead_mask,
                skull_mask,
                iterations=weld_iters,
                blend=1.0,
                move_mask=seam_move,
                pin_mask=pin,
            )
    print(
        f"[example] forehead/skull bump flatten: "
        f"candidate={int(bump_stats.get('candidate', 0))} bumps={int(bump_stats.get('bumps', 0))} "
        f"support={int(bump_stats.get('support', 0))} thresh={bump_stats.get('thresh', 0.0):.4f} "
        f"p90={bump_stats.get('score_p90', 0.0):.4f} "
        f"seam_weld={ce_iters}+{weld_iters} active={int(np.count_nonzero(weights > 1e-6))}",
        flush=True,
    )
    return out


def run_robust_auto_postprocess(
    src_v: np.ndarray,
    src_f: np.ndarray,
    fitted_vertices: np.ndarray,
    result_faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str] | None,
    args: argparse.Namespace,
    prior_region_info: dict,
    landmarks: dict | None,
    correspondence_exclude_vertex_mask: np.ndarray | None,
    face_interior_anchor_vertices: np.ndarray | None,
    face_boundary_band: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Post-ICP non-face smooth, interior restore, optional seam passes (lite when dARAP on)."""
    _ = (src_f, face_boundary_band)
    lite = _postprocess_lite_with_darap(args)
    ultra = _postprocess_ultra_lite(args)
    if ultra:
        print(
            "[example] postprocess ultra-lite (example_fast): "
            "skip non-face smooth and face-interior restore before regional dARAP",
            flush=True,
        )
    elif lite:
        print(
            "[example] postprocess lite (regional dARAP on): "
            "lighter non-face seam smooth, skip redundant final bump/seam passes",
            flush=True,
        )
    nonface_smooth_stats: dict[str, float] = {
        "enabled": 0.0,
        "iterations": 0.0,
        "num_smooth_vertices": 0.0,
        "weight_mean": 0.0,
    }
    face_mask_post = np.asarray(
        prior_region_info.pop("_face_vertex_mask", np.zeros(src_v.shape[0], dtype=bool)),
        dtype=bool,
    )
    nonface_mask_post = np.asarray(
        prior_region_info.pop("_nonface_vertex_mask", ~face_mask_post),
        dtype=bool,
    )
    if ultra:
        pass
    elif not bool(getattr(args, "no_nonface_smooth", False)):
        landmark_pin_ids = None
        if landmarks is not None and "face_vertices" in landmarks:
            landmark_pin_ids = np.unique(np.asarray(landmarks["face_vertices"], dtype=np.int64).reshape(-1))
        face_vertices_frozen = fitted_vertices[face_mask_post].copy()
        junction_reference_vertices = fitted_vertices.copy()
        ear_vertex_mask, ear_region_info = build_seg28_face_vertex_mask(
            src_v.shape[0],
            src_f,
            seg28_pkl_path,
            region_names=HACK_EAR_REGION_NAMES,
        )
        smooth_exclude_mask = np.asarray(ear_vertex_mask, dtype=bool).copy()
        if correspondence_exclude_vertex_mask is not None:
            smooth_exclude_mask |= np.asarray(correspondence_exclude_vertex_mask, dtype=bool)
        ear_vertices_frozen = junction_reference_vertices[ear_vertex_mask].copy()
        forehead_mask, _ = build_seg28_face_vertex_mask(
            src_v.shape[0], src_f, seg28_pkl_path, region_names=HACK_FOREHEAD_REGION_NAMES,
        )
        skull_mask, _ = build_seg28_face_vertex_mask(
            src_v.shape[0], src_f, seg28_pkl_path, region_names=HACK_SKULL_REGION_NAMES,
        )
        chin_mask, _ = build_seg28_face_vertex_mask(
            src_v.shape[0], src_f, seg28_pkl_path, region_names=HACK_CHIN_REGION_NAMES,
        )
        neck_f_mask, _ = build_seg28_face_vertex_mask(
            src_v.shape[0], src_f, seg28_pkl_path, region_names=HACK_NECK_FRONT_REGION_NAMES,
        )
        neck_protect_mask = build_neck_protect_vertex_mask(
            src_v.shape[0],
            src_f,
            seg28_pkl_path,
            fitted_vertices,
            chin_neck_band_scale=float(
                getattr(args, "neck_protect_chin_band_scale", 16.0)
            ),
        )
        (
            forehead_skull_face_relax_mask,
            forehead_skull_nonface_relax_mask,
            forehead_skull_flex_band,
            forehead_skull_nonface_flex_band,
            forehead_skull_band_width,
            forehead_anchor_ramp,
            _forehead_deform_ramp,
        ) = build_forehead_skull_seam_flex_masks(
            fitted_vertices,
            result_faces,
            seg28_pkl_path,
            band_width_scale=float(args.forehead_skull_face_flex_band_scale),
            inner_ramp_threshold=float(
                args.nonface_smooth_forehead_skull_harmonize_inner_threshold
            ),
        )
        forehead_skull_deform_mask = (
            forehead_skull_face_relax_mask | forehead_skull_nonface_relax_mask
        )
        forehead_skull_face_junction_iters = int(
            args.nonface_smooth_forehead_skull_face_junction_iterations
        )
        if lite:
            forehead_skull_face_junction_iters = 0
        chin_neck_seam_pin = (
            region_cross_boundary_seed(result_faces, chin_mask, neck_f_mask)
            & nonface_mask_post
        )
        region_seam_pin_mask = chin_neck_seam_pin
        chin_neck_band_scale = float(args.nonface_smooth_region_seam_band_scale)
        skull_band_scale = float(args.nonface_smooth_forehead_skull_band_scale)
        skull_band_iters = int(args.nonface_smooth_forehead_skull_band_iterations)
        chin_neck_band_iters = int(args.nonface_smooth_region_seam_band_iterations)
        if lite:
            skull_band_iters = min(skull_band_iters, 12)
            chin_neck_band_iters = min(chin_neck_band_iters, 6)
        junction_smooth_weights, junction_width, junction_seam_distance = build_face_nonface_junction_normal_smooth_weights(
            fitted_vertices,
            result_faces,
            face_mask_post,
            nonface_mask_post,
            junction_width_scale=float(args.nonface_junction_width_scale),
            exclude_vertex_mask=smooth_exclude_mask,
            landmark_vertex_ids=landmark_pin_ids,
            weight_smooth_iterations=int(args.nonface_smooth_weight_smooth_iterations),
            weight_smooth_step=float(args.nonface_smooth_weight_smooth_step),
        )
        edge_scale = float(median_edge_length(fitted_vertices, result_faces))
        junction_shell_weights = np.clip(
            np.asarray(junction_smooth_weights, dtype=np.float64).reshape(-1), 0.0, 1.0
        )
        nonface_base_weight = float(np.clip(args.nonface_smooth_extended_weight, 0.0, 1.0))
        combined_smooth_weights = np.zeros_like(junction_shell_weights)
        combined_smooth_weights[nonface_mask_post] = nonface_base_weight
        combined_smooth_weights = np.maximum(combined_smooth_weights, junction_shell_weights)
        combined_smooth_weights[~nonface_mask_post] = 0.0
        combined_smooth_weights[ear_vertex_mask] = 0.0
        if correspondence_exclude_vertex_mask is not None:
            combined_smooth_weights[np.asarray(correspondence_exclude_vertex_mask, dtype=bool)] = 0.0
        if landmark_pin_ids is not None and landmark_pin_ids.size:
            combined_smooth_weights[landmark_pin_ids] = 0.0
        junction_smooth_iterations = int(args.nonface_smooth_iterations)
        projection_iterations = int(args.nonface_smooth_seam_projection_iterations)
        fnf_harmonize_iters = int(args.nonface_smooth_face_nonface_harmonize_iterations)
        fnf_stretch_iters = int(args.nonface_smooth_face_nonface_stretch_iterations)
        fs_stretch_iters = int(args.nonface_smooth_forehead_skull_stretch_iterations)
        fs_intra_stretch_iters = int(args.nonface_smooth_forehead_skull_intra_stretch_iterations)
        if lite:
            junction_smooth_iterations = min(junction_smooth_iterations, 24)
            projection_iterations = min(projection_iterations, 4)
            fnf_harmonize_iters = min(fnf_harmonize_iters, 8)
            fnf_stretch_iters = min(fnf_stretch_iters, 8)
            fs_stretch_iters = 0
            fs_intra_stretch_iters = 0
        post_smooth_vertices = junction_reference_vertices.copy()
        junction_displacement_limits = build_face_nonface_junction_displacement_limits(
            fitted_vertices,
            result_faces,
            face_mask_post,
            nonface_mask_post,
            junction_width=junction_width,
            seam_displacement_scale=float(args.nonface_smooth_seam_displacement_scale),
            outer_displacement_scale=float(args.nonface_smooth_outer_displacement_scale),
            exclude_vertex_mask=smooth_exclude_mask,
        )
        region_seam_vertices_frozen = post_smooth_vertices[region_seam_pin_mask].copy()
        junction_post_enabled = junction_smooth_iterations > 0 and np.any(combined_smooth_weights > 1e-6)
        if junction_post_enabled:
            fitted_vertices = optimize_face_nonface_junction_normal_smoothness(
                fitted_vertices,
                result_faces,
                face_mask_post,
                combined_smooth_weights,
                iterations=junction_smooth_iterations,
                lamb=float(args.nonface_smooth_lambda),
                mu=float(args.nonface_smooth_mu),
                tangential_only=not bool(args.nonface_smooth_normal_off),
                normal_alignment_step=float(args.nonface_smooth_normal_alignment_step),
                max_displacement=junction_displacement_limits,
                max_iteration_displacement_scale=float(
                    args.nonface_smooth_max_iteration_displacement_scale
                ),
            )
            post_smooth_vertices = fitted_vertices.copy()
            region_seam_vertices_frozen = post_smooth_vertices[region_seam_pin_mask].copy()
            skull_band_width = 0.0
            chin_neck_band_width = 0.0
            forehead_skull_union_mask = forehead_mask | skull_mask
            if forehead_skull_face_junction_iters > 0 and np.any(forehead_skull_face_relax_mask):
                fs_face_junction_weights = combined_smooth_weights.copy()
                face_side_peak = float(args.nonface_smooth_forehead_skull_face_junction_weight)
                fs_face_junction_weights[forehead_skull_face_relax_mask] = np.maximum(
                    fs_face_junction_weights[forehead_skull_face_relax_mask],
                    face_side_peak * forehead_skull_flex_band[forehead_skull_face_relax_mask],
                )
                fs_face_pin_mask = face_mask_post.copy()
                fs_face_pin_mask[forehead_skull_face_relax_mask] = False
                fs_face_max_disp = np.full(src_v.shape[0], np.inf, dtype=np.float64)
                face_disp_scale = float(args.nonface_smooth_forehead_skull_face_max_displacement_scale)
                if face_disp_scale > 0.0:
                    fs_face_max_disp[forehead_skull_face_relax_mask] = (
                        face_disp_scale * edge_scale
                    )
                fitted_vertices = optimize_face_nonface_junction_normal_smoothness(
                    fitted_vertices,
                    result_faces,
                    face_mask_post,
                    fs_face_junction_weights,
                    iterations=forehead_skull_face_junction_iters,
                    lamb=float(args.nonface_smooth_lambda),
                    mu=float(args.nonface_smooth_mu),
                    tangential_only=False,
                    normal_alignment_step=float(
                        args.nonface_smooth_forehead_skull_band_normal_step
                    ),
                    max_displacement=fs_face_max_disp,
                    max_iteration_displacement_scale=float(
                        args.nonface_smooth_forehead_skull_band_max_step_scale
                    ),
                    pin_vertex_mask=fs_face_pin_mask,
                )
                post_smooth_vertices = fitted_vertices.copy()
            skull_harmonize_iters = int(args.nonface_smooth_forehead_skull_harmonize_iterations)
            skull_cross_edge_iters = int(args.nonface_smooth_forehead_skull_cross_edge_iterations)
            if lite:
                skull_harmonize_iters = 0
                skull_cross_edge_iters = 0
            skull_band_weights = np.zeros(src_v.shape[0], dtype=np.float64)
            if skull_band_iters > 0 and np.any(skull_mask):
                forehead_skull_shared_geo = forehead_skull_union_mask
                skull_band_weights, skull_band_width = build_region_seam_band_soften_weights(
                    fitted_vertices,
                    result_faces,
                    forehead_mask,
                    skull_mask,
                    band_width_scale=skull_band_scale,
                    peak_weight=float(args.nonface_smooth_forehead_skull_band_weight),
                    bidirectional=True,
                    shared_geodesic_mask=forehead_skull_shared_geo,
                )
                if np.any(skull_band_weights > 1e-6):
                    skull_pass_reference = fitted_vertices.copy()
                    forehead_skull_deform_mask = (
                        forehead_skull_deform_mask
                        | (forehead_mask & (skull_band_weights > 1e-6))
                        | (skull_mask & (skull_band_weights > 1e-6))
                    )
                    if landmark_pin_ids is not None and landmark_pin_ids.size:
                        skull_band_weights[landmark_pin_ids] = 0.0
                        forehead_skull_deform_mask[landmark_pin_ids] = False
                    skull_band_pin_mask = face_mask_post.copy()
                    skull_band_pin_mask[forehead_skull_deform_mask] = False
                    # Keep deep interior forehead pinned; seam band may move with skull.
                    forehead_skull_max_disp = np.full(
                        src_v.shape[0], np.inf, dtype=np.float64
                    )
                    forehead_limit_scale = float(
                        args.nonface_smooth_forehead_skull_forehead_max_displacement_scale
                    )
                    if forehead_limit_scale > 0.0:
                        forehead_band = forehead_mask & (skull_band_weights > 1e-6)
                        limit_scale = forehead_limit_scale * min(
                            4.0,
                            max(1.0, float(skull_band_scale) / 10.0),
                        )
                        forehead_skull_max_disp[forehead_band] = limit_scale * edge_scale
                        skull_band = skull_mask & (skull_band_weights > 1e-6)
                        skull_limit_scale = float(
                            args.nonface_smooth_forehead_skull_skull_max_displacement_scale
                        )
                        if skull_limit_scale > 0.0:
                            forehead_skull_max_disp[skull_band] = (
                                skull_limit_scale
                                * min(4.0, max(1.0, float(skull_band_scale) / 10.0))
                                * edge_scale
                            )
                    fitted_vertices = optimize_face_nonface_junction_normal_smoothness(
                        fitted_vertices,
                        result_faces,
                        face_mask_post,
                        skull_band_weights,
                        iterations=skull_band_iters,
                        lamb=float(args.nonface_smooth_lambda),
                        mu=float(args.nonface_smooth_mu),
                        tangential_only=False,
                        normal_alignment_step=float(
                            args.nonface_smooth_forehead_skull_band_normal_step
                        ),
                        max_displacement=forehead_skull_max_disp,
                        max_iteration_displacement_scale=float(
                            args.nonface_smooth_forehead_skull_band_max_step_scale
                        ),
                        pin_vertex_mask=skull_band_pin_mask,
                    )
                    deform_ramp, anchor_ramp, _ = build_region_junction_motion_ramp(
                        fitted_vertices,
                        result_faces,
                        forehead_mask,
                        skull_mask,
                        ramp_width_scale=skull_band_scale,
                        bidirectional=True,
                        shared_geodesic_mask=forehead_skull_shared_geo,
                    )
                    union_ramp = np.ones(src_v.shape[0], dtype=np.float64)
                    union_ramp[forehead_mask] = anchor_ramp[forehead_mask]
                    union_ramp[skull_mask] = deform_ramp[skull_mask]
                    harmonize_fixed = np.ones(src_v.shape[0], dtype=bool)
                    harmonize_fixed[forehead_skull_union_mask] = False
                    inner_fs_thresh = float(
                        args.nonface_smooth_forehead_skull_harmonize_inner_threshold
                    )
                    harmonize_fixed[
                        forehead_mask
                        & (forehead_anchor_ramp >= inner_fs_thresh)
                    ] = True
                    harmonize_fixed[ear_vertex_mask] = True
                    if landmark_pin_ids is not None and landmark_pin_ids.size:
                        harmonize_fixed[landmark_pin_ids] = True
                    if skull_harmonize_iters > 0:
                        fitted_vertices = harmonize_region_seam_displacement(
                            junction_reference_vertices,
                            fitted_vertices,
                            result_faces,
                            active_mask=forehead_skull_union_mask,
                            fixed_mask=harmonize_fixed,
                            displacement_ramp=union_ramp,
                            iterations=skull_harmonize_iters,
                            inner_ramp_threshold=float(
                                args.nonface_smooth_forehead_skull_harmonize_inner_threshold
                            ),
                        )
                    if skull_cross_edge_iters > 0:
                        fitted_vertices = relax_region_pair_cross_edge_displacement(
                            junction_reference_vertices,
                            fitted_vertices,
                            result_faces,
                            forehead_mask,
                            skull_mask,
                            iterations=skull_cross_edge_iters,
                            blend=float(args.nonface_smooth_forehead_skull_cross_edge_blend),
                            move_mask=forehead_skull_deform_mask,
                        )
                    fs_deep_taubin_iters = int(
                        getattr(args, "nonface_smooth_forehead_skull_deep_taubin_iterations", 0)
                    )
                    if lite:
                        fs_deep_taubin_iters = 0
                    if fs_deep_taubin_iters > 0 and np.any(skull_band_weights > 1e-6):
                        deep_fs_weights = np.clip(skull_band_weights, 0.0, 1.0)
                        deep_fs_weights[face_mask_post] = 0.0
                        if landmark_pin_ids is not None and landmark_pin_ids.size:
                            deep_fs_weights[landmark_pin_ids] = 0.0
                        deep_fs_weights[ear_vertex_mask] = 0.0
                        fitted_vertices = optimize_nonface_region_smoothness(
                            fitted_vertices,
                            result_faces,
                            deep_fs_weights,
                            iterations=fs_deep_taubin_iters,
                            lamb=float(
                                getattr(
                                    args,
                                    "nonface_smooth_forehead_skull_deep_taubin_lambda",
                                    0.52,
                                )
                            ),
                            mu=float(
                                getattr(
                                    args,
                                    "nonface_smooth_forehead_skull_deep_taubin_mu",
                                    -0.54,
                                )
                            ),
                        )
                        fs_deep_weld_iters = int(
                            args.nonface_smooth_forehead_skull_weld_iterations
                        )
                        if fs_deep_weld_iters > 0:
                            fitted_vertices = weld_region_pair_cross_edge_positions(
                                fitted_vertices,
                                result_faces,
                                forehead_mask,
                                skull_mask,
                                iterations=max(10, fs_deep_weld_iters // 3),
                                blend=float(args.nonface_smooth_forehead_skull_weld_blend),
                                move_mask=forehead_skull_deform_mask,
                                pin_mask=skull_band_pin_mask,
                            )
                        print(
                            f"[example] forehead/skull deep Taubin: iters={fs_deep_taubin_iters} "
                            f"verts={int(np.count_nonzero(deep_fs_weights > 1e-6))}",
                            flush=True,
                        )
            if chin_neck_band_iters > 0 and (np.any(neck_f_mask) or np.any(chin_mask)):
                chin_neck_band_weights, chin_neck_band_width = build_region_seam_band_soften_weights(
                    fitted_vertices,
                    result_faces,
                    chin_mask,
                    neck_f_mask,
                    band_width_scale=chin_neck_band_scale,
                    peak_weight=float(args.nonface_smooth_region_seam_band_weight),
                    bidirectional=True,
                )
                if np.any(chin_neck_band_weights > 1e-6):
                    fitted_vertices = optimize_face_nonface_junction_normal_smoothness(
                        fitted_vertices,
                        result_faces,
                        face_mask_post,
                        chin_neck_band_weights,
                        iterations=chin_neck_band_iters,
                        lamb=float(args.nonface_smooth_lambda),
                        mu=float(args.nonface_smooth_mu),
                        tangential_only=True,
                        normal_alignment_step=float(args.nonface_smooth_region_seam_band_normal_step),
                        max_displacement=None,
                        max_iteration_displacement_scale=float(
                            args.nonface_smooth_region_seam_band_max_step_scale
                        ),
                    )
            face_vertex_ids = np.where(face_mask_post)[0]
            restore_on_face = np.ones(face_vertex_ids.shape[0], dtype=bool)
            if np.any(forehead_skull_deform_mask):
                restore_on_face = ~forehead_skull_deform_mask[face_vertex_ids]
            if np.any(restore_on_face):
                restore_ids = face_vertex_ids[restore_on_face]
                fitted_vertices[restore_ids] = face_vertices_frozen[restore_on_face]
            fitted_vertices[ear_vertex_mask] = ear_vertices_frozen
            fitted_vertices[region_seam_pin_mask] = region_seam_vertices_frozen

            def _reapply_face_pins() -> None:
                if np.any(restore_on_face):
                    fitted_vertices[face_vertex_ids[restore_on_face]] = (
                        face_vertices_frozen[restore_on_face]
                    )
                fitted_vertices[ear_vertex_mask] = ear_vertices_frozen

            if projection_iterations > 0:
                fitted_vertices = project_nonface_seam_onto_face_tangent_planes(
                    fitted_vertices,
                    result_faces,
                    face_mask_post,
                    nonface_mask_post,
                    iterations=projection_iterations,
                    blend=float(args.nonface_smooth_seam_projection_blend),
                    max_normal_offset_scale=float(
                        args.nonface_smooth_seam_projection_max_offset_scale
                    ),
                    reference_vertices=junction_reference_vertices,
                    free_face_mask=forehead_skull_face_relax_mask,
                )
                _reapply_face_pins()
            nonface_motion_ramp = np.ones(src_v.shape[0], dtype=np.float64)
            nonface_shell = nonface_mask_post & np.isfinite(junction_seam_distance)
            nonface_motion_ramp[nonface_shell] = _smoothstep(
                np.clip(junction_seam_distance[nonface_shell] / max(junction_width, 1e-12), 0.0, 1.0)
            )
            fnf_harmonize_fixed = ~nonface_mask_post
            fnf_harmonize_fixed |= ear_vertex_mask
            if np.any(forehead_skull_nonface_relax_mask):
                fnf_harmonize_fixed |= forehead_skull_nonface_relax_mask
            if landmark_pin_ids is not None and landmark_pin_ids.size:
                fnf_harmonize_fixed[landmark_pin_ids] = True
            if fnf_harmonize_iters > 0:
                fitted_vertices = harmonize_region_seam_displacement(
                    junction_reference_vertices,
                    fitted_vertices,
                    result_faces,
                    active_mask=nonface_mask_post,
                    fixed_mask=fnf_harmonize_fixed,
                    displacement_ramp=nonface_motion_ramp,
                    iterations=fnf_harmonize_iters,
                    inner_ramp_threshold=float(
                        args.nonface_smooth_face_nonface_harmonize_inner_threshold
                    ),
                )
                _reapply_face_pins()
            stretch_move_mask = build_face_nonface_cross_seam_nonface_mask(
                result_faces,
                face_mask_post,
                nonface_mask_post,
            )
            stretch_band_scale = float(args.nonface_smooth_face_nonface_stretch_band_scale)
            if stretch_band_scale > 0.0:
                stretch_move_mask &= np.isfinite(junction_seam_distance)
                stretch_move_mask &= (
                    junction_seam_distance <= stretch_band_scale * edge_scale
                )
            stretch_move_mask &= ~ear_vertex_mask
            if correspondence_exclude_vertex_mask is not None:
                stretch_move_mask &= ~np.asarray(
                    correspondence_exclude_vertex_mask, dtype=bool
                )
            if fnf_stretch_iters > 0 and np.any(stretch_move_mask):
                fitted_vertices = relax_face_nonface_cross_edge_stretch(
                    src_v,
                    fitted_vertices,
                    result_faces,
                    face_mask_post,
                    nonface_mask_post,
                    max_stretch_ratio=float(args.nonface_smooth_face_nonface_max_stretch_ratio),
                    min_stretch_ratio=float(args.nonface_smooth_face_nonface_min_stretch_ratio),
                    iterations=fnf_stretch_iters,
                    blend=float(args.nonface_smooth_face_nonface_stretch_blend),
                    move_mask=stretch_move_mask,
                )
            fs_stretch_pin = ear_vertex_mask.copy()
            if landmark_pin_ids is not None and landmark_pin_ids.size:
                fs_stretch_pin[landmark_pin_ids] = True
            fs_stretch_pin |= face_mask_post & ~forehead_skull_face_relax_mask
            if fs_stretch_iters > 0 and np.any(forehead_mask) and np.any(skull_mask):
                fitted_vertices = relax_region_pair_cross_edge_stretch(
                    src_v,
                    fitted_vertices,
                    result_faces,
                    forehead_mask,
                    skull_mask,
                    max_stretch_ratio=float(
                        args.nonface_smooth_forehead_skull_max_stretch_ratio
                    ),
                    min_stretch_ratio=float(
                        args.nonface_smooth_forehead_skull_min_stretch_ratio
                    ),
                    iterations=fs_stretch_iters,
                    blend=float(args.nonface_smooth_forehead_skull_stretch_blend),
                    move_mask_a=forehead_skull_face_relax_mask,
                    move_mask_b=forehead_skull_nonface_relax_mask,
                    pin_mask=fs_stretch_pin,
                )
            if fs_intra_stretch_iters > 0 and np.any(skull_mask):
                fs_cross_seed = region_cross_boundary_seed(
                    result_faces, forehead_mask, skull_mask
                )
                skull_stretch_band = skull_mask & np.isfinite(
                    geodesic_distance_from_mask(
                        fitted_vertices,
                        result_faces,
                        fs_cross_seed,
                        max_distance=float(args.nonface_smooth_forehead_skull_intra_stretch_band_scale)
                        * edge_scale,
                        allowed_mask=skull_mask,
                    )
                )
                if np.any(skull_stretch_band):
                    fitted_vertices = relax_edges_in_vertex_mask_stretch(
                        src_v,
                        fitted_vertices,
                        result_faces,
                        skull_stretch_band,
                        max_stretch_ratio=float(
                            args.nonface_smooth_forehead_skull_max_stretch_ratio
                        ),
                        min_stretch_ratio=float(
                            args.nonface_smooth_forehead_skull_min_stretch_ratio
                        ),
                        iterations=fs_intra_stretch_iters,
                        blend=float(args.nonface_smooth_forehead_skull_stretch_blend),
                        pin_mask=fs_stretch_pin,
                    )
            fs_final_cross_iters = int(
                args.nonface_smooth_forehead_skull_final_cross_edge_iterations
            )
            if lite:
                fs_final_cross_iters = 0
            if fs_final_cross_iters > 0 and np.any(forehead_mask) and np.any(skull_mask):
                fitted_vertices = relax_region_pair_cross_edge_displacement(
                    junction_reference_vertices,
                    fitted_vertices,
                    result_faces,
                    forehead_mask,
                    skull_mask,
                    iterations=fs_final_cross_iters,
                    blend=float(
                        args.nonface_smooth_forehead_skull_final_cross_edge_blend
                    ),
                    move_mask=forehead_skull_deform_mask,
                )
            fs_weld_iters = int(args.nonface_smooth_forehead_skull_weld_iterations)
            fs_tangent_iters = int(
                args.nonface_smooth_forehead_skull_tangent_align_iterations
            )
            if lite:
                fs_weld_iters = 0
                fs_tangent_iters = 0
            if np.any(forehead_mask) and np.any(skull_mask) and not lite:
                fs_repair_pin = ear_vertex_mask.copy()
                if landmark_pin_ids is not None and landmark_pin_ids.size:
                    fs_repair_pin[landmark_pin_ids] = True
                fs_repair_move = forehead_skull_deform_mask.copy()
                if fs_weld_iters > 0:
                    fitted_vertices = weld_region_pair_cross_edge_positions(
                        fitted_vertices,
                        result_faces,
                        forehead_mask,
                        skull_mask,
                        iterations=fs_weld_iters,
                        blend=float(args.nonface_smooth_forehead_skull_weld_blend),
                        move_mask=fs_repair_move,
                        pin_mask=fs_repair_pin,
                    )
                if fs_tangent_iters > 0:
                    fitted_vertices = align_region_pair_cross_edges_to_face_tangent(
                        fitted_vertices,
                        result_faces,
                        face_mask_post,
                        forehead_mask,
                        skull_mask,
                        iterations=fs_tangent_iters,
                        blend=float(
                            args.nonface_smooth_forehead_skull_tangent_align_blend
                        ),
                        move_nonface_mask=forehead_skull_nonface_relax_mask,
                        pin_mask=fs_repair_pin,
                    )
                fs_invert_repair_iters = int(
                    args.nonface_smooth_forehead_skull_invert_repair_iterations
                )
                if fs_invert_repair_iters > 0:
                    invert_protect = np.ones(src_v.shape[0], dtype=np.float64)
                    invert_protect[~face_mask_post] = 0.0
                    invert_protect[ear_vertex_mask] = 1.0
                    if landmark_pin_ids is not None and landmark_pin_ids.size:
                        invert_protect[landmark_pin_ids] = 1.0
                    fitted_vertices = repair_inverted_faces(
                        src_v,
                        fitted_vertices,
                        result_faces,
                        iterations=fs_invert_repair_iters,
                        blend_step=float(
                            args.nonface_smooth_forehead_skull_invert_repair_blend
                        ),
                        max_restore_displacement_scale=float(
                            args.nonface_smooth_forehead_skull_invert_repair_max_restore_scale
                        ),
                        protect_vertex_band=invert_protect,
                        protect_strength=float(
                            args.nonface_smooth_forehead_skull_invert_repair_protect_strength
                        ),
                    )
                    if fs_weld_iters > 0:
                        fitted_vertices = weld_region_pair_cross_edge_positions(
                            fitted_vertices,
                            result_faces,
                            forehead_mask,
                            skull_mask,
                            iterations=max(8, fs_weld_iters // 2),
                            blend=float(args.nonface_smooth_forehead_skull_weld_blend),
                            move_mask=fs_repair_move,
                            pin_mask=fs_repair_pin,
                        )
                    if fs_tangent_iters > 0:
                        fitted_vertices = align_region_pair_cross_edges_to_face_tangent(
                            fitted_vertices,
                            result_faces,
                            face_mask_post,
                            forehead_mask,
                            skull_mask,
                            iterations=max(6, fs_tangent_iters // 2),
                            blend=float(
                                args.nonface_smooth_forehead_skull_tangent_align_blend
                            ),
                            move_nonface_mask=forehead_skull_nonface_relax_mask,
                            pin_mask=fs_repair_pin,
                        )
                fs_template_restore_blend = float(
                    args.nonface_smooth_forehead_skull_template_restore_blend
                )
                if lite:
                    fs_template_restore_blend = 0.0
                if fs_template_restore_blend > 0.0:
                    fitted_vertices = restore_region_pair_hairline_template_layout(
                        src_v,
                        fitted_vertices,
                        result_faces,
                        face_mask_post,
                        forehead_mask,
                        skull_mask,
                        skull_anchor_mask=forehead_skull_nonface_relax_mask,
                        forehead_adjust_mask=forehead_skull_face_relax_mask,
                        forehead_blend=fs_template_restore_blend,
                    )
            _reapply_face_pins()
        region_smooth_iters = int(args.nonface_region_smooth_iterations)
        if lite:
            region_smooth_iters = min(region_smooth_iters, 20)
        elif not bool(getattr(args, "no_nonface_region_smooth", False)):
            region_smooth_iters = max(
                region_smooth_iters,
                int(getattr(args, "mandatory_nonface_region_smooth_min_iters", 44)),
            )
        if region_smooth_iters > 0:
            pre_region_smooth_vertices = fitted_vertices.copy()
            region_seam_vertices_frozen = fitted_vertices[region_seam_pin_mask].copy()
            region_exclude = np.asarray(ear_vertex_mask, dtype=bool).copy()
            region_exclude |= np.asarray(neck_protect_mask, dtype=bool)
            # Keep deep forehead interior fixed; skull hairline may receive capped region Taubin.
            if np.any(forehead_mask):
                region_exclude |= forehead_mask & (
                    forehead_anchor_ramp
                    >= float(args.nonface_smooth_forehead_skull_harmonize_inner_threshold)
                )
            if correspondence_exclude_vertex_mask is not None:
                region_exclude |= np.asarray(
                    correspondence_exclude_vertex_mask, dtype=bool
                )
            if landmark_pin_ids is not None and landmark_pin_ids.size:
                region_exclude[landmark_pin_ids] = True
            region_weights = build_nonface_interior_smooth_weights(
                nonface_mask_post,
                face_mask_post,
                junction_seam_distance,
                junction_width,
                seam_floor_weight=float(args.nonface_region_smooth_seam_floor_weight),
                forehead_skull_cap_mask=forehead_skull_nonface_relax_mask,
                forehead_skull_cap_weight=float(
                    args.nonface_region_smooth_forehead_skull_cap_weight
                ),
                exclude_vertex_mask=region_exclude,
                min_seam_distance_scale=float(
                    args.nonface_region_smooth_min_seam_distance_scale
                ),
                edge_scale=edge_scale,
            )
            face_pin_mask = face_mask_post & ~forehead_skull_face_relax_mask
            if np.any(region_weights > 1e-6):
                fitted_vertices = optimize_nonface_region_smoothness(
                    fitted_vertices,
                    result_faces,
                    region_weights,
                    iterations=region_smooth_iters,
                    lamb=float(args.nonface_region_smooth_lambda),
                    mu=float(args.nonface_region_smooth_mu),
                )
                fitted_vertices = reapply_post_nonface_smooth_vertex_pins(
                    fitted_vertices,
                    face_reference_vertices=pre_region_smooth_vertices,
                    face_pin_mask=face_pin_mask,
                    ear_vertex_mask=ear_vertex_mask,
                    ear_vertices_frozen=ear_vertices_frozen,
                    region_seam_pin_mask=region_seam_pin_mask,
                    region_seam_vertices_frozen=region_seam_vertices_frozen,
                )
                if np.any(forehead_mask) and np.any(skull_mask) and not lite:
                    fs_repair_pin = ear_vertex_mask.copy()
                    if landmark_pin_ids is not None and landmark_pin_ids.size:
                        fs_repair_pin[landmark_pin_ids] = True
                    fitted_vertices = apply_forehead_skull_seam_finalize(
                        src_v,
                        fitted_vertices,
                        result_faces,
                        face_mask_post,
                        forehead_mask,
                        skull_mask,
                        forehead_skull_face_relax_mask,
                        forehead_skull_nonface_relax_mask,
                        weld_iterations=max(
                            12,
                            int(args.nonface_smooth_forehead_skull_weld_iterations) // 2,
                        ),
                        weld_blend=float(args.nonface_smooth_forehead_skull_weld_blend),
                        tangent_iterations=max(
                            8,
                            int(
                                args.nonface_smooth_forehead_skull_tangent_align_iterations
                            )
                            // 2,
                        ),
                        tangent_blend=float(
                            args.nonface_smooth_forehead_skull_tangent_align_blend
                        ),
                        template_restore_blend=float(
                            args.nonface_smooth_forehead_skull_template_restore_blend
                        )
                        * 0.55,
                        pin_mask=fs_repair_pin,
                    )
                # fitted_vertices updated
                junction_post_enabled = True
                print(
                    f"[example] nonface region Taubin smooth: iters={region_smooth_iters} "
                    f"verts={int(np.count_nonzero(region_weights > 1e-6))} "
                    f"weight_mean={float(np.mean(region_weights[region_weights > 1e-6])):.3f}",
                    flush=True,
                )
        if junction_post_enabled:
            # fitted_vertices updated
            shell_mask = (
                nonface_mask_post
                & np.isfinite(junction_seam_distance)
                & (junction_seam_distance <= junction_width)
            )
            shell_delta = np.linalg.norm(
                fitted_vertices[shell_mask] - junction_reference_vertices[shell_mask],
                axis=1,
            )
            seam_pin_width = edge_scale
            seam_ring = shell_mask & (junction_seam_distance <= seam_pin_width)
            seam_delta = (
                np.linalg.norm(
                    fitted_vertices[seam_ring] - junction_reference_vertices[seam_ring],
                    axis=1,
                )
                if np.any(seam_ring)
                else np.zeros(0, dtype=np.float64)
            )
            cross_seam = measure_cross_seam_nonface_displacement(
                fitted_vertices,
                junction_reference_vertices,
                result_faces,
                face_mask_post,
                nonface_mask_post,
            )
            forehead_skull_cross, forehead_skull_jump, forehead_skull_stretch = _region_seam_stats(
                src_v,
                fitted_vertices,
                junction_reference_vertices,
                result_faces,
                forehead_mask,
                skull_mask,
                track_mask=forehead_mask | skull_mask,
                max_stretch_ratio=float(args.nonface_smooth_forehead_skull_max_stretch_ratio),
            )
            chin_neck_cross, _, _ = _region_seam_stats(
                src_v,
                fitted_vertices,
                junction_reference_vertices,
                result_faces,
                chin_mask,
                neck_f_mask,
                track_mask=neck_f_mask,
            )
            deformable_nonface_mask = nonface_mask_post & ~ear_vertex_mask
            post_smooth_nonface_delta = (
                np.linalg.norm(
                    post_smooth_vertices[deformable_nonface_mask]
                    - junction_reference_vertices[deformable_nonface_mask],
                    axis=1,
                )
                if np.any(deformable_nonface_mask)
                else np.zeros(0, dtype=np.float64)
            )
            nonface_smooth_stats = {
                "enabled": 1.0,
                "mode": 3.0,
                "iterations": float(junction_smooth_iterations),
                "num_smooth_vertices": float(np.count_nonzero(combined_smooth_weights > 1e-6)),
                "num_junction_shell_vertices": float(np.count_nonzero(junction_smooth_weights > 1e-6)),
                "num_ear_protected_vertices": float(ear_region_info.get("num_face_vertices", np.count_nonzero(ear_vertex_mask))),
                "num_region_seam_pinned_vertices": float(np.count_nonzero(region_seam_pin_mask)),
                "skull_band_scale": float(skull_band_scale),
                "skull_band_iterations": float(skull_band_iters),
                "skull_band_width": float(skull_band_width),
                "chin_neck_band_scale": float(chin_neck_band_scale),
                "chin_neck_band_iterations": float(chin_neck_band_iters),
                "chin_neck_band_width": float(chin_neck_band_width),
                "nonface_base_weight": float(args.nonface_smooth_extended_weight),
                "region_smooth_iterations": float(
                    0.0
                    if bool(getattr(args, "no_nonface_region_smooth", False))
                    else float(args.nonface_region_smooth_iterations)
                ),
                "junction_width": float(junction_width),
                "lambda": float(args.nonface_smooth_lambda),
                "mu": float(args.nonface_smooth_mu),
                "normal_alignment_step": float(args.nonface_smooth_normal_alignment_step),
                "tangential_only": float(not bool(args.nonface_smooth_normal_off)),
                "face_max_abs_delta": float(
                    np.max(np.linalg.norm(fitted_vertices[face_mask_post] - face_vertices_frozen, axis=1))
                    if np.any(face_mask_post)
                    else 0.0
                ),
                "post_smooth_nonface_max_abs_delta": float(
                    np.max(post_smooth_nonface_delta) if post_smooth_nonface_delta.size else 0.0
                ),
                "shell_max_abs_delta": float(np.max(shell_delta) if shell_delta.size else 0.0),
                "shell_mean_abs_delta": float(np.mean(shell_delta) if shell_delta.size else 0.0),
                "nonface_max_abs_delta": float(
                    np.max(
                        np.linalg.norm(
                            fitted_vertices[deformable_nonface_mask]
                            - junction_reference_vertices[deformable_nonface_mask],
                            axis=1,
                        )
                    )
                    if np.any(deformable_nonface_mask)
                    else 0.0
                ),
                "ear_max_abs_delta": float(
                    np.max(
                        np.linalg.norm(
                            fitted_vertices[ear_vertex_mask] - ear_vertices_frozen,
                            axis=1,
                        )
                    )
                    if np.any(ear_vertex_mask)
                    else 0.0
                ),
                "seam_max_abs_delta": float(np.max(seam_delta) if seam_delta.size else 0.0),
                "cross_seam_max_abs_delta": float(cross_seam["max"]),
                "cross_seam_p90_abs_delta": float(cross_seam["p90"]),
                "cross_seam_mean_abs_delta": float(cross_seam["mean"]),
                "forehead_skull_cross_p90_abs_delta": float(forehead_skull_cross["p90"]),
                "forehead_skull_seam_jump_p90": float(forehead_skull_jump["p90"]),
                "forehead_skull_seam_jump_max": float(forehead_skull_jump["max"]),
                "forehead_skull_stretch_median": float(forehead_skull_stretch["median"]),
                "forehead_skull_stretch_p90": float(forehead_skull_stretch["p90"]),
                "forehead_skull_stretch_max": float(forehead_skull_stretch["max"]),
                "chin_neck_cross_p90_abs_delta": float(chin_neck_cross["p90"]),
            }
            print(
                f"[example] nonface junction smooth: iters={junction_smooth_iterations} "
                f"verts={int(nonface_smooth_stats['num_smooth_vertices'])} "
                f"shell={int(nonface_smooth_stats['num_junction_shell_vertices'])} "
                f"ears_pinned={int(nonface_smooth_stats['num_ear_protected_vertices'])} "
                f"region_seam_pinned={int(nonface_smooth_stats['num_region_seam_pinned_vertices'])} "
                f"face_max_delta={nonface_smooth_stats['face_max_abs_delta']:.3e} "
                f"post_smooth_nonface_max={nonface_smooth_stats['post_smooth_nonface_max_abs_delta']:.3e} "
                f"nonface_max_delta={nonface_smooth_stats['nonface_max_abs_delta']:.3e} "
                f"ear_max_delta={nonface_smooth_stats['ear_max_abs_delta']:.3e} "
                f"forehead_skull_p90={nonface_smooth_stats['forehead_skull_cross_p90_abs_delta']:.3e} "
                f"forehead_skull_jump_p90={nonface_smooth_stats['forehead_skull_seam_jump_p90']:.3e} "
                f"forehead_skull_stretch_med={nonface_smooth_stats['forehead_skull_stretch_median']:.3f} "
                f"forehead_skull_stretch_p90={nonface_smooth_stats['forehead_skull_stretch_p90']:.3f} "
                f"forehead_skull_stretch_max={nonface_smooth_stats['forehead_skull_stretch_max']:.3f} "
                f"chin_neck_p90={nonface_smooth_stats['chin_neck_cross_p90_abs_delta']:.3e} "
                f"cross_seam_p90={nonface_smooth_stats['cross_seam_p90_abs_delta']:.3e}",
                flush=True,
            )
    if (
        not ultra
        and face_interior_anchor_vertices is not None
        and not bool(getattr(args, "no_restore_face_interior", False))
    ):
        face_mask_restore, _ = build_seg28_face_vertex_mask(
            src_v.shape[0],
            src_f,
            seg28_pkl_path,
            region_names=HACK_SEG28_FACE_REGION_NAMES,
        )
        forehead_relax_restore, skull_relax_restore, _, _, _, _, _ = (
            build_forehead_skull_seam_flex_masks(
                fitted_vertices,
                result_faces,
                seg28_pkl_path,
                band_width_scale=float(args.forehead_skull_face_flex_band_scale),
                inner_ramp_threshold=float(
                    args.nonface_smooth_forehead_skull_harmonize_inner_threshold
                ),
            )
        )
        forehead_mask_restore, _ = build_seg28_face_vertex_mask(
            src_v.shape[0],
            src_f,
            seg28_pkl_path,
            region_names=HACK_FOREHEAD_REGION_NAMES,
        )
        skull_mask_restore, _ = build_seg28_face_vertex_mask(
            src_v.shape[0],
            src_f,
            seg28_pkl_path,
            region_names=HACK_SKULL_REGION_NAMES,
        )
        seam_deform_restore = forehead_relax_restore | skull_relax_restore
        if bool(getattr(args, "restore_face_interior_exclude_forehead", True)):
            face_interior_mask = face_mask_restore & ~forehead_mask_restore
        else:
            face_interior_mask = build_face_interior_preserve_mask(
                face_mask_restore,
                seam_deform_restore,
                fitted_vertices,
                result_faces,
                forehead_mask_restore,
                skull_mask_restore,
                forehead_seam_exclusion_band_scale=float(
                    args.restore_face_interior_forehead_exclusion_band_scale
                ),
            )
        if np.any(face_interior_mask):
            pre_restore_vertices = fitted_vertices.copy()
            seam_influence_mask = build_forehead_skull_seam_influence_mask(
                pre_restore_vertices,
                result_faces,
                forehead_mask_restore,
                skull_mask_restore,
                band_width_scale=float(
                    args.restore_face_interior_seam_influence_band_scale
                ),
            )
            face_interior_mask &= ~seam_influence_mask
            cheek_band = float(args.restore_face_interior_cheek_exclusion_band_scale)
            if cheek_band > 0.0:
                cross_seed = region_cross_boundary_seed(
                    result_faces, forehead_mask_restore, skull_mask_restore
                )
                cheek_near = face_mask_restore & np.isfinite(
                    geodesic_distance_from_mask(
                        pre_restore_vertices,
                        result_faces,
                        cross_seed,
                        max_distance=cheek_band
                        * float(median_edge_length(pre_restore_vertices, result_faces)),
                        allowed_mask=face_mask_restore,
                    )
                )
                face_interior_mask &= ~cheek_near
            if bool(getattr(args, "restore_face_interior_exclude_frown_0", True)):
                frown_restore_mask, _ = build_seg28_face_vertex_mask(
                    src_v.shape[0],
                    src_f,
                    seg28_pkl_path,
                    region_names=HACK_FROWN_REGION_NAMES,
                )
                face_interior_mask &= ~frown_restore_mask
            fitted_vertices = pre_restore_vertices.copy()
            if np.any(face_interior_mask):
                fitted_vertices[face_interior_mask] = face_interior_anchor_vertices[
                    face_interior_mask
                ]
            fitted_vertices[seam_influence_mask] = pre_restore_vertices[seam_influence_mask]
            # fitted_vertices updated
            print(
                f"[example] restored face-deep-interior verts="
                f"{int(np.count_nonzero(face_interior_mask))} "
                f"seam_influence_verts={int(np.count_nonzero(seam_influence_mask))}",
                flush=True,
            )
            post_interior_region_iters = int(
                args.nonface_region_smooth_post_interior_iterations
            )
            if lite:
                post_interior_region_iters = 0
            elif not bool(getattr(args, "no_nonface_region_smooth", False)):
                post_interior_region_iters = max(
                    post_interior_region_iters,
                    int(
                        getattr(
                            args, "mandatory_nonface_region_smooth_post_interior_min_iters", 28
                        )
                    ),
                )
            if post_interior_region_iters > 0:
                _, junction_width_pi, jsd_pi = build_face_nonface_junction_normal_smooth_weights(
                    fitted_vertices,
                    result_faces,
                    face_mask_restore,
                    ~face_mask_restore,
                    junction_width_scale=float(args.nonface_junction_width_scale),
                )
                nf_mask_pi = ~face_mask_restore
                ear_pi, _ = build_seg28_face_vertex_mask(
                    src_v.shape[0],
                    src_f,
                    seg28_pkl_path,
                    region_names=HACK_EAR_REGION_NAMES,
                )
                _, _, _, nfs_relax_pi, _, _, _ = build_forehead_skull_seam_flex_masks(
                    fitted_vertices,
                    result_faces,
                    seg28_pkl_path,
                    band_width_scale=float(args.forehead_skull_face_flex_band_scale),
                    inner_ramp_threshold=float(
                        args.nonface_smooth_forehead_skull_harmonize_inner_threshold
                    ),
                )
                fh_pi, _ = build_seg28_face_vertex_mask(
                    src_v.shape[0],
                    src_f,
                    seg28_pkl_path,
                    region_names=HACK_FOREHEAD_REGION_NAMES,
                )
                sk_pi, _ = build_seg28_face_vertex_mask(
                    src_v.shape[0],
                    src_f,
                    seg28_pkl_path,
                    region_names=HACK_SKULL_REGION_NAMES,
                )
                seam_inf_pi = build_forehead_skull_seam_influence_mask(
                    fitted_vertices,
                    result_faces,
                    fh_pi,
                    sk_pi,
                    band_width_scale=float(
                        args.restore_face_interior_seam_influence_band_scale
                    ),
                )
                region_exclude_pi = np.asarray(ear_pi, dtype=bool) | seam_inf_pi
                region_exclude_pi |= build_neck_protect_vertex_mask(
                    src_v.shape[0],
                    src_f,
                    seg28_pkl_path,
                    fitted_vertices,
                    chin_neck_band_scale=float(
                        getattr(args, "neck_protect_chin_band_scale", 16.0)
                    ),
                )
                if correspondence_exclude_vertex_mask is not None:
                    region_exclude_pi |= np.asarray(
                        correspondence_exclude_vertex_mask, dtype=bool
                    )
                if landmarks is not None and "face_vertices" in landmarks:
                    region_exclude_pi[
                        np.unique(
                            np.asarray(landmarks["face_vertices"], dtype=np.int64).reshape(
                                -1
                            )
                        )
                    ] = True
                fs_deform_pi = forehead_relax_restore | skull_relax_restore
                region_exclude_pi |= fs_deform_pi
                edge_scale_pi = float(median_edge_length(fitted_vertices, result_faces))
                region_weights_pi = build_nonface_interior_smooth_weights(
                    nf_mask_pi,
                    face_mask_restore,
                    jsd_pi,
                    junction_width_pi,
                    seam_floor_weight=float(args.nonface_region_smooth_seam_floor_weight),
                    forehead_skull_cap_mask=nfs_relax_pi,
                    forehead_skull_cap_weight=float(
                        args.nonface_region_smooth_forehead_skull_cap_weight
                    ),
                    exclude_vertex_mask=region_exclude_pi,
                    min_seam_distance_scale=float(
                        args.nonface_region_smooth_min_seam_distance_scale
                    ),
                    edge_scale=edge_scale_pi,
                )
                pre_region_pi = fitted_vertices.copy()
                ear_frozen_pi = pre_region_pi[ear_pi].copy()
                face_pin_pi = face_mask_restore & ~forehead_relax_restore
                if np.any(region_weights_pi > 1e-6):
                    fitted_vertices = optimize_nonface_region_smoothness(
                        fitted_vertices,
                        result_faces,
                        region_weights_pi,
                        iterations=post_interior_region_iters,
                        lamb=float(args.nonface_region_smooth_lambda),
                        mu=float(args.nonface_region_smooth_mu),
                    )
                    if np.any(face_pin_pi):
                        fitted_vertices[face_pin_pi] = pre_region_pi[face_pin_pi]
                    if np.any(ear_pi):
                        fitted_vertices[ear_pi] = ear_frozen_pi
                    fitted_vertices[seam_inf_pi] = pre_region_pi[seam_inf_pi]
                    # fitted_vertices updated
                    print(
                        f"[example] nonface region smooth after interior restore: "
                        f"iters={post_interior_region_iters} "
                        f"verts={int(np.count_nonzero(region_weights_pi > 1e-6))}",
                        flush=True,
                    )
            if not lite and not bool(getattr(args, "no_forehead_skull_post_interior_finalize", False)):
                fs_finalize_pin_pi = None
                if seg28_pkl_path is not None:
                    ear_finalize_pi, _ = build_seg28_face_vertex_mask(
                        src_v.shape[0],
                        src_f,
                        seg28_pkl_path,
                        region_names=HACK_EAR_REGION_NAMES,
                    )
                    fs_finalize_pin_pi = np.asarray(ear_finalize_pi, dtype=bool)
                if landmarks is not None and "face_vertices" in landmarks:
                    lmk_pin_pi = np.zeros(src_v.shape[0], dtype=bool)
                    lmk_pin_pi[
                        np.unique(
                            np.asarray(landmarks["face_vertices"], dtype=np.int64).reshape(
                                -1
                            )
                        )
                    ] = True
                    fs_finalize_pin_pi = (
                        lmk_pin_pi
                        if fs_finalize_pin_pi is None
                        else (fs_finalize_pin_pi | lmk_pin_pi)
                    )
                fitted_vertices = apply_forehead_skull_seam_finalize(
                    src_v,
                    fitted_vertices,
                    result_faces,
                    face_mask_restore,
                    forehead_mask_restore,
                    skull_mask_restore,
                    forehead_relax_restore,
                    skull_relax_restore,
                    weld_iterations=int(
                        args.nonface_smooth_forehead_skull_post_interior_weld_iterations
                    ),
                    weld_blend=float(
                        args.nonface_smooth_forehead_skull_post_interior_weld_blend
                    ),
                    tangent_iterations=int(
                        args.nonface_smooth_forehead_skull_post_interior_tangent_iterations
                    ),
                    tangent_blend=float(
                        args.nonface_smooth_forehead_skull_post_interior_tangent_blend
                    ),
                    template_restore_blend=float(
                        args.nonface_smooth_forehead_skull_post_interior_template_restore_blend
                    ),
                    pin_mask=fs_finalize_pin_pi,
                )
                # fitted_vertices updated
                print(
                    "[example] forehead/skull seam finalize after interior restore",
                    flush=True,
                )
    lmk_ids_final = None
    if landmarks is not None and "face_vertices" in landmarks:
        lmk_ids_final = np.unique(
            np.asarray(landmarks["face_vertices"], dtype=np.int64).reshape(-1)
        )
    if seg28_pkl_path is not None and not lite:
        if not bool(getattr(args, "no_final_seam_repair", False)):
            fitted_vertices = apply_robust_auto_final_seam_repair(
                src_v,
                fitted_vertices,
                result_faces,
                seg28_pkl_path,
                args,
                landmark_vertex_ids=lmk_ids_final,
                correspondence_exclude_vertex_mask=correspondence_exclude_vertex_mask,
            )
            # fitted_vertices updated
            print(
                "[example] final forehead/skull + face/non-face seam repair",
                flush=True,
            )
        if not bool(getattr(args, "no_mandatory_final_smooth", False)):
            j_it, t_it, h_it, _, _ = resolve_mandatory_local_smooth_iters(
                args, force=True
            )
            fitted_vertices = apply_mandatory_final_seam_smoothness(
                fitted_vertices,
                result_faces,
                seg28_pkl_path,
                args,
                template_vertices=src_v,
                landmark_vertex_ids=lmk_ids_final,
                correspondence_exclude_vertex_mask=correspondence_exclude_vertex_mask,
            )
            # fitted_vertices updated
            print(
                f"[example] mandatory local seam smooth: "
                f"junction={j_it} taubin={t_it} harmonize={h_it} "
                f"refine_taubin={int(args.mandatory_local_smooth_refine_taubin_iters)}",
                flush=True,
            )
        if not bool(getattr(args, "no_mandatory_face_interior_smooth", False)):
            fj, ft = resolve_mandatory_face_interior_smooth_iters(args)
            fitted_vertices = apply_mandatory_face_interior_smoothness(
                fitted_vertices,
                result_faces,
                seg28_pkl_path,
                args,
                landmark_vertex_ids=lmk_ids_final,
                correspondence_exclude_vertex_mask=correspondence_exclude_vertex_mask,
            )
            # fitted_vertices updated
            print(
                f"[example] mandatory face-interior smooth: "
                f"junction={fj} taubin={ft} "
                f"refine={int(args.face_interior_smooth_refine_taubin_iterations)}",
                flush=True,
            )
        if not bool(getattr(args, "no_forehead_skull_bump_flatten", False)):
            fitted_vertices = apply_forehead_skull_seam_bump_flatten(
                fitted_vertices,
                result_faces,
                seg28_pkl_path,
                args,
                landmark_vertex_ids=lmk_ids_final,
                correspondence_exclude_vertex_mask=correspondence_exclude_vertex_mask,
            )

    return fitted_vertices, nonface_smooth_stats


