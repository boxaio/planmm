from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import pickle
import sys
from pathlib import Path
from typing import Optional, Sequence

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _path in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch
import trimesh
from utils.mesh import read_obj

from deformations_MINIMAL import (
    ProcrustesPrecompute,
    SparseLaplaciansSolvers,
    calc_ARAP_global_solve,
    calc_rot_matrices_with_procrustes,
    vertex_procrustes_lambda_from_regions,
)

from geometry import (
    evaluate_mesh_edge_stretch,
    evaluate_mesh_validity,
    extract_edges,
    geodesic_distance_from_mask,
    median_edge_length,
    repair_inverted_faces,
    vertex_normals,
)
from meshes.hack_normals import (
    align_target_mesh_to_hack_by_landmarks as _align_target_mesh_to_hack_by_landmarks,
    apply_similarity_to_vertices,
    bni_lmks_npz_to_lm478,
    landmarks_from_mesh,
    load_hack_lmks478_bary,
)
from cli import build_arg_parser
from run import (
    NonRigidICPConfig,
    _edge_band_from_vertex_band,
    camera_depth_correspondence_weights,
    configure_robust_auto,
    find_pairs,
    make_default_config,
    nonrigid_icp,
    stem_from_raw_obj,
)
from seg28_masks import (
    HACK_CAVITY_REGION_NAMES,
    HACK_EAR_REGION_NAMES,
    HACK_DARAP_EYE_EDGE_REGION_NAMES,
    HACK_DARAP_SMOOTH_AUX_REGION_NAMES,
    HACK_FROWN_REGION_NAMES,
    HACK_FOREHEAD_REGION_NAMES,
    build_region_boundary_ring_vertex_mask,
    HACK_SEG28_FACE_REGION_NAMES,
    HACK_SKULL_REGION_NAMES,
    build_hack_cavity_vertex_mask,
    build_seg28_face_vertex_mask,
    face_boundary_seed,
)
from postprocess import (
    _example_fast_enabled,
    _postprocess_ultra_lite,
    apply_forehead_skull_seam_registration_flex,
    run_robust_auto_postprocess,
)


HACK_FITTING_INDICES = np.array([
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 168, 197, 5, 4,
    98, 97, 2, 326, 327, 246, 159, 157, 173, 153, 144, 398, 385, 387,
    263, 373, 381, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 38, 12, 268, 308, 316, 15, 86,
], dtype=np.int64)

EYE_LANDMARK_INDICES = np.array(
    [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246,
        263, 249, 390, 373, 374, 380, 381, 382,
        362, 398, 384, 385, 386, 387, 388, 466,
        468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
    ],
    dtype=np.int64,
)

LIP_LANDMARK_INDICES = np.array(
    [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    ],
    dtype=np.int64,
)

EYELID_OPENING_LANDMARK_PAIRS = np.array(
    [
        [381, 384],
        [380, 385],
        [374, 386],
        [373, 387],
        [390, 388],
        [153, 158],
        [145, 159],
        [144, 160],
        [163, 161],
        [7, 246],
    ],
    dtype=np.int64,
)

MOUTH_OPENING_LANDMARK_PAIRS = np.array(
    [
        [13, 14],
        [312, 317],
        [82, 87],
        [0, 17],
        [267, 314],
        [37, 84],
        [291, 308],
        [61, 78],
    ],
    dtype=np.int64,
)


def read_phack_mesh(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the PHACK source mesh with the project's OBJ reader.
    """
    obj = read_obj(os.fspath(path), tri=True)
    vertices = np.asarray(obj.vs, dtype=np.float64)
    faces = np.asarray(obj.fvs, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path}: expected triangular PHACK faces, got {faces.shape}")
    return vertices, faces


def read_target_mesh(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the BNI target mesh.
    """
    path = os.fspath(path)
    mesh = trimesh.load(path, process=False, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path}: expected triangular faces, got {faces.shape}")
    return vertices, faces


def load_bni_target_face_mask(
    bni_seg_path: str | os.PathLike[str],
    n_faces: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Build a target-side face mask from BNI segmentation.

    The BNI segmentation lives in target topology. Its named regions describe
    the visible facial surface, so their union excludes cavity/internal target
    triangles from correspondence search.
    """
    bni_seg_path = os.fspath(bni_seg_path)
    with np.load(bni_seg_path, allow_pickle=True) as data:
        if "region_names" in data:
            region_names = [str(name) for name in np.asarray(data["region_names"]).reshape(-1)]
        else:
            region_names = []

        mask = np.zeros(int(n_faces), dtype=bool)
        if "face_region_id" in data:
            face_region_id = np.asarray(data["face_region_id"], dtype=np.int64).reshape(-1)
            if face_region_id.shape[0] != int(n_faces):
                raise ValueError(
                    f"{bni_seg_path}: face_region_id length {face_region_id.shape[0]} "
                    f"!= target face count {int(n_faces)}"
                )
            mask = face_region_id >= 0
        else:
            for region_idx, _name in enumerate(region_names):
                key = f"region_{region_idx:02d}_face_ids"
                if key not in data:
                    continue
                face_ids = np.asarray(data[key], dtype=np.int64).reshape(-1)
                face_ids = face_ids[(face_ids >= 0) & (face_ids < int(n_faces))]
                mask[face_ids] = True

    if not np.any(mask):
        raise ValueError(f"{bni_seg_path}: BNI segmentation selected zero target faces")

    return mask, {
        "bni_seg_path": bni_seg_path,
        "region_names": region_names,
        "target_query_faces": int(np.count_nonzero(mask)),
        "target_total_faces": int(n_faces),
    }


def load_bni_target_region_face_mask(
    bni_seg_path: str | os.PathLike[str],
    n_faces: int,
    region_substrings: Sequence[str],
) -> tuple[np.ndarray, dict[str, object]]:
    """Union of BNI target faces whose region name contains any of ``region_substrings``."""
    bni_seg_path = os.fspath(bni_seg_path)
    patterns = tuple(str(s).strip().lower() for s in region_substrings if str(s).strip())
    with np.load(bni_seg_path, allow_pickle=True) as data:
        if "region_names" in data:
            region_names = [str(name) for name in np.asarray(data["region_names"]).reshape(-1)]
        else:
            region_names = []
        mask = np.zeros(int(n_faces), dtype=bool)
        matched_names: list[str] = []
        for region_idx, name in enumerate(region_names):
            name_l = name.lower()
            if patterns and not any(pat in name_l for pat in patterns):
                continue
            key = f"region_{region_idx:02d}_face_ids"
            if key not in data:
                continue
            face_ids = np.asarray(data[key], dtype=np.int64).reshape(-1)
            face_ids = face_ids[(face_ids >= 0) & (face_ids < int(n_faces))]
            if face_ids.size == 0:
                continue
            mask[face_ids] = True
            matched_names.append(name)
    info = {
        "bni_seg_path": bni_seg_path,
        "region_names": region_names,
        "matched_region_names": matched_names,
        "region_substrings": list(patterns),
        "target_query_faces": int(np.count_nonzero(mask)),
        "target_total_faces": int(n_faces),
    }
    return mask, info


def resolve_pair(
    *,
    stem: str | None,
    raw_path: str | None,
    target_path: str | None,
    raw_dir: str,
    target_dir: str,
) -> tuple[str, str, str]:
    """Resolve (stem, raw OBJ, target OBJ) from explicit paths or directory defaults."""
    if raw_path is not None and target_path is not None:
        raw_path = osp.abspath(raw_path)
        target_path = osp.abspath(target_path)
        if stem is None:
            stem = stem_from_raw_obj(raw_path)
        return str(stem), raw_path, target_path

    if stem is not None:
        stem = str(stem)
        raw_candidate = osp.abspath(raw_path or osp.join(raw_dir, f"{stem}_phack_raw.obj"))
        target_candidate = osp.abspath(target_path or osp.join(target_dir, f"{stem}.obj"))
        if not osp.isfile(raw_candidate):
            raise FileNotFoundError(f"source mesh not found: {raw_candidate}")
        if not osp.isfile(target_candidate):
            raise FileNotFoundError(f"target mesh not found: {target_candidate}")
        return stem, raw_candidate, target_candidate

    pairs = find_pairs(raw_dir, target_dir)
    if not pairs:
        raise FileNotFoundError(
            f"no raw/target pairs under raw_dir={raw_dir!r} and target_dir={target_dir!r}"
        )
    return pairs[0]


def align_target_mesh_to_hack_by_landmarks(
    target_vertices: np.ndarray,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    bni_lmks_npz_path: str | os.PathLike[str],
    hack_lmks478_bary_path: str | os.PathLike[str],
    fitting_indices: np.ndarray = HACK_FITTING_INDICES,
) -> tuple[np.ndarray, dict[str, object]]:
    """Align BNI target to HACK; attach full 478 target landmarks for nonrigid constraints."""
    aligned, info = _align_target_mesh_to_hack_by_landmarks(
        target_vertices,
        source_vertices,
        source_faces,
        bni_lmks_npz_path,
        hack_lmks478_bary_path,
        fitting_indices=fitting_indices,
    )
    tgt_lm478 = bni_lmks_npz_to_lm478(bni_lmks_npz_path)
    tgt_lmks_full = apply_similarity_to_vertices(
        tgt_lm478,
        float(info["scale"]),
        np.asarray(info["rotation"], dtype=np.float64),
        np.asarray(info["translation"], dtype=np.float64),
    )
    face_indices, bary_coords = load_hack_lmks478_bary(hack_lmks478_bary_path)
    hack_lm478 = landmarks_from_mesh(
        source_vertices, source_faces, face_indices, bary_coords
    )
    all_ok = np.isfinite(hack_lm478).all(axis=1) & np.isfinite(tgt_lmks_full).all(axis=1)
    info["tgt_lmks_full"] = tgt_lmks_full
    info["lmk_rmse_all_finite"] = (
        float(np.sqrt(np.mean(np.sum((tgt_lmks_full[all_ok] - hack_lm478[all_ok]) ** 2, axis=1))))
        if np.any(all_ok)
        else float("nan")
    )
    return aligned, info


def config_from_args(args: argparse.Namespace) -> NonRigidICPConfig:
    """Build a robust_auto NonRigidICPConfig from CLI arguments."""
    config = make_default_config(
        quick=bool(getattr(args, "quick", False)),
        verbose=not bool(getattr(args, "quiet", False)),
    )
    config = configure_robust_auto(config)

    if getattr(args, "stiffness", None):
        config.stiffness_schedule = tuple(
            float(x.strip()) for x in str(args.stiffness).split(",") if x.strip()
        )
    for attr, cfg_attr, cast in (
        ("max_iterations", "max_iterations", int),
        ("gamma", "gamma", float),
        ("correspondence_mode", "correspondence_mode", str),
        ("normal_angle", "normal_angle_threshold_degrees", float),
        ("distance_threshold", "distance_threshold", float),
        ("distance_threshold_scale", "distance_threshold_scale", float),
        ("trim_percentile", "trim_percentile", float),
        ("solver", "solver", str),
        ("damping", "damping", float),
        ("solver_tolerance", "solver_tolerance", float),
        ("solver_max_iterations", "solver_max_iterations", int),
        ("affine_prior_weight", "affine_prior_weight", float),
        ("affine_prior_decay", "affine_prior_decay", float),
        ("position_prior_weight", "position_prior_weight", float),
        ("position_prior_decay", "position_prior_decay", float),
        ("laplacian_weight", "laplacian_weight", float),
        ("vertex_mass_weight", "vertex_mass_weight", float),
        ("edge_vector_weight", "edge_vector_weight", float),
        ("geometry_weight_decay", "geometry_weight_decay", float),
        ("landmark_weight", "landmark_weight", float),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(config, cfg_attr, cast(value))

    if bool(getattr(args, "no_icp_camera_depth_attenuation", False)):
        config.camera_depth_attenuation_enabled = False
    if hasattr(args, "icp_camera_depth_axis"):
        config.camera_depth_axis = int(args.icp_camera_depth_axis)
    if hasattr(args, "icp_camera_depth_min_weight"):
        config.camera_depth_min_weight = float(args.icp_camera_depth_min_weight)
    if hasattr(args, "icp_camera_depth_power"):
        config.camera_depth_power = float(args.icp_camera_depth_power)
    if hasattr(args, "icp_camera_depth_far_percentile"):
        config.camera_depth_far_percentile = float(args.icp_camera_depth_far_percentile)

    config.visualize_correspondences = bool(getattr(args, "visualize_correspondences", False))
    config.visualize_correspondence_limit = int(
        getattr(args, "visualize_correspondence_limit", 2000)
    )
    return config


def write_obj_mesh(
    path: str | os.PathLike[str],
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    """Write a triangular mesh as OBJ."""
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.export(os.fspath(path))


def _landmark_bary_on_faces(
    faces: np.ndarray,
    landmark_face_indices: np.ndarray,
    landmark_barycentric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    face_vertices = np.asarray(faces)[np.asarray(landmark_face_indices, dtype=np.int64)]
    barycentric = np.asarray(landmark_barycentric, dtype=np.float64).reshape(-1, 3)
    return face_vertices, barycentric


def _build_opening_delta_landmarks(
    pairs: np.ndarray,
    faces: np.ndarray,
    landmark_face_indices: np.ndarray,
    landmark_barycentric: np.ndarray,
    target_lmks478: np.ndarray,
    *,
    weight: float,
    pair_name: str,
) -> dict[str, np.ndarray]:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pairs.size == 0:
        return {}
    ok = (
        (pairs >= 0).all(axis=1)
        & (pairs[:, 0] < landmark_face_indices.shape[0])
        & (pairs[:, 1] < landmark_face_indices.shape[0])
    )
    pairs = pairs[ok]
    if pairs.size == 0:
        return {}
    idx_a = pairs[:, 0]
    idx_b = pairs[:, 1]
    fva, bca = _landmark_bary_on_faces(faces, landmark_face_indices[idx_a], landmark_barycentric[idx_a])
    fvb, bcb = _landmark_bary_on_faces(faces, landmark_face_indices[idx_b], landmark_barycentric[idx_b])
    tgt_delta = (
        np.asarray(target_lmks478, dtype=np.float64)[idx_a]
        - np.asarray(target_lmks478, dtype=np.float64)[idx_b]
    )
    n_pairs = pairs.shape[0]
    return {
        "delta_face_vertices_a": fva,
        "delta_barycentric_a": bca,
        "delta_face_vertices_b": fvb,
        "delta_barycentric_b": bcb,
        "delta_target_deltas": tgt_delta,
        "delta_weights": np.full(n_pairs, float(weight), dtype=np.float64),
        "delta_pairs": pairs,
        "delta_pair_names": np.array([pair_name] * n_pairs, dtype=object),
    }


def build_nonrigid_landmarks_from_alignment(
    source_faces: np.ndarray,
    hack_lmks478_bary_path: str | os.PathLike[str],
    target_lmks478: np.ndarray,
    *,
    eye_weight: float = 15.0,
    lip_weight: float = 15.0,
    eye_opening_weight: float = 40.0,
    mouth_opening_weight: float = 40.0,
) -> dict[str, np.ndarray]:
    """Barycentric HACK landmarks and opening-pair deltas in the aligned target frame."""
    landmark_face_indices, landmark_barycentric = load_hack_lmks478_bary(hack_lmks478_bary_path)
    face_vertices, barycentric = _landmark_bary_on_faces(
        source_faces, landmark_face_indices, landmark_barycentric
    )
    target_points = np.asarray(target_lmks478, dtype=np.float64).reshape(-1, 3)
    if target_points.shape[0] != face_vertices.shape[0]:
        raise ValueError(
            f"target_lmks478 rows {target_points.shape[0]} != landmark count {face_vertices.shape[0]}"
        )
    weights = np.ones(target_points.shape[0], dtype=np.float64)
    eye_ids = np.asarray(EYE_LANDMARK_INDICES, dtype=np.int64)
    lip_ids = np.asarray(LIP_LANDMARK_INDICES, dtype=np.int64)
    weights[np.isin(np.arange(target_points.shape[0]), eye_ids)] = float(eye_weight)
    weights[np.isin(np.arange(target_points.shape[0]), lip_ids)] = float(lip_weight)

    landmarks: dict[str, np.ndarray] = {
        "face_vertices": face_vertices,
        "barycentric": barycentric,
        "target_points": target_points,
        "weights": weights,
        "landmark_indices": np.arange(target_points.shape[0], dtype=np.int64),
        "eye_landmark_indices": eye_ids,
        "lip_landmark_indices": lip_ids,
        "num_hack_landmarks": int(target_points.shape[0]),
    }

    eye_delta = _build_opening_delta_landmarks(
        EYELID_OPENING_LANDMARK_PAIRS,
        source_faces,
        landmark_face_indices,
        landmark_barycentric,
        target_points,
        weight=eye_opening_weight,
        pair_name="eye",
    )
    mouth_delta = _build_opening_delta_landmarks(
        MOUTH_OPENING_LANDMARK_PAIRS,
        source_faces,
        landmark_face_indices,
        landmark_barycentric,
        target_points,
        weight=mouth_opening_weight,
        pair_name="mouth",
    )
    if eye_delta and mouth_delta:
        landmarks.update(
            {
                "delta_face_vertices_a": np.concatenate(
                    [eye_delta["delta_face_vertices_a"], mouth_delta["delta_face_vertices_a"]]
                ),
                "delta_barycentric_a": np.concatenate(
                    [eye_delta["delta_barycentric_a"], mouth_delta["delta_barycentric_a"]]
                ),
                "delta_face_vertices_b": np.concatenate(
                    [eye_delta["delta_face_vertices_b"], mouth_delta["delta_face_vertices_b"]]
                ),
                "delta_barycentric_b": np.concatenate(
                    [eye_delta["delta_barycentric_b"], mouth_delta["delta_barycentric_b"]]
                ),
                "delta_target_deltas": np.concatenate(
                    [eye_delta["delta_target_deltas"], mouth_delta["delta_target_deltas"]]
                ),
                "delta_weights": np.concatenate(
                    [eye_delta["delta_weights"], mouth_delta["delta_weights"]]
                ),
                "delta_pairs": np.concatenate(
                    [eye_delta["delta_pairs"], mouth_delta["delta_pairs"]]
                ),
                "delta_pair_names": np.concatenate(
                    [eye_delta["delta_pair_names"], mouth_delta["delta_pair_names"]]
                ),
            }
        )
    elif eye_delta:
        landmarks.update(eye_delta)
    elif mouth_delta:
        landmarks.update(mouth_delta)
    return landmarks


def build_auto_registration_controls(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    cavity_pkl_path: Optional[str | os.PathLike[str]] = None,
    *,
    frown_detail_data_weight: float = 2.0,
    frown_query_weight_boost: float = 2.5,
    frown_detail_vertex_mass_multiplier: float = 0.8,
    frown_detail_edge_vector_multiplier: float = 0.0,
    boundary_width_scale: float = 10.0,
    outside_data_weight: float = 0.0,
    outside_affine_prior_weight: float = 12.0,
    outside_shape_weight: float = 8.0,
    outside_position_weight: float = 2.5,
    outside_smoothness_multiplier: float = 5.0,
    outside_edge_vector_multiplier: float = 4.0,
    boundary_smoothness_multiplier: float = 6.0,
    boundary_edge_vector_multiplier: float = 4.0,
    face_data_weight: float = 1.0,
    face_shape_weight: float = 1.0,
    face_position_weight: float = 1.0,
    face_smoothness_multiplier: float = 1.0,
    face_edge_vector_multiplier: float = 1.0,
    face_flex_shape_weight: float = 0.5,
    face_flex_smoothness_multiplier: float = 0.7,
    face_flex_edge_vector_multiplier: float = 0.7,
    frown_detail_position_weight: float = 0.25,
    frown_detail_shape_weight: float = 0.25,
    frown_detail_smoothness_multiplier: float = 0.3,
):
    """
    Minimal automatic controls for robust_auto fitting.
    """
    source_vertices = np.asarray(source_vertices, dtype=np.float64)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    n_vertices = int(source_vertices.shape[0])
    edges = extract_edges(source_faces)
    edge_scale = float(median_edge_length(source_vertices, source_faces))

    face_mask, face_info = build_seg28_face_vertex_mask(
        n_vertices,
        source_faces,
        seg28_pkl_path,
        region_names=HACK_SEG28_FACE_REGION_NAMES,
    )
    nonface_mask = ~np.asarray(face_mask, dtype=bool)
    forehead_mask, _ = build_seg28_face_vertex_mask(
        n_vertices, source_faces, seg28_pkl_path, region_names=HACK_FOREHEAD_REGION_NAMES
    )
    frown_mask, frown_info = build_seg28_face_vertex_mask(
        n_vertices, source_faces, seg28_pkl_path, region_names=HACK_FROWN_REGION_NAMES
    )

    data_vertex_weights = np.where(face_mask, float(face_data_weight), float(outside_data_weight))
    affine_prior_vertex_weights = np.where(
        face_mask, 1.0, float(outside_affine_prior_weight)
    ).astype(np.float64)
    shape_prior_vertex_weights = np.where(
        face_mask, float(face_shape_weight), float(outside_shape_weight)
    ).astype(np.float64)
    position_prior_vertex_weights = np.where(
        face_mask, float(face_position_weight), float(outside_position_weight)
    ).astype(np.float64)
    vertex_mass_vertex_weights = shape_prior_vertex_weights.copy()

    boundary_seed = face_boundary_seed(source_faces, face_mask)
    max_boundary_dist = max(float(boundary_width_scale) * edge_scale, edge_scale)
    boundary_dist = geodesic_distance_from_mask(
        source_vertices,
        source_faces,
        boundary_seed,
        max_distance=max_boundary_dist,
        allowed_mask=face_mask,
    )
    face_boundary_band = np.zeros(n_vertices, dtype=np.float64)
    boundary_shell = face_mask & np.isfinite(boundary_dist)
    face_boundary_band[boundary_shell] = np.clip(
        boundary_dist[boundary_shell] / max(max_boundary_dist, 1e-12),
        0.0,
        1.0,
    )
    interior_flex_band = face_boundary_band * np.asarray(face_mask, dtype=np.float64)
    face_flex_enabled = bool(np.any(interior_flex_band > 0.25))
    if face_flex_enabled:
        flex = interior_flex_band
        shape_prior_vertex_weights[:] *= 1.0 - (1.0 - float(face_flex_shape_weight)) * flex
        position_prior_vertex_weights[:] *= 1.0 - (1.0 - float(face_position_weight)) * flex

    smoothness_vertex_mult = np.where(
        nonface_mask,
        float(outside_smoothness_multiplier),
        float(face_smoothness_multiplier),
    ).astype(np.float64)
    edge_vector_vertex_mult = np.where(
        nonface_mask,
        float(outside_edge_vector_multiplier),
        float(face_edge_vector_multiplier),
    ).astype(np.float64)
    boundary_ring = face_mask & (face_boundary_band > 0.05) & (face_boundary_band < 0.95)
    smoothness_vertex_mult[boundary_ring] = np.maximum(
        smoothness_vertex_mult[boundary_ring],
        float(boundary_smoothness_multiplier),
    )
    edge_vector_vertex_mult[boundary_ring] = np.maximum(
        edge_vector_vertex_mult[boundary_ring],
        float(boundary_edge_vector_multiplier),
    )
    if face_flex_enabled:
        smoothness_vertex_mult[:] *= 1.0 - (
            1.0 - float(face_flex_smoothness_multiplier)
        ) * interior_flex_band
        edge_vector_vertex_mult[:] *= 1.0 - (
            1.0 - float(face_flex_edge_vector_multiplier)
        ) * interior_flex_band

    smoothness_edge_weights = _edge_band_from_vertex_band(edges, smoothness_vertex_mult)
    edge_vector_edge_weights = _edge_band_from_vertex_band(edges, edge_vector_vertex_mult)

    correspondence_exclude_vertex_mask = None
    cavity_info: dict[str, object] = {}
    if cavity_pkl_path is not None:
        cavity_mask, cavity_info = build_hack_cavity_vertex_mask(n_vertices, cavity_pkl_path)
        correspondence_exclude_vertex_mask = np.asarray(cavity_mask, dtype=bool)

    frown_detail_flex_enabled = bool(np.any(frown_mask))
    frown_data_peak = float(frown_detail_data_weight) * float(frown_query_weight_boost)
    if frown_detail_flex_enabled:
        data_vertex_weights[frown_mask] = np.maximum(
            data_vertex_weights[frown_mask], frown_data_peak
        )
        frown_flex = np.asarray(frown_mask, dtype=np.float64)
        position_prior_vertex_weights[frown_mask] = (
            (1.0 - frown_flex[frown_mask]) * position_prior_vertex_weights[frown_mask]
            + frown_flex[frown_mask] * float(frown_detail_position_weight)
        )
        shape_prior_vertex_weights[frown_mask] *= float(frown_detail_shape_weight)
        vertex_mass_vertex_weights[frown_mask] *= float(frown_detail_vertex_mass_multiplier)
        smoothness_edge_weights *= _edge_band_from_vertex_band(
            edges,
            1.0
            - (1.0 - float(frown_detail_smoothness_multiplier)) * frown_flex,
        )
        if float(frown_detail_edge_vector_multiplier) > 0.0:
            edge_vector_edge_weights *= _edge_band_from_vertex_band(
                edges,
                1.0
                - (1.0 - float(frown_detail_edge_vector_multiplier)) * frown_flex,
            )

    darap_eye_edge_band_scale = 2.0
    eye_edge_mask = np.zeros(n_vertices, dtype=bool)
    for region_name in HACK_DARAP_EYE_EDGE_REGION_NAMES:
        region_mask, _ = build_seg28_face_vertex_mask(
            n_vertices,
            source_faces,
            seg28_pkl_path,
            region_names=(region_name,),
        )
        eye_edge_mask |= build_region_boundary_ring_vertex_mask(
            source_vertices,
            source_faces,
            region_mask,
            band_width_scale=darap_eye_edge_band_scale,
        )
    jaw_chin_temple_mask, _ = build_seg28_face_vertex_mask(
        n_vertices,
        source_faces,
        seg28_pkl_path,
        region_names=HACK_DARAP_SMOOTH_AUX_REGION_NAMES,
    )

    prior_region_info: dict[str, object] = {
        "mode": "robust_auto",
        "outside_data_weight": float(outside_data_weight),
        "outside_affine_prior_weight": float(outside_affine_prior_weight),
        "outside_shape_weight": float(outside_shape_weight),
        "outside_position_weight": float(outside_position_weight),
        "outside_smoothness_multiplier": float(outside_smoothness_multiplier),
        "outside_edge_vector_multiplier": float(outside_edge_vector_multiplier),
        "boundary_smoothness_multiplier": float(boundary_smoothness_multiplier),
        "boundary_edge_vector_multiplier": float(boundary_edge_vector_multiplier),
        "boundary_width": float(max_boundary_dist),
        "num_boundary_band_vertices": int(np.count_nonzero(boundary_ring)),
        "num_nonface_protected_vertices": int(np.count_nonzero(nonface_mask)),
        "face": face_info,
        "cavity": cavity_info,
        "face_flex_enabled": face_flex_enabled,
        "face_data_weight": float(face_data_weight),
        "face_shape_weight": float(face_flex_shape_weight),
        "face_smoothness_multiplier": float(face_flex_smoothness_multiplier),
        "face_edge_vector_multiplier": float(face_flex_edge_vector_multiplier),
        "num_face_flex_vertices": int(np.count_nonzero(interior_flex_band > 0.25)),
        "frown_detail": frown_info,
        "frown_detail_flex_enabled": frown_detail_flex_enabled,
        "frown_detail_data_weight": float(frown_detail_data_weight),
        "frown_detail_position_weight": float(frown_detail_position_weight),
        "frown_detail_shape_weight": float(frown_detail_shape_weight),
        "frown_detail_smoothness_multiplier": float(frown_detail_smoothness_multiplier),
        "frown_detail_edge_vector_multiplier": float(frown_detail_edge_vector_multiplier),
        "frown_detail_vertex_mass_multiplier": float(frown_detail_vertex_mass_multiplier),
        "num_frown_detail_vertices": int(np.count_nonzero(frown_mask)),
        "num_frown_detail_flex_vertices": int(np.count_nonzero(frown_mask)),
        "frown_data_weight_peak": frown_data_peak,
        "nose_flex_enabled": False,
        "num_face_vertices": int(np.count_nonzero(face_mask)),
        "num_excluded_vertices": int(
            np.count_nonzero(correspondence_exclude_vertex_mask)
            if correspondence_exclude_vertex_mask is not None
            else 0
        ),
        "num_nonface_vertices": int(np.count_nonzero(nonface_mask)),
        "data_vertex_weight_mean": float(np.mean(data_vertex_weights)),
        "position_prior_vertex_weight_mean": float(np.mean(position_prior_vertex_weights)),
        "shape_prior_vertex_weight_mean": float(np.mean(shape_prior_vertex_weights)),
        "smoothness_edge_weight_mean": float(np.mean(smoothness_edge_weights)),
        "edge_vector_edge_weight_mean": float(np.mean(edge_vector_edge_weights)),
        "darap_eye_edge_band_scale": darap_eye_edge_band_scale,
        "_face_vertex_mask": face_mask,
        "_nonface_vertex_mask": nonface_mask,
        "_forehead_vertex_mask": forehead_mask,
        "_frown_vertex_mask": frown_mask,
        "_darap_smooth_aux_vertex_mask": jaw_chin_temple_mask,
        "_darap_eye_edge_vertex_mask": eye_edge_mask,
    }

    position_prior_axis_weights = None
    return (
        affine_prior_vertex_weights,
        data_vertex_weights,
        position_prior_vertex_weights,
        position_prior_axis_weights,
        shape_prior_vertex_weights,
        vertex_mass_vertex_weights,
        smoothness_edge_weights,
        edge_vector_edge_weights,
        face_boundary_band,
        correspondence_exclude_vertex_mask,
        prior_region_info,
    )


def build_regional_darap_face_interior_vertex_indices(
    n_vertices: int,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str] | None,
    *,
    face_vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Vertex indices in HACK seg28 face regions (same mask as face-interior smoothing)."""
    if face_vertex_mask is not None:
        return np.flatnonzero(np.asarray(face_vertex_mask, dtype=bool)).astype(np.int64, copy=False)
    if seg28_pkl_path is None:
        return np.zeros(0, dtype=np.int64)
    face_mask, _ = build_seg28_face_vertex_mask(
        n_vertices,
        faces,
        seg28_pkl_path,
        region_names=HACK_SEG28_FACE_REGION_NAMES,
    )
    return np.flatnonzero(face_mask).astype(np.int64, copy=False)


def build_regional_darap_forehead_vertex_indices(
    n_vertices: int,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str] | None,
    *,
    forehead_vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Forehead-only vertices (overrides face-interior lambda where they overlap)."""
    if forehead_vertex_mask is not None:
        return np.flatnonzero(np.asarray(forehead_vertex_mask, dtype=bool)).astype(
            np.int64, copy=False
        )
    if seg28_pkl_path is None:
        return np.zeros(0, dtype=np.int64)
    forehead_mask, _ = build_seg28_face_vertex_mask(
        n_vertices,
        faces,
        seg28_pkl_path,
        region_names=HACK_FOREHEAD_REGION_NAMES,
    )
    return np.flatnonzero(forehead_mask).astype(np.int64, copy=False)


def build_regional_darap_frown_vertex_indices(
    n_vertices: int,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str] | None,
    *,
    frown_vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    """HACK seg28 ``frown_0`` glabella detail region."""
    if frown_vertex_mask is not None:
        return np.flatnonzero(np.asarray(frown_vertex_mask, dtype=bool)).astype(
            np.int64, copy=False
        )
    if seg28_pkl_path is None:
        return np.zeros(0, dtype=np.int64)
    frown_mask, _ = build_seg28_face_vertex_mask(
        n_vertices,
        faces,
        seg28_pkl_path,
        region_names=HACK_FROWN_REGION_NAMES,
    )
    return np.flatnonzero(frown_mask).astype(np.int64, copy=False)


def build_regional_darap_jaw_chin_temple_vertex_indices(
    n_vertices: int,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str] | None,
    *,
    jaw_chin_temple_vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    """chin_1, jaw_0/1, temple_0/1 — lower dARAP lambda for smoother sides/chin."""
    if jaw_chin_temple_vertex_mask is not None:
        return np.flatnonzero(np.asarray(jaw_chin_temple_vertex_mask, dtype=bool)).astype(
            np.int64, copy=False
        )
    if seg28_pkl_path is None:
        return np.zeros(0, dtype=np.int64)
    aux_mask, _ = build_seg28_face_vertex_mask(
        n_vertices,
        faces,
        seg28_pkl_path,
        region_names=HACK_DARAP_SMOOTH_AUX_REGION_NAMES,
    )
    return np.flatnonzero(aux_mask).astype(np.int64, copy=False)


def build_regional_darap_eye_edge_vertex_indices(
    vertices: np.ndarray,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str] | None,
    *,
    band_width_scale: float = 2.0,
    eye_edge_vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    """eye_0 / eye_3 boundary rings — lower dARAP lambda at periorbital rims."""
    if eye_edge_vertex_mask is not None:
        return np.flatnonzero(np.asarray(eye_edge_vertex_mask, dtype=bool)).astype(
            np.int64, copy=False
        )
    if seg28_pkl_path is None:
        return np.zeros(0, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    ring = np.zeros(vertices.shape[0], dtype=bool)
    for name in HACK_DARAP_EYE_EDGE_REGION_NAMES:
        region_mask, _ = build_seg28_face_vertex_mask(
            vertices.shape[0],
            faces,
            seg28_pkl_path,
            region_names=(name,),
        )
        ring |= build_region_boundary_ring_vertex_mask(
            vertices,
            faces,
            region_mask,
            band_width_scale=band_width_scale,
        )
    return np.flatnonzero(ring).astype(np.int64, copy=False)


def build_regional_darap_face_boundary_vertex_indices(
    vertices: np.ndarray,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str] | None,
    *,
    band_width_scale: float = 2.0,
    face_vertex_mask: np.ndarray | None = None,
    face_boundary_band: np.ndarray | None = None,
    boundary_width: float | None = None,
) -> np.ndarray:
    """Face vertices near the face/non-face seam (lower dARAP lambda, applied last)."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if face_vertex_mask is not None and face_boundary_band is not None:
        face_mask = np.asarray(face_vertex_mask, dtype=bool).reshape(-1)
        band = np.asarray(face_boundary_band, dtype=np.float64).reshape(-1)
        if face_mask.shape[0] != band.shape[0]:
            raise ValueError("face_vertex_mask and face_boundary_band length mismatch")
        edge_scale = float(median_edge_length(vertices, faces))
        band_dist = float(band_width_scale) * max(edge_scale, 1e-12)
        width = float(boundary_width) if boundary_width is not None else 10.0 * edge_scale
        width = max(width, band_dist + 1e-12)
        thresh = 1.0 - band_dist / width
        boundary_ring = face_mask & (band >= max(thresh, 0.0))
        return np.flatnonzero(boundary_ring).astype(np.int64, copy=False)
    if seg28_pkl_path is None:
        return np.zeros(0, dtype=np.int64)
    face_mask, _ = build_seg28_face_vertex_mask(
        vertices.shape[0],
        faces,
        seg28_pkl_path,
        region_names=HACK_SEG28_FACE_REGION_NAMES,
    )
    if not np.any(face_mask):
        return np.zeros(0, dtype=np.int64)
    boundary_seed = face_boundary_seed(faces, face_mask)
    edge_scale = float(median_edge_length(vertices, faces))
    band_scale = float(band_width_scale)
    max_dist = max(band_scale * max(edge_scale, 1e-12), edge_scale)
    dist = geodesic_distance_from_mask(
        vertices,
        faces,
        boundary_seed,
        max_distance=max_dist,
    )
    boundary_ring = (
        face_mask
        & np.isfinite(dist)
        & (dist <= band_scale * max(edge_scale, 1e-12))
    )
    return np.flatnonzero(boundary_ring).astype(np.int64, copy=False)


def _regional_darap_camera_depth_attenuation_enabled(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "no_regional_darap_camera_depth_attenuation", False)):
        return False
    return bool(getattr(args, "regional_darap_camera_depth_attenuation", True))


def _regional_darap_camera_depth_params(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "depth_axis": int(
            getattr(
                args,
                "regional_darap_camera_depth_axis",
                getattr(args, "icp_camera_depth_axis", 2),
            )
        ),
        "min_weight": float(
            getattr(
                args,
                "regional_darap_camera_depth_min_weight",
                getattr(args, "icp_camera_depth_min_weight", 0.35),
            )
        ),
        "power": float(
            getattr(
                args,
                "regional_darap_camera_depth_power",
                getattr(args, "icp_camera_depth_power", 2.0),
            )
        ),
        "far_percentile": float(
            getattr(
                args,
                "regional_darap_camera_depth_far_percentile",
                getattr(args, "icp_camera_depth_far_percentile", 95.0),
            )
        ),
    }


def modulate_regional_darap_lambda_by_camera_depth(
    local_lambda: torch.Tensor,
    vertices: np.ndarray,
    args: argparse.Namespace,
    *,
    reference_vertex_mask: np.ndarray | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Scale per-vertex Procrustes lambda down for vertices far from the camera plane."""
    stats: dict[str, float] = {"darap_camera_depth_enabled": 0.0}
    if not _regional_darap_camera_depth_attenuation_enabled(args):
        return local_lambda, stats
    params = _regional_darap_camera_depth_params(args)
    depth_w, depth_stats = camera_depth_correspondence_weights(
        vertices,
        depth_axis=int(params["depth_axis"]),
        reference_vertex_mask=reference_vertex_mask,
        min_weight=float(params["min_weight"]),
        power=float(params["power"]),
        far_percentile=float(params["far_percentile"]),
    )
    depth_t = torch.as_tensor(depth_w, dtype=local_lambda.dtype, device=local_lambda.device)
    modulated = local_lambda * depth_t
    ref_mask = (
        np.asarray(reference_vertex_mask, dtype=bool).reshape(-1)
        if reference_vertex_mask is not None
        else np.ones(vertices.shape[0], dtype=bool)
    )
    on_ref = ref_mask & (depth_w < 0.999)
    stats = {
        "darap_camera_depth_enabled": 1.0,
        "darap_camera_depth_reference": float(
            depth_stats.get("camera_depth_reference", float("nan"))
        ),
        "darap_camera_depth_weight_mean": float(np.mean(depth_w[ref_mask])) if np.any(ref_mask) else 1.0,
        "darap_camera_depth_weight_min": float(np.min(depth_w[on_ref])) if np.any(on_ref) else 1.0,
        "darap_camera_depth_atten_vertices": float(np.count_nonzero(on_ref)),
        "darap_lambda_mean_before": float(local_lambda.detach().cpu().numpy().mean()),
        "darap_lambda_mean_after": float(modulated.detach().cpu().numpy().mean()),
    }
    return modulated, stats


def apply_regional_darap_normal_deform(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_normals: np.ndarray,
    face_interior_vertex_indices: np.ndarray | Sequence[int],
    args: argparse.Namespace,
    *,
    forehead_vertex_indices: np.ndarray | Sequence[int] | None = None,
    frown_vertex_indices: np.ndarray | Sequence[int] | None = None,
    jaw_chin_temple_vertex_indices: np.ndarray | Sequence[int] | None = None,
    eye_edge_vertex_indices: np.ndarray | Sequence[int] | None = None,
    face_boundary_vertex_indices: np.ndarray | Sequence[int] | None = None,
    face_vertex_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Same steps as demo_deform_dARAP.py: regional Procrustes lambda, ARAP global solve
    on ``vertices``, return deformed positions directly.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    target_normals = np.asarray(target_normals, dtype=np.float64)
    if vertices.shape[0] != target_normals.shape[0]:
        raise ValueError(
            f"vertices ({vertices.shape[0]}) and target_normals ({target_normals.shape[0]}) "
            "must have the same length"
        )
    stats: dict[str, float] = {
        "enabled": 0.0,
        "n_vertices": float(vertices.shape[0]),
        "face_interior_vertices": 0.0,
        "max_displacement": 0.0,
        "lambda_default": float(getattr(args, "regional_darap_lambda_default", 3.0)),
        "lambda_face_interior": float(getattr(args, "regional_darap_lambda_face_interior", 6.5)),
        "lambda_forehead": float(getattr(args, "regional_darap_lambda_forehead", 1.0)),
        "lambda_frown_0": float(
            getattr(
                args,
                "regional_darap_lambda_frown_0",
                getattr(args, "regional_darap_lambda_frown", 8.5),
            )
        ),
        "lambda_jaw_chin_temple": float(
            getattr(args, "regional_darap_lambda_jaw_chin_temple", 1.0)
        ),
        "lambda_eye_edge": float(getattr(args, "regional_darap_lambda_eye_edge", 1.0)),
        "lambda_face_boundary": float(getattr(args, "regional_darap_lambda_face_boundary", 1.0)),
        "forehead_vertices": 0.0,
        "frown_vertices": 0.0,
        "jaw_chin_temple_vertices": 0.0,
        "eye_edge_vertices": 0.0,
        "face_boundary_vertices": 0.0,
    }
    if bool(getattr(args, "no_regional_darap_normal_deform", False)):
        return vertices, stats

    face_indices = np.asarray(face_interior_vertex_indices, dtype=np.int64).reshape(-1)
    forehead_indices = (
        np.asarray(forehead_vertex_indices, dtype=np.int64).reshape(-1)
        if forehead_vertex_indices is not None
        else np.zeros(0, dtype=np.int64)
    )
    frown_indices = (
        np.asarray(frown_vertex_indices, dtype=np.int64).reshape(-1)
        if frown_vertex_indices is not None
        else np.zeros(0, dtype=np.int64)
    )
    jaw_chin_temple_indices = (
        np.asarray(jaw_chin_temple_vertex_indices, dtype=np.int64).reshape(-1)
        if jaw_chin_temple_vertex_indices is not None
        else np.zeros(0, dtype=np.int64)
    )
    eye_edge_indices = (
        np.asarray(eye_edge_vertex_indices, dtype=np.int64).reshape(-1)
        if eye_edge_vertex_indices is not None
        else np.zeros(0, dtype=np.int64)
    )
    boundary_indices = (
        np.asarray(face_boundary_vertex_indices, dtype=np.int64).reshape(-1)
        if face_boundary_vertex_indices is not None
        else np.zeros(0, dtype=np.int64)
    )
    stats["face_interior_vertices"] = float(face_indices.size)
    stats["forehead_vertices"] = float(forehead_indices.size)
    stats["frown_vertices"] = float(frown_indices.size)
    stats["jaw_chin_temple_vertices"] = float(jaw_chin_temple_indices.size)
    stats["eye_edge_vertices"] = float(eye_edge_indices.size)
    stats["face_boundary_vertices"] = float(boundary_indices.size)
    stats["enabled"] = 1.0

    lambda_default = float(getattr(args, "regional_darap_lambda_default", 3.0))
    lambda_face = float(getattr(args, "regional_darap_lambda_face_interior", 6.5))
    lambda_forehead = float(getattr(args, "regional_darap_lambda_forehead", 1.0))
    lambda_frown = float(
        getattr(
            args,
            "regional_darap_lambda_frown_0",
            getattr(args, "regional_darap_lambda_frown", 8.5),
        )
    )
    lambda_jaw_chin_temple = float(
        getattr(args, "regional_darap_lambda_jaw_chin_temple", 1.0)
    )
    lambda_eye_edge = float(getattr(args, "regional_darap_lambda_eye_edge", 1.0))
    lambda_face_boundary = float(getattr(args, "regional_darap_lambda_face_boundary", 1.0))
    pin_first = bool(getattr(args, "regional_darap_pin_first_vertex", True))
    arap_energy_type = str(getattr(args, "regional_darap_arap_energy_type", "spokes_and_rims_mine"))
    postprocess_mode = str(getattr(args, "regional_darap_postprocess", "recenter_only"))

    verts_t = torch.tensor(vertices, dtype=torch.float32)
    faces_t = torch.tensor(faces, dtype=torch.long)
    target_normals_t = torch.tensor(target_normals, dtype=torch.float32)
    verts_list = [verts_t]
    faces_list = [faces_t]
    target_normals_list = [target_normals_t]

    region_vertex_indices: list[list[int]] = [face_indices.tolist()]
    region_lambdas: list[float] = [lambda_face]
    if forehead_indices.size > 0:
        region_vertex_indices.append(forehead_indices.tolist())
        region_lambdas.append(lambda_forehead)
    if jaw_chin_temple_indices.size > 0:
        region_vertex_indices.append(jaw_chin_temple_indices.tolist())
        region_lambdas.append(lambda_jaw_chin_temple)
    if eye_edge_indices.size > 0:
        region_vertex_indices.append(eye_edge_indices.tolist())
        region_lambdas.append(lambda_eye_edge)
    if boundary_indices.size > 0:
        region_vertex_indices.append(boundary_indices.tolist())
        region_lambdas.append(lambda_face_boundary)
    if frown_indices.size > 0:
        region_vertex_indices.append(frown_indices.tolist())
        region_lambdas.append(lambda_frown)
    local_lambda = vertex_procrustes_lambda_from_regions(
        vertices.shape[0],
        default_lambda=lambda_default,
        region_vertex_indices=region_vertex_indices,
        region_lambdas=region_lambdas,
    )
    local_lambda_pre_depth = local_lambda.clone()
    local_lambda, depth_lambda_stats = modulate_regional_darap_lambda_by_camera_depth(
        local_lambda,
        vertices,
        args,
        reference_vertex_mask=face_vertex_mask,
    )
    stats.update(depth_lambda_stats)
    if (
        frown_indices.size > 0
        and bool(getattr(args, "regional_darap_frown_0_skip_camera_depth_attenuation", True))
    ):
        local_lambda[frown_indices] = local_lambda_pre_depth[frown_indices]
        stats["frown_0_skip_camera_depth_attenuation"] = 1.0

    solvers = SparseLaplaciansSolvers.from_meshes(
        verts_list,
        faces_list,
        pin_first_vertex=pin_first,
        compute_poisson_rhs_lefts=False,
        compute_igl_arap_rhs_lefts=None,
    )
    procrustes_precompute = ProcrustesPrecompute.from_meshes(
        local_step_procrustes_lambda=local_lambda,
        arap_energy_type=arap_energy_type,
        laplacians_solvers=solvers,
        verts_list=verts_list,
        faces_list=faces_list,
    )
    per_vertex_3x3_packed = calc_rot_matrices_with_procrustes(
        procrustes_precompute,
        torch.cat(verts_list),
        torch.cat(target_normals_list),
    )
    per_vertex_3x3_list = torch.split(
        per_vertex_3x3_packed, [int(v.shape[0]) for v in verts_list]
    )
    deformed_list = calc_ARAP_global_solve(
        verts_list,
        faces_list,
        solvers,
        per_vertex_3x3_list,
        arap_energy_type=arap_energy_type,
        postprocess=postprocess_mode,
    )
    deformed = deformed_list[0].detach().cpu().numpy().astype(np.float64, copy=False)
    stats["max_displacement"] = float(np.max(np.linalg.norm(deformed - vertices, axis=1)))
    return deformed, stats



def run_single_example(args: argparse.Namespace) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = osp.abspath(args.raw_dir or repo_root / "test" / "phack_raw")
    target_dir = osp.abspath(args.target_dir or repo_root / "test" / "bni")
    bni_lmks_dir = osp.abspath(args.bni_lmks_dir or repo_root / "test" / "bni_lmks")
    bni_seg_dir = osp.abspath(args.bni_seg_dir or repo_root / "test" / "bni_seg")
    hack_lmks478_bary_path = osp.abspath(
        args.hack_lmks478_bary_path or repo_root / "dataset" / "hack_lmks478_bary.pkl"
    )
    seg28_pkl_path = osp.abspath(args.seg28_pkl_path or repo_root / "data" / "hack_data" / "HACK_seg_28.pkl")
    cavity_pkl_path = osp.abspath(args.hack_cavity_pkl_path or repo_root / "dataset" / "HACK_cavity.pkl")
    output_dir = osp.abspath(args.output_dir or repo_root / "test" / "nonrigid_icp")

    stem, raw_path, target_path = resolve_pair(
        stem=args.stem,
        raw_path=args.raw_path,
        target_path=args.target_path,
        raw_dir=raw_dir,
        target_dir=target_dir,
    )
    out_path = osp.abspath(args.output_path or osp.join(output_dir, f"{stem}_nicp.obj"))
    hist_path = osp.splitext(out_path)[0] + "_history.json"

    src_v, src_f = read_phack_mesh(raw_path)
    tgt_v, tgt_f = read_target_mesh(target_path)
    target_face_mask = None
    target_face_mask_info: dict[str, object] = {
        "enabled": False,
        "bni_seg_path": None,
        "target_query_faces": int(tgt_f.shape[0]),
        "target_total_faces": int(tgt_f.shape[0]),
    }
    bni_seg_path = osp.abspath(args.bni_seg_path or osp.join(bni_seg_dir, f"{stem}_bni_seg.npz"))
    use_target_face_mask = not bool(args.no_target_face_mask)
    if use_target_face_mask:
        if osp.isfile(bni_seg_path):
            target_face_mask, target_face_mask_info = load_bni_target_face_mask(bni_seg_path, tgt_f.shape[0])
            target_face_mask_info["enabled"] = True
            print(
                f"[example] target correspondence face mask: "
                f"faces={target_face_mask_info['target_query_faces']}/{target_face_mask_info['target_total_faces']} "
                f"regions={len(target_face_mask_info['region_names'])} "
                f"path={bni_seg_path}",
                flush=True,
            )
        else:
            target_face_mask_info["bni_seg_path"] = bni_seg_path
            print(
                f"[example] target correspondence face mask unavailable; using full target: {bni_seg_path}",
                flush=True,
            )
    else:
        target_face_mask_info["bni_seg_path"] = bni_seg_path
        if bool(args.no_target_face_mask) or not bool(args.target_face_mask):
            print("[example] target correspondence face mask disabled; using full target", flush=True)
    lmks_path = osp.abspath(args.bni_lmks_path or osp.join(bni_lmks_dir, f"{stem}_bni_lmks.npz"))
    tgt_v, align_info = align_target_mesh_to_hack_by_landmarks(
        tgt_v,
        src_v,
        src_f,
        lmks_path,
        hack_lmks478_bary_path,
    )
    print(
        f"[example] target aligned to HACK by landmarks: "
        f"scale={align_info['scale']:.6f} "
        f"lmk_rmse={align_info['lmk_rmse']:.6f} "
        f"pairs={align_info['num_lmks_used']}/{len(HACK_FITTING_INDICES)}",
        flush=True,
    )
    config = config_from_args(args)
    (
        affine_prior_vertex_weights,
        data_vertex_weights,
        position_prior_vertex_weights,
        position_prior_axis_weights,
        shape_prior_vertex_weights,
        vertex_mass_vertex_weights,
        smoothness_edge_weights,
        edge_vector_edge_weights,
        face_boundary_band,
        correspondence_exclude_vertex_mask,
        prior_region_info,
    ) = build_auto_registration_controls(
        src_v,
        src_f,
        seg28_pkl_path,
        cavity_pkl_path=cavity_pkl_path,
        frown_detail_data_weight=float(getattr(args, "frown_detail_data_weight", 2.0)),
        frown_query_weight_boost=float(
            getattr(
                args,
                "frown_query_weight_boost",
                getattr(args, "frown_0_query_weight_boost", 2.5),
            )
        ),
        frown_detail_vertex_mass_multiplier=float(
            getattr(args, "frown_detail_vertex_mass_multiplier", 0.0)
        ),
        frown_detail_edge_vector_multiplier=float(
            getattr(args, "frown_detail_edge_vector_multiplier", 0.0)
        ),
    )
    fs_seam_flex_info = apply_forehead_skull_seam_registration_flex(
        src_v,
        src_f,
        seg28_pkl_path,
        position_prior_vertex_weights,
        data_vertex_weights,
        shape_prior_vertex_weights,
        vertex_mass_vertex_weights,
        smoothness_edge_weights,
        edge_vector_edge_weights,
        affine_prior_vertex_weights,
        enable_face_flex=not bool(getattr(args, "no_forehead_skull_face_flex", False)),
        enable_skull_flex=not bool(getattr(args, "no_forehead_skull_nonface_flex", False)),
        enable_vertex_mass_boost=not bool(
            getattr(args, "no_forehead_skull_vertex_mass_boost", False)
        ),
        band_width_scale=float(args.forehead_skull_face_flex_band_scale),
        seam_vertex_mass_multiplier=float(args.forehead_skull_vertex_mass_multiplier),
        face_position_weight=float(args.forehead_skull_face_position_weight),
        face_data_weight_boost=float(args.forehead_skull_face_data_weight_boost),
        face_shape_weight=float(args.forehead_skull_face_shape_weight),
        face_smoothness_multiplier=float(args.forehead_skull_face_smoothness_multiplier),
        face_edge_vector_multiplier=float(args.forehead_skull_face_edge_vector_multiplier),
        skull_position_weight=float(args.forehead_skull_nonface_position_weight),
        skull_affine_prior_scale=float(args.forehead_skull_nonface_affine_prior_scale),
        skull_data_weight_boost=float(args.forehead_skull_nonface_data_weight_boost),
        skull_shape_weight=float(args.forehead_skull_nonface_shape_weight),
        skull_smoothness_multiplier=float(args.forehead_skull_nonface_smoothness_multiplier),
        skull_edge_vector_multiplier=float(args.forehead_skull_nonface_edge_vector_multiplier),
    )
    prior_region_info["forehead_skull_seam_flex"] = fs_seam_flex_info
    prior_region_info["vertex_mass_vertex_weight_mean"] = float(
        np.mean(vertex_mass_vertex_weights)
    )
    forehead_mask_icp, _ = build_seg28_face_vertex_mask(
        src_v.shape[0], src_f, seg28_pkl_path, region_names=HACK_FOREHEAD_REGION_NAMES
    )
    skull_mask_icp, _ = build_seg28_face_vertex_mask(
        src_v.shape[0], src_f, seg28_pkl_path, region_names=HACK_SKULL_REGION_NAMES
    )
    flip_guard_vertex_scales = np.zeros(src_v.shape[0], dtype=np.float64)
    flip_guard_vertex_scales[skull_mask_icp] = 1.0
    print(
        f"[example] robust_auto controls: "
        f"face={prior_region_info['num_face_vertices']} "
        f"excluded={prior_region_info['num_excluded_vertices']} "
        f"data_mean={prior_region_info['data_vertex_weight_mean']:.3f} "
        f"pos_mean={prior_region_info['position_prior_vertex_weight_mean']:.3f} "
        f"shape_mean={prior_region_info['shape_prior_vertex_weight_mean']:.3f} "
        f"face_flex={int(bool(prior_region_info.get('face_flex_enabled', False)))} "
        f"nose_flex={int(bool(prior_region_info.get('nose_flex_enabled', False)))} "
        f"frown_0_verts={prior_region_info.get('num_frown_detail_vertices', 0)} "
        f"frown_data_peak={prior_region_info.get('frown_data_weight_peak', 0.0):.2f} "
        f"frown_mass_mult={prior_region_info.get('frown_detail_vertex_mass_multiplier', 1.0):.2f} "
        f"corr_mode={config.correspondence_mode} "
        f"robust_corr={int(bool(config.robust_correspondence))} "
        f"robust_start_stage={config.robust_correspondence_start_stage} "
        f"normal_angle={config.normal_angle_threshold_degrees} "
        f"stiffness={config.stiffness_schedule} "
        f"adapt_reg={float(config.adaptive_regularization_multiplier):.3f} "
        f"adapt_data={float(config.adaptive_data_boost):.3f} "
        f"lmk_decay={float(config.landmark_weight_decay):.3f} "
        f"final_proj={int(bool(config.final_normal_projection))} "
        f"target_mask={int(bool(target_face_mask_info.get('enabled', False)))} "
        f"forehead_skull_seam_flex={int(fs_seam_flex_info.get('enabled', False))} "
        f"forehead_relax_verts={fs_seam_flex_info.get('num_forehead_relax_vertices', 0)} "
        f"skull_relax_verts={fs_seam_flex_info.get('num_skull_relax_vertices', 0)} "
        f"skull_pos_w={fs_seam_flex_info.get('skull_position_weight', 0.0):.3f} "
        f"mass_mult={fs_seam_flex_info.get('seam_vertex_mass_multiplier', 1.0):.2f} "
        f"mass_mean={prior_region_info.get('vertex_mass_vertex_weight_mean', 0.0):.3f} "
        f"camera_depth_atten={int(bool(config.camera_depth_attenuation_enabled))} "
        f"depth_axis={config.camera_depth_axis} "
        f"depth_min_w={config.camera_depth_min_weight:.2f}",
        flush=True,
    )
    face_mask_icp_ref = prior_region_info.get("_face_vertex_mask")
    if bool(config.camera_depth_attenuation_enabled) and face_mask_icp_ref is not None:
        face_ref = np.asarray(face_mask_icp_ref, dtype=bool)
        depth_preview, depth_preview_stats = camera_depth_correspondence_weights(
            src_v,
            depth_axis=int(config.camera_depth_axis),
            reference_vertex_mask=face_ref,
            min_weight=float(config.camera_depth_min_weight),
            power=float(config.camera_depth_power),
            far_percentile=float(config.camera_depth_far_percentile),
        )
        prior_region_info["camera_depth_protection_enabled"] = True
        prior_region_info["camera_depth_protection_vertices"] = int(
            np.count_nonzero((depth_preview < 0.999) & face_ref)
        )
        prior_region_info["camera_depth_reference"] = depth_preview_stats.get(
            "camera_depth_reference", float("nan")
        )
        prior_region_info["camera_depth_weight_mean_face"] = float(
            np.mean(depth_preview[face_ref])
        )
        prior_region_info["camera_depth_weight_min_face"] = float(
            np.min(depth_preview[face_ref])
        )
        print(
            f"[example] camera-depth correspondence preview (src rest): "
            f"ref_z={prior_region_info['camera_depth_reference']:.4f} "
            f"face_w_mean={prior_region_info['camera_depth_weight_mean_face']:.3f} "
            f"face_w_min={prior_region_info['camera_depth_weight_min_face']:.3f} "
            f"atten_verts={prior_region_info['camera_depth_protection_vertices']}",
            flush=True,
        )
    else:
        prior_region_info["camera_depth_protection_enabled"] = False

    landmarks = None
    if float(config.landmark_weight) > 0.0:
        landmark_kwargs = {
            "eye_weight": 15.0,
            "lip_weight": 15.0,
            "eye_opening_weight": 40.0,
            "mouth_opening_weight": 40.0,
        }
        landmarks = build_nonrigid_landmarks_from_alignment(
            src_f,
            hack_lmks478_bary_path,
            np.asarray(align_info["tgt_lmks_full"], dtype=np.float64),
            **landmark_kwargs,
        )
        print(
            f"[example] nonrigid landmark constraints: "
            f"count={landmarks['target_points'].shape[0]} "
            f"eye={landmarks['eye_landmark_indices'].shape[0]} "
            f"lip={landmarks['lip_landmark_indices'].shape[0]} "
            f"opening_pairs={landmarks.get('delta_pairs', np.empty((0, 2))).shape[0]} "
            f"global_weight={config.landmark_weight:.3f}",
            flush=True,
        )
    # show_mesh_pair(src_v, src_f, tgt_v, tgt_f, use_offset=False)

    print(
        f"[example] stem={stem} source V/F={src_v.shape[0]}/{src_f.shape[0]} "
        f"target V/F={tgt_v.shape[0]}/{tgt_f.shape[0]}",
        flush=True,
    )

    frown_query_pin_ids: np.ndarray | None = None
    frown_mask_icp = prior_region_info.get("_frown_vertex_mask")
    if frown_mask_icp is not None and np.any(frown_mask_icp):
        frown_query_pin_ids = np.flatnonzero(np.asarray(frown_mask_icp, dtype=bool)).astype(
            np.int64
        )
        print(
            f"[example] frown_0 ICP: pin {frown_query_pin_ids.size} source query verts "
            f"(data_peak={prior_region_info.get('frown_data_weight_peak', 0.0):.2f})",
            flush=True,
        )

    face_interior_anchor_vertices: np.ndarray | None = None
    result = nonrigid_icp(
        src_v,
        src_f,
        tgt_v,
        tgt_f,
        target_face_mask=target_face_mask,
        config=config,
        landmarks=landmarks,
        data_vertex_weights=data_vertex_weights,
        correspondence_exclude_vertex_mask=correspondence_exclude_vertex_mask,
        smoothness_edge_weights=smoothness_edge_weights,
        edge_vector_edge_weights=edge_vector_edge_weights,
        affine_prior_vertex_weights=affine_prior_vertex_weights,
        position_prior_vertex_weights=position_prior_vertex_weights,
        position_prior_axis_weights=position_prior_axis_weights,
        shape_prior_vertex_weights=shape_prior_vertex_weights,
        vertex_mass_vertex_weights=vertex_mass_vertex_weights,
        flip_guard_vertex_scales=flip_guard_vertex_scales,
        camera_depth_reference_vertex_mask=face_mask_icp_ref,
        correspondence_query_pin_vertex_ids=frown_query_pin_ids,
    )
    fitted_vertices = result.vertices
    cached_face_mask = prior_region_info.get("_face_vertex_mask")
    cached_forehead_mask = prior_region_info.get("_forehead_vertex_mask")
    cached_darap_smooth_aux_mask = prior_region_info.get("_darap_smooth_aux_vertex_mask")
    cached_darap_eye_edge_mask = prior_region_info.get("_darap_eye_edge_vertex_mask")
    if cached_face_mask is not None:
        face_mask_icp = np.asarray(cached_face_mask, dtype=bool)
    else:
        face_mask_icp, _ = build_seg28_face_vertex_mask(
            src_v.shape[0],
            src_f,
            seg28_pkl_path,
            region_names=HACK_SEG28_FACE_REGION_NAMES,
        )
    if not bool(getattr(args, "no_restore_face_interior", False)) and not _postprocess_ultra_lite(
        args
    ):
        face_interior_anchor_vertices = fitted_vertices.copy()
    example_fast = _example_fast_enabled(args)
    if example_fast:
        print(
            "[example] example_fast: skip ICP invert repair, stretch/validity evals, history JSON",
            flush=True,
        )
    icp_invert_iters = int(getattr(args, "icp_invert_repair_iterations", 0))
    if example_fast:
        icp_invert_iters = 0
    if icp_invert_iters > 0:
        icp_repair_protect = np.ones(src_v.shape[0], dtype=np.float64)
        icp_repair_protect[~face_mask_icp] = 0.0
        if correspondence_exclude_vertex_mask is not None:
            icp_repair_protect[np.asarray(correspondence_exclude_vertex_mask, dtype=bool)] = 1.0
        if landmarks is not None and "face_vertices" in landmarks:
            icp_repair_protect[
                np.unique(np.asarray(landmarks["face_vertices"], dtype=np.int64).reshape(-1))
            ] = 1.0
        fitted_vertices = repair_inverted_faces(
            src_v,
            fitted_vertices,
            src_f,
            iterations=icp_invert_iters,
            blend_step=float(args.icp_invert_repair_blend),
            max_restore_displacement_scale=float(args.icp_invert_repair_max_restore_scale),
            protect_vertex_band=icp_repair_protect,
            protect_strength=float(args.icp_invert_repair_protect_strength),
        )
        result.vertices = fitted_vertices
    nonface_smooth_stats: dict[str, float] = {
        "enabled": 0.0,
        "iterations": 0.0,
        "num_smooth_vertices": 0.0,
        "weight_mean": 0.0,
    }
    fitted_vertices, nonface_smooth_stats = run_robust_auto_postprocess(
        src_v,
        src_f,
        fitted_vertices,
        result.faces,
        seg28_pkl_path,
        args,
        prior_region_info,
        landmarks,
        correspondence_exclude_vertex_mask,
        face_interior_anchor_vertices,
        face_boundary_band,
    )
    result.vertices = fitted_vertices
    regional_darap_stats: dict[str, float] = {
        "enabled": 0.0,
        "face_interior_vertices": 0.0,
        "lambda_default": float(getattr(args, "regional_darap_lambda_default", 3.0)),
        "lambda_face_interior": float(getattr(args, "regional_darap_lambda_face_interior", 6.5)),
        "lambda_forehead": float(getattr(args, "regional_darap_lambda_forehead", 1.0)),
        "lambda_frown_0": float(
            getattr(
                args,
                "regional_darap_lambda_frown_0",
                getattr(args, "regional_darap_lambda_frown", 8.5),
            )
        ),
        "lambda_jaw_chin_temple": float(
            getattr(args, "regional_darap_lambda_jaw_chin_temple", 1.0)
        ),
        "lambda_eye_edge": float(getattr(args, "regional_darap_lambda_eye_edge", 1.0)),
        "lambda_face_boundary": float(getattr(args, "regional_darap_lambda_face_boundary", 1.0)),
    }
    if not bool(getattr(args, "no_regional_darap_normal_deform", False)):
        target_normals_for_darap = vertex_normals(fitted_vertices, result.faces)
        face_interior_indices = build_regional_darap_face_interior_vertex_indices(
            src_v.shape[0],
            result.faces,
            seg28_pkl_path,
            face_vertex_mask=cached_face_mask,
        )
        forehead_indices = build_regional_darap_forehead_vertex_indices(
            src_v.shape[0],
            result.faces,
            seg28_pkl_path,
            forehead_vertex_mask=cached_forehead_mask,
        )
        cached_frown_mask = prior_region_info.get("_frown_vertex_mask")
        frown_indices = build_regional_darap_frown_vertex_indices(
            src_v.shape[0],
            result.faces,
            seg28_pkl_path,
            frown_vertex_mask=cached_frown_mask,
        )
        jaw_chin_temple_indices = build_regional_darap_jaw_chin_temple_vertex_indices(
            src_v.shape[0],
            result.faces,
            seg28_pkl_path,
            jaw_chin_temple_vertex_mask=cached_darap_smooth_aux_mask,
        )
        eye_edge_band_scale = float(
            getattr(args, "regional_darap_eye_edge_band_scale", 2.0)
        )
        stored_eye_edge_scale = float(
            prior_region_info.get("darap_eye_edge_band_scale", eye_edge_band_scale)
        )
        if (
            cached_darap_eye_edge_mask is not None
            and abs(stored_eye_edge_scale - eye_edge_band_scale) < 1e-9
        ):
            eye_edge_indices = build_regional_darap_eye_edge_vertex_indices(
                src_v,
                result.faces,
                seg28_pkl_path,
                eye_edge_vertex_mask=cached_darap_eye_edge_mask,
            )
        else:
            eye_edge_indices = build_regional_darap_eye_edge_vertex_indices(
                src_v,
                result.faces,
                seg28_pkl_path,
                band_width_scale=eye_edge_band_scale,
            )
        face_boundary_indices = build_regional_darap_face_boundary_vertex_indices(
            src_v,
            result.faces,
            seg28_pkl_path,
            band_width_scale=float(
                getattr(args, "regional_darap_face_boundary_band_scale", 2.0)
            ),
            face_vertex_mask=cached_face_mask,
            face_boundary_band=face_boundary_band,
            boundary_width=float(prior_region_info.get("boundary_width", 0.0)) or None,
        )
        fitted_vertices, regional_darap_stats = apply_regional_darap_normal_deform(
            src_v,
            result.faces,
            target_normals_for_darap,
            face_interior_indices,
            args,
            forehead_vertex_indices=forehead_indices,
            frown_vertex_indices=frown_indices,
            jaw_chin_temple_vertex_indices=jaw_chin_temple_indices,
            eye_edge_vertex_indices=eye_edge_indices,
            face_boundary_vertex_indices=face_boundary_indices,
            face_vertex_mask=cached_face_mask,
        )
        result.vertices = fitted_vertices
        print(
            "[example] regional dARAP normal deform: "
            f"face_interior={int(regional_darap_stats.get('face_interior_vertices', 0))} "
            f"forehead={int(regional_darap_stats.get('forehead_vertices', 0))} "
            f"frown_0={int(regional_darap_stats.get('frown_vertices', 0))} "
            f"jaw_chin_temple={int(regional_darap_stats.get('jaw_chin_temple_vertices', 0))} "
            f"eye_edge={int(regional_darap_stats.get('eye_edge_vertices', 0))} "
            f"face_boundary={int(regional_darap_stats.get('face_boundary_vertices', 0))} "
            f"lambda_default={regional_darap_stats.get('lambda_default', 3.0):.2f} "
            f"lambda_face={regional_darap_stats.get('lambda_face_interior', 6.5):.2f} "
            f"lambda_forehead={regional_darap_stats.get('lambda_forehead', 1.0):.2f} "
            f"lambda_frown_0={regional_darap_stats.get('lambda_frown_0', 8.5):.2f} "
            f"lambda_jaw_chin_temple={regional_darap_stats.get('lambda_jaw_chin_temple', 1.0):.2f} "
            f"lambda_eye_edge={regional_darap_stats.get('lambda_eye_edge', 1.0):.2f} "
            f"lambda_face_boundary={regional_darap_stats.get('lambda_face_boundary', 1.0):.2f} "
            f"darap_depth_atten={int(regional_darap_stats.get('darap_camera_depth_enabled', 0))} "
            f"darap_depth_w_min={regional_darap_stats.get('darap_camera_depth_weight_min', 1.0):.2f} "
            f"darap_lambda_mean={regional_darap_stats.get('darap_lambda_mean_after', 0.0):.2f} "
            f"max_disp={regional_darap_stats.get('max_displacement', 0.0):.4f}",
            flush=True,
        )
    write_obj_mesh(out_path, fitted_vertices, result.faces)

    edge_stretch_stats: dict[str, object] = {"regions": {}, "skipped": True}
    mesh_validity_stats: dict[str, float] = {}
    source_validity_stats: dict[str, float] = {}
    if not example_fast:
        n_vertices = src_v.shape[0]
        face_eval_mask = cached_face_mask
        if face_eval_mask is None:
            face_eval_mask, _ = build_seg28_face_vertex_mask(
                n_vertices,
                src_f,
                seg28_pkl_path,
                region_names=HACK_SEG28_FACE_REGION_NAMES,
            )
        cavity_eval_mask = None
        if cavity_pkl_path is not None:
            cavity_eval_mask, _ = build_hack_cavity_vertex_mask(n_vertices, cavity_pkl_path)
        ear_eval_mask = None
        forehead_eval_mask = cached_forehead_mask
        skull_eval_mask = None
        if seg28_pkl_path is not None:
            ear_eval_mask, _ = build_seg28_face_vertex_mask(
                n_vertices,
                src_f,
                seg28_pkl_path,
                region_names=HACK_EAR_REGION_NAMES,
            )
            if forehead_eval_mask is None:
                forehead_eval_mask, _ = build_seg28_face_vertex_mask(
                    n_vertices,
                    src_f,
                    seg28_pkl_path,
                    region_names=HACK_FOREHEAD_REGION_NAMES,
                )
            skull_eval_mask, _ = build_seg28_face_vertex_mask(
                n_vertices,
                src_f,
                seg28_pkl_path,
                region_names=HACK_SKULL_REGION_NAMES,
            )
        edge_stretch_stats = evaluate_mesh_edge_stretch(
            src_v,
            fitted_vertices,
            src_f,
            face_vertex_mask=face_eval_mask,
            cavity_vertex_mask=cavity_eval_mask,
            exclude_nonface_vertex_mask=ear_eval_mask,
            region_a_vertex_mask=forehead_eval_mask,
            region_b_vertex_mask=skull_eval_mask,
        )
        mesh_validity_stats = evaluate_mesh_validity(src_v, fitted_vertices, src_f)
        source_validity_stats = evaluate_mesh_validity(src_v, src_v, src_f)

    history = [
        {
            key: (float(value) if isinstance(value, (np.floating, float, int)) else value)
            for key, value in rec.items()
        }
        for rec in result.history
    ]
    if not example_fast:
        with open(hist_path, "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)

    final_valid = result.correspondences.valid_mask
    final_mean = (
        float(np.mean(result.correspondences.distances[final_valid]))
        if np.any(final_valid)
        else float("nan")
    )
    final_controlled = np.isfinite(result.correspondences.distances) & np.isfinite(result.correspondences.points).all(axis=1)
    if correspondence_exclude_vertex_mask is not None:
        final_controlled &= ~np.asarray(correspondence_exclude_vertex_mask, dtype=bool).reshape(-1)
    final_controlled_rmse = (
        float(np.sqrt(np.mean(result.correspondences.distances[final_controlled] ** 2)))
        if np.any(final_controlled)
        else float("nan")
    )
    final_controlled_mean = (
        float(np.mean(result.correspondences.distances[final_controlled]))
        if np.any(final_controlled)
        else float("nan")
    )
    if history and "controlled_rmse" in history[-1]:
        final_controlled_rmse = float(history[-1].get("controlled_rmse", final_controlled_rmse))
        final_controlled_mean = float(history[-1].get("controlled_mean_distance", final_controlled_mean))
        final_controlled_ratio = float(history[-1].get("controlled_ratio", float(np.mean(final_controlled)) if final_controlled.size else 0.0))
    else:
        final_controlled_ratio = float(np.mean(final_controlled)) if final_controlled.size else 0.0
    final_landmark_rmse = float("nan")
    final_eye_landmark_rmse = float("nan")
    final_lip_landmark_rmse = float("nan")
    final_frown_anchor_rmse = float("nan")
    final_opening_delta_rmse = float("nan")
    final_eye_opening_delta_rmse = float("nan")
    final_mouth_opening_delta_rmse = float("nan")
    if landmarks is not None:
        lmk_vertices = np.asarray(landmarks["face_vertices"], dtype=np.int64)
        lmk_bary = np.asarray(landmarks["barycentric"], dtype=np.float64)
        lmk_targets = np.asarray(landmarks["target_points"], dtype=np.float64)
        lmk_points = (fitted_vertices[lmk_vertices] * lmk_bary[:, :, None]).sum(axis=1)
        lmk_sq = np.sum((lmk_points - lmk_targets) ** 2, axis=1)
        final_landmark_rmse = float(np.sqrt(np.mean(lmk_sq)))
        n_base_lmks = int(landmarks.get("num_hack_landmarks", lmk_sq.shape[0]))
        is_frown_anchor = np.asarray(
            landmarks.get("is_frown_target_anchor", np.zeros(lmk_sq.shape[0], dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if is_frown_anchor.shape[0] != lmk_sq.shape[0]:
            is_frown_anchor = np.zeros(lmk_sq.shape[0], dtype=bool)
            if n_base_lmks < lmk_sq.shape[0]:
                is_frown_anchor[n_base_lmks:] = True
        lmk_ids = np.asarray(landmarks["landmark_indices"], dtype=np.int64)
        if lmk_ids.shape[0] == n_base_lmks and n_base_lmks <= lmk_sq.shape[0]:
            eye_sel = np.zeros(lmk_sq.shape[0], dtype=bool)
            lip_sel = np.zeros(lmk_sq.shape[0], dtype=bool)
            eye_sel[:n_base_lmks] = np.isin(
                lmk_ids, np.asarray(landmarks["eye_landmark_indices"], dtype=np.int64)
            )
            lip_sel[:n_base_lmks] = np.isin(
                lmk_ids, np.asarray(landmarks["lip_landmark_indices"], dtype=np.int64)
            )
            if np.any(eye_sel):
                final_eye_landmark_rmse = float(np.sqrt(np.mean(lmk_sq[eye_sel])))
            if np.any(lip_sel):
                final_lip_landmark_rmse = float(np.sqrt(np.mean(lmk_sq[lip_sel])))
        if np.any(is_frown_anchor):
            final_frown_anchor_rmse = float(np.sqrt(np.mean(lmk_sq[is_frown_anchor])))
        else:
            final_frown_anchor_rmse = float("nan")
        if "delta_pairs" in landmarks:
            delta_a = (fitted_vertices[np.asarray(landmarks["delta_face_vertices_a"], dtype=np.int64)] * np.asarray(landmarks["delta_barycentric_a"], dtype=np.float64)[:, :, None]).sum(axis=1)
            delta_b = (fitted_vertices[np.asarray(landmarks["delta_face_vertices_b"], dtype=np.int64)] * np.asarray(landmarks["delta_barycentric_b"], dtype=np.float64)[:, :, None]).sum(axis=1)
            delta_res = np.sum(((delta_a - delta_b) - np.asarray(landmarks["delta_target_deltas"], dtype=np.float64)) ** 2, axis=1)
            final_opening_delta_rmse = float(np.sqrt(np.mean(delta_res)))
            names = np.asarray(landmarks["delta_pair_names"])
            eye_delta_sel = names == "eye"
            mouth_delta_sel = names == "mouth"
            if np.any(eye_delta_sel):
                final_eye_opening_delta_rmse = float(np.sqrt(np.mean(delta_res[eye_delta_sel])))
            if np.any(mouth_delta_sel):
                final_mouth_opening_delta_rmse = float(np.sqrt(np.mean(delta_res[mouth_delta_sel])))
    summary = {
        "stem": stem,
        "fit_mode": "robust_auto",
        "raw_path": raw_path,
        "target_path": target_path,
        "bni_lmks_path": lmks_path,
        "target_face_mask_info": target_face_mask_info,
        "target_face_mask_enabled": bool(target_face_mask_info.get("enabled", False)),
        "bni_seg_path": target_face_mask_info.get("bni_seg_path"),
        "target_query_faces": int(target_face_mask_info.get("target_query_faces", tgt_f.shape[0])),
        "target_total_faces": int(target_face_mask_info.get("target_total_faces", tgt_f.shape[0])),
        "target_align_info": {
            "scale": align_info["scale"],
            "rotation": np.asarray(align_info["rotation"]).tolist(),
            "translation": np.asarray(align_info["translation"]).tolist(),
            "lmk_rmse": align_info["lmk_rmse"],
            "lmk_rmse_all_finite": align_info["lmk_rmse_all_finite"],
            "rotation_det": align_info["rotation_det"],
            "num_lmks_used": align_info["num_lmks_used"],
            "tgt_lmks_npz_path": align_info["tgt_lmks_npz_path"],
            "fitting_indices_used": np.asarray(align_info["fitting_indices_used"]).tolist(),
        } if align_info is not None else None,
        "prior_region_info": prior_region_info,
        "output_path": out_path,
        "history_path": hist_path,
        "iterations": len(result.history),
        "final_valid_ratio": float(np.mean(final_valid)),
        "final_mean_distance": final_mean,
        "final_controlled_rmse": final_controlled_rmse,
        "final_controlled_mean_distance": final_controlled_mean,
        "final_controlled_ratio": final_controlled_ratio,
        "landmark_weight": float(config.landmark_weight),
        "num_landmarks": int(landmarks["target_points"].shape[0]) if landmarks is not None else 0,
        "num_eye_landmarks": int(landmarks["eye_landmark_indices"].shape[0]) if landmarks is not None else 0,
        "num_lip_landmarks": int(landmarks["lip_landmark_indices"].shape[0]) if landmarks is not None else 0,
        "num_opening_landmark_pairs": int(landmarks.get("delta_pairs", np.empty((0, 2))).shape[0]) if landmarks is not None else 0,
        "final_landmark_rmse": final_landmark_rmse,
        "final_eye_landmark_rmse": final_eye_landmark_rmse,
        "final_lip_landmark_rmse": final_lip_landmark_rmse,
        "final_frown_anchor_rmse": final_frown_anchor_rmse,
        "num_frown_target_anchors": int(
            landmarks.get("num_frown_target_anchors", 0)
        )
        if landmarks is not None
        else 0,
        "final_opening_delta_rmse": final_opening_delta_rmse,
        "final_eye_opening_delta_rmse": final_eye_opening_delta_rmse,
        "final_mouth_opening_delta_rmse": final_mouth_opening_delta_rmse,
        "nonface_smooth": nonface_smooth_stats,
        "regional_darap": regional_darap_stats,
        "edge_stretch": edge_stretch_stats,
        "mesh_validity": mesh_validity_stats,
        "source_mesh_validity": source_validity_stats,
    }
    print("[example] wrote", out_path, flush=True)
    if not example_fast:
        print("[example] history", hist_path, flush=True)
        print(
            f"[example] validity inverted_faces={int(mesh_validity_stats['num_inverted_faces'])} "
            f"({100.0 * mesh_validity_stats['fraction_inverted_faces']:.2f}%) "
            f"degenerate_faces={int(mesh_validity_stats['num_degenerate_faces'])} "
            f"(source inverted={int(source_validity_stats['num_inverted_faces'])})",
            flush=True,
        )
        for stretch_region in (
            "face_deep_interior",
            "face_boundary_ring",
            "face_nonface_seam",
            "face_nonface_seam_no_ear",
            "forehead_skull_cross",
            "cavity_rim",
        ):
            region_stats = edge_stretch_stats["regions"].get(stretch_region, {})
            if int(region_stats.get("count", 0)) <= 0:
                continue
            print(
                f"[example] stretch {stretch_region}: "
                f"med={region_stats['ratio_median']:.3f} "
                f"p90={region_stats['ratio_p90']:.3f} "
                f">1.5={100.0 * region_stats['fraction_above_warn']:.1f}% "
                f">2.0={100.0 * region_stats['fraction_above_bad']:.1f}%",
                flush=True,
            )

    if args.visualize:
        visualize_result(src_v, src_f, tgt_v, tgt_f, result.vertices, result.faces, stem)
    return summary


def visualize_result(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    fitted_vertices: np.ndarray,
    fitted_faces: np.ndarray,
    stem: str,
) -> None:
    try:
        import polyscope as ps
    except ImportError as exc:
        raise RuntimeError("polyscope is required for --visualize") from exc

    ps.init()
    ps.set_program_name(f"nonrigid_icp::{stem}")
    ps.set_ground_plane_mode("none")
    ps.register_surface_mesh(
        "source_raw",
        source_vertices,
        source_faces,
        color=(0.8, 0.8, 0.8),
        smooth_shade=True,
        enabled=False,
    )
    ps.register_surface_mesh(
        "target_bni",
        target_vertices,
        target_faces,
        color=(0.2, 0.55, 0.95),
        smooth_shade=True,
        enabled=True,
    )
    ps.register_surface_mesh(
        "fitted_raw_to_bni",
        fitted_vertices,
        fitted_faces,
        color=(0.95, 0.52, 0.16),
        smooth_shade=True,
        enabled=True,
    )
    ps.show()



def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_single_example(args)


if __name__ == "__main__":
    main()
