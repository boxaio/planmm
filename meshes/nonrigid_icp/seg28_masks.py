"""HACK_seg_28 region masks shared by example.py and postprocess.py."""

from __future__ import annotations

import os
import pickle

import numpy as np

from geometry import geodesic_distance_from_mask, median_edge_length

HACK_SEG28_FACE_REGION_NAMES = (
    "cheek_0",
    "cheek_1",
    "chin_0",
    "chin_1",
    "eye_0",
    "eye_1",
    "eye_2",
    "eye_3",
    "eye_4",
    "eye_5",
    "fold_0",
    "fold_1",
    "forehead",
    "frown",
    "frown_0",
    "jaw_0",
    "jaw_1",
    "mouth_0",
    "mouth_1",
    "nose_0",
    "nose_1",
    "philtrum",
    "temple_0",
    "temple_1",
)

HACK_FOREHEAD_REGION_NAMES = ("forehead",)
HACK_SKULL_REGION_NAMES = ("skull",)
HACK_EAR_REGION_NAMES = ("ear_0", "ear_1")
HACK_JAW_REGION_NAMES = ("jaw_0", "jaw_1")
HACK_CHIN_REGION_NAMES = ("chin_0", "chin_1")
HACK_DARAP_SMOOTH_AUX_REGION_NAMES = (
    "chin_1",
    "jaw_0",
    "jaw_1",
    "temple_0",
    "temple_1",
)
HACK_DARAP_EYE_EDGE_REGION_NAMES = ("eye_0", "eye_3")
HACK_NECK_FRONT_REGION_NAMES = ("neck_f",)
HACK_NECK_BACK_REGION_NAMES = ("neck_b",)
HACK_FROWN_REGION_NAMES = ("frown_0",)
HACK_FROWN_DETAIL_REGION_NAMES = HACK_FROWN_REGION_NAMES
HACK_CAVITY_REGION_NAMES = ("nose", "mouth", "eyes")

# Some HACK_seg_28 pickles expose ``frown`` instead of the newer ``frown_0`` label.
HACK_SEG28_REGION_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "frown_0": ("frown_0", "frown"),
}


def resolve_seg28_region_name(seg28: dict[str, object], name: str) -> str | None:
    candidates = HACK_SEG28_REGION_NAME_ALIASES.get(name, (name,))
    for candidate in candidates:
        if candidate in seg28:
            return candidate
    return None


def build_hack_cavity_vertex_mask(
    n_vertices: int,
    cavity_pkl_path: str | os.PathLike[str],
    region_names: tuple[str, ...] = HACK_CAVITY_REGION_NAMES,
) -> tuple[np.ndarray, dict[str, object]]:
    with open(cavity_pkl_path, "rb") as fp:
        cavity_info = pickle.load(fp)
    if not isinstance(cavity_info, dict):
        raise TypeError(f"expected dict in {cavity_pkl_path}, got {type(cavity_info)!r}")

    mask = np.zeros(int(n_vertices), dtype=bool)
    matched_names = []
    for name in region_names:
        if name not in cavity_info:
            continue
        region = cavity_info[name]
        if not isinstance(region, dict) or "vert_ids" not in region:
            continue
        matched_names.append(name)
        vids = np.asarray(region["vert_ids"], dtype=np.int64).reshape(-1)
        vids = vids[(vids >= 0) & (vids < int(n_vertices))]
        mask[vids] = True

    if not matched_names:
        raise KeyError(
            "None of the requested cavity region names were found in HACK_cavity. "
            f"requested={list(region_names)}, available={sorted(cavity_info.keys())}"
        )

    return mask, {
        "cavity_path": os.fspath(cavity_pkl_path),
        "region_names": list(matched_names),
        "num_cavity_vertices": int(np.count_nonzero(mask)),
    }


def load_hack_seg_28(path: str | os.PathLike[str]) -> dict[str, object]:
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    if not isinstance(data, dict):
        raise TypeError(f"expected dict in {path}, got {type(data)!r}")
    return data


def build_seg28_face_vertex_mask(
    n_vertices: int,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    region_names: tuple[str, ...] = HACK_SEG28_FACE_REGION_NAMES,
) -> tuple[np.ndarray, dict[str, object]]:
    seg28 = load_hack_seg_28(seg28_pkl_path)
    faces = np.asarray(faces, dtype=np.int64)
    mask = np.zeros(int(n_vertices), dtype=bool)
    matched_names = []
    resolved_names: set[str] = set()
    for name in region_names:
        resolved = resolve_seg28_region_name(seg28, name)
        if resolved is None or resolved in resolved_names:
            continue
        resolved_names.add(resolved)
        matched_names.append(resolved)
        region = seg28[resolved]
        if not isinstance(region, dict):
            continue
        if "vert_ids" in region:
            vids = np.asarray(region["vert_ids"], dtype=np.int64).reshape(-1)
        elif "vids" in region:
            vids = np.asarray(region["vids"], dtype=np.int64).reshape(-1)
        elif "face_ids" in region:
            fids = np.asarray(region["face_ids"], dtype=np.int64).reshape(-1)
            fids = fids[(fids >= 0) & (fids < faces.shape[0])]
            vids = np.unique(faces[fids].reshape(-1)).astype(np.int64)
        else:
            continue
        vids = vids[(vids >= 0) & (vids < int(n_vertices))]
        mask[vids] = True

    if not matched_names:
        raise KeyError(
            "None of the requested face region names were found in HACK_seg_28. "
            f"requested={list(region_names)}, available={sorted(seg28.keys())}"
        )

    info = {
        "seg28_path": os.fspath(seg28_pkl_path),
        "region_names": list(matched_names),
        "num_face_vertices": int(np.count_nonzero(mask)),
        "num_nonface_vertices": int(n_vertices - np.count_nonzero(mask)),
    }
    return mask, info


def build_region_boundary_ring_vertex_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    region_mask: np.ndarray,
    *,
    band_width_scale: float = 2.0,
) -> np.ndarray:
    """Vertices inside ``region_mask`` within geodesic band of the region's mesh boundary."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    region_mask = np.asarray(region_mask, dtype=bool).reshape(-1)
    if not np.any(region_mask):
        return np.zeros(region_mask.shape[0], dtype=bool)
    seed = face_boundary_seed(faces, region_mask)
    edge_scale = float(median_edge_length(vertices, faces))
    band_dist = float(band_width_scale) * max(edge_scale, 1e-12)
    max_dist = max(band_dist, edge_scale)
    dist = geodesic_distance_from_mask(
        vertices,
        faces,
        seed,
        max_distance=max_dist,
        allowed_mask=region_mask,
    )
    return (
        region_mask
        & np.isfinite(dist)
        & (dist <= band_dist)
    )


def face_boundary_seed(faces: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    seed = np.zeros(face_mask.shape[0], dtype=bool)
    for tri in np.asarray(faces, dtype=np.int64):
        tri_face = face_mask[tri]
        if np.any(tri_face) and not np.all(tri_face):
            seed[tri] = True
    return seed if np.any(seed) else face_mask.copy()


def build_neck_protect_vertex_mask(
    n_vertices: int,
    faces: np.ndarray,
    seg28_pkl_path: str | os.PathLike[str],
    vertices: np.ndarray,
    *,
    chin_neck_band_scale: float = 14.0,
) -> np.ndarray:
    """Vertices that must not move during global non-face Laplacian / Taubin passes."""
    faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=np.float64)
    protect = np.zeros(int(n_vertices), dtype=bool)
    neck_f, _ = build_seg28_face_vertex_mask(
        n_vertices, faces, seg28_pkl_path, region_names=HACK_NECK_FRONT_REGION_NAMES
    )
    protect |= np.asarray(neck_f, dtype=bool)
    try:
        neck_b, _ = build_seg28_face_vertex_mask(
            n_vertices, faces, seg28_pkl_path, region_names=HACK_NECK_BACK_REGION_NAMES
        )
        protect |= np.asarray(neck_b, dtype=bool)
    except KeyError:
        pass
    chin, _ = build_seg28_face_vertex_mask(
        n_vertices, faces, seg28_pkl_path, region_names=HACK_CHIN_REGION_NAMES
    )
    edge_scale = float(median_edge_length(vertices, faces))
    band = float(chin_neck_band_scale) * edge_scale
    if band > 0.0 and edge_scale > 0.0 and np.any(chin) and np.any(neck_f):
        cross_seed = region_cross_boundary_seed(faces, chin, neck_f)
        shell = (neck_f | chin) & np.isfinite(
            geodesic_distance_from_mask(
                vertices,
                faces,
                cross_seed,
                max_distance=band,
                allowed_mask=neck_f | chin,
            )
        )
        protect |= shell
    return protect


def region_cross_boundary_seed(
    faces: np.ndarray,
    region_a_mask: np.ndarray,
    region_b_mask: np.ndarray,
) -> np.ndarray:
    seed = np.zeros(region_a_mask.shape[0], dtype=bool)
    region_a_mask = np.asarray(region_a_mask, dtype=bool).reshape(-1)
    region_b_mask = np.asarray(region_b_mask, dtype=bool).reshape(-1)
    for tri in np.asarray(faces, dtype=np.int64):
        if np.any(region_a_mask[tri]) and np.any(region_b_mask[tri]):
            seed[tri] = True
    return seed
