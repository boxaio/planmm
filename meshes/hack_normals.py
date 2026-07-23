from __future__ import annotations

import argparse
import os
import os.path as osp
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import trimesh
import sys
from scipy.spatial import cKDTree

_THIS_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _path in (str(_REPO_ROOT), str(_THIS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from utils.ps_tools import show_mesh_pair

HACK_FITTING_INDICES = np.array(
    [
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 168, 197, 5, 4,
        98, 97, 2, 326, 327, 246, 159, 157, 173, 153, 144, 398, 385, 387,
        263, 373, 381, 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84,
        181, 78, 38, 12, 268, 308, 316, 15, 86,
    ],
    dtype=np.int64,
)


@dataclass
class ClosestSurfaceResult:
    points: np.ndarray
    distances: np.ndarray
    face_indices: np.ndarray
    method: str


@dataclass
class TargetNormalResult:
    normals: np.ndarray
    valid_mask: np.ndarray
    distances: np.ndarray
    closest_points: np.ndarray
    face_indices: np.ndarray
    method: str


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(
        vectors,
        np.maximum(norms, eps),
        out=np.zeros_like(vectors, dtype=np.float64),
        where=norms > eps,
    )


def validate_vertices_faces(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (n, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (m, 3), got {faces.shape}")
    if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
        raise ValueError("faces contain vertex indices outside vertices")
    return vertices, faces


def face_normals(vertices: np.ndarray, faces: np.ndarray, normalize: bool = True) -> np.ndarray:
    vertices, faces = validate_vertices_faces(vertices, faces)
    if faces.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return normalize_vectors(normals) if normalize else normals


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices, faces = validate_vertices_faces(vertices, faces)
    normals = np.zeros_like(vertices, dtype=np.float64)
    if faces.shape[0] == 0:
        return normals
    fn = face_normals(vertices, faces, normalize=False)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], fn)
    return normalize_vectors(normals)


def _parse_face_vertex_index(token: str) -> int:
    value = token.split("/", 1)[0]
    if not value:
        raise ValueError(f"missing vertex index in face token {token!r}")
    return int(value)


def _triangulate_polygon(indices: list[int]) -> Iterable[list[int]]:
    if len(indices) < 3:
        return
    for i in range(2, len(indices)):
        yield [indices[0], indices[i - 1], indices[i]]


def read_phack_mesh(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Read OBJ vertices and triangular faces without relying on project mesh helpers.
    """
    path = os.fspath(path)
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                toks = line.split()[1:4]
                vertices.append([float(toks[0]), float(toks[1]), float(toks[2])])
            elif line.startswith("f "):
                toks = line.split()[1:]
                idx = [_parse_face_vertex_index(tok) for tok in toks]
                n_vertices_so_far = len(vertices)
                zero_based = [
                    (i - 1 if i > 0 else n_vertices_so_far + i)
                    for i in idx
                ]
                faces.extend(_triangulate_polygon(zero_based))
    vertices_np = np.asarray(vertices, dtype=np.float64)
    faces_np = np.asarray(faces, dtype=np.int64)
    return validate_vertices_faces(vertices_np, faces_np)


def read_target_mesh(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    path = os.fspath(path)
    mesh = trimesh.load(path, process=False, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return validate_vertices_faces(vertices, faces)


def load_hack_lmks478_bary(path: str | os.PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    if "face_indices" not in data or "bary_coords" not in data:
        raise KeyError(f"face_indices/bary_coords missing in {path}")
    face_indices = np.asarray(data["face_indices"], dtype=np.int64).reshape(-1)
    bary_coords = np.asarray(data["bary_coords"], dtype=np.float64).reshape(-1, 3)
    if face_indices.shape[0] != 478 or bary_coords.shape[0] != 478:
        raise ValueError(
            f"expected 478 HACK landmarks, got face_indices={face_indices.shape}, "
            f"bary_coords={bary_coords.shape}"
        )
    return face_indices, bary_coords


def bni_lmks_npz_to_lm478(lmks_npz_path: str | os.PathLike[str]) -> np.ndarray:
    with np.load(os.fspath(lmks_npz_path)) as data:
        if "vertex_coords" not in data:
            raise KeyError(f"'vertex_coords' missing in {lmks_npz_path}")
        coords = np.asarray(data["vertex_coords"], dtype=np.float64).reshape(-1, 3)
        if "lmk_indices" in data:
            lmk_indices = np.asarray(data["lmk_indices"], dtype=np.int64).reshape(-1)
        else:
            lmk_indices = np.arange(coords.shape[0], dtype=np.int64)
    if lmk_indices.shape[0] != coords.shape[0]:
        raise ValueError(
            f"lmk_indices length {lmk_indices.shape[0]} != vertex_coords rows {coords.shape[0]}"
        )
    lm478 = np.full((478, 3), np.nan, dtype=np.float64)
    ok = (lmk_indices >= 0) & (lmk_indices < 478)
    lm478[lmk_indices[ok]] = coords[ok]
    return lm478


def landmarks_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_indices: np.ndarray,
    bary_coords: np.ndarray,
) -> np.ndarray:
    vertices, faces = validate_vertices_faces(vertices, faces)
    face_indices = np.asarray(face_indices, dtype=np.int64).reshape(-1)
    bary_coords = np.asarray(bary_coords, dtype=np.float64).reshape(-1, 3)
    if face_indices.size and (face_indices.min() < 0 or face_indices.max() >= faces.shape[0]):
        raise ValueError("landmark face indices outside mesh faces")
    tri = faces[face_indices]
    return (vertices[tri] * bary_coords[:, :, None]).sum(axis=1)


def similarity_transform_rows_procrustes(
    src: np.ndarray,
    tgt: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Return scale, R, t so tgt ~= (src @ R.T) * scale + t.
    """
    src = np.asarray(src, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    if src.shape != tgt.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/tgt must both have shape (n, 3), got {src.shape} and {tgt.shape}")
    src_mean = src.mean(axis=0, keepdims=True)
    tgt_mean = tgt.mean(axis=0, keepdims=True)
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean
    denom = float((src_centered * src_centered).sum())
    if denom <= 1e-16:
        raise ValueError("source landmarks are degenerate for similarity transform")
    h = src_centered.T @ tgt_centered
    u, svals, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt = vt.copy()
        svals = svals.copy()
        vt[-1, :] *= -1.0
        svals[-1] *= -1.0
        rot = vt.T @ u.T
    scale = float(svals.sum() / denom)
    trans = (tgt_mean - (src_mean @ rot.T) * scale).reshape(3)
    return scale, rot, trans


def apply_similarity_to_vertices(
    vertices: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    return (vertices @ rotation.T) * float(scale) + translation


def bbox_center_and_diag(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (n, 3), got {vertices.shape}")
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return (bbox_min + bbox_max) * 0.5, float(np.linalg.norm(bbox_max - bbox_min))


def align_target_mesh_to_hack_by_landmarks(
    target_vertices: np.ndarray,
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    bni_lmks_npz_path: str | os.PathLike[str],
    hack_lmks478_bary_path: str | os.PathLike[str],
    fitting_indices: np.ndarray = HACK_FITTING_INDICES,
) -> tuple[np.ndarray, dict[str, object]]:
    face_indices, bary_coords = load_hack_lmks478_bary(hack_lmks478_bary_path)
    src_lmks = landmarks_from_mesh(source_vertices, source_faces, face_indices, bary_coords)
    tgt_lmks = bni_lmks_npz_to_lm478(bni_lmks_npz_path)
    fitting_indices = np.asarray(fitting_indices, dtype=np.int64).reshape(-1)
    ok = (
        (fitting_indices >= 0)
        & (fitting_indices < src_lmks.shape[0])
        & np.isfinite(src_lmks[fitting_indices]).all(axis=1)
        & np.isfinite(tgt_lmks[fitting_indices]).all(axis=1)
    )
    used = fitting_indices[ok]
    if used.shape[0] < 3:
        raise ValueError(
            f"need at least 3 valid fitting landmarks, got {used.shape[0]} from {bni_lmks_npz_path}"
        )

    scale, rot, trans = similarity_transform_rows_procrustes(
        tgt_lmks[used],
        src_lmks[used],
    )
    aligned = apply_similarity_to_vertices(target_vertices, scale, rot, trans)
    pred = apply_similarity_to_vertices(tgt_lmks[used], scale, rot, trans)
    residual = pred - src_lmks[used]
    source_center, source_diag = bbox_center_and_diag(source_vertices)
    target_center, target_diag = bbox_center_and_diag(target_vertices)
    aligned_center, aligned_diag = bbox_center_and_diag(aligned)
    info = {
        "scale": scale,
        "rotation": rot,
        "translation": trans,
        "source_bbox_center": source_center,
        "target_bbox_center": target_center,
        "aligned_bbox_center": aligned_center,
        "source_bbox_diag": source_diag,
        "target_bbox_diag": target_diag,
        "aligned_bbox_diag": aligned_diag,
        "rotation_det": float(np.linalg.det(rot)),
        "lmk_rmse": float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))),
        "num_lmks_used": int(used.shape[0]),
        "fitting_indices_used": used,
        "tgt_lmks_npz_path": os.fspath(bni_lmks_npz_path),
    }
    return aligned, info


def closest_surface_points(
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
    query_points: np.ndarray,
) -> ClosestSurfaceResult:
    target_vertices, target_faces = validate_vertices_faces(target_vertices, target_faces)
    query_points = np.asarray(query_points, dtype=np.float64)
    mesh = trimesh.Trimesh(vertices=target_vertices, faces=target_faces, process=False)
    try:
        closest, distances, face_indices = trimesh.proximity.closest_point(mesh, query_points)
        return ClosestSurfaceResult(
            points=np.asarray(closest, dtype=np.float64),
            distances=np.asarray(distances, dtype=np.float64),
            face_indices=np.asarray(face_indices, dtype=np.int64),
            method="surface",
        )
    except Exception as exc:
        tree = cKDTree(target_vertices)
        distances, vertex_indices = tree.query(query_points, k=1)
        face_indices = np.full(query_points.shape[0], -1, dtype=np.int64)
        return ClosestSurfaceResult(
            points=target_vertices[np.asarray(vertex_indices, dtype=np.int64)],
            distances=np.asarray(distances, dtype=np.float64),
            face_indices=face_indices,
            method=f"vertex_fallback:{type(exc).__name__}",
        )


def barycentric_coordinates_on_triangles(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.shape != (points.shape[0], 3, 3):
        raise ValueError(f"triangles must have shape {(points.shape[0], 3, 3)}, got {triangles.shape}")

    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01

    bary = np.zeros((points.shape[0], 3), dtype=np.float64)
    ok = np.abs(denom) > 1e-20
    bary[:, 0] = 1.0
    bary[ok, 1] = (d11[ok] * d20[ok] - d01[ok] * d21[ok]) / denom[ok]
    bary[ok, 2] = (d00[ok] * d21[ok] - d01[ok] * d20[ok]) / denom[ok]
    bary[ok, 0] = 1.0 - bary[ok, 1] - bary[ok, 2]
    return bary


def compute_target_normals_for_raw_vertices(
    raw_vertices: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
) -> TargetNormalResult:
    raw_vertices = np.asarray(raw_vertices, dtype=np.float64)
    target_vertices, target_faces = validate_vertices_faces(target_vertices, target_faces)

    target_vertex_normals = vertex_normals(target_vertices, target_faces)
    closest = closest_surface_points(target_vertices, target_faces, raw_vertices)
    normals = np.zeros_like(raw_vertices, dtype=np.float64)

    valid = np.isfinite(closest.points).all(axis=1) & np.isfinite(closest.distances)
    face_ok = (
        valid
        & (closest.face_indices >= 0)
        & (closest.face_indices < target_faces.shape[0])
    )
    if np.any(face_ok):
        tri_faces = target_faces[closest.face_indices[face_ok]]
        tri_vertices = target_vertices[tri_faces]
        bary = barycentric_coordinates_on_triangles(closest.points[face_ok], tri_vertices)
        interp = (target_vertex_normals[tri_faces] * bary[:, :, None]).sum(axis=1)
        normals[face_ok] = interp

    fallback = valid & (np.linalg.norm(normals, axis=1) <= 1e-12)
    if np.any(fallback):
        tree = cKDTree(target_vertices)
        _, vertex_indices = tree.query(closest.points[fallback], k=1)
        normals[fallback] = target_vertex_normals[np.asarray(vertex_indices, dtype=np.int64)]

    normals = normalize_vectors(normals)
    valid &= np.linalg.norm(normals, axis=1) > 1e-12
    return TargetNormalResult(
        normals=normals,
        valid_mask=valid,
        distances=closest.distances,
        closest_points=closest.points,
        face_indices=closest.face_indices,
        method=closest.method,
    )


def show_raw_vs_target_normal_angle_cloud(
    raw_vertices: np.ndarray,
    raw_faces: np.ndarray,
    target_normals: np.ndarray | TargetNormalResult,
    valid_mask: Optional[np.ndarray] = None,
    *,
    mesh_name: str = "raw_phack",
    angle_name: str = "target/raw normal angle (deg)",
    vector_radius: float = 0.002,
    vector_length: float = 0.02,
) -> np.ndarray:
    """
    Display a Polyscope vertex scalar cloud of the angle between raw mesh normals
    and target-guided normals returned by compute_target_normals_for_raw_vertices.

    Returns the per-vertex angle array in degrees. Invalid vertices are set to NaN.
    """
    import polyscope as ps

    raw_vertices, raw_faces = validate_vertices_faces(raw_vertices, raw_faces)
    closest_points = None
    distances = None
    if isinstance(target_normals, TargetNormalResult):
        if valid_mask is None:
            valid_mask = target_normals.valid_mask
        closest_points = target_normals.closest_points
        distances = target_normals.distances
        target_normals = target_normals.normals
    target_normals = np.asarray(target_normals, dtype=np.float64)
    if target_normals.shape != raw_vertices.shape:
        raise ValueError(
            f"target_normals must have shape {raw_vertices.shape}, got {target_normals.shape}"
        )

    raw_normals = vertex_normals(raw_vertices, raw_faces)
    target_normals = normalize_vectors(target_normals)
    raw_normals = normalize_vectors(raw_normals)
    if valid_mask is None:
        valid_mask = np.ones(raw_vertices.shape[0], dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid_mask.shape[0] != raw_vertices.shape[0]:
            raise ValueError(
                f"valid_mask must have length {raw_vertices.shape[0]}, got {valid_mask.shape[0]}"
            )

    signed_dot_values = np.einsum("ij,ij->i", raw_normals, target_normals)
    dot_values = np.abs(signed_dot_values)
    finite = (
        valid_mask
        & np.isfinite(dot_values)
        & (np.linalg.norm(raw_normals, axis=1) > 1e-12)
        & (np.linalg.norm(target_normals, axis=1) > 1e-12)
    )
    angles = np.full(raw_vertices.shape[0], np.nan, dtype=np.float64)
    angles[finite] = np.rad2deg(np.arccos(np.clip(dot_values[finite], 0.0, 1.0)))
    full_angles = np.full(raw_vertices.shape[0], np.nan, dtype=np.float64)
    full_angles[finite] = np.rad2deg(np.arccos(np.clip(signed_dot_values[finite], -1.0, 1.0)))

    signed_displacement = None
    if closest_points is not None:
        closest_points = np.asarray(closest_points, dtype=np.float64)
        if closest_points.shape == raw_vertices.shape:
            signed_displacement = np.full(raw_vertices.shape[0], np.nan, dtype=np.float64)
            delta = closest_points - raw_vertices
            signed_displacement[finite] = np.einsum("ij,ij->i", delta[finite], raw_normals[finite])

    ps.init()
    ps_mesh = ps.register_surface_mesh(
        mesh_name,
        raw_vertices,
        raw_faces,
        smooth_shade=True,
    )
    ps_mesh.add_scalar_quantity(
        angle_name,
        angles,
        defined_on="vertices",
        enabled=True,
        cmap="reds",
    )
    ps_mesh.add_scalar_quantity(
        "full target/raw normal angle (deg)",
        full_angles,
        defined_on="vertices",
        enabled=False,
        cmap="reds",
    )
    if distances is not None:
        ps_mesh.add_scalar_quantity(
            "raw to target correspondence distance",
            np.asarray(distances, dtype=np.float64),
            defined_on="vertices",
            enabled=False,
            cmap="reds",
        )
    if signed_displacement is not None:
        ps_mesh.add_scalar_quantity(
            "signed displacement along raw normal",
            signed_displacement,
            defined_on="vertices",
            enabled=False,
            cmap="coolwarm",
        )
    ps_mesh.add_vector_quantity(
        "raw mesh normals",
        raw_normals,
        defined_on="vertices",
        enabled=False,
        radius=vector_radius,
        length=vector_length,
    )
    ps_mesh.add_vector_quantity(
        "target-guided normals",
        target_normals,
        defined_on="vertices",
        enabled=False,
        radius=vector_radius,
        length=vector_length,
    )
    ps.show()
    return angles


def stem_from_raw_obj(path: str | os.PathLike[str]) -> str:
    stem = Path(path).stem
    if stem.endswith("_phack_raw"):
        return stem[: -len("_phack_raw")]
    return stem


def find_stems(raw_dir: Path, target_dir: Path, stems: Optional[str | list[str]]) -> list[str]:
    if stems:
        if isinstance(stems, str):
            requested = [stems]
        else:
            requested = list(stems)
        normalized = []
        for stem in requested:
            stem = stem_from_raw_obj(stem)
            if stem.endswith(".obj"):
                stem = stem_from_raw_obj(stem)
            normalized.append(stem)
        return list(dict.fromkeys(normalized))
    found: list[str] = []
    for raw_path in sorted(raw_dir.glob("*_phack_raw.obj")):
        stem = stem_from_raw_obj(raw_path)
        if (target_dir / f"{stem}.obj").is_file():
            found.append(stem)
    return found


def process_one(
    stem: str,
    raw_dir: Path,
    target_dir: Path,
    bni_lmks_dir: Path,
    hack_lmks478_bary_path: Path,
    output_dir: Path,
    show_angle_cloud: bool = False,
) -> dict[str, object]:
    raw_path = raw_dir / f"{stem}_phack_raw.obj"
    target_path = target_dir / f"{stem}.obj"
    bni_lmks_path = bni_lmks_dir / f"{stem}_bni_lmks.npz"
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw mesh not found: {raw_path}")
    if not target_path.is_file():
        raise FileNotFoundError(f"target mesh not found: {target_path}")
    if not bni_lmks_path.is_file():
        raise FileNotFoundError(f"BNI landmarks not found: {bni_lmks_path}")

    raw_vertices, raw_faces = read_phack_mesh(raw_path)
    target_vertices, target_faces = read_target_mesh(target_path)
    aligned_target_vertices, align_info = align_target_mesh_to_hack_by_landmarks(
        target_vertices,
        raw_vertices,
        raw_faces,
        bni_lmks_path,
        hack_lmks478_bary_path,
    )
    result = compute_target_normals_for_raw_vertices(
        raw_vertices,
        aligned_target_vertices,
        target_faces,
    )
    # show_mesh_pair(raw_vertices, raw_faces, aligned_target_vertices, target_faces, use_offset=False)
    if show_angle_cloud:
        show_raw_vs_target_normal_angle_cloud(
            raw_vertices,
            raw_faces,
            result,
            mesh_name=f"{stem}_raw_phack",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}_phack_raw_target_normals.npz"
    np.savez(
        out_path,
        normals=result.normals.astype(np.float32),
        valid_mask=result.valid_mask,
        distances=result.distances.astype(np.float32),
        closest_points=result.closest_points.astype(np.float32),
        face_indices=result.face_indices,
        raw_path=str(raw_path),
        target_path=str(target_path),
        target_align_scale=np.asarray(align_info["scale"], dtype=np.float64),
        target_align_rotation=np.asarray(align_info["rotation"], dtype=np.float64),
        target_align_translation=np.asarray(align_info["translation"], dtype=np.float64),
        source_bbox_center=np.asarray(align_info["source_bbox_center"], dtype=np.float64),
        target_bbox_center=np.asarray(align_info["target_bbox_center"], dtype=np.float64),
        aligned_bbox_center=np.asarray(align_info["aligned_bbox_center"], dtype=np.float64),
        source_bbox_diag=np.asarray(align_info["source_bbox_diag"], dtype=np.float64),
        target_bbox_diag=np.asarray(align_info["target_bbox_diag"], dtype=np.float64),
        aligned_bbox_diag=np.asarray(align_info["aligned_bbox_diag"], dtype=np.float64),
    )

    valid_dist = result.distances[result.valid_mask]
    summary = {
        "stem": stem,
        "raw_vertices": int(raw_vertices.shape[0]),
        "raw_faces": int(raw_faces.shape[0]),
        "target_vertices": int(target_vertices.shape[0]),
        "target_faces": int(target_faces.shape[0]),
        "valid_ratio": float(np.mean(result.valid_mask)) if result.valid_mask.size else 0.0,
        "mean_distance": float(np.mean(valid_dist)) if valid_dist.size else float("nan"),
        "max_distance": float(np.max(valid_dist)) if valid_dist.size else float("nan"),
        "method": result.method,
        "lmk_rmse": float(align_info["lmk_rmse"]),
        "output_path": str(out_path),
    }
    print(
        f"[hack_normals] {stem}: "
        f"raw V/F={summary['raw_vertices']}/{summary['raw_faces']} "
        f"target V/F={summary['target_vertices']}/{summary['target_faces']} "
        f"valid={summary['valid_ratio']:.3f} "
        f"mean_dist={summary['mean_distance']:.6f} "
        f"max_dist={summary['max_distance']:.6f} "
        f"method={summary['method']} "
        f"out={out_path}",
        flush=True,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assign BNI target-correspondence normals to raw PHACK vertices and save NPZ files."
    )
    parser.add_argument("--raw-dir", default="./test/phack_raw")
    parser.add_argument("--target-dir", default="./test/bni")
    parser.add_argument("--bni-lmks-dir", default="./test/bni_lmks")
    parser.add_argument("--hack-lmks478-bary-path", default="./dataset/hack_lmks478_bary.pkl")
    parser.add_argument("--output-dir", default='./test/phack_raw', help="defaults to --raw-dir")
    parser.add_argument("--stem", default='000063_52', help="one or more sample stems")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--no-show-angle-cloud",
        dest="show_angle_cloud",
        action="store_false",
        help="disable Polyscope visualization of target/raw normal angle",
    )
    parser.set_defaults(show_angle_cloud=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    raw_dir = Path(args.raw_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    bni_lmks_dir = Path(args.bni_lmks_dir).resolve()
    hack_lmks478_bary_path = Path(args.hack_lmks478_bary_path).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else raw_dir

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw-dir not found: {raw_dir}")
    if not target_dir.is_dir():
        raise FileNotFoundError(f"target-dir not found: {target_dir}")
    if not bni_lmks_dir.is_dir():
        raise FileNotFoundError(f"bni-lmks-dir not found: {bni_lmks_dir}")
    if not hack_lmks478_bary_path.is_file():
        raise FileNotFoundError(f"hack landmarks bary file not found: {hack_lmks478_bary_path}")

    stems = find_stems(raw_dir, target_dir, args.stem)
    if not stems:
        raise FileNotFoundError(f"no matching *_phack_raw.obj and target obj found in {raw_dir} / {target_dir}")

    failures: list[tuple[str, str]] = []
    for stem in stems:
        out_path = output_dir / f"{stem}_phack_raw_target_normals.npz"
        if args.skip_existing and out_path.is_file():
            print(f"[hack_normals] skip existing {out_path}", flush=True)
            continue
        try:
            process_one(
                stem=stem,
                raw_dir=raw_dir,
                target_dir=target_dir,
                bni_lmks_dir=bni_lmks_dir,
                hack_lmks478_bary_path=hack_lmks478_bary_path,
                output_dir=output_dir,
                show_angle_cloud=bool(args.show_angle_cloud),
            )
        except Exception as exc:
            failures.append((stem, f"{type(exc).__name__}: {exc}"))
            print(f"[hack_normals] failed {stem}: {type(exc).__name__}: {exc}", flush=True)

    if failures:
        print(f"[hack_normals] {len(failures)} failure(s):", flush=True)
        for stem, msg in failures:
            print(f"  - {stem}: {msg}", flush=True)
        return 1
    return 0


if __name__ == "__main__":

    main()
