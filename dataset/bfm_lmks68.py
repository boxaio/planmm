"""
STAR（68 点）+ 仓库内 ``mesh_render.render_mesh``：
多份 BFM OBJ 在共用世界系下渲染，2D 关键点反投影射线与网格求交，多视图 3D 中值融合后
在 ``bfm_template`` 表面求最近点，输出 ``face_indices`` 与 ``bary_coords``。
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pyrender
import torch
import trimesh
from tqdm import tqdm
from trimesh.triangles import points_to_barycentric

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET_DIR = Path(__file__).resolve().parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.bfm_lmks478 import (
    DEFAULT_MESH_FILES,
    _canonical_camera_bundle,
    _closest_on_ray_to_surface,
    _load_obj_vf,
    fuse_hits_prioritize_reference_view_then_median_reference_frames,
    jitter_uv_offsets,
    pixel_ray_opengl_top_left_origin,
    pyrender_projection_matrix,
    ray_mesh_first_positive_hit,
    render_rgb_uint8,
    reprojection_residuals_px_vs_mediapipe,
    save_debug_lmks2d_overlays,
    save_debug_reproj_compare_overlays,
    world_points_to_pixel_uv_openvr_top_left_image,
)
from utils.landmark_detector_star import LandmarkDetectorSTAR

N_LANDMARKS = 68

# 与 BfmHead(recenter=True) 一致：assets 模板 + 前 35709 顶点质心
DEFAULT_BFM_TEMPLATE = Path(
    "/media/ubuntu/xb/cv_resnet50_face-reconstruction/assets/template_bfm.obj"
)
BFM_N_FRONT_VERTS = 35709


def _bfm_recenter_origin(vertices: np.ndarray) -> np.ndarray:
    """与 ``BfmHead`` 相同：减去前 ``BFM_N_FRONT_VERTS`` 顶点均值。"""
    ctr = np.mean(vertices[:BFM_N_FRONT_VERTS], axis=0, keepdims=True)
    return ctr.reshape(3)


def build_star_detector() -> LandmarkDetectorSTAR:
    return LandmarkDetectorSTAR()


def star_landmarks_pixels_batch(
    rgb_images: Sequence[np.ndarray],
    *,
    detector: LandmarkDetectorSTAR | None = None,
) -> np.ndarray:
    """``(K, 68, 2)`` 像素 (u,v)，未检出则为 ``nan``."""
    det = detector if detector is not None else build_star_detector()
    K = len(rgb_images)
    out = np.full((K, N_LANDMARKS, 2), np.nan, dtype=np.float64)
    for k, rgb in enumerate(rgb_images):
        if rgb.dtype != np.uint8:
            rgb = (
                np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                if rgb.max() <= 1.0
                else rgb.astype(np.uint8)
            )
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected HWC RGB, got {rgb.shape}")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        _bbox, lmks = det.detect_single_image(bgr)
        if np.all(lmks[:, 0] < 0):
            continue
        h, w = rgb.shape[:2]
        out[k, :, 0] = lmks[:, 0] * w
        out[k, :, 1] = lmks[:, 1] * h
    return out


def compute_lmks68_barycentric_template(
    *,
    dataset_dir: Path = _DATASET_DIR,
    template_path: Path | None = None,
    template_name: str = "bfm_template.obj",
    mesh_files: Sequence[str] = DEFAULT_MESH_FILES,
    out_pkl: Path | None = None,
    res: int = 512,
    yfov: float = np.pi / 3.0,
    fill_ratio: float = 0.73,
    flat_shading: bool = False,
    jitter_radii_px: Sequence[int] = (0, 1, 2, 4),
    debug_overlay_dir: Path | None = None,
    debug_overlay_pt_radius: int = 3,
) -> dict[str, Any]:
    if template_path is None:
        template_path = (
            DEFAULT_BFM_TEMPLATE
            if DEFAULT_BFM_TEMPLATE.is_file()
            else dataset_dir / template_name
        )
    template_path = Path(template_path)
    mesh_paths = [dataset_dir / name for name in mesh_files]
    if template_path not in mesh_paths:
        mesh_paths = [template_path] + mesh_paths
    for p in mesh_paths:
        if not p.is_file():
            raise FileNotFoundError(p)

    V_tm, faces_t = _load_obj_vf(template_path)
    ctr_template = _bfm_recenter_origin(V_tm)
    print(f"[bfm_lmks68] template={template_path.name} recenter={ctr_template}")

    variants: list[tuple[str, Path, np.ndarray]] = []
    radii: list[float] = []
    seen: set[str] = set()
    for p in mesh_paths:
        if p.stem in seen:
            continue
        seen.add(p.stem)
        Vi, Fi = _load_obj_vf(p)
        if Fi.shape != faces_t.shape or not np.array_equal(Fi, faces_t):
            raise ValueError(f"topology mismatch with template: {p}")
        Vc = Vi - ctr_template
        variants.append((p.stem, p, Vc))
        Ri = float(np.linalg.norm(Vc, axis=1).max())
        if math.isfinite(Ri) and Ri > 0:
            radii.append(Ri)
    R_max = max(radii) if radii else 1.0
    cam_d, znear, zfar, cam_pose = _canonical_camera_bundle(
        R_max, yfov=float(yfov), fill_ratio=float(fill_ratio)
    )
    P_gl = pyrender_projection_matrix(
        yfov=float(yfov), znear=znear, zfar=zfar, width=int(res), height=int(res),
    )

    faces_torch = torch.as_tensor(faces_t, dtype=torch.long)
    rgbs: list[np.ndarray] = []
    meshes_c: list[trimesh.Trimesh] = []
    print(f"[bfm_lmks68] R_max={R_max:.6f}, d={cam_d:.6f}, zn/zf={znear:.6f}/{zfar:.6f}")
    for _stem, _path, Vc in tqdm(variants, desc="render"):
        rgb = render_rgb_uint8(
            Vc,
            faces_torch,
            res=res,
            camera_distance=cam_d,
            znear=znear,
            zfar=zfar,
            fill_ratio=fill_ratio,
            yfov=float(yfov),
            flat_shading=flat_shading,
        )
        rgbs.append(rgb)
        meshes_c.append(trimesh.Trimesh(vertices=Vc, faces=faces_t, process=False))

    star_det = build_star_detector()
    lm_uv = star_landmarks_pixels_batch(rgbs, detector=star_det)

    tmpl_i = next(
        i for i, (_, p, _) in enumerate(variants) if p.resolve() == template_path.resolve()
    )
    tmpl_mesh_c = trimesh.Trimesh(vertices=variants[tmpl_i][2], faces=faces_t, process=False)

    hits_buf = np.full((lm_uv.shape[0], N_LANDMARKS, 3), np.nan, dtype=np.float64)
    jitter = jitter_uv_offsets(jitter_radii_px)

    for view_i, mesh in enumerate(meshes_c):
        uv_k = lm_uv[view_i]
        H, W = rgbs[view_i].shape[0], rgbs[view_i].shape[1]
        aspect_wh = float(W) / float(max(H, 1))
        desc = f"rays {variants[view_i][0]}"
        for k in tqdm(range(N_LANDMARKS), desc=desc, leave=False):
            if not (np.isfinite(uv_k[k, 0]) and np.isfinite(uv_k[k, 1])):
                continue
            u0, v0 = float(uv_k[k, 0]), float(uv_k[k, 1])
            placed = False
            for du, dv in jitter:
                uu, vv = u0 + du, v0 + dv
                if uu < 0 or vv < 0 or uu >= W or vv >= H:
                    continue
                ro, rd = pixel_ray_opengl_top_left_origin(
                    uu, vv, W, H, float(yfov), aspect_wh, cam_pose
                )
                hit = ray_mesh_first_positive_hit(mesh, ro, rd)
                if hit is not None:
                    hits_buf[view_i, k] = hit[0]
                    placed = True
                    break
            if not placed:
                ro, rd = pixel_ray_opengl_top_left_origin(
                    u0, v0, W, H, float(yfov), aspect_wh, cam_pose
                )
                hits_buf[view_i, k] = _closest_on_ray_to_surface(
                    mesh, ro, rd, t_min=cam_d * 0.05, t_max=cam_d * 5.0
                )

    K_views = hits_buf.shape[0]
    reproj_err_px = np.full((K_views, N_LANDMARKS), np.nan, dtype=np.float64)
    for vi in range(K_views):
        H, W = rgbs[vi].shape[0], rgbs[vi].shape[1]
        uv_repr, _ = world_points_to_pixel_uv_openvr_top_left_image(
            cam_pose, P_gl, hits_buf[vi], width=W, height=H
        )
        reproj_err_px[vi] = reprojection_residuals_px_vs_mediapipe(
            lm_uv_mp=lm_uv[vi], lm_uv_repr=uv_repr
        )
        e = reproj_err_px[vi]
        ok_mask = np.isfinite(e)
        if np.any(ok_mask):
            e_ok = np.where(ok_mask, e, np.nan)
            print(
                f"[bfm_lmks68] reproj RMS view={variants[vi][0]}: "
                f"RMS={float(np.sqrt(np.nanmean(np.square(e_ok)))):.3f}px "
                f"median={float(np.nanmedian(e_ok)):.3f}px "
                f"max={float(np.nanmax(e_ok)):.3f}px"
            )
        else:
            print(f"[bfm_lmks68] reproj RMS view={variants[vi][0]}: no finite errors")

    if debug_overlay_dir is not None:
        stems = [v[0] for v in variants]
        save_debug_lmks2d_overlays(
            rgbs,
            lm_uv,
            Path(debug_overlay_dir),
            stems=stems,
            point_radius=debug_overlay_pt_radius,
        )
        save_debug_reproj_compare_overlays(
            rgbs,
            lm_uv,
            hits_buf,
            cam_pose,
            P_gl,
            Path(debug_overlay_dir),
            stems=stems,
            pt_mp_radius=max(1, debug_overlay_pt_radius),
            pt_repr_radius=max(1, debug_overlay_pt_radius - 1),
        )

    P_world, fusion_src_view = fuse_hits_prioritize_reference_view_then_median_reference_frames(
        hits_buf, tmpl_i
    )
    for kk in range(N_LANDMARKS):
        if np.all(np.isfinite(P_world[kk])):
            continue
        vk = tmpl_mesh_c.vertices[int(kk % tmpl_mesh_c.vertices.shape[0])]
        cp, _, _ = trimesh.proximity.closest_point(tmpl_mesh_c, vk.reshape(1, 3))
        P_world[kk] = cp[0]

    closest, _, tri_id = trimesh.proximity.closest_point(tmpl_mesh_c, P_world)
    verts_f = tmpl_mesh_c.vertices[faces_t[tri_id]]
    bc = points_to_barycentric(verts_f.reshape(-1, 3, 3), closest)
    face_indices = tri_id.astype(np.int64)
    bary_coords = np.clip(bc.astype(np.float64), 0.0, 1.0)
    s = bary_coords.sum(axis=1, keepdims=True)
    bary_coords /= np.maximum(s, 1e-12)

    tri_fb = tmpl_mesh_c.vertices[faces_t[face_indices]]
    lm_xyz_from_bary = np.einsum("ij,ijk->ik", bary_coords, tri_fb)
    H_t, W_t = rgbs[tmpl_i].shape[0], rgbs[tmpl_i].shape[1]
    uv_from_bary, _ = world_points_to_pixel_uv_openvr_top_left_image(
        cam_pose, P_gl, lm_xyz_from_bary, width=W_t, height=H_t
    )
    e_bary_repr = reprojection_residuals_px_vs_mediapipe(
        lm_uv_mp=lm_uv[tmpl_i], lm_uv_repr=uv_from_bary
    )
    if np.any(np.isfinite(e_bary_repr)):
        print(
            f"[bfm_lmks68] after closest+bary, template-view reproj: "
            f"RMS={float(np.sqrt(np.nanmean(np.square(e_bary_repr)))):.3f}px "
            f"median={float(np.nanmedian(e_bary_repr)):.3f}px"
        )

    meta: dict[str, Any] = {
        "face_indices": face_indices,
        "bary_coords": bary_coords,
        "landmark_source": "star68",
        "n_landmarks": N_LANDMARKS,
        "bfm_recenter_front_verts": BFM_N_FRONT_VERTS,
        "projection_matrix_gl": P_gl,
        "reproj_err_px_per_view": reproj_err_px,
        "reproj_err_px_bary_template_view": e_bary_repr,
        "fusion_prioritize_template_view_then_median_others": True,
        "fusion_src_view_per_lm": fusion_src_view,
        "template_center": ctr_template.astype(np.float64),
        "template_obj": str(template_path),
        "mesh_files": [str(p) for p in mesh_paths],
        "resolution": int(res),
        "yfov": float(yfov),
        "fill_ratio": float(fill_ratio),
        "camera_distance": float(cam_d),
        "znear": float(znear),
        "zfar": float(zfar),
        "cam_pose_world": cam_pose,
        "P_world_fused": P_world,
        "P_world_median": P_world,
        "hits_per_view": hits_buf,
    }
    if out_pkl is not None:
        op = Path(out_pkl)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[bfm_lmks68] saved {op}")
    return meta


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="BFM STAR68 → template barycentric")
    parser.add_argument("--out", type=Path, default=_DATASET_DIR / "bfm_lmks68_bary.pkl")
    parser.add_argument("--res", type=int, default=512)
    parser.add_argument(
        "--debug-overlay-dir",
        type=Path,
        default=None,
        help="可选：写入红色 68 叠图 *_lmks2d.png 以及红绿对比 *_mp_vs_reproj.png",
    )
    parser.add_argument(
        "--debug-overlay-pt",
        type=int,
        default=2,
        help="叠图关键点半径（像素）",
    )
    args = parser.parse_args()
    compute_lmks68_barycentric_template(
        out_pkl=args.out,
        res=args.res,
        debug_overlay_dir=args.debug_overlay_dir,
        debug_overlay_pt_radius=args.debug_overlay_pt,
    )


if __name__ == "__main__":
    main_cli()
