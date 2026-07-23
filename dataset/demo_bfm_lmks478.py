"""
在 Polyscope 中显示 BFM 模板网格及 ``bfm_lmks478.py`` 生成的 478 个 3D 关键点。

默认读取 ``dataset/bfm_lmks478_bary.pkl``；重心坐标减去模板质心后与 ``compute_lmks478_barycentric_template`` 中一致，
显示时顶点加回同一质心以保持与 OBJ 对齐。
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import polyscope as ps

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATASET_DIR = Path(__file__).resolve().parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mesh import read_obj


DEFAULT_PKL = _DATASET_DIR / "bfm_lmks478_bary.pkl"


def barycentric_to_world_xyz(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_indices: np.ndarray,
    bary_coords: np.ndarray,
    template_center: np.ndarray,
) -> np.ndarray:
    """
    ``vertices``：OBJ 顶点 (V,3)；与 ``compute_lmks478_barycentric_template`` 一样先做 ``Vc = V - center``，
    再按面索引与重心加权，最后 ``+ template_center`` 回到 OBJ 空间。
    """
    V = np.asarray(vertices, dtype=np.float64)
    ctr = np.asarray(template_center, dtype=np.float64).reshape(3)
    F = np.asarray(faces, dtype=np.int64)
    fi = np.asarray(face_indices, dtype=np.int64)
    bc = np.asarray(bary_coords, dtype=np.float64)
    if bc.shape[-1] != 3:
        raise ValueError(f"bary_coords last dim must be 3, got {bc.shape}")
    Vc = V - ctr
    tri_ix = F[fi]
    corners = Vc[tri_ix]
    lm_c = np.sum(bc[:, :, np.newaxis] * corners, axis=1)
    return lm_c + ctr


def run(
    *,
    pkl_path: Path,
    point_radius_rel: float = 2e-3,
) -> None:
    pkl_path = Path(pkl_path)
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"未找到关键点文件：{pkl_path}\n"
            "请先运行：python dataset/bfm_lmks478.py --out <path.pkl>"
        )
    with open(pkl_path, "rb") as f:
        meta = pickle.load(f)

    face_indices = meta["face_indices"]
    bary_coords = meta["bary_coords"]
    template_center = np.asarray(meta["template_center"], dtype=np.float64)
    tmpl_path = Path(meta.get("template_obj", _DATASET_DIR / "bfm_template.obj"))
    if not tmpl_path.is_file():
        raise FileNotFoundError(f"template OBJ 不存在：{tmpl_path}")

    obj = read_obj(str(tmpl_path), tri=True)
    verts = np.asarray(obj.vs, dtype=np.float64)
    faces = np.asarray(obj.fvs, dtype=np.int64)

    lm_xyz = barycentric_to_world_xyz(
        verts, faces, face_indices, bary_coords, template_center
    )

    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    rad = max(diag * point_radius_rel, 1e-6)

    lm_idx = np.arange(lm_xyz.shape[0], dtype=np.float64)

    ps.init()
    ps.set_program_name("bfm_lmks478")
    ps.set_ground_plane_mode("none")

    ps.register_surface_mesh(
        "bfm_template",
        verts,
        faces,
        color=(0.82, 0.82, 0.84),
        edge_width=1.0,
        material="clay",
        smooth_shade=False,
    )
    pcs = ps.register_point_cloud(
        "mediapipe_478",
        lm_xyz,
        # radius=rad,
        radius=0.003,
        color=(0.95, 0.05, 0.05),
    )
    # pcs.add_scalar_quantity("lmk_idx", lm_idx, enabled=True)

    print(f"[demo_bfm_lmks478] mesh {tmpl_path} | V={verts.shape[0]} F={faces.shape[0]} | lm={lm_xyz.shape[0]}")
    ps.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Polyscope：BFM 模板 + MediaPipe478 3D 关键点")
    ap.add_argument("--pkl", type=Path, default=DEFAULT_PKL, help="bfm_lmks478 输出的 pickle")
    ap.add_argument(
        "--point-radius-rel",
        type=float,
        default=1e-3,
        help="点球半径相对包围盒对角线的比例",
    )
    args = ap.parse_args()
    run(pkl_path=args.pkl, point_radius_rel=args.point_radius_rel)


if __name__ == "__main__":
    main()
