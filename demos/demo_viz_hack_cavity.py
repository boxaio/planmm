"""用 Polyscope 可视化 HACK 网格的三个 cavity 区域（nose / mouth / eyes）。"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import polyscope as ps

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mesh import read_obj

DEFAULT_MESH = Path(
    "/home/ubuntu/MyExp/TRACK/data/hack_data/hack_template_turn_left_open_mouth.obj"
)
DEFAULT_CAVITY_PKL = Path(
    "/home/ubuntu/MyExp/TRACK/data/hack_data/HACK_cavity.pkl"
)

REGION_NAMES = ("nose", "mouth", "eyes")

# 与 hack_seg_fids 中 tab10 风格调色板一致
_REGION_COLORS = np.array(
    [
        [0.839, 0.153, 0.157],  # nose  - 红
        [0.122, 0.467, 0.706],  # mouth - 蓝
        [0.173, 0.627, 0.173],  # eyes  - 绿
    ],
    dtype=np.float64,
)
_UNASSIGNED_COLOR = (0.75, 0.75, 0.78)


def load_cavity_info(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    for name in REGION_NAMES:
        if name not in data:
            raise KeyError(f"HACK_cavity.pkl 缺少区域 {name!r}，现有: {sorted(data)}")
        if "face_ids" not in data[name]:
            raise KeyError(f"区域 {name!r} 缺少 face_ids 字段")
    return data


def build_face_region_colors(
    n_faces: int,
    cavity_info: dict,
    *,
    region_names: tuple[str, ...] = REGION_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (face_rgb, region_id)，region_id[i] == -1 表示未分配。"""
    region_id = np.full(n_faces, -1, dtype=np.int32)
    for i, name in enumerate(region_names):
        fids = np.asarray(cavity_info[name]["face_ids"], dtype=np.int64)
        fids = fids[(fids >= 0) & (fids < n_faces)]
        unset = region_id[fids] < 0
        region_id[fids[unset]] = i

    face_rgb = np.tile(np.asarray(_UNASSIGNED_COLOR, dtype=np.float64), (n_faces, 1))
    for i in range(len(region_names)):
        face_rgb[region_id == i] = _REGION_COLORS[i]
    return face_rgb, region_id


def show_hack_cavity_polyscope(
    mesh_path: Path,
    cavity_pkl: Path,
    *,
    mesh_name: str = "hack_cavity",
    show_edge: bool = True,
) -> None:
    obj = read_obj(str(mesh_path), tri=True)
    verts = np.asarray(obj.vs, dtype=np.float64)
    faces = np.asarray(obj.fvs, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"期望三角面 (F,3)，得到 {faces.shape}")

    cavity_info = load_cavity_info(cavity_pkl)
    face_rgb, region_id = build_face_region_colors(faces.shape[0], cavity_info)

    for i, name in enumerate(REGION_NAMES):
        n = int(np.sum(region_id == i))
        print(f"[demo_viz_hack_cavity] {name}: {n} faces")

    n_unassigned = int(np.sum(region_id < 0))
    print(
        f"[demo_viz_hack_cavity] mesh V={verts.shape[0]} F={faces.shape[0]} | "
        f"unassigned faces: {n_unassigned}"
    )

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_shadow_darkness(0.0)
    ps.set_up_dir("y_up")

    ps_mesh = ps.register_surface_mesh(
        mesh_name,
        vertices=verts,
        faces=faces,
        enabled=True,
        color=(0.85, 0.85, 0.85),
        edge_width=0.85 if show_edge else 0.0,
        material="clay",
        smooth_shade=True,
        back_face_policy="custom",
    )
    ps_mesh.add_color_quantity(
        "cavity_region",
        face_rgb,
        defined_on="faces",
        enabled=True,
    )
    ps.show()


def main() -> None:
    p = argparse.ArgumentParser(description="Polyscope 可视化 HACK cavity 三区域")
    p.add_argument(
        "--mesh",
        type=Path,
        default=DEFAULT_MESH,
        help="HACK 模板网格 OBJ",
    )
    p.add_argument(
        "--cavity-pkl",
        type=Path,
        default=DEFAULT_CAVITY_PKL,
        help="HACK_cavity.pkl 路径",
    )
    p.add_argument("--no-edge", action="store_true", help="关闭网格边线")
    args = p.parse_args()

    mesh_path = args.mesh.expanduser().resolve()
    cavity_pkl = args.cavity_pkl.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"网格不存在: {mesh_path}")
    if not cavity_pkl.is_file():
        raise FileNotFoundError(f"cavity pkl 不存在: {cavity_pkl}")

    show_hack_cavity_polyscope(
        mesh_path,
        cavity_pkl,
        show_edge=not args.no_edge,
    )


if __name__ == "__main__":
    main()
