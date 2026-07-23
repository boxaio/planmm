from pathlib import Path
import colorsys
import pickle
import numpy as np
import sys

import polyscope as ps
import polyscope.imgui as psim



_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MESH = _REPO_ROOT / "data" / "hack_data" / "hack_template_turn_left_open_mouth.obj"
_DEFAULT_SEG_PKL = _REPO_ROOT / "data" / "hack_data" / "HACK_seg_28.pkl"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mesh import read_obj

def _load_seg_info(seg_pkl: Path) -> dict:
    with open(seg_pkl, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {seg_pkl}, got {type(data)!r}")
    return data


def _palette(n: int) -> np.ndarray:
    cols = np.zeros((n, 3), dtype=np.float64)
    if n <= 0:
        return cols
    for i in range(n):
        h = (i / n + 0.08) % 1.0
        cols[i] = np.asarray(colorsys.hsv_to_rgb(h, 0.72, 0.95), dtype=np.float64)
    return cols


def show_hack_seg_28(
    mesh_path: Path = _DEFAULT_MESH,
    seg_pkl_path: Path = _DEFAULT_SEG_PKL,
) -> None:
    mesh_path = Path(mesh_path)
    seg_pkl_path = Path(seg_pkl_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    if not seg_pkl_path.is_file():
        raise FileNotFoundError(f"Seg pkl not found: {seg_pkl_path}")

    mesh = read_obj(str(mesh_path))
    verts = np.asarray(mesh.vs, dtype=np.float64)
    faces = np.asarray(mesh.fvs, dtype=np.int64)

    seg_info = _load_seg_info(seg_pkl_path)
    region_names = sorted(seg_info.keys())
    n_regions = len(region_names)
    colors = _palette(n_regions)

    face_region_id = np.full((faces.shape[0],), -1, dtype=np.int64)
    for rid, name in enumerate(region_names):
        fids = np.asarray(seg_info[name]["face_ids"], dtype=np.int64).reshape(-1)
        fids = fids[(fids >= 0) & (fids < faces.shape[0])]
        assign_mask = face_region_id[fids] < 0
        face_region_id[fids[assign_mask]] = rid

    face_color = np.full((faces.shape[0], 3), 0.75, dtype=np.float64)
    ok = face_region_id >= 0
    if np.any(ok):
        face_color[ok] = colors[face_region_id[ok]]

    ps.init()
    ps.set_program_name("hack_seg_28")
    ps.set_ground_plane_mode("none")

    region_mesh_names = []
    region_enabled = []
    for rid, name in enumerate(region_names):
        fids = np.flatnonzero(face_region_id == rid)
        if fids.size == 0:
            continue
        sub_faces = faces[fids]
        ps_name = f"seg28::{name}"
        ps.register_surface_mesh(
            ps_name,
            verts,
            sub_faces,
            color=colors[rid],
            edge_width=0.8,
            smooth_shade=True,
            enabled=True,
        )
        region_mesh_names.append(ps_name)
        region_enabled.append(True)

    if not region_mesh_names:
        raise RuntimeError("No region submeshes registered; check HACK_seg_28.pkl content")

    unassigned_ids = np.flatnonzero(face_region_id < 0)
    if unassigned_ids.size > 0:
        ps.register_surface_mesh(
            "seg28::unassigned",
            verts,
            faces[unassigned_ids],
            color=(0.75, 0.75, 0.75),
            edge_width=0.6,
            smooth_shade=True,
            enabled=False,
        )

    def _set_all(enabled: bool) -> None:
        for i, ps_name in enumerate(region_mesh_names):
            region_enabled[i] = enabled
            ps.get_surface_mesh(ps_name).set_enabled(enabled)

    def _ui_callback() -> None:
        psim.PushItemWidth(220)
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("HACK seg28"):
            if psim.Button("enable all"):
                _set_all(True)
            psim.SameLine()
            if psim.Button("disable all"):
                _set_all(False)
            psim.Separator()
            for i, ps_name in enumerate(region_mesh_names):
                name = ps_name.split("::", 1)[1]
                changed, checked = psim.Checkbox(name, region_enabled[i])
                if changed:
                    region_enabled[i] = checked
                    ps.get_surface_mesh(ps_name).set_enabled(checked)
            psim.TreePop()
        psim.PopItemWidth()

    ps.set_user_callback(_ui_callback)

    print(f"[demo_viz_hack_seg] mesh={mesh_path}")
    print(f"[demo_viz_hack_seg] seg={seg_pkl_path}, regions={n_regions}, faces={faces.shape[0]}")
    missing = int(np.sum(face_region_id < 0))
    if missing > 0:
        print(f"[demo_viz_hack_seg] warning: {missing} faces not assigned to any region")

    ps.show()


if __name__ == "__main__":
    show_hack_seg_28()
