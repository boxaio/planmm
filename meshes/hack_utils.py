import colorsys
import pickle
import numpy as np
from typing import Any, Dict, Optional, Union

import polyscope as ps
import polyscope.imgui as psim
from pathlib import Path



_DEFAULT_HACK_SEG_28_PKL = (
    Path(__file__).resolve().parent.parent / "data" / "hack_data" / "HACK_seg_28.pkl"
)


def load_hack_seg_28_patches(
    pkl_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load all 28 semantic-region patches (submeshes) from HACK_seg_28.pkl.

    Each patch dict contains:
        - face_ids: face indices on the template mesh (list or ndarray)
        - faces: submesh triangles (F, 3), local vertex indices
        - vert_ids: vertex indices on the template mesh
        - verts: submesh vertex positions (V, 3)

    Default path: repo_root/data/hack_data/HACK_seg_28.pkl.
    """
    path = Path(pkl_path) if pkl_path is not None else _DEFAULT_HACK_SEG_28_PKL
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"expected dict in {path}, got {type(data)!r}")
    return data


def extract_submesh(vertices: np.ndarray, faces: np.ndarray, face_indices: np.ndarray):
    """
    Build a submesh induced by a subset of faces.

    Args:
        vertices: (V, 3) full mesh vertices.
        faces: (F, 3) triangle indices into vertices.
        face_indices: indices of faces to keep (1D array-like).

    Returns:
        sub_vertices: (V', 3) extracted vertices.
        sub_faces: (F', 3) faces with remapped local vertex indices.
        vertex_map: old global vertex index -> new local index.
    """
    assert face_indices is not None

    all_vertex_indices = np.unique(faces[face_indices].flatten())

    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(all_vertex_indices)}
    
    sub_vertices = vertices[all_vertex_indices]
    sub_faces = np.vectorize(vertex_map.get)(faces[face_indices])
    
    return sub_vertices, sub_faces, vertex_map



if __name__ == "__main__":
    # Interactive viewer for the 28 patches stored in HACK_seg_28.pkl.
    patches = load_hack_seg_28_patches()
    region_names = sorted(patches.keys())
    num_regions = len(region_names)

    region_meshes = {}
    region_colors = {}
    for i, name in enumerate(region_names):
        p = patches[name]
        region_meshes[name] = {
            "vertices": np.asarray(p["verts"], dtype=np.float64),
            "faces": np.asarray(p["faces"], dtype=np.int64),
        }
        h = (i / max(num_regions, 1) + 0.07) % 1.0
        region_colors[name] = np.array(colorsys.hsv_to_rgb(h, 0.72, 0.96), dtype=np.float64)

    check_boxes = [False] * num_regions
    edge_width = 1.0

    ps.init()
    ps.set_ground_plane_mode("none")

    group = ps.create_group("HACK_seg_28")

    def _patch_ps_name(region: str) -> str:
        # Polyscope names must be unique; keep original region labels readable.
        return f"seg28::{region}"

    for name in region_names:
        mesh = ps.register_surface_mesh(
            name=_patch_ps_name(name),
            vertices=region_meshes[name]["vertices"],
            faces=region_meshes[name]["faces"],
            enabled=False,
            color=region_colors[name],
            edge_color=(0.0, 0.0, 0.0),
            edge_width=edge_width,
            transparency=1.0,
        )
        mesh.add_to_group(group)

    group.set_enabled(True)
    group.set_show_child_details(True)

    total_v = sum(region_meshes[n]["vertices"].shape[0] for n in region_names)
    total_f = sum(region_meshes[n]["faces"].shape[0] for n in region_names)

    def ui_callback():
        global edge_width, check_boxes

        psim.PushItemWidth(240)

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("HACK_seg_28 (pickle)"):
            psim.TextUnformatted(f"regions: {num_regions}")
            psim.TextUnformatted(
                f"sum |V| over patches: {total_v}    sum |F|: {total_f}"
            )
            psim.TreePop()

        psim.Separator()
        changed_ew, edge_width = psim.SliderFloat(
            "patch edge width", edge_width, v_min=0.0, v_max=3.0
        )
        if changed_ew:
            for name in region_names:
                ps.get_surface_mesh(_patch_ps_name(name)).set_edge_width(edge_width)

        psim.Separator()
        if psim.Button("enable all"):
            check_boxes = [True] * num_regions
            for name in region_names:
                ps.get_surface_mesh(_patch_ps_name(name)).set_enabled(True)
        psim.SameLine()
        if psim.Button("disable all"):
            check_boxes = [False] * num_regions
            for name in region_names:
                ps.get_surface_mesh(_patch_ps_name(name)).set_enabled(False)

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Regions"):
            for i, name in enumerate(region_names):
                changed, check = psim.Checkbox(name, check_boxes[i])
                check_boxes[i] = check
                ps_name = _patch_ps_name(name)
                if changed:
                    ps.get_surface_mesh(ps_name).set_enabled(check_boxes[i])
            psim.TreePop()

        psim.PopItemWidth()

    ps.set_user_callback(ui_callback)
    ps.show()




