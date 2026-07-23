"""Visualize all HACK LAMM regions from hack_region_info.pkl in Polyscope.

Reference: LAMM ``demo_region.py`` — base mesh + per-region colored overlays with toggles.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.hack_seg_fids import REGION_NAMES, _default_region_palette, _unwrap_regions_dict, region_names
from utils.mesh import read_obj

DEFAULT_REGION_PKL = _REPO_ROOT / 'dataset' / 'hack_region_info.pkl'
DEFAULT_TEMPLATE_OBJ = _REPO_ROOT / 'dataset' / 'hack_template.obj'

from exp_poisson.hack_control_vertices import control_verts_ids, default_control_vertices


def _control_verts_by_region() -> dict[str, list[int]]:
    """Map LAMM region name -> control vertex ids from ``control_verts_ids``."""
    out: dict[str, list[int]] = {}
    for key, vids in control_verts_ids.items():
        name = REGION_NAMES[int(key)]
        if vids:
            out[name] = [int(v) for v in vids]
    return out

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize HACK region patches (Polyscope)')
    parser.add_argument(
        '--region-info',
        type=Path,
        default=DEFAULT_REGION_PKL,
        help='Path to hack_region_info.pkl',
    )
    parser.add_argument(
        '--template',
        type=Path,
        default=DEFAULT_TEMPLATE_OBJ,
        help='Reference template mesh (.obj)',
    )
    parser.add_argument(
        '--exclude-inner',
        action='store_true',
        help='Deprecated (no inner regions in the 10-patch layout); kept for compatibility.',
    )
    parser.add_argument(
        '--control-radius',
        type=float,
        default=0.0038,
        help='Point cloud radius for control vertices (default: auto from bbox).',
    )
    return parser.parse_args()


def load_regions(pkl_path: Path) -> dict[str, dict]:
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    return _unwrap_regions_dict(raw)


def build_region_submesh(
    verts: np.ndarray,
    region_entry: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sub_verts, sub_faces, global_vids) for one region."""
    vids = np.asarray(region_entry['vids'], dtype=np.int64).reshape(-1)
    patch_faces = np.asarray(region_entry['patch_faces'], dtype=np.int64)
    if patch_faces.ndim != 2 or patch_faces.shape[1] != 3:
        raise ValueError(f'expected patch_faces (F, 3), got {patch_faces.shape}')
    return verts[vids].astype(np.float64), patch_faces, vids


def main():
    args = parse_args()
    if not args.region_info.is_file():
        raise FileNotFoundError(f'region info not found: {args.region_info}')
    if not args.template.is_file():
        raise FileNotFoundError(f'template mesh not found: {args.template}')

    regions = load_regions(args.region_info)
    names = [n for n in region_names if n in regions]
    extra = sorted(set(regions.keys()) - set(names))
    names.extend(extra)
    if args.exclude_inner:
        names = [n for n in names if 'inner' not in n]

    obj = read_obj(str(args.template), tri=True)
    verts = np.asarray(obj.vs, dtype=np.float64)
    faces = np.asarray(obj.fvs, dtype=np.int64)
    num_vertices = verts.shape[0]
    num_faces = faces.shape[0]

    print(f'Loaded {len(names)} regions from {args.region_info}')
    print(f'Template: V={num_vertices}, F={num_faces}')

    region_meshes: dict[str, dict] = {}
    region_vids: dict[str, np.ndarray] = {}
    for name in names:
        sub_v, sub_f, vids = build_region_submesh(verts, regions[name])
        region_meshes[name] = {'vertices': sub_v, 'faces': sub_f}
        region_vids[name] = vids
        print(f'  {name}: |V|={sub_v.shape[0]}, |F|={sub_f.shape[0]}')

    palette = _default_region_palette(len(names))
    default_color = np.array([0.75, 0.75, 0.75], dtype=np.float64)
    region_colors = {name: palette[i] for i, name in enumerate(names)}
    if 'Eyes' in region_colors:
        region_colors['Eyes'] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    check_boxes = {name: False for name in names}
    transparency = 1.0

    control_by_region = _control_verts_by_region()
    bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    control_radius = args.control_radius
    if control_radius is None:
        control_radius = max(bbox_diag * 0.004, 1e-4)

    ps.init()
    ps.set_ground_plane_mode('shadow_only')
    ps.set_shadow_darkness(0.0)
    ps.set_up_dir('y_up')

    group = ps.create_group('HACK_regions')

    control_check_boxes: dict[str, bool] = {}
    control_pc_names: dict[str, str] = {}
    for region_name, vids in control_by_region.items():
        vids_arr = np.asarray(vids, dtype=np.int64)
        if vids_arr.min() < 0 or vids_arr.max() >= num_vertices:
            raise ValueError(
                f'control verts for {region_name!r} out of range: '
                f'max={int(vids_arr.max())}, V={num_vertices}'
            )
        pts = verts[vids_arr]
        pc_name = f'control::{region_name}'
        control_pc_names[region_name] = pc_name
        control_check_boxes[region_name] = True
        rgb = region_colors.get(region_name, np.array([1.0, 0.2, 0.2]))
        ps.register_point_cloud(
            pc_name,
            pts,
            radius=control_radius,
            color=tuple(float(c) for c in rgb),
            enabled=True,
        )
        print(f'  control {region_name}: {len(vids)} verts -> {pc_name}')

    vertex_colors = np.tile(default_color, (num_vertices, 1))
    head_mesh = ps.register_surface_mesh(
        'BASE',
        verts,
        faces,
        color=default_color,
        material='clay',
        smooth_shade=True,
        transparency=transparency,
    )
    head_mesh.add_color_quantity(
        'region_highlight',
        vertex_colors,
        defined_on='vertices',
        enabled=True,
    )
    head_mesh.add_to_group(group)

    region_ps_meshes: dict[str, ps.SurfaceMesh] = {}
    for name in names:
        mesh = ps.register_surface_mesh(
            name=f'region::{name}',
            vertices=region_meshes[name]['vertices'],
            faces=region_meshes[name]['faces'],
            enabled=False,
            color=tuple(float(c) for c in region_colors[name]),
            material='flat',
            smooth_shade=True,
            transparency=1.0,
        )
        mesh.add_to_group(group)
        region_ps_meshes[name] = mesh

    group.set_enabled(True)
    group.set_show_child_details(True)

    def update_region_highlight():
        colors = np.tile(default_color, (num_vertices, 1))
        for name in names:
            if check_boxes[name]:
                colors[region_vids[name]] = region_colors[name]
        head_mesh.add_color_quantity(
            'region_highlight',
            colors,
            defined_on='vertices',
            enabled=True,
        )

    def ui_callback():
        nonlocal transparency, control_radius

        psim.PushItemWidth(220)

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode('HACK template'):
            psim.TextUnformatted(f'#vertices: {num_vertices}   #faces: {num_faces}')
            psim.TextUnformatted(f'#regions shown: {len(names)}')
            psim.TreePop()

        changed, transparency = psim.SliderFloat(
            'base transparency', transparency, v_min=0.0, v_max=1.0,
        )
        if changed:
            head_mesh.set_transparency(transparency)

        psim.Separator()
        if psim.Button('enable all'):
            for name in names:
                check_boxes[name] = True
                region_ps_meshes[name].set_enabled(True)
            update_region_highlight()
        psim.SameLine()
        if psim.Button('disable all'):
            for name in names:
                check_boxes[name] = False
                region_ps_meshes[name].set_enabled(False)
            update_region_highlight()

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode('Regions'):
            highlight_changed = False
            for name in names:
                changed_cb, checked = psim.Checkbox(name, check_boxes[name])
                if changed_cb:
                    check_boxes[name] = checked
                    highlight_changed = True
                region_ps_meshes[name].set_enabled(check_boxes[name])

            if highlight_changed:
                update_region_highlight()
            psim.TreePop()

        if control_pc_names:
            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            if psim.TreeNode('Control vertices'):
                if psim.Button('show all##ctrl'):
                    for rn in control_pc_names:
                        control_check_boxes[rn] = True
                        ps.get_point_cloud(control_pc_names[rn]).set_enabled(True)
                psim.SameLine()
                if psim.Button('hide all##ctrl'):
                    for rn in control_pc_names:
                        control_check_boxes[rn] = False
                        ps.get_point_cloud(control_pc_names[rn]).set_enabled(False)
                changed_r, control_radius_ui = psim.SliderFloat(
                    'point radius', control_radius, v_min=1e-5, v_max=bbox_diag * 0.02,
                )
                if changed_r:
                    control_radius = control_radius_ui
                    for rn in control_pc_names:
                        ps.get_point_cloud(control_pc_names[rn]).set_radius(control_radius)
                psim.Separator()
                for region_name, pc_name in control_pc_names.items():
                    n_ctrl = len(control_by_region[region_name])
                    label = f'{region_name} ({n_ctrl})##ctrl_{region_name}'
                    changed_cb, checked = psim.Checkbox(label, control_check_boxes[region_name])
                    if changed_cb:
                        control_check_boxes[region_name] = checked
                        ps.get_point_cloud(pc_name).set_enabled(checked)
                psim.TreePop()

        psim.PopItemWidth()

    ps.set_user_callback(ui_callback)
    ps.show()


if __name__ == '__main__':
    main()
