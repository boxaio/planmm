import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polyscope as ps
import polyscope.imgui as psim


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from meshes.hack_utils import load_hack_seg_28_patches
from utils.mesh import read_obj

_DATASET_DIR = Path(__file__).resolve().parent

REGION_INFO_PKL = _DATASET_DIR / 'hack_region_info.pkl'
DEFAULT_SEG28_PKL = _REPO_ROOT / 'data' / 'hack_data' / 'HACK_seg_28.pkl'
DEFAULT_TEMPLATE_OBJ = _DATASET_DIR / 'hack_template.obj'

# LAMM-style 10 regions (order matches patch index 0..9).
REGION_NAMES: Dict[int, str] = {
    0: 'LeftFace',
    1: 'RightFace',
    2: 'Ears',
    3: 'Neck',
    4: 'Chin',
    5: 'Skull',
    6: 'Forehead',
    7: 'Eyes',
    8: 'Nose',
    9: 'Mouth',
}

# Each LAMM region is a union of HACK_seg_28 sub-regions (face_ids merged).
REGION_SEG28_PARTS: Dict[str, Tuple[str, ...]] = {
    'LeftFace': ('cheek_0', 'jaw_0', 'fold_0'),
    'RightFace': ('cheek_1', 'jaw_1', 'fold_1'),
    'Ears': ('ear_0', 'ear_1'),
    'Neck': ('neck_f', 'neck_b'),
    'Chin': ('chin_0', 'chin_1'),
    'Skull': ('skull',),
    'Forehead': ('forehead', 'temple_0', 'temple_1'),
    'Eyes': ('eye_0', 'eye_1', 'eye_2', 'eye_3', 'eye_4', 'eye_5', 'frown'),
    'Nose': ('nose_0', 'nose_1'),
    'Mouth': ('mouth_0', 'mouth_1', 'philtrum'),
}

region_names: List[str] = [REGION_NAMES[i] for i in range(len(REGION_NAMES))]


def _seg28_face_ids(seg28: dict, parts: Sequence[str]) -> np.ndarray:
    merged: List[int] = []
    for part in parts:
        if part not in seg28:
            raise KeyError(
                f"seg28 part {part!r} not found; available={sorted(seg28.keys())}"
            )
        entry = seg28[part]
        fids = entry.get('face_ids', entry.get('fids'))
        if fids is None:
            raise KeyError(f"seg28 part {part!r} has no face_ids")
        merged.extend(int(x) for x in np.asarray(fids).reshape(-1))
    return np.unique(np.asarray(merged, dtype=np.int64))


def collect_fids_for_lamm_region(
    name: str,
    seg28: dict | None = None,
    seg28_pkl: str | Path | None = None,
) -> np.ndarray:
    """Merge face_ids of all HACK_seg_28 parts belonging to one LAMM region."""
    if name not in REGION_SEG28_PARTS:
        raise KeyError(f'unknown LAMM region={name!r}, configured: {list(REGION_SEG28_PARTS)}')
    if seg28 is None:
        seg28 = load_hack_seg_28_patches(seg28_pkl or DEFAULT_SEG28_PKL)
    return _seg28_face_ids(seg28, REGION_SEG28_PARTS[name])


def face_indices_to_vertex_indices(faces: np.ndarray, fids: np.ndarray) -> np.ndarray:
    """``faces`` (F,3); ``fids`` face indices. Returns sorted unique vertex indices on those faces."""
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f'faces must be (F,3), got {faces.shape}')
    if fids.size == 0:
        return np.array([], dtype=np.int64)
    if int(fids.min()) < 0 or int(fids.max()) >= faces.shape[0]:
        raise ValueError(
            f'fids out of range: max={int(fids.max())}, num faces F={faces.shape[0]}'
        )
    tri = faces[fids]
    return np.unique(tri.reshape(-1).astype(np.int64, copy=False))


def boundary_vertex_indices(faces: np.ndarray, fids: np.ndarray) -> np.ndarray:
    """Global vertex indices on the boundary of one patch (triangle set given by ``fids``).

    For each undirected mesh edge, count how many faces whose index lies in ``fids`` use it.
    Edges with count ``1`` are boundary edges; their endpoints are boundary vertices.
    Returns sorted unique vertex indices."""
    if fids.size == 0:
        return np.array([], dtype=np.int64)
    edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
    ff = faces.astype(np.int64, copy=False)
    ffids = np.asarray(fids, dtype=np.int64).reshape(-1)
    for fi in ffids:
        a, b, c = map(int, ff[fi])
        for u, v in ((a, b), (b, c), (c, a)):
            e = (u, v) if u < v else (v, u)
            edge_count[e] += 1
    boundary: List[int] = []
    for (u, v), c in edge_count.items():
        if c == 1:
            boundary.extend((u, v))
    return np.unique(np.asarray(boundary, dtype=np.int64))


def patch_faces_local(
    faces: np.ndarray, fids: np.ndarray, vids: np.ndarray, num_vertices: int
) -> np.ndarray:
    """Triangles of a patch as **local** vertex indices into ``vids`` (order preserved), shape ``(F, 3)``.

    ``faces`` is the full mesh ``(F_tot, 3)`` global vertex indices; ``fids`` selects rows; ``vids``
    lists the patch's unique global vertex ids (same rows as in ``build_region_info_dict``).
    """
    faces = np.asarray(faces, dtype=np.int64)
    fids = np.asarray(fids, dtype=np.int64).reshape(-1)
    vids = np.asarray(vids, dtype=np.int64).reshape(-1)
    if fids.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    tri_g = faces[fids]
    g2l = np.full(int(num_vertices), -1, dtype=np.int64)
    g2l[vids] = np.arange(vids.shape[0], dtype=np.int64)
    tri_l = g2l[tri_g]
    if (tri_l < 0).any():
        raise RuntimeError(
            'patch_faces_local: a face references a vertex not in vids; '
            'check fids / face_indices_to_vertex_indices consistency.'
        )
    return tri_l


def build_neck_split_patch_info(
    reference_obj: Union[str, Path],
    neck_region: str = 'Neck',
) -> List[Dict[str, np.ndarray]]:
    """Two-patch split: ``neck_region`` vs all remaining faces (LAMM-style overlap at seam).

    Returns ``[neck_patch, non_neck_patch]`` with ``fids``, ``vids``, ``boundary_vids``, ``patch_faces``.
    Boundary vertices (185 for HACK) appear in both patches; merge outputs by summation.
    """
    ref = Path(reference_obj)
    if not ref.is_file():
        raise FileNotFoundError(f'reference_obj not found: {ref}')

    regions = build_region_info_dict(reference_obj)
    if neck_region not in regions:
        raise KeyError(f'neck_region {neck_region!r} not in {sorted(regions)}')

    obj = read_obj(str(ref), tri=True)
    faces = np.asarray(obj.fvs, dtype=np.int64)
    num_vertices = int(np.asarray(obj.vs).shape[0])

    neck_fids = np.asarray(regions[neck_region]['fids'], dtype=np.int64)
    all_fids = np.arange(faces.shape[0], dtype=np.int64)
    non_neck_fids = np.setdiff1d(all_fids, neck_fids)

    patches: List[Dict[str, np.ndarray]] = []
    for name, fids in (('neck', neck_fids), ('non_neck', non_neck_fids)):
        vids = face_indices_to_vertex_indices(faces, fids)
        patches.append({
            'name': name,
            'fids': fids,
            'vids': vids,
            'boundary_vids': boundary_vertex_indices(faces, fids),
            'patch_faces': patch_faces_local(faces, fids, vids, num_vertices),
        })
    return patches


def save_neck_split_patch_pkl(
    reference_obj: Union[str, Path] = DEFAULT_TEMPLATE_OBJ,
    out_path: Union[str, Path, None] = None,
    neck_region: str = 'Neck',
) -> Path:
    out = Path(out_path or (_DATASET_DIR / 'hack_neck_split_patches.pkl'))
    data = {
        'meta': {'neck_region': neck_region, 'reference_obj': str(reference_obj)},
        'patches': build_neck_split_patch_info(reference_obj, neck_region=neck_region),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_neck_split_patch_pkl(path: Union[str, Path, None] = None) -> List[Dict[str, np.ndarray]]:
    p = Path(path or (_DATASET_DIR / 'hack_neck_split_patches.pkl'))
    if not p.is_file():
        raise FileNotFoundError(f'neck split pkl not found: {p}')
    with open(p, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'patches' in data:
        return data['patches']
    if isinstance(data, list):
        return data
    raise ValueError(f'unexpected neck split pkl format: {type(data)}')


def build_region_info_dict(
    reference_obj: Union[str, Path] = DEFAULT_TEMPLATE_OBJ,
    seg28_pkl: Union[str, Path, None] = None,
) -> dict:
    """
    Build per-region dicts from HACK_seg_28 unions (``REGION_SEG28_PARTS``).

    Each region dict contains ``fids``, ``vids``, ``boundary_vids``, ``patch_faces``.
    """
    ref = Path(reference_obj)
    if not ref.is_file():
        raise FileNotFoundError(f'reference_obj not found: {ref}')

    missing = [n for n in region_names if n not in REGION_SEG28_PARTS]
    if missing:
        raise ValueError(f'region_names missing seg28 mapping: {missing}')

    extra = set(REGION_SEG28_PARTS) - set(region_names)
    if extra:
        raise ValueError(f'REGION_SEG28_PARTS has unlisted regions: {sorted(extra)}')

    seg28 = load_hack_seg_28_patches(seg28_pkl or DEFAULT_SEG28_PKL)

    obj = read_obj(str(ref), tri=True)
    faces = np.asarray(obj.fvs, dtype=np.int64)
    num_vertices = int(np.asarray(obj.vs).shape[0])
    num_faces = faces.shape[0]

    regions_out: Dict[str, Dict[str, np.ndarray]] = {}
    assigned = np.zeros(num_faces, dtype=bool)

    for name in region_names:
        fids = collect_fids_for_lamm_region(name, seg28=seg28)
        if fids.size == 0:
            raise ValueError(f'region {name!r} has zero faces after seg28 merge')
        if int(fids.min()) < 0 or int(fids.max()) >= num_faces:
            raise ValueError(
                f'region {name!r} fids out of range: max={int(fids.max())}, F={num_faces}'
            )
        overlap = int(np.sum(assigned[fids]))
        if overlap > 0:
            print(
                f'[build_region_info_dict] warning: {name} overlaps '
                f'{overlap} faces already assigned to prior regions'
            )
        assigned[fids] = True

        vids = face_indices_to_vertex_indices(faces, fids)
        boundary_vids = boundary_vertex_indices(faces, fids)
        patch_faces = patch_faces_local(faces, fids, vids, num_vertices)
        regions_out[name] = {
            'fids': fids,
            'vids': vids,
            'boundary_vids': boundary_vids,
            'patch_faces': patch_faces,
            'seg28_parts': list(REGION_SEG28_PARTS[name]),
        }
        print(
            f'  {name:10s} parts={REGION_SEG28_PARTS[name]} '
            f'|F|={fids.size} |V|={vids.size} |boundary|={boundary_vids.size}'
        )

    n_unassigned = int(np.sum(~assigned))
    if n_unassigned > 0:
        print(
            f'[build_region_info_dict] warning: {n_unassigned}/{num_faces} '
            f'template faces not covered by any LAMM region'
        )

    return regions_out


def _unwrap_regions_dict(data: dict) -> Dict[str, Dict[str, np.ndarray]]:
    """Accept pickle with top-level ``regions``, flat region dict, or extra ``meta`` key."""
    if 'regions' in data and isinstance(data['regions'], dict):
        return data['regions']
    return {
        k: v
        for k, v in data.items()
        if k != 'meta' and isinstance(v, dict) and 'fids' in v
    }


def _default_region_palette(n: int) -> np.ndarray:
    """``(n, 3)`` RGB in [0,1], similar to tab10 for distinguishable colors."""
    base = np.array(
        [
            [0.122, 0.467, 0.706],
            [1.000, 0.498, 0.055],
            [0.173, 0.627, 0.173],
            [0.839, 0.153, 0.157],
            [0.580, 0.404, 0.741],
            [0.549, 0.337, 0.294],
            [0.890, 0.467, 0.761],
            [0.498, 0.498, 0.498],
        ],
        dtype=np.float64,
    )
    if n <= base.shape[0]:
        return base[:n].copy()
    # Cycle and slightly vary brightness when n exceeds base colors
    out = np.tile(base, (1 + n // base.shape[0], 1))[:n].copy()
    for i in range(base.shape[0], n):
        out[i] *= 0.85 + 0.15 * ((i % 3) / 2.0)
    return out


def show_hack_regions_polyscope(
    reference_obj: Union[str, Path] = DEFAULT_TEMPLATE_OBJ,
    *,
    region_info: Union[None, dict, str, Path] = None,
    names: Optional[Sequence[str]] = None,
    mesh_name: str = 'hack_regions',
    show_edge: bool = True,
    smooth_shade: bool = False,
    unassigned_color: Tuple[float, float, float] = (0.55, 0.55, 0.58),
    boundary_vertex_patches: Union[None, str, Sequence[str]] = None,
    boundary_vertices_gui: bool = False,
    boundary_point_radius: Optional[float] = None,
) -> None:
    """Show the reference mesh in Polyscope with per-region face coloring.

    - ``region_info``: ``None`` builds from this module's rules; or pass a dict from
      ``build_region_info_dict``, ``load_region_info_pkl()``, or a path to a ``.pkl``.
    - ``names``: draw order (default ``region_names``); overlapping faces go to the first region only.
    - ``boundary_vertex_patches``: When **not** using the GUI, ``None``/empty skips boundary point clouds;
      otherwise registers the listed patches. If ``boundary_vertices_gui=True``, still sets **initial**
      visibility for those point clouds.
    - ``boundary_vertices_gui``: If ``True``, preregisters boundary vertex point clouds for each patch and
      exposes **ImGui checkboxes** via ``set_user_callback``; patches with no boundary only show a label.
    - ``boundary_point_radius``: point sprite radius; ``None`` scales from mesh bounding-box diagonal.
    """
    ref = Path(reference_obj)
    if not ref.is_file():
        raise FileNotFoundError(f'reference_obj not found: {ref}')

    obj = read_obj(str(ref), tri=True)
    verts = np.asarray(obj.vs, dtype=np.float64)
    faces = np.asarray(obj.fvs, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f'expected triangle mesh faces (F,3), got {faces.shape}')
    f_count = int(faces.shape[0])

    if region_info is None:
        regions = build_region_info_dict(reference_obj)
    elif isinstance(region_info, (str, Path)):
        with open(Path(region_info), 'rb') as f:
            raw = pickle.load(f)
        regions = _unwrap_regions_dict(raw)
    else:
        regions = _unwrap_regions_dict(region_info)

    order = list(names if names is not None else region_names)
    for nm in order:
        if nm not in regions:
            raise KeyError(f'region_info missing {nm!r}, have: {sorted(regions)}')

    if boundary_vertex_patches is None:
        patch_list: Tuple[str, ...] = ()
    elif isinstance(boundary_vertex_patches, str):
        patch_list = (boundary_vertex_patches,)
    else:
        patch_list = tuple(boundary_vertex_patches)

    name_to_ci = {nm: i for i, nm in enumerate(order)}

    rid = np.full(f_count, -1, dtype=np.int32)
    for i, name in enumerate(order):
        fids = np.asarray(regions[name]['fids'], dtype=np.int64)
        fids = fids[(fids >= 0) & (fids < f_count)]
        unset = rid[fids] < 0
        rid[fids[unset]] = i

    palette = _default_region_palette(len(order))
    face_rgb = np.tile(np.asarray(unassigned_color, dtype=np.float64), (f_count, 1))
    for i in range(len(order)):
        face_rgb[rid == i] = palette[i]

    n_unassigned = int(np.sum(rid < 0))
    if n_unassigned > 0:
        print(f'[show_hack_regions_polyscope] faces not assigned to any region: {n_unassigned}')

    ps.init()
    ps_mesh = ps.register_surface_mesh(
        mesh_name,
        vertices=verts,
        faces=faces,
        enabled=True,
        color=(0.85, 0.85, 0.85),
        edge_width=0.85 if show_edge else 0.0,
        material='clay',
        smooth_shade=smooth_shade,
        back_face_policy='custom',
    )
    ps_mesh.add_color_quantity(
        'region_color',
        face_rgb,
        defined_on='faces',
        enabled=True,
    )

    if patch_list:
        for name in patch_list:
            if name not in regions:
                raise KeyError(
                    f'boundary_vertex_patches has unknown region {name!r}, have: {sorted(regions)}'
                )
            if name not in name_to_ci:
                raise ValueError(
                    f'boundary_vertex_patches: {name!r} not in current draw order={order}; '
                    'add it to ``names`` or remove it from boundary_vertex_patches'
                )

    bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    pt_rad_default = boundary_point_radius
    if pt_rad_default is None:
        pt_rad_default = max(bbox_diag * 0.001, 1e-6)
    v_count = int(verts.shape[0])

    def collect_boundary_xyz(reg_name: str) -> Optional[np.ndarray]:
        """Returns ``None`` if the region has no boundary vertices."""
        entry = regions[reg_name]
        bd = entry.get('boundary_vids')
        if bd is None:
            fids_i = np.asarray(entry['fids'], dtype=np.int64)
            bd = boundary_vertex_indices(faces, fids_i)
        else:
            bd = np.asarray(bd, dtype=np.int64).reshape(-1)
        bd = bd[(bd >= 0) & (bd < v_count)]
        if bd.size == 0:
            return None
        return verts[bd]

    if boundary_vertices_gui:
        boundary_pc_names: List[Optional[str]] = [None] * len(order)
        check_boxes: List[bool] = [False] * len(order)
        for nm in patch_list:
            if nm in name_to_ci:
                check_boxes[name_to_ci[nm]] = True

        for i, name in enumerate(order):
            pts = collect_boundary_xyz(name)
            if pts is None:
                continue
            pc_nm = f'{mesh_name}_boundary_{name}'
            boundary_pc_names[i] = pc_nm
            rgb = tuple(float(x) for x in palette[name_to_ci[name]])
            ps.register_point_cloud(
                pc_nm,
                points=pts,
                radius=pt_rad_default,
                enabled=check_boxes[i],
                color=rgb,
            )

        def boundary_gui_callback() -> None:
            psim.PushItemWidth(280)
            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            if psim.TreeNode('Patch boundary vertices'):
                if psim.Button('Show all'):
                    for j in range(len(order)):
                        pc_nm = boundary_pc_names[j]
                        if pc_nm is None:
                            continue
                        check_boxes[j] = True
                        ps.get_point_cloud(pc_nm).set_enabled(True)
                psim.SameLine()
                if psim.Button('Hide all'):
                    for j in range(len(order)):
                        pc_nm = boundary_pc_names[j]
                        if pc_nm is None:
                            continue
                        check_boxes[j] = False
                        ps.get_point_cloud(pc_nm).set_enabled(False)
                psim.Separator()
                for j, rn in enumerate(order):
                    pc_nm = boundary_pc_names[j]
                    if pc_nm is None:
                        psim.TextUnformatted(f'{rn} (no boundary vertices)')
                        continue
                    chk_label = f'{rn}##bfm_bd_{rn}'
                    changed, ck = psim.Checkbox(chk_label, check_boxes[j])
                    check_boxes[j] = ck
                    if changed:
                        ps.get_point_cloud(pc_nm).set_enabled(check_boxes[j])
                psim.TreePop()
            psim.PopItemWidth()

        ps.set_user_callback(boundary_gui_callback)
    elif patch_list:
        for name in patch_list:
            ci = name_to_ci[name]
            pts = collect_boundary_xyz(name)
            if pts is None:
                continue
            rgb = tuple(float(x) for x in palette[ci])
            ps.register_point_cloud(
                f'{mesh_name}_boundary_{name}',
                points=pts,
                radius=pt_rad_default,
                enabled=True,
                color=rgb,
            )

    ps.show()


def save_region_info_pkl(
    reference_obj: Union[str, Path] = DEFAULT_TEMPLATE_OBJ,
    out_path: Union[str, Path, None] = None,
    seg28_pkl: Union[str, Path, None] = None,
) -> Path:
    """Write ``hack_region_info.pkl`` from HACK_seg_28 region unions."""

    out = Path(REGION_INFO_PKL if out_path is None else out_path)
    print(f'Building {len(region_names)} LAMM regions from {seg28_pkl or DEFAULT_SEG28_PKL}')
    data = build_region_info_dict(reference_obj, seg28_pkl=seg28_pkl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_region_info_pkl(path: Union[str, Path, None] = None) -> dict:
    p = Path(REGION_INFO_PKL if path is None else path)
    if not p.is_file():
        raise FileNotFoundError(f'region_info pkl not found: {p}')
    with open(p, 'rb') as f:
        return pickle.load(f)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build / visualize HACK LAMM region info')
    parser.add_argument('--viz', action='store_true', help='Open Polyscope viewer after saving')
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE_OBJ)
    parser.add_argument('--seg28', type=Path, default=DEFAULT_SEG28_PKL)
    parser.add_argument('--out', type=Path, default=REGION_INFO_PKL)
    args = parser.parse_args()

    saved = save_region_info_pkl(
        reference_obj=args.template,
        out_path=args.out,
        seg28_pkl=args.seg28,
    )
    print(f'saved {saved}')

    if args.viz:
        region_info = load_region_info_pkl(saved)
        show_hack_regions_polyscope(
            reference_obj=args.template,
            region_info=region_info,
            boundary_vertices_gui=True,
        )
