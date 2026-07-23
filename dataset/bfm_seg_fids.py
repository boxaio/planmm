import ast
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

from utils.mesh import read_obj

_DATASET_DIR = Path(__file__).resolve().parent

REGION_INFO_PKL = _DATASET_DIR / 'bfm_region_info.pkl'

region_names: List[str] = [
    'face_0',
    'face_1',
    'eye_0',
    'eye_1',
    'nose',
    'mouth',
    'forehead',
    'frown',
    'eye_0_inner',
    'eye_1_inner',
    'nose_inner',
]

# region_name -> single fids txt path, or tuple of txts for merged regions (e.g. nose).
_REGION_FID_TXT: Dict[str, Union[Path, Tuple[Path, ...]]] = {
    'face_0': _DATASET_DIR / 'bfm_face_0_fids.txt',
    'face_1': _DATASET_DIR / 'bfm_face_1_fids.txt',
    'eye_0': _DATASET_DIR / 'bfm_eye_0_fids.txt',
    'eye_1': _DATASET_DIR / 'bfm_eye_1_fids.txt',
    'nose': (
        _DATASET_DIR / 'bfm_nose_0_fids.txt',
        _DATASET_DIR / 'bfm_nose_1_fids.txt',
    ),
    'mouth': _DATASET_DIR / 'bfm_mouth_fids.txt',
    'forehead': _DATASET_DIR / 'bfm_forehead_fids.txt',
    'frown': _DATASET_DIR / 'bfm_frown_fids.txt',
    'eye_0_inner': _DATASET_DIR / 'bfm_eye_0_inner_fids.txt',
    'eye_1_inner': _DATASET_DIR / 'bfm_eye_1_inner_fids.txt',
    'nose_inner': _DATASET_DIR / 'bfm_nose_inner_fids.txt',
}


def load_bfm_seg_fids_from_txt(txt_path: Union[str, Path]) -> List[int]:
    p = Path(txt_path)
    raw = p.read_text(encoding='utf-8').strip()
    if raw.startswith('['):
        seg_fids = ast.literal_eval(raw)
        if not isinstance(seg_fids, list):
            raise ValueError(
                f'{p}: expected a list literal, got {type(seg_fids).__name__}'
            )
        return [int(x) for x in seg_fids]
    seg_fids: List[int] = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            seg_fids.append(int(line))
    return seg_fids


def _collect_fids_for_region(name: str) -> np.ndarray:
    if name not in _REGION_FID_TXT:
        raise KeyError(f'unknown region={name!r}, configured: {sorted(_REGION_FID_TXT)}')
    src = _REGION_FID_TXT[name]
    if isinstance(src, tuple):
        merged: List[int] = []
        for txt in src:
            merged.extend(load_bfm_seg_fids_from_txt(txt))
        fids = np.asarray(sorted(set(merged)), dtype=np.int64)
    else:
        fids = np.asarray(load_bfm_seg_fids_from_txt(src), dtype=np.int64)
    return fids


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


def build_region_info_dict(reference_obj: Union[str, Path]) -> dict:
    """
    Build per-region dicts with ``fids`` (face ids), ``vids`` (all vertices in the region),
    ``boundary_vids`` (global ids of patch-boundary vertices), and ``patch_faces`` (``(F, 3)``
    int64): each row is one triangle's three vertex indices **local** to that region's ``vids``
    (same convention as slicing ``V[:, vids, :]`` for Poisson nets).

    Region ``nose`` merges face lists from ``bfm_nose_0_fids.txt`` and ``bfm_nose_1_fids.txt``.
    """
    ref = Path(reference_obj)
    if not ref.is_file():
        raise FileNotFoundError(f'reference_obj not found: {ref}')

    missing = [n for n in region_names if n not in _REGION_FID_TXT]
    if missing:
        raise ValueError(f'region_names has unconfigured regions: {missing}')

    extra = set(_REGION_FID_TXT) - set(region_names)
    if extra:
        raise ValueError(f'_REGION_FID_TXT keys do not match region_names, extra: {sorted(extra)}')

    obj = read_obj(str(ref), tri=True)
    faces = np.asarray(obj.fvs, dtype=np.int64)
    num_vertices = int(np.asarray(obj.vs).shape[0])

    regions_out: Dict[str, Dict[str, np.ndarray]] = {}
    for name in region_names:
        fids = _collect_fids_for_region(name)
        vids = face_indices_to_vertex_indices(faces, fids)
        boundary_vids = boundary_vertex_indices(faces, fids)
        patch_faces = patch_faces_local(faces, fids, vids, num_vertices)
        regions_out[name] = {
            'fids': fids,
            'vids': vids,
            'boundary_vids': boundary_vids,
            'patch_faces': patch_faces,
        }

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


def show_bfm_regions_polyscope(
    reference_obj: Union[str, Path]=_DATASET_DIR / 'bfm_template.obj',
    *,
    region_info: Union[None, dict, str, Path] = None,
    names: Optional[Sequence[str]] = None,
    mesh_name: str = 'bfm_regions',
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
        print(f'[show_bfm_regions_polyscope] faces not assigned to any region: {n_unassigned}')

    ps.init()
    ps_mesh = ps.register_surface_mesh(
        mesh_name,
        vertices=verts,
        faces=faces,
        enabled=True,
        color=(0.75, 0.75, 0.75),
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
    reference_obj: Union[str, Path]=_DATASET_DIR / 'bfm_template.obj',
    out_path: Union[str, Path, None] = None,
) -> Path:
    """Write ``bfm_region_info.pkl`` with per-region ``fids``, ``vids``, ``boundary_vids``, and ``patch_faces``."""
    out = Path(REGION_INFO_PKL if out_path is None else out_path)
    data = build_region_info_dict(reference_obj)
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


    # saved = save_region_info_pkl()
    # print(f'saved {saved}')

    region_info = load_region_info_pkl()
    # boundary_vertices_gui=True: use Polyscope ImGui to toggle; boundary_vertex_patches=('nose',) sets initial on
    show_bfm_regions_polyscope(region_info=region_info, boundary_vertices_gui=True)
