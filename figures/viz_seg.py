from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polyscope as ps

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.bfm_seg_fids import (  # noqa: E402
    build_region_info_dict,
    load_region_info_pkl,
    region_names,
    _unwrap_regions_dict,
)
from utils.mesh import read_obj  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent
REGION_INFO_PKL = _REPO_ROOT / 'dataset' / 'bfm_region_info.pkl'
BFM_TEMPLATE = _REPO_ROOT / 'dataset' / 'bfm_template.obj'
PS_MESH_NAME = 'bfm_patch_view'


def submesh_compact(
    verts: np.ndarray,
    faces: np.ndarray,
    fids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract faces ``fids`` and reindex vertices to a compact (V',3) / (F',3) mesh."""
    tris = faces[np.asarray(fids, dtype=np.int64)]
    vuniq = np.unique(tris.reshape(-1))
    old_to_new = np.full(int(verts.shape[0]), -1, dtype=np.int64)
    old_to_new[vuniq] = np.arange(vuniq.shape[0], dtype=np.int64)
    v_sub = np.ascontiguousarray(verts[vuniq], dtype=np.float64)
    f_sub = np.ascontiguousarray(old_to_new[tris], dtype=np.int64)
    return v_sub, f_sub


def _camera_look_at_bounds(verts: np.ndarray) -> None:
    center = verts.mean(axis=0)
    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    dist = max(diag * 2.5, 1e-4)
    eye = center + np.array([0.0, 0.0, dist], dtype=np.float64)
    ps.look_at(eye, center)


def export_patch_screenshots(
    *,
    out_dir: Path | None = None,
    template_obj: Path | None = None,
    region_pkl: Path | None = None,
) -> None:
    out_dir = FIGURES_DIR if out_dir is None else Path(out_dir)
    template_obj = BFM_TEMPLATE if template_obj is None else Path(template_obj)
    region_pkl = REGION_INFO_PKL if region_pkl is None else Path(region_pkl)

    out_dir.mkdir(parents=True, exist_ok=True)
    if not template_obj.is_file():
        raise FileNotFoundError(f'BFM template not found: {template_obj}')

    obj = read_obj(str(template_obj), tri=True)
    verts = np.asarray(obj.vs, dtype=np.float64)
    faces = np.asarray(obj.fvs, dtype=np.int64)

    if region_pkl.is_file():
        regions = _unwrap_regions_dict(load_region_info_pkl(region_pkl))
    else:
        regions = build_region_info_dict(template_obj)

    ps.init()
    ps.set_ground_plane_mode('none')
    ps.set_program_name('bfm_patch_viz')

    for name in region_names:
        if name not in regions:
            raise KeyError(f'missing region {name!r} in region info, have {sorted(regions)}')
        fids = np.asarray(regions[name]['fids'], dtype=np.int64)
        v_sub, f_sub = submesh_compact(verts, faces, fids)
        ps.register_surface_mesh(
            PS_MESH_NAME,
            v_sub,
            f_sub,
            enabled=True,
            color=(0.88, 0.88, 0.88),
            edge_width=0.9,
            material='clay',
            smooth_shade=False,
            back_face_policy='custom',
        )
        _camera_look_at_bounds(v_sub)
        out_png = out_dir / f'bfm_{name}.png'
        ps.screenshot(str(out_png), transparent_bg=True)
        ps.remove_surface_mesh(PS_MESH_NAME)
        print(f'wrote {out_png}')


def main() -> None:
    export_patch_screenshots()


if __name__ == '__main__':
    main()
