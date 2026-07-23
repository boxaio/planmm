import polyscope as ps
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from dataset.bfm_dataset import BFMDataset, VAL_BFM_NOSE_PT


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = verts.astype(np.float64)
    f = faces.astype(np.int64)
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    fn = np.cross(e1, e2)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn = np.zeros_like(v)
    for j in range(3):
        np.add.at(vn, f[:, j], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    return vn.astype(np.float32)


def _tangent_frames(
    normals: np.ndarray, ref_direction: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """与顶点法向正交的切向 t 与副切向 b，满足 b = n × t（右手系）。"""
    if ref_direction is None:
        ref_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    n = normals.astype(np.float64)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    ref = np.broadcast_to(ref_direction, n.shape)
    t = ref - np.sum(ref * n, axis=1, keepdims=True) * n
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    bad = (tn < 1e-8).flatten()
    if np.any(bad):
        alt = np.array([1.0, 0.0, 0.0])
        ref2 = np.broadcast_to(alt, n.shape)
        t_alt = ref2 - np.sum(ref2 * n, axis=1, keepdims=True) * n
        t[bad] = t_alt[bad]
        tn = np.linalg.norm(t, axis=1, keepdims=True)
    t /= np.maximum(tn, 1e-12)
    b = np.cross(n, t)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return t.astype(np.float32), b.astype(np.float32)


def _axis_arrows(
    origins: np.ndarray, dirs: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    origins = np.asarray(origins, dtype=np.float64)
    dirs = np.asarray(dirs, dtype=np.float64)
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    tips = origins + scale * dirs
    k = origins.shape[0]
    nodes = np.empty((2 * k, 3), dtype=np.float32)
    edges = np.empty((k, 2), dtype=np.int32)
    for i in range(k):
        nodes[2 * i] = origins[i]
        nodes[2 * i + 1] = tips[i]
        edges[i] = [2 * i, 2 * i + 1]
    return nodes, edges


dataset = BFMDataset(packed_pt=VAL_BFM_NOSE_PT)
s0 = dataset[0]

# 鼻尖等关键点在子网格上的顶点下标（可改为 list）
kpts_vid = [1003]

verts = s0["source_verts"].cpu().numpy()
faces = s0["faces"].cpu().numpy()
kpts_vid = np.atleast_1d(np.asarray(kpts_vid, dtype=np.int64))
if np.any((kpts_vid < 0) | (kpts_vid >= verts.shape[0])):
    raise IndexError(
        f"kpts_vid 需在 [0, {verts.shape[0]}), 当前: {kpts_vid}"
    )

vn = _vertex_normals(verts, faces)
Ts, Bs = _tangent_frames(vn[kpts_vid])
Ns = vn[kpts_vid]
P = verts[kpts_vid]

diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
axis_scale = 0.06 * diag

ps.init()

ps.register_surface_mesh(
    "nose_source",
    vertices=verts,
    faces=faces,
    enabled=True,
    color=(0.75, 0.75, 0.75),
    edge_width=0.95,
    material="clay",
    smooth_shade=False,
    back_face_policy="custom",
)

nt, et = _axis_arrows(P, Ts, axis_scale)
nb, eb = _axis_arrows(P, Bs, axis_scale)
nn, en = _axis_arrows(P, Ns, axis_scale)

cn_t = ps.register_curve_network("kpt_frame_t", nt, et)
cn_t.set_color((1.0, 0.2, 0.2))
cn_b = ps.register_curve_network("kpt_frame_b", nb, eb)
cn_b.set_color((0.2, 1.0, 0.2))
cn_n = ps.register_curve_network("kpt_frame_n", nn, en)
cn_n.set_color((0.25, 0.45, 1.0))

pts = ps.register_point_cloud("kpt_centers", P)
pts.set_radius(0.008 * diag, relative=False)
pts.set_color((1.0, 0.85, 0.2))

ps.show()
