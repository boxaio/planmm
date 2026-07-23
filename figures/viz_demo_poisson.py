"""
Local patch around a single seed vertex on the **full** template mesh.

``SEED_VERTEX`` is a **global** vertex index (same as the loaded OBJ ``0 … V−1``).
We build the mesh graph from **all** triangles, take every vertex within ``MAX_HOP`` graph
hops of the seed, and keep triangles whose three corners all lie in that set (induced
submesh on the k-ring). ``MAX_HOP = 1`` is typically “one ring” around the seed.

Renders gray faces, black edges, black vertex points; writes ``figures/viz_demo_poisson.png``.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import polyscope as ps
import trimesh

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FIGURES_DIR = Path(__file__).resolve().parent
TEMPLATE_MESH = _REPO_ROOT / 'dataset' / 'hack_template.obj'
OUT_PNG = FIGURES_DIR / 'viz_demo_poisson.png'

# Global vertex index into TEMPLATE_MESH: 0 <= SEED_VERTEX < V
SEED_VERTEX = 7285
# Graph hops along triangle edges; 1 = seed plus its mesh neighbors (“one ring” support).
MAX_HOP = 1


def _build_vertex_adjacency(n_verts: int, tris: np.ndarray) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n_verts)]
    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))
    return adj


def vertices_within_hops(
    adj: list[set[int]],
    start: int,
    max_hop: int,
) -> np.ndarray:
    """BFS from global ``start``; return global vertex indices with hop distance <= ``max_hop``."""
    n = len(adj)
    dist = np.full(n, -1, dtype=np.int32)
    if start < 0 or start >= n:
        return np.array([], dtype=np.int64)
    dist[start] = 0
    q: deque[int] = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return np.flatnonzero((dist >= 0) & (dist <= max_hop)).astype(np.int64)


def _compact_submesh(
    verts: np.ndarray,
    faces: np.ndarray,
    face_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    f_sub = faces[face_mask]
    if f_sub.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    used = np.unique(f_sub.reshape(-1))
    old_to_new = np.full(verts.shape[0], -1, dtype=np.int64)
    old_to_new[used] = np.arange(used.shape[0], dtype=np.int64)
    v_sub = np.ascontiguousarray(verts[used], dtype=np.float64)
    f_idx = np.ascontiguousarray(old_to_new[f_sub], dtype=np.int64)
    return v_sub, f_idx


def main() -> None:
    if not TEMPLATE_MESH.is_file():
        raise FileNotFoundError(f'Missing template mesh: {TEMPLATE_MESH}')

    mesh = trimesh.load(str(TEMPLATE_MESH), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    n_v = int(verts.shape[0])
    if not (0 <= SEED_VERTEX < n_v):
        raise ValueError(f'SEED_VERTEX {SEED_VERTEX} out of range [0, {n_v})')

    adj = _build_vertex_adjacency(n_v, faces)
    ball = vertices_within_hops(adj, SEED_VERTEX, MAX_HOP)
    print(
        f'[viz_demo_poisson] full mesh seed={SEED_VERTEX} hops<={MAX_HOP}: '
        f'{ball.size} vertices'
    )

    inside = np.zeros(n_v, dtype=bool)
    inside[ball] = True
    face_keep = inside[faces].all(axis=1)
    if not np.any(face_keep):
        raise RuntimeError(
            'No triangle has all three vertices in the hop ball; increase MAX_HOP or check SEED_VERTEX.'
        )

    v_sub, f_sub = _compact_submesh(verts, faces, face_keep)
    pts_black = np.ascontiguousarray(verts[ball], dtype=np.float64)

    diag = float(np.linalg.norm(v_sub.max(axis=0) - v_sub.min(axis=0)))
    prad = max(diag * 0.02, 1e-6)
    center = v_sub.mean(axis=0)

    ps.init()
    ps.set_ground_plane_mode('none')
    ps.set_program_name('viz_demo_poisson')

    eye = center + np.array([0.0, 0.0, max(diag * 2.8, 1e-4)], dtype=np.float64)
    ps.look_at(eye, center)

    ps.register_surface_mesh(
        'seed_ring_mesh',
        v_sub,
        f_sub,
        enabled=True,
        color=(0.85, 0.85, 0.85),
        edge_color=(0.0, 0.0, 0.0),
        edge_width=3.5,
        material='clay',
        smooth_shade=False,
    )

    ps.register_point_cloud(
        'seed_ring_verts',
        pts_black,
        radius=prad,
        color=(0.0, 0.0, 0.0),
        enabled=True,
    )
    ps.show()
    ps.screenshot(str(OUT_PNG), transparent_bg=True)
    print(f'[viz_demo_poisson] wrote {OUT_PNG}')


if __name__ == '__main__':
    main()
