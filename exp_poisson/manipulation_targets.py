"""Build manipulation targets with per-handle rigid alignment (LAMM inference-style).

Raw ``vt - vs`` mixes global head pose (rigid rotation) with local expression change.
For each HACK_seg_28 handle region we:
  1. Estimate a rigid transform from target boundary vertices to source boundary.
  2. Apply it to the full target handle patch (in source frame).
  3. Alpha-blend aligned patch into ``vs`` to form ``target_manip``.
  4. Encode handle deltas as region-mean displacement for Poisson conditioning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from meshes.hack_utils import load_hack_seg_28_patches

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEG28_PKL = _REPO_ROOT / 'data' / 'hack_data' / 'HACK_seg_28.pkl'


def load_handle_region_boundary_vids(
    handle_regions: list[str],
    template_faces: np.ndarray,
    seg28_pkl: str | Path | None = None,
) -> dict[str, list[int]]:
    """Patch boundary vertex indices on the template mesh for each handle region."""
    from dataset.hack_seg_fids import boundary_vertex_indices

    patches = load_hack_seg_28_patches(seg28_pkl or DEFAULT_SEG28_PKL)
    faces = np.asarray(template_faces, dtype=np.int64)
    out: dict[str, list[int]] = {}

    for name in handle_regions:
        if name not in patches:
            raise KeyError(f"handle region {name!r} not in HACK_seg_28")
        fids = np.asarray(patches[name]['face_ids'], dtype=np.int64).reshape(-1)
        fids = fids[(fids >= 0) & (fids < faces.shape[0])]
        bd = boundary_vertex_indices(faces, fids)
        if bd.size < 3:
            vids = np.asarray(
                patches[name].get('vert_ids', patches[name].get('vids')),
                dtype=np.int64,
            ).reshape(-1)
            bd = vids[: min(3, vids.size)]
        out[name] = [int(v) for v in bd.reshape(-1)]
    return out


def batch_rigid_align(
    mobile: torch.Tensor,
    fixed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rigid transform mapping ``mobile`` correspondence to ``fixed``.

    Both inputs are [B, N, 3] corresponding points (e.g. boundary vertices).
    Returns R [B, 3, 3], t [B, 3] with ``mobile @ R^T + t ≈ fixed`` (row-vector convention).
    """
    if mobile.shape != fixed.shape or mobile.ndim != 3 or mobile.shape[-1] != 3:
        raise ValueError(f'expected matching [B, N, 3] tensors, got {mobile.shape} vs {fixed.shape}')
    if mobile.shape[1] < 3:
        raise ValueError(f'need at least 3 correspondence points, got N={mobile.shape[1]}')

    mu_m = mobile.mean(dim=1, keepdim=True)
    mu_f = fixed.mean(dim=1, keepdim=True)
    m = mobile - mu_m
    f = fixed - mu_f
    cov = torch.bmm(m.transpose(1, 2), f)
    u, _, vh = torch.linalg.svd(cov)

    r = torch.bmm(vh.transpose(-2, -1), u.transpose(-2, -1))
    det = torch.linalg.det(r)
    reflect = (det < 0).unsqueeze(-1).unsqueeze(-1)
    vh_adj = vh.clone()
    vh_adj[:, -1, :] = torch.where(reflect.squeeze(-1), -vh_adj[:, -1, :], vh_adj[:, -1, :])
    r = torch.bmm(vh_adj.transpose(-2, -1), u.transpose(-2, -1))

    t = (mu_f - torch.bmm(mu_m, r.transpose(1, 2))).squeeze(1)
    return r, t


def apply_rigid(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Apply batched rigid transform to [B, V, 3] points."""
    return torch.bmm(points, rotation.transpose(1, 2)) + translation.unsqueeze(1)


def build_local_manipulation_targets(
    source: torch.Tensor,
    target: torch.Tensor,
    handle_vid_tensors: list[torch.Tensor],
    boundary_vid_tensors: list[torch.Tensor],
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Build locally aligned manipulation targets and handle delta signals.

    Args:
        source: [B, V, 3] input mesh ``vs``.
        target: [B, V, 3] paired mesh ``vt`` (e.g. ``roll(vs, 1)`` in batch).
        handle_vid_tensors: per-handle region vertex indices.
        boundary_vid_tensors: per-handle patch boundary indices on template.
        alpha: [B] blend factor in [alpha_min, alpha_max].

    Returns:
        target_manip: [B, V, 3] — ``vs`` with each handle patch blended toward
            rigidly-aligned target geometry; vertices outside handles stay ``vs``.
        delta_handles: list of [B, 3] region-mean displacements (x alpha) for Poisson cond.
    """
    if len(handle_vid_tensors) != len(boundary_vid_tensors):
        raise ValueError('handle and boundary tensor lists must have the same length')

    batch_size = source.shape[0]
    alpha_w = alpha.view(batch_size, 1, 1)
    target_manip = source.clone()
    delta_handles: list[torch.Tensor] = []

    for vids, bvids in zip(handle_vid_tensors, boundary_vid_tensors):
        src_boundary = source[:, bvids]
        tgt_boundary = target[:, bvids]
        rotation, translation = batch_rigid_align(tgt_boundary, src_boundary)

        region_src = source[:, vids]
        region_tgt = target[:, vids]
        region_aligned = apply_rigid(region_tgt, rotation, translation)
        region_blended = (1 - alpha_w) * region_src + alpha_w * region_aligned
        target_manip[:, vids] = region_blended

        delta_handles.append((alpha_w * (region_aligned - region_src)).mean(dim=1))

    return target_manip, delta_handles
