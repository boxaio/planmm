"""Comprehensive mesh quality metrics for PLANMM (no region masks).

Used by overfit smoke and offline diagnosis. Metrics cover geometry fidelity,
expression capture, diversity, stretch/fold proxies, and path consistency.
"""

from __future__ import annotations

from typing import Any

import torch


def _face_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """verts [B,V,3], faces [F,3] or [B,F,3] → normals [B,F,3] (unnormalized)."""
    if faces.ndim == 2:
        f = faces
        a, b, c = verts[:, f[:, 0]], verts[:, f[:, 1]], verts[:, f[:, 2]]
    else:
        a = torch.gather(verts, 1, faces[..., 0:1].expand(-1, -1, 3))
        b = torch.gather(verts, 1, faces[..., 1:2].expand(-1, -1, 3))
        c = torch.gather(verts, 1, faces[..., 2:3].expand(-1, -1, 3))
    return torch.cross(b - a, c - a, dim=-1)


def mesh_quality_report(
    recon: torch.Tensor,
    gt: torch.Tensor,
    scaffold: torch.Tensor,
    faces: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    *,
    gen: torch.Tensor | None = None,
    njf_or_base: torch.Tensor | None = None,
    local_disp: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Return a flat dict of scalar metrics (floats) for logging / gates."""
    device = recon.device
    src = edge_src.to(device)
    dst = edge_dst.to(device)
    f = faces[0] if faces.ndim == 3 else faces
    f = f.to(device)

    disp_r = recon - scaffold
    disp_g = gt - scaffold

    # Mass-free mean center (uniform) — translation-invariant expression metric.
    disp_r_c = disp_r - disp_r.mean(dim=1, keepdim=True)
    disp_g_c = disp_g - disp_g.mean(dim=1, keepdim=True)
    deform_capture = (
        1.0
        - (disp_r_c - disp_g_c).abs().mean()
        / disp_g_c.abs().mean().clamp_min(1e-8)
    ).item()
    b = recon.shape[0]
    if b >= 2:
        def_pw_r = (disp_r_c[0] - disp_r_c[1]).abs().mean().item()
        def_pw_g = (disp_g_c[0] - disp_g_c[1]).abs().mean().item()
        deform_div = def_pw_r / max(def_pw_g, 1e-8)
    else:
        deform_div = 0.0

    l1 = (recon - gt).abs().mean().item()
    l1_max = (recon - gt).abs().mean(dim=-1).amax().item()
    capture = (
        1.0
        - (disp_r - disp_g).abs().mean()
        / disp_g.abs().mean().clamp_min(1e-8)
    ).item()
    rn = (recon - scaffold).abs().mean().item()
    gn = (gt - scaffold).abs().mean().item()

    r_ext = (recon.amax(1) - recon.amin(1)).mean(0)
    g_ext = (gt.amax(1) - gt.amin(1)).mean(0).clamp_min(1e-8)
    s_ext = (scaffold.amax(1) - scaffold.amin(1)).mean(0).clamp_min(1e-8)
    ext_rg = (r_ext / g_ext).detach().cpu().tolist()
    ext_rs = (r_ext / s_ext).detach().cpu().tolist()

    axis_r = disp_r.abs().mean(dim=(0, 1))
    axis_g = disp_g.abs().mean(dim=(0, 1)).clamp_min(1e-8)
    axis_ratio = (axis_r / axis_g).detach().cpu().tolist()

    len_r = torch.norm(recon[:, src] - recon[:, dst], dim=-1)
    len_g = torch.norm(gt[:, src] - gt[:, dst], dim=-1).clamp_min(1e-8)
    len_s = torch.norm(scaffold[:, src] - scaffold[:, dst], dim=-1).clamp_min(1e-8)
    ratio_g = len_r / len_g
    ratio_s = len_r / len_s

    # Face orientation / area (fold & collapse proxies).
    n_r = _face_normals(recon, f)
    n_g = _face_normals(gt, f)
    flip_frac = ((n_r * n_g).sum(-1) < 0).float().mean().item()
    area_r = n_r.norm(dim=-1).mean().item()
    area_g = n_g.norm(dim=-1).mean().clamp_min(1e-8).item()
    area_ratio = area_r / area_g

    # Jump metrics use centered disp so translation doesn't inflate smoothness.
    jump_r = (disp_r_c[:, src] - disp_r_c[:, dst]).norm(dim=-1).mean().item()
    jump_g = (disp_g_c[:, src] - disp_g_c[:, dst]).norm(dim=-1).mean().item()

    if b >= 2:
        pairwise = (recon[0] - recon[1]).abs().mean().item()
        disp_pw = (disp_r[0] - disp_r[1]).abs().mean().item()
        gt_pw = (disp_g[0] - disp_g[1]).abs().mean().item()
        disp_div = disp_pw / max(gt_pw, 1e-8)
    else:
        pairwise = 0.0
        disp_div = 0.0

    out: dict[str, Any] = {
        'l1': l1,
        'l1_max_vert': l1_max,
        'capture': capture,
        'deform_capture': deform_capture,
        'deform_div': deform_div,
        'recon_neutral': rn,
        'gt_neutral': gn,
        'extent_rg_x': ext_rg[0],
        'extent_rg_y': ext_rg[1],
        'extent_rg_z': ext_rg[2],
        'extent_rs_x': ext_rs[0],
        'extent_rs_y': ext_rs[1],
        'extent_rs_z': ext_rs[2],
        'axis_ratio_x': axis_ratio[0],
        'axis_ratio_y': axis_ratio[1],
        'axis_ratio_z': axis_ratio[2],
        'edge_gt_p50': float(ratio_g.quantile(0.50)),
        'edge_gt_p95': float(ratio_g.quantile(0.95)),
        'edge_gt_max': float(ratio_g.max()),
        'edge_sc_p95': float(ratio_s.quantile(0.95)),
        'edge_sc_max': float(ratio_s.max()),
        'flip_frac': flip_frac,
        'area_ratio': area_ratio,
        'disp_jump_pred': jump_r,
        'disp_jump_gt': jump_g,
        'disp_jump_ratio': jump_r / max(jump_g, 1e-8),
        'pairwise': pairwise,
        'disp_div': disp_div,
        'disp_pred_max': float(disp_r.norm(dim=-1).max()),
        'disp_gt_max': float(disp_g.norm(dim=-1).max()),
        'disp_excess_max': float(torch.relu(disp_r.norm(dim=-1) - disp_g.norm(dim=-1)).max()),
    }

    if gen is not None:
        out['gen_l1'] = (gen - gt).abs().mean().item()
        if b >= 2:
            out['gen_pairwise'] = (gen[0] - gen[1]).abs().mean().item()
            c = gen - gen.mean(dim=1, keepdim=True)
            s = (c.amax(1) - c.amin(1)).norm(dim=-1).clamp_min(1e-8).view(-1, 1, 1)
            sn = c / s
            out['gen_shape_pw'] = (sn[0] - sn[1]).abs().mean().item()
        else:
            out['gen_pairwise'] = 0.0
            out['gen_shape_pw'] = 0.0

    if njf_or_base is not None:
        out['base_neutral'] = (njf_or_base - scaffold).abs().mean().item()
        out['local_contrib'] = (recon - njf_or_base).abs().mean().item()
    if local_disp is not None:
        out['local_rms'] = float(local_disp.pow(2).mean().sqrt())
        out['local_max'] = float(local_disp.abs().max())

    return out


def gate_mesh_quality(m: dict[str, Any], *, stage: str = 'overfit') -> tuple[bool, list[str]]:
    """Hard gates for overfit / short-train. Returns (ok, failure_reasons)."""
    fails: list[str] = []

    def need(cond: bool, msg: str):
        if not cond:
            fails.append(msg)

    # Fidelity
    need(m['l1'] < 0.06, f"l1={m['l1']:.4f} (want <0.06)")
    need(m['capture'] > 0.55, f"capture={m['capture']:.3f} (want >0.55)")
    # Expression must not hide behind translation-dominated capture.
    need(
        m.get('deform_capture', 0) > 0.45,
        f"deform_capture={m.get('deform_capture', 0):.3f} (want >0.45)",
    )
    need(
        m.get('deform_div', 0) > 0.35,
        f"deform_div={m.get('deform_div', 0):.3f} (want >0.35)",
    )
    if 'mouth_open_ratio' in m:
        need(
            m['mouth_open_ratio'] > 0.40,
            f"mouth_open_ratio={m['mouth_open_ratio']:.3f} (want >0.40)",
        )
    need(m['recon_neutral'] > 0.5 * m['gt_neutral'], f"recon≈neutral (rn={m['recon_neutral']:.4f})")

    # No flatten / squash (all axes)
    for ax, key in zip('xyz', ('extent_rg_x', 'extent_rg_y', 'extent_rg_z')):
        need(0.88 <= m[key] <= 1.15, f"extent_{ax}={m[key]:.3f} (want 0.88-1.15)")
    for ax, key in zip('xyz', ('axis_ratio_x', 'axis_ratio_y', 'axis_ratio_z')):
        # LAMM-only L1 has no axis/edge regularizers; keep soft floor.
        need(m[key] > 0.40, f"axis_{ax}={m[key]:.3f} (want >0.40)")

    # Stretch / fold: p95 is the visual signal; max is outlier-dominated on HACK
    # (hard-mouth overfit can spike a few edges without looking broken).
    need(m['edge_gt_p95'] < 2.5, f"edge_gt_p95={m['edge_gt_p95']:.3f} (want <2.5)")
    need(m['edge_gt_max'] < 120.0, f"edge_gt_max={m['edge_gt_max']:.3f} (want <120)")
    need(m['edge_sc_max'] < 200.0, f"edge_sc_max={m['edge_sc_max']:.3f} (want <200)")
    need(m['flip_frac'] < 0.05, f"flip_frac={m['flip_frac']:.4f} (want <0.05)")
    need(0.5 <= m['area_ratio'] <= 1.8, f"area_ratio={m['area_ratio']:.3f} (want 0.5-1.8)")
    need(m['disp_excess_max'] < 1.0, f"disp_excess={m['disp_excess_max']:.3f} (want <1.0)")

    # Diversity (not mean face)
    need(m['disp_div'] > 0.35, f"disp_div={m['disp_div']:.3f} (want >0.35)")
    need(m['pairwise'] > 0.01, f"pairwise={m['pairwise']:.4f} (want >0.01)")

    # Local structure: edge-jumps should track GT (too high = rough, too low = over-smooth)
    need(
        m['disp_jump_ratio'] > 0.70,
        f"disp_jump_ratio={m['disp_jump_ratio']:.3f} (want >0.70; too smooth)",
    )
    need(
        m['disp_jump_ratio'] < 1.30,
        f"disp_jump_ratio={m['disp_jump_ratio']:.3f} (want <1.30; too rough)",
    )
    if 'gen_l1' in m:
        need(m['gen_l1'] < m['l1'] * 1.8 + 0.02, f"gen_l1={m['gen_l1']:.4f} vs recon")
        need(m.get('gen_shape_pw', 0) > 0.003, f"gen_shape_pw={m.get('gen_shape_pw', 0):.4f}")

    del stage
    return len(fails) == 0, fails


def format_mesh_qa(m: dict[str, Any], fails: list[str] | None = None) -> str:
    lines = [
        '--- Mesh Quality ---',
        f"l1={m['l1']:.4f} l1_max={m['l1_max_vert']:.4f} capture={m['capture']:.3f} "
        f"deform_cap={m.get('deform_capture', 0):.3f} deform_div={m.get('deform_div', 0):.3f} "
        f"||r-n||={m['recon_neutral']:.4f} ||g-n||={m['gt_neutral']:.4f}",
        f"extent r/g xyz={[round(m[k],3) for k in ('extent_rg_x','extent_rg_y','extent_rg_z')]}",
        f"axis energy r/g xyz={[round(m[k],3) for k in ('axis_ratio_x','axis_ratio_y','axis_ratio_z')]}",
        f"edge_gt p50/p95/max={m['edge_gt_p50']:.3f}/{m['edge_gt_p95']:.3f}/{m['edge_gt_max']:.3f} "
        f"edge_sc max={m['edge_sc_max']:.3f}",
        f"flip={m['flip_frac']:.4f} area_ratio={m['area_ratio']:.3f} "
        f"disp_jump r/g={m['disp_jump_pred']:.4f}/{m['disp_jump_gt']:.4f} "
        f"(ratio={m['disp_jump_ratio']:.3f})",
        f"pairwise={m['pairwise']:.4f} disp_div={m['disp_div']:.3f} "
        f"disp_max p/g={m['disp_pred_max']:.3f}/{m['disp_gt_max']:.3f} "
        f"excess={m['disp_excess_max']:.3f}",
    ]
    if 'gen_l1' in m:
        lines.append(
            f"gen_l1={m['gen_l1']:.4f} gen_pw={m.get('gen_pairwise',0):.4f} "
            f"gen_shape={m.get('gen_shape_pw',0):.4f}"
        )
    if 'local_contrib' in m:
        lines.append(
            f"base||·-n||={m.get('base_neutral',0):.4f} local_contrib={m['local_contrib']:.4f} "
            f"local_rms={m.get('local_rms',0):.4f}"
        )
    if fails:
        lines.append('FAILS: ' + '; '.join(fails))
    return '\n'.join(lines)
