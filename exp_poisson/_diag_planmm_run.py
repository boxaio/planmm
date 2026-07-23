"""One-shot diagnosis for a PLANMM AE run (capture + seams + global/regional/local)."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from configs.config_utils import read_yaml
from exp_poisson.planmm_mesh_qa import mesh_quality_report
from dataset.hack_dataset import HACKDataset, VAL_HACK_PT
from networks.planmm import mass_center as _regional_center_feat
from exp_poisson.train_planmm_ae import (
    _compute_x_neutral,
    _per_sample_disp_capture,
    _resolve_dataset_sources,
    build_hackhead_for_mean_shape,
    build_planmm_net,
)
from utils.torch_utils import load_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=str, required=True)
    p.add_argument('--ckpt', type=str, default='best.pth')
    p.add_argument('--n', type=int, default=64)
    p.add_argument('--device', type=str, default='cuda:1')
    p.add_argument('--batch', type=int, default=1)
    return p.parse_args()


@torch.no_grad()
def decode_parts(net, z, x_neutral):
    """Mirror final-stage decode; returns (local_proxy, full, d_g, d_r, d_l_or_feat, gate)."""
    z = net._bound_latent(z)
    patches = net.latent_to_patch_tokens(z)
    cond, gate = net._build_decoder_cond(z)
    B = z.shape[0]
    mass, solver, G, M, faces = net._get_batch_operators(B, z.device)

    g_feat = cond
    r_feat = cond
    for block in net.dec_global_blocks:
        g_feat = net._run_poisson_block(
            g_feat, block, M, G, solver, faces, mass, checkpoint=False,
        )
    for block in net.dec_reg_blocks:
        r_feat = net._run_regional_block(
            r_feat, block, M, G, solver, faces, mass, checkpoint=False,
        )

    n_stages = net.decoder_depth + 1
    local_as_feat = bool(getattr(net, 'local_as_feat', False))
    g_in = g_feat
    if local_as_feat:
        local_feat = net._build_local_feat(patches)
        g_in = g_feat + net.local_disp_scale * local_feat

    d_regional = net._regional_displacement(
        n_stages - 1, n_stages, r_feat, M, G, solver, faces, mass,
    )
    use_grad_hint = bool(getattr(net, 'global_grad_hint', False)) and net.global_njf_final
    if use_grad_hint:
        hint = net.dec_global_heads[-1](g_in) if net.global_njf_residual else None
        if hint is not None:
            hint = hint + net.regional_disp_scale * d_regional
        else:
            hint = net.regional_disp_scale * d_regional
        d_global = net._global_displacement(
            n_stages - 1, n_stages, g_in, M, G, solver, faces, mass, disp_hint=hint,
        )
        # Effective local = full - (projected global without local feat)
        g_base = net._global_displacement(
            n_stages - 1, n_stages, g_feat, M, G, solver, faces, mass,
            disp_hint=(
                net.dec_global_heads[-1](g_feat) + net.regional_disp_scale * d_regional
                if net.global_njf_residual else net.regional_disp_scale * d_regional
            ),
        )
        d_local = d_global - g_base
        local_only = x_neutral + gate * d_local
        full = x_neutral + gate * d_global
        return local_only, full, d_global, d_regional, d_local, gate

    d_global = net._global_displacement(
        n_stages - 1, n_stages, g_in, M, G, solver, faces, mass,
    )
    if local_as_feat:
        d_global_base = net._global_displacement(
            n_stages - 1, n_stages, g_feat, M, G, solver, faces, mass,
        )
        d_local = d_global - d_global_base
        local_only = x_neutral + gate * d_local
        full = x_neutral + gate * (d_global + net.regional_disp_scale * d_regional)
        return local_only, full, d_global, d_regional, d_local, gate

    d_local_raw = net._build_d_local(patches)
    if net.local_njf_smooth:
        d_local = net._smooth_d_local(d_local_raw, M, G, solver, faces, mass)
    else:
        d_local = d_local_raw
        if net.local_disp_smooth_blend > 0:
            d_local = net._edge_smooth_displacement(
                d_local, blend=net.local_disp_smooth_blend,
            )
    local_only = x_neutral + gate * net.local_disp_scale * d_local
    full = x_neutral + gate * (
        d_global + net.regional_disp_scale * d_regional + net.local_disp_scale * d_local
    )
    return local_only, full, d_global, d_regional, d_local, gate


def _infer_model_cfg_from_ckpt(model_cfg: dict, ckpt_path: Path) -> dict:
    """Match module layout to checkpoint (v3/v4/v5)."""
    saved = torch.load(str(ckpt_path), map_location='cpu')
    keys = set(saved.keys())
    cfg = dict(model_cfg)
    has_local_xyz = any(k.startswith('dec_local_heads.') for k in keys)
    has_local_feat_mlp3 = 'dec_local_token_mlp.2.weight' in keys
    cfg['local_as_feat'] = (not has_local_xyz) and has_local_feat_mlp3
    if 'dec_global_njf.njf.grad_mlp.layers.0.lin_real.weight' not in keys:
        cfg['global_njf_final'] = False
        cfg['global_njf_residual'] = False
        cfg.setdefault('local_final_only', False)
        cfg.setdefault('regional_final_only', False)
        cfg.setdefault('local_disp_smooth_blend', 0.0)
    else:
        # Residual if final-stage Linear global head exists alongside NJF.
        n_g = sum(1 for k in keys if k.startswith('dec_global_heads.') and k.endswith('.2.weight'))
        cfg['global_njf_residual'] = n_g > int(cfg.get('decoder_depth', 3))
        if not cfg.get('global_njf_residual') and n_g >= int(cfg.get('decoder_depth', 3)) + 1:
            cfg['global_njf_residual'] = True
        # v4 pure-NJF: heads only for intermediate stages
        if n_g == int(cfg.get('decoder_depth', 3)) and not has_local_feat_mlp3:
            cfg['global_njf_residual'] = False
    if 'dec_reg_njf.njf.grad_mlp.layers.0.lin_real.weight' not in keys:
        cfg['regional_njf_final'] = False
    if 'dec_local_njf.njf.grad_mlp.layers.0.lin_real.weight' not in keys:
        cfg['local_njf_smooth'] = False
    if has_local_xyz:
        cfg['local_as_feat'] = False
    return cfg


def region_capture(recon, gt, neu, vids):
    dp = recon[:, vids] - neu[:, vids]
    dg = gt[:, vids] - neu[:, vids]
    err = (dp - dg).abs().mean(dim=(1, 2))
    gn = dg.abs().mean(dim=(1, 2)).clamp_min(1e-8)
    return 1.0 - err / gn


def edge_disp_jump(disp, src, dst, mask=None):
    j = (disp[:, src] - disp[:, dst]).norm(dim=-1)
    if mask is not None:
        j = j[:, mask]
    return j.mean(dim=-1)


def mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    return a.mean(), a.std(), a.min(), a.max()


def main():
    args = parse_args()
    run = Path(args.run)
    device = torch.device(args.device)
    config = read_yaml(str(run / 'config_file.yaml'))
    ckpt_path = run / args.ckpt
    model_cfg = _infer_model_cfg_from_ckpt(config['MODEL'], ckpt_path)
    net = build_planmm_net(model_cfg, device)
    load_from_checkpoint(net, str(ckpt_path), partial_restore=False)
    net.eval()
    hackhead = build_hackhead_for_mean_shape(config['MODEL'], device=device)

    w = net._soft_patch_weight
    entropy = -(w * w.clamp_min(1e-12).log()).sum(-1)
    maxw = w.max(-1).values
    print('=== Soft scatter weights ===')
    print(f'  V={w.shape[0]} K={w.shape[1]}')
    print(
        f'  max_weight mean/p50/p10='
        f'{maxw.mean():.3f}/{maxw.median():.3f}/{maxw.quantile(0.1):.3f}'
    )
    print(
        f'  entropy mean/p50={entropy.mean():.3f}/{entropy.median():.3f} '
        f'(logK={np.log(w.shape[1]):.3f})'
    )
    print(
        f'  frac maxw<0.9={float((maxw < 0.9).float().mean()):.3f}  '
        f'maxw<0.7={float((maxw < 0.7).float().mean()):.3f}'
    )

    src, dst = net._mesh_edge_src, net._mesh_edge_dst
    seam_edge_mask = w.argmax(-1)[src] != w.argmax(-1)[dst]
    seam_verts = torch.unique(torch.cat([src[seam_edge_mask], dst[seam_edge_mask]]))
    print(f'  seam_edges={int(seam_edge_mask.sum())}/{len(src)}  seam_verts={len(seam_verts)}')

    names = net.region_names
    mouth_vids = net.region_vert_ids[names.index('Mouth')]
    eyes_vids = net.region_vert_ids[names.index('Eyes')]
    chin_vids = net.region_vert_ids[names.index('Chin')]

    sources = _resolve_dataset_sources(config['DATASETS'])
    ds = HACKDataset(packed_pt=VAL_HACK_PT, sources=sources)
    print(f'full eval pool size={len(ds)} (sources={sources})')

    rng = np.random.default_rng(42)
    n_probe = min(args.n, len(ds))
    probe_idx = rng.choice(len(ds), size=n_probe, replace=False)

    acc = defaultdict(list)
    sample_rows = []

    for start in range(0, n_probe, args.batch):
        idxs = probe_idx[start:start + args.batch]
        batch = [ds[int(i)] for i in idxs]
        vs = torch.stack([b['verts'] for b in batch]).to(device)
        neck = torch.stack([b['neck_pose'] for b in batch]).to(device)
        x_neu = _compute_x_neutral(hackhead, neck, net.template_verts)
        z, mu, logvar = net.encode(vs, x_neutral=x_neu, return_layer_outputs=False)
        local_only, full, d_global, d_regional, d_local, gate = decode_parts(net, mu, x_neu)
        recon = full
        if start == 0:
            recon_check = net.decode(mu, x_neutral=x_neu, return_layer_outputs=False)
            if not torch.allclose(recon_check, full, atol=1e-3):
                print('WARN decode mismatch', float((recon_check - full).abs().max()))
            del recon_check
        net._ops_cache.clear()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        cap = _per_sample_disp_capture(full, vs, x_neu)
        cap_local = _per_sample_disp_capture(local_only, vs, x_neu)
        cap_m = region_capture(full, vs, x_neu, mouth_vids)
        cap_e = region_capture(full, vs, x_neu, eyes_vids)
        cap_c = region_capture(full, vs, x_neu, chin_vids)
        cap_m_local = region_capture(local_only, vs, x_neu, mouth_vids)
        cap_e_local = region_capture(local_only, vs, x_neu, eyes_vids)

        dg_n = d_global.norm(dim=-1).mean(dim=1)
        dr_n = d_regional.norm(dim=-1).mean(dim=1)
        dl_n = d_local.norm(dim=-1).mean(dim=1)
        ratio_reg_global = dr_n / dg_n.clamp_min(1e-8)

        disp_full = full - x_neu
        disp_local = local_only - x_neu
        disp_gt = vs - x_neu
        jump_full = edge_disp_jump(disp_full, src, dst)
        jump_local = edge_disp_jump(disp_local, src, dst)
        jump_gt = edge_disp_jump(disp_gt, src, dst)
        jump_full_seam = edge_disp_jump(disp_full, src, dst, seam_edge_mask)
        jump_local_seam = edge_disp_jump(disp_local, src, dst, seam_edge_mask)
        jump_gt_seam = edge_disp_jump(disp_gt, src, dst, seam_edge_mask)

        seam_l1 = (full[:, seam_verts] - vs[:, seam_verts]).abs().mean(dim=(1, 2))
        all_l1 = (full - vs).abs().mean(dim=(1, 2))
        local_l1 = (local_only - vs).abs().mean(dim=(1, 2))

        if start == 0:
            z0 = torch.zeros_like(mu)
            neu0 = net.decode(z0, x_neutral=x_neu, return_layer_outputs=False)
            z0_err = (neu0 - x_neu).abs().mean(dim=(1, 2))
            net._ops_cache.clear()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        else:
            z0_err = torch.zeros(vs.shape[0], device=device)

        for i in range(vs.shape[0]):
            row = {
                'capture': float(cap[i]),
                'capture_local': float(cap_local[i]),
                'cap_mouth': float(cap_m[i]),
                'cap_eyes': float(cap_e[i]),
                'cap_chin': float(cap_c[i]),
                'cap_mouth_local': float(cap_m_local[i]),
                'cap_eyes_local': float(cap_e_local[i]),
                'd_global_mean': float(dg_n[i]),
                'd_regional_mean': float(dr_n[i]),
                'd_local_mean': float(dl_n[i]),
                'reg_global_ratio': float(ratio_reg_global[i]),
                'jump_ratio': float(jump_full[i] / max(float(jump_gt[i]), 1e-8)),
                'jump_seam_ratio': float(
                    jump_full_seam[i] / max(float(jump_gt_seam[i]), 1e-8)
                ),
                'jump_local_seam_ratio': float(
                    jump_local_seam[i] / max(float(jump_gt_seam[i]), 1e-8)
                ),
                'l1': float(all_l1[i]),
                'local_l1': float(local_l1[i]),
                'seam_l1': float(seam_l1[i]),
                'gate': float(gate[i].mean()),
                'z0_err': float(z0_err[i]),
                'gt_disp': float(disp_gt[i].abs().mean()),
                'recon_disp': float(disp_full[i].abs().mean()),
            }
            sample_rows.append(row)
            for k, v in row.items():
                acc[k].append(v)

    print(f'\n=== Aggregate over {len(sample_rows)} samples (mu decode) ===')
    keys_show = [
        'l1', 'local_l1', 'seam_l1', 'capture', 'capture_local',
        'cap_mouth', 'cap_eyes', 'cap_chin', 'cap_mouth_local', 'cap_eyes_local',
        'd_global_mean', 'd_regional_mean', 'd_local_mean', 'reg_global_ratio',
        'jump_ratio', 'jump_seam_ratio', 'jump_local_seam_ratio',
        'gate', 'z0_err', 'gt_disp', 'recon_disp',
    ]
    for k in keys_show:
        if k not in acc or len(acc[k]) == 0:
            continue
        m, s, mn, mx = mean_std(acc[k])
        print(f'  {k:28s}  mean={m:.4f}  std={s:.4f}  [{mn:.4f},{mx:.4f}]')

    print('\n=== Mesh QA (first 8) ===')
    idxs = probe_idx[:8]
    qa_full = defaultdict(list)
    qa_reg = defaultdict(list)
    for i in idxs:
        b = ds[int(i)]
        vs_i = b['verts'].unsqueeze(0).to(device)
        neck_i = b['neck_pose'].unsqueeze(0).to(device)
        x_neu_i = _compute_x_neutral(hackhead, neck_i, net.template_verts)
        _, mu_i, _ = net.encode(vs_i, x_neutral=x_neu_i, return_layer_outputs=False)
        local_only_i, full_i, d_global_i, d_regional_i, d_local_i, _ = decode_parts(
            net, mu_i, x_neu_i,
        )
        net._ops_cache.clear()
        qa_full_i = mesh_quality_report(
            full_i, vs_i, x_neu_i, net.template_faces, src, dst, local_disp=d_local_i,
        )
        qa_reg_i = mesh_quality_report(
            local_only_i, vs_i, x_neu_i, net.template_faces, src, dst, local_disp=d_local_i,
        )
        for k, v in qa_full_i.items():
            qa_full[k].append(v)
        for k, v in qa_reg_i.items():
            qa_reg[k].append(v)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    m = {k: float(np.mean(v)) for k, v in qa_full.items()}
    m_reg = {k: float(np.mean(v)) for k, v in qa_reg.items()}
    keys = [
        'l1', 'capture', 'disp_jump_pred', 'disp_jump_gt', 'disp_jump_ratio',
        'recon_neutral', 'gt_neutral', 'disp_pred_max', 'disp_gt_max',
    ]
    print('FULL:', {k: round(m[k], 4) for k in keys})
    print('LOCAL_ONLY:', {k: round(m_reg[k], 4) for k in keys if k in m_reg})

    print('\n=== Parameter norms ===')
    g_fro = sum(h[2].weight.norm().item() for h in net.dec_global_heads)
    r_fro = sum(h[2].weight.norm().item() for h in net.dec_reg_heads)
    print(f'  dec_global_readout fro_sum={g_fro:.4f}')
    print(f'  dec_reg_readout fro_sum={r_fro:.4f}')
    print(f'  latent_to_token fro={net.latent_to_token.weight.norm():.4f}')

    rows_sorted = sorted(sample_rows, key=lambda r: r['capture'])
    show_keys = [
        'capture', 'cap_mouth', 'cap_eyes', 'd_global_mean', 'd_regional_mean',
        'd_local_mean', 'jump_seam_ratio', 'jump_local_seam_ratio', 'gt_disp',
    ]
    print('\n=== Worst 3 capture ===')
    for r in rows_sorted[:3]:
        print({k: round(r[k], 3) for k in show_keys})
    print('=== Best 3 capture ===')
    for r in rows_sorted[-3:]:
        print({k: round(r[k], 3) for k in show_keys})

    js = np.array(acc['jump_local_seam_ratio'])
    jf = np.array(acc['jump_seam_ratio'])
    print('\n=== Seam artifact attribution ===')
    print(f'  local_seam_ratio mean={js.mean():.3f}  full_seam_ratio mean={jf.mean():.3f}')
    print(f'  frac full_seam_ratio > 1.2 = {(jf > 1.2).mean():.3f}')
    print(f'  frac full_seam_ratio > 1.5 = {(jf > 1.5).mean():.3f}')
    print(f'  frac local_seam_ratio > 1.2 = {(js > 1.2).mean():.3f}')

    print('\n=== Expression amplitude (recon/gt) ===')
    for k in ['recon_disp', 'gt_disp']:
        m, s, mn, mx = mean_std(acc[k])
        print(f'  {k}: mean={m:.4f}')
    amp = np.array(acc['recon_disp']) / np.maximum(np.array(acc['gt_disp']), 1e-8)
    print(f'  amp_ratio recon/gt mean={amp.mean():.3f} [{amp.min():.3f},{amp.max():.3f}]')


if __name__ == '__main__':
    main()
