"""PLANMM overfit smoke: multilayer path L1 + VAE KL + anti-cheat checks.

Usage:
    PYTHONPATH=. python exp_poisson/planmm_overfit_smoke.py --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import read_yaml
from dataset.hack_dataset import HACKDataset, TRAIN_HACK_PT
from exp_poisson.planmm_mesh_qa import (
    format_mesh_qa,
    gate_mesh_quality,
    mesh_quality_report,
)
from exp_poisson.train_planmm_ae import (
    build_hackhead_for_mean_shape,
    build_planmm_net,
    compute_mean_shape_from_neck_pose,
    train_step,
    unwrap_net,
    _control_vertices_l1,
    _region_control_vids,
)
from exp_poisson.loss import get_loss
from networks.planmm import build_layer_blend_weights, build_layer_loss_weights
from utils.helpers import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description='PLANMM VAE overfit + mesh QA')
    p.add_argument('--config', type=str, default=str(_REPO_ROOT / 'exp_poisson' / 'planmm_smoke.yaml'))
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', type=str, default='0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-ctrl-tokens', action='store_true', help='Disable CtrlToken dual-path')
    p.add_argument(
        '--ablate-ctrl', action='store_true',
        help='Run overfit with use_ctrl_tokens on/off and print comparison',
    )
    return p.parse_args()


@torch.no_grad()
def evaluate_batch(core, batch, hackhead, device):
    vs = batch['verts'].to(device)
    neck = batch['neck_pose'].to(device)
    xn = compute_mean_shape_from_neck_pose(hackhead, neck, core.template_verts)
    _z, mu, logvar, _enc = core.encode(vs, x_neutral=xn, return_layer_outputs=True)
    recon = core.decode(mu, x_neutral=xn, return_layer_outputs=False)
    gen = core.generate_from_latent(mu, x_neutral=xn)
    # Random prior diversity
    z_rand = torch.randn_like(mu)
    gen_rand = core.generate_from_latent(z_rand, x_neutral=xn)
    zero = core.decode(mu, x_neutral=xn, drop_cond=True, return_layer_outputs=False)

    report = mesh_quality_report(
        recon, vs, xn, core.template_faces,
        core._mesh_edge_src, core._mesh_edge_dst,
        gen=gen,
    )
    report['gen_recon_l1'] = (gen - recon).abs().mean().item()
    # LAMM mean = x_neutral: drop_cond / z=0 → scaffold (neutral face).
    report['zero_err'] = (zero - xn).abs().mean().item()
    report['rand_pw'] = (
        (gen_rand[0] - gen_rand[1]).abs().mean().item() if vs.shape[0] >= 2 else 0.0
    )
    if vs.shape[0] >= 2:
        disp_rand = gen_rand - xn
        report['rand_disp_pw'] = (disp_rand[0] - disp_rand[1]).abs().mean().item()
    else:
        report['rand_disp_pw'] = 0.0
    zero_z = torch.zeros_like(mu)
    zero_from_z = core.decode(zero_z, x_neutral=xn, return_layer_outputs=False)
    report['zero_z_err'] = (zero_from_z - xn).abs().mean().item()
    report['kl'] = core.kl_divergence(mu, logvar).item()
    vids = core._control_vert_ids
    if vids.numel() > 0:
        report['ctrl_l1'] = _control_vertices_l1(recon, vs, vids.to(device)).item()
    mouth_vids = _region_control_vids(core, 9, device)
    if mouth_vids.numel() > 0:
        report['ctrl_l1_mouth'] = _control_vertices_l1(recon, vs, mouth_vids).item()
    eyes_vids = _region_control_vids(core, 7, device)
    if eyes_vids.numel() > 0:
        report['ctrl_l1_eyes'] = _control_vertices_l1(recon, vs, eyes_vids).item()
    id_noise = 0.1 * torch.randn_like(mu)
    report['id_sens'] = (
        core.decode(mu + id_noise, x_neutral=xn, return_layer_outputs=False) - recon
    ).abs().mean().item()
    # Mouth opening recovery vs neutral (translation-invariant expression check).
    if 'Mouth' in getattr(core, 'region_names', []):
        vids = core.region_vert_ids[core.region_names.index('Mouth')].to(device)
        gy = (vs[:, vids, 1].amax(1) - vs[:, vids, 1].amin(1))
        ry = (recon[:, vids, 1].amax(1) - recon[:, vids, 1].amin(1))
        ny = (xn[:, vids, 1].amax(1) - xn[:, vids, 1].amin(1))
        denom = (gy - ny).clamp_min(1e-4)
        report['mouth_open_ratio'] = float(((ry - ny) / denom).clamp(0, 2).mean())
    else:
        report['mouth_open_ratio'] = 0.0
    return report


def _disp_jump_ratio(qa: dict) -> float:
    if 'disp_jump_ratio' in qa:
        return float(qa['disp_jump_ratio'])
    r = qa.get('disp_jump_pred', 0.0)
    g = qa.get('disp_jump_gt', 1.0)
    return float(r) / max(float(g), 1e-8)


def _pick_expressive_indices(ds, core, k: int = 4, scan: int = 1500) -> list[int]:
    """Prefer large mouth openings so smoke cannot pass on near-neutral faces."""
    if 'Mouth' not in getattr(core, 'region_names', []):
        return list(range(min(k, len(ds))))
    vids = core.region_vert_ids[core.region_names.index('Mouth')].cpu()
    scored = []
    n = min(len(ds), scan)
    for i in range(n):
        vs = ds[i]['verts']
        span = float(vs[vids, 1].max() - vs[vids, 1].min())
        scored.append((span, i))
    scored.sort(reverse=True)
    # diversify a bit: take top, then spaced
    picks = [scored[0][1]]
    for span, i in scored[1:]:
        if len(picks) >= k:
            break
        if all(abs(i - p) > 5 for p in picks):
            picks.append(i)
    while len(picks) < k and len(picks) < len(scored):
        picks.append(scored[len(picks)][1])
    return picks[:k]


def run_overfit(model_cfg: dict, config: dict, args, device: torch.device) -> tuple[dict, dict, bool]:
    """Train overfit smoke; returns (qa_final, qa_start, passed)."""
    net = build_planmm_net(model_cfg, device)
    hackhead = build_hackhead_for_mean_shape(model_cfg, device)

    ds = HACKDataset(packed_pt=TRAIN_HACK_PT)
    core = unwrap_net(net)
    indices = _pick_expressive_indices(ds, core, k=args.batch_size)
    print(f'[data] expressive indices={indices}')
    batch = next(iter(DataLoader(Subset(ds, indices), batch_size=len(indices), shuffle=False)))

    print(
        f'[arch] mode={core.decoder_mode} τ+Linear+NJF '
        f'path=τ+global/region-deform '
        f'τ/def/face='
        f'{core.tau_loss_weight}/{core.def_loss_weight}/{core.face_grad_weight} '
        f'enc={core.encoder_depth} dec={core.decoder_depth} '
        f'C={core.C_width} bd={core.bottleneck_dim} K={core.Npatches}'
    )

    lambda_blend = build_layer_blend_weights(
        core.encoder_depth, core.decoder_depth, device=device,
    )
    layer_weights = build_layer_loss_weights(
        core.encoder_depth, core.decoder_depth, device=device,
    )
    optimizer = optim.Adam(net.parameters(), lr=float(config['SOLVER'].get('lr_base', 5e-4)))
    loss_fn = get_loss(config, reduction='none')
    train_decode_mu = bool(config.get('SOLVER', {}).get('train_decode_mu', False))

    core.eval()
    qa0 = evaluate_batch(core, batch, hackhead, device)
    print('[start]\n' + format_mesh_qa(qa0))
    core.train()

    losses = []
    for step in range(1, args.steps + 1):
        _, loss, id_mean, step_stats = train_step(
            net, batch, loss_fn, optimizer, device, lambda_blend, layer_weights,
            hackhead=hackhead,
            train_decode_mu=train_decode_mu,
        )
        losses.append(float(loss.item()))
        if step in (1, 20, args.steps // 2, args.steps) or step % max(args.steps // 5, 1) == 0:
            core.eval()
            qa = evaluate_batch(core, batch, hackhead, device)
            core.train()
            kl_v = step_stats.get('loss_kl', torch.tensor(0.0))
            kl_f = float(kl_v.item() if hasattr(kl_v, 'item') else kl_v)
            print(
                f'step {step:4d}/{args.steps}: loss={loss.item():.4f} '
                f'l1={qa["l1"]:.4f} capture={qa["capture"]:.3f} '
                f'deform={qa.get("deform_capture", 0):.3f} '
                f'ddef={qa.get("deform_div", 0):.3f} '
                f'mouth_open={qa.get("mouth_open_ratio", 0):.3f} '
                f'jump={qa.get("disp_jump_ratio", 0):.3f} '
                f'tau={float(step_stats.get("loss_tau", 0)):.4f} '
                f'def={float(step_stats.get("loss_def", 0)):.4f} '
                f'int={float(step_stats.get("loss_int", 0)):.4f}'
            )

    core.eval()
    qa = evaluate_batch(core, batch, hackhead, device)
    early_min = min(losses[: max(10, len(losses) // 5)])
    mesh_ok, fails = gate_mesh_quality(qa, stage='overfit')
    if qa['l1'] >= qa0['l1'] * 0.7:
        fails.append(f"l1 not reduced ({qa0['l1']:.4f}->{qa['l1']:.4f})")
    if early_min >= losses[0] * 0.95:
        fails.append('early loss did not drop')
    if qa['gen_recon_l1'] > 1e-4:
        fails.append(f"gen!=recon ({qa['gen_recon_l1']:.2e}); must be identical path")
    if qa['rand_pw'] < 0.005:
        fails.append(f"rand id pairwise={qa['rand_pw']:.4f} (want >0.005)")
    if qa['zero_err'] >= 0.02:
        fails.append(f"zero_err={qa['zero_err']:.4f} (drop_cond must be exact scaffold/mean)")
    if qa.get('zero_z_err', 0) >= 0.02:
        fails.append(f"zero_z_err={qa['zero_z_err']:.4f} (z=0 must be near mean face)")
    if qa['id_sens'] <= 5e-5:
        fails.append(f"id_sens={qa['id_sens']:.6f}")
    if not (qa['kl'] == qa['kl']):
        fails.append('kl is NaN')

    ok = len(fails) == 0
    print('\n=== PLANMM Overfit (LAMM residual path, recon==gen) ===')
    print(format_mesh_qa(qa, fails if not ok else None))
    print(
        f'gen_recon_l1={qa["gen_recon_l1"]:.2e} rand_pw={qa["rand_pw"]:.4f} '
        f'rand_disp_pw={qa.get("rand_disp_pw", 0.0):.4f} '
        f'zero_err={qa["zero_err"]:.4f} zero_z_err={qa.get("zero_z_err", float("nan")):.4f} '
        f'ctrl_l1={qa.get("ctrl_l1", float("nan")):.4f} '
        f'mouth={qa.get("ctrl_l1_mouth", float("nan")):.4f} '
        f'eyes={qa.get("ctrl_l1_eyes", float("nan")):.4f} '
        f'disp_jump_ratio={_disp_jump_ratio(qa):.3f} '
        f'id_sens={qa["id_sens"]:.6f} kl={qa["kl"]:.4e}'
    )
    print(f'STATUS: {"PASS" if ok else "FAIL"}')
    for f in fails:
        print(f'  - {f}')
    return qa, qa0, ok


def main():
    args = parse_args()
    seed_everything(args.seed)
    config = read_yaml(args.config)
    device = torch.device(
        f'cuda:{args.device}' if args.device != 'cpu' and torch.cuda.is_available() else 'cpu',
    )

    if args.ablate_ctrl:
        results = {}
        for use_ctrl, label in ((True, 'ctrl_on'), (False, 'ctrl_off')):
            print(f'\n########## ABLATION: {label} ##########')
            model_cfg = dict(config['MODEL'])
            model_cfg['use_ctrl_tokens'] = use_ctrl
            seed_everything(args.seed)
            qa, _qa0, ok = run_overfit(model_cfg, config, args, device)
            results[label] = qa
        print('\n=== CtrlToken ablation summary ===')
        for key in ('ctrl_on', 'ctrl_off'):
            qa = results[key]
            print(
                f'{key}: l1={qa["l1"]:.4f} capture={qa["capture"]:.3f} '
                f'ctrl={qa.get("ctrl_l1", float("nan")):.4f} '
                f'mouth={qa.get("ctrl_l1_mouth", float("nan")):.4f} '
                f'eyes={qa.get("ctrl_l1_eyes", float("nan")):.4f} '
                f'disp_jump={_disp_jump_ratio(qa):.3f}'
            )
        on, off = results['ctrl_on'], results['ctrl_off']
        ctrl_better = on.get('ctrl_l1', 1e9) < off.get('ctrl_l1', 1e9)
        print(f'ctrl_l1 improved with tokens: {ctrl_better}')
        return 0 if ctrl_better else 1

    model_cfg = dict(config['MODEL'])
    if args.no_ctrl_tokens:
        model_cfg['use_ctrl_tokens'] = False
    _qa, _qa0, ok = run_overfit(model_cfg, config, args, device)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
