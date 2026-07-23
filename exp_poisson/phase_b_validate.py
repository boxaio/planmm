"""Phase B acceptance: recon retention vs Phase A + random-z generation diversity."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from configs.config_utils import read_yaml
from dataset.hack_dataset import HACKDataset, VAL_HACK_PT
from exp_poisson.train_planmm_ae import (
    _compute_x_neutral,
    _per_sample_disp_capture,
    _resolve_dataset_sources,
    build_hackhead_for_mean_shape,
    build_planmm_net,
    unwrap_net,
)
from exp_poisson._diag_planmm_run import decode_parts, edge_disp_jump
from utils.torch_utils import load_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-a', type=str, required=True, help='Phase A run dir')
    p.add_argument('--run-b', type=str, required=True, help='Phase B run dir')
    p.add_argument('--ckpt', type=str, default='best.pth')
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--device', type=str, default='cuda:0')
    return p.parse_args()


@torch.no_grad()
def eval_run(run_dir: Path, ckpt: str, device: torch.device, idxs: np.ndarray):
    config = read_yaml(str(run_dir / 'config_file.yaml'))
    net = build_planmm_net(config['MODEL'], device)
    load_from_checkpoint(net, str(run_dir / ckpt), partial_restore=False)
    net.eval()
    hackhead = build_hackhead_for_mean_shape(config['MODEL'], device=device)
    sources = _resolve_dataset_sources(config['DATASETS'])
    ds = HACKDataset(packed_pt=VAL_HACK_PT, sources=sources)

    src, dst = net._mesh_edge_src, net._mesh_edge_dst
    acc = {'l1': [], 'capture': [], 'jump_ratio': [], 'cap_mouth': []}
    mouth_vids = net.region_vert_ids[net.region_names.index('Mouth')]

    for i in idxs:
        b = ds[int(i)]
        vs = b['verts'].unsqueeze(0).to(device)
        neck = b['neck_pose'].unsqueeze(0).to(device)
        x_neu = _compute_x_neutral(hackhead, neck, net.template_verts)
        _, mu, _ = net.encode(vs, x_neutral=x_neu, return_layer_outputs=False)
        recon = net.decode(mu, x_neutral=x_neu, return_layer_outputs=False)
        acc['l1'].append(float((recon - vs).abs().mean()))
        acc['capture'].append(float(_per_sample_disp_capture(recon, vs, x_neu)[0]))
        disp_r = recon - x_neu
        disp_g = vs - x_neu
        jr = float(
            edge_disp_jump(disp_r, src, dst)[0]
            / max(float(edge_disp_jump(disp_g, src, dst)[0]), 1e-8)
        )
        acc['jump_ratio'].append(jr)
        dp = recon[:, mouth_vids] - x_neu[:, mouth_vids]
        dg = vs[:, mouth_vids] - x_neu[:, mouth_vids]
        err = (dp - dg).abs().mean()
        gn = dg.abs().mean().clamp_min(1e-8)
        acc['cap_mouth'].append(float(1.0 - err / gn))

    out = {k: float(np.mean(v)) for k, v in acc.items()}
    core = unwrap_net(net)
    B = 8
    x_neu = core.template_verts.unsqueeze(0).expand(B, -1, -1)
    z_rand = torch.randn(B, core.Npatches, core.dim, device=device)
    meshes = []
    for i in range(B):
        meshes.append(
            core.decode(z_rand[i:i + 1], x_neutral=x_neu[i:i + 1], return_layer_outputs=False)
        )
    mstack = torch.cat(meshes, dim=0)
    pw = []
    for i in range(B):
        for j in range(i + 1, B):
            pw.append(float((mstack[i] - mstack[j]).abs().mean()))
    out['rand_pw'] = float(np.mean(pw))

    z0 = torch.zeros(1, core.Npatches, core.dim, device=device)
    neu0 = core.decode(z0, x_neutral=x_neu[:1], return_layer_outputs=False)
    out['z0_err'] = float((neu0 - x_neu[:1]).abs().mean())
    return out


def main():
    args = parse_args()
    device = torch.device(args.device)
    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    sources = _resolve_dataset_sources(read_yaml(str(run_a / 'config_file.yaml'))['DATASETS'])
    ds = HACKDataset(packed_pt=VAL_HACK_PT, sources=sources)
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    a = eval_run(run_a, args.ckpt, device, idxs)
    b = eval_run(run_b, args.ckpt, device, idxs)

    print('=== Phase A ===')
    for k, v in a.items():
        print(f'  {k}: {v:.4f}')

    print('=== Phase B ===')
    for k, v in b.items():
        print(f'  {k}: {v:.4f}')

    print('=== Retention (B vs A, target drop <10%) ===')
    for k in ['l1', 'capture', 'jump_ratio', 'cap_mouth']:
        if a[k] == 0:
            continue
        if k == 'l1':
            drop = (b[k] - a[k]) / a[k] * 100
            ok = drop < 10
        else:
            drop = (a[k] - b[k]) / a[k] * 100
            ok = drop < 10
        print(f'  {k}: A={a[k]:.4f} B={b[k]:.4f}  change={drop:+.1f}%  {"PASS" if ok else "FAIL"}')

    print('=== Generation ===')
    print(f'  rand_pw={b["rand_pw"]:.4f}  (target >0.01)  {"PASS" if b["rand_pw"] > 0.01 else "FAIL"}')
    print(f'  z0_err={b["z0_err"]:.6f}  (target <0.02)  {"PASS" if b["z0_err"] < 0.02 else "FAIL"}')


if __name__ == '__main__':
    main()
