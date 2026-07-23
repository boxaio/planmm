"""Check latent-only reconstruction quality for PLAMM."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from configs.config_utils import read_yaml
from dataset.hack_dataset import HACKDataset, TRAIN_HACK_PT, VAL_HACK_PT
from exp_poisson.loss import get_loss
from networks.plamm import PLAMM, load_template_from_config


def mean_l1(a, b):
    return (a - b).abs().mean().item()


@torch.no_grad()
def decompose_recon(core, x):
    B = x.shape[0]
    device = x.device
    mass, solver, G, M, faces = core._get_batch_operators(B, device)
    template = core._template_batch(B, device)

    id_token, _enc_feats = core.encode(x)
    cond = core._build_decoder_cond(id_token, B, device, None)
    njf_out, _ = core.decoder(
        template, M, G, solver, faces, mass,
        extra_features=cond,
    )
    njf_delta = njf_out - template

    recon_full, _ = core.decode(id_token)
    recon_njf_only = njf_out

    zero_id = torch.zeros_like(id_token)
    zero_latent, _ = core.decode(zero_id)

    return {
        'full': recon_full,
        'njf_only': recon_njf_only,
        'zero_latent': zero_latent,
        'njf_delta_norm': njf_delta.norm(dim=-1).mean().item(),
        'id_token_mean': id_token.abs().mean().item(),
    }


def eval_loader_loss(net, loader, loss_fn, device, max_batches=20):
    net.eval()
    losses = []
    with torch.no_grad():
        for i, sample in enumerate(loader):
            if i >= max_batches:
                break
            x = sample['verts'].to(device)
            recon = net(x)
            losses.append(loss_fn(recon, x).mean().item())
    return float(np.mean(losses))


def main():
    config = read_yaml(_REPO_ROOT / 'exp_poisson/plamm_hack.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = config['SOLVER']['seed']

    net = PLAMM(config['MODEL']).to(device)
    tv, tf = load_template_from_config(config['MODEL'], device=device)
    net.register_template_mesh(tv, tf)

    ckpt = _REPO_ROOT / 'results/plamm_hack/best.pth'
    if ckpt.is_file():
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f'loaded {ckpt}')
    else:
        print('no best.pth — using current weights in memory if any')

    loss_fn = get_loss(config, reduction='none')

    for split, pt, frac in (
        ('train', TRAIN_HACK_PT, 0.01),
        ('eval', VAL_HACK_PT, 1.0),
    ):
        ds = HACKDataset(packed_pt=pt)
        n = max(1, int(len(ds) * frac))
        idx = np.random.default_rng(seed + (0 if split == 'train' else 1)).choice(len(ds), n, replace=False)
        loader = DataLoader(Subset(ds, idx.tolist()), batch_size=4, shuffle=False)
        loss = eval_loader_loss(net, loader, loss_fn, device)
        print(f'{split} latent-only L1 ({frac * 100:.0f}% samples, {n} total): {loss:.6f}')

    ds = HACKDataset(packed_pt=VAL_HACK_PT)
    x = ds[0]['verts'].unsqueeze(0).to(device)
    gt = x[0]
    tmpl = net.template_verts

    d = decompose_recon(net, x)
    full = net(x)[0]

    print('\n=== Latent-only decomposition (1 eval sample) ===')
    print(f"GT vs full forward:     {mean_l1(full, gt):.6f}")
    print(f"GT vs decomposed full:  {mean_l1(d['full'][0], gt):.6f}")
    print(f"GT vs template:         {mean_l1(tmpl, gt):.6f}")
    print(f"GT vs template+njf:     {mean_l1(d['njf_only'][0], gt):.6f}")
    print(f"GT vs zero_id decode:     {mean_l1(d['zero_latent'][0], gt):.6f}")
    print(f"\n|njf_delta|  mean: {d['njf_delta_norm']:.6f}")
    print(f"|id_token|   mean: {d['id_token_mean']:.6f}")

    zero_frac = mean_l1(d['zero_latent'][0], gt) / (mean_l1(full, gt) + 1e-8)
    print(f"\nzero_id error / full error: {zero_frac:.3f}  (>>1 = id_token drives reconstruction)")

    if mean_l1(full, gt) < 0.5 * mean_l1(tmpl, gt):
        print('\n结论: latent-only 重建显著优于 template')
    else:
        print('\n结论: latent-only 重建仍接近 template，bottleneck 可能未充分学习')


if __name__ == '__main__':
    main()
