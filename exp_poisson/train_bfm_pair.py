import os
import json
import math
import torch
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pathlib import Path
from typing import Tuple
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader

from pytorch3d.transforms import so3_exponential_map



_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.bfm_pair_dataset import BFMDataset, TRAIN_BFM_NOSE_PT, VAL_BFM_NOSE_PT, build_bfm_nose_pt_cache
from meshes.operators import construct_mesh_operators
from networks.PoissonNet import PoissonNet
from utils.helpers import *
from utils.ps_tools import show_mesh, show_mesh_pair, show_mesh_scalar_field, show_mesh_grad_cloud
from render.mesh_render import render_mesh_on, render_vertex_error_map, add_text, image_grid


seed_everything(31415)
device = torch.device('cuda:0')

# === Config ===
with open('/media/ubuntu/SSD/PHACK_code/exp_poisson/config.json', 'r') as f:
    config = json.load(f)


config['exp_name'] = 'demo_train_bfm_pair_nose'
batch_size = config['batch_size']
grad_accum = config['grad_accum']
lr = config['lr']
clip_grad_norm = config['clip_grad_norm']
train_steps = config['train_steps']
mass_mse = config['mass_mse']
viz_steps = config['viz_steps']
num_extra_features = config['num_extra_features']

lambda_v = config.get('lambda_v', 1.0)
lambda_g = config.get('lambda_g', 1.0)
NOSE_KPT_VID = int(config.get('kpt_vid', 1003))

exp_name = config['exp_name']
os.makedirs(os.path.join('results', exp_name), exist_ok=True)
outfile = lambda x: os.path.join('results', exp_name, x)



# === Data ===（先运行 ``python -m dataset.bfm_dataset`` 或 ``build_bfm_nose_pt_cache()`` 生成 .pt）
if not TRAIN_BFM_NOSE_PT.is_file() or not VAL_BFM_NOSE_PT.is_file():
    build_bfm_nose_pt_cache(
        regions=['nose_0', 'nose_1'], pairing_seed=0, split_seed=0,
    )

train_dataset = BFMDataset(packed_pt=TRAIN_BFM_NOSE_PT)
test_dataset = BFMDataset(packed_pt=VAL_BFM_NOSE_PT)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
train_loader = cycle(train_loader)
test_loader = cycle(test_loader)


model = PoissonNet( 
    C_in=3,
    C_out=3,
    C_width=config['width'],
    n_blocks=config['nblocks'],
    head='njf',
    extra_features=num_extra_features,
    config=config,
)
    
print('Model parameters:', count_parameters(model))
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


def _faces_topology(faces: torch.Tensor) -> torch.Tensor:
    if faces.dim() == 3:
        return faces[0].long()
    return faces.long()


def batched_vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """verts (B, V, 3), faces (F,3) 或与 batch 相同拓扑的 (B,F,3)。"""
    f = _faces_topology(faces)
    B, V, _ = verts.shape
    i0, i1, i2 = f[:, 0], f[:, 1], f[:, 2]
    v0 = verts[:, i0]
    v1 = verts[:, i1]
    v2 = verts[:, i2]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn = F.normalize(fn, dim=-1, eps=1e-8)
    vn = torch.zeros_like(verts)
    F_n = f.shape[0]
    for ij in (i0, i1, i2):
        idx = ij.unsqueeze(0).expand(B, F_n)
        for d in range(3):
            vn[:, :, d].scatter_add_(1, idx, fn[:, :, d])
    return F.normalize(vn, dim=-1, eps=1e-8)


def minimal_rotation_matrices(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    行向量 a,b (B,3)，求最短路径旋转 R (B,3,3)，使得 a @ R^T ≈ b（与 so3 列向量惯例一致）。
    """
    a = F.normalize(a, dim=-1, eps=1e-8)
    b = F.normalize(b, dim=-1, eps=1e-8)
    axb = torch.linalg.cross(a, b, dim=-1)
    s = axb.norm(dim=-1)
    c = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    device, dtype = a.device, a.dtype

    axis = axb / (s.unsqueeze(-1) + 1e-8)
    theta = torch.atan2(s, c)
    log_rot = axis * theta.unsqueeze(-1)

    aux = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).expand_as(a)
    perp = torch.linalg.cross(a, aux, dim=-1)
    alt = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).expand_as(a)
    perp = torch.where(
        perp.norm(dim=-1, keepdim=True) < 1e-6,
        torch.linalg.cross(a, alt, dim=-1),
        perp,
    )
    perp = F.normalize(perp, dim=-1, eps=1e-8)

    log_rot = torch.where((c < -0.9999).unsqueeze(-1), perp * math.pi, log_rot)
    log_rot = torch.where((c > 0.9999).unsqueeze(-1), torch.zeros_like(log_rot), log_rot)
    return so3_exponential_map(log_rot)


def align_pred_to_target_kpt(
    preds: torch.Tensor,
    preds_grad: torch.Tensor,
    verts_tar: torch.Tensor,
    faces: torch.Tensor,
    kpt_vid: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    按鼻尖（或指定）关键点：先将 preds 上该点法线与 verts_tar 对齐（整体旋转），
    再平移使该点与目标重合。对 preds_grad 施加相同旋转（平移不影响梯度）。
    """
    if kpt_vid < 0 or kpt_vid >= preds.shape[1]:
        raise IndexError(
            f'kpt_vid={kpt_vid} out of range V={preds.shape[1]}'
        )
    n_pred = batched_vertex_normals(preds, faces)[:, kpt_vid, :]
    n_tar = batched_vertex_normals(verts_tar, faces)[:, kpt_vid, :]
    R = minimal_rotation_matrices(n_pred, n_tar)
    Rt = R.transpose(1, 2)
    preds_rot = torch.bmm(preds, Rt)
    preds_grad_rot = torch.bmm(preds_grad, Rt)
    p_pred = preds_rot[:, kpt_vid, :]
    p_tar = verts_tar[:, kpt_vid, :]
    t = p_tar - p_pred
    preds_al = preds_rot + t.unsqueeze(1)
    return preds_al, preds_grad_rot



@torch.no_grad()
def form_batch(data_loader, augment=False):
    _dict = next(data_loader)
    verts_src = _dict['source_verts'].to(device)
    verts_tar = _dict['target_verts'].to(device)
    faces = _dict['faces'].to(device)
    target_pose = _dict['target_pose'].to(device)

    if augment: 
        # augment src/target global scale before computing operators
        scale_xyz = torch.rand(verts_src.shape[0], 1, 1, device=verts_src.device) * 0.6 + 0.7
        verts_src = verts_src * scale_xyz
        verts_tar = verts_tar * scale_xyz

        # augment mesh position:
        shift_xyz = torch.randn(verts_src.shape[0], 1, 3, device=verts_src.device) * 0.15
        verts_src = verts_src + shift_xyz
        verts_tar = verts_tar + shift_xyz

    mass_src, solver_src, G_src, M_src = construct_mesh_operators(verts_src, faces, high_precision=True)
    return verts_src, verts_tar, target_pose, faces, mass_src, solver_src, G_src, M_src

def compute_loss(pred_v, pred_grad, tar_v, G, v_mass, f_mass, mass_weighted=True):
    tar_grad = torch.bmm(G, tar_v)
    loss_v = MSE_loss(pred_v, tar_v, v_mass, mass_weighted)
    loss_g = MSE_loss(pred_grad, tar_grad, f_mass, mass_weighted)
    return loss_v, loss_g


def train_batch(batch_i):
    model.train()

    batch_loss_v = 0
    batch_loss_g = 0
    batch_loss = 0
    accums = 0

    while accums < grad_accum:
        verts_src, verts_tar, target_pose, faces, mass_src, solver_src, G_src, M_src \
            = form_batch(train_loader, augment=False)

        preds, preds_grad = model(
            x_in=verts_src,
            M=M_src, 
            G=G_src, 
            solver=solver_src, 
            faces=faces, 
            vertex_mass=mass_src,
            extra_features=target_pose,
        )

        preds, preds_grad = align_pred_to_target_kpt(
            preds, preds_grad, verts_tar, faces, NOSE_KPT_VID
        )

        # show_mesh_pair(
        #     verts_tar[0].detach().cpu().numpy(), 
        #     faces[0].detach().cpu().numpy(),
        #     preds[0].detach().cpu().numpy(), 
        #     faces[0].detach().cpu().numpy(),
        #     use_offset=False,
        # )
        # show_mesh_scalar_field(
        #     preds[0].detach().cpu().numpy(),
        #     faces[0].detach().cpu().numpy(),
        #     mass_src[0].detach().cpu().numpy(),
        #     defined_on='vertices',
        #     show_edge=True,
        #     smooth_shade=False,
        # )
        # show_mesh_grad_cloud(
        #     preds[0].detach().cpu().numpy(),
        #     faces[0].detach().cpu().numpy(),
        #     preds_grad[0].detach().cpu().numpy(),
        #     batch_index=0,
        #     agg='mean_norm',
        #     mesh_name='preds',
        #     quantity_name='preds_grad',
        # )

        # L = λ_v * ‖v_tar - v_pred‖^2 + λ_g * ‖∇_src v_tar - ∇_src v_pred‖^2
        loss_v, loss_g = compute_loss(
            pred_v=preds, 
            pred_grad=preds_grad, 
            tar_v=verts_tar, 
            G=G_src, 
            v_mass=mass_src, 
            f_mass=M_src, 
            mass_weighted=mass_mse,
        )
        loss_v = loss_v * lambda_v
        loss_g = loss_g * lambda_g
        loss = (loss_v + loss_g) / grad_accum
        loss.backward() 

        batch_loss += loss.item()
        batch_loss_v += loss_v.item() / grad_accum
        batch_loss_g += loss_g.item() / grad_accum
        accums += 1

    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

    optimizer.step()
    optimizer.zero_grad()

    if batch_i % viz_steps == 0:
        vsrc_np = to_np(verts_src[0])
        f_np = to_np(faces[0])
        vtar_np = to_np(verts_tar[0])
        preds_np = to_np(preds[0])
        
        render_src = add_text(render_mesh_on(vsrc_np, f_np), caption='source')
        render_tar = add_text(render_mesh_on(vtar_np, f_np), caption='target')
        render_pred = add_text(render_mesh_on(preds_np, f_np), caption='output')
        render_err = add_text(
            render_vertex_error_map(preds_np, vtar_np, f_np, mass_src), caption='error'
        )
        render = torch.cat([render_src, render_tar, render_pred, render_err], dim=-1)
        save_image(render, outfile('viz_train.png'))

    return batch_loss, batch_loss_v, batch_loss_g

@torch.no_grad()
def test():
    model.eval()
    
    total_loss = 0
    total_loss_v = 0
    total_loss_g = 0
    total_samples = 0

    renders = []
    render_idxs = random.sample(range(len(test_dataset)), 9) if len(test_dataset) > 9 else range(len(test_dataset))
    for i in range(len(test_dataset)):
        verts_src, verts_tar, target_pose, faces, mass_src, solver_src, G_src, M_src = form_batch(test_loader)

        preds, preds_grad = model(
            x_in=verts_src,
            M=M_src, 
            G=G_src, 
            solver=solver_src, 
            faces=faces, 
            vertex_mass=mass_src,
            extra_features=target_pose,
        )

        preds, preds_grad = align_pred_to_target_kpt(
            preds, preds_grad, verts_tar, faces, NOSE_KPT_VID
        )

        loss_v, loss_g = compute_loss(preds, preds_grad, verts_tar, G_src, mass_src, M_src, mass_weighted=mass_mse)
        loss_v = loss_v * lambda_v
        loss_g = loss_g * lambda_g
        loss = loss_v + loss_g

        total_loss += loss.item()
        total_loss_v += loss_v.item()
        total_loss_g += loss_g.item()
        total_samples += 1

        if i in render_idxs:
            vsrc_np = to_np(verts_src[0])
            f_np = to_np(faces[0])
            vtar_np = to_np(verts_tar[0])
            preds_np = to_np(preds[0])

            render_src = add_text(render_mesh_on(vsrc_np, f_np), caption='source')
            render_tar = add_text(render_mesh_on(vtar_np, f_np), caption='target')
            render_pred = add_text(render_mesh_on(preds_np, f_np), caption='output')
            render_err = add_text(
                render_vertex_error_map(preds_np, vtar_np, f_np, mass_src), caption='error'
            )
            render = torch.cat([render_src, render_tar, render_pred, render_err], dim=-1)
            renders += [render]

    renders = image_grid(renders)
    save_image(renders, outfile('viz_test.png'))

    total_loss /= total_samples
    total_loss_v /= total_samples
    total_loss_g /= total_samples

    return total_loss


train_losses = []
train_losses_v = []
train_losses_g = []
test_losses = []
test_steps = []

# pbar = tqdm(range(train_steps), dynamic_ncols=True)
pbar = tqdm(range(train_steps))
for step_i in pbar:
    train_loss, train_loss_v, train_loss_g = train_batch(step_i)
    
    train_losses += [train_loss]
    train_losses_v += [train_loss_v]
    train_losses_g += [train_loss_g]

    if step_i % viz_steps == 0 and step_i > 0:
        test_loss = test()
        test_losses += [test_loss]
        test_steps += [step_i]
        torch.save(model.state_dict(), outfile(f'bfm_nose_{step_i}_{test_loss:.4f}.pt'))

    if step_i % 500 == 0:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
        ax[0].plot(train_losses, label='Train')
        ax[0].plot(train_losses_v, label='vertex')
        ax[0].plot(train_losses_g, label='gradient')
        ax[0].set_ylim(0, 0.026)
        ax[0].legend()
        ax[0].set_title('Train loss')
        ax[1].plot(test_steps, test_losses, label='Test')
        ax[1].set_title('Test loss')
        plt.tight_layout()
        plt.savefig(outfile('loss.png'))
        plt.close()

    pbar.set_description(f"Train loss: {train_loss:.5f}")

torch.save(model.state_dict(), outfile(f'bfm_nose_final.pt'))
print("Training complete")
