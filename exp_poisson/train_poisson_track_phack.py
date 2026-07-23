import os
import re
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from meshes.operators import construct_mesh_operators
from networks.PoissonAGC import build_poisson_model
from networks.PoissonDetailResidual import build_expression_region_mask
from utils.helpers import cycle, seed_everything, count_parameters, MSE_loss, to_np
from utils.mesh import normalize_mesh, read_obj
from render.mesh_render import render_mesh_on, render_vertex_error_map, add_text
from utils.video import save_tensor_as_video


DEFAULT_REPAIR_DIR = Path(__file__).resolve().parent / "phack_nersemble" / "phack_repair"
SUBJECT_CONFIGS = {
    "224": "224_cam_222200037",
    "473": "473_cam_222200037",
}
_FRAME_RE = re.compile(r"_(\d+)_repair$")


def _parse_orig_frame_id(path: Path) -> int:
    match = _FRAME_RE.search(path.stem)
    if match is None:
        raise ValueError(f"Cannot parse frame id from {path.name}")
    return int(match.group(1))


def collect_phack_repair_paths(
    repair_dir: Path,
    subject_id: str,
) -> list[Path]:
    if subject_id not in SUBJECT_CONFIGS:
        raise ValueError(f"subject_id must be one of {list(SUBJECT_CONFIGS.keys())}, got {subject_id!r}")

    repair_dir = Path(repair_dir)
    if not repair_dir.is_dir():
        raise FileNotFoundError(f"repair dir not found: {repair_dir}")

    prefix = SUBJECT_CONFIGS[subject_id]
    paths = sorted(
        repair_dir.glob(f"{prefix}_*_repair.obj"),
        key=_parse_orig_frame_id,
    )
    if not paths:
        raise FileNotFoundError(
            f"No repair meshes matched prefix {prefix!r} under {repair_dir}"
        )
    return paths


class PHACKMeshDataset(Dataset):
    """PHACK repair mesh sequence for one NeRSemble subject (224 or 473).

    Meshes are ordered by the original frame id in the filename, but ``t_vals``
    and ``__getitem__`` indices use contiguous re-numbering 0..N-1.
    """

    def __init__(self, subject_id, repair_dir=DEFAULT_REPAIR_DIR):
        self.subject_id = str(subject_id)
        self.repair_dir = Path(repair_dir or DEFAULT_REPAIR_DIR)
        self.obj_paths = collect_phack_repair_paths(self.repair_dir, self.subject_id)
        self.orig_frame_ids = [_parse_orig_frame_id(p) for p in self.obj_paths]
        self.seq_frame_ids = list(range(len(self.obj_paths)))

        nframes = len(self.obj_paths)
        self.t_vals = torch.linspace(0, 1, nframes).unsqueeze(1)

        verts_list = []
        faces_ref = None
        for path in tqdm(self.obj_paths, desc=f"load PHACK {self.subject_id}"):
            obj = read_obj(str(path), tri=True)
            verts = np.asarray(obj.vs, dtype=np.float32)
            faces = np.asarray(obj.fvs, dtype=np.int64)
            if faces_ref is None:
                faces_ref = faces
            elif faces.shape != faces_ref.shape or not np.array_equal(faces, faces_ref):
                raise ValueError(f"{path.name}: face topology differs from reference mesh")

            verts = normalize_mesh(verts, faces_ref, mode="surface_area")
            verts = verts - np.mean(verts, axis=0)
            verts_list.append(verts)

        self.verts = torch.from_numpy(np.stack(verts_list, axis=0)).float()
        self.faces = torch.from_numpy(faces_ref).long()
        print(
            f"Loaded PHACK subject {self.subject_id}: "
            f"{nframes} frames, verts {tuple(self.verts.shape)}, faces {tuple(self.faces.shape)}"
        )

    def __len__(self):
        return len(self.verts)

    def __getitem__(self, idx):
        return self.verts[idx], self.faces, self.t_vals[idx]

    def get_source_mesh(self):
        """Reference mesh: first frame of the re-numbered sequence."""
        return self.verts[0], self.faces

    def get_source_cloth(self):
        return self.get_source_mesh()


def main():
    seed_everything(31415)
    device = torch.device('cuda:1')

    with open('./exp_poisson/nersemble_phack_config.json', 'r') as f:
        config = json.load(f)

    detail_cfg = config.get('detail_residual', {})
    detail_enabled = detail_cfg.get('enabled', False)

    model_type = config.get('model_type', 'poissonnet')
    if detail_enabled:
        default_exp_name = detail_cfg.get('exp_name', 'poisson_agc_detail_nersemble_phack')
    elif model_type == 'poisson_agc':
        default_exp_name = 'poisson_agc_nersemble_phack'
    else:
        default_exp_name = 'poissonnet_nersemble_phack'
    config['exp_name'] = config.get('exp_name', default_exp_name)
    config['device'] = str(device)
    batch_size = config['batch_size']
    grad_accum = config['grad_accum']
    lr = detail_cfg.get('lr', config['lr']) if detail_enabled else config['lr']
    clip_grad_norm = config['clip_grad_norm']
    train_steps = detail_cfg.get('train_steps', 5000) if detail_enabled else config['train_steps']
    mass_mse = config['mass_mse']
    viz_steps = config['viz_steps']

    lambda_v = config['lambda_v']
    lambda_g = config['lambda_g']
    lambda_res = detail_cfg.get('lambda_res', 0.0) if detail_enabled else 0.0
    mask_only_loss = detail_enabled and detail_cfg.get('mask_only_loss', True)
    lambda_v_mask = detail_cfg.get('lambda_v_mask', 100.0) if mask_only_loss else 0.0
    lambda_g_mask = detail_cfg.get('lambda_g_mask', 0.0) if mask_only_loss else 0.0
    center_outputs = not mask_only_loss and config.get('center_outputs', True)

    exp_name = config['exp_name']
    os.makedirs(os.path.join('results', exp_name), exist_ok=True)
    outfile = lambda x: os.path.join('results', exp_name, x)

    train_dataset = PHACKMeshDataset(subject_id="224")
    test_dataset = PHACKMeshDataset(subject_id="224")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    train_loader = cycle(train_loader)
    test_loader = cycle(test_loader)

    _, faces_ref = train_dataset.get_source_cloth()
    region_mask = None
    if detail_enabled:
        regions = detail_cfg.get('regions')
        region_kwargs = {}
        if regions:
            region_kwargs['region_names'] = tuple(regions)
        region_mask = build_expression_region_mask(
            n_vertices=train_dataset.verts.shape[1],
            faces=faces_ref.numpy(),
            **region_kwargs,
        )

    model = build_poisson_model(
        config,
        region_mask=region_mask,
        C_in=3,
        C_out=3,
        C_width=config['width'],
        n_blocks=config['nblocks'],
        head_type='njf',
        extra_features=1,
    )

    print(f'Model type: {model_type}')
    if detail_enabled:
        print('Detail residual: enabled (frozen PoissonAGC base)')
        if mask_only_loss:
            print(
                f'Mask-only detail loss: lambda_v_mask={lambda_v_mask}, '
                f'lambda_g_mask={lambda_g_mask}, center_outputs={center_outputs}'
            )
    print('Model parameters:', count_parameters(model))
    model = model.to(device)

    if detail_enabled:
        base_ckpt = Path(detail_cfg['base_checkpoint'])
        if not base_ckpt.is_file():
            raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt}")
        model.load_base_checkpoint(base_ckpt, device)
        model.set_base_trainable(False)
        region_mask = region_mask.to(device)
        print(f'Loaded frozen base from {base_ckpt}')
        print('Detail head parameters:', count_parameters(model.detail_head))
        optimizer = torch.optim.Adam(model.detail_parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    verts_src, faces_src = train_dataset.get_source_cloth()
    verts_src, faces_src = verts_src.to(device).unsqueeze(0), faces_src.to(device).unsqueeze(0)
    verts_src = verts_src.repeat(batch_size, 1, 1)
    faces_src = faces_src.repeat(batch_size, 1, 1)
    mass_src, solver_src, G_src, M_src = construct_mesh_operators(verts_src, faces_src, high_precision=True)

    if hasattr(model, 'register_source_mesh'):
        model.register_source_mesh(verts_src[0], faces_src[0])
        print('Registered digeo source mesh for PoissonAGC')

    @torch.no_grad()
    def form_batch(data_loader):
        verts, _, time = next(data_loader)
        verts = verts.to(device)
        time = time.to(device)

        if time.ndim == 1:
            time = time.unsqueeze(0)

        return verts, time

    def compute_loss(pred_v, pred_grad, tar_v, G, v_mass, f_mass, mass_weighted=True):
        tar_grad = torch.bmm(G, tar_v)
        loss_v = MSE_loss(pred_v, tar_v, v_mass, mass_weighted)
        loss_g = MSE_loss(pred_grad, tar_grad, f_mass, mass_weighted)
        return loss_v, loss_g

    def compute_mask_vertex_loss(pred_v, tar_v, mask):
        return torch.mean(torch.abs(pred_v[:, mask] - tar_v[:, mask]))

    def compute_residual_loss(delta, v_low, tar_v, mask):
        delta_tar = tar_v - v_low
        return torch.mean((delta[:, mask] - delta_tar[:, mask]) ** 2)

    def train_batch(batch_i):
        model.train()

        batch_loss_v = 0
        batch_loss_g = 0
        batch_loss_res = 0
        batch_loss = 0
        batch_loss_v_mask = 0
        accums = 0

        while accums < grad_accum:
            verts_tar, time = form_batch(train_loader)
            expr_mask = model.region_mask.bool()

            if detail_enabled:
                preds, preds_grad, aux = model(
                    x_in=verts_src, M=M_src, G=G_src, solver=solver_src,
                    faces=faces_src, vertex_mass=mass_src, extra_features=time,
                    return_aux=True,
                )
            else:
                preds, preds_grad = model(
                    x_in=verts_src, M=M_src, G=G_src, solver=solver_src,
                    faces=faces_src, vertex_mass=mass_src, extra_features=time,
                )

            if center_outputs:
                preds_mean = preds.mean(dim=1, keepdim=True)
                tar_mean = verts_tar.mean(dim=1, keepdim=True)
                verts_tar_centered = verts_tar - tar_mean
                preds_centered = preds - preds_mean
            else:
                verts_tar_centered = verts_tar
                preds_centered = preds

            if mask_only_loss:
                loss_v = torch.tensor(0.0, device=device)
                loss_g = torch.tensor(0.0, device=device)
                loss_v_mask = compute_mask_vertex_loss(preds, verts_tar, expr_mask)
                loss_v_mask = loss_v_mask * lambda_v_mask
                loss = loss_v_mask / grad_accum

                if lambda_g_mask > 0:
                    tar_grad = torch.bmm(G_src, verts_tar)
                    loss_g_mask = torch.mean(
                        torch.abs(preds_grad[:, expr_mask] - tar_grad[:, expr_mask])
                    ) * lambda_g_mask
                    loss = loss + loss_g_mask / grad_accum
                    batch_loss_g += loss_g_mask.item() / grad_accum

                batch_loss_v_mask += loss_v_mask.item() / grad_accum
            else:
                loss_v, loss_g = compute_loss(
                    pred_v=preds_centered,
                    pred_grad=preds_grad,
                    tar_v=verts_tar_centered,
                    G=G_src,
                    v_mass=mass_src,
                    f_mass=M_src,
                    mass_weighted=mass_mse,
                )
                loss_v = loss_v * lambda_v
                loss_g = loss_g * lambda_g
                loss = (loss_v + loss_g) / grad_accum
                batch_loss_v += loss_v.item() / grad_accum
                batch_loss_g += loss_g.item() / grad_accum

            if detail_enabled and lambda_res > 0:
                loss_res = compute_residual_loss(aux['delta'], aux['v_low'], verts_tar, expr_mask)
                loss = loss + (loss_res * lambda_res) / grad_accum
                batch_loss_res += loss_res.item() / grad_accum

            loss.backward()

            batch_loss += loss.item()
            accums += 1

        if clip_grad_norm is not None:
            params = model.detail_parameters() if detail_enabled else model.parameters()
            torch.nn.utils.clip_grad_norm_(params, clip_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        if batch_i % viz_steps == 0:
            vsrc_np = to_np(verts_src[0])
            f_np = to_np(faces_src[0])
            vtar_np = to_np(verts_tar[0])
            preds_np = to_np(preds[0])

            render_src = add_text(render_mesh_on(vsrc_np, f_np), caption='source')
            render_tar = add_text(render_mesh_on(vtar_np, f_np), caption='target')
            render_pred = add_text(render_mesh_on(preds_np, f_np), caption='output')
            render_err = add_text(
                render_vertex_error_map(preds_np, vtar_np, f_np, mass_src),
                caption='error',
            )
            panels = [render_src, render_tar, render_pred, render_err]
            if detail_enabled and 'v_low' in aux:
                render_base = add_text(
                    render_mesh_on(to_np(aux['v_low'][0]), f_np),
                    caption='base',
                )
                panels = [render_base] + panels
            save_image(torch.cat(panels, dim=-1), outfile('viz_train.png'))

        if mask_only_loss:
            return batch_loss, batch_loss_v_mask, batch_loss_g, batch_loss_res
        return batch_loss, batch_loss_v, batch_loss_g, batch_loss_res

    @torch.no_grad()
    def render_video(step_i):
        print('Rendering nersemble_phack video...')
        model.eval()

        nframes = len(train_dataset)
        time = torch.linspace(0, 1, nframes).unsqueeze(-1).to(device)

        _verts_src = verts_src[0].unsqueeze(0)
        _faces_src = faces_src[0].unsqueeze(0)
        _mass_src = mass_src[0].unsqueeze(0)
        _M_src = M_src[0].unsqueeze(0)
        _G_src = G_src[0].unsqueeze(0)
        _solver_src = solver_src[:1]

        all_frames = []
        for i in tqdm(range(nframes), dynamic_ncols=True, desc='Decoding and rendering nersemble_phack video'):
            t = time[i:i + 1]
            preds, _ = model(
                x_in=_verts_src, M=_M_src, G=_G_src, solver=_solver_src,
                faces=_faces_src, vertex_mass=_mass_src, extra_features=t,
            )
            if center_outputs:
                preds = preds - preds.mean(dim=1, keepdim=True)
            preds_np = to_np(preds[0])
            f_np = to_np(_faces_src[0])
            vgt_np = to_np(train_dataset[i][0])
            render_pred = add_text(render_mesh_on(preds_np, f_np), caption=f'Prediction - time={t.item():.2f}')
            render_gt = add_text(render_mesh_on(vgt_np, f_np), caption=f'Ground truth - time={t.item():.2f}')
            all_frames.append(torch.cat([render_pred, render_gt], dim=-1))

        save_tensor_as_video(
            torch.stack(all_frames, dim=0),
            outfile(f'nersemble_phack_{step_i}'),
            reencode=True,
            boomerang=True,
            fps=24,
        )

    train_losses = []
    train_losses_v = []
    train_losses_g = []
    train_losses_res = []
    train_losses_v_mask = []
    pbar = tqdm(range(train_steps), dynamic_ncols=True)
    for step_i in pbar:
        batch_out = train_batch(step_i)
        if detail_enabled:
            if mask_only_loss:
                train_loss, train_loss_v, train_loss_g, train_loss_res = batch_out
                train_losses_v_mask.append(train_loss_v)
            else:
                train_loss, train_loss_v, train_loss_g, train_loss_res = batch_out
            train_losses_res.append(train_loss_res)
        else:
            train_loss, train_loss_v, train_loss_g = batch_out
        train_losses.append(train_loss)
        train_losses_v.append(train_loss_v)
        train_losses_g.append(train_loss_g)

        if (step_i % 4000 == 0 or step_i == train_steps - 1) and step_i > 0:
            render_video(step_i)
            torch.save(model.state_dict(), outfile(f'{exp_name}_{step_i}.pt'))

        if step_i % config.get('print_freq', 100) == 0:
            print("Step {} - Train overall: {:.5f}".format(step_i, train_loss))
            fig, ax = plt.subplots(figsize=(20, 5))
            ax.plot(train_losses, label='Train')
            ax.plot(train_losses_v, label='vertex' if not mask_only_loss else 'mask_vertex')
            ax.plot(train_losses_g, label='gradient')
            if detail_enabled:
                ax.plot(train_losses_res, label='residual')
            ax.set_ylim(0, max(0.1, max(train_losses) * 1.2) if train_losses else 0.1)
            ax.legend()
            ax.set_title('Train loss')
            plt.tight_layout()
            plt.savefig(outfile('viz_loss.png'))
            plt.close()

        desc = f"Train loss: {train_loss:.5f}"
        if detail_enabled:
            if mask_only_loss:
                desc += f" mask_v: {train_loss_v:.5f}"
            desc += f" res: {train_loss_res:.5f}"
        pbar.set_description(desc)

    torch.save(model.state_dict(), outfile(f'{exp_name}_final.pt'))


if __name__ == '__main__':
    main()
