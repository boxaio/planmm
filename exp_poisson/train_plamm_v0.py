import numpy as np
import torch
import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    
import polyscope as ps
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from configs.config_utils import *
from dataset.bfm_dataset import BFMDataset, TRAIN_BFM_PT, VAL_BFM_PT
from networks.plamm_v0 import PLAMM
from utils.helpers import *
from utils.torch_utils import load_from_checkpoint, get_net_trainable_params

from exp_poisson.loss import get_loss
from exp_poisson.lr_scheduler import build_scheduler
from exp_poisson.summaries import write_mean_summaries



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PyTorch LAMM Training')
    parser.add_argument(
        '--config', type=str, 
        default="/media/ubuntu/SSD/PHACK_code/exp_poisson/plamm_ae.yaml",
        help='configuration (.yaml) file to use',
    )
    parser.add_argument(
        '--device', type=str, default='0',
        help='devices to use, use comma separated values for multiple gpus e.g. "0,1"',
    )
    return parser.parse_args()



def get_dataloaders(config):
    dataloaders = {}
    # TRAIN data -----------------------
    train_dataset = BFMDataset(packed_pt=TRAIN_BFM_PT)
    dataloaders['training'] = DataLoader(
        train_dataset,
        batch_size=config['DATASETS']['train']['batch_size'], 
        num_workers=4, shuffle=True,
    )
    # EVAL data -------------------------------
    val_dataset = BFMDataset(packed_pt=VAL_BFM_PT)
    dataloaders['eval'] = DataLoader(
        val_dataset,
        batch_size=config['DATASETS']['eval']['batch_size'], 
        num_workers=4, shuffle=False,
    )

    return dataloaders


def evaluate(net, eval_loader, loss_fn, device):
    """
    Performs model evaluation for mesh dimensionality reduction using the LAMM network.

    Parameters:
    - net (torch.nn.Module): The LAMM network instance to be trained.
    - eval_loader (torch.utils.data.dataloader.DataLoader): A torch DataLoader.
    - loss_fn (torch.nn.modules.loss._Loss): The loss function used for training.
    - device (str): The device on which to perform calculations. Accepts strings like 'cpu' or 'cuda:0'.
    """
    losses_all = []
    net.eval()
    with torch.no_grad():
        for step, sample in enumerate(eval_loader):
            Vs = sample['verts'].to(device)
            Vt = sample['verts'].to(device)
            outputs = net(Vs)[-1]
            # For evaluation calculate the loss only at the output layer of the model
            loss = loss_fn(outputs, Vt).mean()
            losses_all.append(loss.cpu().detach().numpy())
    mean_loss = np.mean(losses_all)
    # return the mean loss at the output layer (averaged over num batches, num vertices and dimensions)
    return mean_loss


def train_step(net, sample, loss_fn, lambda_target, optimizer, device):
    """
    Performs a single training step for mesh dimensionality reduction using the LAMM network.

    This function executes a forward pass, computes the loss, performs a backward pass,
    and updates the network's weights accordingly.

    Parameters:
    - net (torch.nn.Module): The LAMM network instance to be trained.
    - sample (dict): A batch of training samples.
    - loss_fn (torch.nn.modules.loss._Loss): The loss function used for training.
    - lambda_target (torch.Tensor): A tensor representing **per-layer** loss weights for the network.
    - optimizer (torch.optim.Optimizer): The optimizer used for adjusting the network's weights.
    - device (str): The device on which to perform calculations. Accepts strings like 'cpu' or 'cuda:0'.
    """
    # For dimensionality reduction task, source and target are the same
    Vs = sample['verts'].to(device)
    Vt = sample['verts'].to(device)
    outputs = net(Vs)  # [m, B, 11916, 3], where m = encoder_depth + decoder_depth + 2

    # Calculate target geometry at every layer of the network for multilayer loss (eqs.1,2). 
    # We use a zero tensor object of the same dimensionality as data samples 
    # to define mean geometry assuming all data are centered by the
    # dataloader which this codebase assumes to be the case
    Vt_expanded = (1 - lambda_target) * torch.zeros_like(Vt) + lambda_target * Vt  # [m, B, 11916, 3]
    loss = loss_fn(outputs, Vt_expanded)  # [m, 1, 11916, 3]

    optimizer.zero_grad()
    (lambda_target * loss).sum().backward()
    optimizer.step()
    # Return the mean loss per layer, averaged over batches, vertices and dimensions
    return loss.mean(dim=[1, 2, 3])



def main():

    args = parse_args()

    # This script should work fine in most cases using only single GPU training since LAMM is very lightweight in terms
    # of memory. Training on 12k vertex meshes with batch size 32 requires ~2GB of GPU RAM.
    device_ids = [int(d) for d in args.device.split(',')]
    device = f'cuda:{device_ids[0]}'

    config = read_yaml(args.config)
    seed_everything(31415)

    dataloaders = get_dataloaders(config)

    net = PLAMM(config['MODEL']).to(device)

    # number of trainable parameters
    print(f"number of trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

    # Training hyperparameters
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_dir = config['CHECKPOINT']["save_dir"]
    num_steps_epoch = len(dataloaders['training'])
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    # lambda_target is used to weight the contribution of the ground truths in the loss at each layer (l) in the
    # multilayer loss formulation presented in eqs.(1,2) (equivalent to the term l/L).
    lambda_target = torch.cat((
        torch.linspace(1, 0, config['MODEL']['encoder_depth'] + 1),
        torch.linspace(0, 1, config['MODEL']['decoder_depth'] + 1),
    )).view(config['MODEL']['encoder_depth'] + config['MODEL']['decoder_depth'] + 2, 1, 1, 1).to(device)

    # lambda_target is used to weight the contribution of each layer (l) in the multilayer loss formulation
    # presented in eqs.(1,2).
    lambda_target = torch.tensor(config['SOLVER']['weights'], dtype=torch.float32, device=device).view(lambda_target.shape)

    checkpoint_file = config['CHECKPOINT']["load_from_checkpoint"]
    if checkpoint_file:
        load_from_checkpoint(net, checkpoint_file, partial_restore=False)
    print("current learning rate: ", lr)

    if len(device_ids) > 1:
        net = nn.DataParallel(net, device_ids=device_ids)
    net.to(device)

    if save_dir and (not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    copy_yaml(args.config)

    loss_fn = get_loss(config, reduction='none')
    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    scheduler = build_scheduler(config, optimizer, num_steps_epoch)
    writer = SummaryWriter(save_dir)

    best_eval_loss = 1e10
    net.train()
    for epoch in range(1, num_epochs+1):
        for step, sample in enumerate(dataloaders['training']):
            abs_step = (epoch - 1) * num_steps_epoch + step + 1

            loss = train_step(net, sample, loss_fn, lambda_target, optimizer, device)

            if abs_step % train_metrics_steps == 0:
                write_mean_summaries(
                    writer, {f'train_loss_{i}': loss[i].item() for i in range(loss.shape[0])},
                    abs_step, mode="training", optimizer=optimizer,
                )
                print(
                    f"abs_step: {abs_step}, epoch: {epoch}, step: {step+1}, loss: {loss.data.tolist()}, "
                    f"learn rate: {optimizer.param_groups[0]['lr']}"
                )

            if abs_step % eval_steps == 0:
                print('--------------------- EVAL ----------------------------')
                eval_loss = evaluate(net, dataloaders['eval'], loss_fn, device)

                if eval_loss < best_eval_loss:
                    if len(device_ids) > 1:
                        torch.save(net.module.state_dict(), f"{save_dir}/best.pth")
                    else:
                        torch.save(net.state_dict(), f"{save_dir}/best.pth")
                    best_eval_loss = eval_loss

                write_mean_summaries(writer, {'eval_loss': eval_loss}, abs_step, mode="eval_micro", optimizer=None)
                print(f"abs_step: {abs_step}, epoch: {epoch}, step: {step + 1}, loss: {eval_loss}")
                net.train()

        scheduler.step_update(abs_step)


if __name__ == "__main__":
    main()

