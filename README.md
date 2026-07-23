# PlanMM

PlanMM is a Poisson-integrated mesh autoencoder for the **HACK** head topology. It tokenizes mesh patches with Poisson operators, encodes/decodes with an MLP-Mixer stack, and reconstructs vertices via a hybrid Poisson mesh readout.

## Dataset

Training uses the **HACK** mesh dataset (`dataset/hack_dataset.py`), built from repaired HACK fits of three sources:

| Source | Role |
|--------|------|
| FaceScape | Multi-identity / expression head scans |
| [ImHead](https://rolpotamias.github.io/imHead/) | Large-scale implicit head shapes used as HACK-fit sources |
| FFHQ | Photo-driven refined HACK meshes |

All samples share the same HACK template connectivity (~14k vertices / ~28k faces). Splits are listed in `dataset/train_hack.pt`, `val_hack.pt`, and `test_hack.pt` (paths point to your local mesh roots; update them before training). Template and region metadata live under `dataset/` and `data/hack_data/`.

## Training

Two stages are supported under `exp_poisson/`:

1. **Autoencoder** — `train_planmm_ae.py`  
   Multilayer L1 on predicted meshes, plus Poisson gradient / seam terms. Typical config: `planmm_hack.yaml` (copy from your local experiment configs and point `DATASETS` / `CHECKPOINT` paths to this repo).

2. **Manipulation** — `train_planmm_manipulation.py`  
   Continues from an AE checkpoint; trains handle-delta control with hybrid mesh readout (`planmm_hack_manipulation.yaml`).

Example:

```bash
# AE
python exp_poisson/train_planmm_ae.py --config exp_poisson/planmm_hack.yaml

# Manipulation (after AE)
python exp_poisson/train_planmm_manipulation.py --config exp_poisson/planmm_hack_manipulation.yaml
```

Evaluation / demos: `planmm_eval_ae.py`, `viz_planmm_generate_local.py`, `viz_mesh_quality_*.py`, etc.

## Checkpoints

Large weights are **not** in git. Place them under `checkpoint/`:

| File | Download |
|------|----------|
| `normal-model-vitl16_384.onnx` | Surface-normal model (Large) from [microsoft/DAViD](https://github.com/microsoft/DAViD) — direct link: [normal-model-vitl16_384.onnx](https://facesyntheticspubwedata.z6.web.core.windows.net/iccv-2025/models/normal-model-vitl16_384.onnx) |
| `epoch_latest.pth` | [Google Drive folder](https://drive.google.com/drive/folders/1qYs-UKUBVvDGw-0dZBzU7eIdkRz7kxT-?usp=drive_link) |

## Layout

```
blender_addon/   # Blender plugin
checkpoint/      # local weights (see above)
configs/         # PHACK stage configs
data/            # HACK assets
dataset/         # dataloaders & split lists
demos/           # quick visualization demos
exp_poisson/     # train / eval / viz scripts
meshes/          # mesh operators, HACK utilities
networks/        # PlanMM / PoissonNet / encoders
render/          # rendering helpers
utils/           # shared utilities
```
