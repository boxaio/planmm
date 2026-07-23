"""Load PoissonNet checkpoint and render 4 uniformly spaced time frames with Blender."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from meshes.operators import construct_mesh_operators
from networks.PoissonAGC import build_poisson_model
from networks.PoissonDetailResidual import build_expression_region_mask
from utils.helpers import seed_everything, to_np
from exp_poisson.train_poisson_track_phack import PHACKMeshDataset
from figures.demo_blender_render_hack import blender_render, crop_rgba_whitespace

DEFAULT_CKPT = (
    _REPO_ROOT / "results" / "poissonnet_nersemble_phack" / "poissonnet_nersemble_phack_final.pt"
)
DEFAULT_OUTPUT_DIR = (
    _REPO_ROOT / "results" / "poissonnet_nersemble_phack" / "blender_viz_frames"
)
CONFIG_PATH = _REPO_ROOT / "exp_poisson" / "nersemble_phack_config.json"
NUM_FRAMES = 4


def save_obj_mesh(vertices: np.ndarray, faces: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def load_model(checkpoint: Path, config: dict, device: torch.device, dataset: PHACKMeshDataset):
    detail_cfg = config.get("detail_residual", {})
    region_mask = None
    if detail_cfg.get("enabled", False):
        regions = detail_cfg.get("regions")
        region_kwargs = {}
        if regions:
            region_kwargs["region_names"] = tuple(regions)
        _, faces_ref = dataset.get_source_cloth()
        region_mask = build_expression_region_mask(
            n_vertices=dataset.verts.shape[1],
            faces=faces_ref.numpy(),
            **region_kwargs,
        )
    model = build_poisson_model(
        config,
        region_mask=region_mask,
        C_in=3,
        C_out=3,
        C_width=config["width"],
        n_blocks=config["nblocks"],
        head_type="njf",
        extra_features=1,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_meshes_at_times(
    model,
    dataset: PHACKMeshDataset,
    times: list[float],
    device: torch.device,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    verts_src, faces = dataset.get_source_cloth()
    verts_src = verts_src.to(device).unsqueeze(0)
    faces = faces.to(device).unsqueeze(0)
    mass, solver, G, M = construct_mesh_operators(verts_src, faces, high_precision=True)

    results: list[tuple[float, np.ndarray, np.ndarray]] = []
    for t_val in times:
        t = torch.tensor([[t_val]], dtype=torch.float32, device=device)
        preds, _ = model(
            x_in=verts_src,
            M=M,
            G=G,
            solver=solver,
            faces=faces,
            vertex_mass=mass,
            extra_features=t,
        )
        preds = preds - preds.mean(dim=1, keepdim=True)
        results.append((t_val, to_np(preds[0]), to_np(faces[0])))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render PoissonNet PHACK predictions at uniformly spaced times."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subject-id", type=str, default="224")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--res", type=int, default=80, help="Blender render resolution percent")
    parser.add_argument("--trim-pad", type=int, default=4)
    args = parser.parse_args()

    seed_everything(31415)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    times = np.linspace(0.0, 1.0, args.num_frames).tolist()
    output_dir = args.output_dir.expanduser().resolve()
    raw_dir = output_dir / "_raw"
    mesh_dir = output_dir / "_meshes"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset subject {args.subject_id}...")
    dataset = PHACKMeshDataset(subject_id=args.subject_id)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, config, device, dataset)

    verts_src, faces_src = dataset.get_source_cloth()
    if hasattr(model, "register_source_mesh"):
        model.register_source_mesh(
            verts_src.to(device),
            faces_src.to(device),
        )

    print(f"Predicting {args.num_frames} frames at t = {[round(t, 3) for t in times]}")
    predictions = predict_meshes_at_times(model, dataset, times, device)

    for t_val, verts_np, faces_np in predictions:
        name = f"poissonnet_t{t_val:.2f}"
        mesh_path = mesh_dir / f"{name}.obj"
        raw_png = raw_dir / f"{name}.png"
        final_png = output_dir / f"{name}.png"

        save_obj_mesh(verts_np, faces_np, mesh_path)
        print(f"Blender rendering {name}...")
        blender_render(mesh_path=str(mesh_path), out_file=str(raw_png), res=args.res)

        img = np.array(Image.open(raw_png).convert("RGBA"))
        cropped = crop_rgba_whitespace(img, pad=args.trim_pad)
        Image.fromarray(cropped, mode="RGBA").save(final_png)
        print(f"Saved → {final_png}")

    print(f"Done. {len(predictions)} PNGs saved to {output_dir}")


if __name__ == "__main__":
    main()
