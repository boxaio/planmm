"""Render PoissonAGC predictions and dARAP-refined meshes at key frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exp_poisson.demo_poissonagc_dARAP import (
    CONFIG_PATH,
    DEFAULT_CKPT,
    deform_pred_to_target_normals,
    load_poisson_agc,
)
from exp_poisson.train_poisson_track_phack import (
    DEFAULT_REPAIR_DIR,
    PHACKMeshDataset,
    SUBJECT_CONFIGS,
    _parse_orig_frame_id,
)
from figures.demo_blender_render_hack import align_rgba_images, blender_render, crop_rgba_whitespace
from render.mesh_render import render_mesh, vertex_error_colors
from utils.mesh import read_obj

SUBJECT_ID = "224"
CHOSEN_FRAMES = ["000", "018", "078", "196", "226"]
DEFAULT_OUTPUT = (
    _REPO_ROOT / "results" / "poisson_agc_nersemble_phack" / "viz_poissonagc_dARAP.png"
)
DEFAULT_WORK_DIR = DEFAULT_OUTPUT.parent / "_viz_poissonagc_dARAP"
DEFAULT_MESH_DIR = DEFAULT_WORK_DIR / "meshes"


def repair_mesh_path(subject_id: str, frame_token: str, repair_dir: Path = DEFAULT_REPAIR_DIR) -> Path:
    prefix = SUBJECT_CONFIGS[subject_id]
    token = frame_token.strip()
    if token.endswith(".obj"):
        token = token[:-4]
    if token.endswith("_repair"):
        return Path(repair_dir) / f"{token}.obj"
    frame_id = int(token)
    return Path(repair_dir) / f"{prefix}_{frame_id:03d}_repair.obj"


def resolve_chosen_frames(
    dataset: PHACKMeshDataset,
    chosen: list[str],
    *,
    repair_dir: Path = DEFAULT_REPAIR_DIR,
) -> list[tuple[int, Path, int]]:
    orig_to_seq = {orig_id: seq_idx for seq_idx, orig_id in enumerate(dataset.orig_frame_ids)}
    resolved: list[tuple[int, Path, int]] = []

    for token in chosen:
        path = repair_mesh_path(dataset.subject_id, token, repair_dir)
        if not path.is_file():
            raise FileNotFoundError(f"repair mesh not found: {path}")

        orig_id = _parse_orig_frame_id(path)
        if orig_id not in orig_to_seq:
            raise ValueError(
                f"{path.name}: orig frame id {orig_id} not found in subject {dataset.subject_id} sequence"
            )
        seq_idx = orig_to_seq[orig_id]
        resolved.append((seq_idx, path, orig_id))

    return resolved


def mesh_job_paths(mesh_dir: Path, repair_stems: list[str]) -> tuple[list[Path], list[Path]]:
    pred_paths = [mesh_dir / f"pred_{stem}.obj" for stem in repair_stems]
    darap_paths = [mesh_dir / f"darap_{stem}.obj" for stem in repair_stems]
    return pred_paths, darap_paths


def meshes_are_ready(mesh_dir: Path, repair_stems: list[str]) -> bool:
    pred_paths, darap_paths = mesh_job_paths(mesh_dir, repair_stems)
    return all(p.is_file() for p in pred_paths + darap_paths)


def save_obj_mesh(vertices: np.ndarray, faces: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    obj = read_obj(str(path), tri=True)
    verts = np.asarray(obj.vs, dtype=np.float64)
    faces = np.asarray(obj.fvs, dtype=np.int64)
    return verts, faces


def _chw_to_rgba(img_chw) -> np.ndarray:
    rgb = (
        img_chw.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)
    fg = (np.abs(rgb.astype(np.float32) - 255.0) > 9.0).any(axis=2)
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(fg, 255, 0).astype(np.uint8)
    return rgba


def compute_error_vmax(
    mesh_paths: list[Path],
    targets: list[np.ndarray],
    *,
    user_vmax: float | None = None,
) -> float:
    if user_vmax is not None and user_vmax > 0:
        return float(user_vmax)
    peak = 0.0
    for mesh_path, tgt in zip(mesh_paths, targets):
        verts, _ = load_obj_mesh(mesh_path)
        err = np.linalg.norm(verts - tgt, axis=-1)
        peak = max(peak, float(err.max()))
    return max(peak, 1e-6)


def postprocess_render_rgba(
    img: np.ndarray,
    *,
    trim_pad: int,
    face_crop: bool,
    face_width_ratio: float,
    face_height_ratio: float,
    face_center_y: float,
    face_pad_px: int,
) -> np.ndarray:
    img = crop_rgba_whitespace(img, pad=trim_pad)
    if face_crop:
        img = crop_rgba_to_face_region(
            img,
            width_ratio=face_width_ratio,
            height_ratio=face_height_ratio,
            center_y_frac=face_center_y,
            pad_px=face_pad_px,
        )
    return img


def _foreground_bbox(img: np.ndarray, *, alpha_thresh: int = 10) -> tuple[int, int, int, int]:
    if img.shape[2] == 4:
        fg = img[:, :, 3] > alpha_thresh
    else:
        fg = (np.abs(img.astype(np.float32) - 255.0) > 9.0).any(axis=2)
    ys, xs = np.where(fg)
    if ys.size == 0:
        h, w = img.shape[:2]
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_rgba_to_face_region(
    img: np.ndarray,
    *,
    width_ratio: float = 0.70,
    height_ratio: float = 0.58,
    center_y_frac: float = 0.34,
    pad_px: int = 6,
    alpha_thresh: int = 10,
) -> np.ndarray:
    """Crop a trimmed Blender render to the upper-center face region (image space)."""
    img = np.asarray(img)
    x0, y0, x1, y1 = _foreground_bbox(img, alpha_thresh=alpha_thresh)
    fg_w = max(x1 - x0, 1)
    fg_h = max(y1 - y0, 1)

    crop_w = fg_w * width_ratio
    crop_h = fg_h * height_ratio
    cx = x0 + fg_w * 0.5
    cy = y0 + fg_h * center_y_frac

    px0 = int(np.floor(cx - crop_w * 0.5 - pad_px))
    py0 = int(np.floor(cy - crop_h * 0.5 - pad_px))
    px1 = int(np.ceil(cx + crop_w * 0.5 + pad_px))
    py1 = int(np.ceil(cy + crop_h * 0.5 + pad_px))

    h, w = img.shape[:2]
    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(w, max(px1, px0 + 1))
    py1 = min(h, max(py1, py0 + 1))
    return img[py0:py1, px0:px1]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        import PIL

        font_path = os.path.join(os.path.dirname(PIL.__file__), "fonts", "DejaVuSans.ttf")
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def _vertical_caption_layer(
    caption: str,
    *,
    row_h: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    caption_gap: int,
) -> tuple[Image.Image, int]:
    """Render caption rotated 90° CCW, scaled to fit within the row height."""
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = measure.textbbox((0, 0), caption, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    ImageDraw.Draw(text_img).text(
        (-bbox[0], -bbox[1]),
        caption,
        fill=(0, 0, 0, 255),
        font=font,
    )
    rotated = text_img.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)

    max_h = max(row_h - 2 * caption_gap, 1)
    if rotated.height > max_h:
        scale = max_h / rotated.height
        rotated = rotated.resize(
            (max(1, int(round(rotated.width * scale))), max(1, int(round(rotated.height * scale)))),
            Image.Resampling.LANCZOS,
        )

    cap_w = rotated.width + 2 * caption_gap
    cap_layer = Image.new("RGBA", (cap_w, row_h), (0, 0, 0, 0))
    cap_layer.paste(
        rotated,
        ((cap_w - rotated.width) // 2, (row_h - rotated.height) // 2),
        rotated,
    )
    return cap_layer, cap_w


def render_mesh_with_error_colors(
    mesh_verts: np.ndarray,
    tar_verts: np.ndarray,
    faces: np.ndarray,
    raw_png: Path,
    *,
    render_px: int,
    trim_pad: int,
    face_crop: bool,
    face_width_ratio: float,
    face_height_ratio: float,
    face_center_y: float,
    face_pad_px: int,
    error_vmax: float,
    error_cmap: str,
    skip_render: bool = False,
) -> np.ndarray:
    """Render ``mesh_verts`` geometry with per-vertex error colors vs ``tar_verts``."""
    raw_png.parent.mkdir(parents=True, exist_ok=True)
    if not skip_render or not raw_png.is_file():
        vertex_colors = vertex_error_colors(
            mesh_verts,
            tar_verts,
            vmax=error_vmax,
            cmap=error_cmap,
        )
        img_chw = render_mesh(
            mesh_verts,
            faces,
            vertex_colors=vertex_colors,
            flat_shading=False,
            res=render_px,
            center_mesh=True,
        )
        rgba = _chw_to_rgba(img_chw)
        Image.fromarray(rgba, mode="RGBA").save(raw_png)
    else:
        rgba = np.array(Image.open(raw_png).convert("RGBA"))
    return postprocess_render_rgba(
        rgba,
        trim_pad=trim_pad,
        face_crop=face_crop,
        face_width_ratio=face_width_ratio,
        face_height_ratio=face_height_ratio,
        face_center_y=face_center_y,
        face_pad_px=face_pad_px,
    )


def render_tile_png(
    mesh_path: Path,
    tar_verts: np.ndarray,
    faces: np.ndarray,
    raw_png: Path,
    *,
    error_color: bool,
    render_px: int,
    blender_res: int,
    error_vmax: float | None,
    error_cmap: str,
    skip_cached_render: bool,
    **crop_kw,
) -> np.ndarray:
    """Render one collage tile from ``mesh_path`` (pred or dARAP OBJ on disk)."""
    if error_color:
        mesh_verts, _ = load_obj_mesh(mesh_path)
        if error_vmax is None:
            raise ValueError("error_vmax is required when error_color=True")
        return render_mesh_with_error_colors(
            mesh_verts,
            tar_verts,
            faces,
            raw_png,
            render_px=render_px,
            error_vmax=error_vmax,
            error_cmap=error_cmap,
            skip_render=skip_cached_render,
            **crop_kw,
        )
    return render_mesh_png(
        mesh_path,
        raw_png,
        res=blender_res,
        skip_blender=skip_cached_render,
        **crop_kw,
    )


def render_mesh_png(
    mesh_path: Path,
    raw_png: Path,
    *,
    res: int,
    trim_pad: int,
    face_crop: bool,
    face_width_ratio: float,
    face_height_ratio: float,
    face_center_y: float,
    face_pad_px: int,
    skip_blender: bool = False,
) -> np.ndarray:
    raw_png.parent.mkdir(parents=True, exist_ok=True)
    if not skip_blender or not raw_png.is_file():
        blender_render(mesh_path=str(mesh_path), out_file=str(raw_png), res=res)
    img = np.array(Image.open(raw_png).convert("RGBA"))
    return postprocess_render_rgba(
        img,
        trim_pad=trim_pad,
        face_crop=face_crop,
        face_width_ratio=face_width_ratio,
        face_height_ratio=face_height_ratio,
        face_center_y=face_center_y,
        face_pad_px=face_pad_px,
    )


def compose_2x5_figure(
    row0_images: list[np.ndarray],
    row1_images: list[np.ndarray],
    *,
    row0_caption: str = "PoissonNet",
    row1_caption: str = "PoissonNet+dARAP",
    gap: int = 8,
    caption_gap: int = 6,
    font_size: int = 18,
    bg: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> Image.Image:
    if len(row0_images) != 5 or len(row1_images) != 5:
        raise ValueError("expected 5 row-0 images and 5 row-1 images")

    aligned = align_rgba_images(row0_images + row1_images)
    row0_aligned = aligned[:5]
    row1_aligned = aligned[5:]
    tile_h, tile_w = row0_aligned[0].shape[:2]

    font = _load_font(font_size)
    cap_layer0, cap_w0 = _vertical_caption_layer(
        row0_caption, row_h=tile_h, font=font, caption_gap=caption_gap
    )
    cap_layer1, cap_w1 = _vertical_caption_layer(
        row1_caption, row_h=tile_h, font=font, caption_gap=caption_gap
    )
    cap_w = max(cap_w0, cap_w1)

    row_w = cap_w + gap + 5 * tile_w + 4 * gap
    row_h = tile_h
    canvas = Image.new("RGBA", (row_w, 2 * row_h + gap), bg)

    def paste_row(y: int, cap_layer: Image.Image, images: list[np.ndarray]) -> None:
        canvas.paste(cap_layer, (0, y), cap_layer)
        x = cap_w + gap
        for img in images:
            canvas.paste(Image.fromarray(img, mode="RGBA"), (x, y))
            x += tile_w + gap

    paste_row(0, cap_layer0, row0_aligned)
    paste_row(row_h + gap, cap_layer1, row1_aligned)
    return canvas


def compute_and_save_meshes(
    *,
    checkpoint: Path,
    config: dict,
    dataset: PHACKMeshDataset,
    frame_indices: list[int],
    targets: list[np.ndarray],
    pred_paths: list[Path],
    darap_paths: list[Path],
    device,
    center_outputs: bool,
) -> np.ndarray:
    import torch

    from meshes.operators import construct_mesh_operators
    from utils.helpers import to_np

    model = load_poisson_agc(checkpoint, config, device)
    verts_src, faces = dataset.get_source_cloth()
    if hasattr(model, "register_source_mesh"):
        model.register_source_mesh(verts_src.to(device), faces.to(device))

    verts_src_b = verts_src.to(device).unsqueeze(0)
    faces_b = faces.to(device).unsqueeze(0)
    mass, solver, G, M = construct_mesh_operators(verts_src_b, faces_b, high_precision=True)
    faces_np = to_np(faces_b[0])

    for frame_idx, tgt, pred_path, darap_path in tqdm(
        zip(frame_indices, targets, pred_paths, darap_paths),
        total=len(frame_indices),
        desc="predict + dARAP",
        dynamic_ncols=True,
    ):
        t_val = dataset.t_vals[frame_idx].to(device).view(1, 1)
        preds, _ = model(
            x_in=verts_src_b,
            M=M,
            G=G,
            solver=solver,
            faces=faces_b,
            vertex_mass=mass,
            extra_features=t_val,
        )
        if center_outputs:
            preds = preds - preds.mean(dim=1, keepdim=True)
        pred_v = to_np(preds[0])
        darap_v = deform_pred_to_target_normals(pred_v, faces_np, tgt)

        save_obj_mesh(pred_v, faces_np, pred_path)
        save_obj_mesh(darap_v, faces_np, darap_path)

    return faces_np


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render 5 key PoissonAGC / dARAP frames into a 2×5 figure.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--subject-id", type=str, default=SUBJECT_ID)
    parser.add_argument(
        "--frames",
        type=str,
        nargs="+",
        default=CHOSEN_FRAMES,
        help=(
            "phack_repair frame tokens or filenames, e.g. 018 or "
            "224_cam_222200037_018_repair.obj"
        ),
    )
    parser.add_argument(
        "--repair-dir",
        type=Path,
        default=DEFAULT_REPAIR_DIR,
        help="directory containing *_repair.obj meshes",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_MESH_DIR,
        help="directory with pred_*.obj / darap_*.obj; reused by default if complete",
    )
    parser.add_argument(
        "--recompute-meshes",
        action="store_true",
        help="re-run checkpoint inference and dARAP even if mesh files exist",
    )
    parser.add_argument(
        "--skip-blender",
        action="store_true",
        help="reuse existing raw PNGs in work_dir/raw when present",
    )
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--res", type=int, default=100, help="Blender render resolution percent")
    parser.add_argument("--trim-pad", type=int, default=4)
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument("--no-face-crop", action="store_true", help="disable face-region crop")
    parser.add_argument("--face-width-ratio", type=float, default=0.88)
    parser.add_argument("--face-height-ratio", type=float, default=0.68)
    parser.add_argument(
        "--face-center-y",
        type=float,
        default=0.34,
        help="vertical face center as fraction from top of foreground bbox",
    )
    parser.add_argument("--face-pad-px", type=int, default=3)
    parser.add_argument(
        "--error-color",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="color vertices by L2 error to target and render with pyrender",
    )
    parser.add_argument(
        "--error-vmax",
        type=float,
        default=None,
        help="shared error color scale max; auto from all meshes if omitted",
    )
    parser.add_argument("--error-cmap", type=str, default="YlOrRd")
    parser.add_argument(
        "--render-px",
        type=int,
        default=None,
        help="pyrender resolution when --error-color is set (default: 512 * res / 100)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    chosen_tokens = list(args.frames)
    if len(chosen_tokens) != 5:
        raise ValueError(f"expected 5 chosen frames, got {len(chosen_tokens)}")

    output_path = args.output.expanduser().resolve()
    work_dir = output_path.parent / "_viz_poissonagc_dARAP"
    mesh_dir = args.mesh_dir.expanduser().resolve()
    raw_dir = work_dir / "raw"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset subject {args.subject_id}...")
    dataset = PHACKMeshDataset(subject_id=args.subject_id, repair_dir=args.repair_dir)
    frame_entries = resolve_chosen_frames(dataset, chosen_tokens, repair_dir=args.repair_dir)
    frame_indices = [seq_idx for seq_idx, _, _ in frame_entries]
    frame_paths = [path for _, path, _ in frame_entries]
    repair_stems = [path.stem for path in frame_paths]
    pred_paths, darap_paths = mesh_job_paths(mesh_dir, repair_stems)

    print("Chosen repair meshes:")
    for token, path, seq_idx, (_, _, orig_id) in zip(
        chosen_tokens, frame_paths, frame_indices, frame_entries
    ):
        print(f"  {token!r} -> {path.name} (seq={seq_idx}, orig={orig_id:03d})")

    use_cached_meshes = meshes_are_ready(mesh_dir, repair_stems) and not args.recompute_meshes
    if use_cached_meshes:
        print(f"Using cached meshes from {mesh_dir}")
    else:
        import torch

        from utils.helpers import seed_everything

        seed_everything(31415)
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        config["detail_residual"] = {**config.get("detail_residual", {}), "enabled": False}

        targets = [dataset.verts[idx].numpy() for idx in frame_indices]
        print(f"Computing meshes for sequence indices {frame_indices}...")
        compute_and_save_meshes(
            checkpoint=checkpoint,
            config=config,
            dataset=dataset,
            frame_indices=frame_indices,
            targets=targets,
            pred_paths=pred_paths,
            darap_paths=darap_paths,
            device=device,
            center_outputs=not args.no_center,
        )

    _, faces_t = dataset.get_source_cloth()
    faces_np = faces_t.numpy()
    targets = [dataset.verts[idx].numpy() for idx in frame_indices]

    render_px = args.render_px or max(256, int(512 * args.res / 100))
    error_vmax: float | None = None
    if args.error_color:
        all_mesh_paths = pred_paths + darap_paths
        all_targets = targets + targets
        error_vmax = compute_error_vmax(
            all_mesh_paths,
            all_targets,
            user_vmax=args.error_vmax,
        )
        print(f"Error color scale vmax={error_vmax:.6f}, cmap={args.error_cmap!r}, render_px={render_px}")

    crop_kw = dict(
        trim_pad=args.trim_pad,
        face_crop=not args.no_face_crop,
        face_width_ratio=args.face_width_ratio,
        face_height_ratio=args.face_height_ratio,
        face_center_y=args.face_center_y,
        face_pad_px=args.face_pad_px,
    )

    # Row 0: checkpoint prediction meshes (pred_*.obj)
    # Row 1: dARAP-refined meshes (darap_*.obj)
    row_specs: tuple[tuple[str, list[Path], str], ...] = (
        ("PoissonNet", pred_paths, "pred"),
        ("PoissonNet+dARAP", darap_paths, "darap"),
    )
    row_images: list[list[np.ndarray]] = [[], []]
    for row_idx, (row_caption, mesh_paths, raw_subdir) in enumerate(row_specs):
        print(f"Row {row_idx + 1} ({row_caption}):")
        for mesh_path, tgt_v in zip(mesh_paths, targets):
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Missing {raw_subdir} mesh: {mesh_path}")
            raw_png = raw_dir / raw_subdir / f"{mesh_path.stem}.png"
            print(f"  {mesh_path.name}")
            row_images[row_idx].append(
                render_tile_png(
                    mesh_path,
                    tgt_v,
                    faces_np,
                    raw_png,
                    error_color=args.error_color,
                    render_px=render_px,
                    blender_res=args.res,
                    error_vmax=error_vmax,
                    error_cmap=args.error_cmap,
                    skip_cached_render=args.skip_blender,
                    **crop_kw,
                )
            )

    figure = compose_2x5_figure(
        row_images[0],
        row_images[1],
        font_size=args.font_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output_path)
    print(f"Saved 2x5 figure → {output_path}")


if __name__ == "__main__":
    main()
