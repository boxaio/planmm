"""可视化全局生成网格的法线光滑性质量。

三行对比：
  1) ablations/lamm_hack_ae_mlp/generated_global
  2) ablations/lamm_hack_ae_transformer/generated_global
  3) results/planmm_hack_ae/<最新时间戳>/generated_global

每行 4 个网格；每个网格输出「渲染 | 法线光滑热图」紧挨无留白，
不同网格之间留少量白边。最终写入 ``figures/mesh_quality_global.png``。
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from render.mesh_render import render_mesh
from utils.mesh import vertex_normals

# ----- paths -----
MLP_GLOBAL_DIR = _REPO_ROOT / "ablations" / "lamm_hack_ae_mlp" / "generated_global"
TRANSFORMER_GLOBAL_DIR = (
    _REPO_ROOT / "ablations" / "lamm_hack_ae_transformer" / "generated_global"
)
PLANMM_AE_ROOT = _REPO_ROOT / "results" / "planmm_hack_ae"
REGION_INFO_FILE = _REPO_ROOT / "dataset" / "hack_region_info.pkl"
DEFAULT_OUTPUT = _REPO_ROOT / "figures" / "mesh_quality_global.png"

# 每行展示的样本索引（LAMM 用 ``{i}.obj``，PLANMM 用 ``{i:04d}.obj``）
MESH_INDICES: tuple[int, ...] = (1, 11, 15, 22)

ROWS: tuple[tuple[str, str], ...] = (
    ("LAMM-MLPMixer", "mlp"),
    ("LAMM-Transformer", "transformer"),
    ("PLANMM-MLPMixer", "planmm"),
)

# ----- render / mosaic -----
RENDER_RES = 640
RENDER_FILL_RATIO = 0.97
RENDER_YFOV = np.pi / 3.0

MOSAIC_TRIM_BG = 1.0
MOSAIC_TRIM_TOL = 0.035
MOSAIC_TRIM_PAD = 4
FACE_CROP_WIDTH_RATIO = 0.78
FACE_CROP_HEIGHT_RATIO = 0.72
FACE_CROP_CENTER_X = 0.5
FACE_CROP_CENTER_Y = 0.36

PAIR_GAP_PX = 18  # 不同网格之间的留白
ROW_GAP_PX = 14
ROW_LABEL_FONT_SIZE = 28
ROW_LABEL_GAP = 8
ROW_LABEL_BG = (255, 255, 255, 255)
ROW_LABEL_FG = (0, 0, 0, 255)

# HEATMAP_CMAP = "turbo"
# HEATMAP_CMAP = "viridis"
HEATMAP_CMAP = "RdYlBu_r"
HEATMAP_PERCENTILE = 98.0  # 全局归一化上界（跨方法可比）
HEATMAP_GAMMA = 0.55

_FACE_REGION_NAMES = (
    "LeftFace",
    "RightFace",
    "Ears",
    "Chin",
    "Forehead",
    "Eyes",
    "Nose",
    "Mouth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 3x4 mesh-quality mosaic (shaded + normal-smoothness heatmap).",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(MESH_INDICES),
        help="Mesh indices to show (default: 0 1 2 3).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--res", type=int, default=RENDER_RES)
    parser.add_argument(
        "--planmm-run",
        type=Path,
        default=None,
        help="Optional planmm run dir (default: latest timestamp with generated_global).",
    )
    parser.add_argument(
        "--region-info",
        type=Path,
        default=REGION_INFO_FILE,
        help="hack_region_info.pkl for face-centered framing.",
    )
    return parser.parse_args()


def latest_planmm_generated_global(root: Path = PLANMM_AE_ROOT) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"planmm ae root missing: {root}")
    candidates = [
        p / "generated_global"
        for p in root.iterdir()
        if p.is_dir() and (p / "generated_global").is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"no generated_global under {root}")
    return max(candidates, key=lambda p: p.parent.name)


def resolve_mesh_path(folder: Path, index: int, *, naming: str) -> Path:
    if naming == "planmm":
        path = folder / f"{index:04d}.obj"
    else:
        path = folder / f"{index}.obj"
    if not path.is_file():
        raise FileNotFoundError(f"missing mesh: {path}")
    return path


def load_obj_mesh(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(obj_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return verts, faces


def load_face_vertex_ids(region_info_file: Path) -> np.ndarray:
    with open(region_info_file, "rb") as f:
        region_info = pickle.load(f)
    vids: list[int] = []
    for name in _FACE_REGION_NAMES:
        if name in region_info:
            vids.extend(
                np.asarray(region_info[name]["vids"], dtype=np.int64).reshape(-1).tolist()
            )
    if not vids:
        raise RuntimeError(f"no face region vids found in {region_info_file}")
    return np.unique(np.asarray(vids, dtype=np.int64))


def build_vertex_adjacency(faces: np.ndarray, n_verts: int) -> list[np.ndarray]:
    nbrs: list[set[int]] = [set() for _ in range(n_verts)]
    for a, b, c in faces.astype(np.int64):
        nbrs[a].update((int(b), int(c)))
        nbrs[b].update((int(a), int(c)))
        nbrs[c].update((int(a), int(b)))
    return [np.fromiter(s, dtype=np.int64) for s in nbrs]


def normal_smoothness_score(
    verts: np.ndarray,
    faces: np.ndarray,
    adjacency: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Per-vertex normal discontinuity: mean over neighbors of ``1 - n_i·n_j``.

    Larger = less smooth = worse local mesh quality.
    """
    v = torch.from_numpy(np.asarray(verts, dtype=np.float32)).unsqueeze(0)
    f = torch.from_numpy(np.asarray(faces, dtype=np.int64)).unsqueeze(0)
    normals = vertex_normals(v, f)[0].numpy().astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)

    if adjacency is None:
        adjacency = build_vertex_adjacency(faces, len(verts))

    scores = np.zeros(len(verts), dtype=np.float64)
    for i, nbr in enumerate(adjacency):
        if nbr.size == 0:
            continue
        dots = np.clip((normals[i] * normals[nbr]).sum(axis=1), -1.0, 1.0)
        scores[i] = float(np.mean(1.0 - dots))
    return scores


def scores_to_vertex_colors(
    scores: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str = HEATMAP_CMAP,
    gamma: float = HEATMAP_GAMMA,
) -> np.ndarray:
    denom = max(float(vmax) - float(vmin), 1e-12)
    linear = np.clip((scores - float(vmin)) / denom, 0.0, 1.0)
    norm = np.power(linear, max(float(gamma), 1e-6))
    rgb = plt.get_cmap(cmap)(norm)[:, :3].astype(np.float32)
    return rgb


def fg_bbox(
    img: torch.Tensor,
    bg: float = MOSAIC_TRIM_BG,
    tol: float = MOSAIC_TRIM_TOL,
    pad: int = MOSAIC_TRIM_PAD,
) -> tuple[int, int, int, int]:
    """Return inclusive-exclusive crop box ``(y0, y1, x0, x1)`` from shaded render."""
    _, h, w = img.shape
    fg = (torch.abs(img - bg) > tol).any(dim=0)
    ys, xs = torch.where(fg)
    if ys.numel() == 0:
        return 0, h, 0, w
    y0 = max(0, int(ys.min().item()) - pad)
    y1 = min(h, int(ys.max().item()) + 1 + pad)
    x0 = max(0, int(xs.min().item()) - pad)
    x1 = min(w, int(xs.max().item()) + 1 + pad)
    return y0, y1, x0, x1


def face_crop_box_from_trim(
    trim_h: int,
    trim_w: int,
    width_ratio: float = FACE_CROP_WIDTH_RATIO,
    height_ratio: float = FACE_CROP_HEIGHT_RATIO,
    center_x_frac: float = FACE_CROP_CENTER_X,
    center_y_frac: float = FACE_CROP_CENTER_Y,
    pad: int = MOSAIC_TRIM_PAD,
) -> tuple[int, int, int, int]:
    crop_w = trim_w * width_ratio
    crop_h = trim_h * height_ratio
    cx = trim_w * center_x_frac
    cy = trim_h * center_y_frac
    px0 = int(np.floor(cx - crop_w * 0.5 - pad))
    py0 = int(np.floor(cy - crop_h * 0.5 - pad))
    px1 = int(np.ceil(cx + crop_w * 0.5 + pad))
    py1 = int(np.ceil(cy + crop_h * 0.5 + pad))
    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(trim_w, max(px1, px0 + 1))
    py1 = min(trim_h, max(py1, py0 + 1))
    return py0, py1, px0, px1


def render_shaded_and_heatmap(
    verts: np.ndarray,
    faces: np.ndarray,
    face_vids: np.ndarray,
    vertex_colors: np.ndarray,
    res: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render shaded + heatmap with identical camera / crop so they align."""
    face_ctr = verts[face_vids].mean(axis=0)
    verts_face = verts - face_ctr

    common_kw = dict(
        res=res,
        filename=None,
        flat_shading=False,
        center_mesh=False,
        fill_ratio=RENDER_FILL_RATIO,
        camera_distance=None,
        yfov=RENDER_YFOV,
        reuse_cached_renderer=True,
    )
    shaded = render_mesh(verts_face, faces, **common_kw)
    heat = render_mesh(
        verts_face,
        faces,
        vertex_colors=vertex_colors,
        **common_kw,
    )

    # 用 shaded 的前景框同步裁剪两张图，保证左右对齐且无缝拼接。
    y0, y1, x0, x1 = fg_bbox(shaded)
    shaded_t = shaded[:, y0:y1, x0:x1]
    heat_t = heat[:, y0:y1, x0:x1]
    fy0, fy1, fx0, fx1 = face_crop_box_from_trim(
        trim_h=shaded_t.shape[1],
        trim_w=shaded_t.shape[2],
    )
    return shaded_t[:, fy0:fy1, fx0:fx1], heat_t[:, fy0:fy1, fx0:fx1]


def pad_chw_to(
    img: torch.Tensor,
    out_h: int,
    out_w: int,
    bg: float = MOSAIC_TRIM_BG,
) -> torch.Tensor:
    _, h, w = img.shape
    if h > out_h or w > out_w:
        raise ValueError(f"cannot pad {h}x{w} into smaller {out_h}x{out_w}")
    pad_h = out_h - h
    pad_w = out_w - w
    return F.pad(
        img,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        value=bg,
    )


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        import PIL

        font_path = os.path.join(os.path.dirname(PIL.__file__), "fonts", "DejaVuSans.ttf")
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def _pil_from_chw(img: torch.Tensor) -> Image.Image:
    arr = (
        img.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _chw_from_pil(img: Image.Image) -> torch.Tensor:
    return (
        torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
        .permute(2, 0, 1)
        .contiguous()
    )


def _vertical_caption_layer(
    caption: str,
    row_h: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    caption_gap: int = ROW_LABEL_GAP,
    fg: tuple[int, int, int, int] = ROW_LABEL_FG,
) -> tuple[Image.Image, int]:
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = measure.textbbox((0, 0), caption, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_img = Image.new("RGBA", (max(text_w, 1), max(text_h, 1)), (0, 0, 0, 0))
    ImageDraw.Draw(text_img).text((-bbox[0], -bbox[1]), caption, fill=fg, font=font)
    rotated = text_img.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)

    max_h = max(row_h - 2 * caption_gap, 1)
    if rotated.height > max_h:
        scale = max_h / rotated.height
        rotated = rotated.resize(
            (
                max(1, int(round(rotated.width * scale))),
                max(1, int(round(rotated.height * scale))),
            ),
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


def prepend_row_labels(
    rows: list[torch.Tensor],
    labels: list[str],
) -> list[torch.Tensor]:
    font = _load_font(ROW_LABEL_FONT_SIZE)
    layers: list[Image.Image] = []
    widths: list[int] = []
    for row, label in zip(rows, labels):
        layer, cap_w = _vertical_caption_layer(
            label, row_h=int(row.shape[1]), font=font
        )
        layers.append(layer)
        widths.append(cap_w)
    label_w = max(widths)
    out: list[torch.Tensor] = []
    for row, layer in zip(rows, layers):
        canvas = Image.new("RGBA", (label_w, int(row.shape[1])), ROW_LABEL_BG)
        canvas.paste(layer, ((label_w - layer.width) // 2, 0), layer)
        out.append(torch.cat([_chw_from_pil(canvas), row], dim=2))
    return out


def assemble_mosaic(
    grid: list[list[tuple[torch.Tensor, torch.Tensor]]],
    *,
    row_labels: list[str],
    pair_gap: int = PAIR_GAP_PX,
    row_gap: int = ROW_GAP_PX,
) -> torch.Tensor:
    """``grid[r][c] = (shaded, heatmap)`` → labeled mosaic."""
    n_rows = len(grid)
    n_cols = len(grid[0])
    pairs: list[list[torch.Tensor]] = []
    for r in range(n_rows):
        row_pairs: list[torch.Tensor] = []
        for c in range(n_cols):
            shaded, heat = grid[r][c]
            h = max(shaded.shape[1], heat.shape[1])
            shaded_p = pad_chw_to(shaded, out_h=h, out_w=shaded.shape[2])
            heat_p = pad_chw_to(heat, out_h=h, out_w=heat.shape[2])
            row_pairs.append(torch.cat([shaded_p, heat_p], dim=2))
        pairs.append(row_pairs)

    col_ws = [
        max(int(pairs[r][c].shape[2]) for r in range(n_rows)) for c in range(n_cols)
    ]
    row_hs = [
        max(int(pairs[r][c].shape[1]) for c in range(n_cols)) for r in range(n_rows)
    ]

    row_tensors: list[torch.Tensor] = []
    for r in range(n_rows):
        cells: list[torch.Tensor] = []
        for c in range(n_cols):
            cells.append(pad_chw_to(pairs[r][c], out_h=row_hs[r], out_w=col_ws[c]))
            if c < n_cols - 1 and pair_gap > 0:
                cells.append(
                    torch.ones(3, row_hs[r], pair_gap, dtype=torch.float32) * MOSAIC_TRIM_BG
                )
        row_tensors.append(torch.cat(cells, dim=2))

    row_tensors = prepend_row_labels(row_tensors, row_labels)

    max_w = max(int(r.shape[2]) for r in row_tensors)
    aligned: list[torch.Tensor] = []
    for i, row in enumerate(row_tensors):
        aligned.append(pad_chw_to(row, out_h=int(row.shape[1]), out_w=max_w))
        if i < n_rows - 1 and row_gap > 0:
            aligned.append(
                torch.ones(3, row_gap, max_w, dtype=torch.float32) * MOSAIC_TRIM_BG
            )
    return torch.cat(aligned, dim=1)


def main() -> None:
    args = parse_args()
    indices = tuple(int(i) for i in args.indices)
    if len(indices) != 4:
        raise ValueError(f"expected exactly 4 indices, got {indices}")

    if args.planmm_run is not None:
        planmm_dir = Path(args.planmm_run)
        if planmm_dir.name != "generated_global":
            planmm_dir = planmm_dir / "generated_global"
    else:
        planmm_dir = latest_planmm_generated_global()

    folders = {
        "mlp": MLP_GLOBAL_DIR,
        "transformer": TRANSFORMER_GLOBAL_DIR,
        "planmm": planmm_dir,
    }
    for key, folder in folders.items():
        if not folder.is_dir():
            raise FileNotFoundError(f"{key} generated_global missing: {folder}")

    face_vids = load_face_vertex_ids(Path(args.region_info))
    print(f"[viz_mesh_quality_global] planmm dir = {planmm_dir}")
    print(f"[viz_mesh_quality_global] indices = {indices}")

    # 第一遍：加载网格并计算光滑分数，确定全局色标
    loaded: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        key: [] for _, key in ROWS
    }
    all_scores: list[np.ndarray] = []
    adjacency_cache: dict[int, list[np.ndarray]] = {}

    for label, key in ROWS:
        naming = "planmm" if key == "planmm" else "lamm"
        for idx in indices:
            obj_path = resolve_mesh_path(folders[key], idx, naming=naming)
            verts, faces = load_obj_mesh(obj_path)
            n_v = len(verts)
            if n_v not in adjacency_cache:
                adjacency_cache[n_v] = build_vertex_adjacency(faces, n_v)
            scores = normal_smoothness_score(
                verts, faces, adjacency=adjacency_cache[n_v]
            )
            loaded[key].append((verts, faces, scores))
            all_scores.append(scores)
            print(
                f"  [{label}] {obj_path.name}: "
                f"mean={scores.mean():.4f}  p98={np.percentile(scores, 98):.4f}"
            )

    stacked = np.concatenate(all_scores)
    vmin = 0.0
    vmax = float(np.percentile(stacked, HEATMAP_PERCENTILE))
    vmax = max(vmax, 1e-6)
    print(f"[viz_mesh_quality_global] heatmap scale: vmin={vmin:.4f} vmax={vmax:.4f}")

    # 第二遍：渲染
    grid: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    for label, key in ROWS:
        row_cells: list[tuple[torch.Tensor, torch.Tensor]] = []
        for verts, faces, scores in loaded[key]:
            colors = scores_to_vertex_colors(scores, vmin=vmin, vmax=vmax)
            shaded, heat = render_shaded_and_heatmap(
                verts,
                faces,
                face_vids,
                colors,
                res=int(args.res),
            )
            row_cells.append((shaded, heat))
        grid.append(row_cells)
        print(f"[viz_mesh_quality_global] rendered row {label}")

    mosaic = assemble_mosaic(
        grid,
        row_labels=[label for label, _ in ROWS],
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(mosaic, str(out_path))
    print(
        f"[viz_mesh_quality_global] wrote {out_path} "
        f"({mosaic.shape[2]}x{mosaic.shape[1]} px)"
    )


if __name__ == "__main__":
    main()
