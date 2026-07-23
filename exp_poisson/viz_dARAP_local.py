"""多区域局部 dARAP 形变展示图。

对 eyes / nose / mouth / chin 各取一对 source/target，用两个 ``lambda_deform``
分别形变；每行四格：source | target | λ1 | λ2。四行拼成
``figures/dARAP_local_show.png``。

形变逻辑对齐 ``demo_deform_dARAP.py`` / ``demo_switch_local.py``。
各区域的索引、λ、相机角在下方 ``REGION_SHOW_JOBS`` 中配置。
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NICP_DIR = _REPO_ROOT / "meshes" / "nonrigid_icp"
for _path in (str(_REPO_ROOT), str(_NICP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from configs.config_utils import read_yaml
from dataset.hack_dataset import HACKDataset, VAL_HACK_PT
from dataset.hack_seg_fids import REGION_NAMES
from meshes.nonrigid_icp.deformations_MINIMAL import (
    ProcrustesPrecompute,
    SparseLaplaciansSolvers,
    calc_ARAP_global_solve,
    calc_rot_matrices_with_procrustes,
    vertex_procrustes_lambda_from_regions,
)
from render.mesh_render import render_mesh
from utils.mesh import vertex_normals

# ----- paths -----
CONFIG_FILE = _REPO_ROOT / "exp_poisson" / "planmm_hack_manipulation.yaml"
REGION_INFO_FILE = _REPO_ROOT / "dataset" / "hack_region_info.pkl"
PACKED_PT = VAL_HACK_PT
DEFAULT_OUTPUT = _REPO_ROOT / "figures" / "dARAP_local_show.png"

# ----- shared deform / render defaults -----
LAMBDA_DEFAULT = 3.0

RENDER_RES = 960
RENDER_FILL_RATIO = 0.97
RENDER_YFOV = np.pi / 3.0
RENDER_CAMERA_DISTANCE = None  # None => auto from fill_ratio

MOSAIC_TRIM_BG = 1.0
MOSAIC_TRIM_TOL = 0.035
MOSAIC_TRIM_PAD = 4
FACE_CROP = True
FACE_CROP_WIDTH_RATIO = 0.9
FACE_CROP_HEIGHT_RATIO = 0.7
FACE_CROP_CENTER_X = 0.5
FACE_CROP_CENTER_Y = 0.34

ROW_LABEL_FONT_SIZE = 35
ROW_LABEL_GAP = 10
ROW_LABEL_BG = (255, 255, 255, 255)
ROW_LABEL_FG = (0, 0, 0, 255)

COL_LABEL_FONT_SIZE = 16
COL_LABEL_PAD_Y = 8
COL_LABEL_BG = (255, 255, 255)

_FACE_REGION_NAMES = (
    "LeftFace", "RightFace", "Ears", "Chin", "Forehead", "Eyes", "Nose", "Mouth",
)
_REGION_NAME_TO_KEY = {name.lower(): key for key, name in REGION_NAMES.items()}


@dataclass(frozen=True)
class RegionShowJob:
    """One row in the 4x4 show figure."""

    region: str
    src_idx: int
    tgt_idx: int
    lambda_deform: tuple[float, float]
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    # 图像空间裁剪（相对前景 bbox）：宽/高比例 + 水平/竖直中心位置
    face_crop: bool = True
    crop_width: float = FACE_CROP_WIDTH_RATIO
    crop_height: float = FACE_CROP_HEIGHT_RATIO
    crop_center_x: float = FACE_CROP_CENTER_X
    crop_center_y: float = FACE_CROP_CENTER_Y


# 行顺序固定：eyes / nose / mouth / chin。按需改各行的索引、λ、相机角、裁剪。
REGION_SHOW_JOBS: tuple[RegionShowJob, ...] = (
    RegionShowJob(
        region="eyes",
        src_idx=0,
        tgt_idx=6,
        lambda_deform=(6.5, 9.5),
        yaw=10.0,
        pitch=0.0,
        roll=0.0,
        crop_width=0.9,
        crop_height=0.7,
        crop_center_x=0.5,
        crop_center_y=0.30,
    ),
    RegionShowJob(
        region="nose",
        src_idx=0,
        tgt_idx=9,
        lambda_deform=(6.5, 9.5),
        yaw=25.0,
        pitch=10.0,
        roll=0.0,
        crop_width=0.9,
        crop_height=0.7,
        crop_center_x=0.58,
        crop_center_y=0.4,
    ),
    RegionShowJob(
        region="mouth",
        src_idx=0,
        tgt_idx=1,
        lambda_deform=(6.5, 9.5),
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        crop_width=0.9,
        crop_height=0.7,
        crop_center_x=0.5,
        crop_center_y=0.42,
    ),
    RegionShowJob(
        region="chin",
        src_idx=0,
        tgt_idx=3,
        lambda_deform=(6.5, 9.5),
        yaw=28.0,
        pitch=5.0,
        roll=0.0,
        crop_width=0.9,
        crop_height=0.65,
        crop_center_x=0.55,
        crop_center_y=0.45,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render 4x4 local-dARAP show: rows=eyes/nose/mouth/chin, "
            "cols=source/target/λ1/λ2 → figures/dARAP_local_show.png"
        ),
    )
    parser.add_argument(
        "--packed-pt",
        type=Path,
        default=PACKED_PT,
        help="Packed HACK .pt (default: val_hack.pt).",
    )
    parser.add_argument(
        "--region-info",
        type=Path,
        default=None,
        help="hack_region_info.pkl (default: MODEL.region_info_file or dataset/hack_region_info.pkl).",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=CONFIG_FILE,
        help="YAML used to resolve region_info_file when --region-info is omitted.",
    )
    parser.add_argument("--lambda-default", type=float, default=LAMBDA_DEFAULT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path (default: figures/dARAP_local_show.png).",
    )
    render = parser.add_argument_group("shared render options")
    render.add_argument("--res", type=int, default=RENDER_RES)
    render.add_argument("--fill-ratio", type=float, default=RENDER_FILL_RATIO)
    render.add_argument(
        "--camera-distance",
        type=float,
        default=RENDER_CAMERA_DISTANCE,
        help="Fixed camera distance; default auto from --fill-ratio.",
    )
    render.add_argument(
        "--yfov",
        type=float,
        default=None,
        help="Vertical FOV in degrees (default: ~60°).",
    )
    render.add_argument(
        "--no-face-crop",
        action="store_true",
        help="Disable image-space face crop after rendering.",
    )
    render.add_argument("--face-crop-width", type=float, default=FACE_CROP_WIDTH_RATIO)
    render.add_argument("--face-crop-height", type=float, default=FACE_CROP_HEIGHT_RATIO)
    render.add_argument("--face-crop-center-x", type=float, default=FACE_CROP_CENTER_X)
    render.add_argument("--face-crop-center-y", type=float, default=FACE_CROP_CENTER_Y)
    return parser.parse_args()


def rotation_matrix_from_view_deg(
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray | None:
    """Build 3x3 rotation for ``render_mesh(..., rot_mat=...)``, or None if identity."""
    if abs(yaw_deg) < 1e-8 and abs(pitch_deg) < 1e-8 and abs(roll_deg) < 1e-8:
        return None
    ax = np.deg2rad(float(pitch_deg))
    ay = np.deg2rad(float(yaw_deg))
    az = np.deg2rad(float(roll_deg))
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def resolve_region(spec: str) -> tuple[int, str]:
    """Parse region patch id (0-9) or name. Returns (key, canonical Name e.g. Mouth)."""
    spec = spec.strip()
    if spec.isdigit():
        key = int(spec)
        if key not in REGION_NAMES:
            raise ValueError(f"invalid region id {key}, expected 0-9")
        return key, REGION_NAMES[key]

    name_l = spec.lower()
    if name_l in _REGION_NAME_TO_KEY:
        key = _REGION_NAME_TO_KEY[name_l]
        return key, REGION_NAMES[key]

    raise ValueError(
        f"unknown region {spec!r}; use patch id 0-9 or name: "
        f"{', '.join(sorted(_REGION_NAME_TO_KEY.keys()))}"
    )


def resolve_region_info_file(args: argparse.Namespace) -> Path:
    if args.region_info is not None:
        path = Path(args.region_info)
    else:
        cfg_path = Path(args.config_file)
        if cfg_path.is_file():
            cfg = read_yaml(cfg_path)
            path = Path(cfg["MODEL"]["region_info_file"])
        else:
            path = REGION_INFO_FILE
    if not path.is_file():
        raise FileNotFoundError(f"region_info not found: {path}")
    return path


def load_region_vids(region_info_file: Path, region_name: str) -> np.ndarray:
    with open(region_info_file, "rb") as f:
        region_info = pickle.load(f)
    if region_name not in region_info:
        available = sorted(k for k in region_info if "inner" not in k)
        raise KeyError(
            f"region {region_name!r} missing in {region_info_file}; "
            f"available={available}"
        )
    return np.asarray(region_info[region_name]["vids"], dtype=np.int64).reshape(-1)


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


def deform_source_to_target_normals(
    src_verts: np.ndarray,
    faces: np.ndarray,
    tgt_normals: np.ndarray,
    region_vertex_indices: list[list[int]],
    *,
    default_lambda: float = LAMBDA_DEFAULT,
    region_lambda: float,
) -> np.ndarray:
    """dARAP local step + global solve (same pipeline as demo_deform_dARAP)."""
    verts_list = [torch.tensor(src_verts, dtype=torch.float32)]
    faces_list = [torch.tensor(faces, dtype=torch.long)]
    target_normals_list = [torch.tensor(tgt_normals, dtype=torch.float32)]

    local_step_procrustes_lambda = vertex_procrustes_lambda_from_regions(
        len(src_verts),
        default_lambda=default_lambda,
        region_vertex_indices=region_vertex_indices,
        region_lambdas=[region_lambda] * len(region_vertex_indices),
    )

    solvers = SparseLaplaciansSolvers.from_meshes(
        verts_list,
        faces_list,
        pin_first_vertex=True,
        compute_poisson_rhs_lefts=False,
        compute_igl_arap_rhs_lefts=None,
    )
    procrustes_precompute = ProcrustesPrecompute.from_meshes(
        local_step_procrustes_lambda=local_step_procrustes_lambda,
        arap_energy_type="spokes_and_rims_mine",
        laplacians_solvers=solvers,
        verts_list=verts_list,
        faces_list=faces_list,
    )
    per_vertex_3x3matrices_packed = calc_rot_matrices_with_procrustes(
        procrustes_precompute,
        torch.cat(verts_list),
        torch.cat(target_normals_list),
    )
    per_vertex_3x3matrices_list = torch.split(
        per_vertex_3x3matrices_packed,
        [len(v) for v in verts_list],
    )
    deformed_verts_list = calc_ARAP_global_solve(
        verts_list,
        faces_list,
        solvers,
        per_vertex_3x3matrices_list,
        arap_energy_type="spokes_and_rims_mine",
        postprocess="recenter_only",
    )
    return deformed_verts_list[0].detach().cpu().numpy()


def crop_chw_near_uniform_bg(
    img: torch.Tensor,
    *,
    bg: float = MOSAIC_TRIM_BG,
    tol: float = MOSAIC_TRIM_TOL,
    pad: int = MOSAIC_TRIM_PAD,
) -> torch.Tensor:
    if img.ndim != 3:
        raise ValueError(f"expected CHW image, got shape {tuple(img.shape)}")
    _, h, w = img.shape
    fg = (torch.abs(img - bg) > tol).any(dim=0)
    ys, xs = torch.where(fg)
    if ys.numel() == 0:
        return img
    y0 = max(0, int(ys.min().item()) - pad)
    y1 = min(h, int(ys.max().item()) + 1 + pad)
    x0 = max(0, int(xs.min().item()) - pad)
    x1 = min(w, int(xs.max().item()) + 1 + pad)
    return img[:, y0:y1, x0:x1]


def crop_chw_to_face_region(
    img: torch.Tensor,
    *,
    width_ratio: float = FACE_CROP_WIDTH_RATIO,
    height_ratio: float = FACE_CROP_HEIGHT_RATIO,
    center_x_frac: float = FACE_CROP_CENTER_X,
    center_y_frac: float = FACE_CROP_CENTER_Y,
    pad: int = MOSAIC_TRIM_PAD,
    bg: float = MOSAIC_TRIM_BG,
    tol: float = MOSAIC_TRIM_TOL,
) -> torch.Tensor:
    if img.ndim != 3:
        raise ValueError(f"expected CHW image, got shape {tuple(img.shape)}")
    _, h, w = img.shape
    fg = (torch.abs(img - bg) > tol).any(dim=0)
    ys, xs = torch.where(fg)
    if ys.numel() == 0:
        return img

    x0, y0 = int(xs.min().item()), int(ys.min().item())
    x1, y1 = int(xs.max().item()) + 1, int(ys.max().item()) + 1
    fg_w, fg_h = x1 - x0, y1 - y0
    crop_w = fg_w * width_ratio
    crop_h = fg_h * height_ratio
    cx = x0 + fg_w * center_x_frac
    cy = y0 + fg_h * center_y_frac

    px0 = int(np.floor(cx - crop_w * 0.5 - pad))
    py0 = int(np.floor(cy - crop_h * 0.5 - pad))
    px1 = int(np.ceil(cx + crop_w * 0.5 + pad))
    py1 = int(np.ceil(cy + crop_h * 0.5 + pad))

    px0 = max(0, px0)
    py0 = max(0, py0)
    px1 = min(w, max(px1, px0 + 1))
    py1 = min(h, max(py1, py0 + 1))
    return img[:, py0:py1, px0:px1]


def align_stripe_heights_centervpad(
    stripes: list[torch.Tensor],
    *,
    bg: float = MOSAIC_TRIM_BG,
) -> list[torch.Tensor]:
    if not stripes:
        return stripes
    max_h = max(s.shape[1] for s in stripes)
    out: list[torch.Tensor] = []
    for t in stripes:
        _, h, _ = t.shape
        if h >= max_h:
            out.append(t)
            continue
        pad_total = max_h - h
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        out.append(F.pad(t, (0, 0, pad_top, pad_bottom), value=bg))
    return out


def align_row_widths_centerhpad(
    rows: list[torch.Tensor],
    *,
    bg: float = MOSAIC_TRIM_BG,
) -> list[torch.Tensor]:
    if not rows:
        return rows
    max_w = max(r.shape[2] for r in rows)
    out: list[torch.Tensor] = []
    for t in rows:
        _, _, w = t.shape
        if w >= max_w:
            out.append(t)
            continue
        pad_total = max_w - w
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        out.append(F.pad(t, (pad_left, pad_right, 0, 0), value=bg))
    return out


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
    caption_gap: int = ROW_LABEL_GAP,
    fg: tuple[int, int, int, int] = ROW_LABEL_FG,
) -> tuple[Image.Image, int]:
    """Render caption rotated 90° CCW, scaled to fit within the row height."""
    measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = measure.textbbox((0, 0), caption, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_img = Image.new("RGBA", (max(text_w, 1), max(text_h, 1)), (0, 0, 0, 0))
    ImageDraw.Draw(text_img).text(
        (-bbox[0], -bbox[1]),
        caption,
        fill=fg,
        font=font,
    )
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


def pad_chw_to(
    img: torch.Tensor,
    *,
    out_h: int,
    out_w: int,
    bg: float = MOSAIC_TRIM_BG,
) -> torch.Tensor:
    """Center-pad a CHW image to ``(out_h, out_w)``."""
    if img.ndim != 3:
        raise ValueError(f"expected CHW image, got shape {tuple(img.shape)}")
    _, h, w = img.shape
    if h > out_h or w > out_w:
        raise ValueError(f"cannot pad {h}x{w} into smaller {out_h}x{out_w}")
    pad_h = out_h - h
    pad_w = out_w - w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=bg)


def assemble_panel_grid(
    grid: list[list[torch.Tensor]],
    *,
    bg: float = MOSAIC_TRIM_BG,
) -> tuple[list[torch.Tensor], list[int]]:
    """Pad cells to a regular grid; return aligned rows and per-column widths."""
    if not grid:
        return [], []
    n_cols = len(grid[0])
    if any(len(row) != n_cols for row in grid):
        raise ValueError("all rows must have the same number of panels")

    col_ws = [
        max(int(grid[r][c].shape[2]) for r in range(len(grid)))
        for c in range(n_cols)
    ]
    row_hs = [
        max(int(grid[r][c].shape[1]) for c in range(n_cols))
        for r in range(len(grid))
    ]

    rows: list[torch.Tensor] = []
    for r, row in enumerate(grid):
        cells = [
            pad_chw_to(cell, out_h=row_hs[r], out_w=col_ws[c], bg=bg)
            for c, cell in enumerate(row)
        ]
        rows.append(torch.cat(cells, dim=2))
    return rows, col_ws


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


def render_text_label_rgba(
    text: str,
    *,
    fontsize: int = COL_LABEL_FONT_SIZE,
    math: bool = False,
    color: str = "black",
    dpi: int = 200,
) -> Image.Image:
    """Render plain or mathtext label via matplotlib; returns cropped RGBA PIL image."""
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = f"${text}$" if math else text
    fig = plt.figure(figsize=(4.0, 1.0), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.5,
        display,
        fontsize=fontsize,
        color=color,
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)
    buf.seek(0)
    rgba = Image.open(buf).convert("RGBA")

    # Trim near-white margins while keeping a small pad.
    arr = np.asarray(rgba)
    rgb = arr[:, :, :3].astype(np.float32)
    fg = (np.abs(rgb - 255.0) > 12.0).any(axis=2)
    ys, xs = np.where(fg)
    if ys.size == 0:
        return rgba
    pad = 4
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(rgba.height, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(rgba.width, int(xs.max()) + 1 + pad)
    return rgba.crop((x0, y0, x1, y1))


def build_column_header_row(
    col_widths: list[int],
    labels: list[str],
    math_mask: list[bool] | None = None,
    fontsize: int = COL_LABEL_FONT_SIZE,
    pad_y: int = COL_LABEL_PAD_Y,
    bg_rgb: tuple[int, int, int] = COL_LABEL_BG,
    left_gutter_w: int = 0,
) -> torch.Tensor:
    """Horizontal header strip aligned to ``col_widths`` (+ optional left gutter)."""
    if len(col_widths) != len(labels):
        raise ValueError("col_widths/labels length mismatch")
    if math_mask is None:
        math_mask = [False] * len(labels)
    if len(math_mask) != len(labels):
        raise ValueError("math_mask/labels length mismatch")

    label_imgs = [
        render_text_label_rgba(lab, fontsize=fontsize, math=is_math)
        for lab, is_math in zip(labels, math_mask)
    ]
    content_h = max(im.height for im in label_imgs) + 2 * pad_y
    cells: list[Image.Image] = []
    for w, im in zip(col_widths, label_imgs):
        cell_w = int(w)
        # Shrink label if wider than its column so neighboring headers never overlap.
        if im.width > cell_w - 4:
            scale = (cell_w - 4) / float(im.width)
            new_size = (
                max(1, int(round(im.width * scale))),
                max(1, int(round(im.height * scale))),
            )
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        cell = Image.new("RGB", (cell_w, content_h), bg_rgb)
        x = max(0, (cell_w - im.width) // 2)
        y = max(0, (content_h - im.height) // 2)
        cell.paste(im, (x, y), im)
        cells.append(cell)

    header = Image.new("RGB", (sum(col_widths) + left_gutter_w, content_h), bg_rgb)
    x = left_gutter_w
    for cell in cells:
        header.paste(cell, (x, 0))
        x += cell.width
    return _chw_from_pil(header)


def prepend_row_labels(
    rows: list[torch.Tensor],
    labels: list[str],
    *,
    font_size: int = ROW_LABEL_FONT_SIZE,
    caption_gap: int = ROW_LABEL_GAP,
    bg_rgba: tuple[int, int, int, int] = ROW_LABEL_BG,
) -> tuple[list[torch.Tensor], int]:
    """Prepend a vertical region label column; returns (rows, label_width)."""
    if len(rows) != len(labels):
        raise ValueError(f"rows/labels length mismatch: {len(rows)} vs {len(labels)}")
    if not rows:
        return rows, 0

    font = _load_font(font_size)
    layers: list[Image.Image] = []
    widths: list[int] = []
    for row, label in zip(rows, labels):
        row_h = int(row.shape[1])
        layer, cap_w = _vertical_caption_layer(
            label,
            row_h=row_h,
            font=font,
            caption_gap=caption_gap,
        )
        layers.append(layer)
        widths.append(cap_w)

    label_w = max(widths)
    out: list[torch.Tensor] = []
    for row, layer in zip(rows, layers):
        row_h = int(row.shape[1])
        canvas = Image.new("RGBA", (label_w, row_h), bg_rgba)
        canvas.paste(layer, ((label_w - layer.width) // 2, 0), layer)
        label_chw = _chw_from_pil(canvas)
        out.append(torch.cat([label_chw, row], dim=2))
    return out, label_w


def render_mesh_frame(
    verts: np.ndarray,
    faces: np.ndarray,
    face_vids: np.ndarray,
    *,
    rot_mat: np.ndarray | None = None,
    res: int = RENDER_RES,
    fill_ratio: float = RENDER_FILL_RATIO,
    camera_distance: float | None = RENDER_CAMERA_DISTANCE,
    yfov: float = RENDER_YFOV,
    face_crop: bool = FACE_CROP,
    face_crop_width: float = FACE_CROP_WIDTH_RATIO,
    face_crop_height: float = FACE_CROP_HEIGHT_RATIO,
    face_crop_center_x: float = FACE_CROP_CENTER_X,
    face_crop_center_y: float = FACE_CROP_CENTER_Y,
) -> torch.Tensor:
    face_ctr = verts[face_vids].mean(axis=0)
    verts_face = verts - face_ctr
    img_chw = render_mesh(
        verts_face,
        faces,
        rot_mat=rot_mat,
        res=res,
        filename=None,
        flat_shading=False,
        center_mesh=False,
        fill_ratio=fill_ratio,
        camera_distance=camera_distance,
        yfov=yfov,
    )
    img_chw = crop_chw_near_uniform_bg(img_chw)
    if not face_crop:
        return img_chw
    return crop_chw_to_face_region(
        img_chw,
        width_ratio=face_crop_width,
        height_ratio=face_crop_height,
        center_x_frac=face_crop_center_x,
        center_y_frac=face_crop_center_y,
    )


def load_pair_from_dataset(
    dataset: HACKDataset,
    src_idx: int,
    tgt_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    n = len(dataset)
    if not (0 <= src_idx < n and 0 <= tgt_idx < n):
        raise IndexError(f"src/tgt idx out of range: {src_idx}, {tgt_idx} (len={n})")

    src = dataset[src_idx]
    tgt = dataset[tgt_idx]
    src_verts = np.asarray(src["verts"], dtype=np.float64)
    tgt_verts = np.asarray(tgt["verts"], dtype=np.float64)
    faces = np.asarray(src["faces"], dtype=np.int64)

    if src_verts.shape != tgt_verts.shape:
        raise ValueError(
            f"source/target vertex count mismatch: "
            f"{src_verts.shape[0]} vs {tgt_verts.shape[0]}"
        )
    if not np.array_equal(faces, np.asarray(tgt["faces"], dtype=np.int64)):
        raise ValueError("source/target face topology mismatch")

    return src_verts, tgt_verts, faces, str(src["stem"]), str(tgt["stem"])


def run_region_row(
    job: RegionShowJob,
    *,
    dataset: HACKDataset,
    region_info_file: Path,
    face_vids: np.ndarray,
    lambda_default: float,
    render_shared: dict,
) -> list[torch.Tensor]:
    """Return 4 panels: source | target | λ1 | λ2."""
    region_key, region_name = resolve_region(job.region)
    src_verts, tgt_verts, faces, src_stem, tgt_stem = load_pair_from_dataset(
        dataset, job.src_idx, job.tgt_idx,
    )
    region_vids = load_region_vids(region_info_file, region_name)
    max_vid = int(region_vids.max()) if region_vids.size else -1
    if max_vid >= src_verts.shape[0]:
        raise ValueError(
            f"region vid {max_vid} out of range for mesh with {src_verts.shape[0]} verts"
        )

    print(
        f"[viz_dARAP_local] row={job.region}  src[{job.src_idx}]={src_stem}  "
        f"tgt[{job.tgt_idx}]={tgt_stem}  region={region_name} "
        f"(key={region_key}, {len(region_vids)} verts)  "
        f"λ={job.lambda_deform}  yaw/pitch/roll="
        f"({job.yaw:.1f},{job.pitch:.1f},{job.roll:.1f})  "
        f"crop=({job.crop_width:.2f},{job.crop_height:.2f},"
        f"cx={job.crop_center_x:.2f},cy={job.crop_center_y:.2f})",
        flush=True,
    )

    tgt_normals = vertex_normals(
        torch.tensor(tgt_verts[None], dtype=torch.float32),
        torch.tensor(faces[None], dtype=torch.long),
    )[0].numpy()

    deformed_list: list[np.ndarray] = []
    for lam in job.lambda_deform:
        deformed = deform_source_to_target_normals(
            src_verts,
            faces,
            tgt_normals,
            [region_vids.tolist()],
            default_lambda=lambda_default,
            region_lambda=float(lam),
        )
        max_disp = float(np.max(np.linalg.norm(deformed - src_verts, axis=1)))
        print(
            f"[viz_dARAP_local]   λ={lam:.2f} deform done, max disp={max_disp:.6f}",
            flush=True,
        )
        deformed_list.append(deformed)

    rot_mat = rotation_matrix_from_view_deg(job.yaw, job.pitch, job.roll)
    face_crop = bool(job.face_crop) and bool(render_shared.get("face_crop", True))
    render_kw = {
        **render_shared,
        "rot_mat": rot_mat,
        "face_crop": face_crop,
        "face_crop_width": float(job.crop_width),
        "face_crop_height": float(job.crop_height),
        "face_crop_center_x": float(job.crop_center_x),
        "face_crop_center_y": float(job.crop_center_y),
    }

    panel_labels = (
        "source",
        "target",
        f"λ={job.lambda_deform[0]:.1f}",
        f"λ={job.lambda_deform[1]:.1f}",
    )
    meshes = (src_verts, tgt_verts, deformed_list[0], deformed_list[1])
    panels: list[torch.Tensor] = []
    for label, verts in zip(panel_labels, meshes):
        panels.append(render_mesh_frame(verts, faces, face_vids, **render_kw))
        print(f"[viz_dARAP_local]   rendered {label}", flush=True)

    return panels


def column_header_labels(jobs: tuple[RegionShowJob, ...]) -> tuple[list[str], list[bool]]:
    """Top labels: source, target, $\\lambda=...$, $\\lambda=...$."""
    lam0 = float(jobs[0].lambda_deform[0])
    lam1 = float(jobs[0].lambda_deform[1])
    for job in jobs[1:]:
        if job.lambda_deform != jobs[0].lambda_deform:
            print(
                f"[viz_dARAP_local] warning: lambda_deform differs across rows "
                f"({job.region}={job.lambda_deform}); header uses first row "
                f"{jobs[0].lambda_deform}",
                flush=True,
            )
            break
    labels = [
        "source",
        "target",
        rf"\lambda={lam0:.1f}",
        rf"\lambda={lam1:.1f}",
    ]
    math_mask = [False, False, True, True]
    return labels, math_mask


def main() -> None:
    args = parse_args()
    region_info_file = resolve_region_info_file(args)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path

    packed_pt = Path(args.packed_pt)
    if not packed_pt.is_absolute():
        packed_pt = _REPO_ROOT / packed_pt
    dataset = HACKDataset(packed_pt=packed_pt)
    face_vids = load_face_vertex_ids(region_info_file)

    yfov = float(np.deg2rad(args.yfov)) if args.yfov is not None else RENDER_YFOV
    render_shared = dict(
        res=int(args.res),
        fill_ratio=float(args.fill_ratio),
        camera_distance=args.camera_distance,
        yfov=yfov,
        face_crop=not args.no_face_crop,
        face_crop_width=float(args.face_crop_width),
        face_crop_height=float(args.face_crop_height),
        face_crop_center_x=float(args.face_crop_center_x),
        face_crop_center_y=float(args.face_crop_center_y),
    )

    grid: list[list[torch.Tensor]] = []
    row_labels: list[str] = []
    for job in REGION_SHOW_JOBS:
        panels = run_region_row(
            job,
            dataset=dataset,
            region_info_file=region_info_file,
            face_vids=face_vids,
            lambda_default=float(args.lambda_default),
            render_shared=render_shared,
        )
        grid.append(panels)
        row_labels.append(job.region.capitalize())

    rows, col_widths = assemble_panel_grid(grid)
    rows, label_w = prepend_row_labels(rows, row_labels)

    col_labels, math_mask = column_header_labels(REGION_SHOW_JOBS)
    header = build_column_header_row(
        col_widths,
        col_labels,
        math_mask=math_mask,
        left_gutter_w=label_w,
    )
    mosaic = torch.cat([header, *rows], dim=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(mosaic, str(out_path))
    print(
        f"[viz_dARAP_local] wrote {out_path} "
        f"({mosaic.shape[2]}x{mosaic.shape[1]} px, "
        f"{len(REGION_SHOW_JOBS)}x4 panels + row/col labels)",
        flush=True,
    )


if __name__ == "__main__":
    main()
