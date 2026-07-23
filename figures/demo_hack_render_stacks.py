from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

DEMO_IMG_DIR = Path(__file__).resolve().parent / "demo_hack_render_pngs"
OUTPUT_PATH = Path(__file__).resolve().parent / "fig_demo_hack_render_stacks.png"

demo_imgs = [
    "ffhq_00998_nicp.png",
    "ffhq_14179_nicp.png",
    "ffhq_16098_nicp.png",
    "ffhq_16387_nicp.png",
    "ffhq_16619_nicp.png",
    "ffhq_17064_nicp.png",
    "ffhq_17628_nicp.png",
    "ffhq_18379_repair.png",
    "facescape_1_3_mouth_stretch_repair.png",
    "facescape_2_15_lip_roll_repair.png",
    "facescape_3_7_jaw_forward_repair.png",
    "facescape_3_18_eye_closed_repair.png",
    "facescape_8_16_grin_repair.png",
    "facescape_77_20_brow_lower_repair.png",
    "facescape_779_17_cheek_blowing_repair.png",
    "facescape_844_17_cheek_blowing_repair.png",
    "imhead_04050.000008_repair.png",
    "imhead_04050.000016_repair.png",
    "imhead_04050.000041_repair.png",
    "imhead_04050.000062_repair.png",
    "imhead_04118.000005_repair.png",
    "imhead_04118.000008_repair.png",
    "imhead_04118.000036_repair.png",
    "imhead_04118.000065_repair.png",
]

NCOLS = 8
NROWS = 3
PADDING = 0


def load_chw_rgba(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGBA")
    arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return arr


def align_rgba_tensors(tiles: list[torch.Tensor]) -> list[torch.Tensor]:
    """将各张 RGBA 图居中 pad 到同一尺寸，便于 make_grid 拼接。"""
    max_h = max(t.shape[1] for t in tiles)
    max_w = max(t.shape[2] for t in tiles)
    aligned: list[torch.Tensor] = []
    for tile in tiles:
        _, h, w = tile.shape
        canvas = torch.zeros(tile.shape[0], max_h, max_w)
        y0 = (max_h - h) // 2
        x0 = (max_w - w) // 2
        canvas[:, y0 : y0 + h, x0 : x0 + w] = tile
        aligned.append(canvas)
    return aligned


def save_rgba_grid(grid: torch.Tensor, path: Path) -> None:
    arr = (grid.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGBA").save(path)


def main() -> None:
    if len(demo_imgs) != NCOLS * NROWS:
        raise ValueError(f"期望 {NCOLS * NROWS} 张图片，当前为 {len(demo_imgs)}")

    tiles: list[torch.Tensor] = []
    for name in demo_imgs:
        path = DEMO_IMG_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"缺少图片: {path}")
        tiles.append(load_chw_rgba(path))

    aligned_tiles = align_rgba_tensors(tiles)
    grid = make_grid(
        torch.stack(aligned_tiles, dim=0),
        nrow=NCOLS,
        padding=PADDING,
        pad_value=0.0,
    )
    save_rgba_grid(grid, OUTPUT_PATH)
    print(f"[demo_hack_stacks] 已保存 {NROWS}×{NCOLS} 拼图 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
