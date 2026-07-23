import argparse
import glob
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

# BLEND_SCENE = "/media/ubuntu/SSD/PHACK_code/figures/demo_ffhq.blend"
# BLENDER_EXEC = "/home/ubuntu/Downloads/blender-4.4.3-linux-x64/blender"

BLEND_SCENE = "/media/ubuntu/SSD/PHACK_code/figures/demo_ffhq_5.0.blend"
BLENDER_EXEC = "/home/ubuntu/Downloads/blender-5.0.0-linux-x64/blender"

BLENDER_TOOL = "/media/ubuntu/SSD/PHACK_code/figures/blender_tool.py"

demo_ffhq_meshes_dir = "/media/ubuntu/xb/FFHQ_Dataset/phack_refine_demos"
demo_facescape_meshes_dir = "/media/ubuntu/xb/FaceScape_Dataset/phack_refine_demos"
demo_imhead_meshes_dir = "/media/ubuntu/xb/ImHead_dataset/phack_refine_demos"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "demo_hack_render_pngs"

DATASET_CONFIGS: list[tuple[str, str]] = [
    ("ffhq", demo_ffhq_meshes_dir),
    ("facescape", demo_facescape_meshes_dir),
    ("imhead", demo_imhead_meshes_dir),
]


def blender_render(
    mesh_path,
    out_file,
    res=100,
    scale_ratio=0.8,
    offset_ratio=0.02,
    wireframe=True,
    t=0.008,
):
    args = [
        BLENDER_EXEC,
        "-b",
        BLEND_SCENE,
        "--python",
        BLENDER_TOOL,
        "--",
        "-i",
        mesh_path,
        "-o",
        out_file,
        "-r",
        f"{res}",
        "--scale_ratio",
        f"{scale_ratio}",
        "--offset_ratio",
        f"{offset_ratio}",
        "-t",
        f"{t}",
    ]
    if not wireframe:
        args.append("-s")
    subprocess.run(args, check=True)


def crop_rgba_whitespace(img: np.ndarray, *, pad: int = 4, alpha_thresh: int = 10) -> np.ndarray:
    """裁掉透明或近白背景边框。"""
    if img.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {img.shape}")
    if img.shape[2] == 4:
        fg = img[:, :, 3] > alpha_thresh
    else:
        fg = (np.abs(img.astype(np.float32) - 255.0) > 9.0).any(axis=2)
    ys, xs = np.where(fg)
    if ys.size == 0:
        return img
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(img.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(img.shape[1], int(xs.max()) + 1 + pad)
    return img[y0:y1, x0:x1]


def align_rgba_images(images: list[np.ndarray]) -> list[np.ndarray]:
    """将裁剪后的 RGBA 图像居中 pad 到同一尺寸。"""
    if not images:
        return images
    max_h = max(im.shape[0] for im in images)
    max_w = max(im.shape[1] for im in images)
    aligned: list[np.ndarray] = []
    for img in images:
        h, w = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1
        canvas = np.zeros((max_h, max_w, channels), dtype=np.uint8)
        y0 = (max_h - h) // 2
        x0 = (max_w - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = img
        aligned.append(canvas)
    return aligned


def list_mesh_paths(mesh_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(mesh_dir, "*.obj")))


def prefixed_name(category: str, stem: str) -> str:
    return f"{category}_{stem}"


def find_raw_png(raw_dir: Path, name: str) -> Path | None:
    direct = raw_dir / f"{name}.png"
    if direct.is_file():
        return direct
    matches = sorted(raw_dir.glob(f"*_{name}.png"))
    return matches[0] if matches else None


def final_png_path(output_dir: Path, name: str) -> Path:
    return output_dir / f"{name}.png"


def collect_mesh_jobs(
    categories: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """返回 (category, mesh_path, output_name) 列表。"""
    jobs: list[tuple[str, str, str]] = []
    for category, mesh_dir in DATASET_CONFIGS:
        if categories is not None and category not in categories:
            continue
        mesh_paths = list_mesh_paths(mesh_dir)
        if not mesh_paths:
            print(f"[{category}] 警告: 未在 {mesh_dir} 中找到 OBJ 文件")
            continue
        for mesh_path in mesh_paths:
            stem = Path(mesh_path).stem
            jobs.append((category, mesh_path, prefixed_name(category, stem)))
    return jobs


def main() -> None:
    valid_categories = [c for c, _ in DATASET_CONFIGS]
    parser = argparse.ArgumentParser(
        description="渲染 FFHQ / FaceScape / ImHead demo 目录下全部网格并裁剪留白"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=valid_categories,
        default=valid_categories,
        help="要渲染的数据集类别，默认全部",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--res", type=int, default=100, help="Blender 渲染分辨率百分比")
    parser.add_argument("--trim-pad", type=int, default=4)
    args = parser.parse_args()

    jobs = collect_mesh_jobs(args.categories)
    if not jobs:
        raise FileNotFoundError("未找到任何 OBJ 文件")

    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo_hack] 共 {len(jobs)} 个网格，开始 Blender 渲染...")
    rendered = 0
    skipped = 0
    for idx, (category, mesh_path, output_name) in enumerate(jobs):
        existing_raw = find_raw_png(raw_dir, output_name)
        if existing_raw is not None:
            skipped += 1
            print(f"  [{idx + 1}/{len(jobs)}] {output_name} — 已存在，跳过")
            continue
        raw_png = raw_dir / f"{output_name}.png"
        print(f"  [{idx + 1}/{len(jobs)}] {output_name} ({category})")
        blender_render(mesh_path=mesh_path, out_file=str(raw_png), res=args.res)
        rendered += 1

    print(f"[demo_hack] 渲染完成：新增 {rendered}，跳过 {skipped}")

    cropped_images: list[np.ndarray] = []
    output_names: list[str] = []
    for _, _, output_name in jobs:
        raw_png = find_raw_png(raw_dir, output_name)
        if raw_png is None:
            raise FileNotFoundError(f"缺少原始渲染图: {output_name}.png")
        img = np.array(Image.open(raw_png).convert("RGBA"))
        cropped_images.append(crop_rgba_whitespace(img, pad=args.trim_pad))
        output_names.append(output_name)

    aligned_images = align_rgba_images(cropped_images)
    target_h, target_w = aligned_images[0].shape[:2]
    print(f"[demo_hack] 统一输出尺寸: {target_w} x {target_h}")

    for output_name, img in zip(output_names, aligned_images):
        Image.fromarray(img, mode="RGBA").save(final_png_path(output_dir, output_name))

    print(f"[demo_hack] 已保存 {len(aligned_images)} 张 PNG → {output_dir}")


if __name__ == "__main__":
    main()
