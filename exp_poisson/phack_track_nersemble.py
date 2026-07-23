"""Run PHACK step-0 + step-1 pipeline on NeRSemble image sequences."""

from __future__ import annotations

import argparse
import os
import os.path as osp
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pred_ffhq_hack_step_0 import (
    bni,
    eval_stage1_PHACK,
    predict_bni_lmks,
    save_tgt_seg_28_by_landmarks,
)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

nersemble_root_dir = "/media/ubuntu/xb/nersemble_data/"
_ids = ["224", "473"]
camera_id = "cam_222200037"
sequence_name = "FREE"
images_subdir = "images_4"
frame_stride = 2  # 每隔一帧取一帧：0, 2, 4, ...
frame_offset = 0

OUTPUT_ROOT = Path(__file__).resolve().parent / "phack_nersemble"


def collect_nersemble_image_paths(
    subject_ids: list[str],
    *,
    camera_id: str = camera_id,
    sequence_name: str = sequence_name,
    images_subdir: str = images_subdir,
    nersemble_root: str = nersemble_root_dir,
    frame_stride: int = frame_stride,
    frame_offset: int = frame_offset,
) -> list[str]:
    """Collect image paths with subject-prefixed stems via symlinks in staging dir."""
    staging_dir = OUTPUT_ROOT / "images"
    staging_dir.mkdir(parents=True, exist_ok=True)

    stride = max(int(frame_stride), 1)
    offset = max(int(frame_offset), 0)

    image_paths: list[str] = []
    for subject_id in subject_ids:
        src_dir = osp.join(
            nersemble_root,
            subject_id,
            "sequences",
            sequence_name,
            images_subdir,
        )
        if not osp.isdir(src_dir):
            raise FileNotFoundError(f"images dir not found: {src_dir}")

        matched_names = []
        for name in sorted(os.listdir(src_dir)):
            if not name.startswith(camera_id):
                continue
            ext = osp.splitext(name)[1].lower()
            if ext not in _IMAGE_EXTENSIONS:
                continue
            matched_names.append(name)

        selected_names = matched_names[offset::stride]
        for name in selected_names:
            ext = osp.splitext(name)[1].lower()
            src_path = osp.abspath(osp.join(src_dir, name))
            stem = f"{subject_id}_{osp.splitext(name)[0]}"
            link_path = staging_dir / f"{stem}{ext}"
            if not link_path.exists():
                link_path.symlink_to(src_path)
            image_paths.append(str(link_path))
    return image_paths


def run_step0(
    image_paths: list[str],
    *,
    raw_dir: str,
    bni_dir: str,
    bni_lmks_dir: str,
    bni_seg_dir: str,
    checkpoint_path: str,
    filtered_output_path: str | None,
    skip_existing_normal_map: bool,
    no_render: bool,
) -> None:
    print(f"[step0] images={len(image_paths)} raw_dir={raw_dir}", flush=True)
    eval_stage1_PHACK(
        image_paths=image_paths,
        checkpoint_path=checkpoint_path,
        out_dir=raw_dir,
        render=not no_render,
        filtered_output_path=filtered_output_path,
        skip_existing_normal_map=skip_existing_normal_map,
    )
    bni(raw_dir=raw_dir, output_dir=bni_dir)
    predict_bni_lmks(mesh_dir=bni_dir, save_dir=bni_lmks_dir, num_workers=1)
    save_tgt_seg_28_by_landmarks(
        raw_mesh_dir=raw_dir,
        tgt_mesh_dir=bni_dir,
        bni_lmks_dir=bni_lmks_dir,
        save_dir=bni_seg_dir,
    )


def run_step1(
    *,
    raw_dir: str,
    target_dir: str,
    bni_lmks_dir: str,
    bni_seg_dir: str,
    output_dir: str,
    workers: int,
    limit: int | None,
    continue_on_error: bool,
) -> None:
    from pred_ffhq_hack_step_1 import build_arg_parser as build_step1_arg_parser
    from pred_ffhq_hack_step_1 import run_batch

    parser = build_step1_arg_parser()
    args = parser.parse_args([])
    args.raw_dir = raw_dir
    args.target_dir = target_dir
    args.bni_lmks_dir = bni_lmks_dir
    args.bni_seg_dir = bni_seg_dir
    args.output_dir = output_dir
    args.visualize = False
    args.workers = workers
    args.limit = limit
    args.continue_on_error = continue_on_error
    run_batch(args)


def run_step2_repair_cavity(
    *,
    refine_dir: str,
    repair_dir: str,
    repair_workers: int,
    repair_device: str,
    repair_overwrite: bool,
) -> dict[str, object]:
    nicp_dir = _REPO_ROOT / "meshes" / "nonrigid_icp"
    if str(nicp_dir) not in sys.path:
        sys.path.insert(0, str(nicp_dir))
    from batch_repair_cavity import run_batch_repair

    print(
        f"[step2] repair cavity self-intersections "
        f"input={refine_dir} output={repair_dir}",
        flush=True,
    )
    return run_batch_repair(
        refine_dir,
        repair_dir,
        workers=repair_workers,
        device=repair_device,
        overwrite=repair_overwrite,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PHACK step-0 and step-1 on NeRSemble image sequences.",
    )
    parser.add_argument(
        "--subject-ids",
        nargs="+",
        default=_ids,
        help="NeRSemble subject ids to process",
    )
    parser.add_argument(
        "--camera-id",
        default=camera_id,
        help="camera prefix filter, e.g. cam_222200037",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=frame_stride,
        help="keep every N-th frame after sorting; 2 means every other frame",
    )
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=frame_offset,
        help="start index offset before applying frame stride",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="root directory for all pipeline outputs",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=str(_REPO_ROOT / "checkpoint" / "epoch_latest.pth"),
        help="PHACK encoder checkpoint path",
    )
    parser.add_argument(
        "--step",
        choices=("all", "0", "1", "2"),
        default="2",
        help="run step-0/1/2 only, or all (0→1→2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most this many images / mesh stems",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="worker processes for step-1 batch registration",
    )
    parser.add_argument(
        "--repair-workers",
        type=int,
        default=8,
        help="worker threads for step-2 cavity self-intersection repair",
    )
    parser.add_argument(
        "--repair-device",
        default="cuda:0",
        help="device for step-2 batch_repair_cavity",
    )
    parser.add_argument(
        "--repair-overwrite",
        action="store_true",
        help="overwrite existing *_repair.obj in phack_repair",
    )
    parser.add_argument(
        "--skip-existing-normal-map",
        action="store_true",
        help="skip DAViD re-estimation if _normal_map.png already exists",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="skip overlap rendering in step-0",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue step-1 batch when one stem fails",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    raw_dir = str(output_root / "phack_raw")
    bni_dir = str(output_root / "bni")
    bni_lmks_dir = str(output_root / "bni_lmks")
    bni_seg_dir = str(output_root / "bni_seg")
    refine_dir = str(output_root / "phack_refine")
    repair_dir = str(output_root / "phack_repair")
    filtered_path = str(output_root / "filtered_stems.txt")

    for subdir in (raw_dir, bni_dir, bni_lmks_dir, bni_seg_dir, refine_dir, repair_dir):
        os.makedirs(subdir, exist_ok=True)

    global OUTPUT_ROOT
    OUTPUT_ROOT = output_root

    image_paths = collect_nersemble_image_paths(
        args.subject_ids,
        camera_id=args.camera_id,
        frame_stride=args.frame_stride,
        frame_offset=args.frame_offset,
    )
    if args.limit is not None:
        image_paths = image_paths[: max(int(args.limit), 0)]
    if not image_paths:
        raise SystemExit("No NeRSemble images matched the given subject/camera filters.")

    print(
        f"[phack_track_nersemble] subjects={args.subject_ids} "
        f"camera={args.camera_id} frame_stride={args.frame_stride} "
        f"frame_offset={args.frame_offset} frames={len(image_paths)} "
        f"output={output_root}",
        flush=True,
    )

    if args.step in ("all", "0"):
        run_step0(
            image_paths,
            raw_dir=raw_dir,
            bni_dir=bni_dir,
            bni_lmks_dir=bni_lmks_dir,
            bni_seg_dir=bni_seg_dir,
            checkpoint_path=args.checkpoint_path,
            filtered_output_path=filtered_path,
            skip_existing_normal_map=args.skip_existing_normal_map,
            no_render=args.no_render,
        )

    if args.step in ("all", "1"):
        run_step1(
            raw_dir=raw_dir,
            target_dir=bni_dir,
            bni_lmks_dir=bni_lmks_dir,
            bni_seg_dir=bni_seg_dir,
            output_dir=refine_dir,
            workers=args.workers,
            limit=args.limit,
            continue_on_error=args.continue_on_error,
        )

    if args.step in ("all", "2"):
        run_step2_repair_cavity(
            refine_dir=refine_dir,
            repair_dir=repair_dir,
            repair_workers=args.repair_workers,
            repair_device=args.repair_device,
            repair_overwrite=args.repair_overwrite,
        )


if __name__ == "__main__":
    main()
