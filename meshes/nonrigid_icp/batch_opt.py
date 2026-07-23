from __future__ import annotations

import argparse
import copy
import json
import os
import os.path as osp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from example import build_arg_parser as build_single_arg_parser
from example import run_single_example


def collect_bni_stems(target_dir: str | os.PathLike[str]) -> list[str]:
    target_dir = Path(target_dir)
    stems = [path.stem for path in sorted(target_dir.glob("*.obj")) if path.is_file()]
    return stems


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _make_item_args(
    args: argparse.Namespace,
    stem: str,
    target_dir: str,
    output_dir: str,
) -> argparse.Namespace:
    item_args = copy.copy(args)
    item_args.stem = stem
    item_args.target_dir = target_dir
    item_args.output_dir = output_dir
    item_args.raw_path = None
    item_args.target_path = None
    item_args.bni_lmks_path = None
    item_args.output_path = None
    item_args.visualize = False
    if not bool(getattr(args, "no_example_fast", False)):
        item_args.example_fast = True
    return item_args


def _run_one_stem(
    args: argparse.Namespace,
    stem: str,
    target_dir: str,
    output_dir: str,
) -> dict[str, object]:
    item_args = _make_item_args(args, stem, target_dir, output_dir)
    summary = run_single_example(item_args)
    return {
        "stem": stem,
        "status": "ok",
        "summary": summary,
    }


def _write_batch_summary(path: str, records: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(_json_safe(records), fp, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = build_single_arg_parser()
    parser.description = "Batch-register every BNI OBJ stem with example.py settings."
    parser.set_defaults(
        stem=None,
        visualize=False,
    )
    parser.add_argument(
        "--batch-summary-path",
        default=None,
        help="path for the batch summary json; defaults to <output-dir>/batch_summary.json",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a stem when its output OBJ already exists",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="continue with later stems if one registration fails",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most this many stems after sorting",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of worker processes for batch registration; use 1 for sequential execution",
    )
    return parser


def run_batch(args: argparse.Namespace) -> list[dict[str, object]]:
    repo_root = Path(__file__).resolve().parents[2]
    target_dir = osp.abspath(args.target_dir or repo_root / "test" / "bni")
    output_dir = osp.abspath(args.output_dir or repo_root / "test_nicp")
    stems = collect_bni_stems(target_dir)
    if args.limit is not None:
        stems = stems[: max(int(args.limit), 0)]
    if not stems:
        raise FileNotFoundError(f"No .obj files found in {target_dir}")
    if args.output_path is not None:
        raise ValueError("--output-path is for a single example; use --output-dir for batch output")

    os.makedirs(output_dir, exist_ok=True)
    summary_path = osp.abspath(args.batch_summary_path or osp.join(output_dir, "batch_summary.json"))
    print(f"[batch] target_dir={target_dir}", flush=True)
    print(f"[batch] stems={len(stems)} output_dir={output_dir}", flush=True)

    records: list[dict[str, object]] = []
    tasks: list[tuple[int, str]] = []
    for index, stem in enumerate(stems, start=1):
        out_path = osp.abspath(osp.join(output_dir, f"{stem}_nicp.obj"))
        if args.skip_existing and osp.isfile(out_path):
            record = {
                "stem": stem,
                "status": "skipped_existing",
                "output_path": out_path,
            }
            records.append(record)
            print(f"[batch] {index}/{len(stems)} skip existing {stem}", flush=True)
            continue
        tasks.append((index, stem))

    _write_batch_summary(summary_path, records)
    workers = max(int(args.workers), 1)
    workers = min(workers, max(len(tasks), 1))
    print(f"[batch] active_tasks={len(tasks)} workers={workers}", flush=True)
    if not bool(getattr(args, "no_example_fast", False)):
        print("[batch] example_fast enabled per mesh (see --no-example-fast)", flush=True)

    if workers <= 1:
        for index, stem in tasks:
            print(f"[batch] {index}/{len(stems)} start {stem}", flush=True)
            try:
                record = _run_one_stem(args, stem, target_dir, output_dir)
            except Exception as exc:
                record = {
                    "stem": stem,
                    "status": "failed",
                    "error": repr(exc),
                }
                records.append(record)
                print(f"[batch] {index}/{len(stems)} failed {stem}: {exc!r}", flush=True)
                _write_batch_summary(summary_path, records)
                if not args.continue_on_error:
                    raise
                continue

            records.append(record)
            _write_batch_summary(summary_path, records)
            print(f"[batch] {index}/{len(stems)} done {stem}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_item = {}
            for index, stem in tasks:
                print(f"[batch] {index}/{len(stems)} submit {stem}", flush=True)
                future = executor.submit(_run_one_stem, args, stem, target_dir, output_dir)
                future_to_item[future] = (index, stem)

            for future in as_completed(future_to_item):
                index, stem = future_to_item[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = {
                        "stem": stem,
                        "status": "failed",
                        "error": repr(exc),
                    }
                    records.append(record)
                    print(f"[batch] {index}/{len(stems)} failed {stem}: {exc!r}", flush=True)
                    _write_batch_summary(summary_path, records)
                    if not args.continue_on_error:
                        for pending in future_to_item:
                            pending.cancel()
                        raise
                    continue

                records.append(record)
                _write_batch_summary(summary_path, records)
                print(f"[batch] {index}/{len(stems)} done {stem}", flush=True)

    stem_order = {stem: idx for idx, stem in enumerate(stems)}
    records.sort(key=lambda rec: stem_order.get(str(rec.get("stem")), len(stems)))
    _write_batch_summary(summary_path, records)
    print(f"[batch] wrote {summary_path}", flush=True)
    return records


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
