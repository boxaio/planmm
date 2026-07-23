#!/usr/bin/env python3
"""Bilateral Normal Integration CLI.

Usage:
    python cli.py run <path> [-k 2] [--iter 150] [--tol 1e-4] [--output-dir DIR] [--json]
    python cli.py info <path> [--json]
    python cli.py batch <paths...> [-k 2] [--iter 150] [--tol 1e-4] [--json]
"""

import argparse
import contextlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mesh import write_mesh_obj

warnings.filterwarnings("ignore")

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_SOLVER_ERROR = 2


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr so library prints don't corrupt JSON output."""
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


_NORMAL_MAP_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _find_normal_map_files(path):
    """Return sorted filenames in path whose names contain 'normal_map'."""
    names = []
    with os.scandir(path) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            name = entry.name
            if "normal_map" not in name.lower():
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in _NORMAL_MAP_IMAGE_EXTS:
                names.append(name)
    names.sort()
    return names


def _read_normal_map(normal_path):
    """Read and normalize a single normal-map image."""
    import cv2
    import numpy as np

    raw = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"failed to read image: {normal_path}")
    normal_map = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    bit_depth = 16 if normal_map.dtype == np.uint16 else 8
    if bit_depth == 16:
        normal_map = normal_map / 65535 * 2 - 1
    else:
        normal_map = normal_map / 255 * 2 - 1
    return normal_map, bit_depth


def _mask_from_normal_image(normal_path):
    """Build a boolean mask from non-zero pixels in a normal-map image."""
    import cv2
    import numpy as np

    raw = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"failed to read image: {normal_path}")
    if raw.ndim == 2:
        return raw > 0
    return np.any(raw > 0, axis=-1)


def _load_bool_mask_image(mask_path, target_shape):
    """Load grayscale/bool mask PNG and resize to ``target_shape`` (H, W) if needed."""
    import cv2
    import numpy as np

    raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"failed to read mask image: {mask_path}")
    mask = raw > 0
    th, tw = int(target_shape[0]), int(target_shape[1])
    if mask.shape[0] != th or mask.shape[1] != tw:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (tw, th),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def load_dataset(path, mask_path=None):
    """Load a normal-map image and build integration mask.

    ``path`` can be either a single normal-map image file or a directory
    containing files whose names include ``normal_map``.

    If ``mask_path`` is set, load an external bool mask (e.g. Segformer face
    region) and intersect with non-zero normal-map pixels.

    Returns dict with keys: normal_maps, normal_map_files, normal_map, mask,
    masks, K, projection, img_shape, bit_depth, num_normals.
    """
    import numpy as np

    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext not in _NORMAL_MAP_IMAGE_EXTS:
            raise FileNotFoundError(f"unsupported normal-map image: {path}")
        normal_files = [os.path.basename(path)]
        normal_paths = {normal_files[0]: os.path.abspath(path)}
    elif os.path.isdir(path):
        normal_files = _find_normal_map_files(path)
        if not normal_files:
            raise FileNotFoundError(f"no normal_map image found in {path}")
        normal_paths = {name: os.path.join(path, name) for name in normal_files}
    else:
        raise FileNotFoundError(f"normal-map image or directory not found: {path}")

    external_mask = None
    if mask_path is not None:
        mask_path = os.path.abspath(mask_path)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"mask image not found: {mask_path}")

    normal_maps = {}
    masks = {}
    bit_depths = {}
    for name in normal_files:
        normal_path = normal_paths[name]
        normal_map, bit_depth = _read_normal_map(normal_path)
        normal_maps[name] = normal_map
        base_mask = _mask_from_normal_image(normal_path)
        if mask_path is not None:
            if external_mask is None or external_mask.shape != normal_map.shape[:2]:
                external_mask = _load_bool_mask_image(mask_path, normal_map.shape[:2])
            masks[name] = base_mask & external_mask
        else:
            masks[name] = base_mask
        bit_depths[name] = bit_depth

    primary_name = normal_files[0]
    normal_map = normal_maps[primary_name]
    bit_depth = bit_depths[primary_name]
    mask = masks[primary_name]

    K_path = os.path.join(path, "K.txt") if os.path.isdir(path) else os.path.join(os.path.dirname(path), "K.txt")
    K = np.loadtxt(K_path) if os.path.isfile(K_path) else None

    return {
        "normal_maps": normal_maps,
        "normal_map_files": normal_files,
        "normal_map": normal_map,
        "mask": mask,
        "masks": masks,
        "K": K,
        "projection": "perspective" if K is not None else "orthographic",
        "img_shape": list(normal_map.shape[:2]),
        "bit_depth": bit_depth,
        "num_normals": int(np.sum(mask)),
    }


def inspect_dataset(path):
    """Inspect a dataset folder and return metadata without running integration."""
    import cv2
    import numpy as np

    if not os.path.isdir(path):
        raise FileNotFoundError(f"directory not found: {path}")

    normal_files = _find_normal_map_files(path)
    files = {name: True for name in normal_files}
    for name in ("mask.png", "K.txt"):
        files[name] = os.path.isfile(os.path.join(path, name))

    normal_maps = {}
    for name in normal_files:
        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            normal_maps[name] = {
                "image_size": [h, w],
                "bit_depth": 16 if img.dtype == np.uint16 else 8,
            }

    primary_name = normal_files[0] if normal_files else None
    primary_info = normal_maps.get(primary_name)

    result = {
        "path": os.path.abspath(path),
        "normal_map_files": normal_files,
        "normal_maps": normal_maps,
        "files": files,
        "image_size": primary_info["image_size"] if primary_info else None,
        "bit_depth": primary_info["bit_depth"] if primary_info else None,
        "num_valid_pixels": None,
        "projection": "orthographic",
    }

    if files.get("mask.png"):
        raw_mask = cv2.imread(os.path.join(path, "mask.png"), cv2.IMREAD_GRAYSCALE)
        if raw_mask is not None:
            mask = raw_mask.astype(bool)
            result["num_valid_pixels"] = int(np.sum(mask))
    elif result["image_size"]:
        result["num_valid_pixels"] = result["image_size"][0] * result["image_size"][1]

    if files.get("K.txt"):
        result["projection"] = "perspective"

    return result


# def save_outputs(depth_map, surface, wu_map, wv_map, energy_list, mask, output_dir, k):
#     """Save all output files. Returns list of saved file paths."""
#     import cv2
#     import numpy as np

#     os.makedirs(output_dir, exist_ok=True)
#     k_str = f"{k:g}"
#     saved = []

#     mesh_path = os.path.join(output_dir, f"mesh_k_{k_str}.ply")
#     surface.save(mesh_path, binary=False)
#     saved.append(mesh_path)

#     for name, wmap in [("wu", wu_map), ("wv", wv_map)]:
#         colored = cv2.applyColorMap((255 * wmap).astype(np.uint8), cv2.COLORMAP_JET)
#         colored[~mask] = 255
#         p = os.path.join(output_dir, f"{name}_k_{k_str}.png")
#         cv2.imwrite(p, colored)
#         saved.append(p)

#     depth_path = os.path.join(output_dir, "depth.npy")
#     np.save(depth_path, depth_map)
#     saved.append(depth_path)

#     energy_path = os.path.join(output_dir, "energy.npy")
#     np.save(energy_path, np.array(energy_list))
#     saved.append(energy_path)

#     return saved

def _prepare_mesh_for_obj(vertices, faces):
    """Convert bilateral-integration mesh to OBJ-friendly coordinates and scale.

    Input vertices are in OpenCV camera space (x right, y down, z into scene).
    Output vertices are centered and scaled to approximately [-1, 1]^3, using
    the same bounding-box normalization as ``read_obj(..., normalize=True)``.
    """
    import numpy as np

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)

    # OpenCV -> OpenGL/OBJ: y up, z toward viewer.
    v[:, 1] *= -1
    v[:, 2] *= -1

    min_coords = v.min(axis=0)
    max_coords = v.max(axis=0)
    center = (min_coords + max_coords) / 2.0
    extent = np.max(max_coords - min_coords)
    if extent > 0:
        v = (v - center) * (2.0 / extent)

    return v.astype(np.float32), f


def _output_stem_from_normal_map_name(normal_map_name):
    """Derive output basename from a normal-map filename without 'normal_map'."""
    import re

    stem = os.path.splitext(normal_map_name)[0]
    stem = re.sub(r"normal_map", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[._-]+", "_", stem).strip("_")
    return stem or "mesh"


def save_outputs(surface, output_dir, normal_map_name):
    """Save integration output mesh. Returns list of saved file paths."""
    os.makedirs(output_dir, exist_ok=True)
    stem = _output_stem_from_normal_map_name(normal_map_name)
    saved = []

    mesh_path = os.path.join(output_dir, f"{stem}.obj")
    vertices, faces = _prepare_mesh_for_obj(surface.points, surface.regular_faces)
    mesh_info = {
        "v": vertices,
        "fv": faces,
    }
    write_mesh_obj(mesh_info, mesh_path)
    saved.append(mesh_path)

    return saved


def run_normal_map_file(
    normal_path,
    k,
    max_iter,
    tol,
    output_dir,
    skip_existing=False,
    show_progress=True,
    K=None,
    mask_path=None,
):
    """Run bilateral normal integration on a single normal-map image."""
    import numpy as np

    from bilateral_normal_integration_cpu import bilateral_normal_integration

    normal_path = os.path.abspath(normal_path)
    normal_map_name = os.path.basename(normal_path)
    out_dir = output_dir or os.path.dirname(normal_path)
    os.makedirs(out_dir, exist_ok=True)

    stem = _output_stem_from_normal_map_name(normal_map_name)
    mesh_path = os.path.join(out_dir, f"{stem}.obj")
    if skip_existing and os.path.isfile(mesh_path):
        return {
            "status": "skipped",
            "path": normal_path,
            "output_files": [os.path.abspath(mesh_path)],
        }

    try:
        data = load_dataset(normal_path, mask_path=mask_path)
    except (FileNotFoundError, ValueError) as e:
        return {"status": "error", "path": normal_path, "error": str(e)}

    normal_map = data["normal_map"]
    mask = data["mask"]
    if mask.shape != normal_map.shape[:2]:
        mask = np.any(normal_map != 0, axis=-1)
        if mask_path is not None:
            mask = mask & _load_bool_mask_image(mask_path, normal_map.shape[:2])

    camera_K = K if K is not None else data["K"]
    projection = "perspective" if camera_K is not None else "orthographic"

    try:
        t0 = time.time()
        depth_map, surface, wu_map, wv_map, energy_list = bilateral_normal_integration(
            normal_map=normal_map,
            normal_mask=mask,
            k=k,
            K=camera_K,
            max_iter=max_iter,
            tol=tol,
            show_progress=show_progress,
        )
        saved = save_outputs(surface, out_dir, normal_map_name)
        wall_time = time.time() - t0
    except Exception as e:
        return {"status": "error", "path": normal_path, "error": str(e)}

    return {
        "status": "success",
        "path": normal_path,
        "params": {"k": k, "max_iter": max_iter, "tol": tol, "projection": projection},
        "stats": {
            "num_normals": int(np.sum(mask)),
            "iterations": len(energy_list),
            "final_energy": energy_list[-1] if energy_list else None,
            "wall_time": round(wall_time, 3),
        },
        "output_files": [os.path.abspath(p) for p in saved],
    }


def run_single(path, k, max_iter, tol, output_dir):
    """Run bilateral normal integration on a single dataset. Returns result dict."""
    path = os.path.abspath(path)
    out_dir = output_dir or path

    if os.path.isfile(path):
        return run_normal_map_file(
            path, k, max_iter, tol, out_dir, show_progress=True,
        )

    if not os.path.isdir(path):
        return {"status": "error", "path": path, "error": f"path not found: {path}"}

    normal_files = _find_normal_map_files(path)
    if not normal_files:
        return {"status": "error", "path": path, "error": f"no normal_map image found in {path}"}

    saved_all = []
    last_stats = None
    num_success = 0
    num_error = 0
    t0 = time.time()

    for normal_map_name in normal_files:
        result = run_normal_map_file(
            os.path.join(path, normal_map_name),
            k,
            max_iter,
            tol,
            out_dir,
            show_progress=False,
        )
        if result["status"] == "success":
            num_success += 1
            last_stats = result["stats"]
            saved_all.extend(result["output_files"])
        elif result["status"] == "error":
            num_error += 1

    wall_time = time.time() - t0
    if num_success == 0:
        return {
            "status": "error",
            "path": path,
            "error": f"failed on all {len(normal_files)} normal_map files",
        }

    return {
        "status": "success",
        "path": path,
        "params": {"k": k, "max_iter": max_iter, "tol": tol, "projection": "orthographic"},
        "stats": {
            "num_normals": last_stats["num_normals"] if last_stats else None,
            "iterations": last_stats["iterations"] if last_stats else None,
            "final_energy": last_stats["final_energy"] if last_stats else None,
            "wall_time": round(wall_time, 3),
            "num_maps": len(normal_files),
            "num_success": num_success,
            "num_error": num_error,
        },
        "output_files": saved_all,
    }


# ── Subcommand handlers ─────────────────────────────────────────────

def cmd_info(args):
    path = args.path
    if not os.path.isdir(path):
        err = f"directory not found: {path}"
        if args.json:
            json.dump({"status": "error", "path": path, "error": err}, sys.stdout, indent=2)
            print(file=sys.stdout)
        else:
            print(f"Error: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    info = inspect_dataset(path)

    if args.json:
        json.dump(info, sys.stdout, indent=2)
        print(file=sys.stdout)
        return EXIT_SUCCESS

    # human-readable
    print(f"Dataset: {info['path']}")
    print("-" * 40)
    for fname, exists in info["files"].items():
        tag = "found" if exists else "MISSING"
        extra = ""
        if fname in info.get("normal_maps", {}) and exists:
            meta = info["normal_maps"][fname]
            h, w = meta["image_size"]
            extra = f" ({w}x{h}, {meta['bit_depth']}-bit)"
        if fname == "K.txt" and exists:
            extra = " (perspective)"
        print(f"  {fname:<20s} {tag}{extra}")
    print()
    if info["num_valid_pixels"] is not None:
        print(f"  Valid pixels:  {info['num_valid_pixels']:,}")
    print(f"  Projection:    {info['projection']}")
    return EXIT_SUCCESS


def cmd_run(args):
    path = args.path
    if not os.path.isdir(path):
        err = f"directory not found: {path}"
        if args.json:
            json.dump({"status": "error", "path": path, "error": err}, sys.stdout, indent=2)
            print(file=sys.stdout)
        else:
            print(f"Error: {err}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if args.json:
        with _redirect_stdout_to_stderr():
            result = run_single(path, args.k, args.iter, args.tol, args.output_dir)
        json.dump(result, sys.stdout, indent=2)
        print(file=sys.stdout)
    else:
        result = run_single(path, args.k, args.iter, args.tol, args.output_dir)
        if result["status"] == "success":
            s = result["stats"]
            print(f"\nDone: {result['path']}")
            print(f"  iterations={s['iterations']}  energy={s['final_energy']:.4f}  time={s['wall_time']:.1f}s")
            for f in result["output_files"]:
                print(f"  -> {f}")
        else:
            print(f"Error: {result['error']}", file=sys.stderr)

    return EXIT_SUCCESS if result["status"] == "success" else EXIT_SOLVER_ERROR


def cmd_batch(args):
    results = []
    has_error = False

    for path in args.paths:
        if not os.path.isdir(path):
            r = {"status": "error", "path": os.path.abspath(path), "error": f"directory not found: {path}"}
            results.append(r)
            has_error = True
            if not args.json:
                print(f"[SKIP] {path}: directory not found", file=sys.stderr)
            continue

        if args.json:
            with _redirect_stdout_to_stderr():
                r = run_single(path, args.k, args.iter, args.tol, None)
        else:
            r = run_single(path, args.k, args.iter, args.tol, None)

        results.append(r)
        if r["status"] != "success":
            has_error = True

    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print(file=sys.stdout)
    else:
        # summary table
        print()
        header = f"{'Dataset':<30s} {'Status':<8s} {'Normals':>10s} {'Iters':>6s} {'Energy':>12s} {'Time':>8s}"
        print(header)
        print("-" * len(header))
        total_time = 0.0
        ok_count = 0
        for r in results:
            name = os.path.basename(r["path"])
            if r["status"] == "success":
                s = r["stats"]
                total_time += s["wall_time"]
                ok_count += 1
                print(f"{name:<30s} {'ok':<8s} {s['num_normals']:>10,d} {s['iterations']:>6d} {s['final_energy']:>12.4f} {s['wall_time']:>7.1f}s")
            else:
                print(f"{name:<30s} {'ERROR':<8s} {'-':>10s} {'-':>6s} {'-':>12s} {'-':>8s}")
        print("-" * len(header))
        print(f"Total: {ok_count}/{len(results)} succeeded{' ' * 38}{total_time:>7.1f}s")

    if has_error:
        return EXIT_SOLVER_ERROR
    return EXIT_SUCCESS


# ── Argument parser ──────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(prog="bini", description="Bilateral Normal Integration — CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # shared algorithm args
    algo = argparse.ArgumentParser(add_help=False)
    algo.add_argument("-k", type=float, default=2, help="sigmoid sharpness for discontinuity preservation (default: 2)")
    algo.add_argument("--iter", type=int, default=150, help="max IRLS iterations (default: 150)")
    algo.add_argument("--tol", type=float, default=1e-4, help="relative energy convergence tolerance (default: 1e-4)")

    # shared --json flag
    json_flag = argparse.ArgumentParser(add_help=False)
    json_flag.add_argument("--json", action="store_true", help="output structured JSON to stdout")

    # run
    p_run = sub.add_parser("run", parents=[algo, json_flag], help="run integration on a single dataset")
    p_run.add_argument("path", help="path to dataset folder")
    p_run.add_argument("--output-dir", default=None, help="output directory (default: same as input path)")
    p_run.set_defaults(func=cmd_run)

    # info
    p_info = sub.add_parser("info", parents=[json_flag], help="inspect dataset metadata (no computation)")
    p_info.add_argument(
        "path", 
        default='/media/ubuntu/SSD/PHACK_code/demos/bilateral_normals', 
        help="path to dataset folder",
    )
    p_info.set_defaults(func=cmd_info)

    # batch
    p_batch = sub.add_parser("batch", parents=[algo, json_flag], help="run integration on multiple datasets")
    p_batch.add_argument("paths", nargs="+", help="paths to dataset folders")
    p_batch.set_defaults(func=cmd_batch)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
