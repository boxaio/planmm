import argparse
import concurrent.futures
from pathlib import Path
import sys
import traceback
import pickle
import numpy as np
import potpourri3d as pp3d
import torch
import torch.nn.functional as Func

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_NICP_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_NICP_DIR) not in sys.path:
    sys.path.insert(0, str(_NICP_DIR))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from mesh_intersection.bvh_search_tree import BVH
from meshes.largesteps.geometry import compute_matrix_from_lap
from meshes.largesteps.parameterize import from_differential, to_differential
from meshes.largesteps.optimize import AdamUniform

import constraints
import mutils
import optimizers
from TPE import signed_TPE_verts_movable


HACK_MESH_DIR = Path("/media/ubuntu/SSD/hack-3dgs/test_nicp")
SAVE_DIR = Path("/media/ubuntu/SSD/hack-3dgs/test_nicp")

DEFAULT_CONFIG = {
    "optimizer": "MomentumBrake",
    "lr": 0.002,
    "max_collisions": 16,
    "num_iters": 50,
    "post_iters": 200,
    "fixed_verts_weight": 150.0,
    "constraints": ["area"],
}

HACK_CAVITY_PKL_PATH = "/media/ubuntu/SSD/PHACK_code/dataset/HACK_cavity.pkl"
with open(HACK_CAVITY_PKL_PATH, "rb") as f:
    cavity_info = pickle.load(f)


def _make_cavity_face_mask(cavity_info, num_faces, device):
    """Return a boolean face mask for all HACK cavity face ids."""
    cavity_face_ids = []
    for region_info in cavity_info.values():
        cavity_face_ids.extend(region_info["face_ids"])

    cavity_face_ids = np.asarray(cavity_face_ids, dtype=np.int64)
    invalid_face_ids = cavity_face_ids[
        (cavity_face_ids < 0) | (cavity_face_ids >= num_faces)
    ]
    if invalid_face_ids.size > 0:
        raise ValueError(
            "cavity_info contains face ids outside the current mesh face range: "
            f"{invalid_face_ids[:20].tolist()}"
        )

    movable_faces = torch.zeros(num_faces, dtype=torch.bool, device=device)
    movable_faces[
        torch.as_tensor(np.unique(cavity_face_ids), dtype=torch.long, device=device)
    ] = True
    return movable_faces


def _make_vertex_mask_from_face_mask(faces, face_mask, num_vertices):
    """Return a vertex mask for vertices touched by the selected faces."""
    vertex_mask = torch.zeros(num_vertices, dtype=torch.bool, device=faces.device)
    if torch.any(face_mask):
        vertex_mask[torch.unique(faces[face_mask].reshape(-1))] = True
    return vertex_mask

def _progress(iterable, total):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc="repair", unit="mesh")


def _output_path(obj_path, input_dir, save_dir):
    rel_path = obj_path.relative_to(input_dir)
    output_name = obj_path.name[: -len("_nicp.obj")] + "_repair.obj"
    return save_dir / rel_path.parent / output_name


def _make_optimizer(name, params, lr):
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "GD":
        return optimizers.GradientDescent(params, lr=lr)
    if name == "MomentumBrake":
        return optimizers.MomentumBrake(params, lr=lr)
    if name == "AdamUniform":
        return AdamUniform(params, lr=lr)
    raise NotImplementedError(f"Not implemented optimizer type: {name}")


def _count_movable_collision_faces(collision_idxs, movable_faces):
    if collision_idxs.shape[0] == 0:
        return 0
    collision_faces = torch.unique(collision_idxs.reshape(-1))
    return int(movable_faces[collision_faces].sum().item())


def _detect_collisions(vertices, faces, search_tree):
    triangles = vertices[faces]
    collision_idxs = search_tree(triangles.unsqueeze(0)).squeeze(0)
    return collision_idxs[collision_idxs[:, 0] >= 0, :]


def repair_obj(obj_path, output_path, config, device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np_v, np_f = pp3d.read_mesh(str(obj_path))
    v_range = np_v.max() - np_v.min()
    normalize_scale = 16.5 / v_range
    np_v = normalize_scale * np_v

    write_faces = np_f.copy()
    if np_f.shape[1] == 4:
        np_f = mutils.quad_to_tri(np_f)

    vertices = torch.tensor(np_v, dtype=torch.float32, device=device)
    faces = torch.tensor(np_f, dtype=torch.long, device=device)

    movable_faces = _make_cavity_face_mask(cavity_info, np_f.shape[0], device)
    movable_faces_idx = faces[movable_faces]
    movable_verts = _make_vertex_mask_from_face_mask(
        faces, movable_faces, vertices.shape[0]
    )
    fixed_verts = ~movable_verts

    search_tree = BVH(max_collisions=config["max_collisions"])
    lap = mutils.cotan_laplacian(np_v, np_f).float().to(device)
    M = compute_matrix_from_lap(lap, vertices, lambda_=0, alpha=0.99)

    u = to_differential(M, vertices)
    u.requires_grad = True
    opt = _make_optimizer(config["optimizer"], [u], config["lr"])

    with torch.no_grad():
        V0 = (
            constraints.total_volume(vertices, movable_faces_idx)
            if "volume" in config["constraints"]
            else None
        )
        A0 = (
            constraints.total_area(vertices, movable_faces_idx)
            if "area" in config["constraints"]
            else None
        )
        L0 = (lap @ vertices).clone() if "curvature" in config["constraints"] else None

        collision_idxs = _detect_collisions(vertices, faces, search_tree)
        num_col = int(collision_idxs.shape[0])
        movable_col_faces = _count_movable_collision_faces(
            collision_idxs, movable_faces
        )

    best_vertices = vertices.detach().clone()
    best_movable_col_faces = movable_col_faces
    best_num_col = num_col
    final_vertices = vertices.detach().clone()
    max_steps = config["num_iters"] + config["post_iters"]

    if movable_col_faces > 0:
        for step in range(max_steps):
            opt.zero_grad()
            verts = from_differential(M, u, "Cholesky")

            pen_loss = signed_TPE_verts_movable(
                verts, faces, search_tree, movable_faces
            )

            reg_loss = torch.tensor(0.0, device=device)
            if V0 is not None:
                reg_loss += Func.l1_loss(
                    constraints.total_volume(verts, movable_faces_idx),
                    V0,
                )
            if A0 is not None:
                reg_loss += Func.l1_loss(
                    constraints.total_area(verts, movable_faces_idx),
                    A0,
                )
            if L0 is not None:
                reg_loss += 1e6 * Func.mse_loss(lap @ verts, L0)
            if config["fixed_verts_weight"] > 0 and torch.any(fixed_verts):
                reg_loss += config["fixed_verts_weight"] * Func.mse_loss(
                    verts[fixed_verts],
                    vertices[fixed_verts],
                )

            loss = pen_loss + reg_loss
            loss.backward()
            opt.step()

            with torch.no_grad():
                final_vertices = from_differential(M, u, "Cholesky").detach()
                collision_idxs = _detect_collisions(final_vertices, faces, search_tree)
                num_col = int(collision_idxs.shape[0])
                movable_col_faces = _count_movable_collision_faces(
                    collision_idxs, movable_faces
                )

                if (
                    movable_col_faces < best_movable_col_faces
                    or (
                        movable_col_faces == best_movable_col_faces
                        and num_col < best_num_col
                    )
                ):
                    best_movable_col_faces = movable_col_faces
                    best_num_col = num_col
                    best_vertices = final_vertices.clone()

            if movable_col_faces == 0:
                best_vertices = final_vertices.clone()
                best_movable_col_faces = 0
                best_num_col = num_col
                break

    repaired_vertices = best_vertices.cpu().numpy() / normalize_scale
    pp3d.write_mesh(repaired_vertices, write_faces, str(output_path))
    torch.cuda.empty_cache()

    return {
        "input": str(obj_path),
        "output": str(output_path),
        "movable_col_faces": best_movable_col_faces,
        "num_col": best_num_col,
    }


def _worker(task):
    obj_path, output_path, config, device = task
    try:
        result = repair_obj(obj_path, output_path, config, device)
        return True, result
    except Exception:
        return False, {
            "input": str(obj_path),
            "output": str(output_path),
            "error": traceback.format_exc(),
        }


def run_batch_repair(
    input_dir: Path | str,
    save_dir: Path | str | None = None,
    *,
    workers: int = 8,
    device: str = "cuda:0",
    overwrite: bool = False,
    config: dict | None = None,
) -> dict[str, object]:
    """Repair all ``*_nicp.obj`` meshes under ``input_dir``."""
    input_dir = Path(input_dir)
    save_dir = Path(save_dir) if save_dir is not None else input_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    repair_config = dict(DEFAULT_CONFIG)
    if config:
        repair_config.update(config)

    obj_paths = sorted(input_dir.rglob("*_nicp.obj"))
    tasks = []
    for obj_path in obj_paths:
        output_path = _output_path(obj_path, input_dir, save_dir)
        if output_path.exists() and not overwrite:
            continue
        tasks.append((obj_path, output_path, repair_config, device))

    print(f"[batch_repair_cavity] found={len(obj_paths)} queued={len(tasks)} save_dir={save_dir}", flush=True)
    if not tasks:
        return {
            "found": len(obj_paths),
            "queued": 0,
            "success": 0,
            "failed": 0,
            "ok_results": [],
            "failed_results": [],
        }

    ok_results = []
    failed_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_worker, task) for task in tasks]
        for future in _progress(concurrent.futures.as_completed(futures), len(futures)):
            ok, result = future.result()
            if ok:
                ok_results.append(result)
            else:
                failed_results.append(result)

    print(
        f"[batch_repair_cavity] done success={len(ok_results)} failed={len(failed_results)}",
        flush=True,
    )
    if failed_results:
        failed_log = save_dir / "failed_repairs.txt"
        failed_log.parent.mkdir(parents=True, exist_ok=True)
        with failed_log.open("w") as f:
            for item in failed_results:
                f.write(f"{item['input']}\n")
                f.write(item["error"])
                f.write("\n" + "=" * 80 + "\n")
        print(f"[batch_repair_cavity] failure log: {failed_log}", flush=True)

    return {
        "found": len(obj_paths),
        "queued": len(tasks),
        "success": len(ok_results),
        "failed": len(failed_results),
        "ok_results": ok_results,
        "failed_results": failed_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch repair *_nicp.obj meshes with repair_local defaults."
    )
    parser.add_argument("--input_dir", type=Path, default=HACK_MESH_DIR)
    parser.add_argument("--save_dir", type=Path, default=SAVE_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument(
        "--max_collisions", type=int, default=DEFAULT_CONFIG["max_collisions"]
    )
    parser.add_argument("--num_iters", type=int, default=DEFAULT_CONFIG["num_iters"])
    parser.add_argument("--post_iters", type=int, default=DEFAULT_CONFIG["post_iters"])
    parser.add_argument(
        "--fixed_verts_weight",
        type=float,
        default=DEFAULT_CONFIG["fixed_verts_weight"],
    )
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update(
        {
            "lr": args.lr,
            "max_collisions": args.max_collisions,
            "num_iters": args.num_iters,
            "post_iters": args.post_iters,
            "fixed_verts_weight": args.fixed_verts_weight,
        }
    )

    run_batch_repair(
        args.input_dir,
        args.save_dir,
        workers=args.workers,
        device=args.device,
        overwrite=args.overwrite,
        config=config,
    )


if __name__ == "__main__":
    main()
