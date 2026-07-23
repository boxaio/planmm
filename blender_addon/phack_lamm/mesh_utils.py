"""Mesh / disk helpers shared by operators (Blender-only)."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

MESH_NAME = "PHACK_HackMesh"
DISK_PREFIX = "PHACK_Disk_"
COLLECTION_NAME = "PHACK_LAMM"
SOURCE_NPZ_KEY = "phack_source_npz"
REST_CENTERS_KEY = "phack_disk_rest"

# HACK / LAMM meshes are Y-up (Skull +Y, face +Z). Blender is Z-up.
# (x, y, z)_model -> (x, -z, y)_blender  == rotate +90° about X


def model_to_blender(verts: np.ndarray) -> np.ndarray:
    v = np.asarray(verts, dtype=np.float64)
    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = -v[..., 2]
    out[..., 2] = v[..., 1]
    return out


def blender_to_model(verts: np.ndarray) -> np.ndarray:
    v = np.asarray(verts, dtype=np.float64)
    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = v[..., 2]
    out[..., 2] = -v[..., 1]
    return out

# Distinct colors for 10 facial patches
REGION_COLORS = (
    (0.92, 0.35, 0.35, 1.0),  # LeftFace
    (0.35, 0.55, 0.95, 1.0),  # RightFace
    (0.95, 0.75, 0.25, 1.0),  # Ears
    (0.55, 0.85, 0.45, 1.0),  # Neck
    (0.85, 0.45, 0.85, 1.0),  # Chin
    (0.45, 0.85, 0.85, 1.0),  # Skull
    (0.95, 0.55, 0.25, 1.0),  # Forehead
    (0.35, 0.85, 0.65, 1.0),  # Eyes
    (0.75, 0.45, 0.35, 1.0),  # Nose
    (0.65, 0.55, 0.95, 1.0),  # Mouth
)

REGION_NAMES = {
    0: "LeftFace",
    1: "RightFace",
    2: "Ears",
    3: "Neck",
    4: "Chin",
    5: "Skull",
    6: "Forehead",
    7: "Eyes",
    8: "Nose",
    9: "Mouth",
}


def addon_dir() -> Path:
    return Path(__file__).resolve().parent


def guess_repo_root() -> Path:
    # blender_addon/phack_lamm -> PHACK_code
    return addon_dir().parent.parent


def ensure_settings_defaults(settings) -> None:
    if not settings.repo_root:
        settings.repo_root = str(guess_repo_root())
    if not settings.python_exe:
        # Prefer env that has torch; fall back to sys.executable
        candidates = [
            os.environ.get("PHACK_PYTHON", ""),
            "/home/ubuntu/anaconda3/bin/python3",
            sys.executable,
        ]
        for c in candidates:
            if c and Path(c).is_file():
                settings.python_exe = c
                break


def get_or_create_collection() -> bpy.types.Collection:
    col = bpy.data.collections.get(COLLECTION_NAME)
    if col is None:
        col = bpy.data.collections.new(COLLECTION_NAME)
        bpy.context.scene.collection.children.link(col)
    return col


def link_object(obj: bpy.types.Object, collection: bpy.types.Collection | None = None) -> None:
    collection = collection or get_or_create_collection()
    for col in list(obj.users_collection):
        col.objects.unlink(obj)
    collection.objects.link(obj)


def clear_phack_objects() -> None:
    names = [o.name for o in bpy.data.objects if o.name == MESH_NAME or o.name.startswith(DISK_PREFIX)]
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        mesh = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def verts_faces_from_mesh(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        mat = obj.matrix_world
        verts = np.array([mat @ v.co for v in mesh.vertices], dtype=np.float64)
        faces = np.array([p.vertices[:] for p in mesh.polygons], dtype=np.int64)
    finally:
        eval_obj.to_mesh_clear()
    return verts, faces


def create_or_update_mesh(name: str, verts: np.ndarray, faces: np.ndarray) -> bpy.types.Object:
    """Create / replace a mesh from model-space verts (converted to Blender Z-up)."""
    verts = model_to_blender(np.asarray(verts, dtype=np.float64))
    faces = np.asarray(faces, dtype=np.int64)

    existing = bpy.data.objects.get(name)
    if existing is not None and existing.type == "MESH":
        mesh = existing.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.delete(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], context="VERTS")
        bm.to_mesh(mesh)
        bm.free()
        mesh.clear_geometry()
        mesh.from_pydata(verts.tolist(), [], faces.tolist())
        mesh.update()
        obj = existing
    else:
        if existing is not None:
            bpy.data.objects.remove(existing, do_unlink=True)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(verts.tolist(), [], faces.tolist())
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        link_object(obj)

    # Neutral look
    if not obj.data.materials:
        mat = bpy.data.materials.get("PHACK_MeshMat")
        if mat is None:
            mat = bpy.data.materials.new("PHACK_MeshMat")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (0.82, 0.72, 0.65, 1.0)
                bsdf.inputs["Roughness"].default_value = 0.45
        obj.data.materials.append(mat)

    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.location = (0.0, 0.0, 0.0)
    obj.scale = (1.0, 1.0, 1.0)
    return obj


def _disk_material(region_key: int, opacity: float) -> bpy.types.Material:
    name = f"PHACK_DiskMat_{region_key}"
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        mat.blend_method = "BLEND"
        try:
            mat.shadow_method = "NONE"
        except Exception:
            pass
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        color = list(REGION_COLORS[region_key % len(REGION_COLORS)])
        color[3] = float(opacity)
        bsdf.inputs["Base Color"].default_value = tuple(color[:4])
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = float(opacity)
        if "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = tuple(color[:4])
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = 0.35
        elif "Emission" in bsdf.inputs:
            bsdf.inputs["Emission"].default_value = tuple(color[:4])
    return mat


def _orient_disk_matrix(center: Vector, normal: Vector) -> Matrix:
    """Build matrix: disk lies in plane with given normal, centered at center."""
    n = normal.normalized()
    if abs(n.z) < 0.9:
        up = Vector((0.0, 0.0, 1.0))
    else:
        up = Vector((0.0, 1.0, 0.0))
    x = up.cross(n).normalized()
    y = n.cross(x).normalized()
    # Blender circle is in XY; map local Z -> normal
    rot = Matrix((
        (x.x, y.x, n.x, 0.0),
        (x.y, y.y, n.y, 0.0),
        (x.z, y.z, n.z, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    return Matrix.Translation(center) @ rot


def estimate_normal(verts: np.ndarray, center: np.ndarray) -> Vector:
    """PCA-ish fallback: use vector from mesh centroid to disk center."""
    mesh_c = verts.mean(axis=0)
    n = center - mesh_c
    if np.linalg.norm(n) < 1e-8:
        n = np.array([0.0, 0.0, 1.0])
    return Vector(n.tolist())


def create_region_disks(
    regions: list[dict],
    source_verts: np.ndarray,
    opacity: float = 0.45,
    visible: bool = True,
) -> dict[int, bpy.types.Object]:
    """Create filled circle disks for each region; store rest centers on scene.

    ``source_verts`` / region centers from LAMM are model (Y-up); disks are placed
    in Blender (Z-up). Rest centers are stored in Blender space.
    """
    # Remove old disks
    for obj in list(bpy.data.objects):
        if obj.name.startswith(DISK_PREFIX):
            mesh = obj.data if obj.type == "MESH" else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)

    verts_b = model_to_blender(source_verts)
    rest = {}
    disks = {}
    for reg in regions:
        key = int(reg["key"])
        name = reg["name"]
        center_m = np.asarray(reg["center"], dtype=np.float64)
        center = model_to_blender(center_m.reshape(1, 3))[0]
        radius = float(reg["radius"])
        rest[str(key)] = center.tolist()

        disk_name = f"{DISK_PREFIX}{key}_{name}"
        mesh = bpy.data.meshes.new(disk_name)
        bm = bmesh.new()
        bmesh.ops.create_circle(bm, cap_ends=True, radius=1.0, segments=48)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        obj = bpy.data.objects.new(disk_name, mesh)
        link_object(obj)
        mat = _disk_material(key, opacity)
        if mesh.materials:
            mesh.materials[0] = mat
        else:
            mesh.materials.append(mat)

        normal = estimate_normal(verts_b, center)
        obj.matrix_world = _orient_disk_matrix(Vector(center.tolist()), normal)
        obj.scale = (radius, radius, radius * 0.05)
        obj.show_name = False
        obj.hide_viewport = not visible
        obj.hide_render = True
        obj["phack_region_key"] = key
        obj["phack_region_name"] = name
        disks[key] = obj

    scene = bpy.context.scene
    scene[REST_CENTERS_KEY] = json.dumps(rest)
    return disks


def read_disk_offsets() -> dict[int, list[float]]:
    """Disk translation offsets in **model** space (for LAMM inference)."""
    scene = bpy.context.scene
    raw = scene.get(REST_CENTERS_KEY, "{}")
    rest = json.loads(raw) if isinstance(raw, str) else dict(raw)
    offsets: dict[int, list[float]] = {}
    for obj in bpy.data.objects:
        if not obj.name.startswith(DISK_PREFIX):
            continue
        key = int(obj.get("phack_region_key", -1))
        if key < 0:
            continue
        rest_c = rest.get(str(key))
        if rest_c is None:
            continue
        cur = obj.matrix_world.translation
        delta_b = np.array(
            [float(cur.x - rest_c[0]), float(cur.y - rest_c[1]), float(cur.z - rest_c[2])],
            dtype=np.float64,
        )
        if np.linalg.norm(delta_b) <= 1e-8:
            continue
        delta_m = blender_to_model(delta_b.reshape(1, 3))[0]
        offsets[key] = [float(delta_m[0]), float(delta_m[1]), float(delta_m[2])]
    return offsets


def reset_disks_to_rest() -> None:
    scene = bpy.context.scene
    raw = scene.get(REST_CENTERS_KEY, "{}")
    rest = json.loads(raw) if isinstance(raw, str) else dict(raw)

    verts_b = None
    mesh_obj = bpy.data.objects.get(MESH_NAME)
    if mesh_obj is not None:
        verts_b, _ = verts_faces_from_mesh(mesh_obj)
    src_path = scene.get(SOURCE_NPZ_KEY, "")
    if src_path and Path(src_path).is_file():
        data = np.load(str(src_path), allow_pickle=True)
        verts_b = model_to_blender(np.asarray(data["verts"], dtype=np.float64))

    for obj in bpy.data.objects:
        if not obj.name.startswith(DISK_PREFIX):
            continue
        key = int(obj.get("phack_region_key", -1))
        rest_c = rest.get(str(key))
        if rest_c is None:
            continue
        center = Vector(rest_c)
        if verts_b is not None:
            normal = estimate_normal(verts_b, np.asarray(rest_c))
            scale = obj.scale.copy()
            obj.matrix_world = _orient_disk_matrix(center, normal)
            obj.scale = scale
        else:
            obj.location = center


def set_disk_offset(region_key: int, offset: np.ndarray) -> None:
    """Move disk by a **model-space** offset (from LAMM applied_offsets)."""
    scene = bpy.context.scene
    raw = scene.get(REST_CENTERS_KEY, "{}")
    rest = json.loads(raw) if isinstance(raw, str) else dict(raw)
    rest_c = rest.get(str(region_key))
    if rest_c is None:
        return
    offset_b = model_to_blender(np.asarray(offset, dtype=np.float64).reshape(1, 3))[0]
    for obj in bpy.data.objects:
        if not obj.name.startswith(DISK_PREFIX):
            continue
        if int(obj.get("phack_region_key", -1)) != region_key:
            continue
        target = Vector(rest_c) + Vector(offset_b.tolist())
        mw = obj.matrix_world.copy()
        mw.translation = target
        obj.matrix_world = mw
        break


def cache_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "phack_lamm_blender"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_inference_cli(
    settings,
    mode: str,
    *,
    out_npz: Path,
    source_npz: Path | None = None,
    region: int | None = None,
    random_region: bool = False,
    offsets: dict | None = None,
) -> dict:
    """Call inference/runtime.py with the configured Python interpreter."""
    ensure_settings_defaults(settings)
    runtime_py = addon_dir() / "inference" / "runtime.py"
    if not runtime_py.is_file():
        raise FileNotFoundError(f"missing runtime: {runtime_py}")

    seed = int(settings.seed)
    cmd = [
        str(settings.python_exe),
        str(runtime_py),
        "--mode", mode,
        "--repo_root", str(settings.repo_root),
        "--device", str(settings.device),
        "--k_std", str(float(settings.k_std)),
        "--out_npz", str(out_npz),
    ]
    ckpt = str(settings.checkpoint or "")
    if ckpt:
        cmd.extend(["--checkpoint", ckpt])
    ae_ckpt = str(getattr(settings, "ae_checkpoint", "") or "")
    if ae_ckpt:
        cmd.extend(["--ae_checkpoint", ae_ckpt])
    if seed >= 0:
        cmd.extend(["--seed", str(seed)])
    if source_npz is not None:
        cmd.extend(["--source_npz", str(source_npz)])
    if region is not None:
        cmd.extend(["--region", str(region)])
    if random_region:
        cmd.append("--random_region")
    if offsets:
        cmd.extend(["--offsets_json", json.dumps({str(k): v for k, v in offsets.items()})])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"inference failed (code {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    # Last JSON line
    info = {}
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            info = json.loads(line)
            break
    return info


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(str(path), allow_pickle=True)
    verts = np.asarray(data["verts"], dtype=np.float64)
    faces = np.asarray(data["faces"], dtype=np.int64)
    meta = {}
    if "meta" in data.files:
        raw = data["meta"]
        meta = raw.item() if hasattr(raw, "item") else dict(raw)
    return verts, faces, meta
