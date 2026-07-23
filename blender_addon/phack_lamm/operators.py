"""Blender operators for PHACK LAMM generate / slider local manipulation."""

from pathlib import Path

import bpy
import numpy as np

from . import mesh_utils as mu
from . import disk_pad


def _slider_offsets(settings) -> dict:
    """Build model-space offset from disk pad (u,v,depth) × scale × amount."""
    region = int(settings.region)
    dx, dy, dz = disk_pad.settings_to_delta_blender(settings)
    delta_b = np.array([dx, dy, dz], dtype=np.float64)
    if np.linalg.norm(delta_b) <= 1e-12:
        return {}
    delta_m = mu.blender_to_model(delta_b.reshape(1, 3))[0]
    return {region: [float(delta_m[0]), float(delta_m[1]), float(delta_m[2])]}


def _sync_sliders_from_model_offset(settings, region_key: int, offset_m) -> None:
    """Write applied model-space offset back to disk pad controls."""
    off_b = mu.model_to_blender(np.asarray(offset_m, dtype=np.float64).reshape(1, 3))[0]
    disk_pad.delta_blender_to_disk(settings, float(off_b[0]), float(off_b[1]), float(off_b[2]))
    settings.region = str(int(region_key))


class PHACK_OT_generate_random(bpy.types.Operator):
    bl_idname = "phack.generate_random"
    bl_label = "Generate Random Hack Mesh"
    bl_description = "Sample a random HACK identity with LAMM"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.phack_lamm
        mu.ensure_settings_defaults(settings)
        out_npz = mu.cache_dir() / "source.npz"
        try:
            settings.status = "Generating..."
            mu.run_inference_cli(settings, "generate", out_npz=out_npz)
            verts, faces, _meta = mu.load_npz(out_npz)
        except Exception as exc:
            settings.status = "Generate failed"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        mu.clear_phack_objects()
        mesh_obj = mu.create_or_update_mesh(mu.MESH_NAME, verts, faces)
        context.scene[mu.SOURCE_NPZ_KEY] = str(out_npz)

        # Reset local edit pad for a fresh identity
        settings.disk_u = 0.0
        settings.disk_v = 0.0
        settings.disk_depth = 0.0
        settings.amount = 1.0

        bpy.ops.object.select_all(action="DESELECT")
        mesh_obj.select_set(True)
        context.view_layer.objects.active = mesh_obj

        settings.status = f"Generated ({verts.shape[0]} verts)"
        self.report({"INFO"}, settings.status)
        return {"FINISHED"}


class PHACK_OT_apply_slider_manip(bpy.types.Operator):
    bl_idname = "phack.apply_slider_manip"
    bl_label = "Apply Local Edit"
    bl_description = (
        "Apply Delta × Amount as control-vertex displacement on the selected region "
        "(LAMM local manipulation)"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.phack_lamm
        mu.ensure_settings_defaults(settings)
        source_npz = Path(context.scene.get(mu.SOURCE_NPZ_KEY, ""))
        if not source_npz.is_file():
            self.report({"ERROR"}, "No source mesh. Run Generate first.")
            return {"CANCELLED"}

        offsets = _slider_offsets(settings)
        out_npz = mu.cache_dir() / "manipulated.npz"

        # Zero delta → restore source
        if not offsets:
            try:
                verts, faces, _ = mu.load_npz(source_npz)
                mesh_obj = mu.create_or_update_mesh(mu.MESH_NAME, verts, faces)
                context.view_layer.objects.active = mesh_obj
                settings.status = "Restored source (delta=0)"
                return {"FINISHED"}
            except Exception as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}

        try:
            region = int(settings.region)
            settings.status = f"Applying region {region}..."
            mu.run_inference_cli(
                settings,
                "manipulate",
                out_npz=out_npz,
                source_npz=source_npz,
                offsets=offsets,
            )
            verts, faces, _ = mu.load_npz(out_npz)
        except Exception as exc:
            settings.status = "Local edit failed"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        mesh_obj = mu.create_or_update_mesh(mu.MESH_NAME, verts, faces)
        context.view_layer.objects.active = mesh_obj
        name = mu.REGION_NAMES.get(int(settings.region), settings.region)
        settings.status = f"Applied {name}"
        self.report({"INFO"}, settings.status)
        return {"FINISHED"}


class PHACK_OT_reset_sliders(bpy.types.Operator):
    bl_idname = "phack.reset_sliders"
    bl_label = "Reset Disk"
    bl_description = "Zero the disk pad / depth and restore the source mesh"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.phack_lamm
        settings.disk_u = 0.0
        settings.disk_v = 0.0
        settings.disk_depth = 0.0
        settings.amount = 1.0
        source_npz = Path(context.scene.get(mu.SOURCE_NPZ_KEY, ""))
        if source_npz.is_file():
            verts, faces, _ = mu.load_npz(source_npz)
            mesh_obj = mu.create_or_update_mesh(mu.MESH_NAME, verts, faces)
            context.view_layer.objects.active = mesh_obj
        settings.status = "Disk reset"
        self.report({"INFO"}, settings.status)
        return {"FINISHED"}


class PHACK_OT_random_manipulate(bpy.types.Operator):
    bl_idname = "phack.random_manipulate"
    bl_label = "Random Manipulate Region"
    bl_description = "Sample a random local edit for the selected facial patch (LAMM generate_local)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.phack_lamm
        mu.ensure_settings_defaults(settings)
        source_npz = Path(context.scene.get(mu.SOURCE_NPZ_KEY, ""))
        if not source_npz.is_file():
            self.report({"ERROR"}, "No source mesh. Run Generate first.")
            return {"CANCELLED"}

        region = int(settings.region)
        out_npz = mu.cache_dir() / "manipulated.npz"
        try:
            settings.status = f"Random manip region {region}..."
            mu.run_inference_cli(
                settings,
                "manipulate",
                out_npz=out_npz,
                source_npz=source_npz,
                region=region,
                random_region=True,
            )
            verts, faces, meta = mu.load_npz(out_npz)
        except Exception as exc:
            settings.status = "Random manip failed"
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        mesh_obj = mu.create_or_update_mesh(mu.MESH_NAME, verts, faces)
        applied = meta.get("applied_offsets") or {}
        if str(region) in applied or region in applied:
            off = applied.get(region, applied.get(str(region)))
            # Avoid triggering live_apply while syncing
            was_live = settings.live_apply
            settings.live_apply = False
            _sync_sliders_from_model_offset(settings, region, off)
            settings.live_apply = was_live
        context.view_layer.objects.active = mesh_obj
        settings.status = f"Random {mu.REGION_NAMES.get(region, region)} applied"
        self.report({"INFO"}, settings.status)
        return {"FINISHED"}


class PHACK_OT_frame_mesh(bpy.types.Operator):
    bl_idname = "phack.frame_mesh"
    bl_label = "Focus Hack Mesh"
    bl_options = {"REGISTER"}

    def execute(self, context):
        obj = bpy.data.objects.get(mu.MESH_NAME)
        if obj is None:
            self.report({"ERROR"}, "No PHACK mesh in scene")
            return {"CANCELLED"}
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                with context.temp_override(
                    area=area,
                    region=next(r for r in area.regions if r.type == "WINDOW"),
                ):
                    bpy.ops.view3d.view_selected()
                break
        return {"FINISHED"}


class PHACK_OT_restore_source(bpy.types.Operator):
    bl_idname = "phack.restore_source"
    bl_label = "Restore Source Mesh"
    bl_description = "Reload the last generated source identity and zero sliders"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.phack_lamm
        source_npz = Path(context.scene.get(mu.SOURCE_NPZ_KEY, ""))
        if not source_npz.is_file():
            self.report({"ERROR"}, "No cached source.npz")
            return {"CANCELLED"}
        try:
            verts, faces, _ = mu.load_npz(source_npz)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        was_live = settings.live_apply
        settings.live_apply = False
        settings.disk_u = 0.0
        settings.disk_v = 0.0
        settings.disk_depth = 0.0
        settings.amount = 1.0
        settings.live_apply = was_live

        mesh_obj = mu.create_or_update_mesh(mu.MESH_NAME, verts, faces)
        context.view_layer.objects.active = mesh_obj
        settings.status = "Restored source"
        self.report({"INFO"}, settings.status)
        return {"FINISHED"}


from . import registry


CLASSES = (
    PHACK_OT_generate_random,
    PHACK_OT_apply_slider_manip,
    PHACK_OT_reset_sliders,
    PHACK_OT_random_manipulate,
    PHACK_OT_frame_mesh,
    PHACK_OT_restore_source,
)


def register():
    registry.register_classes(CLASSES)


def unregister():
    registry.unregister_classes(CLASSES)
