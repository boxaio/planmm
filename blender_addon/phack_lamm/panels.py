"""N-panel UI for PHACK LAMM."""

import bpy

from . import registry


class PHACK_PT_lamm_panel(bpy.types.Panel):
    bl_label = "PHACK LAMM"
    bl_idname = "PHACK_PT_lamm_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PHACK"

    def draw(self, context):
        layout = self.layout
        try:
            settings = context.scene.phack_lamm
        except Exception as exc:
            layout.label(text="Settings missing — re-enable the addon.", icon="ERROR")
            layout.label(text=str(exc))
            return

        try:
            from . import mesh_utils as mu
            mu.ensure_settings_defaults(settings)
        except Exception as exc:
            layout.label(text=f"Init warning: {exc}", icon="ERROR")

        box = layout.box()
        box.label(text="Paths", icon="FILE_FOLDER")
        box.prop(settings, "repo_root")
        box.prop(settings, "python_exe")
        box.prop(settings, "ae_checkpoint")
        box.prop(settings, "checkpoint")
        box.prop(settings, "device")

        box = layout.box()
        box.label(text="1. Random Hack Mesh", icon="MESH_MONKEY")
        row = box.row(align=True)
        row.prop(settings, "seed")
        row.prop(settings, "k_std")
        box.operator("phack.generate_random", icon="FILE_REFRESH")
        row = box.row(align=True)
        row.operator("phack.restore_source", icon="LOOP_BACK")
        row.operator("phack.frame_mesh", icon="VIEWZOOM")

        box = layout.box()
        box.label(text="2. Local Region Edit", icon="MESH_CIRCLE")
        box.prop(settings, "region")
        box.label(text="Disk (right of view, near sidebar)")
        row = box.row(align=True)
        row.operator("phack.disk_pad", icon="MOUSE_LMB", text="Drag Disk Pad")
        row.operator("phack.toggle_disk_pad", icon="HIDE_OFF", text="")
        box.prop(settings, "show_disk_pad")
        box.prop(settings, "disk_scale", slider=True)
        box.prop(settings, "disk_depth", slider=True, text="Depth (Y)")
        box.prop(settings, "amount", slider=True)
        row = box.row(align=True)
        row.label(text=f"U={settings.disk_u:+.2f}  V={settings.disk_v:+.2f}")
        box.prop(settings, "live_apply")
        row = box.row(align=True)
        row.operator("phack.apply_slider_manip", icon="CHECKMARK", text="Apply")
        row.operator("phack.reset_sliders", icon="X", text="Reset")

        box = layout.box()
        box.label(text="Or: Random Sample", icon="MOD_NOISE")
        box.operator("phack.random_manipulate", icon="SHADERFX")

        layout.separator()
        layout.label(text=f"Status: {settings.status}")


CLASSES = (PHACK_PT_lamm_panel,)


def register():
    registry.register_classes(CLASSES)


def unregister():
    registry.unregister_classes(CLASSES)
