"""Scene / object property definitions for PHACK LAMM addon."""

# IMPORTANT for Blender 4.x:
# - Use annotation style: `name: StringProperty(...)`
# - Do NOT use `from __future__ import annotations` (turns props into strings)
# - Do NOT use assignment style `name = StringProperty(...)` (stays as _PropertyDeferred)

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup, Scene


REGION_ITEMS = (
    ("0", "LeftFace", "Patch 0"),
    ("1", "RightFace", "Patch 1"),
    ("2", "Ears", "Patch 2"),
    ("3", "Neck", "Patch 3"),
    ("4", "Chin", "Patch 4"),
    ("5", "Skull", "Patch 5"),
    ("6", "Forehead", "Patch 6"),
    ("7", "Eyes", "Patch 7"),
    ("8", "Nose", "Patch 8"),
    ("9", "Mouth", "Patch 9"),
)

_LIVE_APPLYING = False


def _live_apply_update(self, context):
    global _LIVE_APPLYING
    if not getattr(self, "live_apply", False):
        return
    if _LIVE_APPLYING:
        return
    _LIVE_APPLYING = True
    try:
        bpy.ops.phack.apply_slider_manip("EXEC_DEFAULT")
    except Exception:
        pass
    finally:
        _LIVE_APPLYING = False


def _disk_redraw(self, context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
    _live_apply_update(self, context)


class PHACK_LammSettings(PropertyGroup):
    repo_root: StringProperty(
        name="Repo Root",
        description="PHACK_code repository root",
        default="",
        subtype="DIR_PATH",
    )
    python_exe: StringProperty(
        name="Python",
        description="Python interpreter with torch + PHACK deps (used via subprocess)",
        default="",
        subtype="FILE_PATH",
    )
    ae_checkpoint: StringProperty(
        name="AE Checkpoint",
        description=(
            "AE weights for Generate; empty = results/lamm_hack_ae/20260706_094816/best.pth"
        ),
        default="",
        subtype="FILE_PATH",
    )
    checkpoint: StringProperty(
        name="Manip Checkpoint",
        description=(
            "Manipulation weights for local edit; empty = "
            "results/lamm_hack_manipulation/best_alpha_max.pth "
            "(applied as residual on AE source)"
        ),
        default="",
        subtype="FILE_PATH",
    )
    device: StringProperty(
        name="Device",
        description='CUDA index ("0") or "cpu"',
        default="0",
    )
    seed: IntProperty(
        name="Seed",
        description="RNG seed (-1 = random)",
        default=26,
        min=-1,
    )
    k_std: FloatProperty(
        name="k_std",
        description="Std multiplier for Gaussian / displacement sampling",
        default=1.0,
        min=0.0,
        soft_max=3.0,
    )
    region: EnumProperty(
        name="Region",
        description="Facial patch to edit (10 LAMM regions)",
        items=REGION_ITEMS,
        default="8",
        update=_live_apply_update,
    )
    # Unit-disk coordinates for the viewport pad (clamped to circle).
    disk_u: FloatProperty(
        name="Disk U",
        description="Horizontal pad position → Blender X",
        default=0.0,
        min=-1.0,
        max=1.0,
        update=_disk_redraw,
    )
    disk_v: FloatProperty(
        name="Disk V",
        description="Vertical pad position → Blender Z (up)",
        default=0.0,
        min=-1.0,
        max=1.0,
        update=_disk_redraw,
    )
    disk_depth: FloatProperty(
        name="Depth",
        description="Out-of-plane offset → Blender Y (forward/back)",
        default=0.0,
        min=-1.0,
        max=1.0,
        subtype="FACTOR",
        update=_live_apply_update,
    )
    disk_scale: FloatProperty(
        name="Scale",
        description="Max displacement at disk rim (Blender units)",
        default=0.03,
        min=0.001,
        soft_max=0.1,
        precision=4,
        update=_live_apply_update,
    )
    amount: FloatProperty(
        name="Amount",
        description="Extra multiplier on disk displacement",
        default=1.0,
        min=0.0,
        soft_max=3.0,
        subtype="FACTOR",
        update=_live_apply_update,
    )
    live_apply: BoolProperty(
        name="Live Apply",
        description="Re-run LAMM when the disk/depth changes (slower; needs GPU)",
        default=False,
    )
    show_disk_pad: BoolProperty(
        name="Show Disk Pad",
        description="Show the 2D disk pad overlay in the 3D View",
        default=True,
        update=_disk_redraw,
    )
    status: StringProperty(
        name="Status",
        default="Ready",
    )


CLASSES = (PHACK_LammSettings,)


def register():
    from . import registry
    registry.register_classes(CLASSES)
    Scene.phack_lamm = PointerProperty(type=PHACK_LammSettings)


def unregister():
    from . import registry
    if hasattr(Scene, "phack_lamm"):
        del Scene.phack_lamm
    registry.unregister_classes(CLASSES)
