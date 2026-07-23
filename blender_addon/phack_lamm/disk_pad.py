"""Viewport 2D disk pad: drag a point inside a circle to set local-edit XY offset."""

from __future__ import annotations

import math

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

# Screen-space pad: right side of the 3D view (next to N-panel / PHACK sidebar)
PAD_MARGIN = 36
PAD_RADIUS = 78
HANDLE_R = 7
# Vertical placement as fraction of region height (0=bottom, 1=top)
PAD_CENTER_Y_FRAC = 0.52

_draw_handle = None
_modal_running = False


def pad_center_radius(region) -> tuple[float, float, float]:
    """Return (cx, cy, radius) in region pixel coords (origin bottom-left).

    Anchored to the **right** edge so it sits next to the sidebar.
    """
    r = float(PAD_RADIUS)
    cx = float(region.width) - PAD_MARGIN - r
    cy = float(region.height) * PAD_CENTER_Y_FRAC
    # Keep fully inside the region
    cx = max(r + 4.0, min(cx, float(region.width) - PAD_MARGIN - r))
    cy = max(r + 4.0, min(cy, float(region.height) - PAD_MARGIN - r))
    return cx, cy, r


def uv_from_mouse(region, mx: float, my: float) -> tuple[float, float]:
    cx, cy, r = pad_center_radius(region)
    u = (mx - cx) / r
    v = (my - cy) / r
    # Clamp to unit disk
    d = math.hypot(u, v)
    if d > 1.0 and d > 1e-8:
        u /= d
        v /= d
    return float(u), float(v)


def mouse_in_pad(region, mx: float, my: float, slop: float = 12.0) -> bool:
    cx, cy, r = pad_center_radius(region)
    return math.hypot(mx - cx, my - cy) <= (r + slop)


def settings_to_delta_blender(settings) -> tuple[float, float, float]:
    """Map disk (u,v) + depth → Blender-space XYZ (Z-up head).

    Disk horizontal → X (left/right)
    Disk vertical   → Z (up/down)
    Depth slider    → Y (forward/back, face facing -Y)
    """
    scale = float(settings.disk_scale) * float(settings.amount)
    u = float(settings.disk_u)
    v = float(settings.disk_v)
    depth = float(settings.disk_depth)
    return (u * scale, depth * scale, v * scale)


def delta_blender_to_disk(settings, dx: float, dy: float, dz: float) -> None:
    """Write Blender delta back into disk_u/v/depth (best-effort)."""
    scale = float(settings.disk_scale) * float(settings.amount)
    if scale < 1e-12:
        settings.disk_u = 0.0
        settings.disk_v = 0.0
        settings.disk_depth = 0.0
        return
    u = dx / scale
    depth = dy / scale
    v = dz / scale
    d = math.hypot(u, v)
    if d > 1.0:
        u /= d
        v /= d
    settings.disk_u = float(max(-1.0, min(1.0, u)))
    settings.disk_v = float(max(-1.0, min(1.0, v)))
    settings.disk_depth = float(max(-1.0, min(1.0, depth)))


def _circle_verts(cx, cy, r, n=64):
    return [(cx + r * math.cos(2 * math.pi * i / n),
             cy + r * math.sin(2 * math.pi * i / n)) for i in range(n + 1)]


def _draw_pad():
    context = bpy.context
    if context.area is None or context.area.type != "VIEW_3D":
        return
    region = context.region
    if region is None or region.type != "WINDOW":
        return
    settings = getattr(context.scene, "phack_lamm", None)
    if settings is None or not settings.show_disk_pad:
        return

    cx, cy, r = pad_center_radius(region)
    u, v = float(settings.disk_u), float(settings.disk_v)
    hx = cx + u * r
    hy = cy + v * r

    gpu.state.blend_set("ALPHA")
    gpu.state.line_width_set(1.5)

    try:
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    except Exception:
        shader = gpu.shader.from_builtin("2D_UNIFORM_COLOR")

    # Soft fill
    fill = [(cx, cy)] + _circle_verts(cx, cy, r, 48)
    batch = batch_for_shader(shader, "TRI_FAN", {"pos": fill})
    shader.bind()
    shader.uniform_float("color", (0.12, 0.14, 0.18, 0.55))
    batch.draw(shader)

    # Outer ring
    ring = _circle_verts(cx, cy, r, 64)
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": ring})
    shader.bind()
    shader.uniform_float("color", (0.85, 0.85, 0.9, 0.95))
    batch.draw(shader)

    # Crosshair
    cross = [
        (cx - r, cy), (cx + r, cy),
        (cx, cy - r), (cx, cy + r),
    ]
    batch = batch_for_shader(shader, "LINES", {"pos": cross})
    shader.bind()
    shader.uniform_float("color", (0.55, 0.58, 0.62, 0.7))
    batch.draw(shader)

    # Handle
    handle = _circle_verts(hx, hy, HANDLE_R, 24)
    fill_h = [(hx, hy)] + handle
    batch = batch_for_shader(shader, "TRI_FAN", {"pos": fill_h})
    shader.bind()
    shader.uniform_float("color", (0.95, 0.45, 0.25, 0.95))
    batch.draw(shader)
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": handle + [handle[0]]})
    shader.bind()
    shader.uniform_float("color", (1.0, 1.0, 1.0, 0.95))
    batch.draw(shader)

    gpu.state.blend_set("NONE")
    gpu.state.line_width_set(1.0)


def ensure_draw_handler():
    global _draw_handle
    if _draw_handle is None:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw_pad, (), "WINDOW", "POST_PIXEL",
        )


def remove_draw_handler():
    global _draw_handle
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, "WINDOW")
        _draw_handle = None


def _tag_redraw_all_view3d():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _maybe_live_apply(context):
    settings = context.scene.phack_lamm
    if settings.live_apply:
        try:
            bpy.ops.phack.apply_slider_manip("EXEC_DEFAULT")
        except Exception:
            pass


class PHACK_OT_disk_pad(bpy.types.Operator):
    """Drag the orange handle inside the on-screen disk to set local offset."""

    bl_idname = "phack.disk_pad"
    bl_label = "Disk Pad Drag"
    bl_description = (
        "Drag the point inside the circle (viewport right, near sidebar). "
        "Horizontal=X, Vertical=Z; use Depth for Y"
    )
    bl_options = {"REGISTER"}

    _dragging: bool = False

    def invoke(self, context, event):
        global _modal_running
        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"WARNING"}, "Open a 3D View to use the disk pad")
            return {"CANCELLED"}

        settings = context.scene.phack_lamm
        settings.show_disk_pad = True
        ensure_draw_handler()
        _tag_redraw_all_view3d()

        if _modal_running:
            return {"CANCELLED"}

        _modal_running = True
        self._dragging = False
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(
            "Disk Pad: drag orange point · LMB drag · Enter/Esc apply & exit · R reset"
        )
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        global _modal_running
        settings = context.scene.phack_lamm
        region = context.region

        if event.type == "MOUSEMOVE" and self._dragging and region:
            u, v = uv_from_mouse(region, event.mouse_region_x, event.mouse_region_y)
            settings.disk_u = u
            settings.disk_v = v
            _tag_redraw_all_view3d()
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE":
            if event.value == "PRESS" and region and mouse_in_pad(
                region, event.mouse_region_x, event.mouse_region_y,
            ):
                self._dragging = True
                u, v = uv_from_mouse(region, event.mouse_region_x, event.mouse_region_y)
                settings.disk_u = u
                settings.disk_v = v
                _tag_redraw_all_view3d()
                return {"RUNNING_MODAL"}
            if event.value == "RELEASE" and self._dragging:
                self._dragging = False
                _maybe_live_apply(context)
                _tag_redraw_all_view3d()
                return {"RUNNING_MODAL"}

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == "PRESS":
            try:
                bpy.ops.phack.apply_slider_manip("EXEC_DEFAULT")
            except Exception as exc:
                self.report({"ERROR"}, str(exc))
            return self._finish(context)

        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            return self._finish(context)

        if event.type == "R" and event.value == "PRESS" and not event.ctrl:
            settings.disk_u = 0.0
            settings.disk_v = 0.0
            settings.disk_depth = 0.0
            _tag_redraw_all_view3d()
            _maybe_live_apply(context)
            return {"RUNNING_MODAL"}

        # Pass through navigation when not dragging on the pad
        if self._dragging:
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def _finish(self, context):
        global _modal_running
        _modal_running = False
        self._dragging = False
        context.workspace.status_text_set(None)
        _tag_redraw_all_view3d()
        return {"FINISHED"}


class PHACK_OT_toggle_disk_pad(bpy.types.Operator):
    bl_idname = "phack.toggle_disk_pad"
    bl_label = "Show / Hide Disk Pad"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = context.scene.phack_lamm
        settings.show_disk_pad = not settings.show_disk_pad
        if settings.show_disk_pad:
            ensure_draw_handler()
        _tag_redraw_all_view3d()
        return {"FINISHED"}


CLASSES = (
    PHACK_OT_disk_pad,
    PHACK_OT_toggle_disk_pad,
)


def register():
    from . import registry
    registry.register_classes(CLASSES)
    ensure_draw_handler()


def unregister():
    from . import registry
    remove_draw_handler()
    registry.unregister_classes(CLASSES)
