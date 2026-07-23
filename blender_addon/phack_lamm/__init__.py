bl_info = {
    "name": "PHACK LAMM Hack Generator",
    "author": "PHACK",
    "version": (1, 4, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > PHACK",
    "description": (
        "Generate random HACK meshes with LAMM and edit 10 facial patches "
        "via a 2D disk pad (local manipulation)."
    ),
    "category": "Mesh",
    "doc_url": "",
}

from . import disk_pad, mesh_utils, operators, panels, properties


def register():
    # Idempotent: leftover classes from a failed prior reload
    try:
        unregister()
    except Exception:
        pass
    properties.register()
    operators.register()
    disk_pad.register()
    panels.register()
    print("[PHACK LAMM] registered")


def unregister():
    panels.unregister()
    disk_pad.unregister()
    operators.unregister()
    properties.unregister()
    print("[PHACK LAMM] unregistered")


if __name__ == "__main__":
    register()
