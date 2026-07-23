import numpy as np
import os
import sys
import bpy
import argparse


# Remove annoying argument that messes up argparse
sys.argv.remove('--')


parser = argparse.ArgumentParser(description="Render OBJ meshes from a given viewpoint, with wireframe.")
parser.add_argument("--input", "-i", required=True, type=os.path.abspath, help="Meshes to render.", nargs="+")
# parser.add_argument("--collection", "-c", type=str, default="14", help="Camera collection to use.")
parser.add_argument("--smooth", "-s", action="store_true", help="Render without wireframe.")
parser.add_argument("--thickness", "-t", type=float, default=0.008, help="Thickness of the wireframe.")
# parser.add_argument("--viewpoint", "-v", type=int, default=0, help="Index of the camera with which to render.")
parser.add_argument("--output_file", "-o", required=True, type=str, help="Output filename.")
parser.add_argument("--resolution", "-r", type=int, default=100, help="Rendering resolution fraction.")
parser.add_argument("--background", action="store_true", help="whether render the background")
parser.add_argument("--scale_ratio", type=float, default=0.8)
parser.add_argument("--offset_ratio", type=float, default=0.02)
# parser.add_argument("--ours", action="store_true", help="Color mesh as ours")
# parser.add_argument("--baseline", action="store_true", help="Color mesh as baseline")
parser.add_argument("--area", action="store_true", help="Color mesh depending on surface area")
parser.add_argument("--sequence", action="store_true", help="Handle naming so that it is compatible with premiere sequence import")
# parser.add_argument("--lines", action="store_true", help="Show self intersection as lines")
# parser.add_argument("--faces", action="store_true", help="Show self intersection as faces")
parser.add_argument("--it", type=int, default=-1)


params = parser.parse_known_args()[0]
if params.sequence and params.it == -1:
    raise ValueError("Invalid iteration number!")


# assert params.collection in bpy.data.collections.keys(), "Wrong collection name!"

bpy.ops.object.select_all(action='DESELECT')


bpy.context.scene.render.engine = 'CYCLES'
# bpy.context.scene.render.engine = 'BLENDER_EEVEE'

bpy.context.scene.cycles.device = 'GPU'

material = bpy.data.materials["Material"]


def set_mesh_visibility(active_obj=None):
    """仅让 active_obj 参与渲染，其余 MESH 全部隐藏。"""
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        visible = obj is active_obj
        obj.hide_render = not visible
        obj.hide_viewport = not visible


set_mesh_visibility(None)

i = 0
for filename in params.input:
    folder, obj_file = os.path.split(filename)
    name, ext = os.path.splitext(obj_file)
    
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filename)
        # bpy.ops.import_scene.obj(filepath=filename)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=filename)
        # bpy.ops.import_mesh.ply(filepath=filename)
    else:
        raise ValueError(f"Unsupported extension: {ext} ! This script only supports OBJ and PLY files.")
    # Make the imported object the active one
    obj = bpy.context.selected_objects[-1]
    bpy.context.view_layer.objects.active = obj
    if ext == ".ply":
        # NOTE the coordinate system in Blender for .ply and .obj files differ by a rotation of 90 degrees
        obj.rotation_euler[0] = np.pi / 2


    obj.data.materials.clear()
    obj.data.materials.append(material)
    # bpy.ops.object.shade_smooth()
    set_mesh_visibility(obj)

    # Set the active camera
    # bpy.data.scenes["Scene"].camera = bpy.data.collections[params.collection].objects[params.viewpoint]
    # Render
    bpy.data.scenes["Scene"].render.image_settings.file_format = 'PNG'
    bpy.data.scenes["Scene"].render.film_transparent = not params.background
    bpy.data.scenes["Scene"].render.resolution_percentage = params.resolution
    bpy.data.scenes["Scene"].render.filepath = params.output_file
    
    bpy.ops.render.render(write_still=True)
    # bpy.ops.render.render()

    # Delete the object
    bpy.ops.object.delete()
    i+=1

