import os
import sys
import bpy
import argparse
import glob
import numpy as np


# Remove annoying argument that messes up argparse
sys.argv.remove('--')


parser = argparse.ArgumentParser(description="Render OBJ meshes to Animation.")
parser.add_argument("--mesh_folder", "-i", required=True, type=str, help="folder of Meshes to render.")
# parser.add_argument("--collection", "-c", type=str, default="14", help="Camera collection to use.")
parser.add_argument("--smooth", "-s", action="store_true", help="Render without wireframe.")
parser.add_argument("--thickness", "-t", type=float, default=0.008, help="Thickness of the wireframe.")
# parser.add_argument("--viewpoint", "-v", type=int, default=0, help="Index of the camera with which to render.")
parser.add_argument("--output_file_prefix", "-o", required=True, type=str, help="Output filename.")
parser.add_argument("--resolution", "-r", type=int, default=100, help="Rendering resolution fraction.")
parser.add_argument("--background", action="store_true", help="whether render the background")
parser.add_argument("--scale_ratio", type=float, default=0.8)
parser.add_argument("--offset_ratio", type=float, default=0.02)
# parser.add_argument("--ours", action="store_true", help="Color mesh as ours")
# parser.add_argument("--baseline", action="store_true", help="Color mesh as baseline")
parser.add_argument("--area", action="store_true", help="Color mesh depending on surface area")
parser.add_argument("--it", type=int, default=-1)

params = parser.parse_known_args()[0]
obj_files = glob.glob(os.path.join(params.mesh_folder, "*.obj")) \
             or glob.glob(os.path.join(params.mesh_folder, "*.ply"))
obj_files = sorted(obj_files)
# print(params.mesh_folder)
# print(obj_files)

name, ext = os.path.splitext(obj_files[0])

bpy.ops.object.select_all(action='DESELECT')

# face_mat = bpy.data.materials["FaceColor"]
# black_mat = bpy.data.materials["Black"]

scene = bpy.context.scene
# 设置渲染引擎为Cycles
scene.render.engine = 'CYCLES'
# bpy.context.scene.render.engine = 'BLENDER_EEVEE'

# 设置使用CPU进行渲染
scene.cycles.device = 'GPU'
scene.cycles.samples = 128  # 降低采样数以加速
scene.cycles.use_denoising = True  # 启用降噪

# scene.render.image_settings.file_format = 'FFMPEG'
# scene.render.ffmpeg.format = 'MPEG4'
# scene.render.ffmpeg.codec = 'H264'

# 创建空物体作为父级
parent = bpy.data.objects.new("MeshParent", None)
bpy.context.collection.objects.link(parent)

# 设置动画总帧数
scene.frame_start = 0
scene.frame_end = len(obj_files)-1

# 全局变量用于跟踪当前对象
current_obj = None

def load_obj_for_frame(scene):
    global current_obj
    
    frame_num = scene.frame_current
    if frame_num > len(obj_files):
        return
    
    # 删除前一个对象
    if current_obj:
        bpy.data.objects.remove(current_obj, do_unlink=True)

    obj_file = obj_files[frame_num-1]
    
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=obj_file)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=obj_file)
    
    # 获取并配置新导入的对象
    imported = bpy.context.selected_objects[0]
    imported.location = (0, 0, 0)
    current_obj = imported

    if ext == ".ply":
        # NOTE the coordinate system in Blender for .ply and .obj files differ by a rotation of 90 degrees
        current_obj.rotation_euler[0] = np.pi / 2

    bpy.ops.object.material_slot_add()
    bpy.ops.object.material_slot_add()
    if ext == ".ply":
        bpy.ops.object.material_slot_add()

    # current_obj.data.materials[0] = face_mat
    # current_obj.data.materials[1] = black_mat

    bpy.ops.object.shade_smooth()

    # 自动缩放适配视图
    bpy.ops.object.select_all(action='DESELECT')
    current_obj.select_set(True)
    # bpy.ops.view3d.camera_to_view_selected()

     # 输出当前帧数和总帧数信息
    print(f"Rendering Frame {frame_num + 1}/{len(obj_files)}...")

# 注册帧更新处理器
bpy.app.handlers.frame_change_pre.append(load_obj_for_frame)


scene.render.film_transparent = not params.background
scene.render.resolution_percentage = params.resolution
scene.render.filepath = params.output_file_prefix


# bpy.context.window_manager.progress_begin(0, len(obj_files)-1)

bpy.ops.render.render(animation=True)
# bpy.ops.render.render(write_still=True)

# bpy.context.window_manager.progress_end()


