import numpy as np
import subprocess
import sys
import os
import seaborn as sns
import trimesh
import glob

BLEND_SCENE = "/home/ubuntu/MyExp/head_blender_scene/render_THACK.blend"
BLENDER_EXEC = "/home/ubuntu/Downloads/blender-4.4.3-linux-x64/blender" 


def blender_render(mesh_path, out_file, res=100, scale_ratio=0.8, offset_ratio=0.02,
                   area=False, wireframe=False, t=0.008):
    """
    Render a mesh with blender. This method calls a blender python script using
    subprocess and an associated blender file containing a readily available
    rendering setup.

    Parameters
    -------------------

    mesh_path : str
        Path to the model to render. PLY and OBJ files are supported.
    out_file : str
        Output jpg/png file to which the rendering will be saved.
    collection: str
        Name of the collection in the blend file from which to choose a camera.
    viewpoint : int
        Index of the camera in the given collection.
    res : int
        Percentage of the full resolution in blender (default 100%)
    area : bool
        Vizualize vertex area as vertex colors (assumes vertex colors have been precomputed)
    wireframe : bool
        Render the model with or without wireframe.
    t : float
        Wireframe thickness
    """
    args = [BLENDER_EXEC, "-b" , 
        BLEND_SCENE, "--python",
        "/home/ubuntu/MyExp/SHACK/render/blender_tool.py", "--", 
        "-i", mesh_path,
        "-o", out_file, 
        "-r", f"{res}", 
        "--scale_ratio", f"{scale_ratio}",
        "--offset_ratio", f"{offset_ratio}",
        "-t", f"{t}"]
    
    if not wireframe:
        args.append("-s")
    if area:
        args.append("--area")
    # Redirect stdout and stderr to DEVNULL to suppress output
    subprocess.run(
        args, 
        check=True,
        # stdout=subprocess.DEVNULL, 
        # stderr=subprocess.DEVNULL
    )



if __name__ == "__main__":

    base_path = "/media/ubuntu/SSD/nersemble_preprocessed/TRACK_SPEECH_LOGS/EMO-4-disgust+happy-cam_222200037/2025-08-26_17-02-40"
    mesh_path = os.path.join(base_path, "eval_30/mesh/frame_000.obj")

    rendering_filename = os.path.join(base_path, "render/rendering.png")
    blender_render(
        mesh_path=mesh_path, 
        out_file=rendering_filename, 
        res=80,
        scale_ratio=0.8,
        offset_ratio=0.02,
        # area=True,
        wireframe=False,
    )