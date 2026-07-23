import os
import glob
import subprocess
import trimesh


BLEND_SCENE = "/home/ubuntu/MyExp/head_blender_scene/render_SHACK.blend"
BLENDER_EXEC = "/home/ubuntu/Downloads/blender-4.4.3-linux-x64/blender" 


def blender_animation(
    mesh_folder, output_file_prefix, res=100, area=False, wireframe=False, t=0.008,
):
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
    args = [
        BLENDER_EXEC, "-b" , BLEND_SCENE, 
        "--python", "/home/ubuntu/MyExp/SHACK/render/meshes_to_animation.py", "--", 
        "-i", mesh_folder,
        "-o", output_file_prefix, 
        "-r", f"{res}", 
        "-t", f"{t}",
    ]
    
    if not wireframe:
        args.append("-s")
    if area:
        args.append("--area")
    # Redirect stdout and stderr to DEVNULL to suppress output
    subprocess.run(
        args, 
        check=True, 
        # stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    
if __name__ == "__main__":

    base_path = "/media/ubuntu/SSD/nersemble_preprocessed/TRACK_SPEECH_LOGS/EMO-4-disgust+happy-cam_222200037/2025-08-26_17-02-40"
    mesh_folder = os.path.join(base_path, "eval_30/mesh/")

    

    rendering_filename = os.path.join(base_path, "render/rendering.png")
    blender_animation(
        mesh_folder=mesh_folder, 
        out_file_prefix=rendering_filename, 
        res=80,
        scale_ratio=0.8,
        offset_ratio=0.02,
        # area=True,
        wireframe=False,
    )


# if __name__ == "__main__":

#     wireframes = []
#     renderings = []
#     res = 80

#     mesh_folder="/media/box/Elements/Exp/ScanTalk/Demo/mydemo/mj/Meshes/"
#     obj_files = glob.glob(mesh_folder + "*.obj") or glob.glob(mesh_folder + "*.ply")
#     mesh = trimesh.load(obj_files[0])
#     thickness = mesh.edges_unique_length.min()

#     wireframe_filename = "./results/mj"


    # blender_animation(
    #     mesh_folder=mesh_folder, 
    #     out_mp4_file=wireframe_filename, 
    #     res=res,
    #     wireframe=True,
    #     t=thickness,
    # )

    # fps = 30
    # audio_fname = '/media/box/Elements/Exp/ScanTalk/src/examples/1446-122614-0094.flac'
    # video_fname = './results/mj.mp4'

    # image_files = sorted(glob.glob('./results'+'*.png'))
    # pattern = "mj%4d.png"
    # # cmd = ('/usr/bin/ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p -ar 22050 {2}'.format(
    # #         audio_fname, image_files[0].rsplit('.', 1)[0] + '%03d.png', video_fname)).split()
    # # call(cmd)
    # cmd = [
    #     '/usr/bin/ffmpeg',
    #     '-y',  # overwrite existing file
    #     '-framerate', str(fps),  
    #     '-i', os.path.join('./results', pattern),  
    #     '-i', audio_fname,
    #     '-c:v', 'libx264',  
    #     # '-vcodec', 'h264',
    #     '-ac', '2',
    #     '-channel_layout', 'stereo',
    #     '-pix_fmt', 'yuv420p',  
    #     '-ar', '22050', 
    #     # '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # 确保偶数分辨率（H264要求）
    #     # '-r', str(frame_rate),  # 
    #     '-crf', '20',  # quality（0-51，smaller the better）
    #     video_fname
    # ]
    # subprocess.run(cmd)
