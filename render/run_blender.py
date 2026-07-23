import numpy as np
import subprocess
import sys
import os
import seaborn as sns
import trimesh
import glob
import re
from datetime import datetime

from render.blender_animation import blender_animation
from utils.path_utils import find_latest_timestamp_folder



def write_video(
    subject_id: str="EMO-4-disgust+happy-cam_222200037_60fps", 
    log_folder: str="/media/ubuntu/SSD/nersemble_preprocessed/TRACK_SPEECH_LOGS/",
    with_audio: bool=True, audio_file: str="",
    fps: int=60,
):
        
    base_folder = os.path.join(log_folder, f"{subject_id}")
    target_folder = find_latest_timestamp_folder(base_folder)

    mesh_folder = os.path.join(target_folder, 'eval_30/mesh')
    render_folder = os.path.join(target_folder, 'render')
    if not os.path.exists(render_folder):
        os.mkdir(render_folder)
    out_fname = f'{subject_id}_.mp4'  # output mp4

    blender_animation(
        mesh_folder=mesh_folder, 
        output_file_prefix=os.path.join(render_folder, out_fname.split('.', -1)[0]), 
        res=90,
    )

    mp4_pattern = os.path.join(render_folder, "*.mp4")
    mp4_files = sorted(glob.glob(mp4_pattern))
    mp4_path = mp4_files[0]

    if with_audio:
        assert audio_file is not None
        # audio_file = f'/media/ubuntu/SSD/MEAD_preprocessed/wav_aug/{subject_id}_0_0_001.wav'
        save_name = mp4_path.replace('.mp4', '_audio.mp4')
        cmd = [
            '/usr/bin/ffmpeg',
            '-y',  # overwrite existing file
            '-r', str(fps),  # ensure this framerate is before the input file
            # '-framerate', str(30),  
            '-i', mp4_path,  
            '-i', audio_file,
            '-c:v', 'libx264',  
            # '-vcodec', 'h264',
            '-ac', '2',
            '-channel_layout', 'stereo',
            '-pix_fmt', 'yuv420p',  
            '-ar', '22050', 
            # '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # 确保偶数分辨率（H264要求）
            '-crf', '10',  # quality（0-51，smaller the better）
            save_name,
        ]
    else:
        save_name = mp4_path
        cmd = [
            '/usr/bin/ffmpeg',
            '-y',  # overwrite existing file
            '-r', str(fps),  
            '-i', mp4_path,  
            '-c:v', 'libx264',  
            # '-vcodec', 'h264',
            '-ac', '2',
            '-channel_layout', 'stereo',
            '-pix_fmt', 'yuv420p',  
            '-ar', '22050', 
            # '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # 确保偶数分辨率（H264要求）
            # '-r', str(frame_rate),  # 
            '-crf', '20',  # quality（0-51，smaller the better）
            save_name
        ]
    subprocess.run(cmd)




if __name__ == "__main__":

    # write_video(
    #     subject_id='EMO-4-disgust+happy-cam_222200037_60fps',
    #     log_folder="/media/ubuntu/SSD/nersemble_preprocessed/TRACK_SPEECH_LOGS/",
    #     with_audio=False,
    #     fps=60,
    # )


    write_video(
        subject_id='000624_180',
        log_folder="/media/ubuntu/SSD/hallo3_preprocessed/TRACK_SPEECH_LOGS/",
        with_audio=False,
        fps=25,
    )
