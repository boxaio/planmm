import os
import cv2
import tempfile
import numpy as np
from subprocess import call
import pyrender
import trimesh
import glob
import librosa
import subprocess
import torch
from pytorch3d.transforms import rotation_6d_to_matrix
from tqdm import tqdm

from src.hackhead import HACKHead
from utils.mesh import read_obj
from utils.path_utils import find_latest_timestamp_folder

# os.environ['PYOPENGL_PLATFORM'] = 'egl'



def render_mesh_helper(
    verts, face, center, camera_params,
    rot=np.zeros(3), z_offset=0.0, size=(900, 900), background_black=False,
):
    h, w = size

    f = camera_params['focal length'] * max(h, w)
    cx, cy = 0.5*w, 0.5*h

    frustum = {
        'near': 0.01, 'far': 20.0, 'height': h, 'width': w,
    }

    R = rotation_6d_to_matrix(torch.from_numpy(camera_params['R_vec'])).numpy()

    verts_copy = verts.copy()
    face_copy = face.copy()

    verts_copy = cv2.Rodrigues(rot)[0].dot((verts_copy - center).T).T + center
    # verts_copy = R.dot((verts_copy - center).T).T + center

    intensity = 2.0
    rgb_per_v = None

    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    tri_mesh = trimesh.Trimesh(
        vertices=verts_copy, faces=face_copy, vertex_colors=rgb_per_v,
    )
    render_mesh = pyrender.Mesh.from_trimesh(
        tri_mesh, material=primitive_material, smooth=True,
    )

    if background_black:
        scene = pyrender.Scene(
            ambient_light=[.2, .2, .2], bg_color=[0, 0, 0],
        )  # [0, 0, 0] black,[255, 255, 255] white
    else:
        scene = pyrender.Scene(
            ambient_light=[.2, .2, .2], bg_color=[255, 255, 255],
        )  # [0, 0, 0] black,[255, 255, 255] white

    camera = pyrender.IntrinsicsCamera(
        fx=f, fy=f, cx=cx, cy=cy,
        znear=frustum['near'],
        zfar=frustum['far'],
    )

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    # camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
    RT = np.eye(4)
    RT[:3, :3] = R
    RT[:3, 3] = -camera_params['t']
    scene.add(
        camera, 
        pose=RT,
    )

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(
            viewport_width=frustum['width'], viewport_height=frustum['height'],
        )
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]


def render_sequence_meshes(
    mesh_folder: str="1001_DFA_NEU_XX/2025-07-24_19-41-44/eval_20/mesh",
    vertice_list: list=None,
    face: np.ndarray=None,
    out_folder: str="1001_DFA_NEU_XX/2025-07-24_19-41-44/mesh_video",
    audio_folder: str="AudioWAV",
    audio_file: str="1001_DFA_NEU_XX.wav",
    fps=30, 
    size=(900, 900),
    camera_params=None,
):
    sequence_vertices = []
    
    if vertice_list is None:
        # meshes = sorted(glob.glob(os.path.join(mesh_folder, '*.obj*')))
        meshes = [f for f in os.listdir(f'{mesh_folder}/') if f.endswith(".obj")]
        meshes = sorted(meshes)

        for i, mesh_path in enumerate(meshes):
            obj = read_obj(os.path.join(mesh_folder, mesh_path))
            sequence_vertices.append(obj.vs)

        sequence_vertices = np.stack(sequence_vertices)
        face = obj.fvs

    else:
        assert face is not None, "face must be provided when vertice_list is provided"
        sequence_vertices = vertice_list

    os.makedirs(out_folder, exist_ok=True)
    output_path = os.path.join(out_folder, audio_file.replace('.wav', '_mesh.mp4'))


    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_folder)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(
            tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), fps, size, True,
        )
    else:
        writer = cv2.VideoWriter(
            tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True,
        )

    num_frames = len(sequence_vertices)
    center = np.mean(sequence_vertices[0], axis=0)
    i = 0
    for frame_id in range(num_frames - 2):
        img = render_mesh_helper(
            verts=sequence_vertices[frame_id], face=face,
            center=center, 
            camera_params=camera_params,
        )  # [H, W, 3]
        writer.write(img)
        cv2.imwrite(
            os.path.join(out_folder, 'Images', str(frame_id).zfill(3) + '.png'), 
            img,
        ) 
        i = i + 1
    writer.release()

    cmd = [
        '/usr/bin/ffmpeg',
        '-y',  # overwrite existing file
        '-r', str(fps),  # ensure this framerate is before the input file
        '-i', os.path.join(audio_folder, audio_file),  
        '-i', tmp_video_file.name,
        '-c:v', 'libx264',  
        # '-vcodec', 'h264',
        '-ac', '2',
        '-channel_layout', 'stereo',
        '-pix_fmt',  'yuv420p',  
        '-ar', '22050', 
        output_path,
    ]
    subprocess.run(cmd)


def render_flame_meshes(
    track_flame_params: str,
    out_folder: str="1001_DFA_NEU_XX/2025-07-24_19-41-44/mesh_video",
    audio_folder: str="AudioWAV",
    audio_file: str="1001_DFA_NEU_XX.wav",
    fps=24, 
    size=(900, 900),
):
    from dataset.run_spectre import spectre_cfg
    from external.spectre.src.models.FLAME import FLAME, FLAMETex
    
    params = np.load(track_flame_params)
    num_frames = len(params['cam'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flame = FLAME(spectre_cfg.model).to(device)

    camera_params = {
        'focal length': 1.0, # float
        "principal point": np.array([0.0, 0.0]), # [2, ]
        "R_vec": np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),  # [6, ]
        "t": np.array([0.0, 0.0, -0.5]),  # [3, ]
    }

    verts, landmarks2d, landmarks3d = flame(
        shape_params=torch.from_numpy(params['shape']).to(device), 
        expression_params=torch.from_numpy(params['exp']).to(device), 
        pose_params=torch.from_numpy(params['pose']).to(device),
    )
    # print(verts.shape)  # [T, 5023, 3]
    verts = verts.cpu().numpy()
    faces = flame.faces_tensor.cpu().numpy()
    
    os.makedirs(out_folder, exist_ok=True)
    output_path = os.path.join(out_folder, audio_file.replace('.wav', '_mesh.mp4'))

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_folder)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(
            tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), fps, size, True,
        )
    else:
        writer = cv2.VideoWriter(
            tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True,
        )

    center = np.mean(verts[0], axis=0)
    i = 0
    for frame_id in tqdm(range(num_frames)):
        img = render_mesh_helper(
            verts=verts[frame_id], face=faces,
            center=center, 
            camera_params=camera_params,
        )  # [H, W, 3]
        writer.write(img)
        cv2.imwrite(
            os.path.join(out_folder, 'Images', str(frame_id).zfill(3) + '.png'), 
            img,
        ) 
        i = i + 1
    writer.release()

    cmd = [
        '/usr/bin/ffmpeg',
        '-y',  # overwrite existing file
        '-r', str(fps),  # ensure this framerate is before the input file
        '-i', os.path.join(audio_folder, audio_file),  
        '-i', tmp_video_file.name,
        
        # '-c:v', 'libx264',  
        '-c:v', 'h264_nvenc',  # NVIDIA硬件加速编码器
        '-preset', 'fast',     # 硬件加速的预设（fast/medium/slow）
        '-threads', '0',
        
        '-c:a', 'aac', '-ac', '2', '-ar', '22050', '-threads', '0',

        # '-ac', '2',
        # '-channel_layout', 'stereo',
        '-pix_fmt',  'yuv420p',  
        # '-ar', '22050', 
        output_path,
    ]
    subprocess.run(cmd)


    

def render_hack_meshes(
    track_hack_params: str,
    out_folder: str="1001_DFA_NEU_XX/2025-07-24_19-41-44/mesh_video",
    audio_folder: str="AudioWAV",
    audio_file: str="1001_DFA_NEU_XX.wav",
    fps=30, 
    size=(900, 900),
):
    params = np.load(track_hack_params)
    
    camera_params = {
        'focal length': params['cam_fl'][0], # float
        "principal point": params['cam_pp'][0], # [2, ]
        "R_vec": params['cam_R_vec'][0],  # [6, ]
        "t": params['cam_t'][0],  # [3, ]
        "RT": params['cam_RT'][0],  # [4, 4]
    }
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hackhead = HACKHead(use_teeth=False).to(device)
    tau = torch.zeros(1, 1).to(device)
    gamma = torch.ones(1, 1).to(device)

    num_frames = len(params['expression'])
    # template shape only
    shape = torch.from_numpy(np.zeros_like(params['shape'])).reshape(1,-1).to(device)  # [1, 200]
    neck_pose = torch.from_numpy(np.zeros_like(params['neck_pose'])).to(device)  # [T, 8, 3]

    expression = torch.from_numpy(params['expression']).to(device)  # [T, 55]

    sequence_vertices = []
    for i in range(num_frames):
        results = hackhead(
            shape=shape, 
            expression=expression[[i]], 
            neck_pose=neck_pose[[i]], 
            tau=tau, 
            gamma=gamma,
            return_landmarks=False,
        )
        verts_base = results[0].squeeze().cpu().numpy()
        sequence_vertices.append(verts_base)
    sequence_vertices = np.stack(sequence_vertices)
    face = hackhead.faces.cpu().numpy()
    
    os.makedirs(out_folder, exist_ok=True)
    output_path = os.path.join(out_folder, audio_file.replace('.wav', '_mesh.mp4'))

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_folder)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(
            tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), fps, size, True,
        )
    else:
        writer = cv2.VideoWriter(
            tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, True,
        )

    num_frames = len(sequence_vertices)
    center = np.mean(sequence_vertices[0], axis=0)
    i = 0
    for frame_id in range(num_frames - 2):
        img = render_mesh_helper(
            verts=sequence_vertices[frame_id], face=face,
            center=center, 
            camera_params=camera_params,
        )  # [H, W, 3]
        writer.write(img)
        cv2.imwrite(
            os.path.join(out_folder, 'Images', str(frame_id).zfill(3) + '.png'), 
            img,
        ) 
        i = i + 1
    writer.release()

    cmd = [
        '/usr/bin/ffmpeg',
        '-y',  # overwrite existing file
        '-r', str(fps),  # ensure this framerate is before the input file
        '-i', os.path.join(audio_folder, audio_file),  
        '-i', tmp_video_file.name,
        
        # '-c:v', 'libx264',  
        '-c:v', 'h264_nvenc',  # NVIDIA硬件加速编码器
        '-preset', 'fast',     # 硬件加速的预设（fast/medium/slow）
        '-threads', '0',
        
        '-c:a', 'aac', '-ac', '2', '-ar', '22050', '-threads', '0',

        # '-ac', '2',
        # '-channel_layout', 'stereo',
        '-pix_fmt',  'yuv420p',  
        # '-ar', '22050', 
        output_path,
    ]
    subprocess.run(cmd)


if __name__ == "__main__":

    dataset = 'HDTF'
    # dataset = 'TFHP'

    flame_params_folder = f"/media/ubuntu/SSD/{dataset}_preprocessed/SPECTRE_FLAME_PARAMS"
    subject = 'WRA_KellyAyotte_000'
    # subject = 'TH_002_000'

    flame_params_path = f"{flame_params_folder}/{subject}.npz"

    render_flame_meshes(
        track_flame_params=flame_params_path,
        out_folder=f"/media/ubuntu/SSD/{dataset}_preprocessed/SPECTRE_OUTPUT/{subject}",
        audio_folder=f"/media/ubuntu/SSD/{dataset}_preprocessed/AudioWAV",
        audio_file=f"{subject}.wav",
        fps=24, 
        size=(900, 900),
    )



# if __name__ == "__main__":

#     base_path = "/media/ubuntu/SSD/HDTF_preprocessed/TRACK_HDTF_SPEECH_LOGS"
#     subject_id = 'RD_Radio34_002'

#     exp_path = find_latest_timestamp_folder(
#         os.path.join(base_path, f'{subject_id}'), # 
#     )
#     mesh_dir = os.path.join(base_path, exp_path, "eval/mesh")

#     track_params_path = os.path.join(
#         base_path, exp_path, "tracked_hack_params.npz",
#     )

#     render_hack_meshes(
#         track_hack_params=track_params_path,
#         out_folder=os.path.join(base_path, exp_path, "mesh_video"),
#         audio_folder="/media/ubuntu/SSD/HDTF_preprocessed/AudioWAV",
#         audio_file=f"{subject_id}.wav",
#         fps=30, 
#         size=(900, 900),
#     )
