import os
from pathlib import Path


BASE_DIR = "/home/ubuntu/MyExp/TRACK/"

mead_data_dir = "/media/ubuntu/BoxAI/Datasets/MEAD/"
mead_metadata_path = '/media/ubuntu/SSD/TMEAD_data/mead_metadata.pkl'

mead_preprocessed_dir = "/media/ubuntu/SSD/MEAD_preprocessed/"

FLAME_ASSETS = os.path.join(BASE_DIR, 'assets/flame/')

FLAME_MODEL_PATH = os.path.join(FLAME_ASSETS, 'flame2023.pkl')
FLAME_MESH_PATH = os.path.join(FLAME_ASSETS, 'head_template_mesh.obj')
FLAME_TEX_PATH = os.path.join(FLAME_ASSETS, 'FLAME_texture.npz')
FLAME_LMK_PATH = os.path.join(FLAME_ASSETS, 'landmark_embedding_with_eyes.npy')
FLAME_PARTS_PATH = os.path.join(FLAME_ASSETS, 'FLAME_masks.pkl')
FLAME_UVMASK_PATH = os.path.join(FLAME_ASSETS, 'uv_masks.npz')
FLAME_PAINTED_TEX_PATH = os.path.join(FLAME_ASSETS, 'tex_mean_painted.png')

FLAME_LMKS3D_PATH = os.path.join(BASE_DIR, "widgets/flame_mp_lmks3d_new.npz")


head_template = os.path.join(FLAME_ASSETS, 'head_template.obj')
head_template_color = os.path.join(FLAME_ASSETS, 'head_template_color.obj')
head_template_ply = os.path.join(FLAME_ASSETS, 'test_rigid.ply')
VALID_VERTICES_WIDE_REGION = os.path.join(FLAME_ASSETS, 'uv_valid_verty_noEyes_debug.npy')
VALID_VERTS_UV_MESH = os.path.join(FLAME_ASSETS, 'uv_valid_verty.npy')
VERTEX_WEIGHT_MASK = os.path.join(FLAME_ASSETS, 'flame_vertex_weights.npy')
MIRROR_INDEX = os.path.join(FLAME_ASSETS, 'flame_mirror_index.npy')
EYE_MASK = os.path.join(FLAME_ASSETS, 'uv_mask_eyes.png')
FLAME_UV_COORDS = os.path.join(FLAME_ASSETS, 'flame_uv_coords.npy')
VALID_VERTS_NARROW = os.path.join(FLAME_ASSETS, 'uv_valid_verty_noEyes.npy')
VALID_VERTS = os.path.join(FLAME_ASSETS, 'uv_valid_verty_noEyes_noEyeRegion_debug_wEars.npy')



HACK_DATA_PATH = os.path.join(BASE_DIR, "data/hack_data/")

HACK_TEMPLATE_TRI_PATH = os.path.join(HACK_DATA_PATH, "normalized_hack_template_uv.obj")
HACK_TEMPLATE_QUAD_PATH = os.path.join(HACK_DATA_PATH, "000_generic_neutral_mesh_newuv.obj")

HACK_SHAPE_PATH = os.path.join(HACK_DATA_PATH, "S.npy")
HACK_EXPRESSION_PATH = os.path.join(HACK_DATA_PATH, "E.npy")
HACK_POSE_PATH = os.path.join(HACK_DATA_PATH, "P.npy")
HACK_WEIGHT_PATH = os.path.join(HACK_DATA_PATH, "weight_map_smooth.npy")
HACK_BLENDSHAPE_PATH = os.path.join(HACK_DATA_PATH, "blendshape.npy")

HACK_LMKS3D_PATH = os.path.join(HACK_DATA_PATH, "hack_mp_lmks3d_merged.npz")
HACK_LMKS68_PATH = "/home/ubuntu/MyExp/HACK-Model/hack_lmks_68/hack_lmk68_modified.npz"
HACK_Lc_PATH = os.path.join(HACK_DATA_PATH, "Lc_mid.png")

HACK_template_larynx_PATH = os.path.join(HACK_DATA_PATH, "ts_larynx.npy")
HACK_template_bones_PATH = os.path.join(HACK_DATA_PATH, "bones_neutral.json")

HACK_SCALE_INFO = HACK_DATA_PATH + "scale_info.npz"


# paths to pretrained pixel3dmm checkpoints
CKPT_UV_PRED = os.path.join(BASE_DIR, 'pretrained_weights/uv.ckpt')
CKPT_NORMALS_PRED = os.path.join(BASE_DIR, 'pretrained_weights/normals.ckpt')


texgan_ffhq_uv_path = "/media/ubuntu/BoxAI/FFHQ-UV/checkpoints/texgan_model/texgan_ffhq_uv.pth"    

MEAD_text_path = os.path.join(BASE_DIR, "assets/list_full_mead_annotated.txt")