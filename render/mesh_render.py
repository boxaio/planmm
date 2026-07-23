import os
import sys
import torch
import numpy as np
import math
import PIL
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as F
from PIL import ImageFont, ImageDraw

# os.environ['PYOPENGL_PLATFORM'] = 'egl' # you may need to use this for headless rendering
import trimesh
import pyrender

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from utils.helpers import to_np

# 同一分辨率下复用 OffscreenRenderer，显著加速批量多视图渲染（fit_imhead / bfm_lmks478）。
_offscreen_renderer_cache: dict[int, pyrender.OffscreenRenderer] = {}

default_colors = [
    'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 
    'purple', 'brown', 'pink', 'gray', 'olive', 'lime', 'teal', 'aqua', 'navy', 'black',
]

seg_colors8 = [
    (0.122, 0.467, 0.706),
    (1.000, 0.498, 0.055),
    (0.173, 0.627, 0.173),
    (0.839, 0.153, 0.157),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.498, 0.498, 0.498),
]

segmentation_colors10 = plt.get_cmap('tab10').colors
segmentation_colors20 = plt.get_cmap('tab20').colors
segmentation_colors30 = [
    (0.968, 0.441, 0.536),
    (0.969, 0.454, 0.408),
    (0.954, 0.478, 0.196),
    (0.862, 0.536, 0.195),
    (0.793, 0.571, 0.195),
    (0.735, 0.595, 0.194),
    (0.68, 0.615, 0.194),
    (0.623, 0.633, 0.194),
    (0.557, 0.651, 0.193),
    (0.468, 0.67, 0.193),
    (0.313, 0.693, 0.192),
    (0.196, 0.697, 0.361),
    (0.201, 0.691, 0.48),
    (0.205, 0.686, 0.549),
    (0.208, 0.681, 0.6),
    (0.21, 0.677, 0.643),
    (0.213, 0.673, 0.684),
    (0.216, 0.668, 0.726),
    (0.22, 0.663, 0.773),
    (0.225, 0.654, 0.834),
    (0.233, 0.64, 0.926),
    (0.433, 0.607, 0.959),
    (0.583, 0.57, 0.958),
    (0.697, 0.528, 0.958),
    (0.8, 0.477, 0.958),
    (0.908, 0.402, 0.958),
    # (0.96, 0.375, 0.893),
    # (0.962, 0.398, 0.801),
    # (0.964, 0.414, 0.719),
    # (0.966, 0.428, 0.637)
]

def render_mesh(
    verts,
    faces,
    face_colors=None,
    vertex_colors=None,
    rot_mat=None,
    flat_shading=False,
    res=512,
    filename=None,
    *,
    center_mesh=True,
    camera_distance=None,
    fill_ratio=0.73,
    yfov=None,
    znear_override=None,
    zfar_override=None,
    reuse_cached_renderer=False,
):

    mesh = trimesh.Trimesh(vertices=to_np(verts), faces=to_np(faces), process=False)

    if rot_mat is not None:
        mesh.vertices = mesh.vertices @ rot_mat.T

    if face_colors is not None:
        if face_colors.dtype == np.int64 or face_colors.dtype == np.int32:
            face_colors = np.array([to_rgb(default_colors[i % len(default_colors)]) for i in face_colors])
        mesh.visual.face_colors = face_colors
    elif vertex_colors is not None:
        mesh.visual.vertex_colors = vertex_colors
    else:
        # 无颜色时用肤色顶点色 + 平滑着色；统一灰面片 + flat 在人脸上会像一块灰板。
        skin = np.asarray([0.86, 0.75, 0.69], dtype=np.float32)
        mesh.visual.vertex_colors = np.tile(skin, (len(mesh.vertices), 1))

    if yfov is None:
        yfov = np.pi / 3.0
    half = yfov * 0.5

    if center_mesh:
        ctr = mesh.vertices.mean(axis=0)
        mesh.vertices = mesh.vertices - ctr

    radius = float(np.linalg.norm(mesh.vertices, axis=1).max())
    if not np.isfinite(radius) or radius < 1e-8:
        radius = 1.0

    if camera_distance is not None:
        d = float(camera_distance)
    else:
        # 直径 2*radius 约等于 fill_ratio * (垂直方向可见世界高度 2*d*tan(half))
        d = radius / (fill_ratio * math.tan(half))

    if znear_override is not None and zfar_override is not None:
        znear = float(znear_override)
        zfar = float(zfar_override)
    else:
        znear = max(0.01, d - radius * 3.0)
        zfar = max(znear + 0.1, d + radius * 6.0, 50.0)

    flags = pyrender.constants.RenderFlags.OFFSCREEN | pyrender.constants.RenderFlags.SKIP_CULL_FACES
    if flat_shading:
        flags |= pyrender.constants.RenderFlags.FLAT

    camera = pyrender.PerspectiveCamera(yfov=yfov, znear=znear, zfar=zfar)
    camera = pyrender.Node(camera=camera, translation=[0, 0.0, d], rotation=[0, 0, 0, 1])
    # 本仓库 pyrender 无 AmbientLight；DirectionalLight 仅沿节点局部 -Z，无 direction 参数。
    # 主光与相机同向 + 相机系内轻微旋转的补光，比单聚光灯更均匀、无锥形衰减。
    light_key = pyrender.DirectionalLight(color=np.ones(3), intensity=1.8)
    light_fill = pyrender.DirectionalLight(color=np.ones(3), intensity=1.3)
    ax = np.deg2rad(28.0)
    ay = np.deg2rad(-22.0)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(ax), -np.sin(ax)],
            [0.0, np.sin(ax), np.cos(ax)],
        ],
        dtype=np.float64,
    )
    Ry = np.array(
        [
            [np.cos(ay), 0.0, np.sin(ay)],
            [0.0, 1.0, 0.0],
            [-np.sin(ay), 0.0, np.cos(ay)],
        ],
        dtype=np.float64,
    )
    fill_rot = np.eye(4, dtype=np.float64)
    fill_rot[:3, :3] = Ry @ Rx

    # 与 render_mesh_video 一致：少量环境光可避免纯直射光下过暗或对比过低
    ambient = np.asarray([0.32, 0.32, 0.32], dtype=np.float64)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=ambient)
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=not flat_shading)
    scene.add(light_key, pose=camera.matrix)
    scene.add(light_fill, pose=camera.matrix @ fill_rot)
    scene.add_node(camera)
    scene.add(mesh, pose=np.eye(4))

    if reuse_cached_renderer:
        if res not in _offscreen_renderer_cache:
            _offscreen_renderer_cache[res] = pyrender.OffscreenRenderer(res, res)
        r = _offscreen_renderer_cache[res]
    else:
        r = pyrender.OffscreenRenderer(res, res)
    color, _ = r.render(scene, flags=flags)
    color = color.astype(np.float32) / 255.
    color = torch.from_numpy(color).permute(2, 0, 1)

    if filename is not None:
        save_image(color, filename)

    return color


def render_mesh_on(verts, faces, vertex_colors=None, res=512):
    verts = to_np(verts)
    faces = to_np(faces)
    if vertex_colors is None:
        vertex_colors = np.ones_like(verts) * [1.0, 0.5, 0.5]
    render = render_mesh(verts, faces, vertex_colors=vertex_colors, flat_shading=False, res=res)
    return render


def vertex_error_colors(
    pred_verts,
    tar_verts,
    err_mode: str = 'l2',
    vmin: float = 0.0,
    vmax: float | None = None,
    # cmap: str = 'hot',
    cmap: str = 'YlOrRd',
    vertex_mass: np.ndarray | None = None,
    mass_gamma: float = 0.5,
) -> np.ndarray:

    p = np.asarray(to_np(pred_verts), dtype=np.float64)
    t = np.asarray(to_np(tar_verts), dtype=np.float64)
    if p.shape != t.shape:
        raise ValueError(f'pred/tar shape mismatch: {p.shape} vs {t.shape}')
    if err_mode == 'l2':
        err = np.linalg.norm(p - t, axis=-1)
    else:
        raise ValueError(f'unknown err_mode: {err_mode!r}')

    if vertex_mass is not None:
        m = np.asarray(to_np(vertex_mass), dtype=np.float64).reshape(-1)
        if m.shape[0] != err.shape[0]:
            raise ValueError(
                f'vertex_mass length {m.shape[0]} != num vertices {err.shape[0]}'
            )
        m = m / (np.mean(m) + 1e-12)
        err = err * (np.power(np.maximum(m, 1e-12), float(mass_gamma)))

    if vmax is None:
        vmax = err.max()
        if vmax < 1e-10:
            vmax = 1.0
    u = (err - vmin) / (vmax - vmin + 1e-12)
    u = np.clip(u, 0.0, 1.0)

    cmap_obj = plt.get_cmap(cmap)
    rgba = cmap_obj(u)  # float (N,4)
    rgba = (np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgba


def render_vertex_error_map(
    pred_verts,
    tar_verts,
    faces,
    mass_src=None,
    res=512,
    mass_gamma: float = 0.5,
    **color_kw,
):
    """
    在预测网格上绘制相对目标网格的逐顶点误差伪彩（默认欧氏距离 + turbo colormap）。

    mass_src: 与 construct_mesh_operators 的 vertex_mass 一致，形状 [V] 或 [B,V]。
    为 None 时不做质量加权。否则将显示误差按 (m / mean(m)) ** mass_gamma 放大，
    质量（通常等价于局部面积权重）较大的顶点在色标上更突出。
    """
    vmass = None
    if mass_src is not None:
        m = np.asarray(to_np(mass_src), dtype=np.float64)
        if m.ndim == 2:
            m = m[0]
        vmass = m.reshape(-1)

    n_pred = np.asarray(to_np(pred_verts), dtype=np.float64).shape[0]
    if vmass is not None and vmass.shape[0] != n_pred:
        raise ValueError(
            f'mass_src len {vmass.shape[0]} != pred vertex count {n_pred}'
        )

    vc = vertex_error_colors(
        pred_verts,
        tar_verts,
        vertex_mass=vmass,
        mass_gamma=mass_gamma,
        **color_kw,
    )
    return render_mesh(
        to_np(pred_verts),
        to_np(faces),
        vertex_colors=vc,
        flat_shading=False,
        res=res,
    )

def render_overlayed_meshes(vert_list, faces_list):
    all_verts = []
    all_faces = []
    vert_offset = 0

    # use different colors for each mesh:
    color_pal = default_colors
    vertex_colors = []

    for i, (verts, faces) in enumerate(zip(vert_list, faces_list)):
        verts_np = to_np(verts)
        faces_np = to_np(faces)
        
        all_verts.append(verts_np)
        # Add faces to the combined list, but offset the indices
        all_faces.append(faces_np + vert_offset)
        vert_offset += verts_np.shape[0]

        # assume color:
        color = to_rgb(color_pal[i % len(color_pal)])
        v_color = 0.6*np.ones_like(verts_np) * color
        vertex_colors.append(v_color)
    
    combined_verts = np.concatenate(all_verts, axis=0)
    combined_faces = np.concatenate(all_faces, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    return render_mesh_on(combined_verts, combined_faces, vertex_colors=vertex_colors)

def add_text(img, caption, coords=(5, 5), color=(0, 0, 0), text_size=20):
    img_pil = F.to_pil_image(img)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        default_font_path = os.path.join(os.path.dirname(PIL.__file__), "fonts", "DejaVuSans.ttf")
        font = ImageFont.truetype(default_font_path, text_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text(coords, caption, fill=color, font=font)
    return F.to_tensor(img_pil)

def image_grid(img_list, fp=None):
    '''
    Exports a stack of images as a large square gallery.
    (or as close to square as possible)

    `img_list`: (N, C, H, W)
    '''
    if isinstance(img_list, list):
        # check if all images have the same shape
        if not all([img.shape == img_list[0].shape for img in img_list]):
            # fix this by resizing all images to the same shape
            max_shape = max([img.shape[-1] for img in img_list])
            img_list = [
                torch.nn.functional.interpolate(img[None], size=(max_shape, max_shape), mode='bilinear', align_corners=False)[0] 
                for img in img_list
            ]

        img_list = torch.stack(img_list, dim=0)
        
    nrows = int(math.sqrt(len(img_list)))
    grid = make_grid(img_list, nrow=nrows)
    if fp is not None:
        save_image(grid, fp)
    return grid
