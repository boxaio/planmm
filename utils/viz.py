import os
import os.path as osp
from typing import Optional
import matplotlib.pyplot as plt
import torch
import numpy as np
import io
import cv2
import time

from torchvision.utils import draw_bounding_boxes, draw_keypoints
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from scipy.spatial.distance import cdist

from utils.mediapipe_landmarks import mediapipe_indices, profile_line_lmks

from PIL import Image
try:
    from PIL.Image import Resampling
    RESAMPLING_METHOD = Resampling.BICUBIC
except ImportError:
    from PIL.Image import BICUBIC
    RESAMPLING_METHOD = BICUBIC

connectivity_face = (
    [(i, i + 1) for i in list(range(0, 16))]
    + [(i, i + 1) for i in list(range(17, 21))]
    + [(i, i + 1) for i in list(range(22, 26))]
    + [(i, i + 1) for i in list(range(27, 30))]
    + [(i, i + 1) for i in list(range(31, 35))]
    + [(i, i + 1) for i in list(range(36, 41))]
    + [(36, 41)]
    + [(i, i + 1) for i in list(range(42, 47))]
    + [(42, 47)]
    + [(i, i + 1) for i in list(range(48, 59))]
    + [(48, 59)]
    + [(i, i + 1) for i in list(range(60, 67))]
    + [(60, 67)]
)


def make_grid_from_opencv_images(images, nrow=12):
    """ Create a grid of images from the list of cv2 images in images"""
    images = np.array(images)
    images = images[..., ::-1]
    images = np.array(images)
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()/255.
    grid = make_grid(images, nrow=nrow)
    return grid

def plot_landmarks_2d(
    img: torch.tensor,
    lmks: torch.tensor,
    connectivity=None,
    colors="white",
    unit=1,
    input_float=False,
):
    if input_float:
        img = (img * 255).byte()

    img = draw_keypoints(
        img,
        lmks,
        connectivity=connectivity,
        colors=colors,
        radius=2 * unit,
        width=2 * unit,
    )

    if input_float:
        img = img.float() / 255
    return img


def plot_mp_landmarks_on_face(
    img: torch.Tensor,  # [3, h, w], uint8
    mp_lmks: torch.Tensor,  # [1, 478, 2], float32
    connectivity: None,
    colors=(255,0,0),
    unit=1,
    input_float=False,
):
    if input_float:
        img = (img * 255).byte()
    
    img = draw_keypoints(
        img,
        mp_lmks,
        connectivity=connectivity,
        colors=colors,
        radius=2 * unit,
        width=2 * unit,
    )  # [3, h, w], uint8

    if input_float:
        img = img.float() / 255
    
    return img


def plot_all_kpts(image, kpts, color='b', radius: int=1):
    if color == 'r':
        c = (0, 0, 255)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    elif color == 'p':
        c = (255, 100, 100)

    image = image.copy()
    kpts = kpts.copy()

    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv2.circle(img=image, center=(int(st[0]), int(st[1])), radius=radius, color=c, thickness=1)

    return image


def get_img_from_fig(fig, w=256, h=256, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.resize(cv2.imdecode(img_arr, 1), (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def merge_views(views):
    grid = []
    for view in views:
        grid.append(np.concatenate(view, axis=2))
    grid = np.concatenate(grid, axis=1)

    # tonemapping
    return to_image(grid)


def to_image(img):
    """
    img: [3, H, W]  numpy array
    """
    img = (img.transpose(1, 2, 0) * 255)[:, :, [2, 1, 0]]
    img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
    return img


def stack_images(images, grid_shape=None, scale=1.0):
    """
    images (list): list of images numpy array, shape [3,h,w] or [1,h,w]
    grid_shape (tuple): grid shape (rows, cols)
    scale (float): scale factor
    """
    processed_images = []
    for img in images:
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]  
            img = img  
        elif img.ndim == 3 and img.shape[0] == 3:
            img = img
        
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).transpose(2,0,1)
        
        processed_images.append(img)
    
    if grid_shape is None:
        n_images = len(processed_images)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        grid_shape = (rows, cols)
    else:
        rows, cols = grid_shape
    
    max_height = max(img.shape[1] for img in processed_images)
    max_width = max(img.shape[2] for img in processed_images)
    
    resized_images = []
    for img in processed_images:
        h, w = img.shape[1:]
        if h < max_height or w < max_width:
            resized = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            resized[:h, :w, :] = to_image(img)
            resized_images.append(resized)
        else:
            resized_images.append(to_image(img))
    
    grid_image = np.zeros((rows * max_height, cols * max_width, 3), dtype=np.uint8)
    
    for i, img in enumerate(resized_images):
        if i >= rows * cols:
            break
        
        row = i // cols
        col = i % cols
        
        start_y = row * max_height
        end_y = start_y + max_height
        start_x = col * max_width
        end_x = start_x + max_width
        
        grid_image[start_y:end_y, start_x:end_x] = img
    
    if scale != 1.0:
        new_width = int(grid_image.shape[1] * scale)
        new_height = int(grid_image.shape[0] * scale)
        grid_image = cv2.resize(grid_image, (new_width, new_height))
    
    return grid_image


def tensor_vis_landmarks(images, landmarks, color='g'):
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()

    for i in range(images.shape[0]):
        image = images[i]
        image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]].copy()
        image = (image * 255)
        predicted_landmark = predicted_landmarks[i]
        image_landmarks = plot_all_kpts(image, predicted_landmark, color)
        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(
        vis_landmarks[:, :, :, [2, 1, 0]].transpose(0, 3, 1, 2)) / 255.  # , dtype=torch.float32)
    return vis_landmarks

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), RESAMPLING_METHOD)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), RESAMPLING_METHOD)
    image_pil.save(image_path)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array, range(0, 1)
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.clamp(0.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def batch_draw_keypoints(images, landmarks, color=(255, 255, 255), radius=1, flip_y=False):
    """
    images: torch.tensor, [B, 3, H, W], uint8
    landmarks: torch.tensor, [B, N, 2], float, [-1.0, 1.0]
    """
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
        landmarks = landmarks.copy()*112 + 112
    if flip_y:
        landmarks[..., 1] = 224 - 1 - landmarks[..., 1]

    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().transpose(0, 2, 3, 1)
        images = (images * 255).astype('uint8')
        images = np.ascontiguousarray(images[..., ::-1])

    plotted_images = []
    for image, landmark in zip(images, landmarks):
        for point in landmark:
            image = cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
        plotted_images.append(image)

    return plotted_images


def show_mesh(vertices, faces):

    import polyscope as ps

    ps.init()
    ps.register_surface_mesh("mesh", vertices, faces, enabled=True)
    ps.show()




def plot_face_mesh_2d_error(
    mesh1, mesh2, point_size=20, cmap='viridis', figsize=(9, 9),
    save_path="face_mesh_error_2d.png", dpi=300,
):
    """
    【正脸2D版】拓扑一致人脸网格误差热力图 + 自动保存高清PNG
    :param mesh1: 原始人脸网格顶点数组, numpy.ndarray, shape=(N,3)
    :param mesh2: 待对比人脸网格顶点数组, numpy.ndarray, shape=(N,3) 拓扑完全一致
    :param point_size: 顶点显示大小，正脸2D推荐20，顶点数>1000可调小至12
    :param cmap: 误差配色映射，蓝→红，默认viridis(最优)，可选jet/coolwarm/plasma
    :param figsize: 画布尺寸，正脸推荐(8,10)，适配人脸长宽比
    :param save_path: 保存PNG的路径+文件名，默认当前目录
    :param dpi: 保存的图片分辨率，300=高清，600=超高清，150=普通
    :return: error_distances: 每个顶点的误差值数组 shape=(N,)
    """
    assert isinstance(mesh1, np.ndarray) and isinstance(mesh2, np.ndarray), "网格必须是numpy数组"
    assert mesh1.shape == mesh2.shape and mesh1.shape[1]==3, "网格必须是(N,3)拓扑一致的数组"
    N = mesh1.shape[0]

    error_distances = cdist(mesh1, mesh2, metric='euclidean').diagonal()
    mean_error = np.mean(error_distances)
    median_error = np.median(error_distances)
    max_error = np.max(error_distances)
    min_error = np.min(error_distances)
    std_error = np.std(error_distances)
    
    print("=" * 60)
    print("✨ 人脸网格顶点误差量化统计 (正脸2D可视化 | 三维欧氏距离) ✨")
    print(f"顶点总数：{N} 个 | 误差单位：坐标单位")
    print(f"平均误差：{mean_error:.4f} | 中位数误差：{median_error:.4f}")
    print(f"最大误差：{max_error:.4f} | 最小误差：{min_error:.4f}")
    print(f"误差标准差：{std_error:.4f} (越小=误差分布越集中)")
    print("=" * 60)

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        mesh1[:, 0], mesh1[:, 1],  
        c=error_distances,        
        cmap=cmap,
        s=point_size,        
        alpha=0.95,           
        edgecolors='none'  
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('', fontsize=26, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    # ax.set_title('人脸网格【正脸】误差热力图 (蓝=极小误差 · 红=极大误差)', fontsize=14, fontweight='bold', pad=20)
    # ax.set_xlabel('人脸左右 (X轴)', fontsize=11)
    # ax.set_ylabel('人脸上下 (Y轴)', fontsize=11)
    ax.grid(False)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)

    return error_distances



class Visualizer():
    def __init__(self, cfg):
        self.cfg = cfg 
        self.name = cfg['exp_name']
        self.img_dir = osp.join(cfg['checkpoints_dir'], cfg['exp_name'], 'results')
        
        if cfg['phase'] != 'test':
            self.writer = SummaryWriter(osp.join(cfg['checkpoints_dir'], cfg['exp_name'], 'logs'))
            # create a logging file to store training losses
            self.log_name = osp.join(cfg['checkpoints_dir'], cfg['exp_name'], 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, 
        vis_dict, total_iters, epoch, dataset='train', save_results=False, count=0, name=None,
        add_image=True,
    ):
        """Display current results on tensorboad; save current results to an HTML file.

        Parameters:
            vis_dict (OrderedDict) - - dictionary of images to display or save
            total_iters (int) -- total iterations
            epoch (int) - - the current epoch
            dataset (str) - - 'train' or 'val' or 'test'
        """
        # if (not add_image) and (not save_results): return
        
        for label, image in vis_dict.items():
            for i in range(image.shape[0]):
                image_numpy = tensor2im(image[i])
                if add_image:
                    self.writer.add_image(
                        label + '%s_%02d'%(dataset, i + count), image_numpy, total_iters, dataformats='HWC',
                    )

                if save_results:
                    save_path = osp.join(self.img_dir, dataset, 'epoch_%s_%06d'%(epoch, total_iters))
                    if not osp.isdir(save_path):
                        os.makedirs(save_path)
                    
                    if label == 'overlap_vis':
                        name = name + '_overlap'
                    if label == 'input_vis':
                        name = name + '_input'

                    if name is not None:
                        img_path = osp.join(save_path, '%s.png' % name)
                    else:
                        img_path = osp.join(save_path, '%s_%03d.png' % (label, i + count))
                    save_image(image_numpy, img_path)


    def plot_current_losses(self, total_iters, losses, dataset='train'):
        for name, value in losses.items():
            self.writer.add_scalar(name + '/%s'%dataset, value, total_iters)

    def print_current_losses(self, epoch, batch_idx, losses, t_comp, phase='train'):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
        """
        # message = '(phase: %s, epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (
        #     phase, epoch, iters, t_comp, t_data)
        
        log_msg = f"{phase.upper()} [Epoch:{epoch}|Batch:{batch_idx} time: {t_comp:.3f}] "
        for k, v in losses.items():
            log_msg += "{}: {:.6f}  ".format(k, v)

        print(log_msg) 
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % log_msg)  

    def save_plot(self, vis, save_path, show_landmarks=True, flip_y=False):
        for key in vis.keys():
            vis[key] = vis[key].detach().cpu()
            
        B = vis['input_img'].shape[0]
        m = vis['rendered_img'].shape[0] // vis['input_img'].shape[0]
        show_lmks_ids = mediapipe_indices + profile_line_lmks
        if show_landmarks:
            original_img_with_landmarks = batch_draw_keypoints(
                vis['input_normals'], vis['gt_lmks_mp'][:,show_lmks_ids,:], color=(0, 255, 0), flip_y=flip_y, # BGR
            )
            original_img_with_landmarks = batch_draw_keypoints(
                original_img_with_landmarks, vis['pred_lmks_mp'][:,show_lmks_ids,:], color=(0, 0, 255), flip_y=flip_y, # BGR
            )
            original_grid = make_grid_from_opencv_images(original_img_with_landmarks, nrow=B)
        else:
            original_img_with_landmarks = vis['input_normals']
            original_grid = make_grid(original_img_with_landmarks, nrow=B)
        
        more_grid = []
        if 'reconstruction_img' in vis.keys():
            more_grid.append(make_grid(vis['reconstruction_img'].detach().cpu(), nrow=B))
            for i in range(m):
                pred_normals = vis['pred_normals'][i*B:(i+1)*B]
                pred_normals = torch.tensor(np.array(pred_normals))
                more_grid.append(make_grid(pred_normals.detach().cpu(), nrow=B))
        else:
            for i in range(m):
                render_img_with_lmks = batch_draw_keypoints(
                    vis['rendered_img'][i*B:(i+1)*B], 
                    vis['pred_lmks_mp'][i*B:(i+1)*B][:,show_lmks_ids,:], 
                    color=(255, 0, 0), flip_y=flip_y, # BGR
                )
                # vis['rendered_img_with_lmks'] = torch.tensor(np.array(render_img_with_lmks)).permute(0,3,1,2)/255.0
                rendered_img_with_lmks = torch.tensor(np.array(render_img_with_lmks)).permute(0,3,1,2)/255.0
                more_grid.append(make_grid(rendered_img_with_lmks.detach().cpu(), nrow=B))

        # Stage-2: deformed mesh RGB (same pred landmarks overlaid for alignment check)
        if 'rendered_img_stage2' in vis.keys():
            render_s2 = batch_draw_keypoints(
                vis['rendered_img_stage2'],
                vis['pred_lmks_mp'][:, show_lmks_ids, :],
                color=(0, 165, 255),
                flip_y=flip_y,
            )
            rendered_s2 = torch.tensor(np.array(render_s2)).permute(0, 3, 1, 2) / 255.0
            more_grid.append(make_grid(rendered_s2.detach().cpu(), nrow=B))
        
        grid = torch.cat([original_grid] + more_grid, dim=1)
        
        grid = grid.permute(1, 2, 0).cpu().numpy()*255.0
        grid = np.clip(grid, 0, 255)
        grid = grid.astype(np.uint8)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cv2.imwrite(save_path, grid)


def save_deformer_stage2_images(
    outputs: dict,
    save_dir: str,
    epoch: int,
    batch_idx: int,
    total_iters: int,
    nrow: Optional[int] = None,
) -> None:
    """
    保存 stage_2 deformer 相关可视化：UV GT / refined / |diff| / uv_disp(minmax) / 屏幕法线 / rgb_stage2。
    需 ``outputs`` 含 ``uv_normal_gt``、``uv_normal_refined``（[B,3,H,W]）。
    """
    if 'uv_normal_gt' not in outputs or 'uv_normal_refined' not in outputs:
        return
    os.makedirs(save_dir, exist_ok=True)
    tag = f"e{epoch:03d}_b{batch_idx:05d}_it{total_iters:08d}"
    gt = outputs['uv_normal_gt'].detach().cpu().float().clamp(0.0, 1.0)
    ref = outputs['uv_normal_refined'].detach().cpu().float().clamp(0.0, 1.0)
    b = gt.shape[0]
    if nrow is None:
        nrow = max(1, min(b, 4))
    diff = (gt - ref).abs()
    save_image(make_grid(gt, nrow=nrow), os.path.join(save_dir, f"{tag}_uv_gt.png"))
    save_image(make_grid(ref, nrow=nrow), os.path.join(save_dir, f"{tag}_uv_refined.png"))
    save_image(make_grid(diff, nrow=nrow), os.path.join(save_dir, f"{tag}_uv_absdiff.png"))
    if 'uv_disp' in outputs:
        disp = outputs['uv_disp'].detach().cpu().float()
        lo = disp.amin(dim=(2, 3), keepdim=True)
        hi = disp.amax(dim=(2, 3), keepdim=True)
        disp_vis = (disp - lo) / (hi - lo + 1e-8)
        save_image(make_grid(disp_vis.clamp(0.0, 1.0), nrow=nrow), os.path.join(save_dir, f"{tag}_uv_disp.png"))
    if 'pred_normals' in outputs:
        pn = outputs['pred_normals'].detach().cpu().float().clamp(0.0, 1.0)
        save_image(make_grid(pn, nrow=nrow), os.path.join(save_dir, f"{tag}_screen_normals.png"))
    if 'rendered_img_stage2' in outputs and outputs['rendered_img_stage2'] is not None:
        r2 = outputs['rendered_img_stage2'].detach().cpu().float().clamp(0.0, 1.0)
        save_image(make_grid(r2, nrow=nrow), os.path.join(save_dir, f"{tag}_rgb_stage2.png"))