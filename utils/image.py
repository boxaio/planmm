import cv2
import numpy as np
import torch

import os
import stat
import os
import h5py
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


def secure_delete_jpg(img_dir, dry_run=False, secure_erase=False, rm_dir=False):
    """
    Securely delete all JPG files (including hidden files) in the current directory
    
    :param dry_run: Simulation mode (default False)
    :param secure_erase: Secure erase mode (default False)
    """

    # Collect all JPG files (including hidden files)
    jpg_files = []
    for item in img_dir.iterdir():
        if item.is_file() and item.suffix.lower() in {'.jpg', '.jpeg'}:
            jpg_files.append(item)

    deleted = 0
    for file in jpg_files:
        try:
            # Secure erase mode processing
            if secure_erase and not dry_run:
                _secure_erase(file)
                
            # Modify file permissions (handle read-only files)
            if not dry_run:
                file.chmod(stat.S_IWUSR | stat.S_IRUSR)  # Set read-write permissions
                file.unlink()  # Execute deletion
                
            deleted += 1
        except Exception as e:
            print(f"Failed to delete {file.name}: {str(e)}")
    
    if not dry_run and deleted == len(jpg_files) and rm_dir:
        try:
            img_dir.rmdir()  
            print(f"Directory {img_dir.name} has been deleted.")
        except OSError as e:
            print(f"Failed to delete directory: {str(e)}")
            print("Note: Directory may not be empty or permission denied.")

    # Output results
    mode = "Simulation Mode" if dry_run else "Normal Deletion"
    if secure_erase and not dry_run:
        mode = "Secure Erase Mode"
    print(f"\nOperation Complete - {mode}")
    print(f"Files Matched: {len(jpg_files)}")
    print(f"Successfully Deleted: {deleted}")


def _secure_erase(file_path, passes=3):
    try:
        file_size = file_path.stat().st_size
        with open(file_path, "ba+") as f:
            for _ in range(passes):
                f.seek(0)
                f.write(os.urandom(file_size))
            f.truncate()
    except Exception as e:
        raise RuntimeError(f"Secure Erase Failed: {str(e)}")



class HDF5ImageWriter:
    def __init__(self, output_path, target_size=(540, 600), compression="gzip", chunk_size=100):
        """
        初始化HDF5写入器
        :param output_path: 输出HDF5文件路径
        :param target_size: 统一图像尺寸 (height, width)
        :param compression: 压缩算法 (gzip/lzf)
        :param chunk_size: HDF5分块大小（影响IO性能）
        """
        self.output_path = Path(output_path)
        self.target_size = target_size
        self.compression = compression
        self.chunk_size = chunk_size
        
        # 预检查输出路径
        if self.output_path.exists():
            raise FileExistsError("file already exists")

    def process_folder(self, image_folder, max_workers=4):
        """
        处理整个图片文件夹
        :param image_folder: 包含JPG的文件夹路径
        :param max_workers: 并行工作线程数
        """
        image_folder = Path(image_folder)
        if not image_folder.is_dir():
            raise NotADirectoryError("invalid image folder path")

        image_paths = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.jpeg"))
        image_paths = [p for p in image_paths if p.is_file()]

        sample_img = self._load_image(image_paths[0])
        self.channels = sample_img.shape[-1]
        
        # create HDF5 file
        with h5py.File(self.output_path, "w") as h5_file:
            images_ds = h5_file.create_dataset(
                "images",
                shape=(0, *self.target_size, self.channels),
                maxshape=(None, *self.target_size, self.channels),
                chunks=(self.chunk_size, *self.target_size, self.channels),
                compression=self.compression,
                dtype=np.uint8
            )
            
            # create meta data
            meta_ds = h5_file.create_dataset(
                "metadata",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.special_dtype(vlen=str)
            )

            # parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for batch in self._batch_generator(image_paths, batch_size=500):
                    futures.append(executor.submit(self._process_batch, batch))

                with tqdm(total=len(image_paths), desc="processing hdf5") as pbar:
                    for future in futures:
                        batch_images, batch_meta = future.result()
                        self._write_batch(h5_file, images_ds, meta_ds, batch_images, batch_meta)
                        pbar.update(len(batch_images))

            # file attributes
            h5_file.attrs["total_images"] = images_ds.shape[0]
            h5_file.attrs["image_size"] = self.target_size
            h5_file.attrs["source_folder"] = str(image_folder)

    def _batch_generator(self, items, batch_size=100):
        """生成批量处理的数据块"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def _process_batch(self, batch_paths):
        """处理单批图片"""
        batch_images = []
        batch_meta = []
        
        for path in batch_paths:
            try:
                img = self._load_image(path)
                batch_images.append(img)
                batch_meta.append(f"{path.name}|{path.stat().st_size}")
            except Exception as e:
                print(f"跳过文件 {path.name}: {str(e)}")
                continue
                
        return np.array(batch_images), batch_meta

    def _load_image(self, path):
        with Image.open(path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if img.size != self.target_size[::-1]:  # PIL使用(width, height)
                img = img.resize(self.target_size[::-1], Image.Resampling.LANCZOS)
            
            return np.array(img)

    def _write_batch(self, h5_file, images_ds, meta_ds, batch_images, batch_meta):
        if len(batch_images) == 0:
            return

        new_size = images_ds.shape[0] + len(batch_images)
        images_ds.resize(new_size, axis=0)
        meta_ds.resize(new_size, axis=0)

        images_ds[-len(batch_images):] = batch_images
        meta_ds[-len(batch_meta):] = batch_meta


class HDF5ImageReader:
    def __init__(self, hdf5_path):
        """
        初始化HDF5读取器
        :param hdf5_path: HDF5文件路径
        """
        self.hdf5_path = hdf5_path
        self.file = None
        self.images_ds = None
        self.meta_ds = None
        
    def __enter__(self):
        """支持上下文管理器"""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """自动关闭文件"""
        self.close()
        
    def open(self):
        """打开HDF5文件"""
        self.file = h5py.File(self.hdf5_path, 'r')
        self.images_ds = self.file['images']
        self.meta_ds = self.file['metadata']
        
    def close(self):
        """关闭HDF5文件"""
        if self.file:
            self.file.close()
            self.file = None
            self.images_ds = None
            self.meta_ds = None
            
    def get_attributes(self):
        """获取文件全局属性"""
        return dict(self.file.attrs)
    
    def get_total_images(self):
        """获取图像总数"""
        return self.images_ds.shape[0]
    
    def get_image(self, index):
        """
        按索引获取单张图像
        :param index: 图像索引 (0-based)
        :return: (图像数组, 元数据字符串)
        """
        return self.images_ds[index], self.meta_ds[index].decode('utf-8')
    
    def get_batch(self, start_idx, end_idx):
        """
        批量获取图像 (高效读取连续区块)
        :param start_idx: 起始索引 (包含)
        :param end_idx: 结束索引 (不包含)
        :return: (图像数组, 元数据列表)
        """
        images = self.images_ds[start_idx:end_idx]
        meta = [m.decode('utf-8') for m in self.meta_ds[start_idx:end_idx]]
        return images, meta
    
    def iterate_images(self, batch_size=100):
        """
        迭代器：分批生成图像数据
        :param batch_size: 每批数量 (默认100)
        """
        total = self.get_total_images()
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            yield self.get_batch(start, end)



def create_mask(landmarks, shape):
    if isinstance(landmarks, np.ndarray):
        landmarks = landmarks.astype(np.int32)[...,:2]
        hull = cv2.convexHull(landmarks)
        mask = np.ones(shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 0)
    elif isinstance(landmarks, torch.Tensor):
        if landmarks.is_cuda:
            landmarks = landmarks.detach().cpu()
        landmarks = landmarks.to(torch.int32)
        hull = cv2.convexHull(landmarks.numpy())
        mask = torch.ones(shape, device=landmarks.device, dtype=torch.uint8)
        cv2.fillConvexPoly(mask.numpy(), hull, 0)

    return mask


def get_bbox(image, lmks, bb_scale=2.0):
    h, w, c = image.shape
    lmks = lmks.astype(np.int32)
    x_min, x_max, y_min, y_max = np.min(lmks[:, 0]), np.max(lmks[:, 0]), np.min(lmks[:, 1]), np.max(lmks[:, 1])
    x_center, y_center = int((x_max + x_min) / 2.0), int((y_max + y_min) / 2.0)
    size = int(bb_scale * 2 * max(x_center - x_min, y_center - y_min))
    xb_min, xb_max, yb_min, yb_max = max(x_center - size // 2, 0), min(x_center + size // 2, w - 1), \
        max(y_center - size // 2, 0), min(y_center + size // 2, h - 1)

    yb_max = min(yb_max, h - 1)
    xb_max = min(xb_max, w - 1)
    yb_min = max(yb_min, 0)
    xb_min = max(xb_min, 0)

    if (xb_max - xb_min) % 2 != 0:
        xb_min += 1

    if (yb_max - yb_min) % 2 != 0:
        yb_min += 1

    return np.array([xb_min, xb_max, yb_min, yb_max])


def crop_image(image, x_min, y_min, x_max, y_max):
    return image[max(y_min, 0):min(y_max, image.shape[0] - 1), max(x_min, 0):min(x_max, image.shape[1] - 1), :]


def squarefiy(image, size=512):
    h, w, c = image.shape
    if w != h:
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        image = np.pad(image, [(vp, vp), (hp, hp), (0, 0)], mode='constant')

    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        input_image = torch.clamp(input_image, -1.0, 1.0)
        image_tensor = input_image.data
    else:
        return input_image.reshape(3, 512, 512).transpose()
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def crop_image_bbox(image, bbox):
    xb_min = bbox[0]
    xb_max = bbox[1]
    yb_min = bbox[2]
    yb_max = bbox[3]
    cropped = crop_image(image, xb_min, yb_min, xb_max, yb_max)
    return cropped


def get_gaussian_pyramid(levels, input, kernel_size, sigma):
    pyramid = []
    images = input.clone()
    for k, level in enumerate(reversed(levels)):
        image_size, iters = level
        size = [int(image_size[0]), int(image_size[1])]
        images = F.interpolate(images, size, mode='bilinear', align_corners=False)
        images = gaussian_blur(images, [kernel_size, kernel_size], sigma=[sigma, sigma] if sigma is not None else None)
        pyramid.append((images, iters, size, image_size))

    return list(reversed(pyramid))

def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)

def get_aspect_ratio(images):
    h, w = images.shape[2:4]
    ratio = w / h
    if ratio > 1.0:
        aspect_ratio = torch.tensor([1. / ratio, 1.0]).float().cuda()[None]
    else:
        aspect_ratio = torch.tensor([1.0, ratio]).float().cuda()[None]
    return aspect_ratio
