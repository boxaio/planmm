import torch
import numpy as np
import random
import os.path as osp

from PIL import Image
try:
    from PIL.Image import Resampling
    RESAMPLING_METHOD = Resampling.BICUBIC
except ImportError:
    from PIL.Image import BICUBIC
    RESAMPLING_METHOD = BICUBIC

from skimage import transform as trans
from scipy.io import loadmat, savemat

import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)



# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2 * npts, 8])

    A[0:2 * npts - 1:2, 0:3] = x.transpose()
    A[0:2 * npts - 1:2, 3] = 1

    A[1:2 * npts:2, 4:7] = x.transpose()
    A[1:2 * npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2 * npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    k = np.asarray(k).reshape(-1)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

def resize_n_crop_img(img, lmks, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    img = img.resize((w, h), resample=RESAMPLING_METHOD)
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=RESAMPLING_METHOD)
        mask = mask.crop((left, up, right, below))

    lmks = np.stack([lmks[:, 0] - t[0] + w0/2, lmks[:, 1] - t[1] + h0/2], axis=1) * s
    lmks = lmks - np.reshape(np.array([(w/2 - target_size/2), (h/2 - target_size/2)]), [1, 2])

    return img, lmks, mask


def apply_align_crop_from_trans_params(img, trans_params, target_size=224.):
    """
    Apply the same resize+crop as ``align_img_by_lmks478`` / ``resize_n_crop_img`` using
    ``trans_params`` from the first alignment: ``(w0, h0, s, tx, ty)``.
    Use for a second image (e.g. parsing) so it stays pixel-aligned with the aligned
    normals/RGB. ``img`` must have the same ``(w0, h0)`` as the image used to compute
    ``trans_params``.
    """
    tp = np.asarray(trans_params, dtype=np.float64).reshape(-1)
    if tp.size != 5:
        raise ValueError(f"trans_params must have 5 elements (w0,h0,s,tx,ty), got shape {tp.shape}")
    w0, h0, s, tx, ty = tp.tolist()
    # Match resize_n_crop_img: (w0*s) must be numpy so .astype(np.int32) exists; plain float has no .astype.
    w0 = int(w0)
    h0 = int(h0)
    s = np.float64(s)
    t = np.array([tx, ty], dtype=np.float64)
    w = (w0 * s).astype(np.int32)
    h = (h0 * s).astype(np.int32)
    left = (w / 2 - target_size / 2 + float((t[0] - w0 / 2) * s)).astype(np.int32)
    right = left + target_size
    up = (h / 2 - target_size / 2 + float((h0 / 2 - t[1]) * s)).astype(np.int32)
    below = up + target_size
    img = img.resize((int(w), int(h)), resample=RESAMPLING_METHOD)
    img = img.crop((int(left), int(up), int(right), int(below)))
    return img


def extract_5p(lmks):
    idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lmks5p = np.stack([
        lmks[idx[0], :], 
        np.mean(lmks[idx[[1, 2]], :], 0), 
        np.mean(lmks[idx[[3, 4]], :], 0), 
        lmks[idx[5], :], 
        lmks[idx[6], :],
    ], axis=0)
    lmks5p = lmks5p[[1, 2, 0, 3, 4], :]
    return lmks5p

def extract_5p_from478(lmks):
    idx = np.array([
        5,   # nose
        130, 133, 463, 359,  # eyes 
        185, 409,  # lips
    ])
    lmks5p = np.stack([
        lmks[idx[0], :], 
        np.mean(lmks[idx[[1, 2]], :], 0), 
        np.mean(lmks[idx[[3, 4]], :], 0), 
        lmks[idx[5], :], 
        lmks[idx[6], :],
    ], axis=0)
    lmks5p = lmks5p[[1, 2, 0, 3, 4], :]
    return lmks5p

def align_img_by_lmks478(img, lmks_mp, lmks3D, mask=None, target_size=224., rescale_factor=82.):
    """
    Return:
        transparams   -- numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new       -- PIL.Image  (target_size, target_size, 3)
        lmks_mp_new        -- numpy.array  (478, 2), y direction is opposite to v direction
        mask_new      -- PIL.Image  (target_size, target_size)
    
    Parameters:
        img       -- PIL.Image  (raw_H, raw_W, 3)
        lmksmp    -- numpy.array  (478, 2), y direction is opposite to v direction
        lmks3D    -- numpy.array  (5, 3)
        mask      -- PIL.Image  (raw_H, raw_W, 3)
    """

    w0, h0 = img.size
    lmks_mp[..., 1] = h0 - 1 - lmks_mp[..., 1]

    if lmks_mp.shape[0] != 5:
        lmks5p = extract_5p_from478(lmks_mp)
    else:
        lmks5p = lmks_mp

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lmks5p.transpose(), lmks3D.transpose())
    s = rescale_factor/s  # this s should be about 1.0
    # processing the image
    img_new, lmks_mp_new, mask_new = resize_n_crop_img(img, lmks_mp, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lmks_mp_new, mask_new

def align_img(img, lmks68, lmks_mp, lmks3D, mask=None, target_size=224., rescale_factor=82.):
    """
    Return:
        transparams   -- numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new       -- PIL.Image  (target_size, target_size, 3)
        lm_new        -- numpy.array  (68, 2), y direction is opposite to v direction
        mask_new      -- PIL.Image  (target_size, target_size)
    
    Parameters:
        img       -- PIL.Image  (raw_H, raw_W, 3)
        lmks68    -- numpy.array  (68, 2), y direction is opposite to v direction
        lmks3D    -- numpy.array  (5, 3)
        mask      -- PIL.Image  (raw_H, raw_W, 3)
    """

    w0, h0 = img.size
    lmks68[..., 1] = h0 - 1 - lmks68[..., 1]
    lmks_mp[..., 1] = h0 - 1 - lmks_mp[..., 1]

    if lmks68.shape[0] != 5:
        lmks5p = extract_5p(lmks68)
    else:
        lmks5p = lmks68

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lmks5p.transpose(), lmks3D.transpose())
    s = rescale_factor/s  # this s should be about 1.0

    # processing the image
    img_new, lmks68_new, mask_new = resize_n_crop_img(img, lmks68, t, s, target_size=target_size, mask=mask)
    _, lmks_mp_new, _ = resize_n_crop_img(img, lmks_mp, t, s, target_size=target_size, mask=mask)

    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lmks68_new, lmks_mp_new, mask_new


def estimate_norm(lmks, H):
    # from 
    # https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
    """
    Return:
        trans_m  -- numpy.array  (2, 3)
    Parameters:
        lmks     -- numpy.array  (68, 2) or (478, 2), y direction is opposite to v direction
        H        -- int/float, image height
    """
    if lmks.shape[0] == 68:
        lmks5p = extract_5p(lmks)
    elif lmks.shape[0] == 478:
        lmks5p = extract_5p_from478(lmks)
    else:
        raise ValueError('lmks.shape[0] must be either 68 or 478')
    
    lmks5p[:, -1] = H - 1 - lmks5p[:, -1]
    tform = trans.SimilarityTransform()
    src = np.array([
        [38.2946, 51.6963],  # right eye center
        [73.5318, 51.5014],  # left eye center
        [56.0252, 71.7366],  # nose center
        [41.5493, 92.3655],  # rightmost lip
        [70.7299, 92.2041]   # leftmost lip
    ], dtype=np.float32)

    tform.estimate(lmks5p, src)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)

    return M[0:2, :]


def estimate_norm_torch(lmks_68p, H):
    lmks_68p_ = lmks_68p.detach().cpu().numpy()
    M = []
    for i in range(lmks_68p_.shape[0]):
        M.append(estimate_norm(lmks_68p_[i], H))
    M = torch.tensor(np.array(M), dtype=torch.float32).to(lmks_68p.device)
    return M

def load_lm3d(bfm_folder):

    Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D


def get_affine_mat(opt, size):
    shift_x, shift_y, scale, rot_angle, flip = 0., 0., 1., 0., False
    w, h = size

    if 'shift' in opt['preprocess']:
        shift_pixs = int(opt.shift_pixs)
        shift_x = random.randint(-shift_pixs, shift_pixs)
        shift_y = random.randint(-shift_pixs, shift_pixs)
    if 'scale' in opt['preprocess']:
        scale = 1 + opt.scale_delta * (2 * random.random() - 1)
    if 'rot' in opt['preprocess']:
        rot_angle = opt.rot_angle * (2 * random.random() - 1)
        rot_rad = -rot_angle * np.pi/180
    if 'flip' in opt['preprocess']:
        flip = random.random() > 0.5

    shift_to_origin = np.array([
        [1, 0, -w//2], 
        [0, 1, -h//2], 
        [0, 0, 1],
    ])
    flip_mat = np.array([
        [-1 if flip else 1, 0, 0], 
        [0, 1, 0], 
        [0, 0, 1],
    ])
    shift_mat = np.array([
        [1, 0, shift_x], 
        [0, 1, shift_y], 
        [0, 0, 1],
    ])
    rot_mat = np.array([
        [np.cos(rot_rad), np.sin(rot_rad), 0], 
        [-np.sin(rot_rad), np.cos(rot_rad), 0], 
        [0, 0, 1],
    ])
    scale_mat = np.array([
        [scale, 0, 0], 
        [0, scale, 0], 
        [0, 0, 1],
    ])
    shift_to_center = np.array([
        [1, 0, w//2], 
        [0, 1, h//2], 
        [0, 0, 1],
    ])
    
    affine = shift_to_center @ scale_mat @ rot_mat @ shift_mat @ flip_mat @ shift_to_origin    
    affine_inv = np.linalg.inv(affine)
    return affine, affine_inv, flip

def apply_img_affine(img, affine_inv, method=RESAMPLING_METHOD):
    return img.transform(img.size, Image.AFFINE, data=affine_inv.flatten()[:6], resample=RESAMPLING_METHOD)

def apply_lmks_affine(landmark, affine, flip, size):
    _, h = size
    lmks = landmark.copy()
    lmks[:, 1] = h - 1 - lmks[:, 1]
    lmks = np.concatenate((lmks, np.ones([lmks.shape[0], 1])), -1)
    lmks = lmks @ np.transpose(affine)
    lmks[:, :2] = lmks[:, :2] / lmks[:, 2:]
    lmks = lmks[:, :2]
    lmks[:, 1] = h - 1 - lmks[:, 1]
    if flip:
        lmks_ = lmks.copy()
        lmks_[:17] = lmks[16::-1]
        lmks_[17:22] = lmks[26:21:-1]
        lmks_[22:27] = lmks[21:16:-1]
        lmks_[31:36] = lmks[35:30:-1]
        lmks_[36:40] = lmks[45:41:-1]
        lmks_[40:42] = lmks[47:45:-1]
        lmks_[42:46] = lmks[39:35:-1]
        lmks_[46:48] = lmks[41:39:-1]
        lmks_[48:55] = lmks[54:47:-1]
        lmks_[55:60] = lmks[59:54:-1]
        lmks_[60:65] = lmks[64:59:-1]
        lmks_[65:68] = lmks[67:64:-1]
        lmks = lmks_
    return lmks
