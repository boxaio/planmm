import os
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as Func
import matplotlib.pyplot as plt
from torchvision.transforms import Normalize, CenterCrop, Resize, Compose
from torchvision.transforms.functional import gaussian_blur

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator

from configs.env_paths import BASE_DIR

separator = Separator(phone='-', word=' ')
backend = EspeakBackend('en-us', words_mismatch='ignore', with_stress=False)

# phonemes to visemes map. this was created using Amazon Polly
# https://docs.aws.amazon.com/polly/latest/dg/polly-dg.pdf


def get_phoneme_to_viseme_map():
    pho2vi = {}
    # pho2vi_counts = {}
    all_vis = []

    p2v = os.path.join(BASE_DIR, "assets/phonemes2visemes.csv")

    with open(p2v) as file:
        lines = file.readlines()
        # for line in lines[2:29]+lines[30:50]:
        for line in lines:
            if line.split(",")[0] in pho2vi:
                if line.split(",")[4].strip() != pho2vi[line.split(",")[0]]:
                    print('error')
            pho2vi[line.split(",")[0]] = line.split(",")[4].strip()

            all_vis.append(line.split(",")[4].strip())
            # pho2vi_counts[line.split(",")[0]] = 0
    return pho2vi, all_vis

pho2vi, all_vis = get_phoneme_to_viseme_map()


def convert_text_to_visemes(text):
    phonemized = backend.phonemize([text], separator=separator)[0]

    text = ""
    for word in phonemized.split(" "):
        visemized = []
        for phoneme in word.split("-"):
            if phoneme == "":
                continue
            try:
                visemized.append(pho2vi[phoneme.strip()])
                if pho2vi[phoneme.strip()] not in all_vis:
                    all_vis.append(pho2vi[phoneme.strip()])
                # pho2vi_counts[phoneme.strip()] += 1
            except:
                print('Count not find', phoneme)
                continue
        text += " " + "".join(visemized)
    return text


def save2avi(filename, data=None, fps=25):
    """save2avi. - function taken from Visual Speech Recognition repository
    args:
        filename: str, the filename to save the video (.avi).
        data: numpy.ndarray, the data to be saved.
        fps: the chosen frames per second.
    """
    assert data is not None, "data is {}".format(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
    writer = cv2.VideoWriter(filename, fourcc, fps, (data[0].shape[1], data[0].shape[0]), 0)
    for frame in data:
        writer.write(frame)
    writer.release()


def predict_text(lipreader, mouth_sequence):
    from external.Visual_Speech_Recognition_for_Multiple_Languages_v1.espnet.asr.asr_utils import add_results_to_json
    
    lipreader.model.eval()
    with torch.no_grad():
        enc_feats, _ = lipreader.model.encoder(mouth_sequence, None)
        enc_feats = enc_feats.squeeze()

        nbest_hyps = lipreader.beam_search(
            x=enc_feats,
            maxlenratio=lipreader.maxlenratio,
            minlenratio=lipreader.minlenratio,
        )
        nbest_hyps = [
            h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), lipreader.nbest)]
        ]

        transcription = add_results_to_json(nbest_hyps, lipreader.char_list)

    return transcription.replace("<eos>", "")


def predict_text_deca(lipreader, mouth_sequence):
    from external.Visual_Speech_Recognition_for_Multiple_Languages.espnet.asr.asr_utils import add_results_to_json
    
    lipreader.model.eval()
    with torch.no_grad():
        enc_feats, _ = lipreader.model.encoder(mouth_sequence, None)
        enc_feats = enc_feats.squeeze(0)

        ys_hat = lipreader.model.ctc.ctc_lo(enc_feats)
        # print(ys_hat)
        ys_hat = ys_hat.argmax(1)
        ys_hat = torch.unique_consecutive(ys_hat, dim=-1)

        ys = [lipreader.model.args.char_list[x] for x in ys_hat if x != 0]

        ys = "".join(ys)
        ys = ys.replace("<space>", " ")

    return ys.replace("<eos>", "")


def extract_mouth(images, mouth_lmks, convert_grayscale=False, padding=5, target_size=(96, 96)):
    """
    images: Tensor of shape (N, C, H, W), pixel values
    mouth_lmks: Tensor of shape (N, L, 2), pixel values 
    """
    mouth_transform = Compose([
        Normalize(0.0, 1.0),
        # CenterCrop((crop_h, crop_w)),
        # CenterCrop((88, 88)),
        # Normalize(mean, std)
    ])
    H, W = images.shape[-2:]

    if mouth_lmks.shape[1] == 68: # FLAME
        lmk_idx_upper = 30
        lmk_idx_lower = 8
        lmk_idx_left = 12
        lmk_idx_right = 4
        # crop the rectangle around the mouth
        y_min = torch.min(mouth_lmks[:, lmk_idx_upper, 1]) - padding
        y_max = torch.max(mouth_lmks[:, lmk_idx_lower, 1]) + padding
        x_min = torch.min(mouth_lmks[:, lmk_idx_right, 0]) - padding
        x_max = torch.max(mouth_lmks[:, lmk_idx_left, 0]) + padding
        
    else:
        # HACK
        pass

    mouth_sequence = []
    mouth_mask = None
    for i, frame in enumerate(images):
        # x_coords = mouth_lmks[i, :, 0]
        # y_coords = mouth_lmks[i, :, 1]
        
        # x_min = torch.min(x_coords) - padding
        # x_max = torch.max(x_coords) + padding
        # y_min = torch.min(y_coords) - padding
        # y_max = torch.max(y_coords) + padding
        
        x_min = torch.clamp(x_min, 0, W-1).int()
        x_max = torch.clamp(x_max, 0, W-1).int()
        y_min = torch.clamp(y_min, 0, H-1).int()
        y_max = torch.clamp(y_max, 0, H-1).int()

        if convert_grayscale:
            img = Func.rgb_to_grayscale(frame)
        else:
            img = frame

        # mouth = img * mouth_mask[None].float()/255.0
        cropped_mouth = img[:, y_min:y_max+1, x_min:x_max+1].float()/255.0
        # cropped_mouth = F.interpolate(
        #     cropped_mouth.unsqueeze(0),  
        #     size=target_size,
        #     mode='bilinear',
        #     align_corners=False
        # ).squeeze(0)  
        # cropped_mouth = Func.resize(
        #     cropped_mouth, target_size, 
        #     interpolation=Func.InterpolationMode.BILINEAR, antialias=True)
        mouth_sequence.append(cropped_mouth)

    mouth_sequence = torch.stack(mouth_sequence)
    
    return mouth_sequence, mouth_transform


def cut_mouth(
    images, landmarks, convert_grayscale=True,
    crop_width: int=96, crop_height: int=96, window_margin: int=12, start_idx: int=48, stop_idx: int=68,
    crop_size=(88, 88), mean: float=0.421, std: float=0.165,
):
    """ function adapted from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages"""

    # ---- transform mouths before going into the lipread network for loss ---- #
    mouth_transform = Compose([
        Normalize(0.0, 1.0),
        CenterCrop(crop_size),
        Normalize(mean, std)]
    )

    mouth_sequence = []

    # landmarks = landmarks * 112 + 112
    for frame_idx, frame in enumerate(images):
        window_margin = min(window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
        smoothed_landmarks = landmarks[frame_idx-window_margin:frame_idx + window_margin + 1].mean(dim=0)
        smoothed_landmarks += landmarks[frame_idx].mean(dim=0) - smoothed_landmarks.mean(dim=0)

        center_x, center_y = torch.mean(smoothed_landmarks[start_idx : stop_idx], dim=0)
        center_x = center_x.round()
        center_y = center_y.round()

        height = crop_height//2
        width = crop_width//2

        threshold = 5

        if convert_grayscale:
            img = Func.rgb_to_grayscale(frame).squeeze()
        else:
            img = frame

        if center_y - height < 0:
            center_y = height
        if center_y - height < 0 - threshold:
            raise Exception('too much bias in height')
        if center_x - width < 0:
            center_x = width
        if center_x - width < 0 - threshold:
            raise Exception('too much bias in width')

        if center_y + height > img.shape[-2]:
            center_y = img.shape[-2] - height
        if center_y + height > img.shape[-2] + threshold:
            raise Exception('too much bias in height')
        if center_x + width > img.shape[-1]:
            center_x = img.shape[-1] - width
        if center_x + width > img.shape[-1] + threshold:
            raise Exception('too much bias in width')

        mouth = img[...,
                    int(center_y-height): int(center_y+height),
                    int(center_x-width): int(center_x+round(width))]
        mouth_sequence.append(mouth)

    mouth_sequence = torch.stack(mouth_sequence, dim=0)

    return mouth_sequence, mouth_transform