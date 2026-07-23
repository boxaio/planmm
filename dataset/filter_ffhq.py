


# ------------------------------------------------------------------------
# Filter the FFHQ dataset images to remove those with severe occlusion
# ------------------------------------------------------------------------

import os
import random
from collections import Counter

import cv2
import numpy as np


# Load the FFHQ dataset
ffhq_root = "/media/ubuntu/xb/FFHQ_Dataset/"

parsing_dir = os.path.join(ffhq_root, "parsing_224")

# save the filtered dataset
filtered_filename = "ffhq_filtered_by_occlusion.txt"


label2id = {
    "background": 0,
    "skin": 1,
    "nose": 2,
    "eye_g": 3,
    "l_eye": 4,
    "r_eye": 5,
    "l_brow": 6,
    "r_brow": 7,
    "l_ear": 8,
    "r_ear": 9,
    "mouth": 10,
    "u_lip": 11,
    "l_lip": 12,
    "hair": 13,
    "hat": 14,
    "ear_r": 15,
    "neck_l": 16,
    "neck": 17,
    "cloth": 18
}

id2label = {v: k for k, v in label2id.items()}

# Face parsing labels used for occlusion heuristics
SKIN_LABEL = label2id["skin"]
EYE_G_LABEL = label2id["eye_g"]
L_EYE_LABEL = label2id["l_eye"]
R_EYE_LABEL = label2id["r_eye"]
HAIR_LABEL = label2id["hair"]
CHIN_OCCLUDER_LABELS = {
    label2id["background"],
    label2id["hair"],
    label2id["hat"],
    label2id["cloth"],
}

# Reject if any threshold is exceeded
HAIR_ON_FACE_THRESHOLD = 0.22
CHIN_OCCLUSION_THRESHOLD = 0.52
GLASSES_AREA_THRESHOLD = 0.004
EYES_UNDER_GLASSES_THRESHOLD = 0.0012
SIDE_FACE_EYE_RATIO_THRESHOLD = 0.10
SIDE_FACE_SMALL_EYE_THRESHOLD = 0.00035
SIDE_FACE_LARGE_EYE_THRESHOLD = 0.0010


def compute_occlusion_metrics(parsing: np.ndarray) -> dict | None:
    """Compute parsing-based metrics for occlusion filtering."""
    skin_mask = parsing == SKIN_LABEL
    if skin_mask.sum() < 50:
        return None

    total_pixels = parsing.size
    left_eye_area = (parsing == L_EYE_LABEL).sum() / total_pixels
    right_eye_area = (parsing == R_EYE_LABEL).sum() / total_pixels
    glasses_area = (parsing == EYE_G_LABEL).sum() / total_pixels

    ys, xs = np.where(skin_mask)
    face_x1, face_y1 = xs.min(), ys.min()
    face_x2, face_y2 = xs.max(), ys.max()
    face_patch = parsing[face_y1:face_y2 + 1, face_x1:face_x2 + 1]
    face_area = face_patch.size

    hair_on_face = (face_patch == HAIR_LABEL).sum() / face_area

    chin_height = max(1, int((face_y2 - face_y1 + 1) * 0.28))
    chin_patch = face_patch[-chin_height:, :]
    chin_occlusion = sum((chin_patch == label).sum() for label in CHIN_OCCLUDER_LABELS) / chin_patch.size

    if max(left_eye_area, right_eye_area) < 1e-6:
        eye_area_ratio = 0.0
    else:
        eye_area_ratio = min(left_eye_area, right_eye_area) / max(left_eye_area, right_eye_area)

    return {
        "left_eye_area": left_eye_area,
        "right_eye_area": right_eye_area,
        "glasses_area": glasses_area,
        "eye_area_sum": left_eye_area + right_eye_area,
        "eye_area_ratio": eye_area_ratio,
        "hair_on_face": hair_on_face,
        "chin_occlusion": chin_occlusion,
    }


def get_occlusion_rejection_reasons(metrics: dict) -> list[str]:
    """Return rejection reasons; empty list means the sample is kept."""
    reasons = []

    if (
        metrics["glasses_area"] > GLASSES_AREA_THRESHOLD
        and metrics["eye_area_sum"] < EYES_UNDER_GLASSES_THRESHOLD
    ):
        reasons.append("glasses_occlusion")

    one_eye_visible = (
        metrics["left_eye_area"] < SIDE_FACE_SMALL_EYE_THRESHOLD
        and metrics["right_eye_area"] > SIDE_FACE_LARGE_EYE_THRESHOLD
    ) or (
        metrics["right_eye_area"] < SIDE_FACE_SMALL_EYE_THRESHOLD
        and metrics["left_eye_area"] > SIDE_FACE_LARGE_EYE_THRESHOLD
    )
    if one_eye_visible:
        reasons.append("one_eye_visible")

    side_face = (
        metrics["eye_area_ratio"] < SIDE_FACE_EYE_RATIO_THRESHOLD
        and max(metrics["left_eye_area"], metrics["right_eye_area"]) > SIDE_FACE_LARGE_EYE_THRESHOLD
        and min(metrics["left_eye_area"], metrics["right_eye_area"]) < SIDE_FACE_SMALL_EYE_THRESHOLD
    )
    if side_face:
        reasons.append("extreme_side_face")

    if metrics["hair_on_face"] > HAIR_ON_FACE_THRESHOLD:
        reasons.append("hair_occlusion")

    if metrics["chin_occlusion"] > CHIN_OCCLUSION_THRESHOLD:
        reasons.append("chin_occlusion")

    return reasons


def filter_ffhq_by_occlusion(
    parsing_dir: str,
    output_path: str,
    progress_interval: int = 1000,
) -> list[str]:
    """Filter FFHQ samples by parsing-based occlusion rules and save kept filenames."""
    parsing_files = sorted(
        filename
        for filename in os.listdir(parsing_dir)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not parsing_files:
        raise FileNotFoundError(f"No parsing images found in {parsing_dir}")

    kept_stems: list[str] = []
    rejection_stats: Counter[str] = Counter()
    unreadable_count = 0

    for index, filename in enumerate(parsing_files, start=1):
        parsing_path = os.path.join(parsing_dir, filename)
        parsing = cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE)
        if parsing is None:
            unreadable_count += 1
            continue

        metrics = compute_occlusion_metrics(parsing)
        stem = os.path.splitext(filename)[0]
        if metrics is None:
            rejection_stats["invalid_parsing"] += 1
            continue

        reasons = get_occlusion_rejection_reasons(metrics)
        if reasons:
            for reason in reasons:
                rejection_stats[reason] += 1
            continue

        kept_stems.append(stem)

        if index % progress_interval == 0 or index == len(parsing_files):
            print(
                f"[{index}/{len(parsing_files)}] kept={len(kept_stems)} "
                f"rejected={index - len(kept_stems) - unreadable_count}"
            )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for stem in kept_stems:
            file.write(f"{stem}\n")

    print(f"Saved {len(kept_stems)} filenames to {output_path}")
    print(f"Unreadable parsing images: {unreadable_count}")
    print("Rejection stats:")
    for reason, count in rejection_stats.most_common():
        print(f"  {reason}: {count}")

    return kept_stems


def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


# BGR colors for each parsing label (#RRGGBB in comments for IDE color preview)
label_colors = {
    0: (0, 0, 0),         # #000000 background
    1: (180, 200, 255),   # #ffc8b4 skin
    2: (0, 165, 255),     # #ffa500 nose
    3: (255, 255, 0),     # #00ffff eye_g
    4: (255, 0, 0),       # #0000ff l_eye
    5: (0, 0, 255),       # #ff0000 r_eye
    6: (128, 0, 128),     # #800080 l_brow
    7: (255, 0, 255),     # #ff00ff r_brow
    8: (0, 128, 255),     # #ff8000 l_ear
    9: (255, 128, 0),     # #0080ff r_ear
    10: (0, 255, 255),    # #ffff00 mouth
    11: (0, 200, 100),    # #64c800 u_lip
    12: (100, 200, 0),    # #00c864 l_lip
    13: (80, 80, 80),     # #505050 hair
    14: (200, 100, 50),   # #3264c8 hat
    15: (50, 200, 200),   # #c8c832 ear_r
    16: (150, 150, 0),    # #009696 neck_l
    17: (200, 150, 150),  # #9696c8 neck
    18: (100, 100, 200),  # #c86464 cloth
}


def render_label_color_preview(swatch_width: int=80, row_height: int=28, padding: int=8):
    """Render a color legend for label_colors."""
    rows = len(label_colors)
    width = 420
    height = padding * 2 + row_height * rows
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for i, (label_id, color) in enumerate(sorted(label_colors.items())):
        y1 = padding + i * row_height
        y2 = y1 + row_height - 4
        x1 = padding
        x2 = x1 + swatch_width
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness=-1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)

        name = id2label.get(label_id, str(label_id))
        text = f"{label_id:2d}  {name:<8}  {bgr_to_hex(color)}"
        cv2.putText(
            canvas, text, (x2 + 10, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    return canvas
def show_label_color_preview():
    preview = render_label_color_preview()
    cv2.imshow("label color preview", preview)

def colorize_parsing(parsing: np.ndarray):
    """Map label ids to colors according to label2id."""
    h, w = parsing.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in label_colors.items():
        colored[parsing == label_id] = color
    return colored

def overlay_parsing(image: np.ndarray, colored_parsing: np.ndarray, alpha: float=0.5):
    """Blend colored parsing mask onto the original image."""
    return cv2.addWeighted(image, 1 - alpha, colored_parsing, alpha, 0)

def visualize_random_parsing(parsing_dir: str, images_dir: str | None = None, alpha: float=0.5):
    parsing_files = [f for f in os.listdir(parsing_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not parsing_files:
        raise FileNotFoundError(f"No parsing images found in {parsing_dir}")

    filename = random.choice(parsing_files)
    parsing_path = os.path.join(parsing_dir, filename)
    parsing = cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE)
    if parsing is None:
        raise RuntimeError(f"Failed to read parsing image: {parsing_path}")

    colored_parsing = colorize_parsing(parsing)
    present_labels = sorted(np.unique(parsing).tolist())
    print(f"Selected: {filename}")
    print("Present labels:", ", ".join(f"{id2label[i]}({i})" for i in present_labels if i in id2label))

    if images_dir is not None:
        image_path = os.path.join(images_dir, filename)
        image = cv2.imread(image_path)
        if image is not None:
            overlay = overlay_parsing(image, colored_parsing, alpha=alpha)
            cv2.imshow("original", image)
            cv2.imshow("parsing overlay", overlay)

    cv2.imshow("colored parsing", colored_parsing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    # show_label_color_preview()
    # images_dir = os.path.join(ffhq_root, "images_224")
    # visualize_random_parsing(parsing_dir, images_dir=images_dir)

    output_path = os.path.join(ffhq_root, filtered_filename)
    filter_ffhq_by_occlusion(parsing_dir, output_path)













