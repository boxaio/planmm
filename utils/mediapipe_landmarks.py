# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de

import numpy as np
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_FACE_OVAL
# from mediapipe.python.solutions.face_mesh_connections import FACEMESH_IRISES
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_IRIS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYEBROW
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_IRIS


# from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION

def keypoints_to_array(keypoint_list):
    return np.unique(np.hstack(([np.array(kp_ids) for kp_ids in keypoint_list])))


def merge_keypoint_ids(keypoint_lists):
    return np.hstack([keypoints_to_array(keypoint_list) for keypoint_list in keypoint_lists])


CONTOUR_LANDMARK_IDS = keypoints_to_array(FACEMESH_FACE_OVAL)
LEFT_EYEBROW_LANDMARK_IDS = keypoints_to_array(FACEMESH_LEFT_EYEBROW)
RIGHT_EYEBROW_LANDMARK_IDS = keypoints_to_array(FACEMESH_RIGHT_EYEBROW)
LEFT_EYE_LANDMARK_IDS = keypoints_to_array(FACEMESH_LEFT_EYE)
RIGHT_EYE_LANDMARK_IDS = keypoints_to_array(FACEMESH_RIGHT_EYE)
NOSE_LANDMARK_IDS = np.array([168, 6, 197, 195, 5, 4, 129, 98, 97, 2, 326, 327, 358])
LIPS_LANDMARK_IDS = keypoints_to_array(FACEMESH_LIPS)
MP_LANDMARKS = np.hstack((LEFT_EYEBROW_LANDMARK_IDS, RIGHT_EYEBROW_LANDMARK_IDS, LEFT_EYE_LANDMARK_IDS, \
                          RIGHT_EYE_LANDMARK_IDS, NOSE_LANDMARK_IDS, LIPS_LANDMARK_IDS))

LEFT_IRIS_LANDMARK_IDS = keypoints_to_array(FACEMESH_LEFT_IRIS)
RIGHT_IRIS_LANDMARK_IDS = keypoints_to_array(FACEMESH_RIGHT_IRIS)

# add one more landmark for the iris
LEFT_IRIS_LANDMARK_IDS = np.concatenate([LEFT_IRIS_LANDMARK_IDS, [473]])
RIGHT_IRIS_LANDMARK_IDS = np.concatenate([RIGHT_IRIS_LANDMARK_IDS, [468]])

EYELIDS_LANDMARK_IDS = np.array([
    [381, 384], [380, 385], [374, 386], [373, 387], [390, 388],
    [153, 158], [145, 159], [144, 160], [163, 161], [7, 246]
])
LOWER_EYELIDS_LANDMARK_IDS = EYELIDS_LANDMARK_IDS[:, 0]
UPPER_EYELIDS_LANDMARK_IDS = EYELIDS_LANDMARK_IDS[:, 1]

# upper part of the eye region
upper_eye_region_lmks_ids = np.array([
    384, 385, 386, 387, 388, 157, 158, 160, 161, 246,
    413, 441, 286, 258, 475, 257, 259, 260, 467, 466,
    159, 247, 359, 33, 130, 113, 173,
])
# lower part of the eye region
lower_eye_region_lmks_ids = np.array([
    381, 380, 374, 390, 249, 153, 145, 144, 163, 7,
    477, 253, 254, 339, 255, 256, 252, 341, 382, 362, 
    226, 163, 472, 463, 25, 110, 154,
])


# forehead landmarks, this is for bangs
FOREHEAD_LANDMARK_IDS = np.array([
    54, 68, 63, 103, 104, 105, 66, 67, 69,
    107, 108, 109, 9, 10, 151, 336, 337, 338, 
    295, 296, 297, 299, 282, 332, 333, 334,
    283, 284, 293, 298,
])

UPPER_LIP_LANDMARK_IDS = np.array(
    [0, 11, 12, 268, 302, 267, 271, 303, 269,
     272, 304, 270, 407, 408, 409, 306, 291, 191,
    38, 72, 37, 41, 73, 39, 42, 74, 40, 183, 184, 185, 61, 76, 
    82, 13, 312,
])
LOWER_LIP_LANDMARK_IDS = np.array(
    [17, 16, 15, 318, 317, 316, 315, 314, 403, 404, 405, 
     319, 320, 321, 325, 307, 375, 308, 292, 77,
     86, 85, 84, 179, 180, 181, 89, 90, 91, 95, 96, 62, 78,
     87, 14, 402,
])

# mouth region landmarks, this is for lipreading
MOUTH_LANDMARK_IDS = np.array([
    57, 186, 92, 165, 167, 164, 393, 391, 322, 410, 287, 
    273, 335, 406, 313, 18, 83, 182, 106, 43,
    212, 214, 202, 210, 204, 211, 194, 32, 201, 208, 200, 199,
    175,  421, 428, 396, 418, 262, 369, 424, 431, 395, 
    422, 430, 394, 432, 434, 216, 436,
    146, 77, 80, 81, 317, 311, 310, 415, 318, 324, 178, 88,
])
MOUTH_LANDMARK_IDS = np.hstack((MOUTH_LANDMARK_IDS, UPPER_LIP_LANDMARK_IDS, LOWER_LIP_LANDMARK_IDS))

# MP_LANDMARKS = np.hstack((LEFT_EYEBROW_LANDMARK_IDS, RIGHT_EYEBROW_LANDMARK_IDS, LEFT_EYE_LANDMARK_IDS, RIGHT_EYE_LANDMARK_IDS, NOSE_LANDMARK_IDS, LIPS_LANDMARK_IDS, LEFT_IRIS_LANDMARK_IDS, RIGHT_IRIS_LANDMARK_IDS))


# key landmark indices that need by deformer
key_landmarks_indices = np.array([
    8, 9, 10, 151, 55, 285, 107, 108, 109, 110, 130,
    336, 337, 338, 65, 66, 67, 69, 22, 23, 24, 25,
    295, 296, 297, 299, 116, 123, 147, 187, 50,
    36, 142, 203, 205, 206, 207, 216, 103, 104, 105,
    252, 253, 254, 255, 256,
    266, 280, 423, 425, 426, 427, 411, 323,
    454, 347, 346, 330, 348, 448, 449, 450, 
    152, 175, 199, 200, 171, 208, 201, 421, 428, 396, 377])



# these are the indices of the mediapipe landmarks that correspond to the 
# mediapipe landmark barycentric coordinates provided by FLAME2020
mediapipe_indices = [
    276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,
    55,  63,  65,  66,  70, 105, 107, 249, 263, 362, 373, 374, 380,
    381, 382, 384, 385, 386, 387, 388, 390, 398, 466,   7,  33, 133,
    144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
    168,   6, 197, 195,   5,   4, 129,  98,  97,   2, 326, 327, 358,
    0,  13,  14,  17,  37,  39,  40,  61,  78,  80,  81,  82,  84,
    87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270, 291,
    308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409,
    415,
]

# flame_fitting_indices = [
#     17, 14, 18, 16, 19, 9, 6, 8, 4, 7, 
#     52, 54, 56, 57, 59, 60, 61, 62, 63,
#     51, 46, 44, 50, 41, 39, 34, 29, 31, 21, 23, 26,
#     72, 70, 69, 65, 87, 88, 90, 102, 95, 68, 77, 84, 
#     73, 
# ]

flame_fitting_indices = [
    70,  63, 105,  66, 107, 336, 296, 334, 293, 300, 168, 197,   5,
    4,  98,  97,   2, 326, 327, 246, 159, 157, 173, 153, 144, 398,
    385, 387, 263, 373, 381,  61,  39,  37,   0, 267, 269, 291, 405,
    314,  17,  84, 181,  78, 38, 12, 268, 308, 316, 15, 86,
]

profile_line_lmks = [
    136, 148, 149, 150, 152, 176, 365, 377, 378, 379, 400,
    58, 172, 132, 93, 234, 127, 162, 
    288, 401, 366, 454, 356, 389,
    21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 
]

def get_idx(index):
    idx = []
    for i, j in enumerate(MP_LANDMARKS):
        if j in index:
            idx.append(i)
    return idx
