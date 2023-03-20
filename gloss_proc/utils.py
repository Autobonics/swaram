import mediapipe as mp
import numpy as np
import os
from typing import List

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Drawing from mediapipe holistic docs : )


def draw_landmarks(image: np.ndarray, res):
    mp_drawing.draw_landmarks(
        image,
        res.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        res.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        res.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        res.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())


def set_gloss_path(gloss: str, gloss_dir: str) -> str:
    data_path = os.path.join(os.getcwd(), gloss_dir)
    req_dir = os.path.join(data_path, gloss)
    if not os.path.exists(req_dir):
        os.mkdir(req_dir)
    return req_dir


def get_all_gloss(gloss_dir: str) -> List[str]:
    if os.path.exists(gloss_dir):
        return list(filter(lambda fname: os.path.isdir(os.path.join(fname, gloss_dir)), os.listdir(gloss_dir)))
    else:
        return []


def gdata_count(gloss: str, gloss_dir: str) -> int:
    req_dir = set_gloss_path(gloss, gloss_dir)
    return len(list(filter(lambda fname: os.path.isfile(
        os.path.join(fname, req_dir)), os.listdir(req_dir))))


def gdata_dir(gloss_dir: str):
    data_path = os.path.join(
        os.getcwd(), gloss_dir)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
