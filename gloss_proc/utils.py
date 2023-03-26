import mediapipe as mp
import numpy as np
import pandas as pd
import os
import torch
from typing import List, Tuple
from functools import reduce

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Drawing from mediapipe holistic docs : )

GData = List[Tuple[np.ndarray, str]]


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
        return list(filter(lambda fname: os.path.isdir(os.path.join(gloss_dir, fname)), os.listdir(gloss_dir)))
    else:
        return []


def gdata_dir(gloss_dir: str) -> str:
    data_path = os.path.join(
        os.getcwd(), gloss_dir)
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    return data_path


def get_all_vid(gloss_dir: str, gloss: str) -> List[str]:
    gpath = os.path.join(gdata_dir(gloss_dir), gloss)
    return list(filter(lambda fname: os.path.isfile(
        fname), map(lambda f: os.path.join(gpath, f), os.listdir(gpath))))


def gdata_count(gloss_dir: str, gloss: str) -> int:
    return len(get_all_vid(gloss_dir, gloss))


def get_vid_data(vid_dir: str) -> np.ndarray:
    return pd.read_csv(vid_dir, header=None).to_numpy()


def _get_gdata(gloss_dir: str, gloss: str) -> GData:
    return list(map(lambda v: (get_vid_data(v), gloss), get_all_vid(gloss_dir, gloss)))


def _get_all_gdata(gloss_dir: str) -> List[GData]:
    return [_get_gdata(gloss_dir, gloss)
            for gloss in get_all_gloss(gloss_dir)]
