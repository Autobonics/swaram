import cv2
import pickle
import numpy as np
import pandas as pd
import mediapipe as mp
from data_preproc import proc_landmarks
from data_preproc.utils import draw_landmarks

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:
            try:
                res = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                draw_landmarks(image, res)
                res_vec = proc_landmarks(res)
            except Exception as err:
                print("Error : ", err)
            cv2.imshow(f"Swaram ", cv2.flip(image, 1))
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
