import cv2
import os
import string
import numpy as np
import pandas as pd
import mediapipe as mp


def gen_classdata(cname, c_path, img_num):
    class_df = []
    for i in range(0, img_num):
        img_path = f"{c_path}\\{i}.png"
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
            with mp.solutions.hands.Hands(
                    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8
            ) as hands:
                try:
                    res = hands.process(img).multi_hand_landmarks[0].landmark
                    res = np.array(
                        list(map(lambda l: [l.x, l.y, l.z], res))).flatten().tolist()
                    res.append(cname)
                    class_df.append(res)
                except Exception as err:
                    print(f"Unable to generate data for image [{cname},{i}] ")

    return class_df


def gen_dataset(img_num):
    print("Generating Dataset  ")
    data_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "img_data")
    res_df = []
    try:
        for char in string.ascii_uppercase:
            res_df += gen_classdata(char,
                                    os.path.join(data_path, char), img_num)
    except Exception as err:
        print("Class img Err :", err)
    df = pd.DataFrame(data=res_df)
    print("Finished generating data \nWriting to file final_data.csv")
    df.to_csv('dataset.csv', index=False, header=False)
