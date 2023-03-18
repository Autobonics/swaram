import cv2
import os
import string
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import List, NamedTuple
from data_preproc.utils import draw_landmarks

Landmark = NamedTuple("Landmark", [('x', float), (
    'y', float), (
        'z', float
)])
landmark = NamedTuple("landmark", [("landmark", List[Landmark])])
Landmarks = NamedTuple(
    'Landmarks', [
        ("right_hand_landmarks", landmark),
        ("left_hand_landmarks", landmark),
        ("pose_landmarks", landmark),
        ("face_landmarks", landmark),
    ])


def proc_landmarks(result_lmks: Landmarks) -> np.ndarray:
    # face_landmarks -> 468 x 3[x,y,z] -> 1404
    # hand_landmarks -> 21 x 2 [left,right] x 3[x,y,z] ->126
    # pose_landmarks -> 33 x 2[x,y, z discarded] -> 66
    # total  landmarks [face+hands+pose] -> 543 x (x,y,z in all landmarks except pose) ->1596

    face_lmks = result_lmks.face_landmarks if result_lmks.face_landmarks else None
    rt_hand_lmks = result_lmks.right_hand_landmarks if result_lmks.right_hand_landmarks else None
    lf_hand_lmks = result_lmks.left_hand_landmarks if result_lmks.left_hand_landmarks else None
    pose_lmks = result_lmks.pose_landmarks if result_lmks.pose_landmarks else None

    def map_lmk(landmarks: landmark, shape: int) -> np.ndarray:
        try:
            # Convert each landmark from [x:x_val,y:y_val,z:z_val] to [x_val,y_val,z_val] list and convert to array
            # and flatten the result getting a numpy vector
            #
            # eg : input => [
            #         { x:0.54, y:0.53, z.0.57},
            #         { x:0.48, y:0.54, z.0.61},
            #         { x:0.22, y:0.39, z.0.77}
            #        ]
            #
            #   output => [0.54,0.53,0.57,0.48,0.54,0.61,0.22,0.39,0.77]
            #
            return np.array(
                list(map(lambda l: [l.x, l.y, l.z], landmarks.landmark))).flatten()
        except Exception:
            # print("MapLmk Error : ", err)
            return np.zeros(shape)
            # all landmarks except pose landmark are passed to mapLmk to generate vector
            # pose landmark was not passed bcos z_val of pose_landmark is discarded
            # the result from map contains 3 numpy vectors  of shape [(1404,),(63,),(63,)]
            # pose landmarks are passed to map_pose function
            # The pose landmark vector is generated with shape (66,)
            # Each landmark is passed with shape bcos zero vector is generated  for each shape in the absence of landmark

    def map_pose(pose_lmks: landmark, shape: int) -> np.ndarray:
        try:
            return np.array(list(map(lambda l: [l.x, l.y], pose_lmks.landmark))).flatten()
        except Exception:
            # print("MapPose err : ", err)
            return np.zeros(shape)

    try:
        res = np.concatenate([
            map_lmk(face_lmks, 1404),
            map_lmk(rt_hand_lmks, 63),
            map_lmk(lf_hand_lmks, 63),
            map_pose(pose_lmks, 66),
        ])
    except Exception:
        res = np.zeros(1596)
        # print("Res error : ", err)
    return res

    # def gen_classdata(cname: str, c_path: str, img_num: int) -> List[list[float]]:
    #     class_df = []
    #     for i in range(0, img_num):
    #         img_path = f"{c_path}\\{i}.png"
    #         if os.path.exists(img_path):
    #             img = cv2.imread(img_path)
    #             img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1)
    #             with mp.solutions.hands.Hands(
    #                     static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8
    #             ) as hands:
    #                 try:
    #                     res = hands.process(img).multi_hand_landmarks[0].landmark
    #                     res = np.array(
    #                         list(map(lambda l: [l.x, l.y, l.z], res))).flatten().tolist()
    #                     res.append(cname)
    #                     class_df.append(res)
    #                 except Exception as err:
    #                     print(f"Unable to generate data for image [{cname},{i}] ")

    #     return class_df

    # def get_all_dirs(data_path: str) -> List[str]:
    #     if os.path.exists(data_path):
    #         return list(filter(lambda fname: os.path.isdir(os.path.join(fname, data_path)), os.listdir(data_path)))
    #     else:
    #         return []

    # def gen_dataset(img_num):
    #     print("Generating Dataset  ")
    #     data_path = os.path.join(os.path.dirname(
    #         os.path.realpath(__file__)), "img_data")
    #     res_df = []
    #     try:
    #         for cname in get_all_dirs(data_path):
    #             res_df += gen_classdata(cname,
    #                                     os.path.join(data_path, cname), img_num)
    #     except Exception as err:
    #         print("Class img Err :", err)
    #     df = pd.DataFrame(data=res_df)
    #     print("Finished generating data \nWriting to file final_data.csv")
    #     df.to_csv('dataset.csv', index=False, header=False)


class GlossProcess():
    def __init__(self, glosses: List[str], frame_count: int, vid_count: int):
        self.glosses = glosses
        self.frame_count = frame_count
        self.vid_count = vid_count

    def __iter__(self):
        for gloss in self.glosses:
            yield self.gen_seq(gloss)

    def __repr__(self):
        return f"""Glosses     : {self.glosses}\n 
                   frame count : {self.frame_count}\n 
                   video count : {self.vid_count}"""

    def gen_seq(self, gloss: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(0)
        result: List[np.ndarray] = []
        i = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_holistic = mp.solutions.holistic
            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as holistic:
                try:
                    res = holistic.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    draw_landmarks(image, res)
                    result.append(proc_landmarks(res))
                    image = cv2.putText(cv2.flip(image, 1), f"frame : {i+1} of [{gloss}]", (10, 35),
                                        cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                except Exception as err:
                    print("Error : ", err)
                cv2.imshow(f"Swaram Data Collection", image)
            if cv2.waitKey(1) == ord("q") or i >= self.frame_count:
                break
        cap.release()
        cv2.destroyAllWindows()
        return result
