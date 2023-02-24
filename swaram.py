import cv2
import pickle
import numpy as np
import pandas as pd
import mediapipe as mp

mp_hands = mp.solutions.hands


def main():
    cap = cv2.VideoCapture(0)
    with open("mlp_model.pkl", "rb") as mod_file:
        model = pickle.load(mod_file)
    cname = [""]
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1)
        with mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.8
        ) as hands:
            try:
                res = hands.process(image).multi_hand_landmarks[0].landmark
                res = pd.DataFrame(data=[np.array(
                    list(map(lambda l: [l.x, l.y, l.z], res))).flatten().tolist()])
                cname = model.predict(res)

            except Exception as err:
                pass
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.flip(image, 1)
        image = cv2.putText(image, cname[0], (150, 250),
                            cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow(f"Swaram ", image)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
