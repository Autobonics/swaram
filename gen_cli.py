import cv2
import argparse
import os
import string
import time
from gen_data import gen_dataset


def gen_dir(cname):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(dir_path, "phrase_data")
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
    path = os.path.join(dataset_path, cname)
    if not os.path.isdir(path):
        os.mkdir(path)
    # else:
    #     print(f"Path : {path} already exists.Data generated for class {cname}")
    return path


def gen_img(cname, cpath, img_num):
    cap = cv2.VideoCapture(0)
    i = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        img_path = f"{cpath}/{str(i)}.png"
        cv2.imwrite(img_path, image)
        i += 1
        cv2.imshow(f"Class {cname} ", image)
        if cv2.waitKey(1) == ord("q") or i >= img_num:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Cli to Generate images and add to dataset")
    parser.add_argument(
        "-c",
        "--cname",
        help="Class name to add image data,i.e alphabet to add images",
    )
    parser.add_argument(
        "-a", "--all", help="Add images to all class from A-Z and autogenerate dataset", action="store_true"
    )
    parser.add_argument(
        "-n", "--no", help="Add number of images for each class [default:600]", type=int
    )
    parser.add_argument(
        "-d", "--data", help="generate dataset", action="store_true"
    )
    img_num = parser.parse_args().no if parser.parse_args().no else 600
    if parser.parse_args().data:
        gen_dataset(img_num)
        exit()

    if not parser.parse_args().all:
        if parser.parse_args().cname:
            cname = parser.parse_args().cname.upper()
        else:
            print("Necessary arguments [ -c ,-a or -d ] not supplied\n")
            parser.print_help()
            exit()
        try:
            c_path = gen_dir(cname)
            gen_img(cname, c_path, img_num)
        except Exception as err:
            print(err)
    else:
        for cname in string.ascii_uppercase:
            print("Getting images of Class : ", cname)
            try:
                c_path = gen_dir(cname)
                gen_img(cname, c_path, img_num)
                print(f"Collection of images of Class  {cname} successful")
            except Exception as err:
                print(err)
        gen_dataset(img_num)


if __name__ == "__main__":
    main()
