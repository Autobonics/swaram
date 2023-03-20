import cv2
import argparse
import os
import string
import time
from gloss_proc import GlossProcess


def main():
    parser = argparse.ArgumentParser(
        description="Cli to Generate data and add to dataset")
    parser.add_argument(
        "-g",
        "--gloss",
        help="Gloss name to add data",
    )
    parser.add_argument(
        "-r", "--require", help="Generate gloss from gloss.txt", action="store_true"
    )
    parser.add_argument(
        "-v", "--vid", help="Add number of videos for each class [default:10]", type=int
    )
    parser.add_argument(
        "-f", "--frame", help="Add number of frames in sequence of each class [default:72]", type=int
    )
    vid_count = parser.parse_args().vid if parser.parse_args().vid else 10
    frame_count = parser.parse_args().frame if parser.parse_args().frame else 72
    if not parser.parse_args().require:
        if parser.parse_args().gloss:
            gloss = parser.parse_args().gloss
        else:
            print("Necessary arguments [ -g or -r ] not supplied\n")
            parser.print_help()
            exit()
        try:
            gp = GlossProcess([gloss], vid_count=vid_count,
                              frame_count=frame_count)
            res = gp.generate()
        except Exception as err:
            print(err)
    else:
        try:
            with open('gloss.txt', 'r') as file:
                glosses = file.readlines()
            gp = GlossProcess(glosses, vid_count=vid_count,
                              frame_count=frame_count)
            gp.generate()
        except Exception as err:
            print(err)


if __name__ == "__main__":
    main()
