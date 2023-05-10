#!/usr/bin/env python3

import os
import sys
import argparse
import glob
import datetime
from pathlib import Path
import cv2
import numpy as np

ap = argparse.ArgumentParser(
        description="combineImages: combines images from two directories into one",
        epilog="Jens Dede, ComNets, Uni Bremen, " + str(datetime.datetime.now().year),
        )
ap.add_argument("--input1", help="The 1st input", default="in1", type=str)
ap.add_argument("--input2", help="The 2nd input", default="in2", type=str)
ap.add_argument("--output", help="The output", default="out", type=str)
ap.add_argument("--filetype", help="The filetype", default="jpg", type=str)
ap.add_argument("--horizontal", help="Left and right image", action="store_true")

args = vars(ap.parse_args())

input1 = {Path(f).stem : f for f in glob.glob(args["input1"])}
input2 = {Path(f).stem : f for f in glob.glob(args["input2"])}

Path(args["output"]).mkdir(parents=True, exist_ok=True)


for i in input1:
    if i in input2.keys():
        img1 = cv2.imread(input1[i])
        img2 = cv2.imread(input2[i])
        output = np.concatenate((img1, img2), axis=int(args["horizontal"]))
        cv2.imwrite(os.path.join(args["output"], i+"."+args["filetype"]), output)
    else:
        print(f"No matching file for {i}.")
