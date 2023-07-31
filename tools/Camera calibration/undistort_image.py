#!/usr/bin/env python3

# Based on https://learnopencv.com/understanding-lens-distortion/

import argparse
import cv2
import numpy as np
import os
import json
import datetime
from pathlib import Path

import pickle

def range_type(astr, min=0.0, max=1.0):
    value = float(astr)
    if min<= value <= max:
        return value
    else:
        raise argparse.ArgumentTypeError('value not in range %s-%s'%(min,max))

parser = argparse.ArgumentParser(
                    prog='undistort_image.py',
                    description='Remove distortion from image. Used for testing.',
                    epilog=f"Jens Dede, {datetime.datetime.now().year}")
parser.add_argument('images', nargs='+', type=str, help="The images")
parser.add_argument("--output", type=str, default="output_images_distorted", help="Remove distortions")
parser.add_argument('pickle', type=str, help="The pickle file for further processing.")
parser.add_argument("-a", type=range_type, metavar="[0...1]", default=1, help="Set the alpha value for the type of distortion. Should be between 0 and 1. Default is %(default)s.")
parser.add_argument("-r", action="store_false", help="Use the cv initUndistortRectifyMap instead of the undistort function")

args = parser.parse_args()

if args.output is not None:
    Path(args.output).mkdir(parents=True, exist_ok=True)

with open(args.pickle, "rb") as f:
    params = pickle.load(f)


for image in args.images:
    img = cv2.imread(image)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(params["camera_matrix"], params["dist"], (w,h), args.a, (w,h))

    if not args.a:
        dst = cv2.undistort(img, params["camera_matrix"], params["dist"], None, newcameramtx)
    else:
        mapx, mapy=cv2.initUndistortRectifyMap(params["camera_matrix"], params["dist"], None , newcameramtx, (w,h), 5)
        dst = cv2.remap(img, mapx ,mapy, cv2.INTER_LINEAR)

    if args.output is not None:
        cv2.imwrite(os.path.join(args.output, os.path.split(image)[-1]), dst)



