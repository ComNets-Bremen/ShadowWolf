#!/usr/bin/env python3

# Based on https://learnopencv.com/camera-calibration-using-opencv/

import argparse
import cv2
import numpy as np
import os
import json
import datetime
from pathlib import Path

import PIL.Image

import pickle

parser = argparse.ArgumentParser(
                    prog='Create_params.py',
                    description='Creates the distortion correction params out of a set of images. The images should contain a checkboard pattern and be from the same camera.',
                    epilog=f"Jens Dede, {datetime.datetime.now().year}")
parser.add_argument('images', nargs='+', type=str, help="The images")
parser.add_argument('--output_images', type=str, default=None, help="Keep the intermediate images in given directory")
parser.add_argument('--pickle', type=str, default=None, help="The pickle output file for further processing.")

args = parser.parse_args()

COMPARE_EXIF_TAGS = ["Make", "Model", "BodySerialNumber"]

def getAllExif(i):
    import PIL.ExifTags
    exif_data_PIL = i._getexif()
    ret = {}

    for k, v in PIL.ExifTags.TAGS.items():
        value = exif_data_PIL.get(k, None)

        if value is not None:
            ret[PIL.ExifTags.TAGS[k]] = value
    return ret

if args.output_images is not None:
    Path(args.output_images).mkdir(parents=True, exist_ok=True)

# Defining the dimensions of checkerboard
CHECKERBOARD = (9,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

first_exif = None # Compare some exif values. Make sure images a suitable

# Extracting path of individual image stored in a given directory
for fname in args.images:
    with PIL.Image.open(fname) as pilImage:
        exif = getAllExif(pilImage)

    if first_exif is None:
        first_exif = exif

    for exif_tag in COMPARE_EXIF_TAGS:
        if first_exif.get(exif_tag, None) != exif.get(exif_tag, None):
            raise ValueError(f"Images from different sources: Tag {exif_tax} is not matching, Aborting")


    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display
    them on the images of checker board
    """
    if ret == True:
        print(f"Found checkboard in image {os.path.split(fname)[-1]}")
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    else:
        print(f"No checkboard found in {os.path.split(fname)[-1]}")

    if args.output_images is not None:
        cv2.imwrite(os.path.join(args.output_images, os.path.split(fname)[-1]), img)


h,w = img.shape[:2]

"""
Performing camera calibration by
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

results = {
        "images" : args.images,
        "camera_matrix" : mtx,
        "dist" : dist,
        "rvecs" : rvecs,
        "tvecs" : tvecs,
        "exif"  : exif,
        "compare_exif_tags" : COMPARE_EXIF_TAGS,
        }

pickle_file = args.pickle

if pickle_file is None:
    pickle_file = "_".join([str(first_exif.get(cet, None)) for cet in COMPARE_EXIF_TAGS]) + ".pickle"


with open(pickle_file, 'wb') as f:
    pickle.dump(results, f)
