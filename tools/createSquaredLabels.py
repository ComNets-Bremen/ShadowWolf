#!/usr/bin/env python3

"""
Create squared boxes around labels with a given resolution

Jens Dede, ComNets, Uni Bremen, 2023, jd@comnets.uni-bremen.de

Idea:
    - We have manually labelled (complete) images
    - We would like to split the image into subimages with a given minimum
      (squared) size
    - This cut the original image and adapt the labels accordingly

Known issues:
    - Depending on the labels, the boxes might not 100% fit to the new patches
    - Manual checking of the labels is recommended
"""

import os
import sys
import glob
import argparse
import datetime
from pathlib import Path
import cv2
import numpy as np
import copy
import itertools
import shutil

# Order of list of box positions
#[xmin, ymin, xmax, ymax]

def isCompletelyInsideBox(innerbox, outerbox):
    # Check if a box is completely inside another box
    return outerbox[0] <= innerbox[0] <= innerbox[2] <= outerbox[2] and\
            outerbox[1] <= innerbox[1] <= innerbox[3] <= outerbox[3]

def intersect(box1, box2):
    # Check if a box is (partly) intersected
    return (box1[0] <= box2[0] <= box1[2] or\
            box1[0] <= box2[2] <= box1[2]) and \
            (box1[1] <= box2[1] <= box1[3] or\
            box1[1] <= box2[3] <= box1[3])


def fitIntoImage(box, width, height):
    # Make sure this box is inside the image itself and shift it if required
    newbox = copy.deepcopy(list(box))
    if newbox[0] < 0:
        newbox[2] += abs(newbox[0])
        newbox[0] = 0

    if newbox[1] < 0:
        newbox[3] += abs(newbox[1])
        newbox[1] = 0

    if newbox[2] > width:
        newbox[0] -= abs(newbox[2] - width)
        newbox[2] = width

    if newbox[3] > height:
        newbox[1] -= abs(newbox[3] - height)
        newbox[3] = height
    return newbox

def cutToBox(box, width, height):
    # Cut the label so it fits into the box
    newbox = copy.deepcopy(list(box))
    newbox[0] = min(max(newbox[0], 0), width)
    newbox[1] = min(max(newbox[1], 0), height)
    newbox[2] = min(max(newbox[2], 0), width)
    newbox[3] = min(max(newbox[3], 0), height)
    return newbox

"""
Start of main app
"""

ap = argparse.ArgumentParser(
        description="combineImages: combines images from two directories into one",
        epilog="Jens Dede, ComNets, Uni Bremen, " + str(datetime.datetime.now().year),
        )
ap.add_argument("--input", help="The input directory", type=str, required=True)
ap.add_argument("--output", help="The output directory", default="output", type=str)
ap.add_argument("--min_size", help="Minimum size of boxes", type=int, default=640)
ap.add_argument("--store_labelled", help="Store labelled images with annotation frames", action="store_true")
args = vars(ap.parse_args())

Path(args["output"]).mkdir(parents=True, exist_ok=True)

# Store the labelled (i.e. with a color frame surrounded detections) ins
# separate files
if args["store_labelled"]:
    Path(os.path.join(args["output"], "labelled")).mkdir(parents=True, exist_ok=True)

# Get all text files in the input directory
files = (f for f in os.listdir(args["input"]) if (os.path.isfile(os.path.join(args["input"], f)) and f.endswith(u".txt")))

# Check if the classes.txt exists. If yes: Copy into output dir
classes_txt = os.path.join(args["input"], "classes.txt")
if os.path.isfile(classes_txt):
    shutil.copy(classes_txt, args["output"])

# Iterate over all txt files
for f in files:
    files_process = glob.glob(os.path.join(args["input"], Path(f).stem+".*"))
    # Each txt file should have a image file connected to this. So we should
    # get two files only differing in the file extension
    if len(files_process) != 2:
        print("Warning: Wrong number of files found:", files_process, "Skipping...")
        continue

    # Split the files according to their format
    file_txt = next((f for f in files_process if f.endswith(u".txt")))
    file_img = next((f for f in files_process if not f.endswith(u".txt")))

    # Load corresponding image and get the absolute size
    img = cv2.imread(file_img)
    (height, width, _) = img.shape

    abs_boxes = [] # Store all boxes with absolute coordinates for the given image

    with open(file_txt, "r") as infile:
        for line in infile:
            if len(line): # If there are detections in this line...
                # Split box information from line
                (cls, box_rel_x_center, box_rel_y_center, box_rel_width, box_rel_height) = line.split()

                # Get absolute boxes in format center x size
                box_abs_x_center = float(box_rel_x_center) * width
                box_abs_y_center = float(box_rel_y_center) * height
                box_abs_width    = float(box_rel_width) * width
                box_abs_height   = float(box_rel_height) * height

                edge_length = max([box_abs_width, box_abs_height, args["min_size"]]) # at least the given size from the params
                edge_length = min([edge_length, width, height]) # but not larger than the image itself

                # Create the bounding box with the least edge length
                s_patch_x_min = int(box_abs_x_center - (edge_length / 2.0))
                s_patch_y_min = int(box_abs_y_center - (edge_length / 2.0))
                s_patch_x_max = int(box_abs_x_center + (edge_length / 2.0))
                s_patch_y_max = int(box_abs_y_center + (edge_length / 2.0))

                outerbox = [s_patch_x_min, s_patch_y_min, s_patch_x_max, s_patch_y_max]
                outerbox = fitIntoImage(outerbox, width, height)

                # Get x_min, y_min, x_max, y_max-format
                abs_boxes.append({
                    "cls" : cls,
                    "labelbox" : (
                        int(box_abs_x_center - (box_abs_width / 2.0)),
                        int(box_abs_y_center - (box_abs_height / 2.0)),
                        int(box_abs_x_center + (box_abs_width / 2.0)),
                        int(box_abs_y_center + (box_abs_height / 2.0)),
                        ),
                    "outerbox" : outerbox,
                    })


    # Remove duplicates from the outer boxes
    outer_boxes = [box["outerbox"] for box in abs_boxes]
    outer_boxes.sort()
    outer_boxes = list(num for num, _ in itertools.groupby(outer_boxes))
    label_boxes = tuple((x["labelbox"], x["cls"]) for x in abs_boxes)

    # Iterate over the outer boxes and check if labels are inside
    for outer_box_i, outer_box in enumerate(outer_boxes):
        cut_image = img.copy()[outer_box[1]:outer_box[3], outer_box[0]:outer_box[2]]
        cut_labelled_image = None
        if args["store_labelled"]:
            cut_labelled_image = cut_image.copy()

        # The output filenames
        filename = Path(file_img).stem + "_cut_"+str(outer_box_i+1)+Path(file_img).suffix
        filename_txt = Path(file_img).stem + "_cut_"+str(outer_box_i+1)+".txt"

        print("Processing:", filename, filename_txt)
        labels = [] # labels for this box
        for label_box in label_boxes:
            if intersect(outer_box, label_box[0]):
                new_label_coordinates = (label_box[0][0] - outer_box[0], label_box[0][1] - outer_box[1], label_box[0][2] - outer_box[0], label_box[0][3] - outer_box[1])
                new_label_coordinates = cutToBox(new_label_coordinates, outer_box[2] - outer_box[0], outer_box[3] - outer_box[1])

                # Convert and normalize to label format
                label_width = new_label_coordinates[2] - new_label_coordinates[0]
                label_height = new_label_coordinates[3] - new_label_coordinates[1]
                x_center = (new_label_coordinates[0] + (label_width/2.0)) / (outer_box[2] - outer_box[0])
                y_center = (new_label_coordinates[1] + (label_height/2.0)) / (outer_box[3] - outer_box[1])
                label_width = label_width / (outer_box[2] - outer_box[0])
                label_height = label_height / (outer_box[3] - outer_box[1])

                labels.append(f"{label_box[1]} {x_center} {y_center} {label_width} {label_height}")

                if cut_labelled_image is not None:
                    cv2.rectangle(cut_labelled_image, new_label_coordinates[:2], new_label_coordinates[2:], (255,255,0), 2)

        if cut_labelled_image is not None:
            cv2.imwrite(os.path.join(args["output"], "labelled", filename), cut_labelled_image)

        cv2.imwrite(os.path.join(args["output"], filename), cut_image)
        with open(os.path.join(args["output"], filename_txt), "w") as f:
            f.writelines("\n".join(labels))

