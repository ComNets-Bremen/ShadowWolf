#!/usr/bin/env python3

"""
Combines images to compare for example the detections from different models

Jens Dede, ComNets, University of Bremen, 2023
"""

import os
import sys
import argparse
import glob
import datetime
from pathlib import Path
import cv2
import numpy as np

def draw_text(img, text,
          pos=(0, 0),
          font=cv2.FONT_HERSHEY_PLAIN,
          font_scale=1,
          font_thickness=1,
          text_color=(0, 0, 0),
          text_color_bg=(255, 255, 255)
          ):
    """
    Draws a text with a background to a given position

    Parameters
    ----------
    img             The image to paint on
    text            The text
    pos             Position of the text
    font            The font (defaults to FONT_HERSHEY_PLAIN)
    font_scale      The scale of the font. Defaults to 1
    font_thickness  The thickness of the font. Defaults to 1
    text_color      The text color. Default: Black (0, 0, 0)
    text_color_bg   The background color. Default: White (255, 255. 255)

    Returns         The text size
    -------
    """

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size


def get_blank_image(height, width):
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)
    return blank_image

ap = argparse.ArgumentParser(
        description="combineImages: combines images from two directories into one",
        epilog="Jens Dede, ComNets, Uni Bremen, " + str(datetime.datetime.now().year),
        )
ap.add_argument("inputs", help="The inputs", type=str, nargs="+")
ap.add_argument("--output", help="The output", default="out", type=str)
ap.add_argument("--filetype", help="The filetype", default="jpg", type=str)
ap.add_argument("--horizontal", help="Left and right image", action="store_true")
ap.add_argument("--height", help="The maximum height of the image", type=int, default=1080/2)


args = vars(ap.parse_args())

if len(args["inputs"]) < 2:
    print("At least two inputs are required. Aborting.")
    sys.exit()

inputs = []

for i in args["inputs"]:
    inputs.append({Path(f).stem : f for f in glob.glob(os.path.join(i, f"*.{args['filetype']}"))})

all_files = list(set([item for i in inputs for item in i.keys()]))
print(all_files)

Path(args["output"]).mkdir(parents=True, exist_ok=True)

for f in all_files:
    print_images = []
    for num, inp in enumerate(inputs):
        if f in inp:
            print_images.append({"src": inp[f], "valid" : True })
        else:
            print_images.append({"src": os.path.join(args["inputs"][num], f)+f".{args['filetype']}", "valid" : False})

    concat_img = []
    for i in print_images:
        if i["valid"] is False:
            img = get_blank_image(1080, 1920)
            draw_text(img, "X", pos=(int(1920/2), int(1080/2)), font_scale=10)
        else:
            img = cv2.imread(i["src"])

        height, width, _ = img.shape
        scale = args["height"] / height
        dim = (int(width*scale), int(height*scale))
        img = cv2.resize(img, dim)
        draw_text(img, i["src"])
        concat_img.append(img)

    for i in concat_img:
        output = np.concatenate((concat_img), axis=int(args["horizontal"]))
        cv2.imwrite(os.path.join(args["output"], f+"."+args["filetype"]), output)



