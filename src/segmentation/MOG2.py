"""
Jens Dede, 2023, jd@comnets.uni-bremen.de

Module for preprocessing images. Uses the MOG2 background subtractor to
find changes in a batch of images.

Images are cut into subimages for further processing
"""

import os
from pathlib import Path
import csv
import logging
import cv2

import imutils
from imutils import contours

from wolf_utils.ProgressBar import printProgressBar
from wolf_utils.ImageHandling import get_avg_image, filter_small_boxes, group_rectangles, get_greycount, extend_boxes

class MOG2:
    def __init__(self, data, config, output_dir, args):
        self.data = data             # Data / batches
        self.config = config         # Opened config file
        self.output_dir = output_dir # Where is the output data stored
        self.args = args             # Data from the argument parser

    def get_segments(self):
        segment_output_dir = os.path.join(self.output_dir, "segments")
        Path(segment_output_dir).mkdir(parents=True, exist_ok=True)

        extra_dir = None # Store intermediate images?
        if self.args["extra"]:
            print("Storing extra images")
            extra_dir = os.path.join(segment_output_dir, "extra")
            Path(extra_dir).mkdir(parents=True, exist_ok=True)

        # Store the mapping original image <-> subimage in a csv file
        csv_filename = os.path.join(segment_output_dir, "split-info.csv")

        total_elements = sum(len(l) for l in self.data)

        with open(csv_filename, "w") as csv_file:
            csvwriter = csv.writer(csv_file)
            csvwriter.writerow(("input filename", "y_min", "y_max", "x_min", "x_max", "output filename", "output width", "output height"))
            imgcount = 0 # For progress bar

            print("Analyzing images")
            printProgressBar(0, total_elements, prefix = 'Progress:', suffix = 'Complete', length = 50)
            for batch_num, batch in enumerate(self.data):
                bgsubtractor = cv2.createBackgroundSubtractorMOG2(
                        history=self.config.getint("DEFAULT", "Segmentation_BG_Detector_history"),
                        varThreshold=self.config.getint("DEFAULT", "Segmentation_BG_Detector_varThreshold"),
                        detectShadows=self.config.getboolean("DEFAULT", "Segmentation_BG_Detector_detectShadows"),
                        ) #history=3, varThreshold=75, detectShadows=False 
                bgsubtractor.apply(get_avg_image(
                    batch,
                    self.config.getint("DEFAULT", "average_image_percentage"),
                    self.config.getint("DEFAULT", "average_image_min_images")
                    )) # Feed with reference (avg) image

                for frame_n, image in enumerate(batch):
                    imgcount += 1
                    printProgressBar(imgcount, total_elements, prefix = 'Progress:', suffix = 'Complete', length = 50)
                    # Get plain image name for later storage
                    image_name = os.path.basename(image).split(".")
                    image_name = (".".join(image_name[:-1]), ".".join(image_name[-1:]))

                    img = cv2.imread(image)

                    if img is None:
                        logging.warning(str(image) + " is not a valid image. Skipping")
                        continue

                    min_area = int(img.shape[0] * img.shape[1] * self.config.getfloat("DEFAULT", "Segmentation_Min_Area"))
                    extend_box = self.config.getint("DEFAULT", "Segmentation_Extend_Boxes")
                    frameDelta = bgsubtractor.apply(img)

                    # Check params: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
                    cnts = cv2.findContours(frameDelta, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
                    cnts = imutils.grab_contours(cnts) # Convenience function for openCV changed behaviour of findContours

                    if len(cnts) == 0:
                        # No contours found. Next image
                        logging.info("No contours found in image " + str(image))
                        continue

                    (cnts, boundingBoxes) = contours.sort_contours(cnts)

                    annotated_img = img.copy()
                    dots_img = frameDelta.copy()

                    for bb_i, bb in enumerate(group_rectangles(filter_small_boxes(extend_boxes(boundingBoxes, extend_box), min_area), 2)):
                        c = get_greycount(bb, dots_img)
                        if c < self.config.getint("DEFAULT", "Segmentation_Grey_Limit"):
                            # Most probably leafs or the like -> ignore
                            continue
                        (x, y, w, h) = bb

                        if min(w, h) < self.config.getint("DEFAULT", "Segmentation_BG_Detector_min_wh"):
                            # too small boxes
                            continue

                        # Limit to max image size
                        y_min = max(y, 0)
                        y_max = min(y+h, img.shape[0])
                        x_min = max(x, 0)
                        x_max = min(x+w, img.shape[1])

                        if extra_dir:
                            # Store the debug images
                            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.putText(annotated_img, str(int(c)), (x+int(w/2),y+int(h/2)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                            cv2.rectangle(dots_img, (x,y), (x+w, y+h), (255,255,255), 2)
                            cv2.putText(dots_img, str(int(c)), (x+int(w/2),y+int(h/2)), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))

                        cut_image = img[y_min:y_max, x_min:x_max]
                        filename = os.path.join(segment_output_dir, image_name[0]+"_cut_"+str(bb_i+1)+"."+image_name[1])
                        cv2.imwrite(filename, cut_image)
                        csvwriter.writerow((image, y_min, y_max, x_min, x_max, filename, cut_image.shape[0], cut_image.shape[1]))

                    if extra_dir:
                        cv2.imwrite(os.path.join(extra_dir, image_name[0] + "_debug." + image_name[1]), annotated_img)
                        cv2.imwrite(os.path.join(extra_dir, image_name[0] + "_dots." + image_name[1]), dots_img)

        return csv_filename # return the csv file
