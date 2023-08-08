#!/usr/bin/env python3

import os
import glob

from pathlib import Path
import cv2
import imutils
from imutils import contours

import datetime

import logging

from wolf_utils.misc import getter_factory

logger = logging.getLogger(__name__)

from BaseClass import BaseClass
from Storage.DataStorage import SegmentDataStorage

from wolf_utils.ImageHandling import get_avg_image, filter_small_boxes, group_rectangles, get_greycount, extend_boxes


## Does nothing. Is used as kind of a template for other classes
class MOG2Class(BaseClass):
    def __init__(self, run_num):
        super().__init__()
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.getStepIdentifier()}")
        logger.info(f"Config: {self.getModuleConfig()}")
        segment_output_dir = os.path.join(self.getCurrentDataDir(ctx), self.getModuleConfig()["segments_dir"])
        Path(segment_output_dir).mkdir(parents=True, exist_ok=True)

        extra_images_dir = self.getModuleConfig()["extra_dir"]
        if extra_images_dir is not None:
            extra_images_dir = os.path.join(self.getCurrentDataDir(ctx), extra_images_dir)
            Path(extra_images_dir).mkdir(parents=True, exist_ok=True)

        segments_db = SegmentDataStorage(self.getSqliteFile(ctx))
        inputs = self.getModuleConfig()["inputs"]
        if len(inputs) != 1:
            raise ValueError(
                f"Wrong number of inputs. This module is currently working with excactly one input file. You gave {len(inputs)}."
            )

        batches = getter_factory(inputs[0]["dataclass"], inputs[0]["getter"], self.getSqliteFile(ctx))()

        for batch_num, batch in enumerate(batches):
            logger.info(f"Handling batch number {batch_num+1}")
            images_raw = [image.fullpath for image in batch[0].images]
            bgsubtractor = cv2.createBackgroundSubtractorMOG2(
                        history=self.getModuleConfig()["segmentation_detector_history"],
                        varThreshold=self.getModuleConfig()["segmentation_detector_varThreshold"],
                        detectShadows=self.getModuleConfig()["segmentation_detector_detectShadows"],
                        ) #history=3, varThreshold=75, detectShadows=False
            bgsubtractor.apply(get_avg_image(
                    images_raw,
                    self.getModuleConfig()["average_image_percentage"],
                    self.getModuleConfig()["average_image_min_images"]
                    )) # Feed with reference (avg) image

            for image_n, image in enumerate(images_raw):
                image_name = os.path.basename(image).split(".")
                image_name = (".".join(image_name[:-1]), ".".join(image_name[-1:]))

                img_array = cv2.imread(image)

                if img_array is None:
                    logging.warning(f"{image} is not a valid image. Skipping")
                    continue

                min_area = int(img_array.shape[0] * img_array.shape[1] * self.getModuleConfig()["segmentation_min_area"])
                extend_box = self.getModuleConfig()["segmentation_extend_boxes"]
                frameDelta = bgsubtractor.apply(img_array)

                # Check params: https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga17ed9f5d79ae97bd4c7cf18403e1689a
                cnts = cv2.findContours(frameDelta, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
                cnts = imutils.grab_contours(cnts) # Convenience function for openCV changed behaviour of findContours

                if len(cnts) == 0:
                    # No contours found. Next image
                    logging.info("No contours found in image " + str(image))
                    continue

                (cnts, boundingBoxes) = contours.sort_contours(cnts)

                annotated_img = img_array.copy()   # original image
                dots_img = frameDelta.copy() # image from bg subtractor


                for bb_i, bb in enumerate(group_rectangles(filter_small_boxes(extend_boxes(boundingBoxes, extend_box), min_area), 2)):
                    c = get_greycount(bb, dots_img)
                    if c < self.getModuleConfig()["segmentation_grey_limit"]:
                        # Most probably leafs or the like -> ignore
                        continue
                    (x, y, w, h) = bb

                    if min(w, h) < self.getModuleConfig()["segmentation_detector_min_wh"]:
                        # too small boxes
                        continue

                    # Limit to max image size
                    y_min = max(y, 0)
                    y_max = min(y+h, img_array.shape[0])
                    x_min = max(x, 0)
                    x_max = min(x+w, img_array.shape[1])

                    if extra_images_dir:
                        # Store the debug images
                        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(annotated_img, str(int(c)), (x+int(w/2),y+int(h/2)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                        cv2.rectangle(dots_img, (x,y), (x+w, y+h), (255,255,255), 2)
                        cv2.putText(dots_img, str(int(c)), (x+int(w/2),y+int(h/2)), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))

                    cut_image = img_array[y_min:y_max, x_min:x_max]
                    filename = os.path.join(segment_output_dir, image_name[0]+"_cut_"+str(bb_i+1)+"."+image_name[1])
                    cv2.imwrite(filename, cut_image)

                    segments_db.store(
                            image=image,
                            segment_fullpath=filename,
                            y_min=y_min,
                            y_max=y_max,
                            x_min=x_min,
                            x_max=x_max,
                            output_width=cut_image.shape[1],
                            output_height=cut_image.shape[0],
                            creator=__name__,
                            )

                if extra_images_dir:
                    cv2.imwrite(os.path.join(extra_images_dir, image_name[0] + "_debug." + image_name[1]), annotated_img)
                    cv2.imwrite(os.path.join(extra_images_dir, image_name[0] + "_dots." + image_name[1]), dots_img)

        ctx["steps"].append({
                "identifier" : self.getStepIdentifier(),
                "sqlite_file" : self.getSqliteFile(ctx),
                "segment_output_dir" : segment_output_dir,
                "extra_images_dir" : extra_images_dir,
            })

        return True, ctx

if __name__ == "__main__":
    pass
