#!/usr/bin/env python3
import itertools
import os
import glob

import datetime

import logging
from pathlib import Path

import cv2

from Storage.BackmappingStorage import BackmappingStorage
from wolf_utils.analysis_helper import condense_votings, to_xmin_xmax
from wolf_utils.misc import draw_text

logger = logging.getLogger(__name__)

from BaseClass import BaseClass

class WeightedDecisionClass(BaseClass):
    def __init__(self, run_num):
        super().__init__()
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.get_step_identifier()}")
        logger.info(f"Config: {self.get_module_config()}")

        output_dir = os.path.join(self.get_current_data_dir(ctx), "boxes")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        backmapping_storage = BackmappingStorage(self.get_sqlite_file(ctx))
        for backmapping_image in backmapping_storage.get_all_images():
            logger.info(f"Checking image {backmapping_image}")
            bms = backmapping_storage.get_backmappings_for_image(backmapping_image)
            logger.info(f"Got {len(bms)} mappings")

            boxes_list = [(v.get_box_center_wh(), v.get_votings_list()) for v in bms]
            len_before = len(boxes_list)
            logger.info(f"Boxes before condensing: {boxes_list}")

            cv = condense_votings(
                boxes_list,
                self.get_module_config()["iou_threshold"],
                self.get_module_config()["weights"],
            )
            logger.info(f"Boxes after condensing: {cv}")

            logger.info(f"Condensed {len_before - len(cv)} Entries")

            # cv contains a list of all boxes with all (weighted) decisions. Now, select the dominant class:

            major_class_votings = []
            for c in cv:
                box, votings = c
                max_class = max(votings, key=votings.get)
                major_class_votings.append((box, {max_class : votings[max_class]}))

            max_allowed_class = self.get_module_config()["ignore_classes_higher"]

            # major_class_votings contains each box with the major voting. Next: remove classes which are too high
            # (i.e. out of context)

            votings_of_interest = [v for v in major_class_votings if int(next(iter(v[1]))) <= max_allowed_class]

            img = cv2.imread(backmapping_image)
            for v in votings_of_interest:
                box, voting = v
                (x_min, x_max, y_min, y_max) = to_xmin_xmax(box)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
                draw_text(
                    img=img,
                    text=str(voting),
                    pos=(x_max, y_max)
                )

            _, fn = os.path.split(backmapping_image)
            cv2.imwrite(os.path.join(output_dir, fn), img)

        ctx["steps"].append({
            "identifier": self.get_step_identifier(),
            "sqlite_file": self.get_sqlite_file(ctx),
            "output_dir" : output_dir,
        })

        return True, ctx


if __name__ == "__main__":
    pass
