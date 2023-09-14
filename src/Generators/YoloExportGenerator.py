#!/usr/bin/env python3

import logging
import os
import shutil
from pathlib import Path
from itertools import repeat, chain

import cv2

from Storage.DataStorage import BaseStorage
from Storage.FinalDetectionStorage import WeightedDecisionStorage
from wolf_utils.analysis_helper import to_xcenter_ycenter
from wolf_utils.ctx_helpers import get_last_variable
from wolf_utils.misc import draw_text

logger = logging.getLogger(__name__)

from BaseClass import BaseClass

class YoloExportClass(BaseClass):
    def __init__(self, run_num, config, *args, **kwargs):
        super().__init__(config=config)
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.get_step_identifier()}")
        logger.info(f"Config: {self.get_module_config()}")

        output_dirs = dict()
        output_dirs["base_path"] = self.get_current_data_dir(ctx)
        output_dirs["labelled_images"] = os.path.join(output_dirs["base_path"], "labelled_images")
        output_dirs["labels_images"] = os.path.join(output_dirs["base_path"], "labels_images")

        for d in output_dirs:
            Path(output_dirs[d]).mkdir(parents=True, exist_ok=True)

        if get_last_variable(ctx, 'classes.txt') is not None:
            # Make sure we have a classes.txt in the label file (if we have one...)
            shutil.copy(get_last_variable(ctx, 'classes.txt'), os.path.join(output_dirs["labels_images"], "classes.txt"))

        bs = BaseStorage(self.get_sqlite_file(ctx))
        wds = WeightedDecisionStorage(self.get_sqlite_file(ctx))
        detection_threshold = self.get_module_config()["detection_threshold"]

        for image in bs.get_all_images_fullpath():
            logger.info(f"Handling image {image}")
            shutil.copy2(image, os.path.join(output_dirs["labels_images"], os.path.split(image)[1]))

            img_fname = os.path.splitext(os.path.split(image)[1])[0] + ".jpg"
            img_path = os.path.join(output_dirs["labelled_images"], img_fname)
            img = cv2.imread(image)

            store_detections = []
            for detection in wds.get_detections_for_image(image, detection_threshold):
                # Export in the format: (class_id, x_centre,  y_centre,  width,  height) with relative sizes
                image_meta = bs.get_image_by_fullpath(image)
                w, h = image_meta.width, image_meta.height
                class_numeric = detection.detection_class_numeric
                class_str     = detection.detection_class_str
                class_probability = detection.detection_probability
                x_center, y_center, b_w, b_h = to_xcenter_ycenter(detection.x_min, detection.x_max, detection.y_min, detection.y_max)
                logger.info(f"converted {(detection.x_min, detection.x_max, detection.y_min, detection.y_max)} to {(x_center, y_center, b_w, b_h)} and further to {(x_center/w, y_center/h, b_w / w, b_h / h)}")
                logger.info (f"Storing {class_str} with probability {class_probability}")
                store_detections.append((str(class_numeric), x_center/w, y_center/h, b_w / w, b_h / h))
                logger.info(f"STORED DETECTIONS: {store_detections}")

                # Export debug images

                cv2.rectangle(img, (detection.x_min, detection.y_min), (detection.x_max, detection.y_max), (0, 0, 255), 2)
                draw_text(
                    img=img,
                    text=f"{class_str} ({class_numeric}, {round(class_probability*100)}%)",
                    pos=(int(detection.x_min+(b_w/2)), int(detection.y_min+(b_h/2)))
                )

            cv2.imwrite(img_path, img)


            if len(store_detections):
                txt_fname = os.path.splitext(os.path.split(image)[1])[0] + ".txt"
                txt_path = os.path.join(output_dirs["labels_images"], txt_fname)
                with open(txt_path, "w") as f:
                    detection_lines = [" ".join([str(item) for item in items]) for items in store_detections]
                    logger.info(f"Writing detections {detection_lines} to file {txt_path}")
                    f.writelines(chain.from_iterable(zip(detection_lines, repeat("\n"))))



        ctx["steps"].append({
            "identifier": self.get_step_identifier(),
            "sqlite_file": self.get_sqlite_file(ctx),
            "labels_images" : output_dirs["labels_images"],
            "labelled_images" : output_dirs["labelled_images"]
        })

        return True, ctx


if __name__ == "__main__":
    pass
