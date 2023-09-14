#!/usr/bin/env python3
import logging
import os
from pathlib import Path
from itertools import repeat, chain

from Storage.DataStorage import BaseStorage
from Storage.FinalDetectionStorage import WeightedDecisionStorage
from wolf_utils.analysis_helper import to_xcenter_ycenter
from wolf_utils.ctx_helpers import get_last_variable

logger = logging.getLogger(__name__)

from BaseClass import BaseClass

"""
Exports the data in a way it can be analyzed using for example an app like

https://github.com/rafaelpadilla/review_object_detection_metrics
"""

class ReviewExportClass(BaseClass):
    def __init__(self, run_num, config, *args, **kwargs):
        super().__init__(config=config)
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.get_step_identifier()}")
        logger.info(f"Config: {self.get_module_config()}")

        output_dirs = dict()
        output_dirs["base_path"] = self.get_current_data_dir(ctx)
        output_dirs["detections_by_id"] = os.path.join(output_dirs["base_path"], "detections_by_id")
        output_dirs["detections_by_name"] = os.path.join(output_dirs["base_path"], "detections_by_name")

        for d in output_dirs:
            Path(output_dirs[d]).mkdir(parents=True, exist_ok=True)

        classes_path = get_last_variable(ctx, 'classes.txt')

        bs = BaseStorage(self.get_sqlite_file(ctx))
        wds = WeightedDecisionStorage(self.get_sqlite_file(ctx))

        for image in bs.get_all_images_fullpath():
            logger.info(f"Handling image {image}")

            store_detections_id = []
            store_detections_name = []
            for detection in wds.get_detections_for_image(image):
                # Export in the format: (class_id, x_centre,  y_centre,  width,  height) with relative sizes
                image_meta = bs.get_image_by_fullpath(image)
                w, h = image_meta.width, image_meta.height
                class_numeric = detection.detection_class_numeric
                class_str     = detection.detection_class_str
                class_probability = detection.detection_probability
                x_center, y_center, b_w, b_h = to_xcenter_ycenter(detection.x_min, detection.x_max, detection.y_min, detection.y_max)
                logger.info (f"Storing {class_str} with probability {class_probability}")
                store_detections_id.append((str(class_numeric), class_probability, x_center/w, y_center/h, b_w / w, b_h / h))
                store_detections_name.append((class_str, class_probability, x_center / w, y_center / h, b_w / w, b_h / h))

            if len(store_detections_id):
                txt_fname = os.path.splitext(os.path.split(image)[1])[0] + ".txt"
                txt_path_id = os.path.join(output_dirs["detections_by_id"], txt_fname)
                txt_path_name = os.path.join(output_dirs["detections_by_name"], txt_fname)

                with open(txt_path_id, "w") as f_id, open(txt_path_name, "w") as f_name:
                    f_id.writelines(
                        chain.from_iterable(zip(
                            [" ".join([str(item) for item in items]) for items in store_detections_id],
                            repeat("\n")
                        ))
                    )
                    f_name.writelines(
                        chain.from_iterable(zip(
                            [" ".join([str(item) for item in items]) for items in store_detections_name],
                            repeat("\n")
                        ))
                    )

        ctx["steps"].append({
            "identifier": self.get_step_identifier(),
            "sqlite_file": self.get_sqlite_file(ctx),
            "detections_by_id" : output_dirs["detections_by_id"],
            "detections_by_name": output_dirs["detections_by_name"],
        })

        return True, ctx


if __name__ == "__main__":
    pass
