#!/usr/bin/env python3

import os
import torch
import time

import cv2

from pathlib import Path

import logging
logger = logging.getLogger(__name__)

from BaseClass import BaseClass
from Storage.DetectionStorage import  DetectionStorage

from wolf_utils.misc import batch, delta_time_format, getter_factory


class YoloDetectionClass(BaseClass):
    """Detect objects using YOLO

        This class is optimized for YOLO-models but should work with everything which can be imported using
        the pytorch hub.
    """
    def __init__(self, run_num):
        super().__init__()
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.get_step_identifier()}")
        logger.info(f"Config: {self.get_module_config()}")

        output_dirs = dict()
        output_dirs["base_path"] = self.get_current_data_dir(ctx)
        output_dirs["labelled_images"] = os.path.join(output_dirs["base_path"], "labelled_images")
        output_dirs["labels_images"] = os.path.join(output_dirs["base_path"], "labels_images")
        output_dirs["cut_to_detection"] = os.path.join(output_dirs["base_path"], "cut_to_detection")

        for d in output_dirs:
            Path(output_dirs[d]).mkdir(parents=True, exist_ok=True)

        inputs = self.get_module_config()["inputs"]
        if len(inputs) != 1:
            raise ValueError(
                f"Wrong number of inputs. This module is currently working with excactly one input file. You gave {len(inputs)}."
                             )

        input_dataclass = inputs[0]["dataclass"]
        input_getter    = inputs[0]["getter"]
        input_images = getter_factory(input_dataclass, input_getter, self.get_sqlite_file(ctx))()

        classes_path = os.path.join(output_dirs["labels_images"], "classes.txt")
        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_file = os.path.join(model_dir, self.get_module_config()["detect_model"])
        logger.info(f"Using model {model_file}. Loading...")

        model = torch.hub.load(
                self.get_module_config()["detect_repository"],
                "custom",
                model_file,
                force_reload = self.get_module_config().get("detect_force_reload", False),
                )

        detection_storage = DetectionStorage(self.get_sqlite_file(ctx), input_dataclass, input_getter)

        total_images = len(input_images)
        handled_images = 0
        start_time_detection = time.time()

        logger.info("Starting detection...")
        for i in batch(input_images, self.get_module_config().get("detect_batchsize", 4)):
            results = model(i)
            classes = results.names
            if not os.path.isfile(classes_path):
                with open(classes_path, "w") as f:
                    for cls in classes:
                        f.writelines(f"{cls} {classes[cls]}\n")

            for image in zip(i, results.xyxy):
                (_, output_filename) = os.path.split(image[0])
                img = cv2.imread(image[0])
                img_w_label = img.copy()
                if img is None:
                    logging.warning(f"Image {image[0]} cannot be loaded. Skipping")
                    continue

                detections_rows = []
                for i, detection in enumerate(image[1].tolist()):
                    (x_min, y_min, x_max, y_max, confidence, cls) = detection
                    w = x_max - x_min
                    h = y_max - y_min
                    cls = int(cls)
                    cv2.rectangle(img_w_label, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,255), 2)
                    cv2.putText(img_w_label, str(int(confidence*100)), (int(x_max)-int(w/2), int(y_max)-int(h/2)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                    cut_to_detection = img.copy()[round(y_min):round(y_max), round(x_min):round(x_max)]
                    f, e = os.path.splitext(output_filename)
                    cut_image = os.path.join(output_dirs["cut_to_detection"], f"{f}_{i}{e}")
                    cv2.imwrite(cut_image, cut_to_detection)
                    logger.info(f"Detection: class {cls} ({classes[cls]}), confidence {(confidence*100.0):3.1f}% in image {output_filename}")
                    detection_storage.store(
                        image[0],
                        cut_image,
                        classes[cls],
                        cls,
                        confidence,
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                    )

                    detections_rows.append(f"{cls} {round(x_min+(w/2.0))} {round(y_min+(h/2.0))} {round(w)} {round(h)}")
                else:
                    logger.info(f"No detections for image {output_filename}")
                cv2.imwrite(os.path.join(output_dirs["labelled_images"], output_filename), img_w_label)
                cv2.imwrite(os.path.join(output_dirs["labels_images"], output_filename), img)
                with open(os.path.join(output_dirs["labels_images"], Path(os.path.split(output_filename)[1]).stem + ".txt"), "w") as f:
                    f.writelines("\n".join(detections_rows))
                handled_images += 1

            # Calculate the remaining time
            time_left = max(((time.time()-start_time_detection)/(handled_images/total_images)) - (time.time() - start_time_detection), 0)
            logger.info(f"Finished image {handled_images} out of {total_images}: {(handled_images/total_images*100.0):3.1f}% done. Estimated time left: {delta_time_format(time_left)}")

        output_dict = dict()
        for key in output_dirs:
            output_dict[key] = output_dirs[key]

        output_dict["identifier"] = self.get_step_identifier()
        output_dict["sqlite_file"] = self.get_sqlite_file(ctx)
        output_dict["classes.txt"] = classes_path
        ctx["steps"].append(output_dict)
        return True, ctx


if __name__ == "__main__":
    pass
