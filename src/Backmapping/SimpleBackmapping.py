#!/usr/bin/env python3
import logging
import os
from pathlib import Path

from wolf_utils.misc import getter_factory, draw_text

logger = logging.getLogger(__name__)

from BaseClass import BaseClass
from Storage.DataStorage import BaseStorage
from Storage.DataStorage import SegmentDataStorage
from Storage.DetectionStorage import DetectionStorage

import cv2


# Create batches based on the image timestamp
class BackmappingClass(BaseClass):
    def __init__(self, run_num):
        super().__init__()
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.get_step_identifier()}")
        logger.info(f"Config: {self.get_module_config()}")

        bs = BaseStorage(self.get_sqlite_file(ctx))
        all_votings = []

        duplicate_getter = getter_factory(
            self.get_module_config()["duplicates"]["dataclass"],
            self.get_module_config()["duplicates"]["getter"],
            self.get_sqlite_file(ctx)
        )

        for input_source in self.get_module_config()["inputs"]:
            input_dataclass = input_source["dataclass"]
            input_getter = input_source["getter"]

            for image in getter_factory(input_dataclass, input_getter, self.get_sqlite_file(ctx))():
                logger.info(f"class: {image['source_class']}.{image['source_getter']}")

                image_duplicates = duplicate_getter(image['source_fullpath'])
                if image_duplicates is not None:
                    logger.info("Image has duplicates!")
                else:
                    image_duplicates = list()

                src_images_getter = getter_factory(image["source_class"], image["source_getter"],
                                                   self.get_sqlite_file(ctx))
                src_images = src_images_getter(img=image["source_fullpath"])

                if len(src_images) != 1:
                    raise ValueError("Number of images is not 1. Aborting...")

                if isinstance(src_images[0], SegmentDataStorage.get_class()):
                    logger.info("SEGMENT")

                    db_image = bs.get_image_by_id(src_images[0].base_image)
                    all_votings.append({
                        "orig_image_name": db_image.name,
                        "orig_image_fullpath": db_image.fullpath,
                        "x_max": src_images[0].x_max,
                        "x_min": src_images[0].x_min,
                        "y_max": src_images[0].y_max,
                        "y_min": src_images[0].y_min,
                        "votings": image["votings"],
                        "source": SegmentDataStorage.get_class().__name__,
                    })

                    for dup in image_duplicates:
                        src_images = src_images_getter(img=dup)[0]
                        logger.info(f"Adding duplicate image data {src_images}")
                        db_image = bs.get_image_by_id(src_images.base_image)
                        all_votings.append({
                            "orig_image_name": db_image.name,
                            "orig_image_fullpath": db_image.fullpath,
                            "x_max": src_images.x_max,
                            "x_min": src_images.x_min,
                            "y_max": src_images.y_max,
                            "y_min": src_images.y_min,
                            "votings": image["votings"],
                            "source": SegmentDataStorage.get_class().__name__,
                        })

                elif isinstance(src_images[0], DetectionStorage.get_class()):
                    logger.info("DETECTION")
                    src_segment_getter = getter_factory(
                        src_images[0].source_dataclass,
                        src_images[0].source_datagetter,
                        self.get_sqlite_file(ctx)
                    )

                    src_segment = src_segment_getter(img=src_images[0].image_fullpath)

                    logger.info(f"{src_segment}")
                    det_x_min = src_images[0].x_min
                    det_x_max = src_images[0].x_max
                    det_y_min = src_images[0].y_min
                    det_y_max = src_images[0].y_max

                    seg_x_min = src_segment[0].x_min
                    seg_x_max = src_segment[0].x_max
                    seg_y_min = src_segment[0].y_min
                    seg_y_max = src_segment[0].y_max

                    x_min = seg_x_min + det_x_min
                    x_max = seg_x_min + det_x_max
                    y_min = seg_y_min + det_y_min
                    y_max = seg_y_min + det_y_max

                    orig_image = bs.get_image_by_id(src_segment[0].base_image)
                    votings = {str(src_images[0].detection_class_numeric): src_images[0].confidence}

                    all_votings.append({
                        "orig_image_name": orig_image.name,
                        "orig_image_fullpath": orig_image.fullpath,
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                        "votings": votings,
                        "source": DetectionStorage.get_class().__name__,
                    })

                    for dup in image_duplicates:
                        src_images = src_images_getter(img=dup)[0]
                        logger.info(f"Adding duplicate image data {src_images}")
                        src_segment = src_segment_getter(img=src_images.image_fullpath)[0]

                        det_x_min = src_images.x_min
                        det_x_max = src_images.x_max
                        det_y_min = src_images.y_min
                        det_y_max = src_images.y_max

                        seg_x_min = src_segment.x_min
                        seg_x_max = src_segment.x_max
                        seg_y_min = src_segment.y_min
                        seg_y_max = src_segment.y_max

                        x_min = seg_x_min + det_x_min
                        x_max = seg_x_min + det_x_max
                        y_min = seg_y_min + det_y_min
                        y_max = seg_y_min + det_y_max

                        orig_image = bs.get_image_by_id(src_segment.base_image)
                        votings = {str(src_images.detection_class_numeric): src_images.confidence}

                        all_votings.append({
                            "orig_image_name": orig_image.name,
                            "orig_image_fullpath": orig_image.fullpath,
                            "x_min": x_min,
                            "x_max": x_max,
                            "y_min": y_min,
                            "y_max": y_max,
                            "votings": votings,
                            "source": DetectionStorage.get_class().__name__,
                        })

                else:
                    logger.error(f"MISSING {type(src_images[0])}")

        logger.info(f"We have {len(all_votings)} votes. Removing duplicate entries.")
        all_votings = [dict(t) for t in [tuple(d.items()) for d in all_votings]]
        logger.info(f"Now, we have {len(all_votings)}. Creating examples boxes for all classes.")

        output_dir = os.path.join(self.get_current_data_dir(ctx), "boxes")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for orig_img in set([i["orig_image_fullpath"] for i in all_votings]):
            img = None
            f_name = None
            for voting in all_votings:
                if voting["orig_image_fullpath"] == orig_img:
                    if img is None:
                        img = cv2.imread(voting["orig_image_fullpath"]).copy()
                        f_name = voting["orig_image_name"]

                    cv2.rectangle(img, (int(voting["x_min"]), int(voting["y_min"])),
                                  (int(voting["x_max"]), int(voting["y_max"])), (0, 0, 255), 2)
                    w = voting["x_max"] - voting["x_min"]
                    h = voting["y_max"] - voting["y_min"]
                    draw_text(
                        img=img,
                        text=str(voting["votings"]) + voting["source"],
                        pos=(int(voting["x_max"]) - int(w / 2), int(voting["y_max"]) - int(h / 2)),
                    )

            cv2.imwrite(os.path.join(output_dir, f_name), img)

        ctx["steps"].append({
            "identifier": self.get_step_identifier(),
            "sqlite_file": self.get_sqlite_file(ctx),
            "output_dir": output_dir,
        })

        return True, ctx


if __name__ == "__main__":
    pass
