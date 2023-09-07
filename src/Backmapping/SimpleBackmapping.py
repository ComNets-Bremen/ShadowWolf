#!/usr/bin/env python3
import json
import logging
import os
import cv2
from pathlib import Path

from Storage.BackmappingStorage import BackmappingStorage
from wolf_utils.misc import getter_factory, draw_text

logger = logging.getLogger(__name__)

from BaseClass import BaseClass
from Storage.DataStorage import BaseStorage
from Storage.DataStorage import SegmentDataStorage
from Storage.DetectionStorage import DetectionStorage


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

                    store_votings = {SegmentDataStorage.get_class().__name__ : json.loads(image["votings"])}

                    db_image = bs.get_image_by_id(src_images[0].base_image)
                    all_votings.append({
                        "orig_image_name": db_image.name,
                        "orig_image_fullpath": db_image.fullpath,
                        "x_max": src_images[0].x_max,
                        "x_min": src_images[0].x_min,
                        "y_max": src_images[0].y_max,
                        "y_min": src_images[0].y_min,
                        "votings": store_votings,
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
                            "votings": store_votings,
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

                    x_min = src_segment[0].x_min + src_images[0].x_min
                    x_max = src_segment[0].x_min + src_images[0].x_max
                    y_min = src_segment[0].y_min + src_images[0].y_min
                    y_max = src_segment[0].y_min + src_images[0].y_max

                    orig_image = bs.get_image_by_id(src_segment[0].base_image)

                    store_votings = {
                        SegmentDataStorage.get_class().__name__ : json.loads(image["votings"]),
                        DetectionStorage.get_class().__name__   : {str(src_images[0].detection_class_numeric): src_images[0].confidence}
                    }

                    all_votings.append({
                        "orig_image_name": orig_image.name,
                        "orig_image_fullpath": orig_image.fullpath,
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                        "votings": store_votings,
                        "source": DetectionStorage.get_class().__name__,
                    })

                    for dup in image_duplicates:
                        src_images = src_images_getter(img=dup)[0]
                        logger.info(f"Adding duplicate image data {src_images}")
                        src_segment = src_segment_getter(img=src_images.image_fullpath)[0]

                        x_min = src_segment.x_min + src_images.x_min
                        x_max = src_segment.x_min + src_images.x_max
                        y_min = src_segment.y_min + src_images.y_min
                        y_max = src_segment.y_min + src_images.y_max

                        orig_image = bs.get_image_by_id(src_segment.base_image)
                        store_votings = {
                            SegmentDataStorage.get_class().__name__: json.loads(image["votings"]),
                            DetectionStorage.get_class().__name__  : {str(src_images.detection_class_numeric): src_images.confidence}
                        }

                        all_votings.append({
                            "orig_image_name": orig_image.name,
                            "orig_image_fullpath": orig_image.fullpath,
                            "x_min": x_min,
                            "x_max": x_max,
                            "y_min": y_min,
                            "y_max": y_max,
                            "votings": store_votings,
                            "source": DetectionStorage.get_class().__name__,
                        })

                else:
                    logger.error(f"MISSING {type(src_images[0])}")

        logger.info(f"We have {len(all_votings)} votes. Removing duplicate entries.")
        all_votings = [dict(t) for t in [tuple(d.items()) for d in all_votings]]
        logger.info(f"Now, we have {len(all_votings)}. Storing everything to the database...")

        backmapping_storage = BackmappingStorage(self.get_sqlite_file(ctx))
        for voting in all_votings:
            backmapping_storage.store(
                image_fullpath=voting["orig_image_fullpath"],
                image_name=voting["orig_image_name"],
                source=voting["source"],
                votings=voting["votings"],
                x_min=voting["x_min"],
                x_max=voting["x_max"],
                y_min=voting["y_min"],
                y_max=voting["y_max"]
            )

        logger.info("Storing boxes in images...")

        output_dir = os.path.join(self.get_current_data_dir(ctx), "boxes")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for backmapping_image in backmapping_storage.get_all_images():
            img =cv2.imread(backmapping_image)
            for bm in backmapping_storage.get_backmappings_for_image(backmapping_image):
                cv2.rectangle(img, (int(bm.x_min), int(bm.y_min)),
                              (int(bm.x_max), int(bm.y_max)), (0, 0, 255), 2)
                w = bm.x_max - bm.x_min
                h = bm.y_max- bm.y_min
                draw_text(
                    img=img,
                    text=str(bm.votings),
                    pos=(int(bm.x_max) - int(w / 2), int(bm.y_max) - int(h / 2)),
                )

            cv2.imwrite(os.path.join(output_dir, bm.image_name), img)

        logger.info("Done")

        ctx["steps"].append({
            "identifier": self.get_step_identifier(),
            "sqlite_file": self.get_sqlite_file(ctx),
            "output_dir": output_dir,
        })

        return True, ctx


if __name__ == "__main__":
    pass
