#!/usr/bin/env python3
import json
import logging
import os
import sys

import cv2
from pathlib import Path

from Storage.BackmappingStorage import BackmappingStorage
from Storage.SimpleLabelStorage import SimpleLabelStorage
from wolf_utils.misc import getter_factory, draw_text


logger = logging.getLogger(__name__)

from BaseClass import BaseClass
from Storage.DataStorage import BaseStorage, SegmentDataStorage
from Storage.DetectionStorage import DetectionStorage



# Create batches based on the image timestamp
class BackmappingClass(BaseClass):
    def __init__(self, run_num, config, *args, **kwargs):
        super().__init__(config=config)
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.get_step_identifier()}")
        logger.info(f"Config: {self.get_module_config()}")

        bs = BaseStorage(self.get_sqlite_file(ctx))
        all_votings = [] # Collect the votings


        # Start the duplicate handler for this instance
        if "duplicates" in self.get_module_config():
            duplicate_getter = getter_factory(
                self.get_module_config()["duplicates"]["dataclass"],
                self.get_module_config()["duplicates"]["getter"],
                self.get_sqlite_file(ctx)
            )
        else:
            logger.info(f"No getter for duplicates.")
            duplicate_getter = None


        # Iterate over all possible inputs. Challenge: Different depths are available.
        for input_source in self.get_module_config()["inputs"]:
            input_dataclass = input_source["dataclass"]
            input_getter = input_source["getter"]
            logger.info(f"Using input_source {input_dataclass} with getter {input_getter} from config.")

            for image in getter_factory(input_dataclass, input_getter, self.get_sqlite_file(ctx))():
                logger.info(f"Image: {image}")
                logger.info(f"class: {image.source_class}.{image.source_getter}")

                # Get the duplicates for each image
                if duplicate_getter is not None:

                    # TODO: Unify names
                    if hasattr(image, "source_fullpath"):
                        path = image.source_fullpath
                    elif hasattr(image, "image_fullpath"):
                        path = image.image_fullpath
                    else:
                        raise ValueError(f"Cannot get path for image type {type(image)}")
                    image_duplicates = duplicate_getter(path)
                    if image_duplicates is not None:
                        logger.info(f"Image has {len(image_duplicates)} duplicates!")
                    else:
                        logger.info(f"Image has no duplicates.")
                        image_duplicates = list()
                else:
                    image_duplicates = list()

                voting = None
                """
                A voting should be in the format:
                    "orig_image_name": db_image.name,
                    "orig_image_fullpath": db_image.fullpath,
                    "x_max": src_images.x_max,
                    "x_min": src_images.x_min,
                    "y_max": src_images.y_max,
                    "y_min": src_images.y_min,
                    "votings": store_votings,
                    "source": type(src_images).__name__,
                """

                if isinstance(image, DetectionStorage.get_class()):
                    # We are a detection -> take directly the votings.
                    voting = DetectionStorage.get_fullscale_voting(image, self.get_sqlite_file(ctx))

                elif isinstance(image, SimpleLabelStorage.get_class()):
                    voting = SimpleLabelStorage.get_fullscale_voting(image, self.get_sqlite_file(ctx))

                else:
                    raise ValueError(f"Unhandled class: {type(image)}")

                all_votings.append(voting)
                for dup in image_duplicates:
                    img = getter_factory(dup.dataclass, dup.getter, self.get_sqlite_file(ctx))(dup.fullpath)[0]


                    if isinstance(img, DetectionStorage.get_class()):
                        # We are a detection -> take directly the votings.
                        dup_voting = DetectionStorage.get_fullscale_voting(img, self.get_sqlite_file(ctx))

                    elif isinstance(img, SimpleLabelStorage.get_class()):
                        dup_voting = SimpleLabelStorage.get_fullscale_voting(img, self.get_sqlite_file(ctx))
                    elif isinstance(img, SegmentDataStorage.get_class()):
                        dup_voting = SegmentDataStorage.get_fullscale_voting(img, self.get_sqlite_file(ctx))
                    else:
                        raise ValueError(f"Unhandled class: {type(img)}")

                    dup_voting["votings"].extend(voting["votings"]) # we get several votings and take the average later on
                    all_votings.append(dup_voting)

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
