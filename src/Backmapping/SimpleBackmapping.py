#!/usr/bin/env python3
import json
import logging
import os
import cv2
from pathlib import Path

from Storage.BackmappingStorage import BackmappingStorage
from Storage.SimpleLabelStorage import SimpleLabelStorage
from wolf_utils.misc import getter_factory, draw_text, find_other_sources

logger = logging.getLogger(__name__)

from BaseClass import BaseClass
from Storage.DataStorage import BaseStorage
from Storage.DataStorage import SegmentDataStorage
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
            duplicate_getter = None


        # Iterate over all possible inputs. Challenge: Different depths are available.
        for input_source in self.get_module_config()["inputs"]:
            input_dataclass = input_source["dataclass"]
            input_getter = input_source["getter"]
            logger.info(f"Using input_source {input_dataclass} with getter {input_getter} from config.")

            for image in getter_factory(input_dataclass, input_getter, self.get_sqlite_file(ctx))():
                logger.info(f"Image: {image}")
                logger.info(f"class: {image.source_class}.{image.source_getter}")

                if duplicate_getter is not None:
                    image_duplicates = duplicate_getter(image.source_fullpath)
                    if image_duplicates is not None:
                        logger.info("Image has duplicates!")
                    else:
                        image_duplicates = list()
                else:
                    image_duplicates = list()

                if isinstance(image, DetectionStorage.get_class()):
                    all_votings.append(DetectionStorage.convert_to_voting_dict(image, bs))

                elif isinstance(image, SimpleLabelStorage.get_class()):
                    src_images_getter = getter_factory(image.source_class, image.source_getter,
                                                   self.get_sqlite_file(ctx))
                    src_images = src_images_getter(img=image.image_fullpath)

                    if len(src_images) != 1:
                        raise ValueError("Number of images is not 1. Aborting...")
                    src_images = src_images[0]

                    if isinstance((src_images, SegmentDataStorage.get_class())):
                        segment_dict = SegmentDataStorage.convert_to_voting_dict(src_images, bs)
                        segment_dict["votings"].append((SegmentDataStorage.get_class().__name__ , json.loads(image.votings)))
                        all_votings.append(segment_dict)

                        for dup in image_duplicates:
                            src_images = src_images_getter(img=dup)
                            logger.info(f"Got {src_images} for duplicate {dup}.")
                            logger.info(
                                f"Using source class \"{image.source_class}\" and source getter \"{image.source_getter}\"")
                            if len(src_images) == 0:
                                # TODO: Store image source in duplicates (getter and dataclass)
                                # TODO: This is a workaround: Try to find the backreferences from the imagededup-handler
                                src_images = find_other_sources(
                                    self.get_other_module_config("Deduplication.Imagededup.ImagededupClass"),
                                    dup, self.get_sqlite_file(ctx))
                            if src_images is not None and len(src_images):
                                src_images = src_images[0]
                                logger.info(f"Adding duplicate image data {src_images}")
                                logger.info(f"{type(src_images)}, {type(src_images) == DetectionStorage.get_class()}")

                                if type(src_images) == DetectionStorage.get_class():
                                    # Handle Detection duplicates
                                    src_segment = getter_factory(
                                        src_images.source_dataclass,
                                        src_images.source_datagetter,
                                        self.get_sqlite_file(ctx)
                                    )(img=src_images.image_fullpath)

                                    x_min = src_segment[0].x_min + src_images.x_min
                                    x_max = src_segment[0].x_min + src_images.x_max
                                    y_min = src_segment[0].y_min + src_images.y_min
                                    y_max = src_segment[0].y_min + src_images.y_max

                                    orig_image = bs.get_image_by_id(src_segment[0].base_image)

                                    all_votings.append({
                                        "orig_image_name": orig_image.name,
                                        "orig_image_fullpath": orig_image.fullpath,
                                        "x_min": x_min,
                                        "x_max": x_max,
                                        "y_min": y_min,
                                        "y_max": y_max,
                                        "votings": {
                                            SegmentDataStorage.get_class().__name__: json.loads(image.votings),
                                        },
                                        "source": DetectionStorage.get_class().__name__,
                                    })

                                elif type(src_images) == SegmentDataStorage.get_class():
                                    # Handle segmentation duplicates
                                    db_image = bs.get_image_by_id(src_images.base_image)
                                    all_votings.append({
                                        "orig_image_name": db_image.name,
                                        "orig_image_fullpath": db_image.fullpath,
                                        "x_max": src_images.x_max,
                                        "x_min": src_images.x_min,
                                        "y_max": src_images.y_max,
                                        "y_min": src_images.y_min,
                                        "votings": store_votings,
                                        "source": type(src_images).__name__,
                                    })
                            else:
                                # Happens if we have same pictures from Detection and segmentation.
                                raise ValueError(f"Cannot resoulve duplicate {dup}")




                if isinstance(src_images[0], SegmentDataStorage.get_class()):
                    logger.info("SEGMENT")




                    store_votings = {SegmentDataStorage.get_class().__name__ : json.loads(image.votings)}

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
                        src_images = src_images_getter(img=dup)
                        logger.info(f"Got {src_images} for duplicate {dup}.")
                        logger.info(f"Using source class \"{image.source_class}\" and source getter \"{image.source_getter}\"")
                        if len(src_images) == 0:
                            # TODO: Store image source in duplicates (getter and dataclass)
                            # TODO: This is a workaround: Try to find the backreferences from the imagededup-handler
                            src_images = find_other_sources(
                                self.get_other_module_config("Deduplication.Imagededup.ImagededupClass"),
                                               dup,self.get_sqlite_file(ctx))
                        if src_images is not None and len(src_images):
                            src_images = src_images[0]
                            logger.info(f"Adding duplicate image data {src_images}")
                            logger.info(f"{type(src_images)}, {type(src_images) == DetectionStorage.get_class()}")

                            if type(src_images) == DetectionStorage.get_class():
                                # Handle Detection duplicates
                                src_segment = getter_factory(
                                    src_images.source_dataclass,
                                    src_images.source_datagetter,
                                    self.get_sqlite_file(ctx)
                                )(img=src_images.image_fullpath)

                                x_min = src_segment[0].x_min + src_images.x_min
                                x_max = src_segment[0].x_min + src_images.x_max
                                y_min = src_segment[0].y_min + src_images.y_min
                                y_max = src_segment[0].y_min + src_images.y_max

                                orig_image = bs.get_image_by_id(src_segment[0].base_image)

                                all_votings.append({
                                    "orig_image_name": orig_image.name,
                                    "orig_image_fullpath": orig_image.fullpath,
                                    "x_min": x_min,
                                    "x_max": x_max,
                                    "y_min": y_min,
                                    "y_max": y_max,
                                    "votings": {
                                        SegmentDataStorage.get_class().__name__: json.loads(image.votings),
                                        },
                                    "source": DetectionStorage.get_class().__name__,
                                })

                            elif type(src_images) == SegmentDataStorage.get_class():
                                # Handle segmentation duplicates
                                db_image = bs.get_image_by_id(src_images.base_image)
                                all_votings.append({
                                    "orig_image_name": db_image.name,
                                    "orig_image_fullpath": db_image.fullpath,
                                    "x_max": src_images.x_max,
                                    "x_min": src_images.x_min,
                                    "y_max": src_images.y_max,
                                    "y_min": src_images.y_min,
                                    "votings": store_votings,
                                    "source": type(src_images).__name__,
                                })
                        else:
                            # Happens if we have same pictures from Detection and segmentation.
                            raise ValueError(f"Cannot resoulve duplicate {dup}")

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
                    raise ValueError(f"MISSING {type(src_images[0])}")

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
