#!/usr/bin/env python3
import glob
import os
import shutil
from pathlib import Path
import uuid
import json
import logging

from wolf_utils.misc import getter_factory

logger = logging.getLogger(__name__)

from BaseClass import BaseClass
from wolf_utils.ctx_helpers import get_last_variable
from Storage.SimpleLabelStorage import SimpleLabelStorage


class SimpleLabelEvaluationClass(BaseClass):
    def __init__(self, run_num, config, *args, **kwargs):
        super().__init__(config=config)
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.get_step_identifier()}")
        logger.info(f"Config: {self.get_module_config()}")

        ds = SimpleLabelStorage(self.get_sqlite_file(ctx))

        simple_eval_output = os.path.join(self.get_current_data_dir(ctx), "simple_eval_out")
        Path(simple_eval_output).mkdir(parents=True, exist_ok=True)

        simple_eval_input = os.path.join(self.get_current_data_dir(ctx), "simple_eval_input")
        Path(simple_eval_input).mkdir(parents=True, exist_ok=True)

        input_json_file = glob.glob(f"{simple_eval_input}/*.json")
        if len(input_json_file):
            logger.info(f"Found json file:{input_json_file[0]}. Will continue with storing results.")
            with open(input_json_file[0], "r") as jf:
                results = json.load(jf)

            for won_images in results["images"]:
                ds.set_relative_voting(won_images["image_original_name"], won_images["relative_class_voting"])

            continue_after_this_step = True

        else:
            shutil.copy(get_last_variable(ctx, "classes.txt"), simple_eval_output)

            dups = self.get_module_config().get("duplicates", None)
            dups_getter = None

            if dups is not None:
                _dataclass = self.get_module_config()["duplicates"]["dataclass"]
                _getter = self.get_module_config()["duplicates"]["getter"]
                dups_getter = getter_factory(_dataclass, _getter, self.get_sqlite_file(ctx))

            for input_source_num, input_source in enumerate(self.get_module_config()["inputs"]):
                images = []

                input_dataclass = input_source["dataclass"]
                input_getter = input_source["getter"]
                logger.info(f"Handling source {input_source_num}: {input_dataclass} -> {input_getter}")
                for image in getter_factory(input_dataclass, input_getter, self.get_sqlite_file(ctx))():
                    images.append({
                        "image": image,
                        "dataclass": input_dataclass,
                        "getter": input_getter,
                    })

                if dups_getter is not None:
                    _len_before = len(images)
                    logger.info(f"Source {input_source_num}: Removing duplicates...")
                    images = [image for image in images if image["image"] == dups_getter(image["image"])]
                    logger.info(f"Source {input_source_num}: Removed {_len_before - len(images)} images.")

                for image_dict in images:
                    image = image_dict["image"]
                    image_dataclass = image_dict["dataclass"]
                    image_getter = image_dict["getter"]
                    logger.info(f"Source {input_source_num}: Handling image {image} from class {image_dataclass}")
                    filename = os.path.split(image)[1]
                    name, ext = os.path.splitext(filename)

                    # Use a random uuid to prevent collisions especially on the SimpleEval side
                    new_filename = f"{name}_shadowwolf__{uuid.uuid4()}{ext}"
                    new_path = os.path.join(simple_eval_output, new_filename)

                    shutil.copy(image, new_path)
                    ds.store(
                        image_dataclass, image_getter,
                        image, new_filename)

            logger.critical(
                f"Exported images to \"{simple_eval_output}\". Upload to your SimpleEval server and wait for the votes.")
            logger.critical(
                f"Upload the resulting json file to \"{simple_eval_input}\" and re-run the script withe the following parameters:")
            logger.critical(f"-d {self.run_num} -o {os.path.split(ctx['output_dir'])[1]}")
            logger.critical(f"Recommended name for the project: \"{os.path.split(ctx['output_dir'])[1]}\"")
            continue_after_this_step = False

        ctx["steps"].append({
            "identifier": self.get_step_identifier(),
            "sqlite_file": self.get_sqlite_file(ctx),
            "images_for_simpleEval": simple_eval_output,
            "return_data_directory": simple_eval_input,
        })

        return continue_after_this_step, ctx


if __name__ == "__main__":
    pass
