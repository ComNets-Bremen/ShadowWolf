#!/usr/bin/env python3
import glob
import os
import shutil
from pathlib import Path
import cv2
import uuid

import importlib
import json

import logging
logger = logging.getLogger(__name__)

from BaseClass import BaseClass

from wolf_utils.ctx_helpers import get_last_variable

from Storage.SimpleLabelStorage import SimpleLabelStorage

## Does nothing. Is used as kind of a template for other classes
class SimpleLabelEvaluationClass(BaseClass):
    def __init__(self, run_num):
        super().__init__()
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.getStepIdentifier()}")
        logger.info(f"Config: {self.getModuleConfig()}")

        ds = SimpleLabelStorage(self.getSqliteFile(ctx))

        last_module_with_getter = self.getLastConfigWithKey("getter")
        input_dataclass, input_getter = last_module_with_getter["dataclass"], last_module_with_getter["getter"]
        logger.info(f"Using {input_getter} from class {input_dataclass}")
        module_path, class_name = input_dataclass.rsplit('.', 1)
        module = importlib.import_module(module_path, class_name)
        importlib.invalidate_caches()
        input_Dataclass = getattr(module, class_name)
        input_dc = input_Dataclass(self.getSqliteFile(ctx))
        getter = getattr(input_dc, input_getter)

        input_images = getter()

        simple_eval_output = os.path.join(self.getCurrentDataDir(ctx), "simple_eval_out")
        Path(simple_eval_output).mkdir(parents=True, exist_ok=True)

        simple_eval_input = os.path.join(self.getCurrentDataDir(ctx), "simple_eval_input")
        Path(simple_eval_input).mkdir(parents=True, exist_ok=True)

        input_json_file = glob.glob(f"{simple_eval_input}/*.json")
        if len(input_json_file):
            logger.info(f"Found json file:{input_json_file[0]}. Will continue with storing results.")
            with open(input_json_file[0], "r") as jf:
                results = json.load(jf)

            for won_images in results["images"]:
                print(won_images)
                ds.set_relative_voting(won_images["image_original_name"], won_images["relative_class_voting"])

            continue_after_this_step = True

        else:

            shutil.copy(get_last_variable(ctx, "classes.txt"), simple_eval_output)

            for image in input_images:
                logger.info(f"Handling image {image['image_fullpath']}")
                filename = os.path.split(image["image_fullpath"])[1]
                name, ext = os.path.splitext(filename)

                # Use a random uuid to prevent collisions especially on the SimpleEval side
                new_filename = f"{name}_shadowwolf__{uuid.uuid4()}{ext}"

                bin_img = cv2.imread(image["image_fullpath"])
                cropped_img = bin_img[round(image["y_min"]):round(image["y_max"]), round(image["x_min"]):round(image["x_max"])]
                cv2.imwrite(os.path.join(simple_eval_output, new_filename), cropped_img)

                ds.store(image["image_fullpath"], new_filename, image["x_min"], image["y_min"], image["x_max"], image["y_max"])

            logger.critical(f"Exported images to \"{simple_eval_output}\". Upload to your SimpleEval server and wait for the votes.")
            logger.critical(f"Upload the resulting json file to \"{simple_eval_input}\" and re-run the script withe the following parameters:")
            logger.critical(f"-d {self.run_num} -o {os.path.split(ctx['output_dir'])[1]}")
            logger.critical(f"Recommended name for the project: \"{os.path.split(ctx['output_dir'])[1]}\"")
            continue_after_this_step = False


        ctx["steps"].append({
            "identifier" : self.getStepIdentifier(),
            "sqlite_file" : self.getSqliteFile(ctx),
            "images_for_simpleEval" : simple_eval_output,
            "return_data_directory" : simple_eval_input,
            })

        return continue_after_this_step, ctx

if __name__ == "__main__":
    pass
