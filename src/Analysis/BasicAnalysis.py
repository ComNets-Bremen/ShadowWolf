#!/usr/bin/env python3

import os
import glob
import PIL.Image

import logging
logger = logging.getLogger(__name__)

from BaseClass import BaseClass
from wolf_utils.PILhelper import get_all_exif, is_gray, get_all_iptcs
from Storage.DataStorage import BasicAnalysisDataStorage



## Stores exif and cam data in sqlite-database
class BasicAnalysisClass(BaseClass):
    def __init__(self, run_num, config, *args, **kwargs):
        super().__init__(config=config)
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.get_step_identifier()}")
        logger.info(f"Config: {self.get_module_config()}")
        images = glob.glob(self.get_input_data() + "/*." + self.get_main_config()["image_filetype"])

        ds = BasicAnalysisDataStorage(self.get_sqlite_file(ctx))

        for image in images:
            logger.info(f"Processing image {image}")
            with PIL.Image.open(image) as img:
                imageIsGray = is_gray(img)
                ds.store(
                        fullpath = image,
                        imageIsGray = imageIsGray,
                        exifs = get_all_exif(img),
                        iptcs = get_all_iptcs(img),
                        )

        ctx["steps"].append({
                "identifier" : self.get_step_identifier(),
                "num_images" : len(images),
                "sqlite_file" : self.get_sqlite_file(ctx)
                })

        return True, ctx

if __name__ == "__main__":
    pass
