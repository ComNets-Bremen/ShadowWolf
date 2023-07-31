#!/usr/bin/env python3

import os
import glob
import PIL.Image

import logging
logger = logging.getLogger(__name__)

from BaseClass import BaseClass
from wolf_utils.PILhelper import getAllExif, isGray
from Storage.DataStorage import BasicAnalysisDataStorage



## Stores exif and cam data in sqlite-database
class BasicAnalysisClass(BaseClass):
    def __init__(self, run_num):
        super().__init__()
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.getStepIdentifier()}")
        logger.info(f"Config: {self.getModuleConfig()}")
        images = glob.glob(self.getInputData() + "/*." + self.getMainConfig()["image_filetype"])

        ds = BasicAnalysisDataStorage(self.getSqliteFile(ctx))

        for image in images:
            logger.info(f"Processing image {image}")
            with PIL.Image.open(image) as img:
                imageIsGray = isGray(img)
                ds.store(
                        fullpath = image,
                        imageIsGray = imageIsGray,
                        exifs = getAllExif(img),
                        )

        ctx["steps"].append({
                "identifier" : self.getStepIdentifier(),
                "num_images" : len(images),
                "sqlite_file" : self.getSqliteFile(ctx)
                })

        return ctx

if __name__ == "__main__":
    pass
