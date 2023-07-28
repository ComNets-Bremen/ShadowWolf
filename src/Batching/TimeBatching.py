#!/usr/bin/env python3

import os
import glob
import PIL.Image
from PIL.ExifTags import TAGS

import datetime

import logging
logger = logging.getLogger(__name__)

from BaseClass import BaseClass
from wolf_utils.PILhelper import getAllExif, isGray
from Storage.DataStorage import BatchingDataStorage, BasicAnalysisDataStorage

## Create batches based on the image timestamp
class TimeBatchingClass(BaseClass):
    def __init__(self, run_num):
        super().__init__()
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.getStepIdentifier()}")
        logger.info(f"Config: {self.getModuleConfig()}")
        images = glob.glob(self.getInputData() + "/*." + self.getMainConfig()["image_filetype"])

        ds = BatchingDataStorage(self.getSqliteFile(ctx))
        analysis_storage = BasicAnalysisDataStorage(self.getSqliteFile(ctx))

        image_db = []

        for image in ds.get_all_images():
            meta, exif = analysis_storage.getByInstance(image[0])
            for e in exif:
                if e.exif_name == self.getModuleConfig()["exif_time_source"]:
                    dt = datetime.datetime.strptime(e.exif_value, "%Y:%m:%d %H:%M:%S")
                    image_db.append({
                        "image" : image[0],
                        "dt"    : dt
                        })
                    continue
        image_db.sort(key=lambda x:x["dt"])

        imgdb_batches = []
        current_subbatch = []
        lastframe = None
        for img in image_db:
            if lastframe is None:
                lastframe = img
                current_subbatch.append(img["image"])
                continue
            if (img["dt"] - lastframe["dt"]).total_seconds() > self.getModuleConfig()["max_timediff_s"]:
                imgdb_batches.append(current_subbatch)
                current_subbatch = []
            current_subbatch.append(img["image"])
            lastframe = img
        imgdb_batches.append(current_subbatch)

        logger.info(f"Database split into {len(imgdb_batches)} batches.")
        logger.info(f"Storing batches into database")
        for batch in imgdb_batches:
            ds.store(batch, __name__)

        ctx["steps"].append({
                "identifier" : self.getStepIdentifier(),
                "sqlite_file" : self.getSqliteFile(ctx),
                "num_batches" : len(imgdb_batches),
                "batch_sizes" : [len(x) for x in imgdb_batches],
                "num_images"  : sum([len(x) for x in imgdb_batches]),
            })

        return ctx

if __name__ == "__main__":
    pass
