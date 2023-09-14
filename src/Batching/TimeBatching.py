#!/usr/bin/env python3

import datetime

import logging

logger = logging.getLogger(__name__)

from BaseClass import BaseClass
from Storage.DataStorage import BatchingDataStorage, BasicAnalysisDataStorage


## Create batches based on the image timestamp
class TimeBatchingClass(BaseClass):
    def __init__(self, run_num, config, *args, **kwargs):
        super().__init__(config=config)
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.get_step_identifier()}")
        logger.info(f"Config: {self.get_module_config()}")

        ds = BatchingDataStorage(self.get_sqlite_file(ctx))
        analysis_storage = BasicAnalysisDataStorage(self.get_sqlite_file(ctx))

        image_db = []

        for image in ds.get_all_images():
            instance = analysis_storage.get_image_by_fullpath(image)
            meta, exif = analysis_storage.get_by_instance(instance)
            for e in exif:
                if e.exif_name == self.get_module_config()["exif_time_source"]:
                    dt = datetime.datetime.strptime(e.exif_value, "%Y:%m:%d %H:%M:%S")
                    image_db.append({
                        "image": instance,
                        "dt": dt
                    })
                    continue
        image_db.sort(key=lambda x: x["dt"])

        imgdb_batches = []
        current_subbatch = []
        lastframe = None
        for img in image_db:
            if lastframe is None:
                lastframe = img
                current_subbatch.append(img["image"])
                continue
            if (img["dt"] - lastframe["dt"]).total_seconds() > self.get_module_config()["max_timediff_s"]:
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
            "identifier": self.get_step_identifier(),
            "sqlite_file": self.get_sqlite_file(ctx),
            "num_batches": len(imgdb_batches),
            "batch_sizes": [len(x) for x in imgdb_batches],
            "num_images": sum([len(x) for x in imgdb_batches]),
        })

        return True, ctx


if __name__ == "__main__":
    pass
