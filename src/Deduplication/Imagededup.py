#!/usr/bin/env python3
from imagededup.methods import PHash
import logging
from wolf_utils.misc import getter_factory

logger = logging.getLogger(__name__)

from BaseClass import BaseClass

from Storage.DuplicateStorage import DuplicateImageStorage

class ImagededupClass(BaseClass):
    def __init__(self, run_num):
        super().__init__()
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.getStepIdentifier()}")
        logger.info(f"Config: {self.getModuleConfig()}")

        input_images = []
        for input_cls in self.getModuleConfig()["inputs"]:
            input_dataclass = input_cls["dataclass"]
            getter = getter_factory(input_dataclass, input_cls["getter"], self.getSqliteFile(ctx))
            input_images.extend(getter())

        hasher = PHash() # TODO: Make configurable

        encodings = dict()

        logger.info(f"Creating hashes for {len(input_images)} images...")
        for image in input_images:
            encodings[image] = hasher.encode_image(image)

        duplicates = hasher.find_duplicates(encoding_map=encodings)

        duplicat_storage = DuplicateImageStorage(self.getSqliteFile(ctx))

        for dup in duplicates:
            if len(duplicates[dup]) > 0:
                for related_dup in duplicates[dup]:
                    duplicat_storage.add_duplicate(dup, related_dup)
                logger.info(f"similar images for {dup}: {duplicat_storage.get_similar(dup)}")



        ctx["steps"].append({
                "identifier" : self.getStepIdentifier(),
                "sqlite_file" : self.getSqliteFile(ctx),
            })

        return True, ctx

if __name__ == "__main__":
    pass
