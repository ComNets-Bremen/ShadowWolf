#!/usr/bin/env python3

import os
import glob
import importlib

import datetime

from imagededup.methods import PHash

import logging
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
        input_dataclass_name = self.getModuleConfig()["deduplication_dataclass"]
        input_dataclass_getter = self.getModuleConfig()["deduplication_getter"]

        logger.info(f"Using getter {input_dataclass_getter} from class {input_dataclass_name}")
        module_path, class_name = input_dataclass_name.rsplit('.', 1)
        module = importlib.import_module(module_path, class_name)
        importlib.invalidate_caches()
        input_Dataclass = getattr(module, class_name)
        input_dc = input_Dataclass(self.getSqliteFile(ctx))
        getter = getattr(input_dc, input_dataclass_getter)

        input_images = getter()

        hasher = PHash() # TODO: Make configurable

        encodings = dict()

        logger.info(f"Creating hashes for {len(input_images)} images...")
        for image in input_images:
            encodings[image] = hasher.encode_image(image)

        duplicates = hasher.find_duplicates(encoding_map=encodings)

        duplicat_storage = DuplicateImageStorage(self.getSqliteFile(ctx), input_dataclass_name)

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
