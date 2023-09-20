#!/usr/bin/env python3
from imagededup.methods import PHash
import logging
from wolf_utils.misc import getter_factory

logger = logging.getLogger(__name__)

from BaseClass import BaseClass

from Storage.DuplicateStorage import DuplicateImageStorage

class ImagededupClass(BaseClass):
    def __init__(self, run_num, config, *args, **kwargs):
        super().__init__(config=config)
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.get_step_identifier()}")
        logger.info(f"Config: {self.get_module_config()}")

        input_images = []
        dataclasses  = {}
        getter       = {}

        for input_cls in self.get_module_config()["inputs"]:
            for img in  getter_factory(input_cls["dataclass"], input_cls["getter"], self.get_sqlite_file(ctx))():
                input_images.append(img)
                dataclasses[img] = input_cls["dataclass"]
                getter[img]      = input_cls["getter"]

        # Unify list, remove duplicates
        input_images = list(dict.fromkeys(input_images))

        hasher = PHash() # TODO: Make configurable

        encodings = dict()

        logger.info(f"Creating hashes for {len(input_images)} images...")
        for image in input_images:
            encodings[image] = hasher.encode_image(image)

        duplicates = hasher.find_duplicates(encoding_map=encodings)

        duplicat_storage = DuplicateImageStorage(self.get_sqlite_file(ctx))

        for dup in duplicates:
            if len(duplicates[dup]) > 0:
                for related_dup in duplicates[dup]:
                    duplicat_storage.add_duplicate((dup, dataclasses[dup], getter[dup]), (related_dup, dataclasses[related_dup], getter[related_dup]))

        ctx["steps"].append({
                "identifier" : self.get_step_identifier(),
                "sqlite_file" : self.get_sqlite_file(ctx),
            })

        return True, ctx

if __name__ == "__main__":
    pass
