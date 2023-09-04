#!/usr/bin/env python3

import os
import glob

import datetime

import logging

logger = logging.getLogger(__name__)

from BaseClass import BaseClass


## Does nothing. Is used as kind of a template for other classes
class NullSegmentationClass(BaseClass):
    def __init__(self, run_num):
        super().__init__()
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.get_step_identifier()}")
        logger.info(f"Config: {self.get_module_config()}")

        ctx["steps"].append({
            "identifier": self.get_step_identifier(),
            "sqlite_file": self.get_sqlite_file(ctx),
        })

        return True, ctx


if __name__ == "__main__":
    pass
