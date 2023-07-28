#!/usr/bin/env python3

import os
import glob

import datetime

import logging
logger = logging.getLogger(__name__)

from BaseClass import BaseClass

## Does nothing. Is used as kind of a template for other classes removing for
# example the distortion etc.
class NullPreprocessingClass(BaseClass):
    def __init__(self, run_num):
        super().__init__()
        self.run_num = run_num

    def run(self, ctx):
        logger.info(f"Identifier: {self.getStepIdentifier()}")
        logger.info(f"Config: {self.getModuleConfig()}")

        ctx["steps"].append({
                "identifier" : self.getStepIdentifier(),
                "sqlite_file" : self.getSqliteFile(ctx),
            })

        return ctx

if __name__ == "__main__":
    pass
