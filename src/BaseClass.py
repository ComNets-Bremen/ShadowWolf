#!/usr/bin/env python3

import os
import json
import importlib
import datetime
from pathlib import Path
import logging
import pickle
import pprint

from wolf_utils.misc import slugify

logfile = "run_" + slugify(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "_logfile.log"
loghandlers = [logging.FileHandler(logfile), logging.StreamHandler()]
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO, handlers=loghandlers)
logger = logging.getLogger(__name__)

class BaseClass:
    def __init__(self, config="config.json", *args, **kwargs):
        self.config = None
        self.config_filename = config
        with open(config, "r") as f:
            self.config = json.load(f)

    def getNewContext(self):
        output_dir = os.path.join(os.getcwd(), "output", slugify(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(self.config_filename)))
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        ctx = dict()
        ctx["start"] = datetime.datetime.now().isoformat()
        ctx["config_filename"] = self.config_filename
        ctx["output_dir"] = output_dir
        ctx["steps"] = []
        return ctx

    def getSqliteFile(self, ctx, name="metadata.sqlite"):
        return "sqlite:///" + os.path.abspath(os.path.join(ctx["output_dir"], name))

    def run(self, ctx=None):
        if ctx is None:
            ctx = self.getNewContext()
        for i, m in enumerate(self.config["modules"]):
            module_path, class_name = m["name"].rsplit('.', 1)
            logger.info(f"running {module_path} {class_name}")
            module = importlib.import_module(module_path, class_name)
            importlib.invalidate_caches()
            cls = getattr(module, class_name)
            ctx = cls(i).run(ctx)

            # Store the end context to a pickle
            with open(os.path.join(ctx["output_dir"], f"{i}_end_ctx.pickle"), "wb") as f:
                pickle.dump(ctx, f)

        ctx["end"] = datetime.datetime.now().isoformat()
        ctx["duration"] = (datetime.datetime.fromisoformat(ctx["end"]) - datetime.datetime.fromisoformat(ctx["start"])).total_seconds()

        pp = pprint.PrettyPrinter(indent=4)
        logger.info(f"Final context:\n {pp.pformat(ctx)}\n")

    def getMainConfig(self):
        return self.config["main_config"]


    def getLastConfig(self, key):
        for i in range(self.run_num, 0, -1):
            if key in self.config["modules"][i]:
                return self.config["modules"][i][key]
        return self.getMainConfig()[key]

    def getLastConfigWithKey(self, key):
        for i in range(self.run_num, 0, -1):
            if key in self.config["modules"][i]:
                return self.config["modules"][i]
        return None


    # Get the images from the last step
    def getInputData(self):
        return self.getLastConfig("image_dir")

    def getModuleConfig(self):
        if self.run_num is None:
            # Get the config by the module name. Might be problematic if one
            # Module is called multiple times with different parameters
            fqn = str(self.__class__.__module__) + "." + str(self.__class__.__name__)
            logger.info(f"Getting config for {fqn}")

            for m in self.config["modules"]:
                if m["name"] == fqn:
                    return m
        elif self.run_num < len(self.config["modules"]):
            # Get the module config from the previous order
            return self.config["modules"][self.run_num]

        logger.info("No config found")
        return dict()

    def getStepIdentifier(self):
        return f"{str(self.run_num)}_{self.__class__.__name__}"

    def getCurrentDataDir(self, ctx):
        p = os.path.join(ctx["output_dir"], self.getStepIdentifier())
        Path(p).mkdir(parents=True, exist_ok=True)
        return p

if __name__ == "__main__":
    bc = BaseClass()
    ctx = bc.getNewContext()
    bc.run(ctx)
