#!/usr/bin/env python3

import os
import json
import importlib
import datetime
import shutil
from pathlib import Path
import logging
import pickle
import pprint

import argparse

from wolf_utils.misc import slugify, getter_factory
from wolf_utils.ColorLogger import CustomFormatter

logfile = "run_" + slugify(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + "_logfile.log"
streamhandler = logging.StreamHandler()
streamhandler.setFormatter(CustomFormatter())
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO, handlers=[logging.FileHandler(logfile), streamhandler])
logger = logging.getLogger(__name__)

class BaseClass:
    def __init__(self, config="config.json", *args, **kwargs):
        self.config = None
        self.config_filename = config
        with open(config, "r") as f:
            self.config = json.load(f)

    def getNewContext(self, output=None, cont=0):
        if output is None:
            output_dir = os.path.join(os.getcwd(), "output", slugify(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + str(self.config_filename)))
        else:
            output_dir = output
        logger.info(f"Using output directory {output_dir}")

        if Path(output_dir).exists() and cont>0:
            with open(os.path.join(output_dir, f"{cont-1}_end_ctx.pickle"), "rb") as f:
                ctx = pickle.load(f)
            ctx["continue_step"] = cont
            ctx["continue_start"] = datetime.datetime.now().isoformat()
            ctx["steps"] = ctx["steps"][:cont] # Remove old steps
        else:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            ctx = dict()
            ctx["start"] = datetime.datetime.now().isoformat()
            ctx["config_filename"] = self.config_filename
            ctx["output_dir"] = output_dir
            ctx["steps"] = []
            shutil.copy(self.config_filename, output_dir)

        if not "logfile" in ctx:
            ctx["logfile"] = []
        ctx["logfile"].append(logfile)
        return ctx

    def getSqliteFile(self, ctx, name="metadata.sqlite"):
        return "sqlite:///" + os.path.abspath(os.path.join(ctx["output_dir"], name))

    def run(self, ctx=None, cont=0):
        if ctx is None:
            ctx = self.getNewContext()
        for i, m in enumerate(self.config["modules"][cont:], start=cont):
            cont, ctx = getter_factory(m["name"], "run", i)(ctx)

            # Store the end context to a pickle
            with open(os.path.join(ctx["output_dir"], f"{i}_end_ctx.pickle"), "wb") as f:
                pickle.dump(ctx, f)

            # Previous step indicated to stop now
            if not cont:
                logger.info(f"Process stop was indicated by {m['name']}. Check output for further information.")
                break

        if "continue_start" in ctx:
            ctx["continue_end"] = datetime.datetime.now().isoformat()
            ctx["continue_duration"] = (datetime.datetime.fromisoformat(ctx["continue_end"]) - datetime.datetime.fromisoformat(ctx["continue_start"])).total_seconds()
        else:
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
    parser = argparse.ArgumentParser(
                    prog='ShadowWolf',
                    description='A workflow for Animal detection and evaluation of the detections.',
                    epilog=f'Jens Dede, ComNets University of Bremen, {datetime.date.today().year}')
    parser.add_argument('-c', '--config', type=str, default="config.json", help="The config file in json format.")
    parser.add_argument('-o', '--output', type=str, default=None, help="The output directory.")
    parser.add_argument('-d', '--cont', type=int, default=0, help="Continue from a certain step. Defaults to 0 (i.e. start from the beginning).")
    args = parser.parse_args()

    bc = BaseClass(config=args.config)
    ctx = bc.getNewContext(output=args.output, cont=args.cont)
    bc.run(ctx, cont=args.cont)
