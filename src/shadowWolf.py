#!/usr/bin/env python3

CONFIGFILE_DEFAULT = "default.cfg"

import os
import sys
import argparse
import configparser
import logging
import datetime
import pkgutil

from pathlib import Path
import importlib

import batching
batching_modules = [name for _, name, _ in pkgutil.iter_modules([os.path.dirname(batching.__file__)])]

import segmentation
segmentation_modules = [name for _, name, _ in pkgutil.iter_modules([os.path.dirname(segmentation.__file__)])]

import detection
detection_modules = [name for _, name, _ in pkgutil.iter_modules([os.path.dirname(detection.__file__)])]

config = configparser.ConfigParser()
config.read_file(open(CONFIGFILE_DEFAULT))

ap = argparse.ArgumentParser(
        description="ShadowWolf: Get annotatable images from dataset using motion detection.",
        epilog="Jens Dede, ComNets, Uni Bremen, " + str(datetime.datetime.now().year),
        )
ap.add_argument("-d", "--debug", help="Enable debug messages and store debug images", action="store_true")
ap.add_argument("images", nargs="+", help="The series of images")
ap.add_argument("--batching", help="The batching module", choices=batching_modules, default=config.get("DEFAULT", "Batching_Module"))
ap.add_argument("--segmentation", help="The segmentation module", choices=segmentation_modules, default=config.get("DEFAULT", "Segmentation_Module"))
ap.add_argument("--detection", help="The detection module", choices=detection_modules, default=config.get("DEFAULT", "Detection_Module"))
ap.add_argument("-o", "--out", type=str, default=None, help="The data output directory, default is an auto-generated path with date, time and the script filename.")
ap.add_argument("-x", "--extra", help="Store intermediate images", action="store_true")
args = vars(ap.parse_args())

if args["debug"]:
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Set logging level to DEBUG")

outdir = args["out"]
if not outdir:
    outdir = "_".join((datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "output", os.path.splitext(os.path.basename(__file__))[0]))

outdir = os.path.join(str(config.get("DEFAULT", "output_main_dir")), outdir)

print("Using output directory", outdir)
Path(outdir).mkdir(parents=True, exist_ok=True)

batching_module = importlib.import_module("batching."+str(args["batching"]))
batching_class = getattr(batching_module, args["batching"])
batching_instance = batching_class(data=args["images"], config=config)
batches = batching_instance.get_batches()

segmentation_module = importlib.import_module("segmentation."+str(args["segmentation"]))
segmentation_class = getattr(segmentation_module, args["segmentation"])
segmentation_instance = segmentation_class(data=batches, config=config, output_dir=outdir, args=args)
segments = segmentation_instance.get_segments()

detection_module = importlib.import_module("detection."+str(args["detection"]))
detection_class = getattr(detection_module, args["detection"])
detection_instance = detection_class(data=segments, config=config, output_dir=outdir, args=args)
detections = detection_instance.get_detections()

print(detections)

