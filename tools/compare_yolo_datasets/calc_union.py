#!/usr/bin/env python3

"""
Jens Dede, jd@comnets-uni-bremen.de

Ref: https://towardsdatascience.com/confusion-matrix-and-object-detection-f0cbcb634157
"""

import os
import sys
import glob
import argparse
import datetime
import pathlib

parser = argparse.ArgumentParser(
                    prog=f"{os.path.basename(__file__)}",
                    description='compare two yolo datasets using the IOU and give the F1 score.',
                    epilog=f"Jens Dede, jd@comnets.uni-bremen.de {datetime.date.year}")



parser.add_argument("--iou", type=float, help="The intersection over union. Defaults to 0.5", default=0.1)
parser.add_argument("--ext", type=str, help="The file extension. Defaults to \"txt\"", default="txt")
parser.add_argument("ground_truth", type=pathlib.Path, help="The path to the ground truth detections.")
parser.add_argument("detections", type=pathlib.Path, help="The path to the detections to compare to.")

args = parser.parse_args()

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    # Intersection area
    #inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
    #    inter_rect_y2 - inter_rect_y1, min=0
    #)
    inter_area = max(0, inter_rect_x2 - inter_rect_x1) * max(0, inter_rect_y2 - inter_rect_y1)
    # Union Area
    b1_area = (b1_x2 - b1_x1 ) * (b1_y2 - b1_y1 )
    b2_area = (b2_x2 - b2_x1 ) * (b2_y2 - b2_y1 )

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def compare_boxes(det_boxes, gt_boxes):
    # Format: Class, xCenter, yCenter, w, h
    list_gt  = []
    list_det = []

    # tn not possible in image recognition
    fp, tp, fn, tn = 0, 0, 0, 0

    for det_box in det_boxes:
        if len(det_box):
            list_det.append({
                "raw" : det_box,
                "class" : int(det_box[0]),
                "detected" : False,
                "box" : [float(x) for x in det_box.split()[1:]],
                })

    for gt_box in gt_boxes:
        if len(gt_box):
            list_gt.append({
                "raw" : gt_box,
                "class" : int(gt_box[0]),
                "detected" : False,
                "box" : [float(x) for x in gt_box.split()[1:]]
                })


    for gt_i in range(len(list_gt)):
        for det_i in range(len(list_det)):
            box_correct = bbox_iou(list_gt[gt_i]["box"], list_det[det_i]["box"]) > args.iou
            same_class  = list_gt[gt_i]["class"] == list_det[det_i]["class"]
            gt_detected = list_gt[gt_i]["detected"]
            det_detected= list_det[det_i]["detected"]

            if box_correct and same_class and not gt_detected and not det_detected:
                tp += 1
                list_gt[gt_i]["detected"] = True
                list_det[det_i]["detected"] = True

            elif box_correct and not same_class and not gt_detected and not det_detected:
                fp += 1
                list_gt[gt_i]["detected"] = True
                list_det[det_i]["detected"] = True
    for gt in list_gt:
        if not gt["detected"]:
            fn += 1
    for det in list_det:
        if not det["detected"]:
            fp += 1
    return fp, tp, fn, tn



detections   = glob.glob(os.path.join(args.detections, f"*.{args.ext}"))
ground_truth = glob.glob(os.path.join(args.ground_truth, f"*.{args.ext}"))

det_filenames = set([os.path.basename(f) for f in detections if not os.path.basename(f) == "classes.txt"])
gt_filenames  = set([os.path.basename(f) for f in ground_truth if not os.path.basename(f) == "classes.txt"])
all_filenames = det_filenames.union(gt_filenames)

print(f"Detections: {len(det_filenames)}, Ground truth: {len(gt_filenames)} -> {len(all_filenames)}")

sumdata =  {"fp" : 0, "tp" : 0, "fn" : 0, "tn" : 0}

for f in all_filenames:
    print(f)
    if f in det_filenames and f in gt_filenames:
        with open(os.path.join(args.detections, f), "r") as det_handler, open(os.path.join(args.ground_truth, f), "r") as gt_handler:
            det_boxes = [det for det in det_handler.read().split("\n") if len(det)]
            gt_boxes  = [gt for gt in gt_handler.read().split("\n") if len(gt)]

            fp, tp, fn, tn = compare_boxes(det_boxes, gt_boxes)
            print(f"Comparing {det_boxes} <-> {gt_boxes} -> fp={fp}, tp={tp}, fn={fn}, fp={fp}")

            sumdata["fp"] += fp
            sumdata["tp"] += tp
            sumdata["fn"] += fn
            sumdata["tn"] += tn


    elif f in det_filenames and f not in gt_filenames:
        with open(os.path.join(args.detections, f), "r") as fhandler:
            d = fhandler.read().split("\n")
            num_fp = len([det for det in d if len(det)])
            print(f"fp: {num_fp} {d}")
            sumdata["fp"] += num_fp

    elif f not in det_filenames and f in gt_filenames:
        with open(os.path.join(args.ground_truth, f), "r") as fhandler:
            d = fhandler.read().split("\n")
            num_fn = len([det for det in d if len(det)])
            print(f"fn: {num_fn}, {d}")
            sumdata["fn"] += num_fn


print(sumdata)

precision = sumdata["tp"] / (sumdata["tp"] + sumdata["fp"])
recall    = sumdata["tp"] / (sumdata["tp"] + sumdata["fn"])
f1        = 2*((precision * recall) / (precision + recall))

print(sumdata)
print("Precision", precision)
print("Recall", recall)
print("F1 score", f1)
