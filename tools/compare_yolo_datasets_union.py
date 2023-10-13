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



parser.add_argument("--iou", type=float, help="The intersection over union. Defaults to 0.5", default=0.5)
parser.add_argument("--ext", type=str, help="The file extension. Defaults to \"txt\"", default="txt")
parser.add_argument("--iext", type=str, help="The file extension for images. Defaults to \"jpg\"", default="jpg")
parser.add_argument("ground_truth", type=pathlib.Path, help="The path to the ground truth detections.")
parser.add_argument("detections", type=pathlib.Path, help="The path to the detections to compare to.")
parser.add_argument("--images", type=pathlib.Path, help="Input images for optical checks.", default=None)
parser.add_argument("--output", type=pathlib.Path, help="The output directory if output is generated.", default="output")

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


def compare_boxes(det_boxes, gt_boxes, iou=0.5, image=None, output_image=None):
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

    # Keep boxes in mind we have already printed
    gt_i_printed  = []
    det_i_printed = []
    im = None

    if image is not None and os.path.isfile(image):
        import cv2
        im = cv2.imread(image)


    for gt_i in range(len(list_gt)):
        for det_i in range(len(list_det)):
            box_iou = bbox_iou(list_gt[gt_i]["box"], list_det[det_i]["box"])


            box_correct = box_iou > iou
            same_class  = list_gt[gt_i]["class"] == list_det[det_i]["class"]
            gt_detected = list_gt[gt_i]["detected"]
            det_detected= list_det[det_i]["detected"]

            if im is not None:
                h, w, _ = im.shape

                if gt_i not in gt_i_printed:
                    box = list_gt[gt_i]["box"]
                    box = box[0] * w, box[1] * h, box[2] * w, box[3] * h
                    x_min, y_min = int(box[0] - (box[2]/2)), int(box[1] - (box[3]/2))
                    x_max, y_max = int(box[0] + (box[2]/2)), int(box[1] + (box[3]/2))
                    text_size, _ = cv2.getTextSize("GT", cv2.FONT_HERSHEY_PLAIN, 1, 2)
                    cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
                    cv2.rectangle(im, (x_min, y_min), (x_min+text_size[0], y_min+text_size[1]), (0,0,255), -1)
                    cv2.putText(im, "GT", (x_min, y_min+text_size[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
                    gt_i_printed.append(gt_i)
                if det_i not in det_i_printed:
                    box = list_det[det_i]["box"]
                    box = box[0] * w, box[1] * h, box[2] * w, box[3] * h
                    x_min, y_min = int(box[0] - (box[2]/2)), int(box[1] - (box[3]/2))
                    x_max, y_max = int(box[0] + (box[2]/2)), int(box[1] + (box[3]/2))
                    text_size, _ = cv2.getTextSize("DET", cv2.FONT_HERSHEY_PLAIN, 1, 2)
                    cv2.rectangle(im, (x_min, y_min), (x_max, y_max), (0,128,0), 2)
                    cv2.rectangle(im, (x_max-text_size[0], y_max-text_size[1]), (x_max, y_max), (0,128, 0), -1)
                    cv2.putText(im, "DET", (x_max-text_size[0], y_max), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
                    gt_i_printed.append(gt_i)
                    det_i_printed.append(det_i)

                if box_iou > 0.1:
                    box1_x_center, box1_y_center =  list_gt[gt_i]["box"][0], list_gt[gt_i]["box"][1]
                    box2_x_center, box2_y_center =  list_det[det_i]["box"][0], list_det[det_i]["box"][1]
                    text = f"IoU={box_iou:.2f}"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)
                    x_center = int(((box1_x_center + box2_x_center) / 2) * w)
                    y_center = int(((box1_y_center + box2_y_center) / 2 ) * h)

                    cv2.putText(im, text, (int(x_center-(text_size[0]/2)), int(y_center-(text_size[1]/2))), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)



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
    if im is not None:
        cv2.imwrite(output_image, im)
    return fp, tp, fn, tn


output_dir = args.output

if args.images is not None:
    # Make sure the output dir exists
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)


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

            image_filename = None

            if args.images is not None:
                image_filename = os.path.join(args.images, f"{os.path.splitext(f)[0]}.{args.iext}")

            output_image_filename = os.path.join(output_dir, f"{os.path.splitext(f)[0]}.{args.iext}")

            fp, tp, fn, tn = compare_boxes(det_boxes, gt_boxes, args.iou, image_filename, output_image_filename)
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
