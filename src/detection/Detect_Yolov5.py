import os
from pathlib import Path
import logging
import torch
import pandas as pd
import cv2

from wolf_utils.misc import batch
from wolf_utils.ProgressBar import printProgressBar


class Detect_Yolov5:

    def __init__(self, data, config, output_dir, args):
        self.data = data
        self.config = config
        self.output_dir = output_dir
        self.args = args


    def get_detections(self):

        output_dirs = dict()
        output_dirs["base"] = os.path.join(self.output_dir, "detections")
        output_dirs["labelled_subimages"] = os.path.join(output_dirs["base"], "labelled_subimages")
        output_dirs["labelled_orig_images"] = os.path.join(output_dirs["base"], "labelled_orig_images")
        output_dirs["labels_images"] = os.path.join(output_dirs["base"], "labels_images")

        for d in output_dirs:
            Path(output_dirs[d]).mkdir(parents=True, exist_ok=True)

        classes_path = os.path.join(output_dirs["labels_images"], "classes.txt")

        model_dir = os.path.dirname(os.path.realpath(__file__))

        print("Loading Models from", model_dir)
        model_file = os.path.join(model_dir, self.config.get("DEFAULT", "Detection_Model"))
        print("Loading Model from", model_file)

        model = torch.hub.load(
                self.config.get("DEFAULT", "Detection_Repo"),
                "custom",
                model_file,
                force_reload = self.config.getboolean("DEFAULT", "Detection_Force_Reload"),
                )


        files = pd.read_csv(self.data)

        printProgressBar(0, len(files["output filename"]), prefix = "Progess:", suffix = "Complete", length = 50)
        finished_labels = 0

        for i in batch(files["output filename"].tolist()):
            # use model.eval, torch.go_grad()??
            results = model(i)
            classes = results.names

            # Store classes.txt for imglabel
            if not os.path.isfile(classes_path):
                with open(classes_path, "w") as f:
                    for cls in classes:
                        f.writelines(f"{cls} {classes[cls]}\n")

            for image in zip(i, results.xyxy):
                (_, output_filename) = os.path.split(image[0])

                img = cv2.imread(image[0])

                if img is None:
                    logging.warning(str(image) + " can not be loaded")
                    continue

                # Get Dataset for original image from pandas frame
                original_data = files[files["output filename"] == image[0]].iloc[0]
                orig_filename = original_data["input filename"]

                orig_image = cv2.imread(orig_filename)

                if orig_image is None:
                    logging.warning(str(orig_filename) + " can not be loaded")

                labelled_orig_image = orig_image.copy()

                export_rows = []

                for detection in image[1].tolist():
                    # xmin    ymin    xmax   ymax  confidence  class
                    # img.shape -> height, width, channels

                    (x_min, y_min, x_max, y_max, confidence, cls) = detection
                    w = x_max - x_min
                    h = y_max - y_min
                    cls = int(cls)

                    cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,0,255), 2)
                    cv2.putText(img, str(int(confidence*100)), (int(x_max)-int(w/2), int(y_max)-int(h/2)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
                    print("Detection: class", cls, "confidence", confidence, "image", output_filename)

                    o_det_x_min = int(x_min + original_data["x_min"])
                    o_det_y_min = int(y_min + original_data["y_min"])
                    o_det_x_max = int(original_data["x_min"] + x_max)
                    o_det_y_max = int(original_data["y_min"] + y_max)

                    cv2.rectangle(labelled_orig_image, (o_det_x_min, o_det_y_min), (o_det_x_max, o_det_y_max), (0,0,255), 2)

                    (height, width, _) = orig_image.shape

                    x_center = (o_det_x_min + ((o_det_x_max - o_det_x_min)/2.0)) / width
                    y_center = (o_det_y_min + ((o_det_y_max - o_det_y_min)/2.0)) / height
                    d_width  = (o_det_x_max - o_det_x_min) / width
                    d_height = (o_det_y_max - o_det_y_min) / height

                    export_rows.append(f"{cls} {x_center} {y_center} {d_width} {d_height}")



                cv2.imwrite(os.path.join(output_dirs["labelled_subimages"], output_filename), img)
                cv2.imwrite(os.path.join(output_dirs["labelled_orig_images"], os.path.split(orig_filename)[1]), labelled_orig_image)
                cv2.imwrite(os.path.join(output_dirs["labels_images"], os.path.split(orig_filename)[1]), orig_image)
                with open(os.path.join(output_dirs["labels_images"], Path(os.path.split(orig_filename)[1]).stem + ".txt"), "w") as f:
                    f.writelines("\n".join(export_rows))

                finished_labels += 1
                printProgressBar(finished_labels, len(files["output filename"]), prefix = "Progess:", suffix = "Complete", length = 50)

        return "Done"
