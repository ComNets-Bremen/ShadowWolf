import random
import cv2
import numpy as np

def get_avg_image(images, percentage=20, min_images = 5):
    number = int(len(images)*percentage/100.0)
    number = min(max(number, min_images), len(images))
    subset = random.sample(images, number)
    img_sub = [cv2.imread(s, 1) for s in subset]
    img = [i for i in img_sub if i is not None] # Make sure all images can be opened

    img = np.mean(img, axis=0)
    return img.astype("uint8")

def filter_small_boxes(boxes, min_area):
    output_boxes = []
    for box in boxes:
        if type(box) == np.ndarray:
            if cv2.contourArea(box) < min_area:
                continue

        elif type(box) in (tuple, list) and len(box) == 4:
            if (box[2] * box[3]) < min_area:
                continue

        output_boxes.append(box)
    return output_boxes

# Create union of two boxes, format: x, y, w, h
def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return [x, y, w, h]

def intersect(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if h<0 or w<0:
        return False
    return True

def group_rectangles(rec, iterations=1):
    rec = list(rec)
    """
    Union intersecting rectangles.
    Args:
        rec - list of rectangles in form [x, y, w, h]
    Return:
        list of grouped rectangles
    """
    for _ in range(iterations):

        tested = [False for i in range(len(rec))]
        final = []
        i = 0
        while i < len(rec):
            if not tested[i]:
                j = i+1
                while j < len(rec):
                    if not tested[j] and intersect(rec[i], rec[j]):
                        rec[i] = union(rec[i], rec[j])
                        tested[j] = True
                        j = i
                    j += 1
                final.append(rec[i])
            i += 1
        rec = final
    return rec

# Extend boxes by for example 40%
def extend_boxes(boxes, percent=40):
    new_boxes = list()
    for box in boxes:
        (x, y, w, h) = box
        center_x = x+(w/2.0)
        center_y = y+(h/2.0)
        w = int(w*(1+(percent/100.0)))
        h = int(h*(1+(percent/100.0)))
        x = int(center_x - (w/2.0))
        y = int(center_y - (h/2.0))
        new_boxes.append((x,y,w,h))
    return new_boxes

# Generate squared boxes as required by ML algorithm
def get_squared_boxes(boxes):
    new_boxes = list()
    for box in boxes:
        (x,y,w,h) = box
        center_x = x+(w/2.0)
        center_y = y+(h/2.0)
        s = max(w, h)
        x = int(center_x - (s/2.0))
        y = int(center_y - (s/2.0))
        new_boxes.append((x, y, s, s))
    return new_boxes

def get_greycount(box, ref_image):
    (x, y, w, h) = box
    y_min = max(y, 0)
    y_max = min(y+h, ref_image.shape[0])
    x_min = max(x, 0)
    x_max = min(x+w, ref_image.shape[1])

    return cv2.mean(ref_image[y_min:y_max, x_min:x_max])[0]
