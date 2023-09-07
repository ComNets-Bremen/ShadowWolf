import itertools
import logging

"""
Some general functions for the analysis and combination of the classes / polls / votes etc.
"""

logger = logging.getLogger(__file__)

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes in the format (x_center, y_center, w, h)
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
    # inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
    #    inter_rect_y2 - inter_rect_y1, min=0
    # )
    inter_area = max(0, inter_rect_x2 - inter_rect_x1) * max(0, inter_rect_y2 - inter_rect_y1)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_union(box1, box2):
    """
    Create the union of two boxes.
    Box format: x_center, y_center, w, h

    Parameters
    ----------
    a 1st box
    b 2nd box

    Returns
    -------
    The new box

    """
    x = min(box1[0], box2[0])
    y = min(box1[1], box2[1])
    w = max(box1[0] + box1[2], box2[0] + box2[2]) - x
    h = max(box1[1] + box1[3], box2[1] + box2[3]) - y
    return [x, y, w, h]


def combine_votings(votings):
    """
    Combine the votings.
    Parameters
    ----------
    votings A list of all votings

    Returns
    -------
    a list of tuples with all votings in the format [(<source>, <voting>),]

    """
    votings_list = []
    for voting in votings:
        for w in voting:
            votings_list.append(w)

    return votings_list


def get_possible_classes(votings):
    """
    Return the possible classes from the function combine_votings

    Parameters
    ----------
    votings list of tuples in the format [(<source>, <voting>),]

    Returns
    -------
    Possible (numeric) classes
    """
    classes = []
    for det in votings:
        classes.extend(det[1].keys())
    return set(classes)


def condense_votings(boxes_list, iou_thres, config_weights):
    """
    Converts an array containig boxes and all possible votings to a condensed array, i.e., combine the votings
    and tidy up everything a little bit.

    Parameters
    ----------
    boxes_list      The boxes in the format [((x_center, y_center, w, h), {"Origin": {<class>:voting}, ...}), ...]
    iou_thres       The threshold for the intersections over union. If > this value -> combine to one box
    config_weights  The weights for the votings. If multiple values are available: combine. Values are unified.

    Returns
    -------

    [((x_center, y_center, w, h), [<weights per class>])]

    """
    while True:
        is_changed = False
        combined_boxes = []
        new_boxes_list = []
        for a, b in itertools.combinations(boxes_list, 2):
            if bbox_iou(a[0], b[0]) > iou_thres:
                is_changed = True
                combined_boxes.append(a)
                combined_boxes.append(b)
                new_boxes_list.append((bbox_union(a[0], b[0]), combine_votings([a[1], b[1]])))

        for b in boxes_list:
            if b not in combined_boxes:
                new_boxes_list.append(b)

        boxes_list = new_boxes_list
        if not is_changed:
            break
    # Right now, we combined boxes and added all votings. Next: Find a meaningful way of combining the votes from all boxes

    new_boxes_list = []

    for box in boxes_list:
        bbox, votings = box
        weights = {}
        for cls in get_possible_classes(votings):
            sum_weights = 0
            total_weight = 0
            for voting in votings:
                if str(cls) in voting[1]:
                    w = config_weights.get(voting[0], None)
                    if w is None:
                        raise ValueError(
                            f"Missing weight in config.json. Possible options are {str(config_weights.keys())}")

                    sum_weights += w
                    total_weight += w * voting[1][str(cls)]

            weights[str(cls)] = total_weight / sum_weights  # Normalize
        new_boxes_list.append((bbox, weights))
    return new_boxes_list


def to_xmin_xmax(box):
    """
    Convert a box (x_center, y_center, w, h) to (x_min, x_max, y_min, y_max)

    Parameters
    ----------
    box     The (x_center, y_center, w, h) box

    Returns
    -------
    The (x_min, x_max, y_min, y_max) box

    """
    x_min = int(box[0] - box[2] / 2)
    x_max = int(box[0] + box[2] / 2)
    y_min = int(box[1] - box[3] / 2)
    y_max = int(box[1] + box[3] / 2)

    return x_min, x_max, y_min, y_max
