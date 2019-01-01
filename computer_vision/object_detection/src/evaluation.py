from collections import namedtuple
import torch


PredBoundingBox = namedtuple("PredBoundingBox", ["probability", "class_id",
                                                 "classname", "bounding_box"
                                                 ])


def center_2_hw(box: torch.Tensor) -> float:
    """
    Converting (cx, cy, w, h) to (x1, y1, x2, y2)
    """

    return torch.cat(
        [box[:, 0, None] - box[:, 2, None]/2,
         box[:, 1, None] - box[:, 3, None]/2,
         box[:, 0, None] + box[:, 2, None]/2,
         box[:, 1, None] + box[:, 3, None]/2
         ], dim=1)


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    # Coverting (cx, cy, w, h) to (x1, y1, x2, y2) since its easier to extract min/max coordinates
    temp_box_a, temp_box_b = center_2_hw(box_a), center_2_hw(box_b)

    max_xy = torch.min(temp_box_a[:, None, 2:], temp_box_b[None, :, 2:])
    min_xy = torch.max(temp_box_a[:, None, :2], temp_box_b[None, :, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def box_area(box: torch.Tensor) -> float:
    return box[:, 2] * box[:, 3]


def jaccard(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    intersection = intersect(box_a, box_b)
    union = box_area(box_a).unsqueeze(1) + box_area(box_b).unsqueeze(0) - intersection
    return intersection / union


def non_max_suppression(bounding_boxes: list, iou_threshold: float = 0.5) -> list:
    filtered_bb = []

    while len(bounding_boxes) != 0:
        best_bb = bounding_boxes.pop(0)
        filtered_bb.append(best_bb)

        remove_items = []
        for bb in bounding_boxes:
            iou = jaccard(torch.tensor(best_bb.bounding_box).unsqueeze(0), 
                          torch.tensor(bb.bounding_box).unsqueeze(0))

            if iou > iou_threshold:
                remove_items.append(bb)
        bounding_boxes = [bb for bb in bounding_boxes if bb not in remove_items]
    return filtered_bb
