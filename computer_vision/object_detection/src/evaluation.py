from collections import namedtuple, defaultdict
import torch
import numpy as np
from src.data_transformer import invert_transformation


PredBoundingBox = namedtuple("PredBoundingBox", ["probability", "class_id",
                                                 "classname", "bounding_box"
                                                 ])


class MAP:
    def __init__(self, model, dataset, jaccard_threshold, anchors):
        self.jaccard_threshold = jaccard_threshold
        self.model = model
        self.eps = np.finfo(np.float32).eps
        self.anchors = anchors
        self.dataset = dataset

    @staticmethod
    def voc_ap(rec, prec):
        """Compute VOC AP given precision and recall with the VOC-07 11-point method."""

        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0.0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
        return ap

    def __call__(self):
        self.model.eval()
        aps = defaultdict(list)

        for i in range(len(self.dataset)):
            (x, bb_true, class_true) = self.dataset[i]
            class_true = class_true.squeeze(0) - 1 # -1 to convert it from 1-21 to 0-20

            x = x[None, :, :, :]
            class_hat, bb_hat = self.model(x)
            class_hat = class_hat[0, :, 1:].sigmoid()

            bb_hat = invert_transformation(bb_hat.squeeze(0), self.anchors)
            jacard_values = jaccard(bb_hat.squeeze(0), bb_true.squeeze(0))

            for j in range(len(class_true)):
                overlap = (jacard_values[:, j] > self.jaccard_threshold).nonzero()
                class_true_j = int(class_true[j].detach().cpu().numpy())

                if len(overlap) > 0:
                    class_hat_j = class_hat[overlap[:,0], :]
                    prob, class_id = class_hat_j.max(1)
                    prob, sort_index = torch.sort(prob, descending=True)
                    class_id = class_id[sort_index].detach().cpu().numpy()

                    tp = np.zeros_like(class_id)
                    fp = np.zeros_like(class_id)

                    found = False
                    for d in range(len(class_id)):
                        if found or class_id[d] != class_true[j]:
                            fp[d] = 1.0
                        else:
                            tp[d] = 1.0
                            found = True

                    fp = np.cumsum(fp)
                    tp = np.cumsum(tp)

                    rec = tp
                    prec = tp / np.maximum(tp + fp, self.eps)

                    temp_ap = MAP.voc_ap(rec, prec)
                    aps[class_true_j].append(temp_ap)
                else:
                    aps[class_true_j].append(0)

        res_list = []
        for _, list_value in aps.items():
            res_list.append(sum(list_value) / len(list_value))

        return res_list, sum(res_list) / len(res_list)


def center_to_minmax(box: torch.Tensor) -> float:
    """
    Converting (cx, cy, w, h) to (x1, y1, x2, y2)
    """

    xmin = box[:, 0] - 0.5 * box[:, 2]
    xmax = box[:, 0] + 0.5 * box[:, 2]

    ymin = box[:, 1] - 0.5 * box[:, 3]
    ymax = box[:, 1] + 0.5 * box[:, 3]
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    # Coverting (cx, cy, w, h) to (x1, y1, x2, y2) since its easier to extract min/max coordinates
    temp_box_a, temp_box_b = center_to_minmax(box_a), center_to_minmax(box_b)

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
