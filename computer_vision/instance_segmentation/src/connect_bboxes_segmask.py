import os
import cv2
import numpy as np


def calc_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


def get_max_box_idx(seg_bbox, bboxes):
    max_iou = 0.0
    max_idx = -1

    for i in range(len(bboxes)):
        iou = calc_iou(seg_bbox, bboxes[i])

        if max_iou < iou:
            max_iou = iou
            max_idx = i

    return max_idx, max_iou


def get_bbox_from_mask(object_mask):
    # Since there might be disconnects between pixels of the same objects,
    # we merge the location of min/max of all boxes to get the largest

    cnts,_ = cv2.findContours(object_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        boxes.append([x,y, x+w,y+h])

    boxes = np.asarray(boxes)
    left, top = np.min(boxes, axis=0)[:2]
    right, bottom = np.max(boxes, axis=0)[2:]

    seg_bbox = [left, top, right, bottom]
    return seg_bbox


def get_boxes_from_segmentation(seg_img):
    assert len(seg_img.shape) == 2, seg_img.shape
    
    seg_bboxes = []
    for cat in np.unique(seg_img):
        # id 0 is background and id 220 seems to be boundries id
        if cat == 220 or cat == 0 :
            continue

        mask = (seg_img == cat).astype(np.uint8)
        object_mask = cv2.bitwise_and(seg_img, seg_img, mask=mask)

        seg_bbox = get_bbox_from_mask(object_mask)
        seg_bboxes.append({"bbox": seg_bbox, "id": cat})
    return seg_bboxes


def connect_bboxes_segmask_func(annotation_dict, seg_img):
    bboxes = [d["bbox"] for d in annotation_dict]
    seg_bboxes = get_boxes_from_segmentation(seg_img)

    if len(bboxes) != len(seg_bboxes):
        # Miss-match between boxes number of seg-boxes and bboxes
        return {}, False

    found_idx = set()
    for i in range(len(seg_bboxes)):
        max_idx, max_iou = get_max_box_idx(seg_bboxes[i]["bbox"], bboxes)

        if max_iou < 0.5:
            # The max iou between seg-box and all bboxes is low
            return {}, False

        if max_idx in found_idx:
            # Already matched the idx with a box
            return {}, False
        found_idx.add(max_idx)

        annotation_dict[i]["seg_mask_id"] = int(seg_bboxes[i]["id"])

    return annotation_dict, True


def add_segmask_ids(annotation_dicts, path):
    # Since VOC is not really made for instance-seg there will be a lot of miss-matches
    # between segmentations and bboxes.
    # Therefore we will filter away any example which seems to have incorrect seg-ids and bboxes

    num_discard = 0
    for i in range(len(annotation_dicts)):
        seg_file = os.path.join(path, "SegmentationObject", annotation_dicts[i]["seg_file_name"])
        seg_img = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
        anno, keep = connect_bboxes_segmask_func(annotation_dicts[i]["annotations"], seg_img)

        if keep:
            annotation_dicts[i]["annotations"] = anno
        else:
            num_discard += 1

    discard_rate = num_discard / len(annotation_dicts)
    print(f"Discard-rate: {discard_rate}")

    return annotation_dicts
