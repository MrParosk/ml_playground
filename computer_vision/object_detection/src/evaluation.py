def calc_iou(box_a: list, box_b: list) -> list:
    """
    Boxes are on the format [bx, by, width, height]
    """

    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # Compute the area of the prediction and ground-truth
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    iou = inter_area / (box_a_area + box_b_area - inter_area + 1e-8)
    return iou


def center_2_hw(bounding_box: list) -> list:
    """
    Coverting boxes to (x1, y1) and (x2, y2)
    """

    return [bounding_box[0] - bounding_box[2]/2,
            bounding_box[1] - bounding_box[3]/2,
            bounding_box[0] + bounding_box[2]/2,
            bounding_box[1] + bounding_box[3]/2
            ]


def non_max_suppression(bounding_boxes: list, iou_threshold: float = 0.5) -> list:
    """ Boxes are on the format:
        [0.7954241, 11, 'dog', 55.436745, 23.547615, 126.517395, 196.1788]
        [confidence, class id, class name, bx, by, width, height]
    """

    filtered_bb = []

    while len(bounding_boxes) != 0:
        best_bb = bounding_boxes.pop(0)
        filtered_bb.append(best_bb)

        remove_items = []
        for bb in bounding_boxes:
            if calc_iou(center_2_hw(best_bb[3:]), center_2_hw(bb[3:])) > iou_threshold:
                remove_items.append(bb)

        bounding_boxes = [bb for bb in bounding_boxes if bb not in remove_items]

    return filtered_bb
