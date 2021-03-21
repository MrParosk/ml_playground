import torch


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
    min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def box_area(box: torch.Tensor) -> float:
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])


def jaccard(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    intersection = intersect(box_a, box_b)
    union = box_area(box_a).unsqueeze(1) + box_area(box_b).unsqueeze(0) - intersection
    return intersection / union


def loc2bbox(anchors, locs):
    # Converting anchors and predicted location deltas to "actual" bounding-boxes.

    if anchors.shape[0] == 0:
        return torch.zeros((0, 4), dtype=locs.dtype)

    w = anchors[:, 2] - anchors[:, 0]
    h = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * w
    ctr_y = anchors[:, 1] + 0.5 * h

    dx = locs[:, 0::4]
    dy = locs[:, 1::4]
    dw = locs[:, 2::4]
    dh = locs[:, 3::4]

    ctr_x = dx * w[:, None] + ctr_x[:, None]
    ctr_y = dy * h[:, None] + ctr_y[:, None]
    w = torch.exp(dw) * w[:, None]
    h = torch.exp(dh) * h[:, None]

    bbox = torch.zeros_like(locs)
    bbox[:, 0::4] = ctr_x - 0.5 * w
    bbox[:, 1::4] = ctr_y - 0.5 * h
    bbox[:, 2::4] = ctr_x + 0.5 * w
    bbox[:, 3::4] = ctr_y + 0.5 * h
    return bbox


def bbox2loc(src_bbox, dst_bbox, eps=1e-6, device="cuda"):
    eps = torch.tensor(eps).float().to(device)

    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    height = torch.max(height, eps)
    width = torch.max(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = torch.log(base_height / height)
    dw = torch.log(base_width / width)

    locs = torch.stack((dx, dy, dw, dh), dim=1)
    return locs
