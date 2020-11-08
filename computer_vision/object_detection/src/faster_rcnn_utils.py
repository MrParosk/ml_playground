import torch
import numpy as np


def generator_anchors_old(img_size, sub_sample=16, ratios=[0.7, 1, 1.3], anchor_scales=[4, 8, 16], device="cuda"):
    # Subsample-factor: how much have have the feature-map decreased from input to last layer
    # in the backbone. I.e. if input image is 224 and the output from the backbone is 7,
    # sub-sample factor is 32.

    feature_size = img_size // sub_sample
    center_x_array = np.arange(
        sub_sample, (feature_size + 1) * sub_sample, sub_sample)
    center_y_array = np.arange(
        sub_sample, (feature_size + 1) * sub_sample, sub_sample)

    num_combinations = len(ratios) * len(anchor_scales)
    anchors = np.zeros((feature_size * feature_size * num_combinations, 4))

    index = 0
    for x in range(len(center_x_array)):
        for y in range(len(center_y_array)):
            ctr_x = center_x_array[x] - sub_sample / 2.0
            ctr_y = center_y_array[y] - sub_sample / 2.0

            for i in range(len(ratios)):
                for j in range(len(anchor_scales)):
                    h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
                    w = sub_sample * \
                        anchor_scales[j] * np.sqrt(1.0 / ratios[i])
                    anchors[index, 0] = ctr_x - w / 2.0
                    anchors[index, 1] = ctr_y - h / 2.0
                    anchors[index, 2] = ctr_x + w / 2.0
                    anchors[index, 3] = ctr_y + h / 2.0
                    index += 1

    anchors = torch.from_numpy(anchors).float().to(device)
    return anchors


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    px = base_size / 2.
    py = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)

    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = px - w / 2.
            anchor_base[index, 1] = py - h / 2.
            anchor_base[index, 2] = px + w / 2.
            anchor_base[index, 3] = py + h / 2.
    return anchor_base


def generator_anchors(img_size, sub_sample=16, ratios=[0.7, 1, 1.3], anchor_scales=[4, 8, 16], device="cuda"):
    feat_stride = sub_sample  # img_size // sub_sample

    anchor_base = generate_anchor_base(
        base_size=sub_sample, ratios=ratios, anchor_scales=anchor_scales)

    shift_y = np.arange(0, img_size * feat_stride, feat_stride)
    shift_x = np.arange(0, img_size * feat_stride, feat_stride)

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift = np.stack((shift_x.ravel(),  shift_y.ravel(),
                      shift_x.ravel(), shift_y.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
        shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    anchor = torch.from_numpy(anchor).float().to(device)
    # print(anchor.shape)
    return anchor


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


def random_choice(arr, size, device="cuda"):
    idx = torch.randperm(len(arr), device=device)
    return arr[idx][0:size]


def normal_init(module, mean, stddev):
    module.weight.data.normal_(mean, stddev)
    module.bias.data.zero_()
