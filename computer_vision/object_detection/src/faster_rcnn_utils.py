import torch
import numpy as np


def generator_anchors(img_size, sub_sample=16, ratios=[0.7, 1, 1.3], anchor_scales=[4, 8, 16], device="cuda"):
    # Subsample-factor: how much have have the feature-map decreased from input to last layer
    # in the backbone. I.e. if input image is 224 and the output from the backbone is 7,
    # sub-sample factor is 32.

    feature_size = img_size // sub_sample
    center_x_array = np.arange(sub_sample, (feature_size + 1) * sub_sample, sub_sample)
    center_y_array = np.arange(sub_sample, (feature_size + 1) * sub_sample, sub_sample)

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
                    w = sub_sample * anchor_scales[j] * np.sqrt(1.0 / ratios[i])
                    anchors[index, 0] = ctr_x - w / 2.0
                    anchors[index, 1] = ctr_y - h / 2.0
                    anchors[index, 2] = ctr_x + w / 2.0
                    anchors[index, 3] = ctr_y + h / 2.0
                    index += 1

    anchors  = torch.from_numpy(anchors).float().to(device)
    return anchors


def loc2bbox(anchors, locs):
    # Converting anchors and predicted location deltas to "actual" bounding-boxes.
    w = anchors[:, 2] - anchors[:, 0]
    h = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * w
    ctr_y = anchors[:, 1] + 0.5 * h
    
    dx = locs[:, 0]
    dy = locs[:, 1]
    dw = locs[:, 2]
    dh = locs[:, 3]
    
    ctr_y = dy * h + ctr_y
    ctr_x = dx * w + ctr_x
    h = torch.exp(dh) * h
    w = torch.exp(dw) * w

    bbox = torch.zeros_like(locs)
    bbox[:, 0] = ctr_x - 0.5 * w
    bbox[:, 1] = ctr_y - 0.5 * h
    bbox[:, 2] = ctr_x + 0.5 * w
    bbox[:, 3] = ctr_y + 0.5 * h
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
