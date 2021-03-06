import numpy as np


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
    feat_stride = sub_sample

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
    return anchor
