import torch
from .utils import random_choice
from .box_functions import jaccard


def sample_labels(labels, n_sample=256, pos_ratio=0.5):
    n_pos = pos_ratio * n_sample

    pos_index = torch.where(labels == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = random_choice(pos_index, size=int(len(pos_index) - n_pos))
        labels[disable_index] = -1

    n_neg = n_sample - torch.sum(labels == 1)
    neg_index = torch.where(labels == 0)[0]
    if len(neg_index) > n_neg:
        disable_index = random_choice(neg_index, size=int(len(neg_index) - n_neg))
        labels[disable_index] = -1

    return labels


def create_anchor_labels(anchors, gt, img_size, pos_iou_threshold=0.7, neg_iou_threshold=0.3,
                         n_sample=256, pos_ratio=0.5):
    index_inside = torch.where((anchors[:, 0] >= 0) &
                               (anchors[:, 1] >= 0) &
                               (anchors[:, 2] <= img_size) &
                               (anchors[:, 3] <= img_size))[0]

    labels = -1 * torch.ones((len(index_inside), )).int()
    valid_anchor_boxes = anchors[index_inside]

    ious = jaccard(valid_anchor_boxes, gt)
    
    argmax_ious = ious.argmax(dim=1)
    max_ious = ious[torch.arange(len(index_inside)), argmax_ious]
    
    gt_argmax_ious = ious.argmax(dim=0)
    
    gt_max_ious = ious[gt_argmax_ious, torch.arange(ious.shape[1])]
    gt_argmax_ious = torch.where(ious == gt_max_ious)[0]
    
    labels[max_ious < neg_iou_threshold] = 0
    labels[gt_argmax_ious] = 1
    labels[max_ious >= pos_iou_threshold] = 1    
    
    labels = sample_labels(labels, n_sample, pos_ratio)
    
    locs = bbox2loc(valid_anchor_boxes, gt[argmax_ious])

    anchor_labels = -1 * torch.ones((len(anchors),)).int()
    anchor_labels[index_inside] = labels
    anchor_labels = anchor_labels.long().to(device)

    anchor_locations = torch.zeros_like(anchors)
    anchor_locations[index_inside, :] = locs
    anchor_locations = anchor_locations.to(device)
    
    return anchor_labels, anchor_locations


def create_target_labels(rois, gt_boxes, label):
    n_sample = 128
    pos_ratio = 0.25
    pos_iou_thresh = 0.5
    neg_iou_thresh_hi = 0.5
    neg_iou_thresh_lo = 0.0
    loc_normalize_mean = torch.tensor([0.0, 0.0, 0.0, 0.0]).view((1, 4)).float().to(device)
    loc_normalize_std = torch.tensor([1.0, 1.0, 1.0, 1.0]).view((1, 4)).float().to(device)

    # Rois comes from the network, we need to disable the grad tracing,
    # since we do some ops which are not differentiable
    with torch.no_grad():          
        pos_roi_per_image = np.round(n_sample * pos_ratio)
        iou = jaccard(rois, gt_boxes)

        gt_assignment = iou.argmax(dim=1)
        max_iou = iou.max(axis=1)[0]
        
        gt_roi_label = label[gt_assignment]

        pos_index = torch.where(max_iou >= pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, len(pos_index)))

        if len(pos_index) > 0:
            pos_index = random_choice(pos_index, pos_roi_per_this_image)

        neg_index = torch.where((max_iou < neg_iou_thresh_hi) & (max_iou >= neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image, len(neg_index)))

        if len(neg_index) > 0:
            neg_index = random_choice(neg_index, neg_roi_per_this_image)

        keep_index = torch.cat([pos_index, neg_index], dim=0)

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels becomes background
        sample_roi = rois[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, gt_boxes[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc - loc_normalize_mean) / loc_normalize_std
    return sample_roi, gt_roi_loc, gt_roi_label
