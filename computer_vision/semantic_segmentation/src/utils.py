import numpy as np


def batch_mean_iou(predicted_batch_segmentation, gt_batch_segmentation):
    assert(len(predicted_batch_segmentation.shape) == 3)

    iou_values = []
    for i in range(predicted_batch_segmentation.shape[0]):
        iou = mean_iou(predicted_batch_segmentation[i, :, :], 
                       gt_batch_segmentation[i, :, :])

        iou_values.append(iou)
    
    return np.mean(iou_values)


def mean_iou(predicted_segmentation, gt_segmentation):
    classes, num_classes  = union_classes(predicted_segmentation, gt_segmentation)
    _, n_classes_gt = extract_classes(gt_segmentation)
    eval_mask, gt_mask = extract_both_masks(predicted_segmentation, gt_segmentation, classes, num_classes)

    iou_list = list([0]) * num_classes

    for i, _ in enumerate(classes):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]
 
        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        intersect = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        union = np.sum(np.logical_or(curr_eval_mask, curr_gt_mask))
        iou_list[i] = intersect / union
 
    mean_iou_value = np.sum(iou_list) / n_classes_gt
    return mean_iou_value


def extract_both_masks(predicted_segmentation, gt_segmentation, classes, num_classes):
    predicted_mask = extract_masks(predicted_segmentation, classes, num_classes)
    gt_mask   = extract_masks(gt_segmentation, classes, num_classes)
    return predicted_mask, gt_mask


def extract_classes(segmentation):
    classes = np.unique(segmentation)
    num_classes = len(classes)

    return classes, num_classes


def union_classes(predicted_segmentation, gt_segmentation):
    predicted_classes, _ = extract_classes(predicted_segmentation)
    gt_classes, _   = extract_classes(gt_segmentation)

    classes = np.union1d(predicted_classes, gt_classes)
    num_classes = len(classes)

    return classes, num_classes


def extract_masks(segmentation, classes, num_classes):
    h, w  = segmentation_size(segmentation)
    masks = np.zeros((num_classes, h, w))

    for i, c in enumerate(classes):
        masks[i, :, :] = segmentation == c

    return masks


def segmentation_size(segmentation):
    height = segmentation.shape[0]
    width  = segmentation.shape[1]

    return height, width


if __name__ == "__main__":
    # Test cases
    
    segm = np.array([[1,0,0,0,0], [0,0,0,0,0]])
    gt = np.array([[0,0,0,0,0], [0,0,0,0,0]])
    res = mean_iou(segm, gt)
    assert(np.allclose(res, 0.9))

    segm = np.array([[0,0,0,0,0], [0,0,0,0,0]])
    gt = np.array([[1,2,0,0,0], [0,0,0,0,0]])
    res = mean_iou(segm, gt)
    assert(np.allclose(res, np.mean([8.0/10.0, 0, 0])))

    np.random.seed(42)
    segm = np.random.randint(2, size=(2, 100, 100))
    gt = np.random.randint(2, size=(2, 100, 100))
    _ = batch_mean_iou(segm, gt)
