def mean_iou(preds, target):
    return (preds & target).float().sum() / (preds | target).float().sum()
