import torch
import torch.nn.functional as F


def filter_smooth_l1(target, pred, labels, ignore_idx=-1):
    filter_target = target[labels != ignore_idx, :]
    filter_pred = pred[labels != ignore_idx, :]
    loss = F.smooth_l1_loss(filter_target, filter_pred, reduction="none")
    return torch.sum(loss) / len(filter_target)
