import torch


def random_choice(arr, size, device="cuda"):
    idx = torch.randperm(len(arr), device=device)
    return arr[idx][0:size]


def normal_init(module, mean, stddev):
    module.weight.data.normal_(mean, stddev)
    module.bias.data.zero_()
