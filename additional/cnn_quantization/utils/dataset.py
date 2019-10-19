import torch
from torch.utils.data import Dataset

class MNIST(Dataset):
    def __init__(self, X, y):
        self.y = torch.from_numpy(y).long()
        X = X.reshape(X.shape[0], 1, 28, 28)
        X = X / 255.0
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx,:,:,:], self.y[idx])
