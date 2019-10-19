import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

class Trainer:
    def __init__(self, model, train_loader, test_loader, seed=42, device="cuda", **kwargs):
        self.device = device
        self.model = model
        # self.optimizer = optim.Adam(model.parameters(), **kwargs)
        self.optimizer = optim.SGD(model.parameters(), **kwargs)
        self.lr_sceduler = MultiStepLR(self.optimizer, gamma=0.5, milestones=[150, 250])

        self.loss = nn.CrossEntropyLoss()

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.seed = seed

    def train(self):
        torch.manual_seed(self.seed)
        self.model.train()

        train_loss = 0.0
        for _, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.model.zero_grad()
                output = self.model(x)

                batch_loss = self.loss(output, y)
                batch_loss.backward()
                self.optimizer.step()
                train_loss += batch_loss.cpu().detach().numpy() / x.shape[0]

        self.lr_sceduler.step()
        train_loss = np.round(train_loss / len(self.train_loader), 6)

        return train_loss

    def validate(self):
        self.model.eval()
        torch.manual_seed(self.seed)

        test_loss, test_acc = 0.0, 0.0
        with torch.no_grad():
            for _, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)

                batch_loss = self.loss(output, y)
                test_loss += batch_loss.cpu().detach().numpy() / x.shape[0]

                y_hat = torch.argmax(output, dim=1)
                acc = (y_hat == y).sum().float() / x.shape[0]
                acc = acc.cpu().detach().numpy()
                test_acc += acc

        test_loss = np.round(test_loss / len(self.test_loader), 6)
        test_acc = np.round(test_acc / len(self.test_loader), 4)
        return test_loss, test_acc

    def run_one_epoch(self, epoch):
        train_loss = self.train()
        test_loss, test_acc = self.validate()
        self.model.train()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train loss {train_loss}, test loss {test_loss}, test accuracy {test_acc}")
