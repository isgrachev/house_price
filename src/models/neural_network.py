from collections import OrderedDict
import torch
from torch import nn
from torch.utils.data import TensorDataset
import numpy as np


def to_tensor(arr: np.ndarray):
        """
        Convert np.array to a tensor.

        Args:
        -----
        arr: np.ndarray
            Array to convert

        Returns:
        --------
        torch.float32 tensor
        """
        
        return torch.tensor(arr, dtype=torch.float32)


class NeuralNetwork(nn.Module):
    """
    Neural Network Regressor.

    Args:
    -----
    features_in: int
        Features passed into 1st FC layer
    features_out: int, default = 1
        Features out. Our problem is regression, so default is 1
    max_epochs: int, default = 500
        Epochs for learning iterations
    device: str, default = 'cpu'
        Device, which is scpecific to this instance
    """
    def __init__(self, features_in: int, features_out: int = 1, max_epochs: int = 500, device: str = 'cpu'):
        # model initiation
        super().__init__()
        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(features_in, 2)),
            ('act1', nn.ReLU()),
            ('fc2', nn.Linear(2, 2)),
            ('act2', nn.ReLU()),
            ('outp', nn.Linear(2, features_out))
        ]))
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.max_epochs = max_epochs
        self.device = {'cpu': 'cpu',
                       'cuda': 'cuda' if torch.cuda.is_available() else 'cpu'}

    def fit(self, X, y, X_val = None, y_val = None):
        """
        Fit neural network
        Validation loss is calculated if X_val and y_val are passed.

        Args:
        -----
        X
            Tensor of features
        y
            Tensor of labels
        X_val, default = None
            Tensor of validation features
        y_val, default = None
            Tensor of validation features
        """

        X = to_tensor(X)
        y = to_tensor(y)
        dataset = TensorDataset(X, y)
        val=False
        try:
            X_val = to_tensor(X_val)
            y_val = to_tensor(y_val)
            val_dataset = TensorDataset(X_val, y_val)
            val=True
        except:
            pass
        self.rmse = {"train": [], "validation": []}
        for epoch in range(self.max_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(dataset.tensors[0])
            loss = self.criterion(pred, dataset.tensors[1])
            loss.backward()
            self.optimizer.step()
            self.rmse['train'].append(np.sqrt(loss.item()))

            # calculate loss on validation dataset
            if val:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(val_dataset.tensors[0])
                    val_loss = np.sqrt(self.criterion(val_pred, val_dataset.tensors[1]).item())
                    self.rmse['validation'].append(val_loss)
            
            if epoch % 10 == 0:
                print(f"\nIteration: {epoch}    |   Train RMSE: {self.rmse['train'][epoch]}    |    Validation RMSE: {self.rmse['validation'][epoch]}")

    def predict(self, X):
        """
        Predict labels with neural network

        Args:
        -----
        X
            Tensor of features

        Returns:
        --------
        pred
            Tensor of labels
        """
        return self.model(X)
