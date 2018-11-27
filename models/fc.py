import torch
import torch.nn as nn


class FCN(nn.Module):
    """
    Fully connected
    """
    def __init__(self, size_layers, dropout=0.5):
        super(FCN, self).__init__()
        self.dropout = dropout
        self.size_layers = size_layers
        self.layers = self.build_layers()

    def build_layers(self):
        sizes = self.size_layers
        layers = []
        for k in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[k], sizes[k+1]))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(dtype=torch.float)
        return self.layers(x)

