import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MLPClassifier']


class MLPClassifier(nn.Module):
    """
    Simple 2 layers perceptron classifier
    """
    def __init__(self, in_features: int, out_features: int, hidden_features: int = 100):
        super(MLPClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = x.to(dtype=torch.float)
        x = F.dropout(F.relu(self.fc(x)), p=0.5)
        return F.softmax(self.fc2(x), dim=-1)
