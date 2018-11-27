from models import FCN
import torch.nn as nn


class AE(nn.Module):
    def __init__(self, n_inputs, size_bottleneck=100, dropout=0.5):
        super(AE, self).__init__()
        self.encoder = FCN([n_inputs, n_inputs, n_inputs//2, size_bottleneck], dropout=dropout)
        self.decoder = FCN([size_bottleneck, n_inputs//2, n_inputs, n_inputs], dropout=dropout)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y, z
