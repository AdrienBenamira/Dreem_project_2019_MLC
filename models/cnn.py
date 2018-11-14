from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.cnn import output_size_seq_conv_layer

__all__ = ['CNN']

use_cuda = torch.cuda.is_available()


class CNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, in_channels: int, number_groups: int = 4, hidden_channels: List[int] = None,
                 kernel_sizes: List[int] = None, size_groups: int = 1):
        """
        Apply convolutions to a signal. Inspired from the Wavenet architecture
        (https://deepmind.com/blog/wavenet-generative-model-raw-audio)
        Args:
            in_features: Number of samples of the input
            out_features: Number of classes
            in_channels: Number of channels in the beginning
            number_groups: Number of groups. Each groups has `size_group` layers.
            hidden_channels (list): For each group, the number of channel to use. Default: 5 for all groups.
            kernel_sizes (list): The size of the kernels for each group. Default: 3 for all groups.
            size_groups: Number of layers per group.
        """
        super(CNN, self).__init__()
        assert hidden_channels is None or len(
            hidden_channels) == number_groups, "The length of hidden_channels must be the number of layers."
        assert kernel_sizes is None or len(
            kernel_sizes) == number_groups, "The length of kernel_sizes must be the number of layers."
        self.in_features = in_features
        self.out_features = out_features
        self.in_channels = in_channels
        self.number_groups = number_groups
        self.hidden_channels = hidden_channels if hidden_channels is not None else [5 for _ in range(number_groups)]
        self.kernel_sizes = kernel_sizes if kernel_sizes is not None else [3 for _ in range(number_groups)]
        self.size_groups = size_groups
        self.conv_layers, self.hidden_features = self.layers()
        self.fc = nn.Linear(self.hidden_features * self.hidden_channels[-1], 1000)
        self.fc2 = nn.Linear(1000, self.out_features)

    def layers(self):
        """
        Generates all layers for the network
        """
        layers = []
        prev_n_channel = self.in_channels
        for k in range(self.number_groups):
            for g in range(self.size_groups):
                # Use dilation to use the signal at different granularity (diff freq)
                conv_transformation = nn.Conv1d(prev_n_channel, self.hidden_channels[k], self.kernel_sizes[k],
                                                dilation=2 ** k)
                conv_gating = nn.Conv1d(prev_n_channel, self.hidden_channels[k], self.kernel_sizes[k], dilation=2 ** k)
                if use_cuda:
                    conv_transformation, conv_gating = conv_transformation.cuda(), conv_gating.cuda()
                prev_n_channel = self.hidden_channels[k]
                # Use multiplicative activations for each channel
                # z = tanh(w_f * x) . sigmoid(w_g * x) where * is the convolution operator and . is the product.
                # We use 2 convolution filters for each channel.
                layers.append({
                    'transfo': conv_transformation,
                    'gate': conv_gating
                })
        size_output = output_size_seq_conv_layer(self.in_features, list(map(lambda x: x['transfo'], layers)))
        return layers, size_output

    def forward(self, x):
        x = x.to(dtype=torch.float)
        for k, layer in enumerate(self.conv_layers):
            tranfo_filter = torch.tanh(layer['transfo'](x))
            gate = torch.sigmoid(layer['gate'](x))
            x = tranfo_filter.mul(gate)  # + x if we want to use residuals
        x = x.view(x.size(0), -1)
        out = self.fc2(F.relu(self.fc(x)))
        return F.relu(out)
