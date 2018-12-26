from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.cnn import output_size_seq_conv_layer, output_size_conv2d_layer

__all__ = ['CNN', 'SimpleCNN']

use_cuda = torch.cuda.is_available()


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5)
        self.conv3 = nn.Conv2d(40, 8, kernel_size=5)
        self.fc1 = nn.Linear(440, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = x.to(dtype=torch.float)
        x = F.relu(F.max_pool2d(self.conv1(x), 5))
        x = F.relu(F.max_pool2d(self.conv2(x), 5))
        x = F.relu(F.max_pool2d(self.conv3(x), 5))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=1)


class CNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, in_channels: int, number_groups: int = 4,
                 hidden_channels: List[int] = None,
                 kernel_sizes: List[int] = None, kernel_pooling: List[int] = None, size_groups: int = 1):
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
        self.kernel_pooling = kernel_pooling if kernel_pooling is not None else [10 for _ in range(number_groups)]
        self.size_groups = size_groups
        self.conv_layers, self.hidden_features = self.layers()
        # conv2d = nn.Conv2d(in_channels=1, out_channels=400, kernel_size=(20, 30))
        # pool2d = nn.MaxPool2d((10, 10))
        # self.conv2d = nn.Sequential(
        #     conv2d,
        #     pool2d,
        #     nn.ReLU()
        # )
        # conv_2d_height, conv_2d_width = output_size_conv2d_layer(self.hidden_channels[-1], self.hidden_features, conv2d)
        # conv2d_dim = output_size_conv2d_layer(conv_2d_height, conv_2d_width, pool2d)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_features * self.hidden_channels[-1], 300),
            nn.ReLU(),
            nn.Linear(300, self.out_features),
            nn.Softmax()
        )

    def layers(self):
        """
        Generates all layers for the network
        """
        layers = []
        prev_n_channel = self.in_channels
        for k in range(self.number_groups):
            for g in range(self.size_groups):
                # Use dilation to use the signal at different granularity (diff freq)
                conv = nn.Conv1d(prev_n_channel, self.hidden_channels[k], self.kernel_sizes[k])
                max_pool = nn.MaxPool1d(self.kernel_pooling[k])
                if use_cuda:
                    conv, max_pool = conv.cuda(), max_pool.cuda()
                prev_n_channel = self.hidden_channels[k]
                # Use multiplicative activations for each channel
                # z = tanh(w_f * x) . sigmoid(w_g * x) where * is the convolution operator and . is the product.
                # We use 2 convolution filters for each channel.
                layers.append(conv)
                layers.append(max_pool)
        size_output = output_size_seq_conv_layer(self.in_features, layers)
        return layers, size_output

    def forward(self, x):
        x = x.to(dtype=torch.float)
        for layer in self.conv_layers:
            x = layer(x)
        # x = x.view(x.size(0), 1, self.hidden_channels[-1], self.hidden_features)
        # x = self.conv2d(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
