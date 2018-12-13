from math import floor
from typing import List, Union

from torch.nn import Conv1d, MaxPool1d

conv_or_pool_t = Union[Conv1d, MaxPool1d]

__all__ = ["output_size_seq_conv_layer", "output_size_conv2d_layer"]


def output_size_conv_layer(length: int, layer: conv_or_pool_t) -> int:
    """
    Compute output size from a conv1d layer
    Args:
        length: width of the image
        layer: convolution layer

    Returns: (height_out, width_out)

    """
    if type(layer.kernel_size) == int:
        kernel_size, stride, padding, dilation = layer.kernel_size, layer.stride, layer.padding, layer.dilation
    else:
        kernel_size, stride, padding, dilation = layer.kernel_size[0], layer.stride[0], layer.padding[0], layer.dilation[0]
    length_out = floor((length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    return length_out


def output_size_seq_conv_layer(length: int, sequence: List[conv_or_pool_t]) -> int:
    """
    Get output height and width from a sequence of conv1d
    Args:
        length: length of the input
        sequence: sequence of layers
    Returns:

    """
    size_seq = len(sequence)
    length_out = length
    for k in range(size_seq):
        if type(sequence[k]) == Conv1d or type(sequence[k]) == MaxPool1d:
            length_out = output_size_conv_layer(length_out, sequence[k])
    return length_out


def output_size_conv2d_layer(height, width, layer):
    kernel_size, stride, padding, dilation = layer.kernel_size, layer.stride, layer.padding, layer.dilation
    padding = (padding, padding) if type(padding) == int else padding
    dilation = (dilation, dilation) if type(dilation) == int else dilation
    height_out = floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    width_out = floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
    return height_out, width_out
