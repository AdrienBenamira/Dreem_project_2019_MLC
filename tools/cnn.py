from math import floor
from typing import Tuple, List, Union
from torch.nn import Conv1d

conv_or_pool_t = Union[Conv1d]

__all__ = ["output_size_seq_conv_layer"]


def output_size_conv_layer(length: int, layer: conv_or_pool_t) -> int:
    """
    Compute output size from a conv1d layer
    Args:
        length: width of the image
        layer: convolution layer

    Returns: (height_out, width_out)

    """
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
        if type(sequence[k]) == Conv1d:
            length_out = output_size_conv_layer(length_out, sequence[k])
    return length_out
