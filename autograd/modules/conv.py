from .base import Module
from ..tensor import Tensor
import numpy as np
from typing import Union


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple], **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2. / in_channels),
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros(out_channels),
            requires_grad=True
        )
        self.padding = kwargs.get('padding', 0)
        self.stride = kwargs.get('stride', 1)
        self.dilation = kwargs.get('dilation', 1)
        self.groups = kwargs.get('groups', 1)
        self.padding_mode = kwargs.get('padding_mode', 'zeros')

    def forward(self, x: Tensor):
        return self._conv2d(x, self.weight, self.bias, self.padding, self.stride, self.dilation, self.groups)
    
    def _conv2d(self, x: Tensor, weight: Tensor, bias: Tensor, padding: int, stride: int, dilation: int, groups: int):
        batch_size, in_channels, height, width = x.data.shape
        out_channels, _, kernel_size, _ = weight.data.shape
        if (in_channels != self.in_channels) | (x.data.ndim != 4):
            raise ValueError(f"Expected input of shape (batch_size, {self.in_channels}, height, width), but got {x.data.shape}")
        if kernel_size != self.kernel_size:
            raise ValueError(f"Expected kernel of size {self.kernel_size}, but got {kernel_size}")
        if groups != 1:
            raise NotImplementedError("Groups > 1 is not supported yet")
        if stride != 1:
            raise NotImplementedError("Stride > 1 is not supported yet")
        if dilation != 1:
            raise NotImplementedError("Dilation > 1 is not supported yet")
        if padding != 0:
            raise NotImplementedError("Padding > 0 is not supported yet")
        if self.padding_mode != 'zeros':
            raise NotImplementedError("Padding mode != 'zeros' is not supported yet")
        x_unf = self._im2col(x, kernel_size, stride, padding)
        weight_unf = weight.data.reshape(out_channels, -1)
        out_unf = x_unf @ weight_unf.T + bias.data
        out = out_unf.reshape(batch_size, out_channels, height, width)
        return Tensor(out)
    
    def _im2col(self, x: Tensor, kernel_size: int, stride: int, padding: int):
        batch_size, in_channels, height, width = x.data.shape
        out_height = (height + 2 * padding - kernel_size) // stride + 1
        out_width = (width + 2 * padding - kernel_size) // stride + 1
        x_unf = np.zeros((batch_size, in_channels, kernel_size, kernel_size, out_height, out_width))
        for i in range(kernel_size):
            for j in range(kernel_size):
                x_unf[:, :, i, j, :, :] = x.data[:, :, i:i + out_height * stride:stride, j:j + out_width * stride:stride]
        return x_unf.reshape(batch_size, in_channels * kernel_size * kernel_size, -1)
    
    def parameters(self):
        return [self.weight, self.bias]