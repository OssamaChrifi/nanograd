from .base import Module
from ..tensor import Tensor
import numpy as np
from typing import Union


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple], **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
                np.sqrt(2. / (in_channels * kernel_size * kernel_size)),
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
        batch_size, in_channels, _, _ = x.data.shape
        out_channels, _, kernel_size, _ = weight.data.shape
        if (in_channels != self.in_channels) or (x.data.ndim != 4):
            raise ValueError(f"Expected input of shape (batch_size, {self.in_channels}, height, width), but got {x.data.shape}")
        if groups != 1:
            raise NotImplementedError("Groups > 1 is not supported yet")
        if self.padding_mode != 'zeros':
            raise NotImplementedError("Padding mode != 'zeros' is not supported yet")
        x_unf, out_height, out_width = self._im2col(x, kernel_size, stride, padding, dilation)
        weight_unf = weight.data.reshape(out_channels, -1)
        out_unf = x_unf.transpose(0,2,1) @ weight_unf.T + bias.data
        out = out_unf.transpose(0,2,1).reshape(batch_size, out_channels, out_height, out_width)
        return Tensor(out)
    
    def _im2col(self, x: Tensor, kernel_size: int, stride: int, padding: int, dilation: int):
        batch_size = x.data.shape[0]
        x_padded = np.pad(x.data, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')

        kernel_effective = dilation * (kernel_size - 1) + 1
        out_height = (x_padded.shape[2] - kernel_effective) // stride + 1
        out_width = (x_padded.shape[3] - kernel_effective) // stride + 1
        
        windows = np.lib.stride_tricks.sliding_window_view(x_padded, (kernel_effective, kernel_effective), axis=(2, 3))
        windows = windows[:, :, ::stride, ::stride, :, :]
        windows = windows[:, :, :, :, ::dilation, ::dilation]

        col = windows.transpose(0, 2, 3, 1, 4, 5)  # (batch_size, out_height, out_width, in_channels, kernel_size, kernel_size)
        col = col.reshape(batch_size, out_height * out_width, -1).transpose(0, 2, 1)  # (batch_size, in_channels*kernel_size*kernel_size, out_height*out_width)
        return col, out_height, out_width

    def parameters(self):
        return [self.weight, self.bias]