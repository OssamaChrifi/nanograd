from .base import Module
from ..tensor import Tensor
import numpy as np
from typing import Union


class Conv2d(Module):

    PAD_MODES = {
            'zeros': ('constant', {'constant_values': 0}),
            'reflect': ('reflect', {}),
            'replicate': ('edge', {}),
            'circular': ('wrap', {})
        }
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, tuple], **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, *self.kernel_size) * \
                np.sqrt(2. / in_channels * self.kernel_size[0] * self.kernel_size[1]),
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
        out_channels, _, kernel_size_h, kernel_size_w = weight.data.shape
        if (in_channels != self.in_channels) or (x.data.ndim != 4):
            raise ValueError(f"Expected input of shape (batch_size, {self.in_channels}, height, width), but got {x.data.shape}")
        if self.padding_mode not in self.PAD_MODES:
            raise NotImplementedError(f"Padding mode {self.padding_mode} is not supported")
        if groups == 1:
            x_unf, out_height, out_width = self._im2col(x, kernel_size_h, kernel_size_w, stride, padding, dilation)
            weight_unf = weight.data.reshape(out_channels, -1)
            out_unf = x_unf.transpose(0,2,1) @ weight_unf.T + bias.data
            out = out_unf.transpose(0,2,1).reshape(batch_size, out_channels, out_height, out_width)
            return Tensor(out)
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups
        outputs = []

        for g in range(groups):
            x_group_data = x.data[:, g * in_channels_per_group : (g + 1) * in_channels_per_group, :, :]
            x_group = Tensor(x_group_data)
            x_unf_group, out_height, out_width = self._im2col(x_group, kernel_size_h, kernel_size_w, stride, padding, dilation)
            weight_group = weight.data[g * out_channels_per_group : (g + 1) * out_channels_per_group, :, :, :]
            weight_unf_group = weight_group.reshape(out_channels_per_group, -1)
            bias_group = bias.data[g * out_channels_per_group : (g + 1) * out_channels_per_group]
            out_unf_group = x_unf_group.transpose(0,2,1) @ weight_unf_group.T + bias_group
            out_group = out_unf_group.transpose(0,2,1).reshape(batch_size, out_channels_per_group, out_height, out_width)
            outputs.append(out_group)

        out = np.concatenate(outputs, axis=1)
        return Tensor(out)
    
    def _im2col(self, x: Tensor, kernel_size_h: int, kernel_size_w: int, stride: int, padding: int, dilation: int):
        batch_size = x.data.shape[0]
        mode, kwargs = self.PAD_MODES[self.padding_mode]
        x_padded = np.pad(x.data, ((0,0), (0,0), (padding, padding), (padding, padding)), mode=mode, **kwargs)
        kernel_effective_h = dilation * (kernel_size_h - 1) + 1
        kernel_effective_w = dilation * (kernel_size_w - 1) + 1
        out_height = (x_padded.shape[2] - kernel_effective_h) // stride + 1
        out_width = (x_padded.shape[3] - kernel_effective_w) // stride + 1
        windows = np.lib.stride_tricks.sliding_window_view(x_padded, (kernel_effective_h, kernel_effective_w), axis=(2, 3))
        windows = windows[:, :, ::stride, ::stride, :, :]
        windows = windows[:, :, :, :, ::dilation, ::dilation]
        col = windows.transpose(0, 2, 3, 1, 4, 5)  
        col = col.reshape(batch_size, out_height * out_width, -1).transpose(0, 2, 1)  
        return col, out_height, out_width

    @property
    def parameters(self):
        return [self.weight, self.bias]