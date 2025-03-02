from .base import Module
from ..tensor import Tensor
import numpy as np
from typing import Union, Tuple


class Conv2d(Module):

    PAD_MODES = {
            'zeros': ('constant', {'constant_values': 0}),
            'reflect': ('reflect', {}),
            'replicate': ('edge', {}),
            'circular': ('wrap', {})
        }
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], **kwargs: Union[int, str]):
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

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim == 3:
            x.data.unsqueeze(0)
        in_channels = x.data.shape[1]
        if (in_channels != self.in_channels) or (x.data.ndim != 4):
            raise ValueError(f"Expected input of shape (batch_size, {self.in_channels}, height, width), but got {x.data.shape}")
        return x.conv2d(self.weight, self.bias, self.padding, self.stride, self.dilation, self.groups)
    
    @property
    def parameters(self):
        return [self.weight, self.bias]