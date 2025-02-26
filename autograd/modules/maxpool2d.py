from .base import Module
from typing import Union, Tuple
from ..tensor import Tensor
import numpy as np

class MaxPool2d(Module):
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], **kwargs: Union[int, Tuple[int, int], bool]):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        stride = kwargs.get('stride', self.kernel_size) 
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        padding = kwargs.get('padding', (0, 0)) 
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        dilation = kwargs.get('dilation', (1, 1)) 
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, _, _ = x.data.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = np.pad(x.data, ((0,0), (0,0), (self.padding[0], self.padding[1]), (self.padding[0], self.padding[1])),\
                               mode='constant', constant_values=0)
        else:
            x_padded = x.data
        out_height = (x_padded.shape[2] - k_h) // s_h + 1
        out_width = (x_padded.shape[3] - k_w) // s_w + 1
        windows = np.lib.stride_tricks.sliding_window_view(x_padded, (k_h, k_w), axis=(2, 3))
        windows = windows[:, :, ::s_h, ::s_w, :, :]
        out = np.max(windows, axis=(-2, -1))
        return Tensor(out.reshape(batch_size, channels, out_height, out_width))