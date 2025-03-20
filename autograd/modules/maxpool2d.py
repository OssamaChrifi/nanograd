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

    def forward(self, x: Tensor) -> Tensor:
        return x.maxPool2d(self.kernel_size, self.padding, self.stride)