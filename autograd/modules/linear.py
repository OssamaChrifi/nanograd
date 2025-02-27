import numpy as np
from typing import Union
from ..tensor import Tensor
from .base import Module

class Linear(Module):
    def __init__(self, in_features : Union[float, int], out_features: Union[float, int]):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2. / in_features),
            requires_grad=True
            )
        self.bias = Tensor(
                np.zeros(self.out_features),
                requires_grad=True
                )

    def forward(self, x: Tensor):
        if x.data.ndim != 2 or x.data.shape[1] != self.in_features:
            raise ValueError(f"Expected input of shape (batch_size, {self.in_features}), but got {x.data.shape}")
        return x @ self.weight + self.bias
    
    @property
    def parameters(self):
        return [self.weight, self.bias]
    
        
    