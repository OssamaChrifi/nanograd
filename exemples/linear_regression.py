from autograd import *
import numpy as np
from typing import Union

class Linear(Function):
    def __init__(self, in_features : Union[float, int], out_features: Union[float, int]):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2. / in_features),
            requires_grad=True
            )
        self.bias = Tensor(
            np.zeros(out_features),
            requires_grad=True
            )

    def forward(self, x: Tensor):
        if x.data.shape != (x.data.shape[0], self.in_features):
            raise ValueError(f"Expected input of shape (*, {self.in_features}), but got {x.data.shape}")
        if x.data.ndim != 2:
            raise ValueError("Expected input to be a 2D tensor (batch_size, in_features)")
        self.input = x
        return x @ self.weight + self.bias
    
        
    