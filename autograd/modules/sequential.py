from .base import Module
from ..tensor import Tensor

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
    
    def forward(self, x: Tensor)-> Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    @property
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters)
        return params