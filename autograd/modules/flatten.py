from .base import Module
from ..tensor import Tensor

class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.flatten()