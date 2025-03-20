from .base import Module
from ..tensor import Tensor
import numpy as np

class MSE(Module):
    def forward(self, x: Tensor, y: Tensor)-> Tensor:
        return ((x - y)**2).mean() 
    
class MAE(Module):
    def forward(self, x: Tensor, y: Tensor)-> Tensor:
        return (x - y).abs()
    
class CrossEntropyLoss(Module):
    def forward(self, x: Tensor, y: Tensor)-> Tensor:
        return x.crossEntropyLoss(y)


        