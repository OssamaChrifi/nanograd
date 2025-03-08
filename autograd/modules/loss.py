from .base import Module
from ..tensor import Tensor
import numpy as np

class MSE(Module):
    def forward(self, x: Tensor, y: Tensor)-> Tensor:
        return ((x - y)**2).mean() 
    
class MAE(Module):
    def forward(self, x: Tensor, y: Tensor)-> Tensor:
        return (x - y).abs()
    
class Cross_Entropie(Module):
    def forward(self, x: Tensor, y: Tensor):
        x_proba = x.softmax()
        class_x_proba = Tensor(x_proba.data[np.arange(y.shape[0]), y.data])
        return -class_x_proba.log()


        