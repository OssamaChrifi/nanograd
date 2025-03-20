from .base import Module
from ..tensor import Tensor

class ReLU(Module):
    def forward(self, x:Tensor) -> Tensor:
        return x.reLU()
    
class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoide()
    
class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh
    
class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.softmax()
    
class LogSoftmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.logSoftmax()