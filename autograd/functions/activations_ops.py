import numpy as np
from typing import Optional
from .base import Function


class ReLU(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return np.maximum(x.data, 0)
    
    def backward(self, grad_output: Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output * (x.data > 0)  
    
class Sigmoid(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return 1 / (1 + np.exp(-x.data))
    
    def backward(self, grad_output: Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output * ((x.data * np.exp(-x.data)) / (1 + np.exp(-x.data)) ** 2)
    
class Tanh(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return np.tanh(x.data)
    
    def backward(self, grad_output: Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output * (1 - np.tanh(x.data) ** 2)
    
class Softmax(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return np.exp(x.data) / np.exp(x.data).sum()
    
    def backward(self, grad_output: Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output * \
            ((x.data * np.exp(x.data) * np.exp(x.data).sum()) / (np.exp(x.data) * np.ones_like(x.shape)))