import numpy as np
from typing import Tuple, Optional
from .base import Function

class Add(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data + y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, grad_output

class Sub(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data - y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, -grad_output
    
class Mul(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data * y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.input
        return grad_output * y.data, grad_output * x.data
    
class Div(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data / y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.input
        return grad_output / y.data, -grad_output * x.data / (y.data ** 2)
    
class Pow(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data ** y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.input
        return grad_output * y.data * (x.data ** (y.data - 1)), grad_output * (x.data ** y.data) * np.log(x.data)
    
class MatMul(Function):
    def forward(self, x, y) -> np.ndarray:
        self.input = (x, y)
        return x.data @ y.data
    
    def backward(self, grad_output : Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        x, y = self.input
        return grad_output @ y.data.T, x.data.T @ grad_output
    
class Sum(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return np.array([x.data.sum()])
    
    def backward(self, grad_output : Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output * np.ones_like(x.data)
    
class Mean(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return np.array([x.data.mean()])
    
    def backward(self, grad_output : Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output * np.ones_like(x.data) / x.data.size

class Exp(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return np.exp(x.data)
    
    def backward(self, grad_output : Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output * np.exp(x.data)

class Log(Function):
    def forward(self, x) -> np.ndarray:
        self.input = x
        return np.log(x.data)
    
    def backward(self, grad_output : Optional[np.ndarray]) -> np.ndarray:
        x = self.input
        return grad_output / x.data
    
