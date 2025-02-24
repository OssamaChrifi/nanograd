import numpy as np
from typing import Optional
from .functions import Add, Sub, Mul, Div, Pow, MatMul, \
    Sum, Mean, Exp, Log

class Tensor:
    """
    Tensor class to store data and gradient
    """
    def __init__(self, 
                 data : np.ndarray, 
                 requires_grad : bool = False):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad = np.zeros_like(data, dtype=float) if requires_grad else None
        self.requires_grad = requires_grad
        self._ctx = None

    def __add__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Add.apply(self, other)
    
    def __mul__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Mul.apply(self, other)
    
    def __sub__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Sub.apply(self, other)
    
    def __truediv__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Div.apply(self, other)
    
    def __pow__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Pow.apply(self, other)
    
    def __matmul__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return MatMul.apply(self, other)
    
    def __radd__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Add.apply(self, other)
    
    def __rmul__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Mul.apply(self, other)
    
    def __rsub__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Sub.apply(self, other)
    
    def __rtruediv__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Div.apply(self, other)

    def __pow__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Pow.apply(self, other)
    
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    def sum(self) -> 'Tensor':
        return Sum.apply(self)
    
    def mean(self) -> 'Tensor':
        return Mean.apply(self)
    
    def exp(self) -> 'Tensor':
        return Exp.apply(self)
    
    def log(self) -> 'Tensor':
        return Log.apply(self)

    def backward(self, grad_output : Optional[np.ndarray] = None) -> None:
        if grad_output is None:
            grad_output = np.ones_like(self.data, dtype=float)
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data, dtype=float)
            self.grad = self.grad + grad_output
        if self._ctx is not None:
            grads_input = self._ctx.backward(grad_output)
            try:
                for tensor, grad in zip(self._ctx.input, grads_input):
                    tensor.backward(grad)
            except:
                self._ctx.input.backward(grads_input)