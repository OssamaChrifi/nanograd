import numpy as np
from typing import Optional, Tuple, Union
from .functions import *

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

    def __rpow__(self, other : 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Pow.apply(self, other)
    
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    @property
    def shape(self):
        return self.data.shape
    
    def sum(self)-> 'Tensor':
        return Sum.apply(self)
    
    def mean(self)-> 'Tensor':
        return Mean.apply(self)
    
    def exp(self)-> 'Tensor':
        return Exp.apply(self)
    
    def log(self)-> 'Tensor':
        return Log.apply(self)
    
    def abs(self)-> 'Tensor':
        return Abs.apply(self)
    
    def flatten(self)-> 'Tensor':
        return Flatten.apply(self)
    
    def reLU(self)-> 'Tensor':
        return ReLU.apply(self)
    
    def sigmoide(self)-> 'Tensor':
        return Sigmoid.apply(self)
    
    def tanh(self)-> 'Tensor':
        return Tanh.apply(self)
    
    def softmax(self)-> 'Tensor':
        return Softmax.apply(self)
    
    def logSoftmax(self)-> 'Tensor':
        return Log.apply(Softmax.apply(self))

    def crossEntropyLoss(self, y : 'Tensor')-> 'Tensor':
        return CrossEntropyLoss.apply(self, y)
    
    def conv2d(self, weight : 'Tensor', bias : 'Tensor', padding : int = 0, stride : int = 1, \
               dilation : int = 1, groups : int = 1, padding_mode: str = 'zeros') -> 'Tensor':
        return Conv2d.apply(self, weight, bias, padding, stride, dilation, groups, padding_mode)
    
    def maxPool2d(self, kernel_size: Union[Tuple[int, int], int], padding: Union[Tuple[int, int], int] = None, \
                  stride: Union[Tuple[int, int], int] = (1, 1)) -> 'Tensor':
        return Maxpool2d.apply(self, kernel_size, padding, stride)

    def backward(self, grad_output : Optional[np.ndarray] = None) -> None:
        if grad_output is None:
            grad_output = np.ones_like(self.data, dtype=float)
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data, dtype=float)
            self.grad = self.grad + grad_output
        if self._ctx is not None:
            grads_input = self._ctx.backward(self.grad)
            try:
                for tensor, grad in zip(self._ctx.input, grads_input):
                    tensor.backward(grad)
            except:
                self._ctx.input.backward(grads_input)