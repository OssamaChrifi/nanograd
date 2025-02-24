import numpy as np
from typing import Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from autograd.tensor import Tensor

class Function:
    def __init__(self, *input: Tuple['Tensor']):
        self.input = input

    @classmethod
    def apply(cls, *input : Tuple['Tensor']) -> 'Tensor':
        fn = cls()
        output = fn.forward(*input)
        from autograd.tensor import Tensor
        result = Tensor(output, requires_grad=any(x.requires_grad for x in input))
        result._ctx = fn
        return result

    def forward(self, *input : Tuple['Tensor']) -> 'Tensor':
        raise NotImplementedError
    
    def backward(self, grad_output : Optional[np.ndarray]):
        raise NotImplementedError