from typing import List
from autograd import Tensor

class SGD:
    def __init__(self, parameters: List[Tensor], learning_rate: float):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for param in self.parameters:
            param.data -= self.learning_rate * param.grad
            param.grad = 0
        
