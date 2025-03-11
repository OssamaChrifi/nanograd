from typing import List
from autograd import Tensor
import numpy as np

class SGD:
    def __init__(self, parameters: List[Tensor], learning_rate: float, momentum: float = 0.0):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.velocities = [np.zeros_like(param.data) for param in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * param.grad
            param.data += self.velocities[i]
            param.grad = 0
        
