import numpy as np
from typing import List, Optional, Tuple
from autograd import Tensor

class Adam:
    def __init__(self, parameters: List[Tensor], learning_rate: float = 0.001,\
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.momentums = [np.zeros_like(param.data) for param in parameters]
        self.velocities = [np.zeros_like(param.data) for param in parameters]
        
    def step(self):
        for i, param in enumerate(self.parameters):
            self.momentums[i] = self.betas[0] * self.momentums[i] + (1 - self.betas[0]) * param.grad
            self.velocities[i] = self.betas[1] * self.velocities[i] + (1 - self.betas[1]) * param.grad**2
            m_hat = self.momentums[i] / (1 - self.betas[0])
            v_hat = self.velocities[i] / (1 - self.betas[1])
            param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            param.grad = 0
