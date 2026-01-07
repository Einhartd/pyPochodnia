import numpy as np
from typing import List
from app.node import Variable


class SGD:

    def __init__(self,
                 parameters: List[Variable],
                 learning_rate: float = 0.01,
                 momentum: float = 0.0
                 ):

        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Initialize momentum buffers
        self.velocity = {}
        if momentum > 0:
            for param in parameters:
                self.velocity[id(param)] = np.zeros_like(param.value)

    def step(self):

        for param in self.parameters:
            if param.grad is None:
                continue

            if self.momentum > 0:
                # Update velocity: v = momentum * v - lr * grad
                param_id = id(param)
                self.velocity[param_id] = (
                    self.momentum * self.velocity[param_id] -
                    self.learning_rate * param.grad
                )
                # Update parameter: param += v
                param.value = param.value + self.velocity[param_id]
            else:
                # Simple SGD: param -= lr * grad
                param.value = param.value - self.learning_rate * param.grad

    def zero_grad(self):

        for param in self.parameters:
            param.zero_grad()


class Adam:

    def __init__(self,
                 parameters: List[Variable],
                 learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 epsilon: float = 1e-8
                 ):

        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize moment estimates
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
        self.t = 0   # Time step

        for param in parameters:
            param_id = id(param)
            self.m[param_id] = np.zeros_like(param.value)
            self.v[param_id] = np.zeros_like(param.value)

    def step(self):

        self.t += 1

        for param in self.parameters:
            if param.grad is None:
                continue

            param_id = id(param)
            grad = param.grad

            # Update biased first moment estimate
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected moment estimates
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.value = param.value - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):

        for param in self.parameters:
            param.zero_grad()

