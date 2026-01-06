import numpy as np
from .node import Node

class Variable(Node):
    """
    Variable node representing inputs or parameters in computational graphs.
    """
    def __init__(
            self, 
            value: np.ndarray | None = None,
            requires_grad: bool = False,
            id: int | None = None,
            name: str | None = None
        ):

        super().__init__(type="Variable", id=id, name=name)
        self.value: np.ndarray | None = np.array(value, dtype=np.float16) if value is not None else None
        self.requires_grad: bool = requires_grad

    def forward(self):
        """
        Forward pass for Variable node simply returns its value.
        """
        return self.value

    def backward(self, grad: np.ndarray | None = None):
        """
        Backward pass for Variable node accumulates gradients if required.
        
        :param self: state of the object
        :param grad: gradient to be accumulated
        :type grad: np.ndarray | None
        """
        if self.requires_grad:
            if grad is None:
                grad = np.ones_like(self.value)

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad
        return

    def zero_grad(self):
        """
        Resets the gradient to zero.
        """
        if self.requires_grad:
            self.grad = np.zeros_like(self.value)
