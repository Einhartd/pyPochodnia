import numpy as np
from .node import Node

class Variable(Node):

    def __init__(
            self, 
            value: np.ndarray | None = None,
            requires_grad: bool = False,
            node_id: int | None = None,
            name: str | None = None
        ):

        super().__init__(node_type="Variable", node_id=node_id, name=name)
        self.value: np.ndarray | None = np.array(value, dtype=np.float32) if value is not None else None
        self.requires_grad: bool = requires_grad

    def forward(self):

        return self.value

    def backward(self, grad: np.ndarray | None = None):

        if self.requires_grad:
            if grad is None:
                grad = np.ones_like(self.value)

            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad
        return

    def zero_grad(self):

        if self.requires_grad:
            self.grad = np.zeros_like(self.value)
