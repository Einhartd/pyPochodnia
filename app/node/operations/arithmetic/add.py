import numpy as np
from app.node import Node

class Add(Node):
    """
    Addition operation node for computational graphs.
    """

    def __init__(self,
                 a: Node,
                 b: Node,
                 id: int | None = None,
                 name: str | None = None
            ):

        super().__init__(a, b, type="Add", id=id, name=name)

        self.a_shape = None
        self.b_shape = None

    def forward(self):
        """
        Forward pass for addition operation.
        """
        a_val = self.children[0].forward()
        b_val = self.children[1].forward()

        self.a_shape = a_val.shape
        self.b_shape = b_val.shape

        self.value = a_val + b_val
        return self.value

    def backward(self, grad: np.ndarray | None = None):
        """
        Backward pass for addition operation.
        
        :param grad: gradient from the subsequent node
        :type grad: np.ndarray | None
        """
        if grad is None:
            grad = np.ones_like(self.value)

        self.grad = grad

        grad_a = grad
        grad_b = grad

        grad_a = self._unbroadcast(grad_a, self.a_shape)
        grad_b = self._unbroadcast(grad_b, self.b_shape)

        self.children[0].backward(grad_a)
        self.children[1].backward(grad_b)

    def _unbroadcast(self, grad: np.array, target_shape: tuple) -> np.array:

        while grad.ndim > len(target_shape):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad