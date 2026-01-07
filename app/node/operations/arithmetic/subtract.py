import numpy as np
from app.node import Node


def _unbroadcast(grad: np.ndarray, target_shape: tuple) -> np.ndarray:

    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Subtract(Node):

    def __init__(self,
                 a: Node,
                 b: Node,
                 node_id: int | None = None,
                 name: str | None = None
                 ):

        super().__init__(a, b, node_type="Subtract", node_id=node_id, name=name)

        self.a_shape = None
        self.b_shape = None

    def forward(self):

        a_val = self.children[0].forward()
        b_val = self.children[1].forward()

        self.a_shape = a_val.shape
        self.b_shape = b_val.shape

        self.value = a_val - b_val
        return self.value

    def backward(self, grad: np.ndarray | None = None):

        if grad is None:
            grad = np.ones_like(self.value)

        self.grad = grad

        # d(a - b)/da = 1, d(a - b)/db = -1
        grad_a = grad
        grad_b = -grad

        grad_a = _unbroadcast(grad_a, self.a_shape)
        grad_b = _unbroadcast(grad_b, self.b_shape)

        self.children[0].backward(grad_a)
        self.children[1].backward(grad_b)

