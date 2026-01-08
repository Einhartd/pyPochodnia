import numpy as np
from typing import Tuple
from app.node import Node


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:

    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Multiply(Node):

    def __init__(self,
                 a: Node,
                 b: Node,
                 node_id: int | None = None,
                 name: str | None = None
                 ) -> None:

        super().__init__(a, b, node_type="Multiply", node_id=node_id, name=name)

        self.a_shape: Tuple[int, ...] | None = None
        self.b_shape: Tuple[int, ...] | None = None

    def forward(self) -> np.ndarray:

        a_val = self.parents[0].forward()
        b_val = self.parents[1].forward()

        self.a_shape = a_val.shape
        self.b_shape = b_val.shape

        self.value = a_val * b_val
        return self.value

    def backward(self, grad: np.ndarray | None = None) -> None:

        if grad is None:
            grad = np.ones_like(self.value)

        self.grad = grad

        a_val = self.parents[0].value
        b_val = self.parents[1].value

        # d(a * b)/da = b, d(a * b)/db = a
        grad_a = grad * b_val
        grad_b = grad * a_val

        grad_a = _unbroadcast(grad_a, self.a_shape)
        grad_b = _unbroadcast(grad_b, self.b_shape)

        self.parents[0].backward(grad_a)
        self.parents[1].backward(grad_b)

