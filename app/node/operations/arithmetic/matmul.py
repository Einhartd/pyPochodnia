import numpy as np
from typing import Tuple
from app.node import Node


class MatMul(Node):

    def __init__(self,
                 a: Node,
                 b: Node,
                 node_id: int | None = None,
                 name: str | None = None
                 ) -> None:

        super().__init__(a, b, node_type="MatMul", node_id=node_id, name=name)

        self.a_shape: Tuple[int, ...] | None = None
        self.b_shape: Tuple[int, ...] | None = None

    def forward(self) -> np.ndarray:

        a_val = self.parents[0].forward()
        b_val = self.parents[1].forward()

        self.a_shape = a_val.shape
        self.b_shape = b_val.shape

        self.value = np.matmul(a_val, b_val)
        return self.value

    def backward(self, grad: np.ndarray | None = None) -> None:

        if grad is None:
            grad = np.ones_like(self.value)

        self.grad = grad

        a_val = self.parents[0].value
        b_val = self.parents[1].value

        # d(A @ B)/dA = grad @ B^T
        # d(A @ B)/dB = A^T @ grad
        grad_a = np.matmul(grad, b_val.T)
        grad_b = np.matmul(a_val.T, grad)

        self.parents[0].backward(grad_a)
        self.parents[1].backward(grad_b)

