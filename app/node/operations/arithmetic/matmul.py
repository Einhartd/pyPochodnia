import numpy as np
from app.node import Node


class MatMul(Node):

    def __init__(self,
                 a: Node,
                 b: Node,
                 node_id: int | None = None,
                 name: str | None = None
                 ):

        super().__init__(a, b, node_type="MatMul", node_id=node_id, name=name)

        self.a_shape = None
        self.b_shape = None

    def forward(self):

        a_val = self.children[0].forward()
        b_val = self.children[1].forward()

        self.a_shape = a_val.shape
        self.b_shape = b_val.shape

        self.value = np.matmul(a_val, b_val)
        return self.value

    def backward(self, grad: np.ndarray | None = None):

        if grad is None:
            grad = np.ones_like(self.value)

        self.grad = grad

        a_val = self.children[0].value
        b_val = self.children[1].value

        # d(A @ B)/dA = grad @ B^T
        # d(A @ B)/dB = A^T @ grad
        grad_a = np.matmul(grad, b_val.T)
        grad_b = np.matmul(a_val.T, grad)

        self.children[0].backward(grad_a)
        self.children[1].backward(grad_b)

