import numpy as np
from app.node import Node


class ReLU(Node):

    def __init__(self,
                 x: Node,
                 node_id: int | None = None,
                 name: str | None = None
                 ) -> None:

        super().__init__(x, node_type="ReLU", node_id=node_id, name=name)

    def forward(self) -> np.ndarray:

        # f(x) = max(0,x)
        x_val = self.parents[0].forward()
        self.value = np.maximum(0, x_val)
        return self.value

    def backward(self, grad: np.ndarray | None = None) -> None:

        if grad is None:
            grad = np.ones_like(self.value)

        self.grad = grad

        x_val = self.parents[0].value

        # Gradient: 1 if x > 0, else 0
        grad_x = grad * (x_val > 0).astype(np.float32)

        self.parents[0].backward(grad_x)


class Sigmoid(Node):

    def __init__(self,
                 x: Node,
                 node_id: int | None = None,
                 name: str | None = None
                 ) -> None:

        super().__init__(x, node_type="Sigmoid", node_id=node_id, name=name)

    def forward(self) -> np.ndarray:

        x_val = self.parents[0].forward()
        # Clip to prevent overflow
        x_val = np.clip(x_val, -500, 500)
        # f(x) = 1 / (1 + exp(-x))
        self.value = 1.0 / (1.0 + np.exp(-x_val))
        return self.value

    def backward(self, grad: np.ndarray | None = None) -> None:

        if grad is None:
            grad = np.ones_like(self.value)

        self.grad = grad

        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        sigmoid_grad = self.value * (1.0 - self.value)
        grad_x = grad * sigmoid_grad

        self.parents[0].backward(grad_x)


class Tanh(Node):

    def __init__(self,
                 x: Node,
                 node_id: int | None = None,
                 name: str | None = None
                 ) -> None:

        super().__init__(x, node_type="Tanh", node_id=node_id, name=name)

    def forward(self) -> np.ndarray:

        x_val = self.parents[0].forward()
        # f(x) = tanh(x)
        self.value = np.tanh(x_val)
        return self.value

    def backward(self, grad: np.ndarray | None = None) -> None:

        if grad is None:
            grad = np.ones_like(self.value)

        self.grad = grad

        # tanh'(x) = 1 - tanh^2(x)
        tanh_grad = 1.0 - self.value ** 2
        grad_x = grad * tanh_grad

        self.parents[0].backward(grad_x)


class Softmax(Node):

    def __init__(self,
                 x: Node,
                 axis: int = -1,
                 node_id: int | None = None,
                 name: str | None = None
                 ) -> None:

        super().__init__(x, node_type="Softmax", node_id=node_id, name=name)
        self.axis: int = axis

    def forward(self) -> np.ndarray:

        x_val = self.parents[0].forward()

        # Subtract max for numerical stability
        x_shifted = x_val - np.max(x_val, axis=self.axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        # exp(x_i) / sum(exp(x_j))
        self.value = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

        return self.value

    def backward(self, grad: np.ndarray | None = None) -> None:

        if grad is None:
            grad = np.ones_like(self.value)

        self.grad = grad

        s = self.value
        grad_x = s * (grad - np.sum(grad * s, axis=self.axis, keepdims=True))

        self.parents[0].backward(grad_x)

