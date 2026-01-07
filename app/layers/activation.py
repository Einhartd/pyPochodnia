from app.node import Node
from app.node.operations.activation import ReLU, Sigmoid, Tanh, Softmax


class Activation:

    def __init__(self, activation: str, name: str = "Activation"):

        self.activation = activation.lower()
        self.name = name

    def forward(self, x: Node) -> Node:

        if self.activation == 'relu':
            return ReLU(x, name=f"{self.name}_relu")
        elif self.activation == 'sigmoid':
            return Sigmoid(x, name=f"{self.name}_sigmoid")
        elif self.activation == 'tanh':
            return Tanh(x, name=f"{self.name}_tanh")
        elif self.activation == 'softmax':
            return Softmax(x, name=f"{self.name}_softmax")
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def __call__(self, x: Node) -> Node:

        return self.forward(x)

    def parameters(self):

        return []

    def zero_grad(self):

        pass

