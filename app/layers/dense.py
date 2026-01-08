import numpy as np
from typing import Optional, List
from app.node import Variable, Node
from app.node.operations.arithmetic import MatMul, Add


class Dense:

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 weight_init: str = "xavier",
                 use_bias: bool = True,
                 name: str = "Dense"
                 ) -> None:

        self.input_size: int = input_size
        self.output_size: int = output_size
        self.use_bias: bool = use_bias
        self.name: str = name

        # Initialize weights
        if weight_init == "xavier":
            limit = np.sqrt(6.0 / (input_size + output_size))
            W = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == "he":
            std = np.sqrt(2.0 / input_size)
            W = np.random.normal(0, std, (input_size, output_size))
        elif weight_init == "normal":
            W = np.random.normal(0, 0.01, (input_size, output_size))
        else:
            raise ValueError(f"Unknown weight initialization: {weight_init}")

        self.W: Variable = Variable(
            value=W.astype(np.float32),
            requires_grad=True,
            name=f"{name}_W"
        )

        # Initialize bias
        if use_bias:
            self.b: Variable | None = Variable(
                value=np.zeros((1, output_size), dtype=np.float32),
                requires_grad=True,
                name=f"{name}_b"
            )
        else:
            self.b = None

    def forward(self, x: Node) -> Node:

        # Compute x @ W
        output = MatMul(x, self.W, name=f"{self.name}_matmul")

        # Add bias if present
        if self.use_bias:
            output = Add(output, self.b, name=f"{self.name}_add_bias")

        return output

    def __call__(self, x: Node) -> Node:

        return self.forward(x)

    def parameters(self) -> List[Variable]:

        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]

    def zero_grad(self) -> None:

        self.W.zero_grad()
        if self.use_bias:
            self.b.zero_grad()

