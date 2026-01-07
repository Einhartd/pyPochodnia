import numpy as np
from app.node import Variable
from app.node.operations.arithmetic import MatMul, Add


class Dense:
    """
    Dense (Fully Connected) layer.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 weight_init: str = "xavier",
                 use_bias: bool = True,
                 name: str = "Dense"
                 ):
        """
        Initialize Dense layer.

        :param input_size: Number of input features
        :param output_size: Number of output features
        :param weight_init: Weight initialization method ('xavier', 'he', 'normal')
        :param use_bias: Whether to use bias
        :param name: Name of the layer
        """
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.name = name

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

        self.W = Variable(
            value=W.astype(np.float32),
            requires_grad=True,
            name=f"{name}_W"
        )

        # Initialize bias
        if use_bias:
            self.b = Variable(
                value=np.zeros((1, output_size), dtype=np.float32),
                requires_grad=True,
                name=f"{name}_b"
            )
        else:
            self.b = None

    def forward(self, x: Variable):
        """
        Forward pass through the layer.

        :param x: Input variable
        :return: Output node
        """
        # Compute x @ W
        output = MatMul(x, self.W, name=f"{self.name}_matmul")

        # Add bias if present
        if self.use_bias:
            output = Add(output, self.b, name=f"{self.name}_add_bias")

        return output

    def __call__(self, x: Variable):
        """
        Allow calling the layer like a function.
        """
        return self.forward(x)

    def parameters(self):
        """
        Get all trainable parameters of the layer.

        :return: List of Variable nodes (weights and biases)
        """
        if self.use_bias:
            return [self.W, self.b]
        else:
            return [self.W]

    def zero_grad(self):
        """
        Reset gradients of all parameters to zero.
        """
        self.W.zero_grad()
        if self.use_bias:
            self.b.zero_grad()

