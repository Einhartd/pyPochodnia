import numpy as np
from .node import Node

class Constant(Node):
    """
    Constant node representing fixed values in computational graphs.
    """
    def __init__(self, 
                 value: np.ndarray | None,
                 id: int | None = None,
                 name: str | None = None
            ):

        super().__init__(type="Constant", id=id, name=name)
        self.value: np.ndarray = np.array(value, dtype=np.float16)

    def forward(self):
        """
        Forward pass for Constant node simply returns its value.
        """
        return self.value

    def backward(self, grad: np.ndarray | None = None):
        """
        Backward pass for Constant node does nothing as constants do not have gradients.
        """
        return
