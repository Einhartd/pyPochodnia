import numpy as np
from .node import Node

class Constant(Node):
    def __init__(self,
                 value: np.ndarray | None,
                 node_id: int | None = None,
                 name: str | None = None
                 ):

        super().__init__(node_type="Constant", node_id=node_id, name=name)
        self.value: np.ndarray = np.array(value, dtype=np.float32)

    def forward(self):
        return self.value

    def backward(self, grad: np.ndarray | None = None):
        return
