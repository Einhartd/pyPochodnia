from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Node(ABC):
    """
    Base class for nodes in computational graphs.
    """
    _global_id_counter: int = 0

    def __init__(
            self,
            *children,
            type: str,
            id: int | None = None,
            name: str | None = None,
        ):

        self.type: str = type

        if id is None:
            self.id = Node._global_id_counter
            Node._global_id_counter += 1
        else:
            self.id = id

        self.name: str = name if name is not None else f"{self.type}_{self.id}"

        self.value: np.ndarray | None = None
        self.grad: np.ndarray | None = None

        self.children: List[Node|None] = []
        self.children.extend(children)

    def __repr__(self):
        """
        String representation of the Node.
        """
        value_str: str = f"shape={self.value.shape}, dtype={self.value.dtype}" if self.value is not None else "value=None"
        grad_str: str = f"shape={self.grad.shape}, dtype={self.grad.dtype}" if self.grad is not None else "grad=None"
        return f"{self.type}(name='{self.name}', id={self.id}, value={value_str}, grad={grad_str}, children={len(self.children)})"

    def __str__(self):
        """
        Detailed string representation of the Node.
        """
        if self.value is not None:
            if self.value.size <=10:
                value_str = np.array2string(self.value, precision=4, suppress_small=True)
            else:
                value_str = f"array(shape={self.value.shape}, dtype={self.value.dtype})"
        else:
            value_str = "None"
        if self.grad is not None:
            if self.grad.size <=10:
                grad_str = np.array2string(self.grad, precision=4, suppress_small=True)
            else:
                grad_str = f"array(shape={self.grad.shape}, dtype={self.grad.dtype})"
        else:
            grad_str = "None"
        return f"{self.name}: value={value_str}, grad={grad_str}, children={len(self.children)}"

    @abstractmethod
    def forward(self):
        """
        Perform the forward pass computation.
        """

    @abstractmethod
    def backward(self, grad):
        """
        Perform the backward pass computation.
        """
