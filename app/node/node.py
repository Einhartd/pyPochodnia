from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Node(ABC):

    _global_id_counter: int = 0

    def __init__(
            self,
            *parents,
            node_type: str,
            node_id: int | None = None,
            name: str | None = None,
        ):

        self.type: str = node_type

        if node_id is None:
            self.id = Node._global_id_counter
            Node._global_id_counter += 1
        else:
            self.id = node_id

        # Always include ID in the name to ensure uniqueness
        if name is not None:
            self.name: str = f"{name}_{self.id}"
        else:
            self.name: str = f"{self.type}_{self.id}"

        self.value: np.ndarray | None = None
        self.grad: np.ndarray | None = None

        self.parents: List[Node|None] = []
        self.parents.extend(parents)

    def __repr__(self) -> str:
        value_str = f"shape={self.value.shape}" if self.value is not None else "None"
        grad_str = f"shape={self.grad.shape}" if self.grad is not None else "None"
        return f"{self.type}('{self.name}', id={self.id}, val={value_str}, grad={grad_str})"

    def __str__(self) -> str:
        # Format value
        if self.value is not None:
            if self.value.size <= 10:
                value_str = np.array2string(self.value, precision=4, suppress_small=True)
            else:
                value_str = f"shape={self.value.shape}, dtype={self.value.dtype}"
        else:
            value_str = "None"

        # Format gradient
        if self.grad is not None:
            if self.grad.size <= 10:
                grad_str = np.array2string(self.grad, precision=4, suppress_small=True)
            else:
                grad_str = f"shape={self.grad.shape}, dtype={self.grad.dtype}"
        else:
            grad_str = "None"

        return (
            f"Node: {self.name}\n"
            f"  ID:      {self.id}\n"
            f"  Type:    {self.type}\n"
            f"  Value:   {value_str}\n"
            f"  Value shape: {self.value.shape}\n"
            f"  Value dtype: {self.value.dtype}\n"
            f"  Grad:    {grad_str}\n"
            f"  Grad shape: {self.grad.shape}\n"
            f"  Grad dtype: {self.grad.dtype}\n"
            f"  Parents: {len(self.parents)}"
        )

    @abstractmethod
    def forward(self) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, grad):
        pass