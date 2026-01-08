from typing import Any

import numpy as np
from numpy import floating

from app.node import Node


class MSELoss(Node):

    def __init__(self,
                 predictions: Node,
                 targets: Node,
                 node_id: int | None = None,
                 name: str | None = None
                 ) -> None:

        super().__init__(predictions, targets, node_type="MSELoss", node_id=node_id, name=name)

    def forward(self) -> floating[Any]:

        pred_val = self.parents[0].forward()
        target_val = self.parents[1].forward()

        # L = mean((predictions - targets)^2)
        diff = pred_val - target_val
        self.value = np.mean(diff ** 2)
        return self.value

    def backward(self, grad: np.ndarray | None = None) -> None:

        if grad is None:
            grad = np.ones_like(self.value)

        self.grad = grad

        pred_val = self.parents[0].value
        target_val = self.parents[1].value

        n = pred_val.size
        # 2 * (predictions - targets) / n
        grad_pred = grad * 2 * (pred_val - target_val) / n

        self.parents[0].backward(grad_pred)


class CrossEntropyLoss(Node):

    def __init__(self, predictions: Node, targets: Node, epsilon: float = 1e-7) -> None:
        super().__init__(predictions, targets, node_type='CrossEntropyLoss')
        self.epsilon: float = epsilon

    def forward(self) -> np.ndarray:
        predictions_node, targets_node = self.parents

        if predictions_node.value is None:
            predictions_node.forward()
        if targets_node.value is None:
            targets_node.forward()

        preds = np.clip(predictions_node.value, self.epsilon, 1 - self.epsilon)
        targets = targets_node.value

        # One-hot encode
        if targets.ndim == 1:
            n_classes = preds.shape[1]
            targets_onehot = np.eye(n_classes)[targets.astype(int)]
        else:
            targets_onehot = targets

        self.targets_onehot = targets_onehot

        # -sum(y_true * log(y_pred))
        batch_size = preds.shape[0]
        loss = -np.sum(targets_onehot * np.log(preds)) / batch_size

        self.value = np.array([loss], dtype=np.float32)
        return self.value

    def backward(self, grad: np.ndarray | None = None) -> None:
        if grad is None:
            grad = np.ones_like(self.value)

        predictions_node, targets_node = self.parents

        # Gradient: (y_pred - y_true) / batch_size
        batch_size = predictions_node.value.shape[0]
        preds = np.clip(predictions_node.value, self.epsilon, 1 - self.epsilon)

        grad_predictions = -(self.targets_onehot / preds) / batch_size

        if predictions_node.grad is None:
            predictions_node.grad = grad_predictions * grad
        else:
            predictions_node.grad += grad_predictions * grad

        if hasattr(predictions_node, 'backward'):
            predictions_node.backward(grad_predictions * grad)


__all__ = ['MSELoss', 'CrossEntropyLoss']

