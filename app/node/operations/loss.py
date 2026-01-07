import numpy as np
from app.node import Node


class MSELoss(Node):

    def __init__(self,
                 predictions: Node,
                 targets: Node,
                 node_id: int | None = None,
                 name: str | None = None
                 ):

        super().__init__(predictions, targets, node_type="MSELoss", node_id=node_id, name=name)

    def forward(self):

        pred_val = self.children[0].forward()
        target_val = self.children[1].forward()

        # L = mean((predictions - targets)^2)
        diff = pred_val - target_val
        self.value = np.mean(diff ** 2)
        return self.value

    def backward(self, grad: np.ndarray | None = None):

        if grad is None:
            grad = np.ones_like(self.value)

        self.grad = grad

        pred_val = self.children[0].value
        target_val = self.children[1].value

        n = pred_val.size
        # 2 * (predictions - targets) / n
        grad_pred = grad * 2 * (pred_val - target_val) / n

        self.children[0].backward(grad_pred)


class CrossEntropyLoss(Node):

    def __init__(self, predictions: Node, targets: Node, epsilon: float = 1e-7):
        super().__init__(predictions, targets, node_type='CrossEntropyLoss')
        self.epsilon = epsilon

    def forward(self):
        predictions_node, targets_node = self.children
        predictions_node.forward()
        targets_node.forward()

        # clip predictions to avoid log(0)
        preds = np.clip(predictions_node.value, self.epsilon, 1 - self.epsilon)

        # -sum(y_true * log(y_pred))
        batch_size = preds.shape[0]
        loss = -np.sum(targets_node.value * np.log(preds)) / batch_size

        self.value = np.array([loss], dtype=np.float32)

    def backward(self, grad):
        predictions_node, targets_node = self.children

        # Gradient: (y_pred - y_true) / batch_size
        batch_size = predictions_node.value.shape[0]
        preds = np.clip(predictions_node.value, self.epsilon, 1 - self.epsilon)

        grad_predictions = -(targets_node.value / preds) / batch_size

        if predictions_node.grad is None:
            predictions_node.grad = grad_predictions * grad
        else:
            predictions_node.grad += grad_predictions * grad

        if hasattr(predictions_node, 'backward'):
            predictions_node.backward(grad_predictions * grad)


__all__ = ['MSELoss']

