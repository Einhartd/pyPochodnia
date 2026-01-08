import numpy as np
from typing import Tuple


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(np.float32)

    return np.mean(y_pred_binary == y_true)


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:

    return accuracy(y_true, y_pred, threshold)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(np.float32)

    # Calculate confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))

    return np.array([[tn, fp], [fn, tp]])


def precision(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:

    cm = confusion_matrix(y_true, y_pred, threshold)
    tp = cm[1, 1]
    fp = cm[0, 1]

    if (tp + fp) == 0:
        return 0.0

    return tp / (tp + fp)


def recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:

    cm = confusion_matrix(y_true, y_pred, threshold)
    tp = cm[1, 1]
    fn = cm[1, 0]

    if (tp + fn) == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:

    prec = precision(y_true, y_pred, threshold)
    rec = recall(y_true, y_pred, threshold)

    if (prec + rec) == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> str:

    acc = accuracy(y_true, y_pred, threshold)
    prec = precision(y_true, y_pred, threshold)
    rec = recall(y_true, y_pred, threshold)
    f1 = f1_score(y_true, y_pred, threshold)
    cm = confusion_matrix(y_true, y_pred, threshold)

    report = []
    report.append("=" * 50)
    report.append("Classification Report")
    report.append("=" * 50)
    report.append(f"Accuracy:  {acc:.4f}")
    report.append(f"Precision: {prec:.4f}")
    report.append(f"Recall:    {rec:.4f}")
    report.append(f"F1 Score:  {f1:.4f}")
    report.append("-" * 50)
    report.append("Confusion Matrix:")
    report.append(f"              Predicted")
    report.append(f"              Neg    Pos")
    report.append(f"Actual  Neg   {cm[0, 0]:<6} {cm[0, 1]:<6}")
    report.append(f"        Pos   {cm[1, 0]:<6} {cm[1, 1]:<6}")
    report.append("=" * 50)

    return "\n".join(report)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:

    mse_val = mse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    rmse_val = rmse(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)

    report = []
    report.append("=" * 50)
    report.append("Regression Report")
    report.append("=" * 50)
    report.append(f"MSE:   {mse_val:.6f}")
    report.append(f"MAE:   {mae_val:.6f}")
    report.append(f"RMSE:  {rmse_val:.6f}")
    report.append(f"R² Score: {r2_val:.6f}")
    report.append("=" * 50)

    return "\n".join(report)

