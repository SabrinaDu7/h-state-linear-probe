from torch import Tensor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_accuracy(y_true: Tensor, y_pred: Tensor) -> float:
    """Binary classification accuracy."""
    return float(accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy()))


def compute_f1(y_true: Tensor, y_pred: Tensor) -> float:
    """Binary F1 score."""
    return float(f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy()))


def compute_auc_roc(y_true: Tensor, y_scores: Tensor) -> float:
    """AUC-ROC. y_scores are positive-class probabilities (or decision scores)."""
    return float(roc_auc_score(y_true.cpu().numpy(), y_scores.cpu().numpy()))


def compute_all_metrics(y_true: Tensor, y_pred: Tensor, y_scores: Tensor) -> dict[str, float]:
    """Return accuracy, f1, and auc_roc in one call."""
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "f1": compute_f1(y_true, y_pred),
        "auc_roc": compute_auc_roc(y_true, y_scores),
    }
