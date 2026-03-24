import pytest
import torch
from sklearn.exceptions import UndefinedMetricWarning

from src.metrics import compute_accuracy, compute_auc_roc, compute_f1, compute_all_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect():
    y_true   = torch.tensor([0, 0, 1, 1])
    y_pred   = torch.tensor([0, 0, 1, 1])
    y_scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
    return y_true, y_pred, y_scores


@pytest.fixture
def all_wrong():
    y_true   = torch.tensor([0, 0, 1, 1])
    y_pred   = torch.tensor([1, 1, 0, 0])
    y_scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    return y_true, y_pred, y_scores


@pytest.fixture
def partial():
    # 3/4 correct; TP=1, FP=1, FN=0 for class 1 → precision=0.5, recall=1.0 → F1=2/3
    y_true   = torch.tensor([0, 0, 1, 1])
    y_pred   = torch.tensor([0, 1, 1, 1])
    y_scores = torch.tensor([0.1, 0.4, 0.8, 0.9])
    return y_true, y_pred, y_scores


# ---------------------------------------------------------------------------
# compute_accuracy
# ---------------------------------------------------------------------------

def test_accuracy_perfect(perfect):
    y_true, y_pred, _ = perfect
    assert compute_accuracy(y_true, y_pred) == 1.0


def test_accuracy_all_wrong(all_wrong):
    y_true, y_pred, _ = all_wrong
    assert compute_accuracy(y_true, y_pred) == 0.0


def test_accuracy_partial(partial):
    y_true, y_pred, _ = partial
    assert compute_accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_accuracy_float_tensors():
    y_true = torch.tensor([0.0, 1.0, 0.0, 1.0])
    y_pred = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert compute_accuracy(y_true, y_pred) == 1.0


# ---------------------------------------------------------------------------
# compute_f1
# ---------------------------------------------------------------------------

def test_f1_perfect(perfect):
    y_true, y_pred, _ = perfect
    assert compute_f1(y_true, y_pred) == 1.0


def test_f1_all_wrong(all_wrong):
    y_true, y_pred, _ = all_wrong
    assert compute_f1(y_true, y_pred) == 0.0


def test_f1_partial(partial):
    y_true, y_pred, _ = partial
    # precision = 2/3, recall = 2/2 = 1.0 → F1 = 2*(2/3*1)/(2/3+1) = 4/3 / 5/3 = 4/5 = 0.8
    assert compute_f1(y_true, y_pred) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# compute_auc_roc
# ---------------------------------------------------------------------------

def test_auc_roc_perfect(perfect):
    y_true, _, y_scores = perfect
    assert compute_auc_roc(y_true, y_scores) == 1.0


def test_auc_roc_all_wrong(all_wrong):
    y_true, _, y_scores = all_wrong
    assert compute_auc_roc(y_true, y_scores) == 0.0


def test_auc_roc_random():
    # Uniform scores → AUC ≈ 0.5
    torch.manual_seed(0)
    y_true   = torch.randint(0, 2, (100,))
    y_scores = torch.rand(100)
    auc = compute_auc_roc(y_true, y_scores)
    assert 0.0 <= auc <= 1.0


def test_auc_roc_single_class_warns():
    """sklearn warns (not raises) when only one class is present."""
    from sklearn.exceptions import UndefinedMetricWarning
    y_true   = torch.tensor([1, 1, 1, 1])
    y_scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    with pytest.warns(UndefinedMetricWarning):
        compute_auc_roc(y_true, y_scores)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

def test_all_metrics_keys(perfect):
    y_true, y_pred, y_scores = perfect
    result = compute_all_metrics(y_true, y_pred, y_scores)
    assert set(result.keys()) == {"accuracy", "f1", "auc_roc"}


def test_all_metrics_perfect(perfect):
    y_true, y_pred, y_scores = perfect
    result = compute_all_metrics(y_true, y_pred, y_scores)
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0
    assert result["auc_roc"] == 1.0


def test_all_metrics_returns_floats(perfect):
    y_true, y_pred, y_scores = perfect
    result = compute_all_metrics(y_true, y_pred, y_scores)
    for v in result.values():
        assert isinstance(v, float)
