"""Simple evaluation metrics for ML models, written from scratch.

Classification metrics assume binary labels (0 and 1) unless a
different positive label is passed in.
"""
import math


def _validate(y_true, y_pred):
    """Check that both inputs are non-empty lists of numbers with the same length."""
    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise TypeError("y_true and y_pred must be lists or tuples.")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, y_pred has {len(y_pred)}."
        )
    for value in list(y_true) + list(y_pred):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"All values must be numbers, got {value!r}.")


def confusion_matrix(y_true, y_pred, positive=1):
    """Count true/false positives and negatives. Returns a dict."""
    _validate(y_true, y_pred)
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for actual, predicted in zip(y_true, y_pred):
        if predicted == positive:
            counts["tp" if actual == positive else "fp"] += 1
        else:
            counts["fn" if actual == positive else "tn"] += 1
    return counts


def accuracy(y_true, y_pred):
    """Share of predictions that are exactly right (0.0 to 1.0)."""
    _validate(y_true, y_pred)
    correct = sum(1 for a, p in zip(y_true, y_pred) if a == p)
    return correct / len(y_true)


def precision(y_true, y_pred, positive=1):
    """Of everything predicted positive, how much was actually positive.
    Returns 0.0 if nothing was predicted positive."""
    cm = confusion_matrix(y_true, y_pred, positive)
    predicted_positive = cm["tp"] + cm["fp"]
    return cm["tp"] / predicted_positive if predicted_positive else 0.0


def recall(y_true, y_pred, positive=1):
    """Of everything actually positive, how much the model found.
    Returns 0.0 if there are no actual positives."""
    cm = confusion_matrix(y_true, y_pred, positive)
    actual_positive = cm["tp"] + cm["fn"]
    return cm["tp"] / actual_positive if actual_positive else 0.0


def f1_score(y_true, y_pred, positive=1):
    """Balance of precision and recall (their harmonic mean)."""
    p = precision(y_true, y_pred, positive)
    r = recall(y_true, y_pred, positive)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def mean_absolute_error(y_true, y_pred):
    """Average size of the error, ignoring direction (for regression)."""
    _validate(y_true, y_pred)
    return sum(abs(a - p) for a, p in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true, y_pred):
    """Like MAE, but punishes large errors more (for regression)."""
    _validate(y_true, y_pred)
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(y_true, y_pred)) / len(y_true))
