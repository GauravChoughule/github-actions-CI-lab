import csv
import math
from pathlib import Path

import pytest

from src import metrics

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "sample_predictions.csv"


# ---------- Fixture: load a small saved set of model predictions ----------
@pytest.fixture
def sample_predictions():
    with open(DATA_FILE) as f:
        rows = list(csv.DictReader(f))
    y_true = [int(r["y_true"]) for r in rows]
    y_pred = [int(r["y_pred"]) for r in rows]
    return y_true, y_pred


def test_confusion_matrix_on_sample_data(sample_predictions):
    # 10 rows: 4 TP, 4 TN, 1 FP, 1 FN
    assert metrics.confusion_matrix(*sample_predictions) == {"tp": 4, "fp": 1, "tn": 4, "fn": 1}


def test_metrics_on_sample_data(sample_predictions):
    assert metrics.accuracy(*sample_predictions) == pytest.approx(0.8)
    assert metrics.precision(*sample_predictions) == pytest.approx(0.8)
    assert metrics.recall(*sample_predictions) == pytest.approx(0.8)
    assert metrics.f1_score(*sample_predictions) == pytest.approx(0.8)


# ---------- Parametrized tests: many cases, one test function ----------
@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], 1.0),   # all correct
        ([1, 1, 1, 1], [0, 0, 0, 0], 0.0),   # all wrong
        ([1, 0, 1, 0], [1, 1, 1, 1], 0.5),   # half right
        ([1], [1], 1.0),                     # single item
    ],
)
def test_accuracy(y_true, y_pred, expected):
    assert metrics.accuracy(y_true, y_pred) == pytest.approx(expected)


@pytest.mark.parametrize(
    "y_true, y_pred, expected_precision, expected_recall",
    [
        ([1, 0, 1, 1], [1, 1, 1, 0], 2 / 3, 2 / 3),
        ([1, 1, 0, 0], [1, 0, 0, 0], 1.0, 0.5),
        ([0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0),   # no positives at all: no divide-by-zero
    ],
)
def test_precision_and_recall(y_true, y_pred, expected_precision, expected_recall):
    assert metrics.precision(y_true, y_pred) == pytest.approx(expected_precision)
    assert metrics.recall(y_true, y_pred) == pytest.approx(expected_recall)


def test_f1_combines_precision_and_recall():
    # precision 1.0, recall 0.5 -> F1 = 2 * 1 * 0.5 / 1.5
    assert metrics.f1_score([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(2 / 3)


def test_custom_positive_label():
    # Treat 0 as the positive class instead of 1
    assert metrics.precision([0, 1, 0], [0, 0, 0], positive=0) == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "y_true, y_pred, expected_mae, expected_rmse",
    [
        ([3.0, 5.0, 2.0], [3.0, 5.0, 2.0], 0.0, 0.0),
        ([1.0, 2.0, 3.0], [2.0, 2.0, 5.0], 1.0, math.sqrt(5 / 3)),
        ([10, 20], [0, 20], 5.0, math.sqrt(50)),
    ],
)
def test_regression_metrics(y_true, y_pred, expected_mae, expected_rmse):
    assert metrics.mean_absolute_error(y_true, y_pred) == pytest.approx(expected_mae)
    assert metrics.root_mean_squared_error(y_true, y_pred) == pytest.approx(expected_rmse)


def test_rmse_is_never_smaller_than_mae():
    y_true, y_pred = [1, 2, 3, 4], [1, 5, 3, 0]
    rmse = metrics.root_mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    assert rmse >= mae


# ---------- Bad input should raise clear errors ----------
@pytest.mark.parametrize(
    "y_true, y_pred, error",
    [
        ([], [], ValueError),                 # empty
        ([1, 0], [1], ValueError),            # different lengths
        ([1, "a"], [1, 0], TypeError),        # not a number
        ([1, True], [1, 0], TypeError),       # booleans are rejected
        ("10", "10", TypeError),              # not a list
    ],
)
def test_invalid_inputs_raise(y_true, y_pred, error):
    with pytest.raises(error):
        metrics.accuracy(y_true, y_pred)
