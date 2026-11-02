import math
import unittest

from src import metrics


class TestClassificationMetrics(unittest.TestCase):

    def test_accuracy(self):
        cases = [
            ([1, 0, 1, 0], [1, 0, 1, 0], 1.0),
            ([1, 1, 1, 1], [0, 0, 0, 0], 0.0),
            ([1, 0, 1, 0], [1, 1, 1, 1], 0.5),
        ]
        for y_true, y_pred, expected in cases:
            # subTest reports each case separately if one fails
            with self.subTest(y_true=y_true, y_pred=y_pred):
                self.assertAlmostEqual(metrics.accuracy(y_true, y_pred), expected)

    def test_confusion_matrix(self):
        result = metrics.confusion_matrix([1, 0, 1, 0], [1, 1, 0, 0])
        self.assertEqual(result, {"tp": 1, "fp": 1, "tn": 1, "fn": 1})

    def test_precision_recall_f1(self):
        y_true, y_pred = [1, 1, 0, 0], [1, 0, 0, 0]
        self.assertAlmostEqual(metrics.precision(y_true, y_pred), 1.0)
        self.assertAlmostEqual(metrics.recall(y_true, y_pred), 0.5)
        self.assertAlmostEqual(metrics.f1_score(y_true, y_pred), 2 / 3)

    def test_no_positives_returns_zero(self):
        self.assertEqual(metrics.precision([0, 0], [0, 0]), 0.0)
        self.assertEqual(metrics.recall([0, 0], [0, 0]), 0.0)
        self.assertEqual(metrics.f1_score([0, 0], [0, 0]), 0.0)


class TestRegressionMetrics(unittest.TestCase):

    def test_mae(self):
        self.assertAlmostEqual(metrics.mean_absolute_error([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]), 1.0)

    def test_rmse(self):
        self.assertAlmostEqual(
            metrics.root_mean_squared_error([1.0, 2.0, 3.0], [2.0, 2.0, 5.0]), math.sqrt(5 / 3)
        )

    def test_perfect_predictions_give_zero_error(self):
        self.assertEqual(metrics.mean_absolute_error([4, 5], [4, 5]), 0.0)
        self.assertEqual(metrics.root_mean_squared_error([4, 5], [4, 5]), 0.0)


class TestInputValidation(unittest.TestCase):

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            metrics.accuracy([], [])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            metrics.mean_absolute_error([1, 2], [1])

    def test_non_numeric_values(self):
        with self.assertRaises(TypeError):
            metrics.recall([1, "x"], [1, 0])

    def test_not_a_list(self):
        with self.assertRaises(TypeError):
            metrics.accuracy("101", "101")


if __name__ == "__main__":
    unittest.main()
