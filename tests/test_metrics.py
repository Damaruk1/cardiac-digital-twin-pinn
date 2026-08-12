"""
test_metrics.py
-----------------
Phase 9 tests: verify precision/recall/F1 computation against a small,
HAND-CALCULATED example -- so we know the expected numbers are correct
independent of the code being tested.

Hand-calculated example:
    True labels:      [0, 0, 0, 1, 1, 1]
    Predicted labels: [0, 0, 1, 1, 1, 0]

    Class 0 (3 true examples): predicted correctly for 2, missed 1 (predicted as 1)
        TP=2, FN=1, FP=1 (one class-1 example was wrongly predicted as 0)
        Precision = 2/(2+1) = 0.667
        Recall    = 2/(2+1) = 0.667

    Class 1 (3 true examples): predicted correctly for 2, missed 1 (predicted as 0)
        TP=2, FN=1, FP=1
        Precision = 2/(2+1) = 0.667
        Recall    = 2/(2+1) = 0.667

Run with:
    pytest tests/test_metrics.py -v
"""

import numpy as np

from src.evaluation.metrics import evaluate_predictions


def test_precision_recall_match_hand_calculation():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 0]

    report = evaluate_predictions(y_true, y_pred, class_names=["A", "B"])

    assert abs(report.precision[0] - 0.667) < 0.01
    assert abs(report.recall[0] - 0.667) < 0.01
    assert abs(report.precision[1] - 0.667) < 0.01
    assert abs(report.recall[1] - 0.667) < 0.01


def test_confusion_matrix_correct_shape_and_values():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 0]

    report = evaluate_predictions(y_true, y_pred, class_names=["A", "B"])

    expected = np.array([[2, 1], [1, 2]])  # rows=true, cols=predicted
    assert np.array_equal(report.confusion, expected)


def test_perfect_predictions_give_f1_of_one():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]

    report = evaluate_predictions(y_true, y_pred, class_names=["N", "S", "V"])

    assert report.macro_f1 == 1.0
    assert report.accuracy == 1.0
    assert all(f == 1.0 for f in report.f1)


def test_always_predicting_majority_class_exposes_low_macro_f1():
    """The core Phase 9 lesson, verified numerically: a model that
    always predicts the majority class can have high accuracy but
    terrible macro F1."""
    # 90 examples of class 0, 10 of class 1 -- classic imbalance
    y_true = [0] * 90 + [1] * 10
    y_pred = [0] * 100  # always predicts class 0

    report = evaluate_predictions(y_true, y_pred, class_names=["N", "V"])

    assert report.accuracy == 0.90  # looks great...
    assert report.recall[1] == 0.0   # ...but catches ZERO of the minority class
    assert report.macro_f1 < 0.5     # macro F1 correctly exposes this failure


def test_format_report_produces_readable_string():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    report = evaluate_predictions(y_true, y_pred, class_names=["N", "V"])

    from src.evaluation.metrics import format_report
    text = format_report(report)

    assert "N" in text
    assert "V" in text
    assert "Macro F1" in text
