import math

import pytest

from experiment.extraction import extract_number
from experiment.metrics import compute_metrics, constant_baseline, summarize_by


@pytest.mark.parametrize(
    "text,expected,method",
    [
        ("7", 7.0, "first_number"),
        ("About 4.73 crystals.", 4.73, "first_number"),
        ("The answer is 19.41", 19.41, "after_answer"),
        ("Roughly 1,250 crystals", 1250.0, "first_number"),
        ("forty-two", 42.0, "number_words"),
        ("one hundred and seven", 107.0, "number_words"),
        ("a dozen or so", 12.0, "number_words"),
        ("between 7 and 9", 8.0, "range_midpoint"),
        ("7-9 rabbits", 8.0, "range_midpoint"),
        ("≈ 83.17", 83.17, "after_≈"),
        ("I cannot know that.", None, "invalid"),
        ("", None, "invalid"),
        ("-5", -5.0, "first_number"),
        ("The average is approximately 137.52 metres", 137.52, "after_is approximately"),
        ("Answer: 6", 6.0, "after_answer"),
    ],
)
def test_extract_number(text, expected, method):
    v, m = extract_number(text)
    assert v == expected
    assert m == method


def test_metrics_basic():
    preds = [10.0, 20.0, None, 40.0]
    truths = [10.0, 22.0, 30.0, 50.0]
    m = compute_metrics(preds, truths)
    assert m["n"] == 4 and m["n_valid"] == 3
    assert m["invalid_rate"] == pytest.approx(0.25)
    assert m["mae"] == pytest.approx((0 + 2 + 10) / 3)
    assert m["rmse"] == pytest.approx(math.sqrt((0 + 4 + 100) / 3))
    assert m["median_ae"] == 2.0
    assert m["within_1pct"] == pytest.approx(1 / 3)
    assert m["within_10pct"] == pytest.approx(2 / 3)
    assert m["pearson_r"] > 0.95
    assert m["bias"] < 0


def test_metrics_all_invalid_and_constant():
    m = compute_metrics([None, None], [1.0, 2.0])
    assert m["invalid_rate"] == 1.0 and math.isnan(m["mae"])
    c = constant_baseline([10.0, 30.0], 20.0)
    assert c["mae"] == 10.0 and math.isnan(c["pearson_r"])


def test_summarize_by():
    recs = [
        {"family": "actual", "predicted_value": 1.0, "true_value": 1.0},
        {"family": "actual", "predicted_value": 3.0, "true_value": 1.0},
        {"family": "fake", "predicted_value": None, "true_value": 5.0},
    ]
    s = summarize_by(recs, "family")
    assert s["actual"]["mae"] == 1.0 and s["fake"]["invalid_rate"] == 1.0
