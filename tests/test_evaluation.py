import pandas as pd

from evaluation import evaluate


def test_evaluate_perfect_match():
    gold = pd.DataFrame([{"Paper DOI": "10.1/x", "Sample ID": "HT1", "Temperature": "650 C"}])
    predicted = gold.copy()
    result = evaluate(predicted, gold)
    assert result.overall_accuracy == 1.0
    assert result.n_matched_rows == 1


def test_evaluate_handles_row_reordering():
    gold = pd.DataFrame(
        [
            {"Sample ID": "HT1", "Temperature": "650 C"},
            {"Sample ID": "HT2", "Temperature": "700 C"},
        ]
    )
    predicted = pd.DataFrame(
        [
            {"Sample ID": "HT2", "Temperature": "700 C"},
            {"Sample ID": "HT1", "Temperature": "650 C"},
        ]
    )
    result = evaluate(predicted, gold)
    assert result.overall_accuracy == 1.0
    assert result.n_matched_rows == 2


def test_evaluate_detects_mismatch():
    gold = pd.DataFrame([{"Sample ID": "HT1", "Initial stress": "625 MPa"}])
    predicted = pd.DataFrame([{"Sample ID": "HT1", "Initial stress": "550 MPa"}])
    result = evaluate(predicted, gold)
    assert result.per_column_accuracy["Initial stress"] == 0.0
    assert result.per_column_accuracy["Sample ID"] == 1.0


def test_evaluate_numeric_tolerant_matching():
    gold = pd.DataFrame([{"Sample ID": "HT1", "Temperature": "650 C"}])
    predicted = pd.DataFrame([{"Sample ID": "HT1", "Temperature": "650C"}])
    result = evaluate(predicted, gold)
    assert result.per_column_accuracy["Temperature"] == 1.0


def test_evaluate_flags_missing_and_extra_rows():
    gold = pd.DataFrame([{"Sample ID": "HT1"}, {"Sample ID": "HT2"}])
    predicted = pd.DataFrame([{"Sample ID": "HT1"}])
    result = evaluate(predicted, gold)
    assert result.n_matched_rows == 1
    assert result.n_gold_rows == 2
    assert result.n_predicted_rows == 1
