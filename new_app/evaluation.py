"""
Evaluation utilities: compare an extracted (predicted) table against a
gold-standard filled workbook of the same shape, producing per-column and
overall accuracy metrics plus a row-aligned diff for manual review.

Row alignment between predicted and gold tables is done greedily by textual
similarity across a set of "key" columns (defaults to columns containing
'doi', 'sample', 'material' — whatever is present), since the model's row
order won't necessarily match the gold file's row order.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

_KEY_HINTS = ("doi", "sample", "material")


def _normalize(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"[^a-z0-9.]+", " ", text.lower())
    return " ".join(text.split())


def _values_match(a: object, b: object) -> bool:
    na, nb = _normalize(a), _normalize(b)
    if na == "" and nb == "":
        return True
    if na == "" or nb == "":
        return False
    if na == nb:
        return True
    num_a = re.findall(r"-?\d+\.?\d*", na)
    num_b = re.findall(r"-?\d+\.?\d*", nb)
    if num_a and num_b and num_a == num_b:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.9


@dataclass
class EvalResult:
    per_column_accuracy: Dict[str, float]
    overall_accuracy: float
    n_matched_rows: int
    n_predicted_rows: int
    n_gold_rows: int
    diff_table: pd.DataFrame


def _key_columns(columns: List[str]) -> List[str]:
    keys = [c for c in columns if any(h in c.lower() for h in _KEY_HINTS)]
    return keys or columns[:1]


def _row_similarity(row_a: pd.Series, row_b: pd.Series, key_cols: List[str]) -> float:
    scores = [1.0 if _values_match(row_a.get(c), row_b.get(c)) else 0.0 for c in key_cols]
    return sum(scores) / len(scores) if scores else 0.0


def align_rows(predicted: pd.DataFrame, gold: pd.DataFrame) -> List[tuple]:
    """Greedy best-match row alignment based on key-column similarity.
    Returns a list of (predicted_index_or_None, gold_index_or_None) pairs.
    """
    common_cols = [c for c in gold.columns if c in predicted.columns]
    key_cols = _key_columns(common_cols)

    pairs = []
    for pi in predicted.index:
        for gi in gold.index:
            sc = _row_similarity(predicted.loc[pi], gold.loc[gi], key_cols)
            pairs.append((sc, pi, gi))
    pairs.sort(key=lambda t: t[0], reverse=True)

    used_p, used_g = set(), set()
    alignment = []
    for sc, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        if sc <= 0:
            continue
        used_p.add(pi)
        used_g.add(gi)
        alignment.append((pi, gi))

    for pi in predicted.index:
        if pi not in used_p:
            alignment.append((pi, None))
    for gi in gold.index:
        if gi not in used_g:
            alignment.append((None, gi))
    return alignment


def evaluate(predicted: pd.DataFrame, gold: pd.DataFrame) -> EvalResult:
    common_cols = [c for c in gold.columns if c in predicted.columns and c.strip().upper() != "ID"]
    alignment = align_rows(predicted, gold)

    per_col_correct = {c: 0 for c in common_cols}
    per_col_total = {c: 0 for c in common_cols}
    diff_rows = []
    matched_rows = 0

    for pi, gi in alignment:
        if pi is not None and gi is not None:
            matched_rows += 1
            p_row = predicted.loc[pi]
            g_row = gold.loc[gi]
            diff_row = {"_predicted_row": pi, "_gold_row": gi}
            for c in common_cols:
                gold_val = g_row.get(c, "")
                pred_val = p_row.get(c, "")
                is_match = _values_match(pred_val, gold_val)
                per_col_total[c] += 1
                if is_match:
                    per_col_correct[c] += 1
                diff_row[f"{c} (gold)"] = gold_val
                diff_row[f"{c} (predicted)"] = pred_val
                diff_row[f"{c} (match)"] = is_match
            diff_rows.append(diff_row)
        elif pi is not None:
            diff_rows.append({"_predicted_row": pi, "_gold_row": None, "note": "extra predicted row (no gold match)"})
        else:
            diff_rows.append({"_predicted_row": None, "_gold_row": gi, "note": "missing — no predicted row matched this gold row"})

    per_col_accuracy = {
        c: (per_col_correct[c] / per_col_total[c] if per_col_total[c] else float("nan")) for c in common_cols
    }
    total_correct = sum(per_col_correct.values())
    total_fields = sum(per_col_total.values())
    overall_accuracy = total_correct / total_fields if total_fields else float("nan")

    return EvalResult(
        per_column_accuracy=per_col_accuracy,
        overall_accuracy=overall_accuracy,
        n_matched_rows=matched_rows,
        n_predicted_rows=len(predicted),
        n_gold_rows=len(gold),
        diff_table=pd.DataFrame(diff_rows),
    )
