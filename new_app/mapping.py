"""
Maps each input-schema field (# | field_name | ontology class | definition) onto
a column of the output template, by NAME rather than by position.

Positional mapping (field #1 -> output column 2, etc.) is fragile: if someone
reorders either sheet, or the two sheets have a different number of fields, the
whole extraction silently misaligns. This module does a greedy best-match
assignment based on token overlap + string similarity between the schema's
`field_name` and the template's header text, and only falls back to position
when no confident textual match exists.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List, Optional

from file_parsers import FieldSpec

CONFIDENCE_THRESHOLD = 0.30


@dataclass
class ColumnMapping:
    column_name: str
    field_spec: Optional[FieldSpec]
    score: float
    method: str  # "name" | "position" | "unmatched"


def _normalize(text: str) -> str:
    text = re.sub(r"_x$", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return text.strip()


def _similarity(a: str, b: str) -> float:
    na, nb = _normalize(a), _normalize(b)
    wa, wb = set(na.split()), set(nb.split())
    jaccard = len(wa & wb) / len(wa | wb) if (wa | wb) else 0.0
    seq_ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    return 0.5 * jaccard + 0.5 * seq_ratio


def map_fields_to_columns(
    field_specs: List[FieldSpec], data_columns: List[str]
) -> List[ColumnMapping]:
    """Greedy best-match assignment of field_specs onto data_columns (1:1 where possible).

    Returns one ColumnMapping per data_column, in data_column order, so the
    result can be used directly to build the LLM prompt / output row schema.
    """
    n_specs, n_cols = len(field_specs), len(data_columns)

    # Score every (field, column) pair.
    scored = []
    for i, spec in enumerate(field_specs):
        for j, col in enumerate(data_columns):
            scored.append((_similarity(spec.field_name, col), i, j))
    scored.sort(key=lambda t: t[0], reverse=True)

    assigned_field = {}
    assigned_col = {}
    for sc, i, j in scored:
        if i in assigned_field or j in assigned_col:
            continue
        if sc < CONFIDENCE_THRESHOLD:
            continue
        assigned_field[i] = (j, sc)
        assigned_col[j] = (i, sc)

    mappings: List[ColumnMapping] = []
    for j, col in enumerate(data_columns):
        if j in assigned_col:
            i, sc = assigned_col[j]
            mappings.append(ColumnMapping(col, field_specs[i], sc, "name"))
        elif j < n_specs:
            # Fallback: no confident textual match found for this column —
            # fall back to positional alignment (only sensible if the two
            # sheets have the same length).
            mappings.append(ColumnMapping(col, field_specs[j], 0.0, "position"))
        else:
            mappings.append(ColumnMapping(col, None, 0.0, "unmatched"))
    return mappings
