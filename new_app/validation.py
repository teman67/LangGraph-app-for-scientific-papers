"""
Lightweight, ontology-informed validation of extracted values.

We don't have a machine-readable "range"/"unit" annotation in the CTO for each
data property, so this infers an expected value *shape* from the column name
(and, as a fallback, from the ontology definition text) and flags values that
don't look right — e.g. a percentage over 100, a temperature with no unit, a
non-numeric stress. This is advisory: it never drops or "corrects" a value,
it only attaches warnings for a human reviewer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class FieldType(str, Enum):
    DOI = "doi"
    PERCENTAGE = "percentage"
    TEMPERATURE = "temperature"
    STRESS = "stress"
    TIME = "time"
    ENERGY = "energy"
    RATE = "rate"
    LENGTH = "length"
    DIMENSIONLESS = "dimensionless"
    TEXT = "text"


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*[eE]\s*-?\d+)?")
_RANGE_RE = re.compile(r"-?\d+(?:\.\d+)?\s*(?:-|to|–)\s*-?\d+(?:\.\d+)?")

_UNIT_PATTERNS = {
    FieldType.TEMPERATURE: re.compile(r"\b(°?\s*[CKF])\b", re.IGNORECASE),
    FieldType.STRESS: re.compile(r"\b(MPa|GPa|ksi|psi)\b", re.IGNORECASE),
    FieldType.TIME: re.compile(r"\b(h|hr|hrs|hours?|s|sec|secs?|seconds?|min|mins?|days?)\b", re.IGNORECASE),
    FieldType.ENERGY: re.compile(r"\b(kJ/mol|J/mol|eV|kcal/mol)\b", re.IGNORECASE),
    FieldType.LENGTH: re.compile(r"\b(nm|um|µm|mm|cm|m)\b"),
    FieldType.RATE: re.compile(r"(s\s*\^?\s*-?1|/\s*s|per\s*second)", re.IGNORECASE),
}

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def classify_field(column_name: str, definition: str = "") -> FieldType:
    name = column_name.lower()
    text = f"{name} {definition.lower()}"

    if "doi" in name:
        return FieldType.DOI
    if "percentage" in name or "elongation" in name or "extension" in name or "reduction of area" in name:
        return FieldType.PERCENTAGE
    if "fraction" in name:
        return FieldType.PERCENTAGE
    if "temperature" in name:
        return FieldType.TEMPERATURE
    if "stress exponent" in name:
        return FieldType.DIMENSIONLESS
    if "stress" in name:
        return FieldType.STRESS
    if "rate" in name:
        return FieldType.RATE
    if "time" in name:
        return FieldType.TIME
    if "energy" in name:
        return FieldType.ENERGY
    if "size" in name or "grain size" in name:
        return FieldType.LENGTH
    return FieldType.TEXT


@dataclass
class ValidationResult:
    warnings: List[str]

    @property
    def ok(self) -> bool:
        return not self.warnings


def validate_value(value: object, field_type: FieldType) -> ValidationResult:
    warnings: List[str] = []
    if value is None:
        return ValidationResult(warnings)
    text = str(value).strip()
    if text == "":
        return ValidationResult(warnings)  # empty is fine — the model was told not to guess

    if field_type == FieldType.DOI:
        if not _DOI_RE.match(text):
            warnings.append(f"Does not look like a DOI (expected '10.xxxx/...'): {text!r}")

    elif field_type == FieldType.PERCENTAGE:
        nums = [float(n.replace(" ", "")) for n in _NUMERIC_RE.findall(text)]
        if not nums:
            warnings.append(f"Expected a numeric percentage, got: {text!r}")
        elif any(n < 0 or n > 100 for n in nums) and not any(0 <= n <= 1 for n in nums):
            warnings.append(f"Percentage value out of 0-100 range: {text!r}")

    elif field_type in (FieldType.TEMPERATURE, FieldType.STRESS, FieldType.TIME, FieldType.ENERGY):
        if not _NUMERIC_RE.search(text):
            warnings.append(f"Expected a numeric value, got: {text!r}")
        pattern = _UNIT_PATTERNS.get(field_type)
        if pattern and not pattern.search(text):
            warnings.append(f"Missing expected unit for {field_type.value}: {text!r}")

    elif field_type == FieldType.RATE:
        if not _NUMERIC_RE.search(text):
            warnings.append(f"Expected a numeric creep-rate value, got: {text!r}")

    elif field_type == FieldType.LENGTH:
        if not _NUMERIC_RE.search(text):
            warnings.append(f"Expected a numeric length value, got: {text!r}")
        elif not _UNIT_PATTERNS[FieldType.LENGTH].search(text):
            warnings.append(f"Missing expected length unit (nm/um/mm/...): {text!r}")

    elif field_type == FieldType.DIMENSIONLESS:
        if not _NUMERIC_RE.search(text):
            warnings.append(f"Expected a dimensionless numeric value, got: {text!r}")

    # FieldType.TEXT: no validation — free text/identifiers are open-ended.
    return ValidationResult(warnings)


def validate_row(row: dict, field_types: dict) -> dict:
    """Return {column_name: [warnings]} for every column with a non-empty warning list."""
    out = {}
    for col, ftype in field_types.items():
        result = validate_value(row.get(col, ""), ftype)
        if not result.ok:
            out[col] = result.warnings
    return out
