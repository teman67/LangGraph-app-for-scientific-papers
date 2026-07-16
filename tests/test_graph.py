from unittest.mock import patch

from file_parsers import ParsedDocument, strip_introduction
from graph import (
    assemble_rows,
    build_graph,
    build_json_schema,
    extract_single_document,
)
from mapping import map_fields_to_columns


def _empty_record(data_columns, **overrides):
    rec = {c: "" for c in data_columns}
    rec["evidence"] = ""
    rec.update(overrides)
    return rec


def test_build_graph_compiles():
    assert build_graph() is not None


def test_json_schema_shape(data_columns):
    schema = build_json_schema(data_columns)
    assert schema["type"] == "object"
    record_schema = schema["properties"]["records"]["items"]
    assert set(record_schema["properties"]) == set(data_columns) | {"evidence"}
    assert record_schema["additionalProperties"] is False


def test_extract_single_document_happy_path(field_specs, data_columns, ontology_terms):
    mappings = map_fields_to_columns(field_specs, data_columns)
    doc = ParsedDocument(name="paper.txt", text="Sample HT1 tested at 650 C, 625 MPa.")

    fake_response = {
        "records": [
            _empty_record(
                data_columns,
                **{"Sample ID": "HT1", "Temperature": "650 C", "Initial stress": "625 MPa"},
            )
        ]
    }

    with patch("graph.run_llm_structured", return_value=fake_response):
        rows, warnings = extract_single_document(
            doc=doc,
            column_mappings=mappings,
            data_columns=data_columns,
            ontology_terms=ontology_terms,
            provider="anthropic",
            api_key="fake-key",
            model="fake-model",
            use_cache=False,
        )

    assert len(rows) == 1
    assert rows[0]["Sample ID"] == "HT1"
    assert warnings == []


def test_extract_single_document_falls_back_on_structured_failure(field_specs, data_columns, ontology_terms):
    import json

    mappings = map_fields_to_columns(field_specs, data_columns)
    doc = ParsedDocument(name="paper.txt", text="Sample HT1 tested at 650 C.")

    fallback_json = json.dumps([_empty_record(data_columns, **{"Sample ID": "HT1"})])

    with patch("graph.run_llm_structured", side_effect=RuntimeError("schema violation")):
        with patch("graph.run_llm", return_value=fallback_json):
            rows, warnings = extract_single_document(
                doc=doc,
                column_mappings=mappings,
                data_columns=data_columns,
                ontology_terms=ontology_terms,
                provider="anthropic",
                api_key="fake-key",
                model="fake-model",
                use_cache=False,
            )

    assert len(rows) == 1
    assert rows[0]["Sample ID"] == "HT1"


def test_extract_single_document_dedupes_across_chunks(field_specs, data_columns, ontology_terms):
    mappings = map_fields_to_columns(field_specs, data_columns)
    # Force chunking with a tiny max_chars so the same short doc is split
    # into >1 chunk, and mock the model to return the *same* record for
    # every chunk (simulating overlap re-extraction).
    long_text = ("Sample HT1 tested at 650 C, 625 MPa. " * 200) + "\n\n" + ("more filler text. " * 200)
    doc = ParsedDocument(name="paper.txt", text=long_text)

    fake_response = {
        "records": [_empty_record(data_columns, **{"Sample ID": "HT1", "Material name": "Inconel 718"})]
    }

    with patch("graph.run_llm_structured", return_value=fake_response):
        rows, warnings = extract_single_document(
            doc=doc,
            column_mappings=mappings,
            data_columns=data_columns,
            ontology_terms=ontology_terms,
            provider="anthropic",
            api_key="fake-key",
            model="fake-model",
            use_cache=False,
            max_chars_per_chunk=500,
        )

    # Regardless of how many chunks were made, identical rows should collapse to one.
    assert len(rows) == 1


def test_assemble_rows_flags_bad_values(data_columns):
    row = {c: "" for c in data_columns}
    row["Paper DOI"] = "not-a-doi"
    row["Temperature"] = "650"  # missing unit
    row["_source_document"] = "paper.pdf"
    row["evidence"] = "quote"

    final = assemble_rows([row], data_columns)
    assert final[0]["ID"] == 1
    assert "Paper DOI" in final[0]["_validation_warnings"]
    assert "Temperature" in final[0]["_validation_warnings"]
    assert final[0]["_source_document"] == "paper.pdf"
    assert final[0]["_evidence"] == "quote"


def test_assemble_rows_clean_value_has_no_warnings(data_columns):
    row = {c: "" for c in data_columns}
    row["Paper DOI"] = "10.1016/j.example.2020"
    row["Temperature"] = "650 C"
    final = assemble_rows([row], data_columns)
    assert final[0]["_validation_warnings"] == ""
