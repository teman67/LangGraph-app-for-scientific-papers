"""
LangGraph pipeline:

    load_schema -> load_ontology -> load_documents -> extract (per document/chunk, via LLM) -> assemble

Each node reads/writes a shared `PipelineState` TypedDict. The graph is kept
linear since the steps are naturally sequential; `extract_single_document` is
also exposed standalone so the Streamlit app can call it per-document in a
loop and update a progress bar (LangGraph's own per-node granularity stops at
the whole `extract` node, which — with several documents — is too coarse for
good UX).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

from file_parsers import (
    FieldSpec,
    ParsedDocument,
    chunk_text,
    load_document,
    parse_input_schema,
    parse_output_headers,
)
from mapping import ColumnMapping, map_fields_to_columns
from ontology import OntologyTerm, find_term, load_ontology
from llm_clients import InvalidAPIKeyError, run_llm, run_llm_structured
from validation import classify_field, validate_row


class PipelineState(TypedDict, total=False):
    # raw inputs
    input_schema_bytes: bytes
    output_template_bytes: bytes
    ontology_bytes: bytes
    documents_raw: List[Dict[str, Any]]  # [{"name": str, "bytes": bytes}]

    # llm config
    provider: str
    api_key: str
    model: str

    # derived
    field_specs: List[FieldSpec]
    ontology_terms: Dict[str, OntologyTerm]
    output_headers: List[str]
    data_columns: List[str]
    column_mappings: List[ColumnMapping]
    parsed_documents: List[ParsedDocument]

    # results
    extracted_rows: List[Dict[str, Any]]
    warnings: List[str]


# ----------------------------------------------------------------------------
# Node: load schema + build name-based column mapping
# ----------------------------------------------------------------------------
def node_load_schema(state: PipelineState) -> PipelineState:
    import io

    specs = parse_input_schema(io.BytesIO(state["input_schema_bytes"]))
    headers = parse_output_headers(io.BytesIO(state["output_template_bytes"]))
    data_columns = headers[1:] if headers and headers[0].strip().upper() == "ID" else headers
    mappings = map_fields_to_columns(specs, data_columns)
    return {
        "field_specs": specs,
        "output_headers": headers,
        "data_columns": data_columns,
        "column_mappings": mappings,
    }


def node_load_ontology(state: PipelineState) -> PipelineState:
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".ttl", delete=False) as tmp:
        tmp.write(state["ontology_bytes"])
        tmp_path = tmp.name
    terms = load_ontology(tmp_path)
    return {"ontology_terms": terms}


def node_load_documents(state: PipelineState) -> PipelineState:
    parsed = []
    for doc in state["documents_raw"]:
        parsed.append(load_document(doc["name"], doc["bytes"]))
    return {"parsed_documents": parsed}


# ----------------------------------------------------------------------------
# Prompting / schema construction
# ----------------------------------------------------------------------------
def build_system_prompt(column_mappings: List[ColumnMapping], ontology_terms: Dict[str, OntologyTerm]) -> str:
    field_blocks = []
    for m in column_mappings:
        if m.field_spec is None:
            continue
        term = find_term(ontology_terms, m.field_spec.ontology_class)
        definition = m.field_spec.ontology_definition or (term.definition if term else "")
        field_blocks.append(
            f'- "{m.column_name}" (ontology class: {m.field_spec.ontology_class})\n'
            f"  Definition: {definition}"
        )
    fields_text = "\n".join(field_blocks)

    return f"""You are a scientific-literature data extraction assistant specialized in materials
science creep-testing publications. You extract structured data governed by the
Creep Testing Ontology (CTO).

TASK
Given the text of one scientific paper (or a chunk of a longer paper; the
Introduction section has already been removed), extract every distinct creep
test condition/result reported in the text and emit one record per distinct
test condition/row. If nothing relevant is present in this chunk, emit an
empty list of records — do not fabricate a row.

Field definitions (ontology-grounded — use these to decide what counts as a match):
{fields_text}

RULES
- **PRIORITY: always prefer values from numbered tables** (Table 1, Table 2, etc.) over
  values mentioned in prose, the abstract, or the conclusions. Prose often rounds
  or approximates; tables contain the authoritative numbers.
- Where a table provides a value with an uncertainty (e.g. "159.74 ± 19.18 h"),
  capture both the central value AND the uncertainty in the same field
  (e.g. "159.74 ± 19.18 h") — do not drop the ± term.
- When the same property (e.g. grain size) is reported at multiple processing stages,
  extract the value that corresponds to the final heat-treated state of that sample,
  and note the stage in parentheses if ambiguous (e.g. "170 µm (after full aging)").
- If a value is not reported in the text, use an empty string "" for that field —
  never invent or guess numeric values.
- Preserve original units in the extracted string (e.g. "650 C", "159.74 h").
- material_name / sample IDs should be copied verbatim as they appear in the text.
- The text may contain `[[page N]]` markers — if you can tell which page a
  record's data came from, put a short verbatim supporting quote plus the page
  number in the "evidence" field (e.g. "p.4: '625 MPa, 650 C, ...'"), otherwise
  leave "evidence" as an empty string.
- Emit one record per distinct test condition/heat-treatment scheme. Do not merge
  multiple conditions into one row and do not emit duplicate rows for the same
  condition.
"""


def build_json_schema(data_columns: List[str]) -> Dict[str, Any]:
    props = {col: {"type": "string"} for col in data_columns}
    props["evidence"] = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": props,
                    "required": list(props.keys()),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["records"],
        "additionalProperties": False,
    }


def _parse_json_array_fallback(raw: str) -> List[Dict[str, Any]]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    data = json.loads(cleaned)
    if isinstance(data, dict):
        data = data.get("records", [data])
    return data


# ----------------------------------------------------------------------------
# Per-document extraction (chunked, structured, with repair retry)
# ----------------------------------------------------------------------------
def extract_single_document(
    doc: ParsedDocument,
    column_mappings: List[ColumnMapping],
    data_columns: List[str],
    ontology_terms: Dict[str, OntologyTerm],
    provider: str,
    api_key: str,
    model: str,
    max_chars_per_chunk: int = 15000,
    reasoning_effort: str = "high",
) -> tuple[List[Dict[str, Any]], List[str]]:
    system_prompt = build_system_prompt(column_mappings, ontology_terms)
    json_schema = build_json_schema(data_columns)

    chunks = chunk_text(doc.text, max_chars=max_chars_per_chunk)
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for chunk_idx, chunk in enumerate(chunks):
        table_hint = "(this chunk contains one or more data tables — extract values from the tables first)" if "Table" in chunk or "±" in chunk else ""
        user_prompt = (
            f"Source document: {doc.name} (chunk {chunk_idx + 1}/{len(chunks)}) {table_hint}\n\n"
            f"--- BEGIN TEXT ---\n{chunk}\n--- END TEXT ---"
        )

        records: Optional[List[Dict[str, Any]]] = None
        last_error = None

        # Attempt 1: schema-enforced structured call.
        try:
            result = run_llm_structured(
                provider=provider,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
                reasoning_effort=reasoning_effort,
            )
            records = result.get("records", [])
        except InvalidAPIKeyError:
            raise  # same bad key would fail the repair retry too — don't bother
        except Exception as e:  # noqa: BLE001
            last_error = e

        # Attempt 2 (repair): free-text call asking the model to fix/re-emit JSON.
        if records is None:
            try:
                repair_prompt = (
                    user_prompt
                    + "\n\nReturn ONLY a JSON array of record objects (no markdown, no commentary), "
                    + f"each with exactly these keys: {json.dumps(data_columns + ['evidence'])}."
                )
                raw = run_llm(
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=repair_prompt,
                    reasoning_effort=reasoning_effort,
                )
                records = _parse_json_array_fallback(raw)
            except InvalidAPIKeyError:
                raise
            except Exception as e2:  # noqa: BLE001
                warnings.append(
                    f"{doc.name} (chunk {chunk_idx + 1}/{len(chunks)}): extraction failed "
                    f"after retry ({last_error} / {e2})"
                )
                records = []

        for r in records:
            r["_source_document"] = doc.name
            r["_chunk_index"] = chunk_idx
            rows.append(r)

    deduped, dedupe_warnings = _dedupe_rows(rows, data_columns)
    warnings.extend(dedupe_warnings)
    return deduped, warnings


def _row_key(row: Dict[str, Any], data_columns: List[str]) -> tuple:
    key_cols = [c for c in data_columns if any(h in c.lower() for h in ("sample", "material", "doi"))]
    key_cols = key_cols or data_columns[:2]

    def norm(v):
        return re.sub(r"[^a-z0-9]+", " ", str(v or "").lower()).strip()

    return tuple(norm(row.get(c, "")) for c in key_cols)


def _dedupe_rows(rows: List[Dict[str, Any]], data_columns: List[str]) -> tuple[List[Dict[str, Any]], List[str]]:
    """When a document is split into overlapping chunks, the same test
    condition can be (re-)extracted from more than one chunk. Collapse rows
    that share the same sample/material/DOI identity, keeping the version
    with the most non-empty fields.
    """
    seen: Dict[tuple, Dict[str, Any]] = {}
    warnings: List[str] = []
    for row in rows:
        key = _row_key(row, data_columns)
        if key == tuple("" for _ in key):
            # no identifying info at all — keep as its own row rather than
            # risk merging unrelated records together.
            seen[(id(row),)] = row
            continue
        if key not in seen:
            seen[key] = row
        else:
            existing = seen[key]
            existing_filled = sum(1 for c in data_columns if str(existing.get(c, "")).strip())
            new_filled = sum(1 for c in data_columns if str(row.get(c, "")).strip())
            if new_filled > existing_filled:
                seen[key] = row
            warnings.append(f"Merged duplicate row for key {key} across chunks.")
    return list(seen.values()), warnings


# ----------------------------------------------------------------------------
# Node: extract (loops over all documents; used when running the graph as a
# whole, e.g. from a script/test rather than the Streamlit app's progress loop)
# ----------------------------------------------------------------------------
def node_extract(state: PipelineState) -> PipelineState:
    all_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for doc in state["parsed_documents"]:
        rows, doc_warnings = extract_single_document(
            doc=doc,
            column_mappings=state["column_mappings"],
            data_columns=state["data_columns"],
            ontology_terms=state["ontology_terms"],
            provider=state["provider"],
            api_key=state["api_key"],
            model=state.get("model", ""),
        )
        all_rows.extend(rows)
        warnings.extend(doc_warnings)

    return {"extracted_rows": all_rows, "warnings": warnings}


def node_assemble(state: PipelineState) -> PipelineState:
    return {"extracted_rows": assemble_rows(state["extracted_rows"], state["data_columns"])}


def assemble_rows(raw_rows: List[Dict[str, Any]], data_columns: List[str]) -> List[Dict[str, Any]]:
    field_types = {col: classify_field(col) for col in data_columns}
    final_rows = []
    for i, row in enumerate(raw_rows, start=1):
        new_row: Dict[str, Any] = {"ID": i}
        for col in data_columns:
            new_row[col] = row.get(col, "")
        warnings_map = validate_row(new_row, field_types)
        new_row["_validation_warnings"] = (
            "; ".join(f"{c}: {'; '.join(ws)}" for c, ws in warnings_map.items()) if warnings_map else ""
        )
        new_row["_source_document"] = row.get("_source_document", "")
        new_row["_evidence"] = row.get("evidence", "")
        final_rows.append(new_row)
    return final_rows


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("load_schema", node_load_schema)
    graph.add_node("load_ontology", node_load_ontology)
    graph.add_node("load_documents", node_load_documents)
    graph.add_node("extract", node_extract)
    graph.add_node("assemble", node_assemble)

    graph.add_edge(START, "load_schema")
    graph.add_edge("load_schema", "load_ontology")
    graph.add_edge("load_ontology", "load_documents")
    graph.add_edge("load_documents", "extract")
    graph.add_edge("extract", "assemble")
    graph.add_edge("assemble", END)

    return graph.compile()
