"""
Creep-Testing Ontology (CTO) Extraction App
--------------------------------------------
Streamlit + LangGraph app that reads scientific papers (PDF or plain text),
an input Excel schema (fields -> ontology classes), an ontology (.ttl) file,
and an output Excel template, and uses an LLM (Claude or OpenAI, user-supplied
API key) to fill the output table with data mined from the papers, grounded
in the ontology's definitions.

Run with:  streamlit run app.py
"""
import io
from datetime import datetime

import pandas as pd
import streamlit as st

from evaluation import evaluate
from file_parsers import load_document, parse_input_schema, parse_output_headers
from graph import assemble_rows, extract_single_document, node_load_ontology
from mapping import map_fields_to_columns

st.set_page_config(page_title="CTO Data Extractor", page_icon="🧪", layout="wide")

st.title("🧪 Creep Testing Ontology — Data Extraction")
st.caption(
    "Extracts structured, ontology-grounded data from scientific papers into a "
    "spreadsheet, using an LLM of your choice. The Introduction section of each "
    "paper is excluded from extraction."
)

# ----------------------------------------------------------------------------
# Sidebar: LLM configuration
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("LLM configuration")
    provider_label = st.radio("Provider", ["Claude (Anthropic)", "OpenAI"], index=0)
    provider = "anthropic" if provider_label.startswith("Claude") else "openai"

    api_key = st.text_input(
        f"{provider_label} API key",
        type="password",
        help="Your key is only used for this session and is never stored.",
    )

    default_model = "claude-sonnet-4-6" if provider == "anthropic" else "gpt-4.1"
    model = st.text_input("Model", value=default_model)

    use_cache = st.checkbox(
        "Cache LLM calls on disk",
        value=True,
        help="Avoids re-billing/re-calling the API for identical (document, prompt) pairs "
        "across re-runs — useful when fixing/re-running just one paper.",
    )
    max_chunk_chars = st.slider(
        "Max characters per chunk sent to the model",
        min_value=5000,
        max_value=30000,
        value=15000,
        step=1000,
        help="Long papers are split into overlapping chunks so results/tables late in the "
        "paper aren't truncated away.",
    )

    st.divider()
    st.caption(
        "Files needed:\n"
        "1. Source paper(s) — PDF or .txt\n"
        "2. Input schema Excel (fields → ontology classes)\n"
        "3. Output template Excel (defines output columns)\n"
        "4. Ontology file (.ttl)"
    )

# ----------------------------------------------------------------------------
# Main: file uploads
# ----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    source_files = st.file_uploader(
        "Source paper(s) — PDF or text",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    input_schema_file = st.file_uploader(
        "Input schema Excel (fields ↔ ontology classes)", type=["xlsx", "xlsm", "xls"]
    )

with col2:
    output_template_file = st.file_uploader(
        "Output template Excel (defines output columns)", type=["xlsx", "xlsm", "xls"]
    )
    ontology_file = st.file_uploader("Ontology file (.ttl)", type=["ttl"])

ready_to_load = bool(source_files and input_schema_file and output_template_file and ontology_file)

# ----------------------------------------------------------------------------
# Stage 1: load schema/ontology/documents, show mapping + intro-stripped
# text for review/editing BEFORE spending any LLM calls.
# ----------------------------------------------------------------------------
if st.button("① Load & preview", disabled=not ready_to_load):
    with st.spinner("Parsing schema, ontology, and documents…"):
        specs = parse_input_schema(io.BytesIO(input_schema_file.getvalue()))
        headers = parse_output_headers(io.BytesIO(output_template_file.getvalue()))
        data_columns = headers[1:] if headers and headers[0].strip().upper() == "ID" else headers
        mappings = map_fields_to_columns(specs, data_columns)

        ontology_terms = node_load_ontology({"ontology_bytes": ontology_file.getvalue()})["ontology_terms"]

        parsed_docs = [load_document(f.name, f.getvalue()) for f in source_files]

    st.session_state["output_headers"] = headers
    st.session_state["data_columns"] = data_columns
    st.session_state["column_mappings"] = mappings
    st.session_state["ontology_terms"] = ontology_terms
    st.session_state["parsed_docs"] = {d.name: d.text for d in parsed_docs}
    st.session_state.pop("extracted_rows", None)  # invalidate any previous run

if "column_mappings" in st.session_state:
    st.subheader("Field → column mapping")
    st.caption(
        "Each input-schema field is matched to an output column by name/ontology similarity, "
        "falling back to position only when no confident textual match exists."
    )
    mapping_rows = [
        {
            "Output column": m.column_name,
            "Matched field": m.field_spec.field_name if m.field_spec else "— none —",
            "Ontology class": m.field_spec.ontology_class if m.field_spec else "",
            "Method": m.method,
            "Confidence": round(m.score, 2),
        }
        for m in st.session_state["column_mappings"]
    ]
    mapping_df = pd.DataFrame(mapping_rows)
    low_conf = mapping_df[(mapping_df["Method"] != "name") | (mapping_df["Confidence"] < 0.5)]
    if not low_conf.empty:
        st.warning(
            f"{len(low_conf)} column(s) matched with low confidence or fell back to position — "
            "double check these before running extraction."
        )
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

if "parsed_docs" in st.session_state:
    st.subheader("Review extracted text (Introduction already removed)")
    st.caption("Edit any document's text below if the Introduction-stripping heuristic over/under-cut a section.")
    for name, text in st.session_state["parsed_docs"].items():
        with st.expander(f"📄 {name} ({len(text):,} chars)"):
            edited = st.text_area("Text sent to the model", value=text, height=250, key=f"doc_text_{name}")
            st.session_state["parsed_docs"][name] = edited

# ----------------------------------------------------------------------------
# Stage 2: run extraction with a per-document progress bar
# ----------------------------------------------------------------------------
ready_to_run = "parsed_docs" in st.session_state and bool(api_key)
run = st.button("② Run extraction", type="primary", disabled=not ready_to_run)
if "parsed_docs" in st.session_state and not api_key:
    st.info("Enter an API key in the sidebar to enable extraction.")

if run:
    docs = list(st.session_state["parsed_docs"].items())
    progress = st.progress(0.0, text="Starting…")
    all_rows, all_warnings = [], []

    from file_parsers import ParsedDocument

    for i, (name, text) in enumerate(docs):
        progress.progress(i / len(docs), text=f"Extracting from {name} ({i + 1}/{len(docs)})…")
        doc = ParsedDocument(name=name, text=text)
        try:
            rows, warns = extract_single_document(
                doc=doc,
                column_mappings=st.session_state["column_mappings"],
                data_columns=st.session_state["data_columns"],
                ontology_terms=st.session_state["ontology_terms"],
                provider=provider,
                api_key=api_key,
                model=model,
                use_cache=use_cache,
                max_chars_per_chunk=max_chunk_chars,
            )
            all_rows.extend(rows)
            all_warnings.extend(warns)
        except Exception as e:  # noqa: BLE001
            all_warnings.append(f"{name}: {e}")

    progress.progress(1.0, text="Assembling table…")
    final_rows = assemble_rows(all_rows, st.session_state["data_columns"])
    progress.empty()

    st.session_state["extracted_rows"] = final_rows
    st.session_state["warnings"] = all_warnings

    for w in all_warnings:
        st.warning(w)
    if not final_rows:
        st.warning(
            "No rows were extracted. Check that the papers actually contain creep-test data "
            "outside the Introduction section, and that your API key/model are valid."
        )
    else:
        n_flagged = sum(1 for r in final_rows if r.get("_validation_warnings"))
        st.success(f"Extracted {len(final_rows)} row(s) — {n_flagged} flagged for review.")

# ----------------------------------------------------------------------------
# Editable results table + download
# ----------------------------------------------------------------------------
if st.session_state.get("extracted_rows"):
    st.subheader("Extracted data — review and edit before downloading")
    st.caption(
        "`_validation_warnings` flags values that don't match the expected shape for that field "
        "(e.g. a percentage over 100, a missing unit). `_evidence` / `_source_document` are "
        "provenance — the paper (and, where the model reported it, page/quote) a row came from."
    )

    headers = st.session_state["output_headers"]
    df = pd.DataFrame(st.session_state["extracted_rows"])
    provenance_cols = ["_source_document", "_evidence", "_validation_warnings"]
    ordered_cols = (
        [c for c in headers if c in df.columns]
        + [c for c in provenance_cols if c in df.columns]
        + [c for c in df.columns if c not in headers and c not in provenance_cols]
    )
    df = df[ordered_cols]

    def _highlight_warnings(row):
        flagged = bool(row.get("_validation_warnings"))
        return ["background-color: #fff3cd" if flagged else "" for _ in row]

    edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="editor")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        edited_df.to_excel(writer, index=False, sheet_name="extracted data")
    buffer.seek(0)

    st.download_button(
        "⬇ Download output Excel",
        data=buffer,
        file_name=f"cto_extraction_{datetime.now():%Y%m%d_%H%M}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ------------------------------------------------------------------------
    # Evaluation against a known gold-standard filled workbook (optional)
    # ------------------------------------------------------------------------
    with st.expander("📊 Evaluate against a gold-standard file"):
        st.caption(
            "Upload a filled workbook with the same columns as the output template "
            "(e.g. a manually-annotated gold standard for this paper) to score the "
            "current extraction against it."
        )
        gold_file = st.file_uploader("Gold-standard Excel", type=["xlsx", "xlsm", "xls"], key="gold_upload")
        if gold_file is not None:
            gold_wb_df = pd.read_excel(io.BytesIO(gold_file.getvalue()), sheet_name=0)
            gold_wb_df = gold_wb_df.dropna(how="all")
            result = evaluate(edited_df, gold_wb_df)

            m1, m2, m3 = st.columns(3)
            m1.metric("Overall field accuracy", f"{result.overall_accuracy:.0%}")
            m2.metric("Rows matched", f"{result.n_matched_rows} / {result.n_gold_rows}")
            m3.metric("Predicted rows", result.n_predicted_rows)

            per_col_df = pd.DataFrame(
                [{"Column": c, "Accuracy": a} for c, a in result.per_column_accuracy.items()]
            ).sort_values("Accuracy")
            st.bar_chart(per_col_df.set_index("Column"))

            st.markdown("**Row-by-row diff**")
            st.dataframe(result.diff_table, use_container_width=True)
