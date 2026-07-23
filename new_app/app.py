"""
Creep-Testing Ontology (CTO) Extraction App — fixed-backend variant
--------------------------------------------------------------------
Streamlit + LangGraph app that reads scientific papers (PDF or plain text)
and uses an LLM (Claude or OpenAI, user-supplied API key) to fill an output
table with data mined from the papers, grounded in the CTO ontology.

Unlike the main app, the input schema, output template, and ontology are
NOT user-uploadable here — they're fixed backend configuration loaded from
the inputs/ folder shipped alongside this app. The user only ever provides
the source paper(s) to extract from.

Run with:  streamlit run app.py
"""
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from evaluation import evaluate
from file_parsers import load_document, parse_input_schema, parse_output_headers
from graph import assemble_rows, extract_single_document, node_load_ontology
from llm_clients import InvalidAPIKeyError
from mapping import map_fields_to_columns

INPUTS_DIR = Path(__file__).parent / "inputs"
BACKEND_FILES = {
    "ontology": INPUTS_DIR / "cto.ttl",
    "input_schema": INPUTS_DIR / "LLM input- terms from CTO.xlsm",
    "output_template": INPUTS_DIR / "LLM output- Gold standard.xlsm",
}

st.set_page_config(page_title="CTO Data Extractor", page_icon="🧪", layout="wide")

missing = [name for name, path in BACKEND_FILES.items() if not path.exists()]
if missing:
    st.error(
        "Missing backend input file(s): "
        + ", ".join(f"`{BACKEND_FILES[m]}`" for m in missing)
        + ". This app expects the ontology, input schema, and output template "
        "to be present in its inputs/ folder."
    )
    st.stop()


@st.cache_data(show_spinner=False)
def _load_backend_config(ontology_path: str, schema_path: str, template_path: str):
    """Parses the fixed schema/ontology/template once per file (cached on
    the file paths, which are constant for the app's lifetime)."""
    specs = parse_input_schema(io.BytesIO(Path(schema_path).read_bytes()))
    headers = parse_output_headers(io.BytesIO(Path(template_path).read_bytes()))
    data_columns = headers[1:] if headers and headers[0].strip().upper() == "ID" else headers
    mappings = map_fields_to_columns(specs, data_columns)
    ontology_terms = node_load_ontology({"ontology_bytes": Path(ontology_path).read_bytes()})["ontology_terms"]
    return headers, data_columns, mappings, ontology_terms

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

    ANTHROPIC_MODELS = [
        # --- Latest (Jun 2026) ---
        "claude-fable-5",          # Most capable – long-running agents
        "claude-opus-4-8",         # Complex agentic coding & enterprise
        "claude-sonnet-5",         # Best speed / intelligence balance
        "claude-haiku-4-5",        # Fastest, near-frontier intelligence
        # --- Previous generation (still supported) ---
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
    ]
    OPENAI_MODELS = [
        # --- Latest GPT-5.6 family ---
        "gpt-5.6-sol",             # Most powerful – complex reasoning & coding
        "gpt-5.6-terra",           # Balanced intelligence and cost
        "gpt-5.6-luna",            # Cost-optimised, high-volume
        # --- Previous generation (still supported) ---
        "gpt-5.4-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
    ]
    model_options = ANTHROPIC_MODELS if provider == "anthropic" else OPENAI_MODELS
    default_model = "claude-sonnet-5" if provider == "anthropic" else "gpt-5.6-terra"
    model = st.selectbox(
        "Model",
        options=model_options,
        index=model_options.index(default_model),
        help=(
            "Powerful: claude-fable-5 / gpt-5.6-sol — highest accuracy.\n"
            "Balanced: claude-sonnet-5 / gpt-5.6-terra — recommended default.\n"
            "Light: claude-haiku-4-5 / gpt-5.6-luna — fastest and cheapest."
        ),
    )

    # Reasoning effort — only relevant for GPT-5.6 and o-series models.
    _supports_effort = provider == "openai" and model.startswith(("gpt-5.", "o1", "o3", "o4"))
    if _supports_effort:
        reasoning_effort = st.select_slider(
            "Reasoning effort",
            options=["low", "medium", "high", "max"],
            value="high",
            help=(
                "How hard the model thinks before answering.\n"
                "low — fastest, cheapest, more variable outputs.\n"
                "medium — balanced.\n"
                "high — recommended for data extraction (default).\n"
                "max — most thorough, slowest, most expensive."
            ),
        )
    else:
        reasoning_effort = "high"  # ignored for non-supporting models

    use_cache = st.checkbox(
        "Cache LLM calls",
        value=True,
        help="Avoids re-billing/re-calling the API for identical (document, prompt) pairs "
        "across re-runs — useful when fixing/re-running just one paper. Backed by Redis "
        "when REDIS_URL is set (e.g. on Heroku), otherwise falls back to local disk.",
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
        "Only the source paper(s) are uploaded here — the input schema, output "
        "template, and ontology are fixed backend configuration:\n"
        f"- Ontology: `{BACKEND_FILES['ontology'].name}`\n"
        f"- Input schema: `{BACKEND_FILES['input_schema'].name}`\n"
        f"- Output template: `{BACKEND_FILES['output_template'].name}`"
    )

    st.divider()
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)]"
        "(https://github.com/teman67/LangGraph-app-for-scientific-papers)"
    )
    st.markdown(
        "**Developer:** Amirhossein Bayani  \n"
        "[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)]"
        "(https://www.linkedin.com/in/amirhosseinbayani/)"
    )

# ----------------------------------------------------------------------------
# Main: file upload (source papers only)
# ----------------------------------------------------------------------------
source_files = st.file_uploader(
    "Source paper(s) — PDF or text",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

ready_to_load = bool(source_files)

# ----------------------------------------------------------------------------
# Stage 1: load schema/ontology/documents, show mapping + intro-stripped
# text for review/editing BEFORE spending any LLM calls.
# ----------------------------------------------------------------------------
if st.button("① Load & preview", disabled=not ready_to_load):
    with st.spinner("Parsing schema, ontology, and documents…"):
        headers, data_columns, mappings, ontology_terms = _load_backend_config(
            str(BACKEND_FILES["ontology"]),
            str(BACKEND_FILES["input_schema"]),
            str(BACKEND_FILES["output_template"]),
        )
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

    invalid_key_error = None
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
                reasoning_effort=reasoning_effort,
            )
            all_rows.extend(rows)
            all_warnings.extend(warns)
        except InvalidAPIKeyError as e:
            invalid_key_error = str(e)
            break  # same key would fail on every remaining document too
        except Exception as e:  # noqa: BLE001
            all_warnings.append(f"{name}: {e}")

    progress.empty()

    if invalid_key_error:
        st.error(f"🔑 {invalid_key_error}")
    else:
        final_rows = assemble_rows(all_rows, st.session_state["data_columns"])
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

# ----------------------------------------------------------------------------
# Model reference guide
# ----------------------------------------------------------------------------
st.divider()
with st.expander("📖 LLM model reference guide", expanded=False):
    st.markdown(
        "Use this table to pick the right model for your workload. "
        "Prices are per **million tokens** (input / output) as listed in official provider docs."
    )

    st.markdown("#### 🟣 Anthropic — Claude models")
    anthropic_data = {
        "Model ID": [
            "claude-fable-5",
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-haiku-4-5",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-sonnet-4-5",
        ],
        "Tier": [
            "🏆 Most capable",
            "💪 Powerful",
            "⚖️ Balanced",
            "⚡ Fast / Light",
            "📦 Legacy",
            "📦 Legacy",
            "📦 Legacy",
            "📦 Legacy",
        ],
        "Best for": [
            "Long-running agents, highest accuracy",
            "Complex agentic coding & enterprise tasks",
            "Best speed / intelligence trade-off (recommended default)",
            "High-volume, latency-sensitive workloads",
            "Previous-generation agentic tasks",
            "Previous-generation tasks with extended thinking",
            "Previous-generation balanced tasks",
            "Previous-generation fast tasks",
        ],
        "Price (input / output per MTok)": [
            "$10 / $50",
            "$5 / $25",
            "$3 / $15 (intro: $2 / $10 until Aug 2026)",
            "$1 / $5",
            "$5 / $25",
            "$5 / $25",
            "$3 / $15",
            "$3 / $15",
        ],
        "Context window": [
            "1 M tokens",
            "1 M tokens",
            "1 M tokens",
            "200 K tokens",
            "1 M tokens",
            "1 M tokens",
            "1 M tokens",
            "200 K tokens",
        ],
    }
    st.dataframe(pd.DataFrame(anthropic_data), use_container_width=True, hide_index=True)

    st.markdown("#### 🟢 OpenAI — GPT models")
    openai_data = {
        "Model ID": [
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.4-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "Tier": [
            "🏆 Most capable",
            "⚖️ Balanced",
            "⚡ Fast / Light",
            "⚡ Fast / Light",
            "📦 Legacy",
            "📦 Legacy",
            "📦 Legacy",
            "📦 Legacy",
        ],
        "Best for": [
            "Complex reasoning, coding, highest accuracy",
            "Balanced intelligence and cost (recommended default)",
            "Cost-sensitive, high-volume workloads",
            "Lightweight tasks at low cost",
            "Previous-generation general tasks",
            "Previous-generation lightweight tasks",
            "Previous-generation multimodal tasks",
            "Previous-generation cost-optimised tasks",
        ],
        "Price (input / output per MTok)": [
            "$5 / $30",
            "$2.50 / $15",
            "$1 / $6",
            "—",
            "—",
            "—",
            "—",
            "—",
        ],
        "Context window": [
            "1.05 M tokens",
            "1.05 M tokens",
            "1.05 M tokens",
            "—",
            "—",
            "—",
            "—",
            "—",
        ],
    }
    st.dataframe(pd.DataFrame(openai_data), use_container_width=True, hide_index=True)

    st.caption(
        "Prices and specs sourced from official provider documentation (Anthropic: platform.claude.com/docs · "
        "OpenAI: developers.openai.com/api/docs/models). Legacy model pricing marked '—' varies — check provider "
        "docs for current rates. Knowledge cutoff for all GPT-5.6 models: Feb 2026; Claude Fable 5 / Opus 4.8 / "
        "Sonnet 5: Jan 2026."
    )
