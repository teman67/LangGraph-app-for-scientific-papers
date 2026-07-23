# CTO Data Extractor — fixed-backend variant

A Streamlit + LangGraph app that reads scientific papers (PDF/text) and uses an LLM
(Claude or OpenAI — your own API key) to fill an output table with structured,
ontology-grounded data mined from the papers. Built around the **Creep Testing
Ontology (CTO)**.

This is a variant of the main app in the parent directory. The difference: the input
schema, output template, and ontology are **not** user-uploadable here — they're fixed
backend configuration shipped in this folder's `inputs/` directory. The only thing a
user ever uploads is the source paper(s) to extract from.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

There's no `tests/` directory in this variant — see the parent app's `tests/` for the
pytest suite covering the shared pipeline code.

## Backend configuration (fixed, not uploaded)

These three files live in `inputs/` and are loaded automatically at startup (parsed
once and cached in-process, not re-parsed per run). If any is missing, the app shows
an error on load instead of failing partway through extraction:

1. **`cto.ttl`** — the ontology, used to pull `rdfs:label` / `skos:definition` (or
   `IAO_0000115`) pairs so the LLM prompt is grounded in the ontology's own wording.
2. **`LLM input- terms from CTO.xlsm`** — the input schema: columns
   `# | field name | related ontology class | related ontology class (definition)`.
3. **`LLM output- Gold standard.xlsm`** — the output template: its **first row**
   defines the exact header row of the table to fill (column 1 is assumed to be `ID`).

To point the app at different files, replace these three in `inputs/` (same filenames
expected — see `BACKEND_FILES` in `app.py`) rather than uploading through the UI.

## What the user provides

**Source paper(s)** only — one or more PDF or `.txt` files. The "Introduction" section
of each paper is automatically detected and stripped out before extraction (a
heuristic based on section headings). You get a chance to review/edit the stripped
text in the app before any LLM call is made.

## Pipeline (LangGraph)

```
load_schema -> load_ontology -> load_documents -> extract (per doc, chunked) -> assemble
```

- **load_schema**: parses the backend input schema workbook and output template header
  row, then runs **name-based mapping** (`mapping.py`) between the two — matching each
  schema field to the output column with the closest name/ontology similarity, and
  only falling back to positional alignment when no confident textual match exists.
  The app shows this mapping (with a confidence score) before you run anything.
- **load_ontology**: parses the `.ttl` file with `rdflib`, builds a label → definition map.
- **load_documents**: extracts text from each PDF/text file (with `[[page N]]` markers
  for provenance) and strips the Introduction section.
- **extract**: for each document:
  - splits long text into overlapping chunks (fixed at 15,000 characters — not
    user-configurable in this variant) so results/tables late in a paper aren't
    truncated away;
  - calls the selected LLM with a **schema-enforced structured output** — Anthropic
    via forced tool-use, OpenAI via `response_format={"type": "json_schema", ...}` —
    instead of asking for "JSON only" in prose and regexing it out;
  - if the structured call fails, **retries once** with a free-text "repair" prompt
    and a looser JSON parser as a fallback (skipped entirely if the failure was an
    invalid API key, since a retry would fail identically);
  - **deduplicates** rows extracted redundantly from overlapping chunks (matched on
    sample ID / material / DOI), keeping the most complete version of each;
  - asks the model for a short **evidence** quote/page per row for provenance.
- **assemble**: aligns rows to the output columns, assigns IDs, and runs
  **validation** (`validation.py`) — infers an expected shape per field (percentage,
  temperature+unit, stress+unit, DOI format, etc.) from the column name and flags
  values that don't match, without ever dropping or "correcting" them itself.

LLM calls are **not cached** in this variant — every extraction call hits the
provider API fresh, even if you re-run the same paper/prompt. (The main app in the
parent directory caches calls via Redis/disk; this one intentionally doesn't.)

## Using the app

1. Pick a provider (Claude/Anthropic or OpenAI) in the sidebar and paste your API key.
   Keys are only held in the Streamlit session and are never written to disk.
2. Upload source paper(s) (PDF/`.txt`), then click **① Load & preview** — review the
   field→column mapping (against the fixed backend schema/template) and the
   intro-stripped text for each paper (edit either if needed).
3. Click **② Run extraction** — a per-document progress bar shows what's running. An
   invalid API key is detected up front and reported once, rather than retried
   uselessly per chunk/document.
4. Review the table: rows/cells with validation warnings are flagged in the
   `_validation_warnings` column; `_source_document` / `_evidence` give provenance.
   Edit any cell directly in the browser.
5. Click **Download output Excel** to save your edits.
6. Optionally, expand **📊 Evaluate against a gold-standard file** and upload a
   manually-annotated version of the same paper(s) to get per-column accuracy, an
   overall score, and a row-by-row diff (row order doesn't need to match — rows are
   aligned by DOI/sample/material similarity).

## Project layout

```
app.py            Streamlit UI (staged: load & preview -> run -> review -> evaluate)
graph.py          LangGraph pipeline nodes, chunking, dedupe, retry, assembly
mapping.py        Name-based field <-> output-column matching
ontology.py       .ttl parsing (rdflib) and term lookup
file_parsers.py   Excel schema/template parsing, PDF/text reading, intro-stripping, chunking
validation.py     Per-field type inference + value validation
evaluation.py     Row-aligned scoring against a gold-standard workbook
llm_clients.py    Anthropic/OpenAI wrappers (free-text + schema-enforced), no caching
inputs/           Fixed backend files: cto.ttl, input schema, output template
```

## Notes & limitations

- Introduction-stripping is heuristic (regex on section headings). Use the preview
  step to check nothing important got cut before spending an LLM call on it.
- The LLM is instructed to leave a field as an empty string rather than guess a value —
  but always check flagged rows (`_validation_warnings`) and spot-check others against
  the source paper, especially numeric values.
- Name-based mapping falls back to position only when no confident textual match is
  found; check the mapping table shown after "Load & preview" if the backend
  schema/template files change significantly.
- The evaluation feature scores against whatever gold file you upload — it does not
  ship with a pre-filled gold standard for the sample papers.
- Since calls aren't cached, re-running extraction on the same paper re-bills the
  provider API every time.
