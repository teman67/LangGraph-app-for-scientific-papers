from file_parsers import chunk_text, strip_introduction


def test_parse_input_schema_count(field_specs):
    assert len(field_specs) == 30
    assert field_specs[0].field_name == "Paper_DOI"
    assert field_specs[0].ontology_class == "cto:creep publication DOI"
    assert field_specs[-1].index == 30


def test_parse_output_headers(output_headers, data_columns):
    assert output_headers[0].strip().upper() == "ID"
    assert len(data_columns) == 30
    assert data_columns[0] == "Paper DOI"


def test_field_and_column_counts_align(field_specs, data_columns):
    # Known invariant of the sample fixtures: 30 input fields <-> 30 output columns.
    assert len(field_specs) == len(data_columns)


def test_strip_introduction_removes_section():
    text = (
        "Title\n\nIntroduction\nBackground info that must not reach the model.\n"
        "More background.\n\nMaterials and Methods\nSample HT1 tested at 650 C.\n"
    )
    cleaned, truncated = strip_introduction(text)
    assert truncated is True
    assert "Background info" not in cleaned
    assert "Sample HT1" in cleaned
    assert "Materials and Methods" in cleaned


def test_strip_introduction_noop_when_absent():
    text = "Title\n\nMaterials and Methods\nSample HT1 tested at 650 C.\n"
    cleaned, truncated = strip_introduction(text)
    assert truncated is False
    assert cleaned == text


def test_chunk_text_short_text_single_chunk():
    text = "short paper text"
    chunks = chunk_text(text, max_chars=1000)
    assert chunks == [text]


def test_chunk_text_long_text_splits_with_overlap():
    paragraph = "Sentence about creep testing results. " * 50 + "\n\n"
    text = paragraph * 20
    chunks = chunk_text(text, max_chars=2000, overlap_chars=200)
    assert len(chunks) > 1
    # every chunk should respect the rough size ceiling
    assert all(len(c) <= 2000 * 1.6 for c in chunks)
    # reassembled chunks should still contain all the distinctive content
    assert all("creep testing results" in c for c in chunks)
