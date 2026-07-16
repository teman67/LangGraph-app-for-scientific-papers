import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def input_schema_path():
    return os.path.join(FIXTURES, "LLM_input-_terms_from_CTO.xlsm")


@pytest.fixture
def output_template_path():
    return os.path.join(FIXTURES, "LLM_output-_Gold_standard.xlsm")


@pytest.fixture
def ontology_path():
    return os.path.join(FIXTURES, "cto.ttl")


@pytest.fixture
def field_specs(input_schema_path):
    from file_parsers import parse_input_schema

    return parse_input_schema(input_schema_path)


@pytest.fixture
def output_headers(output_template_path):
    from file_parsers import parse_output_headers

    return parse_output_headers(output_template_path)


@pytest.fixture
def data_columns(output_headers):
    headers = output_headers
    return headers[1:] if headers and headers[0].strip().upper() == "ID" else headers


@pytest.fixture
def ontology_terms(ontology_path):
    from ontology import load_ontology

    return load_ontology(ontology_path)
