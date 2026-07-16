from ontology import find_term


def test_ontology_loads_terms(ontology_terms):
    assert len(ontology_terms) > 1000


def test_find_term_known_class(ontology_terms):
    term = find_term(ontology_terms, "cto:creep material identifier")
    assert term is not None
    assert "material" in term.definition.lower()


def test_find_term_unknown_class_returns_none(ontology_terms):
    assert find_term(ontology_terms, "cto:this class does not exist") is None


def test_find_term_empty_ref_returns_none(ontology_terms):
    assert find_term(ontology_terms, "") is None
