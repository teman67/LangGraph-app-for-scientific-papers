"""
Utilities to load the Creep Testing Ontology (or any OWL/TTL ontology) and
build a lightweight label -> definition lookup that is used to enrich the
LLM prompt with the authoritative ontology definitions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

import rdflib
from rdflib import RDFS
from rdflib.namespace import SKOS


@dataclass
class OntologyTerm:
    uri: str
    label: str
    definition: str


def load_ontology(path: str) -> Dict[str, OntologyTerm]:
    """Parse a .ttl ontology file and return a dict keyed by lower-cased label."""
    g = rdflib.Graph()
    g.parse(path, format="turtle")

    terms: Dict[str, OntologyTerm] = {}
    for s, _, label in g.triples((None, RDFS.label, None)):
        definition = ""
        for _, _, d in g.triples((s, SKOS.definition, None)):
            definition = str(d)
            break
        if not definition:
            # some ontologies use IAO_0000115 ("definition") annotation property
            iao_def = rdflib.URIRef("http://purl.obolibrary.org/obo/IAO_0000115")
            for _, _, d in g.triples((s, iao_def, None)):
                definition = str(d)
                break
        key = str(label).strip().lower()
        # prefer entries that actually have a definition if the label repeats
        if key not in terms or (not terms[key].definition and definition):
            terms[key] = OntologyTerm(uri=str(s), label=str(label), definition=definition)
    return terms


def find_term(terms: Dict[str, OntologyTerm], class_ref: str) -> Optional[OntologyTerm]:
    """Look up an ontology term given a reference like 'cto:creep material identifier'."""
    if not class_ref:
        return None
    label = re.sub(r"^[a-zA-Z0-9_-]+:", "", class_ref).strip().lower()
    return terms.get(label)
