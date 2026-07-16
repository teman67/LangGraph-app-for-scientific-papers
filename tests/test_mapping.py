from mapping import map_fields_to_columns


def test_mapping_reproduces_known_correct_alignment(field_specs, data_columns):
    mappings = map_fields_to_columns(field_specs, data_columns)
    assert len(mappings) == len(data_columns)

    # For these fixtures, name-based matching should recover the exact same
    # alignment as the (known-correct) positional order.
    for i, m in enumerate(mappings):
        assert m.field_spec is not None
        assert m.field_spec.field_name == field_specs[i].field_name
        assert m.column_name == data_columns[i]


def test_mapping_survives_reordering():
    from file_parsers import FieldSpec

    specs = [
        FieldSpec(1, "material_name", "cto:x", "def"),
        FieldSpec(2, "Paper_DOI", "cto:y", "def"),
    ]
    # Output columns given in a different order than the specs.
    columns = ["Paper DOI", "Material name"]
    mappings = map_fields_to_columns(specs, columns)

    by_column = {m.column_name: m for m in mappings}
    assert by_column["Paper DOI"].field_spec.field_name == "Paper_DOI"
    assert by_column["Material name"].field_spec.field_name == "material_name"
    assert by_column["Paper DOI"].method == "name"
    assert by_column["Material name"].method == "name"


def test_mapping_falls_back_to_position_when_no_textual_match():
    from file_parsers import FieldSpec

    specs = [FieldSpec(1, "zzz_totally_unrelated_xyz", "cto:x", "def")]
    columns = ["Completely Different Header"]
    mappings = map_fields_to_columns(specs, columns)
    assert mappings[0].method == "position"
    assert mappings[0].field_spec.field_name == "zzz_totally_unrelated_xyz"
