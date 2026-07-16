from validation import FieldType, classify_field, validate_value


def test_classify_field_types():
    assert classify_field("Paper DOI") == FieldType.DOI
    assert classify_field("Percentage permanent elongation") == FieldType.PERCENTAGE
    assert classify_field("Delta precepitates fraction") == FieldType.PERCENTAGE
    assert classify_field("Temperature") == FieldType.TEMPERATURE
    assert classify_field("Initial stress") == FieldType.STRESS
    assert classify_field("Stress Exponent (n)") == FieldType.DIMENSIONLESS
    assert classify_field("Creep rupture time") == FieldType.TIME
    assert classify_field("Activation Energy (Qc)") == FieldType.ENERGY
    assert classify_field("Steady state creep rate") == FieldType.RATE
    assert classify_field("Grain size") == FieldType.LENGTH
    assert classify_field("Material name") == FieldType.TEXT


def test_empty_value_never_flagged():
    for ft in FieldType:
        assert validate_value("", ft).ok
        assert validate_value(None, ft).ok


def test_valid_values_pass():
    assert validate_value("10.1016/j.example.2020", FieldType.DOI).ok
    assert validate_value("650 C", FieldType.TEMPERATURE).ok
    assert validate_value("625 MPa", FieldType.STRESS).ok
    assert validate_value("159.74 h", FieldType.TIME).ok
    assert validate_value("276 kJ/mol", FieldType.ENERGY).ok
    assert validate_value("13.8%", FieldType.PERCENTAGE).ok
    assert validate_value("170 um", FieldType.LENGTH).ok
    assert validate_value("n = 5", FieldType.DIMENSIONLESS).ok


def test_invalid_values_flagged():
    assert not validate_value("not-a-doi", FieldType.DOI).ok
    assert not validate_value("650", FieldType.TEMPERATURE).ok  # missing unit
    assert not validate_value("138", FieldType.PERCENTAGE).ok  # out of range
    assert not validate_value("abc", FieldType.STRESS).ok  # non-numeric
