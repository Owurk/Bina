import pytest

from main import (
    FALLBACK_SEVERITY,
    MAX_TITLE_WORDS,
    SEVERITY_LEVELS,
    _clamped_title,
    _extract_json_object,
    _parse_ai_analysis,
)





def test_extract_plain_json_object():
    assert _extract_json_object('{"title": "T"}') == {"title": "T"}


def test_extract_json_from_fenced_output():
    text = '```json\n{"title": "T", "severity": "high"}\n```'
    assert _extract_json_object(text) == {"title": "T", "severity": "high"}


def test_extract_json_from_prose_wrapper():
    text = 'Here is the analysis:\n{"title": "T"}\nHope this helps!'
    assert _extract_json_object(text) == {"title": "T"}


def test_extract_returns_none_without_braces():
    assert _extract_json_object("A person walks by.") is None


def test_extract_returns_none_for_truncated_json():
    assert _extract_json_object('{"title": "truncated') is None


def test_extract_returns_none_for_invalid_json_between_braces():

    assert _extract_json_object('{"title": "weird } case", "description": "d"') is None


def test_extract_returns_none_for_empty_input():
    assert _extract_json_object("") is None


def test_extract_returns_none_for_non_object_payload():

    assert _extract_json_object('{"a": 1} and {"b": 2}') is None





def test_clamped_title_keeps_short_titles():
    assert _clamped_title("Person at the door") == "Person at the door"


def test_clamped_title_caps_word_count():
    title = "one two three four five six seven eight nine ten eleven twelve"
    clamped = _clamped_title(title)
    assert len(clamped.split()) == MAX_TITLE_WORDS
    assert clamped == "one two three four five six seven eight nine ten"


def test_clamped_title_normalizes_whitespace():
    assert _clamped_title("  Person   at\nthe door  ") == "Person at the door"





def test_parse_valid_json_returns_all_three_fields():
    raw = '{"title": "Person at the door", "description": "Someone is ringing the bell.", "severity": "medium"}'
    assert _parse_ai_analysis(raw) == {
        "title": "Person at the door",
        "description": "Someone is ringing the bell.",
        "severity": "medium",
    }


@pytest.mark.parametrize("severity", SEVERITY_LEVELS)
def test_parse_accepts_every_defined_severity(severity):
    raw = f'{{"title": "T", "description": "D.", "severity": "{severity}"}}'
    assert _parse_ai_analysis(raw)["severity"] == severity


def test_parse_normalizes_severity_case():
    raw = '{"title": "T", "description": "D.", "severity": " High "}'
    assert _parse_ai_analysis(raw)["severity"] == "high"


def test_parse_unknown_severity_falls_back():
    raw = '{"title": "T", "description": "D.", "severity": "critical"}'
    parsed = _parse_ai_analysis(raw)
    assert parsed["severity"] == FALLBACK_SEVERITY == "medium"


def test_parse_missing_severity_falls_back():
    raw = '{"title": "T", "description": "D."}'
    assert _parse_ai_analysis(raw)["severity"] == FALLBACK_SEVERITY


def test_parse_non_string_severity_falls_back():
    raw = '{"title": "T", "description": "D.", "severity": 3}'
    assert _parse_ai_analysis(raw)["severity"] == FALLBACK_SEVERITY


def test_parse_missing_title_is_derived_from_description():
    raw = '{"description": "A dog is sleeping on the porch.", "severity": "low"}'
    parsed = _parse_ai_analysis(raw)
    assert parsed["title"] == "A dog is sleeping on the porch"
    assert parsed["severity"] == "low"


def test_parse_clamps_overlong_title():
    raw = (
        '{"title": "one two three four five six seven eight nine ten eleven", '
        '"description": "D.", "severity": "low"}'
    )
    parsed = _parse_ai_analysis(raw)
    assert len(parsed["title"].split()) == MAX_TITLE_WORDS


def test_parse_plain_text_falls_back_with_derived_title():
    raw = "A person is walking across the backyard carrying a ladder towards the shed."
    parsed = _parse_ai_analysis(raw)
    assert parsed["description"] == raw
    assert parsed["title"] == "A person is walking across the backyard carrying a ladder"
    assert parsed["severity"] == FALLBACK_SEVERITY


def test_parse_truncated_json_falls_back_to_raw_text():
    raw = '{"title": "Person at door", "description": "A person is standing at'
    parsed = _parse_ai_analysis(raw)
    assert parsed["description"] == raw
    assert parsed["severity"] == FALLBACK_SEVERITY


def test_parse_json_with_empty_description_uses_raw_text_but_keeps_severity():
    raw = '{"title": "T", "description": "", "severity": "high"}'
    parsed = _parse_ai_analysis(raw)
    assert parsed["description"] == raw
    assert parsed["severity"] == "high"


def test_parse_whitespace_only_returns_none():
    assert _parse_ai_analysis("   \n\t") is None


def test_parse_empty_string_returns_none():
    assert _parse_ai_analysis("") is None
