import json

import pytest

from main import (
    FALLBACK_IS_UNUSUAL,
    FALLBACK_SEVERITY,
    MAX_TITLE_WORDS,
    MAX_UNUSUAL_REASON_WORDS,
    SEVERITY_LEVELS,
    _clamped_title,
    _clamped_words,
    _extract_json_object,
    _parse_ai_analysis,
)


def _raw_json(**overrides) -> str:
    """A well-formed five-key model reply with any fields overridden."""
    payload = {
        "title": "Person at the door",
        "description": "Someone is ringing the bell.",
        "severity": "medium",
    }
    payload.update(overrides)
    return json.dumps(payload)





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


def test_clamped_words_caps_word_count():
    text = "one two three four five six seven eight nine ten eleven twelve"
    clamped = _clamped_words(text, 5)
    assert clamped == "one two three four five"


def test_clamped_words_normalizes_whitespace():
    assert _clamped_words("  a   b\nc  ", 3) == "a b c"





def test_parse_valid_json_returns_all_five_fields():
    raw = '{"title": "Person at the door", "description": "Someone is ringing the bell.", "severity": "medium", "is_unusual": false, "reason": ""}'
    assert _parse_ai_analysis(raw) == {
        "title": "Person at the door",
        "description": "Someone is ringing the bell.",
        "severity": "medium",
        "is_unusual": False,
        "reason": "",
    }


def test_parse_valid_json_returns_unusual_flag_with_reason():
    raw = (
        '{"title": "Person forcing the back window", '
        '"description": "A person is prying open the back window.", '
        '"severity": "high", "is_unusual": true, '
        '"reason": "Prying open a window is threatening."}'
    )
    parsed = _parse_ai_analysis(raw)
    assert parsed["is_unusual"] is True
    assert isinstance(parsed["is_unusual"], bool)
    assert parsed["reason"] == "Prying open a window is threatening."


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



def test_parse_coerces_truthy_is_unusual():
    raw = _raw_json(is_unusual=True, reason="Someone is forcing the door.")
    parsed = _parse_ai_analysis(raw)
    assert parsed["is_unusual"] is True
    assert parsed["reason"] == "Someone is forcing the door."


@pytest.mark.parametrize("flag", ["true", "True", "YES", "1", 1])
def test_parse_coerces_drifted_truthy_spellings(flag):
    """Models drift into string/number spellings of booleans."""
    raw = _raw_json(is_unusual=flag, reason="Climbing over the fence.")
    parsed = _parse_ai_analysis(raw)
    assert parsed["is_unusual"] is True
    assert parsed["reason"] == "Climbing over the fence."


@pytest.mark.parametrize("flag", [False, "false", "False", "no", "0", 0, None])
def test_parse_coerces_falsy_spellings_and_drops_reason(flag):
    raw = _raw_json(is_unusual=flag, reason="Ordinary passers-by.")
    parsed = _parse_ai_analysis(raw)
    assert parsed["is_unusual"] is False
    assert parsed["reason"] == ""


@pytest.mark.parametrize("flag", ["maybe", 2, 1.5, ["true"], {"flag": True}])
def test_parse_invalid_is_unusual_defaults_to_false(flag):
    """An unrecognized answer degrades to "ordinary", never to a false alarm."""
    raw = _raw_json(is_unusual=flag, reason="Volunteered but dropped.")
    parsed = _parse_ai_analysis(raw)
    assert parsed["is_unusual"] is FALLBACK_IS_UNUSUAL == False
    assert parsed["reason"] == ""


def test_parse_missing_is_unusual_defaults_to_false():
    raw = '{"title": "T", "description": "D.", "severity": "low"}'
    parsed = _parse_ai_analysis(raw)
    assert parsed["is_unusual"] == FALLBACK_IS_UNUSUAL == False
    assert parsed["reason"] == ""


def test_parse_reason_is_clamped_when_unusual():
    reason = " ".join(f"word{i}" for i in range(30))
    raw = _raw_json(is_unusual=True, reason=reason)
    parsed = _parse_ai_analysis(raw)
    assert len(parsed["reason"].split()) == MAX_UNUSUAL_REASON_WORDS


def test_parse_non_string_reason_is_dropped():
    raw = _raw_json(is_unusual=True, reason=42)
    parsed = _parse_ai_analysis(raw)
    assert parsed["is_unusual"] is True
    assert parsed["reason"] == ""


def test_parse_plain_text_fallback_has_no_unusual_signal():
    raw = "A person is walking across the backyard carrying a ladder towards the shed."
    parsed = _parse_ai_analysis(raw)
    assert parsed["is_unusual"] is False
    assert parsed["reason"] == ""


def test_parse_truncated_json_fallback_has_no_unusual_signal():
    raw = '{"title": "Person at door", "is_unusual": true, "reason": "Forcing the door'
    parsed = _parse_ai_analysis(raw)
    assert parsed["is_unusual"] is False
    assert parsed["reason"] == ""
