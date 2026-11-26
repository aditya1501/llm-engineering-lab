import pytest

from halludetect.parsing import ParseError, extract_json


@pytest.mark.parametrize(
    "text",
    ['{"a": 1}', 'Sure! ```json\n{"a": 1}\n```', 'Here you go: {"a": 1} hope that helps'],
)
def test_extracts_json(text):
    assert extract_json(text) == {"a": 1}


def test_raises_on_no_json():
    with pytest.raises(ParseError):
        extract_json("no json here")
