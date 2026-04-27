import json

import pytest

from src.skills.unit_converter import _to_float, unit_converter


@pytest.mark.parametrize(
    ("query", "expected_from", "expected_to"),
    [
        ("10 km to miles", "10.0 km", "6.21371 miles"),
        ("5 pounds to kg", "5.0 pounds", "2.26796 kg"),
        ("2 kg to pounds", "2.0 kg", "4.40925 pounds"),
        ("212 f to c", "212.0 F", "100 C"),
        ("0 c to f", "0.0 C", "32 F"),
        ("1 gb to bytes", "1.0 GB", "1073741824 bytes"),
    ],
)
@pytest.mark.asyncio
async def test_unit_converter_supported_patterns(
    query: str,
    expected_from: str,
    expected_to: str,
) -> None:
    payload = json.loads(await unit_converter(query))

    assert payload["status"] == "ok"
    assert payload["from"] == expected_from
    assert payload["to"] == expected_to


@pytest.mark.asyncio
async def test_unit_converter_returns_error_for_empty_query() -> None:
    payload = json.loads(await unit_converter("   "))

    assert payload == {"status": "error", "message": "Missing conversion query"}


@pytest.mark.asyncio
async def test_unit_converter_returns_error_for_unsupported_format() -> None:
    payload = json.loads(await unit_converter("convert apples to oranges"))

    assert payload["status"] == "error"
    assert payload["message"] == "Unsupported conversion format"
    assert payload["query"] == "convert apples to oranges"


def test_to_float_returns_none_for_invalid_input() -> None:
    assert _to_float("not-a-number") is None

