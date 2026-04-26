"""Minimal unit conversion skill with regex parsing for common conversions."""

import asyncio
import json
import re


def _to_float(raw: str):
    try:
        return float(raw)
    except Exception:
        return None


async def unit_converter(query: str) -> str:
    await asyncio.sleep(0)
    text = str(query or "").strip().lower()
    if not text:
        return json.dumps({"status": "error", "message": "Missing conversion query"})

    # km to miles
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:km|kilometer|kilometers)\s*(?:to)?\s*(?:miles?|mi)\b", text)
    if m:
        value = _to_float(m.group(1))
        converted = value * 0.621371
        return json.dumps({"status": "ok", "from": f"{value} km", "to": f"{converted:.6g} miles"})

    # pounds to kg
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:pounds?|lbs?)\s*(?:to|in)?\s*(?:kg|kilograms?)\b", text)
    if m:
        value = _to_float(m.group(1))
        converted = value * 0.45359237
        return json.dumps({"status": "ok", "from": f"{value} pounds", "to": f"{converted:.6g} kg"})

    # kg to pounds
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\s*(?:to|in)?\s*(?:pounds?|lbs?)\b", text)
    if m:
        value = _to_float(m.group(1))
        converted = value * 2.2046226218
        return json.dumps({"status": "ok", "from": f"{value} kg", "to": f"{converted:.6g} pounds"})

    # fahrenheit to celsius
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:fahrenheit|f)\s*(?:to)?\s*(?:celsius|c)\b", text)
    if m:
        value = _to_float(m.group(1))
        converted = (value - 32.0) * 5.0 / 9.0
        return json.dumps({"status": "ok", "from": f"{value} F", "to": f"{converted:.6g} C"})

    # celsius to fahrenheit
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:celsius|c)\s*(?:to)?\s*(?:fahrenheit|f)\b", text)
    if m:
        value = _to_float(m.group(1))
        converted = (value * 9.0 / 5.0) + 32.0
        return json.dumps({"status": "ok", "from": f"{value} C", "to": f"{converted:.6g} F"})

    # data sizes GB to bytes
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(?:gb|gib|gigabytes?)\s*(?:to|in)?\s*(?:bytes?|b)\b", text)
    if m:
        value = _to_float(m.group(1))
        converted = int(round(value * (1024 ** 3)))
        return json.dumps({"status": "ok", "from": f"{value} GB", "to": f"{converted} bytes"})

    return json.dumps(
        {
            "status": "error",
            "message": "Unsupported conversion format",
            "query": query,
        }
    )
