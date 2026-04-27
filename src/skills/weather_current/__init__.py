"""Live weather skill used by fast-path routing and tool execution."""

from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import Any, Dict, Optional, Tuple


_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _fetch_json_sync(url: str) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "AIDEN-weather/1.0"})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = response.read().decode("utf-8")
    parsed = json.loads(payload)
    return parsed if isinstance(parsed, dict) else {}


async def _fetch_json(url: str) -> Dict[str, Any]:
    return await asyncio.to_thread(_fetch_json_sync, url)


def _target_day_from_query(query: str) -> Tuple[str, date]:
    lowered = str(query or "").lower()
    today = date.today()
    if "day after tomorrow" in lowered:
        return "day_after_tomorrow", today + timedelta(days=2)
    if "tomorrow" in lowered:
        return "tomorrow", today + timedelta(days=1)
    return "today", today


def _clean_location_query(query: str) -> str:
    text = str(query or "").strip().strip("?.!")
    in_match = re.search(r"\bin\s+(.+)$", text, flags=re.IGNORECASE)
    if in_match:
        text = in_match.group(1)

    text = re.sub(
        r"\b(?:weather|forecast|current|currently|today|tomorrow|day after tomorrow|temperature|"
        r"rain|raining|snow|wind|humidity|what|about|can|you|tell|me|please|for|the|now|as we speak)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" ,")
    return text


def _build_geocode_url(location: str) -> str:
    search_name = location.split(",", 1)[0].strip() or location
    params = urllib.parse.urlencode(
        {
            "name": search_name,
            "count": 5,
            "language": "en",
            "format": "json",
        }
    )
    return f"{_GEOCODE_URL}?{params}"


def _select_geocode_result(payload: Dict[str, Any], requested_location: str) -> Optional[Dict[str, Any]]:
    results = payload.get("results") or []
    if not isinstance(results, list) or not results:
        return None

    requested_lower = requested_location.lower()
    for item in results:
        if not isinstance(item, dict):
            continue
        country = str(item.get("country") or "").lower()
        country_code = str(item.get("country_code") or "").lower()
        if country and country in requested_lower:
            return item
        if country_code and country_code in requested_lower:
            return item
    return next((item for item in results if isinstance(item, dict)), None)


def _build_forecast_url(latitude: float, longitude: float) -> str:
    params = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,precipitation_sum,wind_speed_10m_max",
            "timezone": "auto",
            "forecast_days": 4,
        }
    )
    return f"{_FORECAST_URL}?{params}"


def _weather_description(code: Any) -> str:
    try:
        return _WEATHER_CODE_DESCRIPTIONS.get(int(code), f"weather code {code}")
    except (TypeError, ValueError):
        return "unknown conditions"


def _daily_value(daily: Dict[str, Any], key: str, index: int) -> Any:
    values = daily.get(key) or []
    if isinstance(values, list) and 0 <= index < len(values):
        return values[index]
    return None


def _format_location(geo: Dict[str, Any], fallback: str) -> str:
    name = str(geo.get("name") or fallback).strip()
    country = str(geo.get("country") or "").strip()
    admin = str(geo.get("admin1") or "").strip()
    parts = [name]
    if admin and admin.lower() != name.lower():
        parts.append(admin)
    if country:
        parts.append(country)
    return ", ".join(part for part in parts if part)


async def weather_current(query: str) -> str:
    location_query = _clean_location_query(query)
    if not location_query:
        return json.dumps(
            {
                "status": "error",
                "message": "Missing location query",
            }
        )

    target_label, target_date = _target_day_from_query(query)

    try:
        geocode_payload = await _fetch_json(_build_geocode_url(location_query))
        geocode_result = _select_geocode_result(geocode_payload, location_query)
        if not geocode_result:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Could not find weather coordinates for '{location_query}'.",
                }
            )

        latitude = float(geocode_result["latitude"])
        longitude = float(geocode_result["longitude"])
        forecast_payload = await _fetch_json(_build_forecast_url(latitude, longitude))

        daily = forecast_payload.get("daily") or {}
        dates = daily.get("time") or []
        target_date_text = target_date.isoformat()
        if target_date_text not in dates:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Forecast for {target_date_text} was not available.",
                    "location": _format_location(geocode_result, location_query),
                    "source": "Open-Meteo",
                }
            )

        day_index = dates.index(target_date_text)
        weather_code = _daily_value(daily, "weather_code", day_index)
        high_c = _daily_value(daily, "temperature_2m_max", day_index)
        low_c = _daily_value(daily, "temperature_2m_min", day_index)
        precip_probability = _daily_value(daily, "precipitation_probability_max", day_index)
        precip_sum = _daily_value(daily, "precipitation_sum", day_index)
        wind_max = _daily_value(daily, "wind_speed_10m_max", day_index)
        description = _weather_description(weather_code)
        location = _format_location(geocode_result, location_query)

        current = forecast_payload.get("current") or {}
        response = {
            "status": "ok",
            "source": "Open-Meteo",
            "location": location,
            "query": str(query or ""),
            "target": target_label,
            "target_date": target_date_text,
            "forecast": {
                "condition": description,
                "temperature_max_c": high_c,
                "temperature_min_c": low_c,
                "precipitation_probability_max_percent": precip_probability,
                "precipitation_sum_mm": precip_sum,
                "wind_speed_max_kmh": wind_max,
            },
            "current": {
                "temperature_c": current.get("temperature_2m"),
                "relative_humidity_percent": current.get("relative_humidity_2m"),
                "condition": _weather_description(current.get("weather_code")),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "time": current.get("time"),
            },
            "summary": (
                f"{target_label.replace('_', ' ').title()} in {location}: {description}, "
                f"high {high_c} C, low {low_c} C, precipitation chance {precip_probability}%."
            ),
        }
        return json.dumps(response, ensure_ascii=False)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "message": "Live weather fetch failed.",
                "details": str(exc),
                "source": "Open-Meteo",
                "location": location_query,
            }
        )
