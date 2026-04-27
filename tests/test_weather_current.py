import json
import asyncio
from datetime import date, timedelta

import pytest

import src.skills.weather_current as weather_module


@pytest.mark.asyncio
async def test_weather_current_fetches_tomorrow_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    async def fake_fetch_json(url: str) -> dict:
        await asyncio.sleep(0)
        if "geocoding-api" in url:
            return {
                "results": [
                    {
                        "name": "Vienna",
                        "admin1": "Vienna",
                        "country": "Austria",
                        "latitude": 48.2085,
                        "longitude": 16.3721,
                    }
                ]
            }
        return {
            "current": {
                "time": f"{date.today().isoformat()}T12:00",
                "temperature_2m": 15.2,
                "relative_humidity_2m": 55,
                "weather_code": 2,
                "wind_speed_10m": 12.0,
            },
            "daily": {
                "time": [tomorrow],
                "weather_code": [61],
                "temperature_2m_max": [17.5],
                "temperature_2m_min": [9.0],
                "precipitation_probability_max": [60],
                "precipitation_sum": [2.4],
                "wind_speed_10m_max": [18.0],
            },
        }

    monkeypatch.setattr(weather_module, "_fetch_json", fake_fetch_json)

    raw_result = await weather_module.weather_current(
        "weather forecast for tomorrow in Vienna, Austria"
    )
    result = json.loads(raw_result)

    assert result["status"] == "ok"
    assert result["source"] == "Open-Meteo"
    assert result["location"] == "Vienna, Austria"
    assert result["target"] == "tomorrow"
    assert result["target_date"] == tomorrow
    assert result["forecast"]["condition"] == "slight rain"
    assert "Live weather fetch is not enabled" not in raw_result