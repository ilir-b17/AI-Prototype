"""Lightweight weather skill used by fast-path routing and tool execution."""

import asyncio
import json


async def weather_current(query: str) -> str:
    await asyncio.sleep(0)
    location = str(query or "").strip()
    if not location:
        return json.dumps(
            {
                "status": "error",
                "message": "Missing location query",
            }
        )

    # Keep this implementation dependency-free and deterministic by default.
    return json.dumps(
        {
            "status": "ok",
            "location": location,
            "note": "Live weather fetch is not enabled in this minimal skill implementation.",
        }
    )
