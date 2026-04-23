import asyncio
import logging

logger = logging.getLogger(__name__)


import json
import os

def _sync_web_search(query: str, max_results: int) -> str:
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return json.dumps({
                "status": "success",
                "message": f"No results found for query: {query}",
                "results": []
            })

        parsed_results = []
        for r in results:
            parsed_results.append({
                "title": r.get('title', 'No title'),
                "url": r.get('href', 'N/A'),
                "snippet": r.get('body', '')[:300]
            })

        return json.dumps({
            "status": "success",
            "results": parsed_results
        }, indent=2)

    except Exception as exc:
        return json.dumps({
            "status": "error",
            "message": "Web search failed",
            "details": str(exc),
            "suggestion": "The search provider may be rate limiting you. Try again later or use different keywords."
        })


async def web_search(query: str, max_results: int = 3) -> str:
    logger.info(f"web_search: {query!r} (max={max_results})")

    if not isinstance(query, str) or not query.strip():
        return json.dumps({
            "status": "error",
            "message": "Invalid query",
            "details": "The query must be a non-empty string."
        })

    try:
        max_results = int(max_results)
        max_results = min(max(1, max_results), 10)
    except ValueError:
        logger.warning(f"Invalid max_results '{max_results}', defaulting to 3")
        max_results = 3

    try:
        timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "15.0"))
    except ValueError:
        timeout_seconds = 15.0

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_sync_web_search, query, max_results),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": f"Web search timed out after {timeout_seconds}s",
            "details": "The search engine took too long to respond."
        })
