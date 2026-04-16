import asyncio
import logging

logger = logging.getLogger(__name__)


def _sync_web_search(query: str, max_results: int) -> str:
    try:
        from ddgs import DDGS
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return f"No results found for: {query}"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. {r.get('title', 'No title')}\n"
                f"   URL: {r.get('href', 'N/A')}\n"
                f"   {r.get('body', '')[:200]}"
            )
        return "\n\n".join(lines)
    except Exception as exc:
        return f"Error: Web search failed — {exc}"


async def web_search(query: str, max_results: int = 3) -> str:
    logger.info(f"web_search: {query!r} (max={max_results})")
    max_results = min(max(1, max_results), 10)
    return await asyncio.to_thread(_sync_web_search, query, max_results)
