import logging

import trafilatura

logger = logging.getLogger(__name__)


import json
import asyncio
import os

def _sync_extract(url: str, max_chars: int) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return json.dumps({
                "status": "error",
                "message": "Failed to fetch URL",
                "details": f"The website ({url}) may be down, blocking bots, or require Javascript.",
                "suggestion": "Check if the URL is correct or if you can find a different source."
            })

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            output_format="txt"
        )
        if not text or not text.strip():
            return json.dumps({
                "status": "error",
                "message": "No extractable article text found",
                "details": "The page might be an image, an empty shell, or highly dynamic requiring a full browser.",
                "suggestion": "This URL might not contain a standard text article."
            })

        text = text.strip()
        is_truncated = False
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n\n[TRUNCATED]"
            is_truncated = True

        logger.info(f"extract_web_article: extracted {len(text)} chars from {url}")
        return json.dumps({
            "status": "success",
            "text": text,
            "truncated": is_truncated
        }, indent=2)
    except Exception as exc:
        logger.error(f"extract_web_article failed: {exc}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Web extraction failed",
            "details": str(exc)
        })

async def extract_web_article(url: str, max_chars: int = 12000) -> str:
    """
    Fetch a URL and extract main article text using trafilatura.
    Returns plain text suitable for summarization.
    """
    if not url or not isinstance(url, str):
        return json.dumps({
            "status": "error",
            "message": "Invalid URL",
            "details": "The URL must be a non-empty string."
        })

    try:
        max_chars = int(max_chars)
        if max_chars < 500:
            max_chars = 500
    except ValueError:
        max_chars = 12000

    try:
        timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "20.0"))
    except ValueError:
        timeout_seconds = 20.0

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_sync_extract, url, max_chars),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": f"Extraction timed out after {timeout_seconds}s",
            "details": f"The website at {url} took too long to download and process."
        })
