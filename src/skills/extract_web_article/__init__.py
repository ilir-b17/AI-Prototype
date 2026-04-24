import asyncio
import json
import logging
import os
from typing import Any, Optional

import trafilatura

logger = logging.getLogger(__name__)

LEGACY_DEFAULT_MAX_CHARS = 12000
TRUNCATION_MARKER = "\n\n[TRUNCATED]"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_char_limit(max_chars: Optional[int], full_context: bool) -> Optional[int]:
    if full_context:
        return _coerce_positive_int(max_chars)
    resolved = _coerce_positive_int(max_chars)
    if resolved is None or resolved < 500:
        return LEGACY_DEFAULT_MAX_CHARS
    return resolved


def _sync_extract(url: str, max_chars: Optional[int], full_context: bool = False) -> str:
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
        char_limit = _resolve_char_limit(max_chars, full_context)
        is_truncated = False
        if char_limit is not None and len(text) > char_limit:
            text = text[:char_limit].rstrip() + TRUNCATION_MARKER
            is_truncated = True

        logger.info(f"extract_web_article: extracted {len(text)} chars from {url}")
        return json.dumps({
            "status": "success",
            "text": text,
            "truncated": is_truncated,
            "truncation_reason": "max_chars_limit" if is_truncated else "",
            "full_context": full_context,
            "applied_max_chars": char_limit,
        }, indent=2)
    except Exception as exc:
        logger.error(f"extract_web_article failed: {exc}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Web extraction failed",
            "details": str(exc)
        })

async def extract_web_article(
    url: str,
    max_chars: Optional[int] = None,
    full_context: bool = False,
) -> str:
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

    full_context = _coerce_bool(full_context, default=False)
    max_chars = _resolve_char_limit(max_chars, full_context)

    try:
        timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "20.0"))
    except ValueError:
        timeout_seconds = 20.0

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_sync_extract, url, max_chars, full_context),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": f"Extraction timed out after {timeout_seconds}s",
            "details": f"The website at {url} took too long to download and process."
        })
    except Exception as exc:
        logger.error(f"extract_web_article failed: {exc}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Web extraction failed",
            "details": str(exc)
        })
