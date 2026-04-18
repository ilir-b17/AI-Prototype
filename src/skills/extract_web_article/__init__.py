import logging

import trafilatura

logger = logging.getLogger(__name__)


async def extract_web_article(url: str, max_chars: int = 12000) -> str:
    """
    Fetch a URL and extract main article text using trafilatura.
    Returns plain text suitable for summarization.
    """
    try:
        if not url or not isinstance(url, str):
            return "Error: url must be a non-empty string."
        if max_chars < 500:
            return "Error: max_chars must be at least 500."

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Error: Failed to fetch URL: {url}"

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            output_format="txt"
        )
        if not text or not text.strip():
            return "Error: No extractable article text found at the URL."

        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "\n\n[TRUNCATED]"
        logger.info(f"extract_web_article: extracted {len(text)} chars from {url}")
        return text
    except Exception as exc:
        logger.error(f"extract_web_article failed: {exc}", exc_info=True)
        return f"Error: Web extraction failed due to [{exc}]."
