import asyncio
import logging
import os

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)

_DOWNLOADS_DIR = os.path.abspath(
    os.getenv(
        "AIDEN_DOWNLOADS_DIR",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "..",
            "downloads",
        ),
    )
)


def _resolve_pdf_path(file_path: str) -> str:
    normalized = file_path.strip()
    if not normalized:
        return normalized

    # Normalize slashes for Windows paths and common /download/ inputs
    normalized = normalized.replace("/", os.sep)

    # If it's an absolute path and exists, use it as-is.
    if os.path.isabs(normalized):
        if os.path.exists(normalized):
            return normalized
        # If it looks like "/download/Tool.pdf", fall back to downloads basename.
        base = os.path.basename(normalized)
        return os.path.join(_DOWNLOADS_DIR, base)

    # Relative path: assume downloads directory.
    return os.path.join(_DOWNLOADS_DIR, normalized)


import json

def _extract_pdf_text_sync(file_path: str, max_pages: int, max_chars: int) -> str:
    resolved = _resolve_pdf_path(file_path)
    if not resolved:
        return json.dumps({
            "status": "error",
            "message": "Invalid file_path",
            "details": "The file_path must be a non-empty string."
        })

    if not os.path.exists(resolved):
        return json.dumps({
            "status": "error",
            "message": f"PDF not found at '{resolved}'",
            "suggestion": "Use manage_file_system 'list' on the downloads folder or the parent directory to confirm the correct filename."
        })

    logger.info(f"extract_pdf_text: reading '{resolved}' (max_pages={max_pages}, max_chars={max_chars})")

    text_chunks = []
    extracted_any = False

    try:
        with pdfplumber.open(resolved) as pdf:
            total_pages = len(pdf.pages)
            page_limit = min(max_pages, total_pages)
            for idx in range(page_limit):
                page_text = pdf.pages[idx].extract_text() or ""
                if page_text.strip():
                    extracted_any = True
                text_chunks.append(page_text)
    except Exception:
        reader = PdfReader(resolved)
        total_pages = len(reader.pages)
        page_limit = min(max_pages, total_pages)
        for idx in range(page_limit):
            page = reader.pages[idx]
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_any = True
            text_chunks.append(page_text)

    if not extracted_any:
        return json.dumps({
            "status": "error",
            "message": "No extractable text found in the PDF",
            "details": "The document may be scanned or image-only.",
            "suggestion": "Consider trying an OCR tool if you must read this document."
        })

    combined = "\n".join(t.strip() for t in text_chunks if t)
    is_truncated = False
    if len(combined) > max_chars:
        combined = combined[:max_chars].rstrip() + "\n\n[TRUNCATED]"
        is_truncated = True

    return json.dumps({
        "status": "success",
        "file": os.path.basename(resolved),
        "text": combined.strip(),
        "truncated": is_truncated
    }, indent=2)


async def extract_pdf_text(
    file_path: str,
    max_pages: int = 20,
    max_chars: int = 12000
) -> str:
    """
    Extract text from a PDF file using pypdf.

    Returns plain text suitable for summarization. If the document has no
    extractable text (e.g., scanned images), returns an explicit error.
    """
    try:
        max_pages = int(max_pages)
        max_chars = int(max_chars)
    except ValueError:
        max_pages = 20
        max_chars = 12000

    if max_pages < 1:
        max_pages = 20
    if max_chars < 200:
        max_chars = 12000

    try:
        timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "30.0"))
    except ValueError:
        timeout_seconds = 30.0

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_extract_pdf_text_sync, file_path, max_pages, max_chars),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        return json.dumps({
            "status": "error",
            "message": f"PDF extraction timed out after {timeout_seconds}s",
            "details": "The PDF might be too large or complex to process in the allotted time."
        })
    except Exception as exc:
        logger.error(f"extract_pdf_text failed: {exc}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "PDF extraction failed",
            "details": str(exc)
        })
