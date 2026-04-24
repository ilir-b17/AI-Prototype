import asyncio
import json
import logging
import os
from typing import Any, Optional

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)

LEGACY_DEFAULT_MAX_PAGES = 20
LEGACY_DEFAULT_MAX_CHARS = 12000
TRUNCATION_MARKER = "\n\n[TRUNCATED]"

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


def _resolve_runtime_limits(
    max_pages: Optional[int],
    max_chars: Optional[int],
    full_context: bool,
) -> tuple[Optional[int], Optional[int]]:
    if full_context:
        return _coerce_positive_int(max_pages), _coerce_positive_int(max_chars)

    resolved_pages = _coerce_positive_int(max_pages) or LEGACY_DEFAULT_MAX_PAGES
    resolved_chars = _coerce_positive_int(max_chars)
    if resolved_chars is None or resolved_chars < 200:
        resolved_chars = LEGACY_DEFAULT_MAX_CHARS
    return resolved_pages, resolved_chars


def _append_limited_text(
    parts: list[str],
    page_text: str,
    current_len: int,
    char_limit: Optional[int],
) -> tuple[int, bool]:
    text = (page_text or "").strip()
    if not text:
        return current_len, False

    candidate = ("\n" if parts else "") + text
    if char_limit is None:
        parts.append(candidate)
        return current_len + len(candidate), False

    remaining = char_limit - current_len
    if remaining <= 0:
        return current_len, True

    if len(candidate) <= remaining:
        parts.append(candidate)
        return current_len + len(candidate), False

    parts.append(candidate[:remaining])
    return current_len + remaining, True


def _collect_page_text(
    *,
    total_pages: int,
    page_cap: Optional[int],
    char_cap: Optional[int],
    read_page_text,
) -> tuple[list[str], bool, int, bool]:
    page_limit = min(page_cap, total_pages) if page_cap is not None else total_pages
    text_chunks: list[str] = []
    extracted_any = False
    current_len = 0
    is_truncated = False

    for idx in range(page_limit):
        page_text = read_page_text(idx) or ""
        if page_text.strip():
            extracted_any = True
        current_len, page_truncated = _append_limited_text(
            text_chunks,
            page_text,
            current_len,
            char_cap,
        )
        if page_truncated:
            is_truncated = True
            break

    return text_chunks, extracted_any, page_limit, is_truncated


def _extract_pdf_text_sync(
    file_path: str,
    max_pages: Optional[int],
    max_chars: Optional[int],
    full_context: bool = False,
) -> str:
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

    page_cap, char_cap = _resolve_runtime_limits(max_pages, max_chars, full_context)
    logger.info(
        "extract_pdf_text: reading '%s' (full_context=%s, max_pages=%s, max_chars=%s)",
        resolved,
        full_context,
        page_cap,
        char_cap,
    )

    text_chunks: list[str] = []
    extracted_any = False
    total_pages = 0
    page_limit = 0
    is_truncated = False

    try:
        with pdfplumber.open(resolved) as pdf:
            total_pages = len(pdf.pages)
            text_chunks, extracted_any, page_limit, is_truncated = _collect_page_text(
                total_pages=total_pages,
                page_cap=page_cap,
                char_cap=char_cap,
                read_page_text=lambda idx: pdf.pages[idx].extract_text(),
            )
    except Exception:
        reader = PdfReader(resolved)
        total_pages = len(reader.pages)
        text_chunks, extracted_any, page_limit, is_truncated = _collect_page_text(
            total_pages=total_pages,
            page_cap=page_cap,
            char_cap=char_cap,
            read_page_text=lambda idx: reader.pages[idx].extract_text(),
        )

    if not extracted_any:
        return json.dumps({
            "status": "error",
            "message": "No extractable text found in the PDF",
            "details": "The document may be scanned or image-only.",
            "suggestion": "Consider trying an OCR tool if you must read this document."
        })

    combined = "".join(text_chunks).strip()
    if is_truncated:
        combined = combined.rstrip() + TRUNCATION_MARKER

    return json.dumps({
        "status": "success",
        "file": os.path.basename(resolved),
        "text": combined.strip(),
        "truncated": is_truncated,
        "truncation_reason": "max_chars_limit" if is_truncated else "",
        "full_context": full_context,
        "applied_max_pages": page_limit,
        "applied_max_chars": char_cap,
        "total_pages": total_pages,
    }, indent=2)


async def extract_pdf_text(
    file_path: str,
    max_pages: Optional[int] = None,
    max_chars: Optional[int] = None,
    full_context: bool = False,
) -> str:
    """
    Extract text from a PDF file using pypdf.

    Returns plain text suitable for summarization. If the document has no
    extractable text (e.g., scanned images), returns an explicit error.
    """
    full_context = _coerce_bool(full_context, default=False)
    max_pages, max_chars = _resolve_runtime_limits(max_pages, max_chars, full_context)

    try:
        timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "30.0"))
    except ValueError:
        timeout_seconds = 30.0

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_extract_pdf_text_sync, file_path, max_pages, max_chars, full_context),
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
