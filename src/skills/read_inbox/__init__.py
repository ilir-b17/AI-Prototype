import asyncio
import json
import os
import re
from email import policy
from email.message import Message
from email.parser import BytesParser
from email.utils import parseaddr
from html import unescape
from pathlib import Path
from typing import Any, Dict, List, Set

import imaplib

from src.skills._common.path_guard import get_default_allowed_roots, resolve_confined_path

MAX_PATH_SEGMENT_LENGTH = 120


def _downloads_dir() -> str:
    return os.path.abspath(
        os.getenv(
            "AIDEN_DOWNLOADS_DIR",
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "..",
                "downloads",
            ),
        )
    )


def _coerce_bool(value: Any, default: bool = True) -> bool:
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
    return bool(value)


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _parse_trusted_senders(raw_value: str) -> Set[str]:
    trusted: Set[str] = set()
    for item in str(raw_value or "").split(","):
        addr = parseaddr(item.strip())[1].strip().lower()
        if addr:
            trusted.add(addr)
    return trusted


def _sanitize_segment(value: str, fallback: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return fallback
    raw = raw.strip("<>")
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", raw)
    return sanitized[:MAX_PATH_SEGMENT_LENGTH] or fallback


def _extract_body(message: Message) -> str:
    plain_parts: List[str] = []
    html_parts: List[str] = []

    for part in message.walk():
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment":
            continue

        content_type = str(part.get_content_type() or "").lower()
        try:
            payload = part.get_payload(decode=True)
            charset = part.get_content_charset() or "utf-8"
            text = (payload or b"").decode(charset, errors="replace")
        except Exception:
            try:
                text = str(part.get_content())
            except Exception:
                text = ""

        text = text.strip()
        if not text:
            continue

        if content_type == "text/plain":
            plain_parts.append(text)
        elif content_type == "text/html":
            html_parts.append(text)

    if plain_parts:
        return "\n\n".join(plain_parts).strip()

    if html_parts:
        merged = "\n\n".join(html_parts)
        no_tags = re.sub(r"<[^>]+>", " ", merged)
        compact = re.sub(r"\s+", " ", unescape(no_tags)).strip()
        return compact

    return ""


def _save_attachments(message: Message, message_id: str, allowed_roots: List[str]) -> List[str]:
    download_root = _downloads_dir()
    message_dir_name = _sanitize_segment(message_id, "message")

    target_dir = os.path.join(download_root, message_dir_name)
    confined_dir = resolve_confined_path(target_dir, allowed_roots)
    os.makedirs(confined_dir, exist_ok=True)

    attachment_paths: List[str] = []
    used_names: Dict[str, int] = {}

    for part in message.walk():
        if part.is_multipart():
            continue
        if part.get_content_disposition() != "attachment":
            continue

        payload = part.get_payload(decode=True)
        if payload is None:
            continue

        filename = _sanitize_segment(part.get_filename() or "attachment", "attachment")
        if filename in used_names:
            used_names[filename] += 1
            stem, suffix = os.path.splitext(filename)
            filename = f"{stem}_{used_names[filename]}{suffix}"
        else:
            used_names[filename] = 0

        target_file = os.path.join(confined_dir, filename)
        confined_file = resolve_confined_path(target_file, allowed_roots)
        with open(confined_file, "wb") as handle:
            handle.write(payload)
        attachment_paths.append(confined_file)

    return attachment_paths


def _mark_seen(mail: imaplib.IMAP4_SSL, msg_id: bytes) -> None:
    try:
        mail.store(msg_id, "+FLAGS", "\\Seen")
    except Exception:
        pass


def _read_inbox_sync(max_emails: int, folder: str, mark_as_read: bool) -> str:
    email_user = str(os.getenv("EMAIL_USER") or "").strip()
    email_pass = str(os.getenv("EMAIL_PASS") or "").strip()
    imap_host = str(os.getenv("IMAP_HOST") or "imap.gmail.com").strip() or "imap.gmail.com"
    imap_port_raw = str(os.getenv("IMAP_PORT") or "993").strip() or "993"

    if not email_user or not email_pass:
        return json.dumps(
            {
                "status": "error",
                "message": "EMAIL_USER and EMAIL_PASS are not configured. Please update your .env file.",
            }
        )

    try:
        imap_port = int(imap_port_raw)
    except ValueError:
        return json.dumps({"status": "error", "message": "IMAP_PORT must be a valid integer"})

    trusted_senders = _parse_trusted_senders(os.getenv("AIDEN_TRUSTED_SENDERS", ""))
    allowed_roots = get_default_allowed_roots()

    mail = None
    try:
        mail = imaplib.IMAP4_SSL(imap_host, imap_port, timeout=30)
        mail.login(email_user, email_pass)

        status, _ = mail.select(folder)
        if status != "OK":
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Failed to open IMAP folder '{folder}'",
                }
            )

        status, data = mail.search(None, "UNSEEN")
        if status != "OK":
            return json.dumps(
                {
                    "status": "error",
                    "message": "Failed to search inbox",
                }
            )

        message_ids = data[0].split() if data and data[0] else []
        selected_ids = list(reversed(message_ids[-max_emails:]))

        emails: List[Dict[str, Any]] = []

        for msg_id in selected_ids:
            status, payload = mail.fetch(msg_id, "(RFC822)")
            if status != "OK" or not payload:
                continue

            raw_message = None
            for item in payload:
                # IMAP fetch response is typically a tuple like:
                # (b'<sequence-id> (RFC822 {bytes})', raw_rfc822_bytes)
                # where index 1 contains the RFC822 body bytes.
                if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (bytes, bytearray)):
                    raw_message = bytes(item[1])
                    break
            if not raw_message:
                continue

            message = BytesParser(policy=policy.default).parsebytes(raw_message)

            sender_raw = str(message.get("From") or "")
            sender_email = parseaddr(sender_raw)[1].strip().lower()
            is_trusted = bool(sender_email) and sender_email in trusted_senders

            if not is_trusted:
                _mark_seen(mail, msg_id)
                continue

            message_id = str(message.get("Message-ID") or f"message-{msg_id.decode(errors='ignore')}").strip()
            email_data = {
                "sender": sender_raw,
                "subject": str(message.get("Subject") or ""),
                "body": _extract_body(message),
                "message_id": message_id,
                "date": str(message.get("Date") or ""),
                "attachment_paths": _save_attachments(message, message_id, allowed_roots),
            }
            emails.append(email_data)

            if mark_as_read:
                _mark_seen(mail, msg_id)

        return json.dumps({"status": "success", "emails": emails}, indent=2)
    except (imaplib.IMAP4.error, OSError, TimeoutError) as exc:
        return json.dumps(
            {
                "status": "error",
                "message": "Failed to read inbox",
                "details": str(exc),
            }
        )
    except PermissionError:
        return json.dumps(
            {
                "status": "error",
                "message": "Attachment storage path is outside the allowed roots",
            }
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "message": "Unexpected inbox read failure",
                "details": str(exc),
            }
        )
    finally:
        if mail is not None:
            try:
                mail.close()
            except Exception:
                pass
            try:
                mail.logout()
            except Exception:
                pass


async def read_inbox(max_emails: int = 5, folder: str = "INBOX", mark_as_read: bool = True) -> str:
    normalized_max = _coerce_positive_int(max_emails, default=5)
    normalized_folder = str(folder or "INBOX").strip() or "INBOX"
    normalized_mark_as_read = _coerce_bool(mark_as_read, default=True)
    return await asyncio.to_thread(_read_inbox_sync, normalized_max, normalized_folder, normalized_mark_as_read)
