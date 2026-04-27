import json
from email.message import EmailMessage
from pathlib import Path

import pytest

from src.skills.read_inbox import read_inbox


class FakeIMAP:
    def __init__(self, host: str, port: int, timeout: int = 30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.stores = []
        self.messages = {}
        self.selected_folder = None

    def login(self, _user: str, _password: str):
        return "OK", [b"Logged in"]

    def select(self, folder: str):
        self.selected_folder = folder
        return "OK", [b""]

    def search(self, _charset, _criterion):
        return "OK", [b"1 2"]

    def fetch(self, msg_id: bytes, _query: str):
        payload = self.messages.get(msg_id)
        if payload is None:
            return "NO", []
        return "OK", [(b"RFC822", payload)]

    def store(self, msg_id: bytes, op: str, flag: str):
        self.stores.append((msg_id, op, flag))
        return "OK", [b""]

    def close(self):
        return "OK", [b""]

    def logout(self):
        return "BYE", [b""]


@pytest.mark.asyncio
async def test_read_inbox_filters_untrusted_and_downloads_attachments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("AIDEN_DOWNLOADS_DIR", str(downloads))
    monkeypatch.setenv("EMAIL_USER", "user@example.com")
    monkeypatch.setenv("EMAIL_PASS", "password")
    monkeypatch.setenv("IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("IMAP_PORT", "993")
    monkeypatch.setenv("AIDEN_TRUSTED_SENDERS", "trusted@example.com")

    fake = FakeIMAP("imap.example.com", 993)

    trusted = EmailMessage()
    trusted["From"] = "Trusted Sender <trusted@example.com>"
    trusted["Subject"] = "Hello"
    trusted["Message-ID"] = "<trusted-message-id>"
    trusted["Date"] = "Mon, 01 Jan 2024 10:00:00 +0000"
    trusted.set_content("trusted body")
    trusted.add_attachment(b"abc", maintype="application", subtype="octet-stream", filename="report.txt")

    untrusted = EmailMessage()
    untrusted["From"] = "Bad Actor <bad@example.com>"
    untrusted["Subject"] = "Ignore"
    untrusted["Message-ID"] = "<untrusted-message-id>"
    untrusted["Date"] = "Mon, 01 Jan 2024 11:00:00 +0000"
    untrusted.set_content("bad body")

    fake.messages = {
        b"1": trusted.as_bytes(),
        b"2": untrusted.as_bytes(),
    }

    monkeypatch.setattr("src.skills.read_inbox.imaplib.IMAP4_SSL", lambda *args, **kwargs: fake)

    result = await read_inbox(max_emails=5, folder="INBOX", mark_as_read=False)
    payload = json.loads(result)

    assert payload["status"] == "success"
    assert len(payload["emails"]) == 1

    email_obj = payload["emails"][0]
    assert email_obj["subject"] == "Hello"
    assert email_obj["body"] == "trusted body"
    assert email_obj["message_id"] == "<trusted-message-id>"
    assert len(email_obj["attachment_paths"]) == 1
    assert Path(email_obj["attachment_paths"][0]).exists()

    # Untrusted messages are always marked as read, even when mark_as_read=False.
    assert (b"2", "+FLAGS", "\\Seen") in fake.stores
    assert (b"1", "+FLAGS", "\\Seen") not in fake.stores


@pytest.mark.asyncio
async def test_read_inbox_returns_empty_when_no_trusted_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AIDEN_DOWNLOADS_DIR", "downloads")
    monkeypatch.setenv("EMAIL_USER", "user@example.com")
    monkeypatch.setenv("EMAIL_PASS", "password")
    monkeypatch.setenv("IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("IMAP_PORT", "993")
    monkeypatch.setenv("AIDEN_TRUSTED_SENDERS", "trusted@example.com")

    fake = FakeIMAP("imap.example.com", 993)
    untrusted = EmailMessage()
    untrusted["From"] = "Bad Actor <bad@example.com>"
    untrusted["Subject"] = "Ignore"
    untrusted["Message-ID"] = "<untrusted-message-id>"
    untrusted.set_content("bad body")
    fake.messages = {b"1": untrusted.as_bytes(), b"2": untrusted.as_bytes()}

    monkeypatch.setattr("src.skills.read_inbox.imaplib.IMAP4_SSL", lambda *args, **kwargs: fake)

    result = await read_inbox()
    payload = json.loads(result)

    assert payload == {"status": "success", "emails": []}
    assert (b"1", "+FLAGS", "\\Seen") in fake.stores
    assert (b"2", "+FLAGS", "\\Seen") in fake.stores


@pytest.mark.asyncio
async def test_read_inbox_handles_connection_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMAIL_USER", "user@example.com")
    monkeypatch.setenv("EMAIL_PASS", "password")
    monkeypatch.setenv("IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("IMAP_PORT", "993")

    def _raise_connect(*_args, **_kwargs):
        raise OSError("network down")

    monkeypatch.setattr("src.skills.read_inbox.imaplib.IMAP4_SSL", _raise_connect)

    result = await read_inbox()
    payload = json.loads(result)

    assert payload["status"] == "error"
    assert payload["message"] == "Failed to read inbox"
    assert "network down" in payload["details"]
