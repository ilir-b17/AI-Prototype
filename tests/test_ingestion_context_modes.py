import json

from src.core.llm_router import CognitiveRouter
import src.skills.extract_pdf_text as pdf_skill
import src.skills.extract_web_article as web_skill


class _FakePdfPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePdf:
    def __init__(self, page_texts: list[str]) -> None:
        self.pages = [_FakePdfPage(text) for text in page_texts]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_extract_pdf_text_legacy_mode_respects_old_limits(monkeypatch):
    page_texts = ["A" * 7000, "B" * 7000]
    monkeypatch.setattr(pdf_skill, "_resolve_pdf_path", lambda _path: "doc.pdf")
    monkeypatch.setattr(pdf_skill.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(pdf_skill.pdfplumber, "open", lambda _path: _FakePdf(page_texts))

    raw = pdf_skill._extract_pdf_text_sync(
        "doc.pdf",
        max_pages=None,
        max_chars=None,
        full_context=False,
    )
    payload = json.loads(raw)

    assert payload["status"] == "success"
    assert payload["full_context"] is False
    assert payload["applied_max_chars"] == 12000
    assert payload["truncated"] is True
    assert payload["truncation_reason"] == "max_chars_limit"
    assert payload["text"].endswith("[TRUNCATED]")


def test_extract_pdf_text_full_context_returns_unfragmented_text(monkeypatch):
    page_texts = ["A" * 7000, "B" * 7000]
    monkeypatch.setattr(pdf_skill, "_resolve_pdf_path", lambda _path: "doc.pdf")
    monkeypatch.setattr(pdf_skill.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(pdf_skill.pdfplumber, "open", lambda _path: _FakePdf(page_texts))

    raw = pdf_skill._extract_pdf_text_sync(
        "doc.pdf",
        max_pages=None,
        max_chars=None,
        full_context=True,
    )
    payload = json.loads(raw)

    assert payload["status"] == "success"
    assert payload["full_context"] is True
    assert payload["applied_max_chars"] is None
    assert payload["truncated"] is False
    assert "[TRUNCATED]" not in payload["text"]
    assert len(payload["text"]) > 12000


def test_extract_web_article_legacy_mode_respects_old_limits(monkeypatch):
    monkeypatch.setattr(web_skill.trafilatura, "fetch_url", lambda _url: "<html></html>")
    monkeypatch.setattr(web_skill.trafilatura, "extract", lambda *_args, **_kwargs: "X" * 15000)

    raw = web_skill._sync_extract(
        "https://example.com",
        max_chars=None,
        full_context=False,
    )
    payload = json.loads(raw)

    assert payload["status"] == "success"
    assert payload["full_context"] is False
    assert payload["applied_max_chars"] == 12000
    assert payload["truncated"] is True
    assert payload["truncation_reason"] == "max_chars_limit"
    assert payload["text"].endswith("[TRUNCATED]")


def test_extract_web_article_full_context_returns_unfragmented_text(monkeypatch):
    monkeypatch.setattr(web_skill.trafilatura, "fetch_url", lambda _url: "<html></html>")
    monkeypatch.setattr(web_skill.trafilatura, "extract", lambda *_args, **_kwargs: "X" * 15000)

    raw = web_skill._sync_extract(
        "https://example.com",
        max_chars=None,
        full_context=True,
    )
    payload = json.loads(raw)

    assert payload["status"] == "success"
    assert payload["full_context"] is True
    assert payload["applied_max_chars"] is None
    assert payload["truncated"] is False
    assert "[TRUNCATED]" not in payload["text"]
    assert len(payload["text"]) == 15000


def test_router_full_context_ingestion_rejected_when_flag_disabled():
    router = CognitiveRouter.__new__(CognitiveRouter)
    router._enable_128k_context = False

    _prepared, result = router._prepare_ingestion_tool_arguments(
        "extract_pdf_text",
        {"full_context": True},
        [{"role": "user", "content": "Please analyze this PDF"}],
    )

    assert result is not None
    assert "disabled" in result.content.lower()


def test_router_full_context_ingestion_applies_dynamic_budget():
    router = CognitiveRouter.__new__(CognitiveRouter)
    router._enable_128k_context = True
    router._target_context_tokens = 131072
    router._ingestion_context_reserve_tokens = 4096
    router._ingestion_min_context_chars = 4000
    router._system_1_context_length = 131072
    router._ollama_options = {}

    prepared, result = router._prepare_ingestion_tool_arguments(
        "extract_web_article",
        {"full_context": True},
        [{"role": "user", "content": "summarize this article"}],
    )

    assert result is None
    assert prepared["full_context"] is True
    assert isinstance(prepared.get("max_chars"), int)
    assert prepared["max_chars"] > 0
    assert prepared["max_chars"] < 131072 * 4
