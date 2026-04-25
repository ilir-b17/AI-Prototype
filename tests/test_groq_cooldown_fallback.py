import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.llm_router import CognitiveRouter


@pytest.mark.asyncio
async def test_direct_groq_path_respects_cooldown() -> None:
    router = CognitiveRouter.__new__(CognitiveRouter)
    router._system2_cooldown_until = time.time() + 600
    router._persist_cooldown_cb = None
    router.groq_model = "llama-3.3-70b-versatile"
    router.registry = MagicMock()
    router.registry.get_schemas.return_value = []

    groq_create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="ok"),
                )
            ]
        )
    )
    router.groq_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=groq_create))
    )

    result = await CognitiveRouter._route_to_groq(
        router,
        [{"role": "user", "content": "hi"}],
        allowed_tools=None,
    )

    groq_create.assert_not_awaited()
    assert result.status == "ok"
    assert "[System 2 - Error]:" in result.content
    assert "Rate limited. Retry in" in result.content


@pytest.mark.asyncio
async def test_ollama_cloud_fallback_respects_groq_cooldown() -> None:
    router = CognitiveRouter.__new__(CognitiveRouter)
    router._system2_cooldown_until = time.time() + 600
    router._persist_cooldown_cb = None
    router.system_2_model = "deepseek-v3.2"
    router.groq_model = "llama-3.3-70b-versatile"
    router.registry = MagicMock()
    router.registry.get_schemas.return_value = []

    router.ollama_cloud_client = MagicMock()
    router.ollama_cloud_client.chat = AsyncMock(side_effect=RuntimeError("ollama cloud unavailable"))

    groq_create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="ok"),
                )
            ]
        )
    )
    router.groq_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=groq_create))
    )

    result = await CognitiveRouter._route_to_ollama_cloud(
        router,
        [{"role": "user", "content": "hello"}],
        allowed_tools=None,
    )

    groq_create.assert_not_awaited()
    assert result.status == "ok"
    assert "[System 2 - Error]:" in result.content
    assert "Rate limited. Retry in" in result.content


def test_cooldown_extension_never_shrinks(monkeypatch: pytest.MonkeyPatch) -> None:
    router = CognitiveRouter.__new__(CognitiveRouter)
    router._persist_cooldown_cb = None
    monkeypatch.setattr("src.core.llm_router.time.time", lambda: 1_000.0)
    router._system2_cooldown_until = 2_800.0  # 30 minutes from mocked now.

    result = CognitiveRouter._handle_groq_rate_limit(
        router,
        "rate limit reached, try again in 5m0s",
    )

    assert result is not None
    assert router._system2_cooldown_until == pytest.approx(2_800.0)


def test_cooldown_extension_grows_when_new_window_is_longer(monkeypatch: pytest.MonkeyPatch) -> None:
    router = CognitiveRouter.__new__(CognitiveRouter)
    router._persist_cooldown_cb = None
    monkeypatch.setattr("src.core.llm_router.time.time", lambda: 1_000.0)
    router._system2_cooldown_until = 1_060.0  # 1 minute from mocked now.

    result = CognitiveRouter._handle_groq_rate_limit(
        router,
        "rate limit reached, try again in 30m0s",
    )

    assert result is not None
    assert router._system2_cooldown_until == pytest.approx(2_800.0)
