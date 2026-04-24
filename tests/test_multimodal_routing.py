import pytest
from unittest.mock import AsyncMock

from src.core.llm_router import CognitiveRouter, RouterResult


def test_prepare_system_1_messages_adds_text_fallback_when_audio_not_supported():
    router = CognitiveRouter.__new__(CognitiveRouter)
    router._enable_native_audio = True
    router._system_1_capabilities = set()

    prepared = router._prepare_system_1_messages(
        [
            {
                "role": "user",
                "content": "Summarize this voice note",
                "audio_bytes": b"\x01\x02\x03",
                "audio_mime_type": "audio/ogg",
            }
        ]
    )

    assert len(prepared) == 1
    assert prepared[0]["role"] == "user"
    assert "Voice note attached" in prepared[0]["content"]
    assert "audios" not in prepared[0]


@pytest.mark.asyncio
async def test_route_to_system_2_strips_audio_bytes_before_provider_call():
    router = CognitiveRouter.__new__(CognitiveRouter)
    router._system2_cooldown_until = 0.0
    router.ollama_cloud_client = None
    router.groq_client = object()
    router.gemini_client = None
    router._route_to_groq = AsyncMock(return_value=RouterResult(status="ok", content="ok"))

    result = await router.route_to_system_2(
        [
            {
                "role": "user",
                "content": "Please process this",
                "audio_bytes": b"raw-audio",
                "audio_mime_type": "audio/ogg",
            }
        ]
    )

    assert result.status == "ok"
    assert result.content == "ok"

    router._route_to_groq.assert_awaited_once()
    args, _kwargs = router._route_to_groq.await_args
    forwarded_messages = args[0]
    assert len(forwarded_messages) == 1
    assert forwarded_messages[0]["role"] == "user"
    assert forwarded_messages[0]["content"] == "Please process this"
    assert "audio_bytes" not in forwarded_messages[0]
