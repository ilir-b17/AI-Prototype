import asyncio
import pytest
import sys
import os

# Complex Mocking for chromadb, google, ollama, groq
from unittest.mock import MagicMock

sys.modules['chromadb'] = MagicMock()
sys.modules['chromadb.config'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()
sys.modules['ollama'] = MagicMock()
sys.modules['groq'] = MagicMock()
sys.modules['langchain'] = MagicMock()
sys.modules['langgraph'] = MagicMock()
sys.modules['langgraph.graph'] = MagicMock()

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock env vars
os.environ['TELEGRAM_BOT_TOKEN'] = 'mock_token'
os.environ['ADMIN_USER_ID'] = '12345'

from src.interfaces import telegram_bot
from src.interfaces.telegram_bot import handle_document_message, handle_message, handle_voice_message, status
from telegram import Update, Message, Chat, User
from unittest.mock import AsyncMock

telegram_bot.ADMIN_USER_ID = 12345

@pytest.mark.asyncio
async def test_telegram_bot_timeout():
    # Setup mocks
    mock_update = MagicMock(spec=Update)
    mock_message = AsyncMock(spec=Message)
    mock_update.message = mock_message
    mock_update.effective_user = MagicMock(spec=User)
    mock_update.effective_user.id = 12345
    mock_update.message.text = "Hello"
    mock_update.effective_chat = MagicMock(spec=Chat)
    mock_update.effective_chat.id = 12345

    mock_context = MagicMock()
    mock_context.bot = AsyncMock()
    mock_context.bot.get_file = AsyncMock()

    # Mock orchestrator to simulate a hung API call
    mock_orchestrator = MagicMock()

    async def slow_process_message(*args, **kwargs):
        await asyncio.sleep(2)
        return "Done"

    mock_orchestrator.process_message = slow_process_message
    telegram_bot.orchestrator = mock_orchestrator
    telegram_bot.AGENT_TIMEOUT_SECONDS = 0.5  # Override the timeout for fast testing

    try:
        # We limit the test execution to 1 second.
        # If telegram_bot doesn't have an internal wait_for timeout, the test will
        # fail with a TimeoutError here because handle_message will sleep for 2s.
        await asyncio.wait_for(handle_message(mock_update, mock_context), timeout=1.0)
    except asyncio.TimeoutError:
        pytest.fail("handle_message hung and did not catch the timeout internally!")

    # After we implement the fix, handle_message should complete in 0.5s internally,
    # catching the TimeoutError from its wait_for and replying to the user.
    mock_update.message.reply_text.assert_called_with("Request timed out. Please try again.")


@pytest.mark.asyncio
async def test_voice_message_disabled_flag_returns_graceful_message():
    mock_update = MagicMock(spec=Update)
    mock_message = AsyncMock(spec=Message)
    mock_update.message = mock_message
    mock_update.effective_user = MagicMock(spec=User)
    mock_update.effective_user.id = 12345
    mock_update.effective_chat = MagicMock(spec=Chat)
    mock_update.effective_chat.id = 12345
    mock_update.message.voice = MagicMock(file_id="voice-file-id", duration=3, file_size=2048)

    mock_context = MagicMock()
    mock_context.bot = AsyncMock()

    mock_orchestrator = MagicMock()
    mock_orchestrator.process_message = AsyncMock(return_value="should-not-run")
    telegram_bot.orchestrator = mock_orchestrator

    prior_flag = telegram_bot.ENABLE_NATIVE_AUDIO
    telegram_bot.ENABLE_NATIVE_AUDIO = False
    try:
        await handle_voice_message(mock_update, mock_context)
    finally:
        telegram_bot.ENABLE_NATIVE_AUDIO = prior_flag

    mock_update.message.reply_text.assert_called_with(
        "Native audio input is disabled. Set ENABLE_NATIVE_AUDIO=true to enable voice notes."
    )
    mock_context.bot.get_file.assert_not_awaited()
    mock_orchestrator.process_message.assert_not_called()


@pytest.mark.asyncio
async def test_status_command_reports_operational_counts():
    mock_update = MagicMock(spec=Update)
    mock_message = AsyncMock(spec=Message)
    mock_update.message = mock_message
    mock_update.effective_user = MagicMock(spec=User)
    mock_update.effective_user.id = 12345

    class _Registry:
        def __len__(self):
            return 7

        def get_load_errors(self):
            return [("broken_skill", "bad schema")]

    mock_orchestrator = MagicMock()
    mock_orchestrator.cognitive_router.registry = _Registry()
    mock_orchestrator.pending_mfa = {"user-1": {}}
    mock_orchestrator.pending_hitl_state = {"user-2": {}, "user-3": {}}
    mock_orchestrator._get_predictive_energy_budget_remaining = AsyncMock(return_value=81)

    mock_context = MagicMock()
    mock_context.bot_data = {"orchestrator": mock_orchestrator}

    await status(mock_update, mock_context)

    sent_text = mock_update.message.reply_text.await_args.args[0]
    assert "Loaded skills: 7" in sent_text
    assert "Failed skills: 1 (broken_skill)" in sent_text
    assert "Pending MFA: 1" in sent_text
    assert "Pending HITL: 2" in sent_text
    assert "Predictive energy budget: 81" in sent_text


@pytest.mark.asyncio
async def test_document_message_uses_caption_as_tool_approval_context():
    mock_update = MagicMock(spec=Update)
    mock_message = AsyncMock(spec=Message)
    mock_update.message = mock_message
    mock_update.effective_user = MagicMock(spec=User)
    mock_update.effective_user.id = 12345
    mock_update.message.caption = "YES"
    mock_update.message.document = MagicMock(file_name="long_tool.py", file_size=4096)

    mock_context = MagicMock()
    mock_context.bot = AsyncMock()

    mock_orchestrator = MagicMock()
    mock_orchestrator.pending_tool_approval = {"12345": {"synthesis": {}}}
    mock_orchestrator.process_message = AsyncMock(return_value="approved")
    telegram_bot.orchestrator = mock_orchestrator
    telegram_bot.AGENT_TIMEOUT_SECONDS = 1.0

    await handle_document_message(mock_update, mock_context)

    mock_orchestrator.process_message.assert_awaited_once()
    args, _kwargs = mock_orchestrator.process_message.await_args
    assert args[0].startswith("YES")
    assert "long_tool.py" in args[0]
    assert args[1] == "12345"
    mock_update.message.reply_text.assert_called_with("approved")


@pytest.mark.asyncio
async def test_voice_message_enabled_passes_in_memory_payload_to_orchestrator():
    mock_update = MagicMock(spec=Update)
    mock_message = AsyncMock(spec=Message)
    mock_update.message = mock_message
    mock_update.effective_user = MagicMock(spec=User)
    mock_update.effective_user.id = 12345
    mock_update.effective_chat = MagicMock(spec=Chat)
    mock_update.effective_chat.id = 12345
    mock_update.message.caption = "Please transcribe"
    mock_update.message.voice = MagicMock(file_id="voice-file-id", duration=4, file_size=4096)

    def _download_to_memory(*args, **kwargs):
        out = kwargs.get("out")
        out.write(b"voice-bytes")

    mock_file = MagicMock()
    mock_file.download_to_memory = AsyncMock(side_effect=_download_to_memory)

    mock_context = MagicMock()
    mock_context.bot = AsyncMock()
    mock_context.bot.get_file = AsyncMock(return_value=mock_file)
    mock_context.bot.send_chat_action = AsyncMock()

    mock_orchestrator = MagicMock()
    mock_orchestrator.process_message = AsyncMock(return_value="voice response")
    telegram_bot.orchestrator = mock_orchestrator
    telegram_bot.AGENT_TIMEOUT_SECONDS = 1.0

    prior_flag = telegram_bot.ENABLE_NATIVE_AUDIO
    telegram_bot.ENABLE_NATIVE_AUDIO = True
    try:
        await handle_voice_message(mock_update, mock_context)
    finally:
        telegram_bot.ENABLE_NATIVE_AUDIO = prior_flag

    mock_context.bot.get_file.assert_awaited_once_with("voice-file-id")
    mock_orchestrator.process_message.assert_awaited_once()
    args, _kwargs = mock_orchestrator.process_message.await_args
    payload = args[0]
    assert payload["text"] == "Please transcribe"
    assert payload["audio_bytes"] == b"voice-bytes"
    assert payload["audio_mime_type"] == "audio/ogg"
    assert payload["audio_source"] == "telegram_voice"
    assert args[1] == "12345"
    mock_update.message.reply_text.assert_called_with("voice response")
