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
from src.interfaces.telegram_bot import handle_message
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
