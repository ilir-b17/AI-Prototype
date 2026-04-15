"""
Telegram Bot Interface for Autonomous Biomimetic AI Agent.

This module implements an asynchronous Telegram bot that serves as the primary
interface for interacting with the AI agent. It includes security checks to
ensure only authorized users can communicate with the bot, and integrates with
the ExecutiveAgent (Prefrontal Cortex) for intelligent message processing.
"""

import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.core.executive import ExecutiveAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/telegram_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration from environment
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ADMIN_USER_ID = int(os.getenv('ADMIN_USER_ID', 0))

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")

if ADMIN_USER_ID == 0:
    raise ValueError("ADMIN_USER_ID environment variable not set")

# Initialize the ExecutiveAgent (will be instantiated in main())
executive_agent: ExecutiveAgent = None


def is_authorized(user_id: int) -> bool:
    """
    Check if the given user_id is authorized to interact with the bot.

    Args:
        user_id: The Telegram user ID to check.

    Returns:
        bool: True if the user is authorized, False otherwise.
    """
    return user_id == ADMIN_USER_ID


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /start command.

    Args:
        update: The update from Telegram.
        context: The context for the command.
    """
    user_id = update.effective_user.id

    if not is_authorized(user_id):
        logger.warning(f"Unauthorized start command attempt from user {user_id}")
        await update.message.reply_text("Unauthorized.")
        return

    logger.info(f"Start command received from authorized user {user_id}")
    await update.message.reply_text("System online. Awaiting input.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle text messages and process them through the ExecutiveAgent.

    Routes messages to the cognitive router for intelligent processing
    based on memory context and message complexity.

    Args:
        update: The update from Telegram.
        context: The context for the message.
    """
    global executive_agent

    user_id = update.effective_user.id

    if not is_authorized(user_id):
        logger.warning(f"Unauthorized message from user {user_id}: {update.message.text}")
        await update.message.reply_text("Unauthorized.")
        return

    user_message = update.message.text
    logger.info(f"Message received from user {user_id}: {user_message}")

    try:
        # Send a typing indicator to show the bot is processing
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )

        # Process the message through the ExecutiveAgent
        if executive_agent is None:
            logger.error("Executive Agent not initialized")
            await update.message.reply_text("System error: Service temporarily unavailable.")
            return

        # Process message (may call external APIs - handle timeouts)
        ai_response = await executive_agent.process_message(user_message)

        # Send the response back to the user
        await update.message.reply_text(ai_response)
        logger.info(f"Response sent to user {user_id}")

    except TimeoutError as e:
        timeout_msg = "Request timed out. Please try again."
        logger.warning(f"Timeout error: {e}")
        await update.message.reply_text(timeout_msg)

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        await update.message.reply_text("An error occurred while processing your message. Please try again.")



def main() -> None:
    """
    Initialize and run the Telegram bot with the ExecutiveAgent.
    """
    global executive_agent

    try:
        # Initialize the ExecutiveAgent (Prefrontal Cortex)
        logger.info("Initializing Executive Agent...")
        executive_agent = ExecutiveAgent()
        logger.info("Executive Agent initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Executive Agent: {e}", exc_info=True)
        raise

    try:
        # Create the Application
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        # Register handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        logger.info("Telegram bot initialized and starting polling...")

        # Run the bot
        application.run_polling()

    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception as e:
        logger.error(f"Bot error: {e}", exc_info=True)
    finally:
        # Cleanup
        if executive_agent:
            logger.info("Shutting down ExecutiveAgent...")
            executive_agent.close()



if __name__ == "__main__":
    main()
