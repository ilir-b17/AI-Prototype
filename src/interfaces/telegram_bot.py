"""
Telegram Bot Interface for Autonomous Biomimetic AI Agent.

This module implements an asynchronous Telegram bot that serves as the primary
interface for interacting with the AI agent. It includes security checks to
ensure only authorized users can communicate with the bot, and integrates with
the ExecutiveAgent (Prefrontal Cortex) for intelligent message processing.
"""

import os
import signal
import logging
import asyncio
import sys
import socket
from io import BytesIO
from dotenv import load_dotenv

# Load .env BEFORE importing any project modules so that module-level os.getenv()
# calls in llm_router, orchestrator, etc. see the correct values.  override=True
# ensures the file always wins over stale process-level env vars.
load_dotenv(override=True)

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import Conflict
from src.core.orchestrator import Orchestrator

# Ensure stdout/stderr use UTF-8 on Windows so Unicode log characters don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/telegram_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Suppress ChromaDB telemetry errors (they don't affect functionality)
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
# Silence noisy HTTP request logs from the Telegram client stack (httpx)
logging.getLogger('httpx').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Load configuration from environment
_UNAUTHORIZED_MSG = "Unauthorized."
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ADMIN_USER_ID = int(os.getenv('ADMIN_USER_ID', 0))
AGENT_TIMEOUT_SECONDS = float(os.getenv('AGENT_TIMEOUT_SECONDS', 300.0))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {'1', 'true', 'yes', 'on'}


ENABLE_NATIVE_AUDIO = _env_bool('ENABLE_NATIVE_AUDIO', default=False)

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")

if ADMIN_USER_ID == 0:
    raise ValueError("ADMIN_USER_ID environment variable not set")

# Initialize the Orchestrator (will be instantiated in main())
orchestrator: Orchestrator = None

# Reference to the running Application (set in main())
_application: Application = None

# Set to True before signalling shutdown when a restart is wanted
_restart_requested: bool = False

# Track consecutive conflicts for backoff mechanism
consecutive_conflicts = 0


def sanitize_for_logging(text: str, max_length: int = 256) -> str:
    """
    Sanitize text for safe logging by escaping control characters
    and truncating excessively long strings to prevent log injection
    and DoS via log bloating.
    """
    if not text:
        return ""

    # Use repr() to escape special characters, but strip the enclosing quotes
    sanitized = repr(text)[1:-1]

    if len(sanitized) > max_length:
        return sanitized[:max_length] + "...[truncated]"
    return sanitized


async def _keep_typing_alive(bot, chat_id: int, stop_event: asyncio.Event) -> None:
    """
    Background task: re-sends the 'typing' action every 4 s until stop_event is
    set. Telegram's typing indicator auto-expires after ~5 s, so without this
    the user sees no activity for long-running requests.
    """
    while not stop_event.is_set():
        try:
            await bot.send_chat_action(chat_id=chat_id, action="typing")
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass


async def _drain_outbound_queue(bot, queue: asyncio.Queue) -> None:
    """
    Background task: drains the orchestrator's outbound queue and sends
    messages to the admin via Telegram. Used by the Proactive Heartbeat.
    """
    while True:
        try:
            message = await queue.get()
            await bot.send_message(chat_id=ADMIN_USER_ID, text=message[:4096])  # Telegram max length
            queue.task_done()
        except Exception as e:
            logger.error(f"Failed to send admin notification: {e}", exc_info=True)


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
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    logger.info(f"Start command received from authorized user {user_id}")
    await update.message.reply_text("System online. Awaiting input.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle text messages and process them through the Orchestrator.

    Routes messages to the state graph for intelligent processing
    based on memory context and message complexity.

    Args:
        update: The update from Telegram.
        context: The context for the message.
    """
    global orchestrator

    user_id = update.effective_user.id

    if not is_authorized(user_id):
        sanitized_msg = sanitize_for_logging(update.message.text)
        logger.warning(f"Unauthorized message from user {user_id}: {sanitized_msg}")
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    user_message = update.message.text
    sanitized_user_msg = sanitize_for_logging(user_message)
    logger.info(f"Message received from user {user_id}: {sanitized_user_msg}")

    try:
        if orchestrator is None:
            logger.error("Orchestrator not initialized")
            await update.message.reply_text("System error: Service temporarily unavailable.")
            return

        # Keep sending 'typing' every 4 s so the indicator stays visible throughout
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            _keep_typing_alive(context.bot, update.effective_chat.id, stop_typing)
        )

        try:
            # Process message (may call external APIs - handle timeouts)
            ai_response = await asyncio.wait_for(
                orchestrator.process_message(user_message, str(user_id)),
                timeout=AGENT_TIMEOUT_SECONDS
            )
        finally:
            stop_typing.set()
            await typing_task

        await update.message.reply_text(ai_response)
        logger.info(f"Response sent to user {user_id}")

    except (TimeoutError, asyncio.TimeoutError) as e:
        timeout_msg = "Request timed out. Please try again."
        logger.warning(f"Timeout error after {AGENT_TIMEOUT_SECONDS}s: {e}")
        await update.message.reply_text(timeout_msg)

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        await update.message.reply_text("An error occurred while processing your message. Please try again.")


async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Telegram voice notes using in-memory buffering only."""
    global orchestrator

    user_id = update.effective_user.id
    voice = update.message.voice

    if not is_authorized(user_id):
        logger.warning(f"Unauthorized voice message from user {user_id}")
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    if voice is None:
        await update.message.reply_text("Unsupported voice payload.")
        return

    if not ENABLE_NATIVE_AUDIO:
        logger.info("Voice message rejected for user %s: ENABLE_NATIVE_AUDIO is disabled.", user_id)
        await update.message.reply_text(
            "Native audio input is disabled. Set ENABLE_NATIVE_AUDIO=true to enable voice notes."
        )
        return

    logger.info(
        "Voice note received from user %s: duration=%ss size=%s bytes",
        user_id,
        voice.duration,
        voice.file_size,
    )

    try:
        if orchestrator is None:
            logger.error("Orchestrator not initialized")
            await update.message.reply_text("System error: Service temporarily unavailable.")
            return

        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            _keep_typing_alive(context.bot, update.effective_chat.id, stop_typing)
        )

        try:
            telegram_file = await context.bot.get_file(voice.file_id)
            buffer = BytesIO()
            await telegram_file.download_to_memory(out=buffer)
            audio_bytes = buffer.getvalue()
            if not audio_bytes:
                await update.message.reply_text("I could not read that voice note. Please try again.")
                return

            prompt_payload = {
                "text": str(update.message.caption or "").strip(),
                "audio_bytes": audio_bytes,
                "audio_mime_type": "audio/ogg",
                "audio_source": "telegram_voice",
                "audio_file_id": voice.file_id,
            }

            ai_response = await asyncio.wait_for(
                orchestrator.process_message(prompt_payload, str(user_id)),
                timeout=AGENT_TIMEOUT_SECONDS,
            )
        finally:
            stop_typing.set()
            await typing_task

        await update.message.reply_text(ai_response)
        logger.info("Voice response sent to user %s", user_id)

    except (TimeoutError, asyncio.TimeoutError) as e:
        timeout_msg = "Request timed out. Please try again."
        logger.warning(f"Voice timeout error after {AGENT_TIMEOUT_SECONDS}s: {e}")
        await update.message.reply_text(timeout_msg)

    except Exception as e:
        logger.error(f"Error processing voice message: {e}", exc_info=True)
        await update.message.reply_text("An error occurred while processing your voice note. Please try again.")


async def goals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/goals — list all active objectives grouped by tier."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    try:
        # Re-use the orchestrator's existing LedgerMemory connection (ISSUE-003).
        # Opening a second aiosqlite connection to the same file can cause
        # "database is locked" errors under concurrent access.
        ledger = context.bot_data.get("ledger")
        if ledger is None:
            await update.message.reply_text("System error: Ledger not available.")
            return
        items = await ledger.get_all_active_goals()

        if not items:
            await update.message.reply_text("Backlog is empty. Use /addgoal to add objectives.")
            return

        lines = ["--- Objective Backlog ---"]
        current_tier = None
        for obj in items:
            if obj["tier"] != current_tier:
                current_tier = obj["tier"]
                lines.append(f"\n[{current_tier}s]")
            status_tag = f"[{obj['status'].upper()}]"
            lines.append(f"  #{obj['id']} {status_tag} {obj['title']} (E:{obj['estimated_energy']})")

        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        logger.error(f"/goals error: {e}", exc_info=True)
        await update.message.reply_text(f"Error fetching goals: {e}")


async def addgoal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/addgoal <Tier> <Energy> <Title> â€" inject a goal into the backlog."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    args = context.args
    if not args or len(args) < 3:
        await update.message.reply_text(
            "Usage: /addgoal <Tier> <Energy> <Title>\n"
            "Example: /addgoal Task 20 Write a weather python script\n"
            "Tiers: Epic, Story, Task"
        )
        return

    tier = args[0].capitalize()
    if tier not in ("Epic", "Story", "Task"):
        await update.message.reply_text("Tier must be: Epic, Story, or Task")
        return

    try:
        energy = int(args[1])
    except ValueError:
        await update.message.reply_text("Energy must be an integer (e.g. 20)")
        return

    title = " ".join(args[2:])

    try:
        # Re-use the orchestrator's existing LedgerMemory connection (ISSUE-003).
        ledger = context.bot_data.get("ledger")
        if ledger is None:
            await update.message.reply_text("System error: Ledger not available.")
            return
        obj_id = await ledger.add_objective(tier=tier, title=title, estimated_energy=energy, origin="Admin")
        decomposition_note = ""
        if tier == "Epic":
            orchestrator_inst = context.bot_data.get("orchestrator")
            if (
                orchestrator_inst is not None
                and getattr(orchestrator_inst, "cognitive_router", None) is not None
                and orchestrator_inst.cognitive_router.get_system_2_available()
            ):
                async def _route(messages):
                    return await orchestrator_inst._route_to_system_2_redacted(
                        messages,
                        allowed_tools=[],
                        purpose="goal_planner_epic",
                        allow_sensitive_context=False,
                    )

                try:
                    plan_result = await orchestrator_inst.goal_planner.plan_goal(
                        title,
                        context="Admin-defined Epic from /addgoal.",
                        route_to_system_2=_route,
                        ledger_memory=ledger,
                        redactor=orchestrator_inst._redact_text_for_cloud,
                        origin="Admin",
                        parent_epic_id=obj_id,
                    )
                    decomposition_note = (
                        f"\nPlanned decomposition only: {plan_result.story_count} Stories, "
                        f"{plan_result.task_count} Tasks."
                    )
                except Exception as plan_err:
                    logger.warning(f"GoalPlanner failed for Epic #{obj_id}: {plan_err}")
                    decomposition_note = "\nEpic added, but automatic decomposition failed."
            else:
                decomposition_note = "\nEpic added, but System 2 is unavailable so decomposition was skipped."

        await update.message.reply_text(
            f"Added to backlog:\n  #{obj_id} [{tier}] {title}\n  Estimated Energy: {energy}{decomposition_note}"
        )
        sanitized_title = sanitize_for_logging(title)
        logger.info(f"Admin added objective #{obj_id}: [{tier}] {sanitized_title}")
    except Exception as e:
        logger.error(f"/addgoal error: {e}", exc_info=True)
        await update.message.reply_text(f"Error adding goal: {e}")


async def shutdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/shutdown — gracefully stop the bot (admin only)."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    logger.info("Shutdown command received from admin.")
    await update.message.reply_text("AIDEN shutting down. Goodbye! \u2705")
    await asyncio.sleep(1)
    # Cancel the main asyncio task so _async_run's finally block runs cleanly
    for task in asyncio.all_tasks():
        if task.get_name() == "main_bot_task":
            task.cancel()
            return
    # Fallback if task name not found
    os.kill(os.getpid(), signal.SIGINT)


async def restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/restart — gracefully restart the bot process (admin only)."""
    global _restart_requested

    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    logger.info("Restart command received from admin.")
    await update.message.reply_text("AIDEN restarting \u2b6f\ufe0f  Back in a moment...")
    await asyncio.sleep(1)
    _restart_requested = True
    for task in asyncio.all_tasks():
        if task.get_name() == "main_bot_task":
            task.cancel()
            return
    os.kill(os.getpid(), signal.SIGINT)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle errors from the Telegram API and application.
    Implements exponential backoff for Conflict (409) errors (ISSUE-014).

    The handler IS async, so the sleep is awaited directly — this is
    intentional: it applies real back-pressure before the next polling
    cycle rather than only logging a "waiting" message that is never acted on.
    """
    global consecutive_conflicts

    error = context.error

    # Log all errors
    logger.error(f"Telegram API error: {error}", exc_info=True)

    # Special handling for Conflict errors - these are recoverable
    if isinstance(error, Conflict):
        consecutive_conflicts += 1

        # Exponential backoff: cap at 60 seconds
        wait_time = min(2 ** consecutive_conflicts, 60)
        logger.warning(
            f"Polling conflict #{consecutive_conflicts} detected. "
            f"Sleeping {wait_time}s before next poll cycle..."
        )
        # Actually enforce the backoff — this is safe to await inside an
        # async error handler (ISSUE-014).
        await asyncio.sleep(wait_time)
        return

    # Reset conflict counter on other errors (indicates we recovered)
    consecutive_conflicts = 0

    # For other errors, try to notify the user if we have an update
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                f"An error occurred: {str(error)[:100]}"
            )
        except Exception as reply_error:
            logger.error(f"Failed to send error message: {reply_error}", exc_info=True)


async def _async_run(application: Application, orchestrator_inst: "Orchestrator") -> None:
    """
    Runs the bot inside a single asyncio.run() call we control.
    Exits cleanly on Ctrl+C (KeyboardInterrupt → CancelledError) or SIGTERM.
    All async cleanup is done here with a hard 5-second timeout so the process
    never hangs on shutdown.
    """
    _bg_tasks: list = []

    async with application:
        # Clear stale webhook
        try:
            await application.bot.delete_webhook(drop_pending_updates=True)
            logger.info("Cleared stale Telegram webhook/session before polling.")
        except Exception as e:
            logger.warning(f"delete_webhook failed (non-fatal): {e}")

        # Async-init the orchestrator
        await orchestrator_inst.async_init()
        ready_event = getattr(orchestrator_inst, "_ready", None)
        if ready_event is not None and not ready_event.is_set():
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=30.0)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Orchestrator did not become ready during startup.") from exc
        outbound_queue: asyncio.Queue = asyncio.Queue()
        orchestrator_inst.outbound_queue = outbound_queue

        # Expose the orchestrator's existing LedgerMemory connection so that
        # command handlers (/goals, /addgoal) can re-use it instead of opening
        # a second parallel connection to the same DB (ISSUE-003).
        application.bot_data["ledger"] = orchestrator_inst.ledger_memory
        application.bot_data["orchestrator"] = orchestrator_inst

        # Re-send any HITL reminders that were queued before the outbound pipe
        # existed (i.e. restored during async_init).  Now that the queue is live
        # the admin will actually receive them in Telegram.
        for _pending_state in orchestrator_inst.pending_hitl_state.values():
            _question = _pending_state.get(
                "_hitl_question",
                "A task is awaiting your guidance. Please reply to continue.",
            )
            await outbound_queue.put(
                f"\u26a0\ufe0f AIDEN restarted — pending task restored:\n\n{_question}"
            )

        _heartbeat_task = asyncio.create_task(orchestrator_inst._heartbeat_loop())
        _sensory_task   = asyncio.create_task(orchestrator_inst._sensory_update_loop())
        _drain_task     = asyncio.create_task(_drain_outbound_queue(application.bot, outbound_queue))
        _memory_task    = asyncio.create_task(orchestrator_inst._memory_consolidation_loop())
        _bg_tasks.extend([_heartbeat_task, _sensory_task, _drain_task, _memory_task])
        logger.info(
            "Orchestrator async_init, heartbeat loop, sensory update loop, memory consolidation loop, "
            "and outbound queue drainer started. "
            f"Tasks: {_heartbeat_task.get_name()}, {_sensory_task.get_name()}, {_memory_task.get_name()}, {_drain_task.get_name()}"
        )

        # Start polling and the PTB application
        await application.updater.start_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
            poll_interval=1.0,
        )
        await application.start()
        logger.info("Telegram bot initialized and polling is active.")
        logger.info(
            "Waiting for Telegram updates. No further terminal output is expected until a message arrives "
            "or shutdown begins."
        )

        try:
            # Block here until cancelled by Ctrl+C or SIGTERM
            await asyncio.Event().wait()
        finally:
            logger.info("Shutting down — stopping polling and cleaning up...")

            # Stop PTB gracefully (best-effort, 5s cap)
            try:
                await asyncio.wait_for(application.updater.stop(), timeout=5.0)
                await asyncio.wait_for(application.stop(), timeout=5.0)
            except Exception:
                pass

            # Cancel background tasks with a hard 5s timeout
            for task in _bg_tasks:
                task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*_bg_tasks, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Background tasks did not cancel in time — forcing exit.")

            # Close async resources
            for coro in (
                orchestrator_inst.cognitive_router.close(),
                orchestrator_inst.ledger_memory.close(),
            ):
                try:
                    await asyncio.wait_for(coro, timeout=3.0)
                except Exception:
                    pass

            # Close synchronous resources
            try:
                orchestrator_inst.vector_memory.close()
            except Exception:
                pass

            logger.info("Shutdown complete.")


def main() -> None:
    """
    Initialize and run the Telegram bot with the Orchestrator.
    """
    global orchestrator, _application, _restart_requested
    lock_sock = None

    try:
        # Initialize the Orchestrator (State Graph)
        logger.info("Initializing Orchestrator...")
        orchestrator = Orchestrator()
        logger.info("Orchestrator initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Orchestrator: {e}", exc_info=True)
        raise

    try:
        # Acquire a simple single-instance lock using a TCP port bind.
        lock_port = int(os.getenv("TELEGRAM_LOCK_PORT", "55678"))
        try:
            lock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # SO_EXCLUSIVEADDRUSE (Windows) prevents other processes from binding
            # the same port even with SO_REUSEADDR set, fixing the lock on Windows.
            # Fall back to SO_REUSEADDR on non-Windows (Linux TIME_WAIT behaviour).
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                lock_sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                lock_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            lock_sock.bind(("127.0.0.1", lock_port))
            lock_sock.listen(1)
            logger.info(f"Acquired single-instance lock on port {lock_port}")
        except OSError:
            logger.error(f"Another bot instance appears to be running (lock port {lock_port} in use). Exiting.")
            return

        # Create the Application and expose it at module level for shutdown/restart
        application = (
            Application.builder()
            .token(TELEGRAM_BOT_TOKEN)
            .get_updates_read_timeout(15)
            .get_updates_write_timeout(15)
            .get_updates_connect_timeout(10)
            .build()
        )
        _application = application

        # Register handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("goals", goals))
        application.add_handler(CommandHandler("addgoal", addgoal))
        application.add_handler(CommandHandler("shutdown", shutdown))
        application.add_handler(CommandHandler("restart", restart))
        application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_error_handler(error_handler)

        async def _named_run():
            task = asyncio.current_task()
            if task:
                task.set_name("main_bot_task")
            await _async_run(application, orchestrator)

        try:
            asyncio.run(_named_run())
        except asyncio.CancelledError:
            logger.info("Bot shutdown requested")
        except KeyboardInterrupt:
            logger.info("Bot interrupted by user")

    except Exception as e:
        logger.error(f"Bot error: {e}", exc_info=True)
    finally:
        try:
            if lock_sock:
                lock_sock.close()
                logger.info("Released single-instance lock")
        except Exception:
            pass

    # If a restart was requested, replace the current process with a fresh one.
    # os.execv replaces the process image in-place â€" the OS PID stays the same
    # and the lock socket is already closed above, so the new instance can
    # acquire it cleanly.
    if _restart_requested:
        logger.info("Restarting process...")
        # Flush log handlers so nothing is lost
        for handler in logging.getLogger().handlers:
            handler.flush()
        os.execv(sys.executable, [sys.executable] + sys.argv)



if __name__ == "__main__":
    main()

