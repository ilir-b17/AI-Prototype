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
import logging.handlers
import asyncio
import sys
import socket
import time
from io import BytesIO
from dotenv import load_dotenv

# Load .env BEFORE importing any project modules so that module-level os.getenv()
# calls in llm_router, orchestrator, etc. see the correct values.  override=True
# ensures the file always wins over stale process-level env vars.
load_dotenv(override=True)

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import Conflict, RetryAfter
from src.core.orchestrator import Orchestrator
from src.interfaces.streaming import DeferredStatusMessage, StatusFormatter
from src.core.progress import ProgressEvent

# Ensure stdout/stderr use UTF-8 on Windows so Unicode log characters don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _resolve_outbound_queue_max_size() -> int:
    return max(1, _parse_int_env("OUTBOUND_QUEUE_MAX_SIZE", 200))


LOG_MAX_BYTES = _parse_int_env("LOG_MAX_BYTES", 10 * 1024 * 1024)
LOG_BACKUP_COUNT = _parse_int_env("LOG_BACKUP_COUNT", 5)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            'logs/telegram_bot.log',
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8',
            delay=True,
        ),
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
_ORCHESTRATOR_NOT_INITIALIZED = "Orchestrator not initialized"
_ORCHESTRATOR_NOT_AVAILABLE_MSG = "System error: Orchestrator not available."
_LEDGER_NOT_AVAILABLE_MSG = "System error: Ledger not available."
_SERVICE_UNAVAILABLE_MSG = "System error: Service temporarily unavailable."
_TIMEOUT_MSG = "Request timed out. Please try again."
_PENDING_RETIRE_UNUSED_KEY = "pending_retire_unused_tools"
_RETIRE_CONFIRM_ALIASES = {"confirm", "yes", "y"}
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ADMIN_USER_ID = int(os.getenv('ADMIN_USER_ID', 0))
AGENT_TIMEOUT_SECONDS = float(os.getenv('AGENT_TIMEOUT_SECONDS', 300.0))

# Minimum delay before showing status message (seconds).
# Responses arriving before this delay show no status message at all.
# This prevents fast-path responses from flashing a status UI.
_STREAMING_DELAY_SECONDS = float(os.getenv("STREAMING_DELAY_SECONDS", "2.5"))

# Minimum token count before streaming is enabled.
# Very short messages get fast-path routing so streaming is never needed.
_STREAMING_MIN_TOKENS = int(os.getenv("STREAMING_MIN_TOKENS", "4"))


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
        message = await queue.get()
        attempts = 0
        while attempts < 3:
            try:
                await _send_admin_outbound_payload(bot, message)
                await asyncio.sleep(0.1)   # rate-limit guard
                break
            except RetryAfter as e:
                wait = getattr(e, "retry_after", 30) + 1
                logger.warning(
                    "Telegram rate limit hit; retrying in %ss", wait
                )
                await asyncio.sleep(wait)
                attempts += 1
            except Exception as e:
                logger.error(
                    "Failed to send admin notification (attempt %s/3): %s",
                    attempts + 1, e, exc_info=True,
                )
                attempts += 1
        else:
            logger.error(
                "Dropped admin notification after 3 failed attempts: %s",
                str(message)[:120],
            )
        queue.task_done()


def _enqueue_outbound_with_drop_oldest(queue: asyncio.Queue, payload, *, source: str) -> bool:
    try:
        queue.put_nowait(payload)
        return True
    except asyncio.QueueFull:
        dropped_payload = None
        try:
            dropped_payload = queue.get_nowait()
            try:
                queue.task_done()
            except ValueError:
                pass
        except asyncio.QueueEmpty:
            pass

        logger.warning(
            "Dropped oldest outbound payload at %s because queue is full (maxsize=%s, dropped_type=%s).",
            source,
            getattr(queue, "maxsize", "unknown"),
            type(dropped_payload).__name__ if dropped_payload is not None else "none",
        )

        try:
            queue.put_nowait(payload)
            return True
        except asyncio.QueueFull:
            logger.error(
                "Could not enqueue outbound payload at %s after dropping oldest.",
                source,
            )
            return False


async def _drain_background_tasks(orchestrator_inst: "Orchestrator", timeout: float = 5.0) -> None:
    """Best-effort drain of orchestrator fire-and-forget tasks before resource shutdown."""
    background_tasks = list(getattr(orchestrator_inst, "_background_tasks", set()) or [])
    if not background_tasks:
        return

    gather_future = asyncio.gather(*background_tasks, return_exceptions=True)
    try:
        await asyncio.wait_for(asyncio.shield(gather_future), timeout=timeout)
    except asyncio.TimeoutError:
        undrained = sum(1 for task in background_tasks if not task.done())
        logger.warning(
            "Orchestrator background task drain timed out after %.1fs; %d background task(s) did not drain in time.",
            timeout,
            undrained,
        )


async def _send_admin_outbound_payload(bot, payload) -> None:
    if not isinstance(payload, dict):
        await bot.send_message(
            chat_id=ADMIN_USER_ID,
            text=str(payload)[:4096],
        )
        return

    text = str(payload.get("text") or "").strip()
    if text:
        await bot.send_message(chat_id=ADMIN_USER_ID, text=text[:4096])

    document_path = str(payload.get("document_path") or "").strip()
    if not document_path:
        return

    document_filename = str(payload.get("document_filename") or os.path.basename(document_path))
    document_bytes = await asyncio.to_thread(_read_binary_file, document_path)
    await bot.send_document(
        chat_id=ADMIN_USER_ID,
        document=BytesIO(document_bytes),
        filename=document_filename,
    )

    if payload.get("delete_after_send"):
        try:
            os.remove(document_path)
        except OSError as remove_error:
            logger.warning("Failed to remove temporary outbound document %s: %s", document_path, remove_error)


def _read_binary_file(path: str) -> bytes:
    with open(path, "rb") as document_file:
        return document_file.read()


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


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/status — show operational health for the admin."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    orchestrator_inst = context.bot_data.get("orchestrator") or orchestrator
    if orchestrator_inst is None:
        await update.message.reply_text(_ORCHESTRATOR_NOT_AVAILABLE_MSG)
        return

    router = getattr(orchestrator_inst, "cognitive_router", None)
    registry = getattr(router, "registry", None)
    try:
        loaded_count = len(registry) if registry is not None else 0
    except Exception:
        loaded_count = 0

    load_errors_getter = getattr(registry, "get_load_errors", None)
    if callable(load_errors_getter):
        failed_skills = list(load_errors_getter())
    else:
        failed_skills = list(getattr(registry, "_load_errors", []) or [])
    failed_names = [str(item[0]) for item in failed_skills]
    failed_display = ", ".join(failed_names[:12]) if failed_names else "none"
    if len(failed_names) > 12:
        failed_display += f", and {len(failed_names) - 12} more"

    try:
        predictive_budget = await orchestrator_inst._get_predictive_energy_budget_remaining()
    except Exception as budget_error:
        logger.warning("/status could not read predictive energy budget: %s", budget_error)
        predictive_budget = "unknown"

    lines = [
        "AIDEN status",
        f"Loaded skills: {loaded_count}",
        f"Failed skills: {len(failed_skills)} ({failed_display})",
        f"Pending MFA: {len(getattr(orchestrator_inst, 'pending_mfa', {}) or {})}",
        f"Pending HITL: {len(getattr(orchestrator_inst, 'pending_hitl_state', {}) or {})}",
        f"Predictive energy budget: {predictive_budget}",
    ]
    await update.message.reply_text("\n".join(lines))


def _retire_confirmation_requested(args) -> bool:
    if not args:
        return False
    return str(args[0] or "").strip().lower() in _RETIRE_CONFIRM_ALIASES


def _ensure_bot_data_dict(context: ContextTypes.DEFAULT_TYPE) -> dict:
    bot_data = getattr(context, "bot_data", None)
    if isinstance(bot_data, dict):
        return bot_data
    bot_data = {}
    context.bot_data = bot_data
    return bot_data


def _normalize_tool_name_list(raw_values) -> list:
    return [
        str(name).strip()
        for name in (raw_values or [])
        if str(name).strip()
    ]


def _format_retire_candidates_message(candidates) -> str:
    lines = ["Unused approved tools (30+ days):"]
    for item in candidates:
        name = str(item.get("name") or "").strip()
        last_used = str(item.get("last_used_at") or "never")
        lines.append(f"- {name} (last_used_at={last_used})")
    lines.append("")
    lines.append("Run /retire_unused_tools confirm to retire these tools.")
    return "\n".join(lines)


async def _confirm_retire_unused_tools(update: Update, ledger, pending: dict, pending_key: str) -> None:
    pending_payload = dict(pending.get(pending_key) or {})
    tool_names = _normalize_tool_name_list(pending_payload.get("tool_names"))
    if not tool_names:
        await update.message.reply_text(
            "No pending retirement list found. Run /retire_unused_tools first."
        )
        return

    retired_count = await ledger.retire_tools(tool_names)
    pending.pop(pending_key, None)
    await update.message.reply_text(
        f"Retired {retired_count} tool(s): {', '.join(tool_names)}"
    )


async def _start_retire_unused_tools_flow(update: Update, ledger, pending: dict, pending_key: str) -> None:
    candidates = await ledger.get_unused_approved_tools(days=30)
    if not candidates:
        pending.pop(pending_key, None)
        await update.message.reply_text("No approved tools are currently unused for 30+ days.")
        return

    tool_names = _normalize_tool_name_list(item.get("name") for item in candidates)
    pending[pending_key] = {
        "tool_names": tool_names,
        "created_at": time.time(),
    }
    await update.message.reply_text(_format_retire_candidates_message(candidates))


async def retire_unused_tools(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/retire_unused_tools [confirm] — retire approved dynamic tools unused for 30+ days."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    bot_data = _ensure_bot_data_dict(context)

    orchestrator_inst = bot_data.get("orchestrator") or orchestrator
    if orchestrator_inst is None:
        await update.message.reply_text(_ORCHESTRATOR_NOT_AVAILABLE_MSG)
        return

    ledger = bot_data.get("ledger") or getattr(orchestrator_inst, "ledger_memory", None)
    if ledger is None:
        await update.message.reply_text(_LEDGER_NOT_AVAILABLE_MSG)
        return

    pending = bot_data.setdefault(_PENDING_RETIRE_UNUSED_KEY, {})
    pending_key = str(update.effective_user.id)

    if _retire_confirmation_requested(getattr(context, "args", [])):
        await _confirm_retire_unused_tools(update, ledger, pending, pending_key)
        return

    await _start_retire_unused_tools_flow(update, ledger, pending, pending_key)


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
            logger.error(_ORCHESTRATOR_NOT_INITIALIZED)
            await update.message.reply_text(_SERVICE_UNAVAILABLE_MSG)
            return

        # Short messages that will be handled by fast-path routing
        # do not need streaming — skip the DeferredStatusMessage overhead.
        token_count = len(user_message.split())
        use_streaming = token_count >= _STREAMING_MIN_TOKENS

        if not use_streaming:
            # Original non-streaming path for very short messages
            stop_typing = asyncio.Event()
            typing_task = asyncio.create_task(
                _keep_typing_alive(context.bot, update.effective_chat.id, stop_typing)
            )
            try:
                ai_response = await asyncio.wait_for(
                    orchestrator.process_message(user_message, str(user_id)),
                    timeout=AGENT_TIMEOUT_SECONDS,
                )
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning(
                    "Timeout after %.0fs for user %s", AGENT_TIMEOUT_SECONDS, user_id
                )
                await update.message.reply_text(_TIMEOUT_MSG)
                return
            except Exception as e:
                logger.error("Error processing message: %s", e, exc_info=True)
                await update.message.reply_text(
                    "An error occurred while processing your message. Please try again."
                )
                return
            finally:
                stop_typing.set()
                await typing_task
            await update.message.reply_text(ai_response)
            logger.info(f"Response sent to user {user_id}")
            return

        # Streaming path — DeferredStatusMessage sends status only if slow
        formatter = StatusFormatter()
        dsm = DeferredStatusMessage(
            update.message,
            delay_seconds=_STREAMING_DELAY_SECONDS,
        )
        await dsm.start()

        # Keep typing indicator running until status message appears.
        # The DeferredStatusMessage replaces the typing indicator once sent.
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            _keep_typing_alive(context.bot, update.effective_chat.id, stop_typing)
        )

        async def on_progress(event: ProgressEvent) -> None:
            # Stop typing indicator on first progress event — the status
            # message is about to appear (or has already appeared).
            if not stop_typing.is_set():
                stop_typing.set()
            formatter.apply(event)
            await dsm.update(formatter.render())

        try:
            ai_response = await asyncio.wait_for(
                orchestrator.process_message(
                    user_message,
                    str(user_id),
                    progress_callback=on_progress,
                ),
                timeout=AGENT_TIMEOUT_SECONDS,
            )
        except (TimeoutError, asyncio.TimeoutError):
            stop_typing.set()
            await typing_task
            logger.warning(
                "Timeout after %.0fs for user %s", AGENT_TIMEOUT_SECONDS, user_id
            )
            delivered = await dsm.finalize(_TIMEOUT_MSG)
            if not delivered:
                await update.message.reply_text(_TIMEOUT_MSG)
            return
        except Exception as e:
            stop_typing.set()
            await typing_task
            logger.error("Error processing message: %s", e, exc_info=True)
            error_msg = (
                "An error occurred while processing your message. Please try again."
            )
            delivered = await dsm.finalize(error_msg)
            if not delivered:
                await update.message.reply_text(error_msg)
            return
        finally:
            # Ensure typing indicator is stopped in all paths
            if not stop_typing.is_set():
                stop_typing.set()
            # Await typing task to prevent ResourceWarning
            try:
                await asyncio.wait_for(typing_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Deliver the final response
        delivered = await dsm.finalize(ai_response)
        if not delivered:
            await update.message.reply_text(ai_response[:4096])
        logger.info("Response delivered to user %s", user_id)

    except Exception as e:
        logger.error("Unhandled error in handle_message: %s", e, exc_info=True)
        try:
            await update.message.reply_text(
                "An error occurred while processing your message. Please try again."
            )
        except Exception:
            pass


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
            logger.error(_ORCHESTRATOR_NOT_INITIALIZED)
            await update.message.reply_text(_SERVICE_UNAVAILABLE_MSG)
            return

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

        formatter = StatusFormatter()
        dsm = DeferredStatusMessage(
            update.message,
            delay_seconds=_STREAMING_DELAY_SECONDS,
            initial_text="⏳ AIDEN — Processing voice note...",
        )
        await dsm.start()

        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            _keep_typing_alive(context.bot, update.effective_chat.id, stop_typing)
        )

        async def on_voice_progress(event: ProgressEvent) -> None:
            if not stop_typing.is_set():
                stop_typing.set()
            formatter.apply(event)
            await dsm.update(formatter.render())

        try:
            ai_response = await asyncio.wait_for(
                orchestrator.process_message(
                    prompt_payload,
                    str(user_id),
                    progress_callback=on_voice_progress,
                ),
                timeout=AGENT_TIMEOUT_SECONDS,
            )
        except (TimeoutError, asyncio.TimeoutError) as e:
            stop_typing.set()
            await typing_task
            logger.warning("Voice timeout after %.0fs: %s", AGENT_TIMEOUT_SECONDS, e)
            delivered = await dsm.finalize(_TIMEOUT_MSG)
            if not delivered:
                await update.message.reply_text(_TIMEOUT_MSG)
            return
        except Exception as e:
            stop_typing.set()
            await typing_task
            logger.error("Error processing voice message: %s", e, exc_info=True)
            error_msg = (
                "An error occurred while processing your voice note. Please try again."
            )
            delivered = await dsm.finalize(error_msg)
            if not delivered:
                await update.message.reply_text(error_msg)
            return
        finally:
            if not stop_typing.is_set():
                stop_typing.set()
            try:
                await asyncio.wait_for(typing_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        delivered = await dsm.finalize(ai_response)
        if not delivered:
            await update.message.reply_text(ai_response[:4096])
        logger.info("Voice response delivered to user %s", user_id)

    except Exception as e:
        logger.error(f"Error processing voice message: {e}", exc_info=True)
        await update.message.reply_text("An error occurred while processing your voice note. Please try again.")


async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Telegram document attachments used as tool-approval context."""
    global orchestrator

    user_id = update.effective_user.id
    if not is_authorized(user_id):
        logger.warning("Unauthorized document message from user %s", user_id)
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    if orchestrator is None:
        logger.error(_ORCHESTRATOR_NOT_INITIALIZED)
        await update.message.reply_text(_SERVICE_UNAVAILABLE_MSG)
        return

    pending_approvals = getattr(orchestrator, "pending_tool_approval", {}) or {}
    if str(user_id) not in pending_approvals:
        await update.message.reply_text("No pending tool approval is waiting for an attachment.")
        return

    caption = str(update.message.caption or "").strip()
    if not caption:
        await update.message.reply_text("Please resend the attachment with YES or NO as the caption.")
        return

    document = update.message.document
    file_name = str(getattr(document, "file_name", "attachment") or "attachment")
    file_size = getattr(document, "file_size", "unknown")
    approval_message = f"{caption}\n[Approval attachment: {file_name}; size={file_size}]"

    try:
        ai_response = await asyncio.wait_for(
            orchestrator.process_message(approval_message, str(user_id)),
            timeout=AGENT_TIMEOUT_SECONDS,
        )
        await update.message.reply_text(ai_response)
    except (TimeoutError, asyncio.TimeoutError) as e:
        logger.warning("Document approval timeout after %ss: %s", AGENT_TIMEOUT_SECONDS, e)
        await update.message.reply_text(_TIMEOUT_MSG)
    except Exception as e:
        logger.error("Error processing document approval: %s", e, exc_info=True)
        await update.message.reply_text("An error occurred while processing the approval attachment. Please try again.")


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
            await update.message.reply_text(_LEDGER_NOT_AVAILABLE_MSG)
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
            await update.message.reply_text(_LEDGER_NOT_AVAILABLE_MSG)
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


async def session_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/session [new|list|switch|link|end|status] — manage conversation sessions."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    orchestrator_inst = context.bot_data.get("orchestrator") or orchestrator
    ledger = context.bot_data.get("ledger")

    if orchestrator_inst is None:
        await update.message.reply_text(_ORCHESTRATOR_NOT_AVAILABLE_MSG)
        return
    if ledger is None:
        await update.message.reply_text(_LEDGER_NOT_AVAILABLE_MSG)
        return

    args = getattr(context, "args", []) or []
    subcommand = args[0].lower() if args else "status"

    try:
        if subcommand == "status":
            await _session_status(update, orchestrator_inst)

        elif subcommand == "new":
            name_parts = args[1:]
            if not name_parts:
                await update.message.reply_text(
                    "Usage: /session new <name> [| description]\n"
                    "Example: /session new AIDEN Refactor | Architectural work"
                )
                return
            raw = " ".join(name_parts)
            parts = raw.split("|", 1)
            name = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""
            await _session_new(update, ledger, orchestrator_inst, name, description)

        elif subcommand == "list":
            await _session_list(update, ledger)

        elif subcommand == "switch":
            if len(args) < 2:
                await update.message.reply_text("Usage: /session switch <id>")
                return
            try:
                session_id = int(args[1])
            except ValueError:
                await update.message.reply_text("Session id must be an integer.")
                return
            await _session_switch(update, ledger, orchestrator_inst, session_id)

        elif subcommand == "link":
            if len(args) < 2:
                await update.message.reply_text(
                    "Usage: /session link <epic_id>\n"
                    "Links the active session to an Epic in the backlog."
                )
                return
            try:
                epic_id = int(args[1])
            except ValueError:
                await update.message.reply_text("Epic id must be an integer.")
                return
            await _session_link(update, ledger, orchestrator_inst, epic_id)

        elif subcommand == "end":
            await _session_end(update, orchestrator_inst)

        elif subcommand == "summary":
            await _session_summary(update, ledger, orchestrator_inst)

        else:
            await update.message.reply_text(
                "Unknown subcommand. Available: status, new, list, switch, link, end, summary"
            )

    except Exception as e:
        logger.error("/session error: %s", e, exc_info=True)
        await update.message.reply_text(f"Session command error: {e}")


async def why_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/why [last|moral|energy|health|backlog] - introspect AIDEN's state.

    Subcommands:
      /why          - explain the last supervisor decision
      /why last     - same as /why
      /why moral    - recent moral audit results
      /why energy   - current energy budget and deferred tasks
      /why backlog  - current backlog status summary
      /why health   - system health: errors and synthesis history
    """
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    orchestrator_inst = context.bot_data.get("orchestrator") or orchestrator
    if orchestrator_inst is None:
        await update.message.reply_text(_ORCHESTRATOR_NOT_AVAILABLE_MSG)
        return

    args = getattr(context, "args", []) or []
    subcommand = (args[0].lower() if args else "last").strip()

    user_id = str(update.effective_user.id)

    await update.message.reply_text(
        f"⏳ Fetching introspection data ({subcommand})..."
    )

    try:
        result = await asyncio.wait_for(
            _execute_why_subcommand(orchestrator_inst, subcommand, user_id),
            timeout=float(os.getenv("AGENT_TIMEOUT_SECONDS", "60")),
        )
        await update.message.reply_text(result[:4096])
    except asyncio.TimeoutError:
        await update.message.reply_text(
            "Introspection query timed out. Try again or use a more specific subcommand."
        )
    except Exception as exc:
        logger.error("/why error: %s", exc, exc_info=True)
        await update.message.reply_text(f"Introspection failed: {exc}")


async def memory_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """/memory [query|stats|compare] - debug two-stage memory retrieval.

    Subcommands:
      /memory stats              - ChromaDB collection statistics
      /memory query <text>       - run two-stage retrieval and show scores
      /memory compare <text>     - show cosine-only vs reranked side-by-side
    """
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    orchestrator_inst = context.bot_data.get("orchestrator") or orchestrator
    if orchestrator_inst is None:
        await update.message.reply_text(_ORCHESTRATOR_NOT_AVAILABLE_MSG)
        return

    args = getattr(context, "args", []) or []
    subcommand = (args[0].lower() if args else "stats").strip()
    query_text = " ".join(args[1:]).strip() if len(args) > 1 else ""

    try:
        if subcommand == "stats":
            await _memory_stats(update, orchestrator_inst)

        elif subcommand == "query":
            if not query_text:
                await update.message.reply_text(
                    "Usage: /memory query <your search text>"
                )
                return
            await update.message.reply_text(f"⏳ Searching: {query_text!r}...")
            await _memory_query(update, orchestrator_inst, query_text)

        elif subcommand == "compare":
            if not query_text:
                await update.message.reply_text(
                    "Usage: /memory compare <your search text>"
                )
                return
            await update.message.reply_text(
                f"⏳ Running cosine vs reranked comparison for: {query_text!r}..."
            )
            await _memory_compare(update, orchestrator_inst, query_text)

        else:
            await update.message.reply_text(
                f"Unknown subcommand: {subcommand!r}\n\n"
                "Available: /memory stats, /memory query <text>, "
                "/memory compare <text>"
            )

    except Exception as exc:
        logger.error("/memory error: %s", exc, exc_info=True)
        await update.message.reply_text(f"Memory command error: {exc}")


async def _memory_stats(update: Update, orchestrator_inst) -> None:
    try:
        count = orchestrator_inst.vector_memory.get_memory_count()
        reranker = getattr(orchestrator_inst, "memory_reranker", None)
        reranker_enabled = reranker.enabled if reranker else False
        reranker_config = reranker.config if reranker else None

        lines = [
            "🧠 Archival Memory Stats",
            "",
            f"Total memories: {count}",
            f"Two-stage reranking: {'✅ enabled' if reranker_enabled else '❌ disabled'}",
        ]
        if reranker_config and reranker_enabled:
            lines += [
                f"  Recall multiplier: {reranker_config.recall_multiplier}x",
                f"  Max candidates: {reranker_config.max_candidates}",
                f"  LLM score weight (alpha): {reranker_config.alpha:.0%}",
                f"  Timeout: {reranker_config.timeout_seconds}s",
            ]
        await update.message.reply_text("\n".join(lines))
    except Exception as exc:
        await update.message.reply_text(f"Stats error: {exc}")


async def _memory_query(update: Update, orchestrator_inst, query: str) -> None:
    """Run two-stage retrieval and show ranked results."""
    _ = orchestrator_inst
    try:
        from src.skills.search_archival_memory import search_archival_memory
        import json

        result_json = await asyncio.wait_for(
            search_archival_memory(query, n_results=5),
            timeout=60.0,
        )
        data = json.loads(result_json)

        if data.get("status") != "success":
            await update.message.reply_text(
                f"Search failed: {data.get('message', 'Unknown error')}"
            )
            return

        results = data.get("results", [])
        reranked = data.get("reranked", False)
        candidates = data.get("candidates_retrieved", 0)

        lines = [
            f"🔍 Memory Search: {query!r}",
            "",
            f"Retrieved {candidates} candidates -> {len(results)} results "
            f"({'reranked ✅' if reranked else 'cosine only'})",
            "",
        ]
        for i, r in enumerate(results, 1):
            doc_preview = str(r.get("document", ""))[:120]
            dist = r.get("distance", 0)
            if reranked:
                combined = r.get("combined_score", 0)
                llm = r.get("rerank_score", 0)
                lines.append(
                    f"{i}. [score={combined:.2f} llm={llm:.1f} cos={dist:.2f}]\n"
                    f"   {doc_preview}..."
                )
            else:
                lines.append(f"{i}. [distance={dist:.3f}]\n   {doc_preview}...")

        await update.message.reply_text("\n".join(lines)[:4096])

    except asyncio.TimeoutError:
        await update.message.reply_text("Search timed out.")
    except Exception as exc:
        await update.message.reply_text(f"Query error: {exc}")


async def _memory_compare(update: Update, orchestrator_inst, query: str) -> None:
    """Show cosine-only vs reranked results side by side."""
    _ = orchestrator_inst
    try:
        from src.skills.search_archival_memory import search_archival_memory
        import json

        # Run cosine-only
        cosine_json = await asyncio.wait_for(
            search_archival_memory(query, n_results=5, skip_reranking=True),
            timeout=30.0,
        )
        cosine_data = json.loads(cosine_json)
        cosine_results = cosine_data.get("results", [])

        # Run with reranking
        reranked_json = await asyncio.wait_for(
            search_archival_memory(query, n_results=5, skip_reranking=False),
            timeout=60.0,
        )
        reranked_data = json.loads(reranked_json)
        reranked_results = reranked_data.get("results", [])
        was_reranked = reranked_data.get("reranked", False)

        lines = [
            f"📊 Cosine vs Reranked: {query!r}",
            "",
            "━━ Cosine only ━━",
        ]
        for i, r in enumerate(cosine_results[:3], 1):
            lines.append(
                f"{i}. [{r.get('distance', 0):.3f}] "
                f"{str(r.get('document', ''))[:80]}..."
            )

        lines += ["", f"━━ Reranked ({'✅' if was_reranked else 'same - reranking disabled'}) ━━"]
        for i, r in enumerate(reranked_results[:3], 1):
            if was_reranked:
                lines.append(
                    f"{i}. [score={r.get('combined_score', 0):.2f}] "
                    f"{str(r.get('document', ''))[:80]}..."
                )
            else:
                lines.append(
                    f"{i}. [{r.get('distance', 0):.3f}] "
                    f"{str(r.get('document', ''))[:80]}..."
                )

        # Highlight any ordering differences
        cosine_order = [
            str(r.get("document", ""))[:30]
            for r in cosine_results[:3]
        ]
        reranked_order = [
            str(r.get("document", ""))[:30]
            for r in reranked_results[:3]
        ]
        if cosine_order != reranked_order and was_reranked:
            lines += ["", "↕️ Ordering changed by reranking"]
        elif was_reranked:
            lines += ["", "= Same ordering (cosine and reranked agree)"]

        await update.message.reply_text("\n".join(lines)[:4096])

    except asyncio.TimeoutError:
        await update.message.reply_text("Comparison timed out.")
    except Exception as exc:
        await update.message.reply_text(f"Compare error: {exc}")


async def _execute_why_subcommand(
    orchestrator_inst,
    subcommand: str,
    user_id: str,
) -> str:
    """Execute the appropriate introspection query for /why subcommand."""
    registry = orchestrator_inst.cognitive_router.registry
    handlers = {
        "": _why_last_subcommand,
        "last": _why_last_subcommand,
        "moral": _why_moral_subcommand,
        "energy": _why_energy_subcommand,
        "backlog": _why_backlog_subcommand,
        "health": _why_health_subcommand,
    }
    handler = handlers.get(str(subcommand or "").strip().lower())
    if handler is None:
        return (
            f"Unknown subcommand: {subcommand!r}\n\n"
            "Available: /why (last decision), /why moral, /why energy, "
            "/why backlog, /why health"
        )

    return await handler(registry, user_id)


def _parse_why_json_payload(raw: str):
    import json as _json

    try:
        parsed = _json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _why_last_subcommand(registry, user_id: str) -> str:
    fn = registry.get_function("query_decision_log")
    if fn is None:
        return "query_decision_log skill not loaded."

    raw = await fn(scope="last_turn", limit=1, user_id=user_id)
    data = _parse_why_json_payload(raw)
    if data is None:
        return raw

    decisions = data.get("decisions", [])
    total = data.get("total_decisions_on_record", 0)
    if not decisions:
        return (
            f"No supervisor decisions recorded yet "
            f"(total on record: {total}).\n\n"
            "Decisions are logged after each completed turn. "
            "Send a message and then /why to see the decision."
        )

    d = decisions[0]
    reasoning = str(d.get("reasoning_preview", "")).strip()
    is_direct = bool(d.get("is_direct", False))
    workers = int(d.get("worker_count", 0) or 0)

    lines = [
        "📋 Last Supervisor Decision",
        "",
        f"Question: {d.get('user_input_preview', '')}",
        f"Plan: {d.get('plan', 'Direct response')}",
        f"Mode: {'Direct response' if is_direct else f'{workers} agent(s)'}",
    ]
    if reasoning:
        lines.append(f"Reasoning: {reasoning}")
    lines += [
        f"Energy before: {d.get('energy_before', 0)}",
        f"Timestamp: {d.get('timestamp', '')}",
        "",
        f"Total decisions on record: {total}",
        "Use /why moral for audit results - /why energy for budget",
    ]
    return "\n".join(lines)


async def _why_moral_subcommand(registry, user_id: str) -> str:
    fn = registry.get_function("query_decision_log")
    if fn is None:
        return "query_decision_log skill not loaded."

    raw = await fn(scope="all", limit=5, user_id=user_id)
    data = _parse_why_json_payload(raw)
    if data is None:
        return raw

    moral_records = data.get("moral_audit_records", [])
    if not moral_records:
        return (
            "✅ No moral audit records found for recent turns.\n\n"
            "This could mean:\n"
            "- Recent outputs passed trivially (read-only fast-path)\n"
            "- No turns have completed yet\n"
            "Moral audit records are created when the critic evaluates output."
        )

    lines = ["⚖️ Recent Moral Audit Results", ""]
    for r in moral_records[:5]:
        approved = bool(r.get("is_approved", True))
        icon = "✅" if approved else "❌"
        lines.append(f"{icon} [{r.get('audit_mode', '')}] {r.get('request_preview', '?')[:60]}")
        if not approved:
            tiers = r.get("violated_tiers", [])
            constraints = r.get("remediation_constraints", [])
            if tiers:
                lines.append(f"   Violated: {', '.join(tiers)}")
            if constraints:
                lines.append(f"   Constraints: {'; '.join(constraints[:2])}")
        lines.append(f"   {r.get('timestamp', '')}")
        lines.append("")
    return "\n".join(lines)


async def _why_energy_subcommand(registry, user_id: str) -> str:
    _ = user_id
    fn = registry.get_function("query_energy_state")
    if fn is None:
        return "query_energy_state skill not loaded."

    raw = await fn(include_deferred_tasks=True, include_blocked_tasks=True)
    data = _parse_why_json_payload(raw)
    if data is None:
        return raw

    energy = data.get("energy", {})
    snapshot = data.get("backlog_snapshot", {})
    deferred = data.get("deferred_tasks", [])
    blocked = data.get("blocked_tasks", [])

    lines = [
        "⚡ Energy State",
        "",
        f"Current budget: {energy.get('current_predictive_budget', '?')}/{energy.get('initial_budget', 100)} ({energy.get('budget_percent', 0)}%)",
        f"ROI threshold: {energy.get('policy', {}).get('roi_threshold', '?')}",
        f"Min reserve: {energy.get('policy', {}).get('min_reserve', '?')}",
        "",
        "Backlog snapshot:",
        f"  Pending: {snapshot.get('pending', 0)}  "
        f"Active: {snapshot.get('active', 0)}  "
        f"Deferred: {snapshot.get('deferred', 0)}  "
        f"Blocked: {snapshot.get('blocked', 0)}",
    ]

    if deferred:
        lines += ["", f"Deferred tasks ({len(deferred)}):"]
        for t in deferred[:5]:
            force_note = " [⚠ will force-execute next]" if t.get("will_force_execute", False) else ""
            lines.append(
                f"  #{t['id']} {t['title'][:50]} "
                f"(defer #{t.get('defer_count', 0)}, "
                f"ROI={t.get('roi_at_deferral', 0):.2f})"
                f"{force_note}"
            )

    if blocked:
        lines += ["", f"Blocked tasks ({len(blocked)}):"]
        for t in blocked[:3]:
            lines.append(f"  #{t['id']} {t['title'][:50]}")

    return "\n".join(lines)


async def _why_backlog_subcommand(registry, user_id: str) -> str:
    _ = user_id
    fn = registry.get_function("query_objective_status")
    if fn is None:
        return "query_objective_status skill not loaded."

    raw = await fn(status_filter="all", include_energy_context=False, limit=20)
    data = _parse_why_json_payload(raw)
    if data is None:
        return raw

    summary = data.get("backlog_summary", {})
    tasks = data.get("tasks", [])

    lines = [
        "📋 Objective Backlog Status",
        "",
        f"Pending:   {summary.get('pending', 0)}",
        f"Active:    {summary.get('active', 0)}",
        f"Deferred:  {summary.get('deferred_due_to_energy', 0)}",
        f"Blocked:   {summary.get('blocked', 0)}",
        f"Completed: {summary.get('completed', 0)}",
    ]

    if tasks:
        lines += ["", "Active objectives (sample):"]
        status_icons = {
            "pending": "⏳",
            "active": "🔄",
            "deferred_due_to_energy": "💤",
            "blocked": "🚫",
            "completed": "✅",
        }
        for t in tasks[:8]:
            status_icon = status_icons.get(str(t.get("status", "")), "❓")
            lines.append(
                f"  {status_icon} [{t.get('tier', '?')}] "
                f"#{t.get('id', '?')} {str(t.get('title', ''))[:50]}"
            )

    lines += [
        "",
        "Use /why energy for deferred task details",
        "Use /goals for full backlog view",
    ]
    return "\n".join(lines)


async def _why_health_subcommand(registry, user_id: str) -> str:
    _ = user_id
    fn = registry.get_function("query_system_health")
    if fn is None:
        return "query_system_health skill not loaded."

    raw = await fn(hours=24, include_synthesis_history=True, include_error_log=True)
    data = _parse_why_json_payload(raw)
    if data is None:
        return raw

    error_data = data.get("error_log", {})
    synthesis_data = data.get("synthesis_history", {})
    lines = [f"🩺 System Health (last {data.get('report_window_hours', 24)}h)", ""]

    error_summary = error_data.get("summary", {})
    total_errors = int(error_data.get("total_entries", 0) or 0)
    if total_errors == 0:
        lines.append("✅ Error log: clean (no WARNING/ERROR/CRITICAL entries)")
    else:
        summary_parts = [f"{lvl}: {cnt}" for lvl, cnt in sorted(error_summary.items())]
        lines.append(f"⚠️  Error log: {' | '.join(summary_parts)}")
        for e in error_data.get("entries", [])[:3]:
            lines.append(f"   [{e.get('level', '?')}] {e.get('message_preview', '')[:80]}")

    lines.append("")

    synth_total = int(synthesis_data.get("total_runs", 0) or 0)
    if synth_total == 0:
        lines.append("🔧 Synthesis: no runs recorded")
        return "\n".join(lines)

    synth_summary = synthesis_data.get("summary", {})
    summary_parts = [
        f"{_STATUS_ICONS_MAP.get(status, status)}{status}: {count}"
        for status, count in sorted(synth_summary.items())
    ]
    lines.append(f"🔧 Synthesis runs: {' | '.join(summary_parts)}")

    blocked_runs = [
        run for run in synthesis_data.get("runs", [])
        if run.get("status") in ("blocked", "rejected")
    ]
    for run in blocked_runs[:2]:
        lines.append(
            f"   {run.get('status_icon', '?')} {run.get('tool_name', '?')}: "
            f"{run.get('blocked_reason_preview', '')[:80]}"
        )
    return "\n".join(lines)


# Status icon map for telegram_bot module scope
_STATUS_ICONS_MAP = {
    "approved": "✅ ",
    "rejected": "❌ ",
    "blocked": "🚫 ",
    "in_progress": "🔄 ",
    "pending_approval": "⏳ ",
}


# --- Session subcommand implementations ---

async def _session_status(
    update: Update, orchestrator_inst
) -> None:
    active = await orchestrator_inst._get_active_session()
    if not active:
        await update.message.reply_text(
            "No active session.\n"
            "Use /session new <name> to create one, or "
            "/session list to see existing sessions."
        )
        return
    epic_id = active.get("epic_id")
    epic_line = f"\nLinked Epic: #{epic_id}" if epic_id else "\nNo linked Epic"
    await update.message.reply_text(
        f"Active session\n"
        f"  id: {active['id']}\n"
        f"  name: {active['name']}\n"
        f"  description: {active.get('description') or '—'}\n"
        f"  turns: {active.get('turn_count', 0)}\n"
        f"  memories: {active.get('memory_count', 0)}\n"
        f"  started: {active.get('started_at') or active.get('created_at', '?')}"
        f"{epic_line}"
    )


async def _session_new(
    update: Update, ledger, orchestrator_inst, name: str, description: str
) -> None:
    session_id = await ledger.create_session(name, description)
    row = await orchestrator_inst._activate_session(session_id)
    if row is None:
        await update.message.reply_text(f"Created session #{session_id} but could not activate it.")
        return
    await update.message.reply_text(
        f"Session created and activated.\n"
        f"  id: {session_id}\n"
        f"  name: {name}\n"
        f"  description: {description or '—'}\n\n"
        f"Future messages will be scoped to this session.\n"
        f"Use /session link <epic_id> to connect to a project Epic."
    )
    logger.info(
        "Admin created and activated session #%s: %r",
        session_id, name,
    )


async def _session_list(update: Update, ledger) -> None:
    sessions = await ledger.list_sessions(limit=15)
    if not sessions:
        await update.message.reply_text("No sessions found. Use /session new <name> to create one.")
        return
    lines = ["Sessions:"]
    for s in sessions:
        active_marker = " [ACTIVE]" if s.get("is_active") else ""
        epic_tag = f" Epic#{s['epic_id']}" if s.get("epic_id") else ""
        lines.append(
            f"  #{s['id']}{active_marker} {s['name']}{epic_tag} "
            f"— {s.get('turn_count', 0)} turns"
        )
    await update.message.reply_text("\n".join(lines))


async def _session_switch(
    update: Update, ledger, orchestrator_inst, session_id: int
) -> None:
    _ = ledger
    row = await orchestrator_inst._activate_session(session_id)
    if row is None:
        await update.message.reply_text(
            f"Session #{session_id} not found. Use /session list to see available sessions."
        )
        return
    epic_line = f"\nLinked Epic: #{row['epic_id']}" if row.get("epic_id") else ""
    await update.message.reply_text(
        f"Switched to session #{session_id}: {row['name']}"
        f"{epic_line}\n\n"
        f"Chat history and memories are now scoped to this session."
    )


async def _session_link(
    update: Update, ledger, orchestrator_inst, epic_id: int
) -> None:
    active = await orchestrator_inst._get_active_session()
    if not active:
        await update.message.reply_text(
            "No active session. Use /session switch <id> first."
        )
        return
    session_id = int(active["id"])
    # Verify the Epic exists in the backlog
    tree = await ledger.get_active_objective_tree(epic_id)
    epic_node = next(
        (n for n in tree if n.get("tier") == "Epic" and int(n.get("id") or 0) == epic_id),
        None,
    )
    if epic_node is None:
        await update.message.reply_text(
            f"Epic #{epic_id} not found in active backlog. "
            f"Use /goals to see available Epics."
        )
        return
    success = await ledger.link_session_to_epic(session_id, epic_id)
    if not success:
        await update.message.reply_text("Failed to link session.")
        return
    # Refresh core memory cache with the new epic_id
    updated_row = await ledger.get_session(session_id)
    if updated_row:
        await orchestrator_inst._sync_active_session_to_core(updated_row)
    await update.message.reply_text(
        f"Session #{session_id} linked to Epic #{epic_id}: "
        f"{epic_node.get('title', '')}\n\n"
        f"Archival memory searches will now prioritise memories "
        f"from this project."
    )


async def _session_end(update: Update, orchestrator_inst) -> None:
    active = await orchestrator_inst._get_active_session()
    if not active:
        await update.message.reply_text("No active session to end.")
        return
    name = active.get("name", "Unknown")
    await orchestrator_inst._deactivate_session()
    await update.message.reply_text(
        f"Session '{name}' deactivated.\n"
        f"AIDEN is now in global context mode — "
        f"all chat history is loaded chronologically."
    )


async def _session_summary(
    update: Update, ledger, orchestrator_inst
) -> None:
    _ = ledger
    active = await orchestrator_inst._get_active_session()
    if not active:
        await update.message.reply_text(
            "No active session. Use /session switch <id> first."
        )
        return
    session_id = int(active["id"])
    # Build a summary by calling process_message with a meta-prompt
    prompt = (
        f"[SESSION SUMMARY REQUEST] "
        f"Summarise the key decisions, findings, and open questions from "
        f"this session so far. Session: {active['name']} (id={session_id}). "
        f"Keep the summary under 400 words. Structure: "
        f"1) What was accomplished 2) Key decisions made 3) Open items."
    )
    await update.message.reply_text("Generating session summary...")
    try:
        summary = await asyncio.wait_for(
            orchestrator_inst.process_message(prompt, str(update.effective_user.id)),
            timeout=float(os.getenv("AGENT_TIMEOUT_SECONDS", "300")),
        )
        await update.message.reply_text(summary[:4096])
    except Exception as e:
        await update.message.reply_text(f"Summary failed: {e}")


async def eval_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """/eval [suite] - run eval suites and report results to admin."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    args = getattr(context, "args", []) or []
    suite_filter = args[0].lower() if args else None

    await update.message.reply_text(
        f"Running eval suites{' (' + suite_filter + ')' if suite_filter else ''}... "
        f"This takes about 10 seconds."
    )

    try:
        cmd = [sys.executable, "scripts/run_evals.py"]
        if suite_filter:
            cmd += ["--suite", suite_filter]

        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=5.0,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=120.0,
        )

        output = stdout.decode(errors="replace").strip()
        err_output = stderr.decode(errors="replace").strip()
        if err_output:
            if output:
                output = f"{output}\n\n[stderr]\n{err_output}"
            else:
                output = err_output

        if not output:
            output = "(no output)"

        status = "PASS" if proc.returncode == 0 else "FAIL"
        message = f"{status}\n\n```\n{output[-3000:]}\n```"
        await update.message.reply_text(message[:4096])

    except asyncio.TimeoutError:
        await update.message.reply_text("Eval runner timed out after 120s.")
    except Exception as exc:
        await update.message.reply_text(f"Eval failed: {exc}")


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

        outbound_queue: asyncio.Queue = asyncio.Queue(maxsize=_resolve_outbound_queue_max_size())
        orchestrator_inst.outbound_queue = outbound_queue

        # Async-init the orchestrator
        await orchestrator_inst.async_init()
        ready_event = getattr(orchestrator_inst, "_ready", None)
        if ready_event is not None and not ready_event.is_set():
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=30.0)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Orchestrator did not become ready during startup.") from exc
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
            _enqueue_outbound_with_drop_oldest(
                outbound_queue,
                f"\u26a0\ufe0f AIDEN restarted — pending task restored:\n\n{_question}",
                source="async_run_pending_hitl_restore",
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

            # Let fire-and-forget persistence writes finish before closing resources.
            await _drain_background_tasks(orchestrator_inst, timeout=5.0)

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
        # Acquire a simple single-instance lock using a UDP port bind.
        lock_port = int(os.getenv("TELEGRAM_LOCK_PORT", "55678"))
        try:
            lock_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # SO_EXCLUSIVEADDRUSE (Windows) prevents other processes from binding
            # the same port even with SO_REUSEADDR set, fixing the lock on Windows.
            # Fall back to SO_REUSEADDR on non-Windows (Linux TIME_WAIT behaviour).
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                lock_sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:
                lock_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            lock_sock.bind(("127.0.0.1", lock_port))
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
        application.add_handler(CommandHandler("status", status))
        application.add_handler(CommandHandler("retire_unused_tools", retire_unused_tools))
        application.add_handler(CommandHandler("goals", goals))
        application.add_handler(CommandHandler("addgoal", addgoal))
        application.add_handler(CommandHandler("session", session_command))
        application.add_handler(CommandHandler("why", why_command))
        application.add_handler(CommandHandler("memory", memory_command))
        application.add_handler(CommandHandler("eval", eval_command))
        application.add_handler(CommandHandler("shutdown", shutdown))
        application.add_handler(CommandHandler("restart", restart))
        application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
        application.add_handler(MessageHandler(filters.Document.ALL, handle_document_message))
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
            # Always release the lock before any restart path. The restart
            # branch below uses os.execv, so this explicit close guarantees
            # the replacement process can re-bind the lock port immediately.
            if lock_sock:
                lock_sock.close()
                logger.info("Released single-instance lock")
        except Exception:
            pass

    # If a restart was requested, replace the current process with a fresh one.
    # os.execv replaces the process image in-place while keeping the same PID.
    # Ordering matters: the lock socket is closed in finally above before execv,
    # preventing a self-deadlock where the new image cannot acquire the lock.
    if _restart_requested:
        logger.info("Restarting process...")
        # Flush log handlers so nothing is lost
        for handler in logging.getLogger().handlers:
            handler.flush()
        os.execv(sys.executable, [sys.executable] + sys.argv)



if __name__ == "__main__":
    main()

