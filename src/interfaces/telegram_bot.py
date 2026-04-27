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
from typing import Any
from dotenv import load_dotenv

# Load .env BEFORE importing any project modules so that module-level os.getenv()
# calls in llm_router, orchestrator, etc. see the correct values.  override=True
# ensures the file always wins over stale process-level env vars.
load_dotenv(override=True)

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import Conflict
from src.core.orchestrator import Orchestrator
from src.interfaces.streaming import DeferredStatusMessage, StatusFormatter
from src.core.progress import ProgressEvent

# Ensure stdout/stderr use UTF-8 on Windows so Unicode log characters don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Configure logging
_log_max_bytes = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # default 10 MB
_log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
_log_file_handler = logging.handlers.RotatingFileHandler(
    'logs/telegram_bot.log',
    maxBytes=_log_max_bytes,
    backupCount=_log_backup_count,
    encoding='utf-8',
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        _log_file_handler,
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
_SERVICE_UNAVAILABLE_MSG = "System error: Service temporarily unavailable."
_TIMEOUT_MSG = "Request timed out. Please try again."
_ORCHESTRATOR_NOT_INITIALIZED = "Orchestrator not initialized"
# Show status bubble after this many seconds of processing time
_STREAMING_DELAY_SECONDS = 2.5
# Messages with at least this many whitespace-separated tokens get the status UI
_STREAMING_MIN_TOKENS = 4
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ADMIN_USER_ID = int(os.getenv('ADMIN_USER_ID', 0))
AGENT_TIMEOUT_SECONDS = float(os.getenv('AGENT_TIMEOUT_SECONDS', 300.0))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {'1', 'true', 'yes', 'on'}


ENABLE_NATIVE_AUDIO = _env_bool('ENABLE_NATIVE_AUDIO', default=False)

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

    The queue may contain either a plain string or a dict produced by
    tool_synthesis_node when the code body exceeds 3 500 chars:
        {"text": str, "document_path": str}
    In the dict case the code is sent as a Telegram document and the
    temporary file is always deleted afterwards to prevent disk leaks.
    """
    while True:
        try:
            message = await queue.get()
            if isinstance(message, dict):
                text = str(message.get("text") or "")[:4096]
                document_path: str | None = message.get("document_path")
                await bot.send_message(chat_id=ADMIN_USER_ID, text=text)
                if document_path and os.path.isfile(document_path):
                    try:
                        with open(document_path, "rb") as doc_file:
                            await bot.send_document(
                                chat_id=ADMIN_USER_ID,
                                document=doc_file,
                                filename=os.path.basename(document_path),
                            )
                    finally:
                        try:
                            os.unlink(document_path)
                        except OSError as unlink_err:
                            logger.warning(
                                "Could not delete synthesis temp file %s: %s",
                                document_path,
                                unlink_err,
                            )
            else:
                await bot.send_message(chat_id=ADMIN_USER_ID, text=str(message)[:4096])
            queue.task_done()
        except Exception as e:
            logger.error(f"Failed to send admin notification: {e}", exc_info=True)


def _coerce_outbound_text(value: object) -> str:
    """Coerce any orchestrator response to a non-empty displayable string."""
    if value is None:
        return "I completed your request but generated an empty response. Please retry."
    text = str(value).strip()
    if not text:
        return "I completed your request but generated an empty response. Please retry."
    return text


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

    Short messages use a simple typing-indicator path.
    Longer messages use DeferredStatusMessage to show live progress,
    then always deliver the final response as a new reply_text (the
    status bubble is deleted first so the answer is clearly visible).
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
            outbound_text = _coerce_outbound_text(ai_response)
            await update.message.reply_text(outbound_text[:4096])
            logger.info(
                "Response sent to user %s (mode=direct len=%s)",
                user_id,
                len(outbound_text),
            )
            return

        # Streaming path — DeferredStatusMessage sends status only if slow.
        # After processing, cancel (delete) the status bubble and always send
        # the final answer as a NEW reply_text so it is clearly visible.
        formatter = StatusFormatter()
        dsm = DeferredStatusMessage(
            update.message,
            delay_seconds=_STREAMING_DELAY_SECONDS,
        )
        await dsm.start()

        # Keep typing indicator running until status message appears.
        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(
            _keep_typing_alive(context.bot, update.effective_chat.id, stop_typing)
        )

        async def on_progress(event: ProgressEvent) -> None:
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
            await dsm.cancel()
            await update.message.reply_text(_TIMEOUT_MSG)
            return
        except Exception as e:
            stop_typing.set()
            await typing_task
            logger.error("Error processing message: %s", e, exc_info=True)
            await dsm.cancel()
            await update.message.reply_text(
                "An error occurred while processing your message. Please try again."
            )
            return
        finally:
            if not stop_typing.is_set():
                stop_typing.set()
            try:
                await asyncio.wait_for(typing_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Delete the status bubble, then send the answer as a fresh reply
        # so the user always sees a proper new message (not an edited bubble).
        outbound_text = _coerce_outbound_text(ai_response)
        await dsm.cancel()
        await update.message.reply_text(outbound_text[:4096])
        logger.info(
            "Response delivered to user %s (mode=reply_text len=%s)",
            user_id,
            len(outbound_text),
        )

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


async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Telegram documents without persisting file contents locally."""
    global orchestrator

    user_id = update.effective_user.id
    if not is_authorized(user_id):
        logger.warning("Unauthorized document message from user %s", user_id)
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    document = getattr(update.message, "document", None)
    if document is None:
        await update.message.reply_text("Unsupported document payload.")
        return

    if orchestrator is None:
        logger.error(_ORCHESTRATOR_NOT_INITIALIZED)
        await update.message.reply_text(_SERVICE_UNAVAILABLE_MSG)
        return

    file_name = sanitize_for_logging(str(getattr(document, "file_name", "document") or "document"), max_length=128)
    file_size = getattr(document, "file_size", None)
    caption = str(getattr(update.message, "caption", "") or "").strip()
    prompt = (
        f"{caption}\n\n"
        f"[Document received: {file_name}"
        f"{f' · {file_size} bytes' if file_size is not None else ''}]"
    ).strip()

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(
        _keep_typing_alive(context.bot, update.effective_chat.id, stop_typing)
    )
    try:
        ai_response = await asyncio.wait_for(
            orchestrator.process_message(prompt, str(user_id)),
            timeout=AGENT_TIMEOUT_SECONDS,
        )
    except (TimeoutError, asyncio.TimeoutError):
        await update.message.reply_text(_TIMEOUT_MSG)
        return
    except Exception as e:
        logger.error("Error processing document message: %s", e, exc_info=True)
        await update.message.reply_text("An error occurred while processing your document. Please try again.")
        return
    finally:
        stop_typing.set()
        try:
            await asyncio.wait_for(typing_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

    await update.message.reply_text(_coerce_outbound_text(ai_response)[:4096])


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


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/status — report basic operational counters."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    orchestrator_inst = context.bot_data.get("orchestrator") if context is not None else orchestrator
    if orchestrator_inst is None:
        await update.message.reply_text(_SERVICE_UNAVAILABLE_MSG)
        return

    registry = getattr(getattr(orchestrator_inst, "cognitive_router", None), "registry", None)
    try:
        loaded_skills = len(registry) if registry is not None else 0
    except Exception:
        loaded_skills = 0

    load_errors = []
    get_load_errors = getattr(registry, "get_load_errors", None)
    if callable(get_load_errors):
        try:
            load_errors = list(get_load_errors())
        except Exception:
            load_errors = []
    failed_names = [str(item[0]) for item in load_errors if item]

    energy = "unknown"
    energy_fn = getattr(orchestrator_inst, "_get_predictive_energy_budget_remaining", None)
    if callable(energy_fn):
        try:
            energy = str(await energy_fn())
        except Exception:
            energy = "unknown"

    failed_preview = f" ({', '.join(failed_names[:5])})" if failed_names else ""

    text = (
        "AIDEN status\n"
        f"Loaded skills: {loaded_skills}\n"
        f"Failed skills: {len(failed_names)}"
        f"{failed_preview}\n"
        f"Pending MFA: {len(getattr(orchestrator_inst, 'pending_mfa', {}) or {})}\n"
        f"Pending HITL: {len(getattr(orchestrator_inst, 'pending_hitl_state', {}) or {})}\n"
        f"Predictive energy budget: {energy}"
    )
    await update.message.reply_text(text)


async def emailstatus(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/emailstatus — report email polling health and queue metrics."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    orchestrator_inst = context.bot_data.get("orchestrator") if context is not None else orchestrator
    if orchestrator_inst is None:
        await update.message.reply_text(_SERVICE_UNAVAILABLE_MSG)
        return

    try:
        stats = await orchestrator_inst.get_email_poll_status()
    except Exception as exc:
        logger.warning("/emailstatus failed: %s", exc, exc_info=True)
        await update.message.reply_text("Email status unavailable right now.")
        return

    text = (
        "AIDEN email status\n"
        f"Last poll time: {stats.get('last_poll_time', 'unknown')}\n"
        f"Emails processed (last 24h): {stats.get('emails_processed_last_24h', 0)}\n"
        f"Pending google-domain tasks: {stats.get('pending_google_tasks', 0)}"
    )
    await update.message.reply_text(text)


async def retire_unused_tools(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/retire_unused_tools [confirm] — retire approved tools that have never been used recently."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    ledger = context.bot_data.get("ledger")
    if ledger is None:
        await update.message.reply_text("System error: Ledger not available.")
        return

    user_key = str(update.effective_user.id)
    pending = context.bot_data.setdefault("pending_retire_unused_tools", {})
    args = [str(arg).lower() for arg in (getattr(context, "args", None) or [])]

    if args and args[0] == "confirm":
        pending_entry = pending.pop(user_key, None)
        tool_names = list((pending_entry or {}).get("tool_names") or [])
        if not tool_names:
            await update.message.reply_text("No unused tools are pending retirement.")
            return
        retired_count = await ledger.retire_tools(tool_names)
        await update.message.reply_text(f"Retired {retired_count} tool(s): {', '.join(tool_names)}")
        return

    items = await ledger.get_unused_approved_tools()
    tool_names = [str(item.get("name") or "").strip() for item in items if str(item.get("name") or "").strip()]
    if not tool_names:
        await update.message.reply_text("No unused approved tools found.")
        return

    pending[user_key] = {"tool_names": tool_names, "created_at": time.time()}
    lines = ["Unused approved tools:"]
    for item in items:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        last_used = item.get("last_used_at") or "never"
        lines.append(f"- {name} (last used: {last_used})")
    lines.append("\nReply with /retire_unused_tools confirm to retire these tools.")
    await update.message.reply_text("\n".join(lines))


async def learnings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/learnings — show outcome score rankings and energy estimation accuracy."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text(_UNAUTHORIZED_MSG)
        return

    ledger = context.bot_data.get("ledger")
    if ledger is None:
        await update.message.reply_text("System error: Ledger not available.")
        return

    try:
        summary = await ledger.get_outcome_learnings(days=30)
    except Exception as e:
        logger.error("/learnings error: %s", e, exc_info=True)
        await update.message.reply_text(f"Error fetching learnings: {e}")
        return

    top = summary.get("top_classes") or []
    bottom = summary.get("bottom_classes") or []
    accuracy = summary.get("energy_accuracy_pct")
    sample = summary.get("energy_accuracy_sample", 0)
    days = summary.get("days_window", 30)

    lines = [f"--- Outcome Learnings (last {days} days) ---"]

    if top:
        lines.append("\n\U0001f3c6 Top 5 task classes (by outcome score):")
        for i, item in enumerate(top, 1):
            lines.append(
                f"  {i}. {item['title'][:60]}  "
                f"avg={item['avg_score']:.1f}/5 (n={item['count']})"
            )
    else:
        lines.append("\nNo scored tasks in this window.")

    if bottom:
        lines.append("\n\U0001f53b Bottom 5 task classes (by outcome score):")
        for i, item in enumerate(bottom, 1):
            lines.append(
                f"  {i}. {item['title'][:60]}  "
                f"avg={item['avg_score']:.1f}/5 (n={item['count']})"
            )

    lines.append(f"\n\u26a1 Energy estimation accuracy (last {days} days):")
    if accuracy is not None:
        lines.append(
            f"  {accuracy}% of tasks estimated within ±50% of actual "
            f"(sample n={sample})"
        )
    else:
        lines.append("  Not enough data yet (need tasks with actual_energy_used recorded).")

    await update.message.reply_text("\n".join(lines))


async def _drain_background_tasks(orchestrator: Any, *, timeout: float = 10.0) -> None:
    """Wait for all orchestrator background tasks to finish, up to *timeout* seconds.

    This is used during graceful shutdown to ensure in-flight writes (e.g. chat-
    turn persistence) are not silently dropped.  If the timeout is exceeded, a
    WARNING is logged and the function returns — it never raises.
    """
    tasks: set = getattr(orchestrator, "_background_tasks", set()) or set()
    if not tasks:
        return

    pending = {t for t in tasks if not t.done()}
    if not pending:
        return

    try:
        await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), timeout=timeout)
    except asyncio.TimeoutError:
        still_pending = [t for t in pending if not t.done()]
        logger.warning(
            "Background task(s) did not drain within %.1fs — %d still pending.",
            timeout,
            len(still_pending),
        )


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
        _sensory_task = asyncio.create_task(orchestrator_inst._sensory_update_loop())
        _email_poll_task = asyncio.create_task(orchestrator_inst._email_poll_loop())
        _drain_task = asyncio.create_task(_drain_outbound_queue(application.bot, outbound_queue))
        _memory_task = asyncio.create_task(orchestrator_inst._memory_consolidation_loop())
        _bg_tasks.extend([_heartbeat_task, _sensory_task, _email_poll_task, _drain_task, _memory_task])
        logger.info(
            "Orchestrator async_init, heartbeat loop, sensory update loop, email poll loop, memory consolidation loop, "
            "and outbound queue drainer started. "
            "Tasks: "
            f"{_heartbeat_task.get_name()}, {_sensory_task.get_name()}, {_email_poll_task.get_name()}, "
            f"{_memory_task.get_name()}, {_drain_task.get_name()}"
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

    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable not set")
    if ADMIN_USER_ID == 0:
        raise ValueError("ADMIN_USER_ID environment variable not set")

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
        application.add_handler(CommandHandler("status", status))
        application.add_handler(CommandHandler("emailstatus", emailstatus))
        application.add_handler(CommandHandler("retire_unused_tools", retire_unused_tools))
        application.add_handler(CommandHandler("learnings", learnings))
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
