"""
Ledger Database Module for Short-Term Memory and Task Management.

This module provides structured storage for operational state and system events
using SQLite via the non-blocking ``aiosqlite`` driver.  All public methods are
async-safe and concurrency-safe through a shared ``asyncio.Lock``.
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

import aiosqlite

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Enumeration of system log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TaskStatus(Enum):
    """Enumeration of task statuses."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Pre-calculate status values for performance optimization
VALID_TASK_STATUSES = {s.value for s in TaskStatus}
VALID_TASK_STATUSES_LIST = [s.value for s in TaskStatus]


class LedgerMemory:
    """
    An async SQLite-backed ledger for short-term memory and task management.

    Construction is synchronous (just sets paths).  Call ``await initialize()``
    before any other method — the Orchestrator's ``async_init()`` does this
    automatically.

    A single ``aiosqlite`` connection is shared across all coroutines, protected
    by an ``asyncio.Lock`` to prevent database-lock errors under concurrent load.
    """

    def __init__(self, db_path: str = "data/ledger.db") -> None:
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Open the async connection and create all tables.  Call once at startup."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._initialize_tables()
        logger.info(f"LedgerMemory initialized (aiosqlite) at {self.db_path}")

    async def _initialize_tables(self) -> None:
        """Create tables and indices if they don't exist."""
        async with self._lock:
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS task_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    priority INTEGER NOT NULL DEFAULT 5,
                    task_description TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context TEXT
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_status ON task_queue(status)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_priority ON task_queue(priority)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_log_timestamp ON system_logs(timestamp)
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS objective_backlog (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tier TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER NOT NULL DEFAULT 5,
                    estimated_energy INTEGER NOT NULL DEFAULT 10,
                    origin TEXT NOT NULL DEFAULT 'Admin',
                    parent_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_id) REFERENCES objective_backlog(id)
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_objective_status ON objective_backlog(status)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_objective_tier ON objective_backlog(tier)
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_user_id ON chat_history(user_id, timestamp)
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS tool_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    code TEXT NOT NULL,
                    schema_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending_approval',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    approved_at TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS pending_tool_approvals (
                    user_id TEXT NOT NULL PRIMARY KEY,
                    synthesis_json TEXT NOT NULL,
                    original_input TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS pending_hitl_states (
                    user_id TEXT NOT NULL PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT NOT NULL PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS pending_mfa_states (
                    user_id TEXT NOT NULL PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db.commit()
        logger.debug("Database tables initialized")

    async def close(self) -> None:
        """Close the aiosqlite connection."""
        try:
            if self._db is not None:
                await self._db.close()
                self._db = None
                logger.info("LedgerMemory connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

    # ─────────────────────────────────────────────────────────────
    # Task Queue
    # ─────────────────────────────────────────────────────────────

    async def add_task(
        self,
        task_description: str,
        priority: int = 5,
        status: str = "PENDING",
    ) -> int:
        """Add a task to the task queue and return its ID."""
        if not task_description or not isinstance(task_description, str):
            raise ValueError("Task description must be a non-empty string")
        if not 1 <= priority <= 10:
            raise ValueError("Priority must be between 1 and 10")
        if status not in VALID_TASK_STATUSES:
            raise ValueError(f"Status must be one of {VALID_TASK_STATUSES_LIST}")

        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO task_queue (priority, task_description, status) VALUES (?, ?, ?)",
                (priority, task_description, status),
            )
            await self._db.commit()
            task_id = cursor.lastrowid

        logger.info(f"Task {task_id} added: {task_description}")
        await self.log_event(
            LogLevel.INFO,
            f"Task created: {task_description}",
            {"task_id": task_id, "priority": priority},
        )
        return task_id

    async def get_pending_tasks(
        self,
        order_by_priority: bool = True,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve pending tasks from the queue."""
        # Use a whitelist instead of interpolating user-controlled strings into SQL
        order = (
            "priority ASC, created_at ASC" if order_by_priority else "created_at ASC"
        )
        query = (
            "SELECT id, priority, task_description, status, created_at, updated_at "
            f"FROM task_queue WHERE status = 'PENDING' ORDER BY {order} LIMIT ?"
        )
        async with self._lock:
            cursor = await self._db.execute(query, (int(limit),))
            rows = await cursor.fetchall()
        tasks = [dict(r) for r in rows]
        logger.info(f"Retrieved {len(tasks)} pending tasks")
        return tasks

    async def update_task_status(self, task_id: int, new_status: str) -> bool:
        """Update the status of a task. Returns True if found and updated."""
        if new_status not in VALID_TASK_STATUSES:
            raise ValueError(f"Status must be one of {VALID_TASK_STATUSES_LIST}")

        async with self._lock:
            cursor = await self._db.execute(
                "UPDATE task_queue SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (new_status, task_id),
            )
            await self._db.commit()
            updated = cursor.rowcount > 0

        if updated:
            logger.info(f"Task {task_id} status updated to {new_status}")
        else:
            logger.warning(f"Task {task_id} not found")
        return updated

    # ─────────────────────────────────────────────────────────────
    # System Logs
    # ─────────────────────────────────────────────────────────────

    async def log_event(
        self,
        log_level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Log a system event to the ledger and return its ID."""
        if not message or not isinstance(message, str):
            raise ValueError("Message must be a non-empty string")

        context_str = str(context) if context else None
        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO system_logs (log_level, message, context) VALUES (?, ?, ?)",
                (log_level.value, message, context_str),
            )
            await self._db.commit()
            log_id = cursor.lastrowid
        logger.debug(f"Log entry {log_id} created: {message}")
        return log_id

    async def get_logs(
        self,
        log_level: Optional[LogLevel] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve system logs, optionally filtered by level."""
        async with self._lock:
            if log_level:
                cursor = await self._db.execute(
                    "SELECT id, log_level, message, timestamp, context "
                    "FROM system_logs WHERE log_level = ? ORDER BY timestamp DESC LIMIT ?",
                    (log_level.value, limit),
                )
            else:
                cursor = await self._db.execute(
                    "SELECT id, log_level, message, timestamp, context "
                    "FROM system_logs ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            rows = await cursor.fetchall()
        logs = [dict(r) for r in rows]
        logger.info(f"Retrieved {len(logs)} log entries")
        return logs

    # ─────────────────────────────────────────────────────────────
    # Objective Backlog  (3-tier: Epic > Story > Task)
    # ─────────────────────────────────────────────────────────────

    async def add_objective(
        self,
        tier: str,
        title: str,
        estimated_energy: int = 10,
        origin: str = "Admin",
        priority: int = 5,
        parent_id: Optional[int] = None,
    ) -> int:
        """Insert a new Epic/Story/Task into the objective_backlog."""
        if tier not in ("Epic", "Story", "Task"):
            raise ValueError("tier must be Epic, Story, or Task")
        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO objective_backlog "
                "(tier, title, status, priority, estimated_energy, origin, parent_id) "
                "VALUES (?, ?, 'pending', ?, ?, ?, ?)",
                (tier, title, priority, estimated_energy, origin, parent_id),
            )
            await self._db.commit()
            obj_id = cursor.lastrowid
        logger.info(f"Objective added: [{tier}] {title!r} (id={obj_id})")
        return obj_id

    async def get_highest_priority_task(self) -> Optional[dict]:
        """Return the single highest-priority pending Task, or None."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, tier, title, status, priority, estimated_energy, "
                "origin, parent_id, created_at FROM objective_backlog "
                "WHERE tier = 'Task' AND status = 'pending' "
                "ORDER BY priority ASC, estimated_energy ASC LIMIT 1"
            )
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_objective_status(self, obj_id: int, new_status: str) -> None:
        """Update status of an objective. Valid: pending, active, completed, suspended."""
        valid = {"pending", "active", "completed", "suspended"}
        if new_status not in valid:
            raise ValueError(f"status must be one of {valid}")
        async with self._lock:
            await self._db.execute(
                "UPDATE objective_backlog SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (new_status, obj_id),
            )
            await self._db.commit()
        logger.info(f"Objective {obj_id} status -> {new_status}")

    async def get_all_active_goals(self) -> List[dict]:
        """Return all non-completed, non-suspended objectives grouped by tier."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, tier, title, status, priority, estimated_energy, origin, parent_id, created_at "
                "FROM objective_backlog WHERE status NOT IN ('completed', 'suspended') "
                "ORDER BY CASE tier WHEN 'Epic' THEN 1 WHEN 'Story' THEN 2 WHEN 'Task' THEN 3 END, priority ASC"
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def seed_initial_goals(self) -> None:
        """Populate the backlog with initial Tasks if it is currently empty."""
        async with self._lock:
            cursor = await self._db.execute("SELECT COUNT(*) FROM objective_backlog")
            row = await cursor.fetchone()
            count = row[0]
        if count > 0:
            return

        seeds = [
            ("Task", "Scan the /src/ directory and summarize the Python architecture to Archival Memory.", 15, 1),
            ("Task", "Update core_memory.json with Admin timezone (Vienna) and basic formatting preferences.", 5, 2),
            ("Task", "Execute a test read/write on the SQLite Ledger and log the latency.", 10, 3),
        ]
        for tier, title, energy, prio in seeds:
            await self.add_objective(tier=tier, title=title, estimated_energy=energy,
                                     origin="System", priority=prio)

        epic_id = await self.add_objective(
            tier="Epic", title="Continuous System Optimization",
            estimated_energy=50, origin="System", priority=5,
        )
        story_id = await self.add_objective(
            tier="Story", title="Analyze System Logs for Tool Failures",
            estimated_energy=20, origin="System", priority=5, parent_id=epic_id,
        )
        await self.add_objective(
            tier="Task", title="Run memory consolidation on recent chat history",
            estimated_energy=15, origin="System", priority=4, parent_id=story_id,
        )
        logger.info("Seeded 3 operational Tasks + Meta-Reflection Epic hierarchy into objective_backlog.")

    # ─────────────────────────────────────────────────────────────
    # Chat History  (short-term conversational memory)
    # ─────────────────────────────────────────────────────────────

    async def save_chat_turn(self, user_id: str, role: str, content: str) -> None:
        """Persist one conversational turn (non-blocking)."""
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        async with self._lock:
            await self._db.execute(
                "INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
                (user_id, role, content),
            )
            await self._db.commit()

    async def get_chat_history(self, user_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieve the most recent *limit* turns for a user in chronological order."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT role, content FROM ("
                "  SELECT id, role, content FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC",
                (user_id, limit),
            )
            rows = await cursor.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    async def trim_chat_history(self, user_id: str, keep_last: int = 5) -> int:
        """Trim chat history to the most recent *keep_last* turns. Returns rows deleted."""
        if keep_last < 1:
            raise ValueError("keep_last must be at least 1")
        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM chat_history WHERE user_id = ? AND id NOT IN ("
                "  SELECT id FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?"
                ")",
                (user_id, user_id, keep_last),
            )
            await self._db.commit()
            deleted = cursor.rowcount
        return deleted

    async def prune_old_chat_history(self, days: int = 90, keep_minimum: int = 20) -> int:
        """Delete chat_history rows older than *days* days, always keeping the most recent
        *keep_minimum* rows per user so active users are never over-pruned."""
        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM chat_history "
                "WHERE timestamp < datetime('now', ? || ' days') "
                "AND id NOT IN ("
                "  SELECT id FROM chat_history ch2 "
                "  WHERE ch2.user_id = chat_history.user_id "
                "  ORDER BY id DESC LIMIT ?"
                ")",
                (f"-{days}", keep_minimum),
            )
            await self._db.commit()
        return cursor.rowcount

    async def get_recent_user_ids(self, limit: int = 20) -> List[str]:
        """Return up to *limit* user_ids ordered by most recent chat activity."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT user_id, MAX(id) AS max_id FROM chat_history "
                "WHERE user_id != 'heartbeat' "
                "GROUP BY user_id ORDER BY max_id DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
        return [r["user_id"] for r in rows]

    # ─────────────────────────────────────────────────────────────
    # Tool Registry  (System-2-synthesised dynamic tools)
    # ─────────────────────────────────────────────────────────────

    async def register_tool(self, name: str, description: str, code: str, schema_json: str) -> int:
        """Insert or replace a pending tool into the registry."""
        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO tool_registry (name, description, code, schema_json, status) "
                "VALUES (?, ?, ?, ?, 'pending_approval') "
                "ON CONFLICT(name) DO UPDATE SET "
                "description=excluded.description, code=excluded.code, "
                "schema_json=excluded.schema_json, status='pending_approval', approved_at=NULL",
                (name, description, code, schema_json),
            )
            await self._db.commit()
            return cursor.lastrowid

    async def approve_tool(self, name: str) -> None:
        """Mark a tool as approved and record the approval timestamp."""
        async with self._lock:
            await self._db.execute(
                "UPDATE tool_registry SET status='approved', approved_at=CURRENT_TIMESTAMP WHERE name=?",
                (name,),
            )
            await self._db.commit()
        logger.info(f"Tool '{name}' approved")

    async def get_approved_tools(self) -> List[Dict[str, Any]]:
        """Return all approved tools (name, code, schema_json)."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT name, description, code, schema_json FROM tool_registry "
                "WHERE status='approved' ORDER BY approved_at ASC"
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────
    # Pending Tool Approvals  (survive bot restarts)
    # ─────────────────────────────────────────────────────────────

    async def save_pending_approval(self, user_id: str, synthesis: dict, original_input: str) -> None:
        """Persist a tool synthesis payload so approval survives a restart."""
        import json
        async with self._lock:
            await self._db.execute(
                "INSERT INTO pending_tool_approvals (user_id, synthesis_json, original_input) "
                "VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
                "synthesis_json=excluded.synthesis_json, original_input=excluded.original_input, "
                "created_at=CURRENT_TIMESTAMP",
                (user_id, json.dumps(synthesis), original_input),
            )
            await self._db.commit()

    async def load_pending_approvals(self) -> dict:
        """Return {user_id: {synthesis, original_input}} for all pending approvals."""
        import json
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT user_id, synthesis_json, original_input FROM pending_tool_approvals"
            )
            rows = await cursor.fetchall()
        result = {}
        for row in rows:
            try:
                result[row["user_id"]] = {
                    "synthesis": json.loads(row["synthesis_json"]),
                    "original_input": row["original_input"],
                }
            except Exception:
                pass
        return result

    async def clear_pending_approval(self, user_id: str) -> None:
        """Remove a pending approval after it is accepted or rejected."""
        async with self._lock:
            await self._db.execute(
                "DELETE FROM pending_tool_approvals WHERE user_id=?", (user_id,)
            )
            await self._db.commit()

    # ─────────────────────────────────────────────────────────────
    # HITL State Persistence  (survive bot restarts)
    # ─────────────────────────────────────────────────────────────

    async def save_hitl_state(self, user_id: str, state: dict) -> None:
        """Persist a HITL state dict so it survives a bot restart."""
        import json as _json
        try:
            state_json = _json.dumps(state)
        except (TypeError, ValueError) as e:
            logger.warning(f"HITL state for '{user_id}' is not JSON-serializable, skipping persistence: {e}")
            return
        async with self._lock:
            await self._db.execute(
                "INSERT INTO pending_hitl_states (user_id, state_json) "
                "VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET "
                "state_json=excluded.state_json, created_at=CURRENT_TIMESTAMP",
                (user_id, state_json),
            )
            await self._db.commit()

    async def load_hitl_states(self) -> dict:
        """Return {user_id: state_dict} for all persisted HITL states."""
        import json as _json
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT user_id, state_json FROM pending_hitl_states"
            )
            rows = await cursor.fetchall()
        result = {}
        for row in rows:
            try:
                result[row["user_id"]] = _json.loads(row["state_json"])
            except Exception as e:
                logger.warning(f"Could not deserialize HITL state for user '{row['user_id']}': {e}")
        return result

    async def clear_hitl_state(self, user_id: str) -> None:
        """Remove a persisted HITL state after it is resumed or abandoned."""
        async with self._lock:
            await self._db.execute(
                "DELETE FROM pending_hitl_states WHERE user_id=?", (user_id,)
            )
            await self._db.commit()

    # ─────────────────────────────────────────────────────────────
    # System State  (generic key-value store for runtime flags)
    # ─────────────────────────────────────────────────────────────

    async def set_system_state(self, key: str, value: str) -> None:
        """Upsert a key-value pair in the system_state table."""
        async with self._lock:
            await self._db.execute(
                "INSERT INTO system_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP",
                (key, value),
            )
            await self._db.commit()

    async def get_system_state(self, key: str) -> Optional[str]:
        """Retrieve a value from the system_state table, or None if absent."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT value FROM system_state WHERE key=?", (key,)
            )
            row = await cursor.fetchone()
        return row["value"] if row else None

    # ─────────────────────────────────────────────────────────────
    # MFA State  (persisted so bot restarts don't lose pending MFA)
    # ─────────────────────────────────────────────────────────────

    async def save_mfa_state(self, user_id: str, tool_name: str, arguments: dict) -> None:
        """Persist a pending MFA challenge to survive bot restarts."""
        import json as _json
        async with self._lock:
            await self._db.execute(
                "INSERT INTO pending_mfa_states (user_id, tool_name, arguments_json) "
                "VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
                "tool_name=excluded.tool_name, arguments_json=excluded.arguments_json, "
                "created_at=CURRENT_TIMESTAMP",
                (user_id, tool_name, _json.dumps(arguments)),
            )
            await self._db.commit()

    async def load_mfa_states(self) -> dict:
        """Return {user_id: {name, arguments, _created_at}} for all persisted MFA states."""
        import json as _json
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT user_id, tool_name, arguments_json, created_at FROM pending_mfa_states"
            )
            rows = await cursor.fetchall()
        result = {}
        for row in rows:
            try:
                result[row["user_id"]] = {
                    "name": row["tool_name"],
                    "arguments": _json.loads(row["arguments_json"]),
                    "_created_at": row["created_at"],
                }
            except Exception as e:
                logger.warning("Could not deserialize MFA state for '%s': %s", row["user_id"], e)
        return result

    async def clear_mfa_state(self, user_id: str) -> None:
        """Remove a persisted MFA state after it is resolved or expired."""
        async with self._lock:
            await self._db.execute(
                "DELETE FROM pending_mfa_states WHERE user_id=?", (user_id,)
            )
            await self._db.commit()
