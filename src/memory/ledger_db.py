"""
Ledger Database Module for Short-Term Memory and Task Management.

This module provides structured storage for operational state and system events
using SQLite via the non-blocking ``aiosqlite`` driver.  All public methods are
async-safe and concurrency-safe through a shared ``asyncio.Lock``.
"""

import os
import asyncio
import logging
import json
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
_SQL_TEXT_DEFAULT_EMPTY = "TEXT NOT NULL DEFAULT ''"
_SQL_INT_DEFAULT_ZERO = "INTEGER NOT NULL DEFAULT 0"
_MAX_PENDING_TTL_SECONDS = 31_536_000


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

    # Lifecycle

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
                CREATE TABLE IF NOT EXISTS supervisor_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    user_input TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    is_direct INTEGER NOT NULL DEFAULT 0,
                    reasoning TEXT NOT NULL DEFAULT '',
                    energy_before INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                CREATE INDEX IF NOT EXISTS idx_supervisor_decisions_user_created
                ON supervisor_decisions(user_id, created_at DESC)
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
                    depends_on_ids TEXT NOT NULL DEFAULT '[]',
                    acceptance_criteria TEXT NOT NULL DEFAULT '',
                    defer_count INTEGER NOT NULL DEFAULT 0,
                    last_energy_eval_at TIMESTAMP,
                    next_eligible_at TIMESTAMP,
                    last_energy_eval_json TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_id) REFERENCES objective_backlog(id)
                )
            """)
            await self._ensure_objective_backlog_column(
                "depends_on_ids",
                "TEXT NOT NULL DEFAULT '[]'",
            )
            await self._ensure_objective_backlog_column(
                "acceptance_criteria",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._ensure_objective_backlog_column(
                "defer_count",
                _SQL_INT_DEFAULT_ZERO,
            )
            await self._ensure_objective_backlog_column(
                "last_energy_eval_at",
                "TIMESTAMP",
            )
            await self._ensure_objective_backlog_column(
                "next_eligible_at",
                "TIMESTAMP",
            )
            await self._ensure_objective_backlog_column(
                "last_energy_eval_json",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._ensure_objective_backlog_column(
                "agent_domain",
                "TEXT",
            )
            await self._ensure_objective_backlog_column(
                "claimed_by",
                "TEXT",
            )
            await self._ensure_objective_backlog_column(
                "claimed_at",
                "TIMESTAMP",
            )
            await self._ensure_objective_backlog_column(
                "result_json",
                "TEXT",
            )
            await self._ensure_objective_backlog_column(
                "result_written_at",
                "TIMESTAMP",
            )
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_objective_status ON objective_backlog(status)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_objective_tier ON objective_backlog(tier)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_objective_parent ON objective_backlog(parent_id)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_objective_next_eligible ON objective_backlog(next_eligible_at)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_objective_blackboard_pending
                ON objective_backlog(agent_domain, status, claimed_by, next_eligible_at)
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    is_active INTEGER NOT NULL DEFAULT 0,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_is_active ON sessions(is_active DESC, id DESC)
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    session_id INTEGER REFERENCES sessions(id),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._ensure_table_column(
                "chat_history",
                "session_id",
                "INTEGER REFERENCES sessions(id)",
            )
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
                CREATE TABLE IF NOT EXISTS cloud_payload_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    purpose TEXT NOT NULL,
                    message_count_before INTEGER NOT NULL DEFAULT 0,
                    message_count_after INTEGER NOT NULL DEFAULT 0,
                    allow_sensitive_context INTEGER NOT NULL DEFAULT 0,
                    payload_sha256 TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_cloud_payload_audit_created
                ON cloud_payload_audit(created_at DESC)
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS pending_mfa_states (
                    user_id TEXT NOT NULL PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS moral_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    audit_mode TEXT NOT NULL,
                    audit_trace TEXT NOT NULL,
                    critic_feedback TEXT NOT NULL,
                    moral_decision_json TEXT NOT NULL,
                    request_redacted TEXT NOT NULL,
                    output_redacted TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_moral_audit_user_id_created_at
                ON moral_audit_log(user_id, created_at DESC)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_moral_audit_created_at
                ON moral_audit_log(created_at DESC)
            """)
            await self._db.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_moral_audit_log_no_update
                BEFORE UPDATE ON moral_audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'Update not allowed on moral_audit_log');
                END;
            """)
            await self._db.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_moral_audit_log_no_delete
                BEFORE DELETE ON moral_audit_log
                BEGIN
                    SELECT RAISE(ABORT, 'Delete not allowed on moral_audit_log');
                END;
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS synthesis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    gap_description TEXT NOT NULL,
                    suggested_tool_name TEXT NOT NULL,
                    original_input TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'in_progress',
                    final_tool_name TEXT NOT NULL DEFAULT '',
                    total_attempts INTEGER NOT NULL DEFAULT 0,
                    successful_attempt INTEGER,
                    max_retries INTEGER NOT NULL DEFAULT 3,
                    code_sha256 TEXT NOT NULL DEFAULT '',
                    test_summary TEXT NOT NULL DEFAULT '',
                    blocked_reason TEXT NOT NULL DEFAULT '',
                    synthesis_json TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await self._ensure_table_column(
                "synthesis_runs",
                "status",
                "TEXT NOT NULL DEFAULT 'in_progress'",
            )
            await self._ensure_table_column(
                "synthesis_runs",
                "final_tool_name",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._ensure_table_column(
                "synthesis_runs",
                "total_attempts",
                _SQL_INT_DEFAULT_ZERO,
            )
            await self._ensure_table_column(
                "synthesis_runs",
                "successful_attempt",
                "INTEGER",
            )
            await self._ensure_table_column(
                "synthesis_runs",
                "max_retries",
                "INTEGER NOT NULL DEFAULT 3",
            )
            await self._ensure_table_column(
                "synthesis_runs",
                "code_sha256",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._ensure_table_column(
                "synthesis_runs",
                "test_summary",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._ensure_table_column(
                "synthesis_runs",
                "blocked_reason",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._ensure_table_column(
                "synthesis_runs",
                "synthesis_json",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_synthesis_runs_user_created
                ON synthesis_runs(user_id, created_at DESC)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_synthesis_runs_status_created
                ON synthesis_runs(status, created_at DESC)
            """)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS synthesis_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    phase TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timed_out INTEGER NOT NULL DEFAULT 0,
                    exit_code INTEGER,
                    duration_ms INTEGER NOT NULL DEFAULT 0,
                    passed_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    skipped_count INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT NOT NULL DEFAULT '',
                    stdout_text TEXT NOT NULL DEFAULT '',
                    stderr_text TEXT NOT NULL DEFAULT '',
                    code_sha256 TEXT NOT NULL DEFAULT '',
                    synthesis_json TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES synthesis_runs(id),
                    UNIQUE(run_id, attempt_number)
                )
            """)
            await self._ensure_table_column(
                "synthesis_attempts",
                "phase",
                "TEXT NOT NULL DEFAULT 'synthesis'",
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "tool_name",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "status",
                "TEXT NOT NULL DEFAULT 'failed'",
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "timed_out",
                _SQL_INT_DEFAULT_ZERO,
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "exit_code",
                "INTEGER",
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "duration_ms",
                _SQL_INT_DEFAULT_ZERO,
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "passed_count",
                _SQL_INT_DEFAULT_ZERO,
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "failed_count",
                _SQL_INT_DEFAULT_ZERO,
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "error_count",
                _SQL_INT_DEFAULT_ZERO,
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "skipped_count",
                _SQL_INT_DEFAULT_ZERO,
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "error_message",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "stdout_text",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "stderr_text",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "code_sha256",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._ensure_table_column(
                "synthesis_attempts",
                "synthesis_json",
                _SQL_TEXT_DEFAULT_EMPTY,
            )
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_synthesis_attempts_run_attempt
                ON synthesis_attempts(run_id, attempt_number ASC)
            """)
            await self._db.execute("""
                CREATE INDEX IF NOT EXISTS idx_synthesis_attempts_status_created
                ON synthesis_attempts(status, created_at DESC)
            """)
            await self._db.commit()
        logger.debug("Database tables initialized")

    async def _ensure_table_column(self, table_name: str, column_name: str, column_sql: str) -> None:
        """Idempotent schema migration helper for adding a missing table column."""
        cursor = await self._db.execute(f"PRAGMA table_info({table_name})")
        rows = await cursor.fetchall()
        existing = {row["name"] for row in rows}
        if column_name in existing:
            return

        await self._db.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"
        )
        logger.info("Migrated %s: added column '%s'", table_name, column_name)

    async def _ensure_objective_backlog_column(self, column_name: str, column_sql: str) -> None:
        """Idempotent schema migration helper for objective_backlog columns."""
        await self._ensure_table_column("objective_backlog", column_name, column_sql)

    @staticmethod
    def _serialize_depends_on_ids(depends_on_ids: Optional[List[int]]) -> str:
        if not depends_on_ids:
            return "[]"

        normalized: List[int] = []
        for raw_id in depends_on_ids:
            try:
                dep_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if dep_id > 0 and dep_id not in normalized:
                normalized.append(dep_id)
        return json.dumps(normalized)

    @staticmethod
    def _deserialize_depends_on_ids(raw_value: Any) -> List[int]:
        if raw_value is None:
            return []
        if isinstance(raw_value, list):
            result: List[int] = []
            for item in raw_value:
                try:
                    dep_id = int(item)
                except (TypeError, ValueError):
                    continue
                if dep_id > 0 and dep_id not in result:
                    result.append(dep_id)
            return result

        raw_text = str(raw_value).strip()
        if not raw_text:
            return []

        try:
            decoded = json.loads(raw_text)
            if isinstance(decoded, list):
                return LedgerMemory._deserialize_depends_on_ids(decoded)
        except (TypeError, ValueError):
            pass

        # Fallback for legacy comma-separated values.
        parts = [part.strip() for part in raw_text.split(",") if part.strip()]
        return LedgerMemory._deserialize_depends_on_ids(parts)

    @classmethod
    def _normalize_objective_row(cls, row: aiosqlite.Row) -> Dict[str, Any]:
        result = dict(row)
        result["depends_on_ids"] = cls._deserialize_depends_on_ids(result.get("depends_on_ids"))
        result["acceptance_criteria"] = str(result.get("acceptance_criteria") or "")
        result["defer_count"] = int(result.get("defer_count") or 0)
        result["last_energy_eval_at"] = result.get("last_energy_eval_at")
        result["next_eligible_at"] = result.get("next_eligible_at")
        result["last_energy_eval_json"] = str(result.get("last_energy_eval_json") or "")
        return result

    @classmethod
    def _normalize_energy_context_row(cls, row: aiosqlite.Row) -> Dict[str, Any]:
        task_row = {
            "id": row["task_id"],
            "tier": "Task",
            "title": row["task_title"],
            "status": row["task_status"],
            "priority": row["task_priority"],
            "estimated_energy": row["task_estimated_energy"],
            "origin": row["task_origin"],
            "parent_id": row["task_parent_id"],
            "depends_on_ids": row["task_depends_on_ids"],
            "acceptance_criteria": row["task_acceptance_criteria"],
            "defer_count": row["task_defer_count"],
            "last_energy_eval_at": row["task_last_energy_eval_at"],
            "next_eligible_at": row["task_next_eligible_at"],
            "last_energy_eval_json": row["task_last_energy_eval_json"],
            "created_at": row["task_created_at"],
            "updated_at": row["task_updated_at"],
        }
        normalized_task = cls._normalize_objective_row(task_row)
        normalized_task["id"] = int(normalized_task["id"])
        normalized_task["priority"] = int(normalized_task.get("priority") or 0)
        normalized_task["estimated_energy"] = int(normalized_task.get("estimated_energy") or 0)

        story = None
        if row["story_id"] is not None:
            story = {
                "id": int(row["story_id"]),
                "title": str(row["story_title"] or ""),
                "status": str(row["story_status"] or ""),
                "acceptance_criteria": str(row["story_acceptance_criteria"] or ""),
            }

        epic = None
        if row["epic_id"] is not None:
            epic = {
                "id": int(row["epic_id"]),
                "title": str(row["epic_title"] or ""),
                "status": str(row["epic_status"] or ""),
                "acceptance_criteria": str(row["epic_acceptance_criteria"] or ""),
            }

        return {
            "task": normalized_task,
            "story": story,
            "epic": epic,
        }

    @staticmethod
    def _objective_energy_context_query(where_clause: str) -> str:
        return (
            "SELECT "
            "task.id AS task_id, task.title AS task_title, task.status AS task_status, "
            "task.priority AS task_priority, task.estimated_energy AS task_estimated_energy, "
            "task.origin AS task_origin, task.parent_id AS task_parent_id, "
            "task.depends_on_ids AS task_depends_on_ids, "
            "task.acceptance_criteria AS task_acceptance_criteria, "
            "task.defer_count AS task_defer_count, "
            "task.last_energy_eval_at AS task_last_energy_eval_at, "
            "task.next_eligible_at AS task_next_eligible_at, "
            "task.last_energy_eval_json AS task_last_energy_eval_json, "
            "task.created_at AS task_created_at, task.updated_at AS task_updated_at, "
            "story.id AS story_id, story.title AS story_title, story.status AS story_status, "
            "story.acceptance_criteria AS story_acceptance_criteria, "
            "epic.id AS epic_id, epic.title AS epic_title, epic.status AS epic_status, "
            "epic.acceptance_criteria AS epic_acceptance_criteria "
            "FROM objective_backlog task "
            "LEFT JOIN objective_backlog story ON task.parent_id = story.id AND story.tier = 'Story' "
            "LEFT JOIN objective_backlog epic ON story.parent_id = epic.id AND epic.tier = 'Epic' "
            "WHERE " + where_clause
        )

    @staticmethod
    def _build_children_index(nodes: Dict[int, Dict[str, Any]]) -> Dict[int, List[int]]:
        children: Dict[int, List[int]] = {}
        for node_id, node in nodes.items():
            parent_id = node.get("parent_id")
            if parent_id is None:
                continue
            children.setdefault(int(parent_id), []).append(node_id)
        return children

    @classmethod
    def _aggregate_task_totals(
        cls,
        node_id: int,
        nodes: Dict[int, Dict[str, Any]],
        children: Dict[int, List[int]],
        memo: Dict[int, Dict[str, int]],
        visiting: Optional[set] = None,
    ) -> Dict[str, int]:
        if node_id in memo:
            return memo[node_id]

        visiting = visiting or set()
        if node_id in visiting:
            return cls._empty_task_totals()
        visiting.add(node_id)

        node = nodes[node_id]
        tier = str(node.get("tier") or "")
        status = str(node.get("status") or "")
        if tier == "Task":
            totals = cls._task_leaf_totals(status)
        else:
            totals = cls._sum_child_task_totals(node_id, nodes, children, memo, visiting)

        memo[node_id] = totals
        visiting.remove(node_id)
        return totals

    @staticmethod
    def _empty_task_totals() -> Dict[str, int]:
        return {"total": 0, "completed": 0, "pending": 0, "active": 0, "suspended": 0}

    @classmethod
    def _task_leaf_totals(cls, status: str) -> Dict[str, int]:
        totals = cls._empty_task_totals()
        totals["total"] = 1
        if status == "completed":
            totals["completed"] = 1
        elif status == "pending":
            totals["pending"] = 1
        elif status == "active":
            totals["active"] = 1
        elif status == "suspended":
            totals["suspended"] = 1
        return totals

    @classmethod
    def _sum_child_task_totals(
        cls,
        node_id: int,
        nodes: Dict[int, Dict[str, Any]],
        children: Dict[int, List[int]],
        memo: Dict[int, Dict[str, int]],
        visiting: set,
    ) -> Dict[str, int]:
        totals = cls._empty_task_totals()
        for child_id in children.get(node_id, []):
            child_totals = cls._aggregate_task_totals(child_id, nodes, children, memo, visiting)
            for key in totals:
                totals[key] += child_totals[key]
        return totals

    @staticmethod
    def _derive_rollup_status(
        fallback_status: str,
        total_tasks: int,
        completed_tasks: int,
        pending_tasks: int,
        active_tasks: int,
        suspended_tasks: int,
    ) -> str:
        if total_tasks == 0:
            return fallback_status or "pending"
        if completed_tasks == total_tasks:
            return "completed"
        if active_tasks > 0:
            return "active"
        if pending_tasks > 0:
            return "pending"
        if suspended_tasks == total_tasks:
            return "suspended"
        return fallback_status or "pending"

    async def close(self) -> None:
        """Close the aiosqlite connection."""
        try:
            if self._db is not None:
                await self._db.close()
                self._db = None
                logger.info("LedgerMemory connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

    @staticmethod
    def _pending_ttl_seconds() -> int:
        try:
            return max(0, int(os.getenv("PENDING_STATE_TTL_SECONDS", "86400")))
        except ValueError:
            return 86400

    async def purge_expired_pending(self, ttl_seconds: Optional[int] = None) -> Dict[str, int]:
        """Delete expired pending MFA/HITL/tool-approval rows and return per-table counts."""
        ttl = self._pending_ttl_seconds() if ttl_seconds is None else max(0, int(ttl_seconds))
        # Cap persisted pending-state TTLs at one year to avoid pathological
        # SQLite datetime intervals while still allowing long local outages.
        ttl = min(ttl, _MAX_PENDING_TTL_SECONDS)
        modifier = f"-{ttl} seconds"
        deleted: Dict[str, int] = {}

        async with self._lock:
            cursor = await self._db.execute(
                "DELETE FROM pending_tool_approvals WHERE datetime(created_at) < datetime('now', ?)",
                (modifier,),
            )
            deleted["pending_tool_approvals"] = int(cursor.rowcount or 0)
            cursor = await self._db.execute(
                "DELETE FROM pending_hitl_states WHERE datetime(created_at) < datetime('now', ?)",
                (modifier,),
            )
            deleted["pending_hitl_states"] = int(cursor.rowcount or 0)
            cursor = await self._db.execute(
                "DELETE FROM pending_mfa_states WHERE datetime(created_at) < datetime('now', ?)",
                (modifier,),
            )
            deleted["pending_mfa_states"] = int(cursor.rowcount or 0)
            await self._db.commit()

        return deleted

    async def append_cloud_payload_audit(
        self,
        *,
        purpose: str,
        message_count_before: int,
        message_count_after: int,
        allow_sensitive_context: bool,
        payload_sha256: str,
    ) -> int:
        """Append an immutable audit record for a payload sent to a cloud LLM."""
        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO cloud_payload_audit "
                "(purpose, message_count_before, message_count_after, allow_sensitive_context, payload_sha256) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(purpose or "system_2"),
                    max(0, int(message_count_before)),
                    max(0, int(message_count_after)),
                    1 if allow_sensitive_context else 0,
                    str(payload_sha256 or ""),
                ),
            )
            await self._db.commit()
            return int(cursor.lastrowid)

    async def get_cloud_payload_audit_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent cloud payload audit entries."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, purpose, message_count_before, message_count_after, "
                "allow_sensitive_context, payload_sha256, created_at "
                "FROM cloud_payload_audit ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            )
            rows = await cursor.fetchall()
        entries = [dict(row) for row in rows]
        for entry in entries:
            entry["allow_sensitive_context"] = bool(entry.get("allow_sensitive_context"))
        return entries

    async def log_supervisor_decision(
        self,
        *,
        user_id: Any,
        user_input: Any,
        plan_json: Any,
        is_direct: bool,
        reasoning: Any,
        energy_before: Any,
    ) -> int:
        """Best-effort append of a supervisor routing decision."""
        try:
            safe_user_id = str(user_id or "")[:128]
            safe_input = str(user_input or "")[:500]
            safe_plan = str(plan_json or "[]")[:4000]
            safe_reasoning = str(reasoning or "")[:1000]
            try:
                safe_energy = max(0, int(energy_before))
            except (TypeError, ValueError):
                safe_energy = 0
            async with self._lock:
                cursor = await self._db.execute(
                    "INSERT INTO supervisor_decisions "
                    "(user_id, user_input, plan_json, is_direct, reasoning, energy_before) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (safe_user_id, safe_input, safe_plan, 1 if is_direct else 0, safe_reasoning, safe_energy),
                )
                await self._db.commit()
                return int(cursor.lastrowid)
        except Exception as e:
            logger.warning("Failed to log supervisor decision: %s", e)
            return 0

    async def get_recent_supervisor_decisions(
        self,
        user_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        query = (
            "SELECT id, user_id, user_input, plan_json, is_direct, reasoning, energy_before, created_at "
            "FROM supervisor_decisions"
        )
        params: List[Any] = []
        if user_id is not None:
            query += " WHERE user_id = ?"
            params.append(str(user_id))
        query += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        async with self._lock:
            cursor = await self._db.execute(query, tuple(params))
            rows = await cursor.fetchall()
        decisions = [dict(row) for row in rows]
        for decision in decisions:
            decision["is_direct"] = bool(decision.get("is_direct"))
        return decisions

    async def get_last_supervisor_decision(self, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        decisions = await self.get_recent_supervisor_decisions(user_id=user_id, limit=1)
        return decisions[0] if decisions else None

    async def count_supervisor_decisions(self, user_id: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) AS count FROM supervisor_decisions"
        params: List[Any] = []
        if user_id is not None:
            query += " WHERE user_id = ?"
            params.append(str(user_id))
        async with self._lock:
            cursor = await self._db.execute(query, tuple(params))
            row = await cursor.fetchone()
        return int(row["count"] if row is not None else 0)

    async def get_deferred_tasks_with_energy_context(self, limit: int = 20) -> List[Dict[str, Any]]:
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, tier, title, status, priority, estimated_energy, defer_count, "
                "last_energy_eval_at, next_eligible_at, last_energy_eval_json "
                "FROM objective_backlog WHERE status IN ('deferred', 'blocked', 'suspended') "
                "ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_blocked_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, tier, title, status, priority, estimated_energy, parent_id, defer_count, "
                "last_energy_eval_at, next_eligible_at, last_energy_eval_json "
                "FROM objective_backlog WHERE status IN ('blocked', 'suspended') "
                "ORDER BY updated_at DESC LIMIT ?",
                (max(1, int(limit)),),
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_system_error_log(self, hours: int = 24, limit: int = 50) -> List[Dict[str, Any]]:
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, log_level, message, timestamp, context FROM system_logs "
                "WHERE log_level IN ('WARNING', 'ERROR', 'CRITICAL') "
                "AND timestamp >= datetime('now', ?) "
                "ORDER BY id DESC LIMIT ?",
                (f"-{max(1, int(hours))} hours", max(1, int(limit))),
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_moral_audit_summary(self, limit: int = 20) -> List[Dict[str, Any]]:
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, user_id, audit_mode, critic_feedback, moral_decision_json, created_at "
                "FROM moral_audit_log ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_recent_synthesis_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, user_id, gap_description, suggested_tool_name, original_input, status, "
                "final_tool_name, total_attempts, successful_attempt, max_retries, code_sha256, "
                "test_summary, blocked_reason, created_at, updated_at "
                "FROM synthesis_runs ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_objective_counts_by_status(self) -> Dict[str, int]:
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT status, COUNT(*) AS count FROM objective_backlog GROUP BY status"
            )
            rows = await cursor.fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

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

    async def append_moral_audit_log(
        self,
        *,
        user_id: str,
        audit_mode: str,
        audit_trace: str,
        critic_feedback: str,
        moral_decision: Dict[str, Any],
        request_redacted: str,
        output_redacted: str,
    ) -> int:
        """Append one immutable moral audit record and return its ID."""
        decision_json = json.dumps(dict(moral_decision or {}), sort_keys=True)
        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO moral_audit_log "
                "(user_id, audit_mode, audit_trace, critic_feedback, moral_decision_json, request_redacted, output_redacted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(user_id or ""),
                    str(audit_mode or ""),
                    str(audit_trace or ""),
                    str(critic_feedback or ""),
                    decision_json,
                    str(request_redacted or ""),
                    str(output_redacted or ""),
                ),
            )
            await self._db.commit()
            return int(cursor.lastrowid)

    async def get_moral_audit_logs(
        self,
        *,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Read immutable moral audit records, most recent first."""
        async with self._lock:
            if user_id is not None:
                cursor = await self._db.execute(
                    "SELECT id, user_id, audit_mode, audit_trace, critic_feedback, moral_decision_json, "
                    "request_redacted, output_redacted, created_at "
                    "FROM moral_audit_log WHERE user_id = ? "
                    "ORDER BY created_at DESC, id DESC LIMIT ?",
                    (str(user_id), int(limit)),
                )
            else:
                cursor = await self._db.execute(
                    "SELECT id, user_id, audit_mode, audit_trace, critic_feedback, moral_decision_json, "
                    "request_redacted, output_redacted, created_at "
                    "FROM moral_audit_log "
                    "ORDER BY created_at DESC, id DESC LIMIT ?",
                    (int(limit),),
                )
            rows = await cursor.fetchall()

        logs: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["moral_decision"] = json.loads(str(item.get("moral_decision_json") or "{}"))
            except Exception:
                item["moral_decision"] = {}
            logs.append(item)
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
        depends_on_ids: Optional[List[int]] = None,
        acceptance_criteria: str = "",
    ) -> int:
        """Insert a new Epic/Story/Task into the objective_backlog."""
        if tier not in ("Epic", "Story", "Task"):
            raise ValueError("tier must be Epic, Story, or Task")
        depends_blob = self._serialize_depends_on_ids(depends_on_ids)
        acceptance_text = str(acceptance_criteria or "").strip()
        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO objective_backlog "
                "(tier, title, status, priority, estimated_energy, origin, parent_id, depends_on_ids, acceptance_criteria) "
                "VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?)",
                (
                    tier,
                    title,
                    priority,
                    estimated_energy,
                    origin,
                    parent_id,
                    depends_blob,
                    acceptance_text,
                ),
            )
            await self._db.commit()
            obj_id = cursor.lastrowid
        logger.info(f"Objective added: [{tier}] {title!r} (id={obj_id})")
        return obj_id

    async def get_highest_priority_task(self) -> Optional[dict]:
        """Return the single highest-priority currently eligible pending Task, or None."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, tier, title, status, priority, estimated_energy, "
                "origin, parent_id, depends_on_ids, acceptance_criteria, "
                "defer_count, last_energy_eval_at, next_eligible_at, last_energy_eval_json, "
                "created_at, updated_at "
                "FROM objective_backlog "
                "WHERE tier = 'Task' AND status = 'pending' "
                "AND (next_eligible_at IS NULL OR datetime(next_eligible_at) <= CURRENT_TIMESTAMP) "
                "ORDER BY priority ASC, estimated_energy ASC LIMIT 1"
            )
            row = await cursor.fetchone()
        return self._normalize_objective_row(row) if row else None

    async def claim_task(self, task_id: int, agent_name: str) -> bool:
        """Atomically claim a pending Task; returns False if already claimed by another agent."""
        owner = str(agent_name or "").strip()
        if not owner:
            raise ValueError("agent_name must be a non-empty string")
        row = None

        async with self._lock:
            tx_started = False
            try:
                await self._db.execute("BEGIN IMMEDIATE")
                tx_started = True
                cursor = await self._db.execute(
                    "UPDATE objective_backlog "
                    "SET claimed_by = ?, claimed_at = CURRENT_TIMESTAMP, status = 'active', "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = ? AND tier = 'Task' AND status = 'pending' AND claimed_by IS NULL "
                    "AND (next_eligible_at IS NULL OR datetime(next_eligible_at) <= CURRENT_TIMESTAMP)",
                    (owner, task_id),
                )
                if bool(cursor.rowcount and cursor.rowcount > 0):
                    await self._db.commit()
                    return True

                row_cursor = await self._db.execute(
                    "SELECT claimed_by FROM objective_backlog WHERE id = ? AND tier = 'Task'",
                    (task_id,),
                )
                row = await row_cursor.fetchone()
                await self._db.commit()
            except Exception:
                if tx_started:
                    await self._db.rollback()
                raise

        if row is None:
            return False
        claimed_by = row["claimed_by"]
        if claimed_by is None:
            return False
        return str(claimed_by) == owner

    async def write_task_result(self, task_id: int, result: dict) -> None:
        """Persist task result payload and mark the Task as completed."""
        if not isinstance(result, dict):
            raise ValueError("result must be a dict")
        result_payload = dict(result)
        result_json = json.dumps(result_payload, sort_keys=True)
        updated = False
        async with self._lock:
            cursor = await self._db.execute(
                "UPDATE objective_backlog "
                "SET result_json = ?, result_written_at = CURRENT_TIMESTAMP, "
                "status = 'completed', updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND tier = 'Task'",
                (result_json, task_id),
            )
            updated = bool(cursor.rowcount and cursor.rowcount > 0)
            await self._db.commit()

        if updated:
            await self.reduce_goal_state_from_task_completion(task_id)

    async def write_task_failure(self, task_id: int, result: dict) -> None:
        """Persist task failure payload and mark the Task as failed.

        Args:
            task_id: Objective backlog Task id.
            result: Failure payload to serialize into ``result_json``.
        """
        if not isinstance(result, dict):
            raise ValueError("result must be a dict")
        result_json = json.dumps(result, sort_keys=True)
        async with self._lock:
            await self._db.execute(
                "UPDATE objective_backlog "
                "SET result_json = ?, result_written_at = CURRENT_TIMESTAMP, "
                "status = 'failed', updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND tier = 'Task'",
                (result_json, task_id),
            )
            await self._db.commit()

    async def set_task_agent_domain(self, task_id: int, agent_domain: str) -> None:
        """Assign or update the agent_domain for a Task row.

        Args:
            task_id: Objective backlog Task id.
            agent_domain: Domain label used by domain agents for filtering.
        """
        async with self._lock:
            await self._db.execute(
                "UPDATE objective_backlog "
                "SET agent_domain = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND tier = 'Task'",
                (str(agent_domain or "").strip(), task_id),
            )
            await self._db.commit()

    async def get_task_row(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Return a normalized Task row by id, or None when not found.

        Args:
            task_id: Objective backlog Task id.

        Returns:
            A normalized task dict containing status, claim, and result fields,
            or ``None`` when no matching Task exists.
        """
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, tier, title, status, priority, estimated_energy, "
                "origin, parent_id, depends_on_ids, acceptance_criteria, "
                "defer_count, last_energy_eval_at, next_eligible_at, last_energy_eval_json, "
                "agent_domain, claimed_by, claimed_at, result_json, result_written_at, "
                "created_at, updated_at "
                "FROM objective_backlog "
                "WHERE id = ? AND tier = 'Task' "
                "LIMIT 1",
                (task_id,),
            )
            row = await cursor.fetchone()
        return self._normalize_objective_row(row) if row else None

    async def get_pending_tasks_for_domain(self, agent_domain: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Return unclaimed pending Tasks for a domain, respecting next_eligible_at."""
        domain = str(agent_domain or "").strip()
        if not domain:
            return []
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, tier, title, status, priority, estimated_energy, "
                "origin, parent_id, depends_on_ids, acceptance_criteria, "
                "defer_count, last_energy_eval_at, next_eligible_at, last_energy_eval_json, "
                "agent_domain, claimed_by, claimed_at, result_json, result_written_at, "
                "created_at, updated_at "
                "FROM objective_backlog "
                "WHERE tier = 'Task' AND status = 'pending' AND claimed_by IS NULL "
                "AND agent_domain = ? "
                "AND (next_eligible_at IS NULL OR datetime(next_eligible_at) <= CURRENT_TIMESTAMP) "
                "ORDER BY priority ASC, estimated_energy ASC, id ASC "
                "LIMIT ?",
                (domain, max(1, int(limit))),
            )
            rows = await cursor.fetchall()
        return [self._normalize_objective_row(row) for row in rows]

    async def get_task_result(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Return parsed task result payload, or None when absent/invalid."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT result_json FROM objective_backlog WHERE id = ? AND tier = 'Task'",
                (task_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        raw_result = row["result_json"]
        if raw_result is None or str(raw_result).strip() == "":
            return None
        try:
            parsed = json.loads(str(raw_result))
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None

    async def update_objective_status(self, obj_id: int, new_status: str) -> None:
        """Update status of an objective.

        Valid states: pending, active, completed, suspended, blocked,
        deferred_due_to_energy.
        """
        valid = {
            "pending",
            "active",
            "completed",
            "suspended",
            "blocked",
            "deferred_due_to_energy",
        }
        if new_status not in valid:
            raise ValueError(f"status must be one of {valid}")

        tier = ""
        async with self._lock:
            row_cursor = await self._db.execute(
                "SELECT tier FROM objective_backlog WHERE id = ?",
                (obj_id,),
            )
            row = await row_cursor.fetchone()
            if row is not None:
                tier = str(row["tier"] or "")

            await self._db.execute(
                "UPDATE objective_backlog SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (new_status, obj_id),
            )
            await self._db.commit()
        logger.info(f"Objective {obj_id} status -> {new_status}")

        # GoalStateReducer: when a Task completes, evaluate Story/Epic completion bottom-up.
        if tier == "Task" and new_status == "completed":
            await self.reduce_goal_state_from_task_completion(obj_id)

    async def record_task_energy_evaluation(
        self,
        task_id: int,
        evaluation_payload: Dict[str, Any],
        *,
        clear_next_eligible: bool = False,
    ) -> None:
        """Persist latest energy evaluation metadata for a Task."""
        payload_json = json.dumps(dict(evaluation_payload or {}), sort_keys=True)
        async with self._lock:
            if clear_next_eligible:
                await self._db.execute(
                    "UPDATE objective_backlog "
                    "SET last_energy_eval_at = CURRENT_TIMESTAMP, "
                    "last_energy_eval_json = ?, "
                    "next_eligible_at = NULL, "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = ? AND tier = 'Task'",
                    (payload_json, task_id),
                )
            else:
                await self._db.execute(
                    "UPDATE objective_backlog "
                    "SET last_energy_eval_at = CURRENT_TIMESTAMP, "
                    "last_energy_eval_json = ?, "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = ? AND tier = 'Task'",
                    (payload_json, task_id),
                )
            await self._db.commit()

    async def defer_task_due_to_energy(
        self,
        task_id: int,
        evaluation_payload: Dict[str, Any],
        *,
        cooldown_seconds: int,
    ) -> None:
        """Mark Task as deferred_due_to_energy and schedule next eligibility."""
        payload_json = json.dumps(dict(evaluation_payload or {}), sort_keys=True)
        cooldown = max(0, int(cooldown_seconds))
        cooldown_sql = f"+{cooldown} seconds"

        async with self._lock:
            await self._db.execute(
                "UPDATE objective_backlog "
                "SET status = 'deferred_due_to_energy', "
                "defer_count = COALESCE(defer_count, 0) + 1, "
                "last_energy_eval_at = CURRENT_TIMESTAMP, "
                "next_eligible_at = datetime(CURRENT_TIMESTAMP, ?), "
                "last_energy_eval_json = ?, "
                "updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND tier = 'Task'",
                (cooldown_sql, payload_json, task_id),
            )
            await self._db.commit()

    async def reduce_goal_state_from_task_completion(self, task_id: int) -> None:
        """Bottom-up reducer: Task completion may complete parent Story and parent Epic."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT parent_id FROM objective_backlog WHERE id = ? AND tier = 'Task'",
                (task_id,),
            )
            task_row = await cursor.fetchone()
            if task_row is None:
                return

            story_id = task_row["parent_id"]
            if story_id is None:
                return

            child_task_cursor = await self._db.execute(
                "SELECT status FROM objective_backlog WHERE tier='Task' AND parent_id = ?",
                (story_id,),
            )
            child_task_rows = await child_task_cursor.fetchall()
            if not child_task_rows:
                return

            story_complete = all(str(row["status"]) == "completed" for row in child_task_rows)
            if not story_complete:
                return

            await self._db.execute(
                "UPDATE objective_backlog "
                "SET status = 'completed', updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ?",
                (story_id,),
            )

            story_cursor = await self._db.execute(
                "SELECT parent_id FROM objective_backlog WHERE id = ? AND tier = 'Story'",
                (story_id,),
            )
            story_row = await story_cursor.fetchone()
            epic_id = story_row["parent_id"] if story_row else None

            if epic_id is not None:
                child_story_cursor = await self._db.execute(
                    "SELECT status FROM objective_backlog WHERE tier='Story' AND parent_id = ?",
                    (epic_id,),
                )
                child_story_rows = await child_story_cursor.fetchall()
                epic_complete = bool(child_story_rows) and all(
                    str(row["status"]) == "completed" for row in child_story_rows
                )
                if epic_complete:
                    await self._db.execute(
                        "UPDATE objective_backlog "
                        "SET status = 'completed', updated_at = CURRENT_TIMESTAMP "
                        "WHERE id = ?",
                        (epic_id,),
                    )

            await self._db.commit()

    async def ensure_parent_chain_active(self, task_id: int) -> None:
        """Ensure parent Story/Epic stay active while a blocked Task awaits remediation."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT parent_id FROM objective_backlog WHERE id = ? AND tier = 'Task'",
                (task_id,),
            )
            task_row = await cursor.fetchone()
            if task_row is None:
                return

            story_id = task_row["parent_id"]
            if story_id is None:
                return

            story_cursor = await self._db.execute(
                "SELECT parent_id FROM objective_backlog WHERE id = ? AND tier = 'Story'",
                (story_id,),
            )
            story_row = await story_cursor.fetchone()
            epic_id = story_row["parent_id"] if story_row else None

            await self._db.execute(
                "UPDATE objective_backlog "
                "SET status = 'active', updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND status != 'suspended'",
                (story_id,),
            )
            if epic_id is not None:
                await self._db.execute(
                    "UPDATE objective_backlog "
                    "SET status = 'active', updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = ? AND status != 'suspended'",
                    (epic_id,),
                )

            await self._db.commit()

    async def get_all_active_goals(self) -> List[dict]:
        """Return all non-completed, non-suspended objectives grouped by tier."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, tier, title, status, priority, estimated_energy, origin, parent_id, "
                "depends_on_ids, acceptance_criteria, defer_count, last_energy_eval_at, "
                "next_eligible_at, last_energy_eval_json, created_at, updated_at "
                "FROM objective_backlog WHERE status NOT IN ('completed', 'suspended') "
                "ORDER BY CASE tier WHEN 'Epic' THEN 1 WHEN 'Story' THEN 2 WHEN 'Task' THEN 3 END, priority ASC"
            )
            rows = await cursor.fetchall()
        return [self._normalize_objective_row(r) for r in rows]

    async def get_active_objective_tree(self, epic_id: Optional[int] = None) -> List[dict]:
        """Return active Epic/Story/Task nodes, optionally scoped to one Epic subtree."""
        select_fields = (
            "id, tier, title, status, priority, estimated_energy, origin, parent_id, "
            "depends_on_ids, acceptance_criteria, defer_count, last_energy_eval_at, "
            "next_eligible_at, last_energy_eval_json, created_at, updated_at"
        )
        prefixed_child_fields = ", ".join(
            f"child.{field.strip()}" for field in select_fields.split(",")
        )
        async with self._lock:
            if epic_id is None:
                cursor = await self._db.execute(
                    "SELECT " + select_fields + " FROM objective_backlog "
                    "WHERE status NOT IN ('completed', 'suspended') "
                    "ORDER BY CASE tier WHEN 'Epic' THEN 1 WHEN 'Story' THEN 2 WHEN 'Task' THEN 3 END, "
                    "priority ASC, id ASC"
                )
            else:
                cursor = await self._db.execute(
                    "WITH RECURSIVE objective_tree AS ("
                    "  SELECT " + select_fields + " FROM objective_backlog WHERE id = ? "
                    "  UNION ALL "
                    "  SELECT " + prefixed_child_fields + " "
                    "  FROM objective_backlog child "
                    "  JOIN objective_tree parent ON child.parent_id = parent.id"
                    ") "
                    "SELECT " + select_fields + " FROM objective_tree "
                    "WHERE status NOT IN ('completed', 'suspended') "
                    "ORDER BY CASE tier WHEN 'Epic' THEN 1 WHEN 'Story' THEN 2 WHEN 'Task' THEN 3 END, "
                    "priority ASC, id ASC",
                    (epic_id,),
                )
            rows = await cursor.fetchall()
        return [self._normalize_objective_row(row) for row in rows]

    async def get_unresolved_depends_on_ids(self, task_id: int) -> List[int]:
        """Return dependency IDs that are not yet completed for a Task."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, tier, depends_on_ids FROM objective_backlog WHERE id = ?",
                (task_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return []

            if str(row["tier"]) != "Task":
                return []

            depends_on_ids = self._deserialize_depends_on_ids(row["depends_on_ids"])
            if not depends_on_ids:
                return []

            placeholders = ",".join("?" for _ in depends_on_ids)
            dep_cursor = await self._db.execute(
                "SELECT id, status FROM objective_backlog WHERE id IN (" + placeholders + ")",
                tuple(depends_on_ids),
            )
            dep_rows = await dep_cursor.fetchall()

        status_by_id = {int(dep_row["id"]): str(dep_row["status"]) for dep_row in dep_rows}
        unresolved = [
            dep_id
            for dep_id in depends_on_ids
            if status_by_id.get(dep_id) != "completed"
        ]
        return unresolved

    async def get_tasks_with_unresolved_dependencies(
        self,
        statuses: Optional[List[str]] = None,
    ) -> List[dict]:
        """Return Task rows that still have unresolved depends_on_ids."""
        statuses = statuses or ["pending", "active", "deferred_due_to_energy"]
        if not statuses:
            return []

        placeholders = ",".join("?" for _ in statuses)
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, tier, title, status, parent_id, depends_on_ids, acceptance_criteria "
                "FROM objective_backlog WHERE tier='Task' AND status IN (" + placeholders + ")",
                tuple(statuses),
            )
            task_rows = await cursor.fetchall()

            status_cursor = await self._db.execute(
                "SELECT id, status FROM objective_backlog"
            )
            all_status_rows = await status_cursor.fetchall()

        status_by_id = {int(item["id"]): str(item["status"]) for item in all_status_rows}
        unresolved_tasks: List[dict] = []
        for row in task_rows:
            normalized = self._normalize_objective_row(row)
            unresolved = [
                dep_id
                for dep_id in normalized["depends_on_ids"]
                if status_by_id.get(dep_id) != "completed"
            ]
            if unresolved:
                normalized["unresolved_depends_on_ids"] = unresolved
                unresolved_tasks.append(normalized)

        return unresolved_tasks

    async def get_task_with_parent_context(self, task_id: int) -> Optional[dict]:
        """Return one Task with its parent Story and Epic context for energy evaluation."""
        query = self._objective_energy_context_query("task.tier = 'Task' AND task.id = ?")
        async with self._lock:
            cursor = await self._db.execute(query, (task_id,))
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._normalize_energy_context_row(row)

    async def get_energy_evaluation_candidates(
        self,
        statuses: Optional[List[str]] = None,
    ) -> List[dict]:
        """Return energy-eligible Task candidates with parent Story/Epic context."""
        statuses = statuses or ["pending", "deferred_due_to_energy"]
        if not statuses:
            return []

        placeholders = ",".join("?" for _ in statuses)
        where_clause = (
            "task.tier = 'Task' AND task.status IN (" + placeholders + ") "
            "AND (task.next_eligible_at IS NULL OR datetime(task.next_eligible_at) <= CURRENT_TIMESTAMP)"
        )
        query = self._objective_energy_context_query(where_clause)
        query += " ORDER BY task.priority ASC, task.estimated_energy ASC, task.id ASC"

        async with self._lock:
            cursor = await self._db.execute(query, tuple(statuses))
            rows = await cursor.fetchall()

        return [self._normalize_energy_context_row(row) for row in rows]

    async def get_objective_hierarchy_rollup(self, epic_id: Optional[int] = None) -> List[dict]:
        """Return roll-up completion/status summaries for Epic and Story tiers."""
        select_fields = (
            "id, tier, title, status, priority, parent_id, depends_on_ids, acceptance_criteria"
        )
        prefixed_child_fields = ", ".join(
            f"child.{field.strip()}" for field in select_fields.split(",")
        )
        async with self._lock:
            if epic_id is None:
                cursor = await self._db.execute(
                    "SELECT " + select_fields + " FROM objective_backlog"
                )
            else:
                cursor = await self._db.execute(
                    "WITH RECURSIVE objective_tree AS ("
                    "  SELECT " + select_fields + " FROM objective_backlog WHERE id = ? "
                    "  UNION ALL "
                    "  SELECT " + prefixed_child_fields + " "
                    "  FROM objective_backlog child "
                    "  JOIN objective_tree parent ON child.parent_id = parent.id"
                    ") "
                    "SELECT " + select_fields + " FROM objective_tree",
                    (epic_id,),
                )
            rows = await cursor.fetchall()

        if not rows:
            return []

        nodes = {int(row["id"]): self._normalize_objective_row(row) for row in rows}
        children = self._build_children_index(nodes)
        memo: Dict[int, Dict[str, int]] = {}

        rollups: List[dict] = []
        for node_id, node in nodes.items():
            tier = str(node.get("tier") or "")
            if tier not in {"Epic", "Story"}:
                continue

            totals = self._aggregate_task_totals(node_id, nodes, children, memo)
            total_tasks = totals["total"]
            completed_tasks = totals["completed"]
            pending_tasks = totals["pending"]
            active_tasks = totals["active"]
            suspended_tasks = totals["suspended"]

            completion_ratio = (
                float(completed_tasks) / float(total_tasks)
                if total_tasks > 0 else 0.0
            )
            rolled_up_status = self._derive_rollup_status(
                fallback_status=str(node.get("status") or "pending"),
                total_tasks=total_tasks,
                completed_tasks=completed_tasks,
                pending_tasks=pending_tasks,
                active_tasks=active_tasks,
                suspended_tasks=suspended_tasks,
            )

            rollups.append(
                {
                    "id": node_id,
                    "tier": tier,
                    "title": str(node.get("title") or ""),
                    "status": str(node.get("status") or ""),
                    "rolled_up_status": rolled_up_status,
                    "completion_ratio": completion_ratio,
                    "total_tasks": total_tasks,
                    "completed_tasks": completed_tasks,
                    "pending_tasks": pending_tasks,
                    "active_tasks": active_tasks,
                    "suspended_tasks": suspended_tasks,
                    "priority": int(node.get("priority") or 0),
                    "parent_id": node.get("parent_id"),
                }
            )

        return sorted(
            rollups,
            key=lambda item: (
                0 if item["tier"] == "Epic" else 1,
                item["priority"],
                item["id"],
            ),
        )

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
        """Return {user_id: {synthesis, original_input, _created_at}} for all pending approvals."""
        await self.purge_expired_pending()
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT user_id, synthesis_json, original_input, "
                "strftime('%s', created_at) AS created_epoch FROM pending_tool_approvals"
            )
            rows = await cursor.fetchall()
        result = {}
        for row in rows:
            try:
                result[row["user_id"]] = {
                    "synthesis": json.loads(row["synthesis_json"]),
                    "original_input": row["original_input"],
                    "_created_at": float(row["created_epoch"] or 0),
                }
            except Exception as e:
                logger.warning("Could not deserialize pending approval for '%s': %s", row["user_id"], e)
        return result

    async def clear_pending_approval(self, user_id: str) -> None:
        """Remove a pending approval after it is accepted or rejected."""
        async with self._lock:
            await self._db.execute(
                "DELETE FROM pending_tool_approvals WHERE user_id=?", (user_id,)
            )
            await self._db.commit()

    # ─────────────────────────────────────────────────────────────
    # Synthesis Run Audit (Phase 7)
    # ─────────────────────────────────────────────────────────────

    async def create_synthesis_run(
        self,
        *,
        user_id: str,
        gap_description: str,
        suggested_tool_name: str,
        original_input: str,
        max_retries: int,
    ) -> int:
        """Create a new synthesis run record and return run_id."""
        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO synthesis_runs "
                "(user_id, gap_description, suggested_tool_name, original_input, status, max_retries) "
                "VALUES (?, ?, ?, ?, 'in_progress', ?)",
                (
                    str(user_id or ""),
                    str(gap_description or ""),
                    str(suggested_tool_name or ""),
                    str(original_input or ""),
                    max(1, int(max_retries or 1)),
                ),
            )
            await self._db.commit()
            return int(cursor.lastrowid)

    @staticmethod
    def _coerce_text(value: Any, default: str = "") -> str:
        if value is None:
            return default
        text = str(value)
        return text if text else default

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            if value is None or value == "":
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _build_synthesis_attempt_insert_values(
        *,
        run_id: int,
        attempt_number: int,
        phase: str,
        synthesis_payload: Dict[str, Any],
        self_test_result: Dict[str, Any],
        code_sha256: str,
    ) -> tuple[Any, ...]:
        payload = dict(synthesis_payload or {})
        result = dict(self_test_result or {})
        synthesis_json = json.dumps(payload, sort_keys=True)
        phase_name = LedgerMemory._coerce_text(phase, "synthesis")
        tool_name = LedgerMemory._coerce_text(payload.get("tool_name"))
        status = LedgerMemory._coerce_text(result.get("status"), "failed")
        duration_ms = LedgerMemory._coerce_int(result.get("duration_ms"))
        passed_count = LedgerMemory._coerce_int(result.get("passed"))
        failed_count = LedgerMemory._coerce_int(result.get("failed"))
        error_count = LedgerMemory._coerce_int(result.get("errors"))
        skipped_count = LedgerMemory._coerce_int(result.get("skipped"))
        error_message = LedgerMemory._coerce_text(result.get("error"))
        stdout_text = LedgerMemory._coerce_text(result.get("stdout"))
        stderr_text = LedgerMemory._coerce_text(result.get("stderr"))
        digest = LedgerMemory._coerce_text(code_sha256)
        return (
            int(run_id),
            max(1, int(attempt_number)),
            phase_name,
            tool_name,
            status,
            1 if bool(result.get("timed_out")) else 0,
            result.get("exit_code"),
            duration_ms,
            passed_count,
            failed_count,
            error_count,
            skipped_count,
            error_message,
            stdout_text,
            stderr_text,
            digest,
            synthesis_json,
        )

    async def _resolve_synthesis_attempt_id(self, run_id: int, attempt_number: int) -> int:
        lookup = await self._db.execute(
            "SELECT id FROM synthesis_attempts WHERE run_id = ? AND attempt_number = ?",
            (int(run_id), max(1, int(attempt_number))),
        )
        row = await lookup.fetchone()
        return int(row["id"]) if row is not None else 0

    async def append_synthesis_attempt(
        self,
        *,
        run_id: int,
        attempt_number: int,
        phase: str,
        synthesis_payload: Dict[str, Any],
        self_test_result: Dict[str, Any],
        code_sha256: str,
    ) -> int:
        """Insert or update one synthesis attempt row for a run."""
        values = self._build_synthesis_attempt_insert_values(
            run_id=run_id,
            attempt_number=attempt_number,
            phase=phase,
            synthesis_payload=synthesis_payload,
            self_test_result=self_test_result,
            code_sha256=code_sha256,
        )

        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO synthesis_attempts "
                "(run_id, attempt_number, phase, tool_name, status, timed_out, exit_code, duration_ms, "
                "passed_count, failed_count, error_count, skipped_count, error_message, stdout_text, stderr_text, "
                "code_sha256, synthesis_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, attempt_number) DO UPDATE SET "
                "phase=excluded.phase, tool_name=excluded.tool_name, status=excluded.status, "
                "timed_out=excluded.timed_out, exit_code=excluded.exit_code, duration_ms=excluded.duration_ms, "
                "passed_count=excluded.passed_count, failed_count=excluded.failed_count, "
                "error_count=excluded.error_count, skipped_count=excluded.skipped_count, "
                "error_message=excluded.error_message, stdout_text=excluded.stdout_text, "
                "stderr_text=excluded.stderr_text, code_sha256=excluded.code_sha256, "
                "synthesis_json=excluded.synthesis_json, created_at=CURRENT_TIMESTAMP",
                values,
            )
            await self._db.commit()

            if cursor.lastrowid:
                return int(cursor.lastrowid)
            return await self._resolve_synthesis_attempt_id(run_id, attempt_number)

    async def update_synthesis_run_status(
        self,
        run_id: int,
        *,
        status: str,
        total_attempts: Optional[int] = None,
        successful_attempt: Optional[int] = None,
        final_tool_name: Optional[str] = None,
        code_sha256: Optional[str] = None,
        test_summary: Optional[str] = None,
        blocked_reason: Optional[str] = None,
        synthesis_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update synthesis run state and optional final metadata."""
        set_clauses = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: List[Any] = [str(status or "in_progress")]

        if total_attempts is not None:
            set_clauses.append("total_attempts = ?")
            params.append(max(0, int(total_attempts)))
        if successful_attempt is not None:
            set_clauses.append("successful_attempt = ?")
            params.append(int(successful_attempt))
        if final_tool_name is not None:
            set_clauses.append("final_tool_name = ?")
            params.append(str(final_tool_name or ""))
        if code_sha256 is not None:
            set_clauses.append("code_sha256 = ?")
            params.append(str(code_sha256 or ""))
        if test_summary is not None:
            set_clauses.append("test_summary = ?")
            params.append(str(test_summary or ""))
        if blocked_reason is not None:
            set_clauses.append("blocked_reason = ?")
            params.append(str(blocked_reason or ""))
        if synthesis_payload is not None:
            set_clauses.append("synthesis_json = ?")
            params.append(json.dumps(dict(synthesis_payload or {}), sort_keys=True))

        params.append(int(run_id))
        async with self._lock:
            await self._db.execute(
                f"UPDATE synthesis_runs SET {', '.join(set_clauses)} WHERE id = ?",
                tuple(params),
            )
            await self._db.commit()

    async def get_synthesis_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """Return one synthesis run by ID, including decoded synthesis payload."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, user_id, gap_description, suggested_tool_name, original_input, status, "
                "final_tool_name, total_attempts, successful_attempt, max_retries, code_sha256, "
                "test_summary, blocked_reason, synthesis_json, created_at, updated_at "
                "FROM synthesis_runs WHERE id = ?",
                (int(run_id),),
            )
            row = await cursor.fetchone()
        if row is None:
            return None

        result = dict(row)
        try:
            result["synthesis"] = json.loads(str(result.get("synthesis_json") or "{}"))
        except Exception:
            result["synthesis"] = {}
        return result

    async def get_synthesis_attempts(self, run_id: int) -> List[Dict[str, Any]]:
        """Return all synthesis attempts for a run ordered by attempt number."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, run_id, attempt_number, phase, tool_name, status, timed_out, exit_code, "
                "duration_ms, passed_count, failed_count, error_count, skipped_count, error_message, "
                "stdout_text, stderr_text, code_sha256, synthesis_json, created_at "
                "FROM synthesis_attempts WHERE run_id = ? ORDER BY attempt_number ASC",
                (int(run_id),),
            )
            rows = await cursor.fetchall()

        attempts: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["timed_out"] = bool(item.get("timed_out"))
            try:
                item["synthesis"] = json.loads(str(item.get("synthesis_json") or "{}"))
            except Exception:
                item["synthesis"] = {}
            attempts.append(item)
        return attempts

    async def get_latest_synthesis_run_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return the most recent synthesis run for a user."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id FROM synthesis_runs WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
                (str(user_id or ""),),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return await self.get_synthesis_run(int(row["id"]))

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
        await self.purge_expired_pending()
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT user_id, state_json, strftime('%s', created_at) AS created_epoch "
                "FROM pending_hitl_states"
            )
            rows = await cursor.fetchall()
        result = {}
        for row in rows:
            try:
                state = json.loads(row["state_json"])
                if isinstance(state, dict) and "_hitl_created_at" not in state:
                    state["_hitl_created_at"] = float(row["created_epoch"] or 0)
                result[row["user_id"]] = state
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
        await self.purge_expired_pending()
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT user_id, tool_name, arguments_json, strftime('%s', created_at) AS created_epoch "
                "FROM pending_mfa_states"
            )
            rows = await cursor.fetchall()
        result = {}
        for row in rows:
            try:
                result[row["user_id"]] = {
                    "name": row["tool_name"],
                    "arguments": json.loads(row["arguments_json"]),
                    "_created_at": float(row["created_epoch"] or 0),
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

    # ─────────────────────────────────────────────────────────────
    # Session Management
    # ─────────────────────────────────────────────────────────────

    async def create_session(self, name: str, description: str = "") -> int:
        """Create a new session and return its id."""
        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO sessions (name, description) VALUES (?, ?)",
                (name, description or ""),
            )
            await self._db.commit()
            return cursor.lastrowid

    async def activate_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Deactivate all sessions, activate the given one, and return its row."""
        async with self._lock:
            await self._db.execute("UPDATE sessions SET is_active=0")
            await self._db.execute(
                "UPDATE sessions SET is_active=1, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (int(session_id),),
            )
            await self._db.commit()
            cursor = await self._db.execute(
                "SELECT id, name, description, is_active, turn_count, created_at FROM sessions WHERE id=?",
                (int(session_id),),
            )
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_active_session(self) -> Optional[Dict[str, Any]]:
        """Return the currently active session, or None."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, name, description, is_active, turn_count, created_at "
                "FROM sessions WHERE is_active=1 LIMIT 1"
            )
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def deactivate_all_sessions(self) -> None:
        """Set all sessions to inactive."""
        async with self._lock:
            await self._db.execute("UPDATE sessions SET is_active=0, updated_at=CURRENT_TIMESTAMP")
            await self._db.commit()

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Return all sessions, active one first, then by descending id."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, name, description, is_active, turn_count, created_at "
                "FROM sessions ORDER BY is_active DESC, id DESC"
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Return a single session row by id."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT id, name, description, is_active, turn_count, created_at FROM sessions WHERE id=?",
                (int(session_id),),
            )
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def increment_session_turn_count(self, session_id: int) -> None:
        """Increment the turn counter for the given session."""
        async with self._lock:
            await self._db.execute(
                "UPDATE sessions SET turn_count=turn_count+1, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (int(session_id),),
            )
            await self._db.commit()

    async def save_chat_turn_with_session(
        self,
        user_id: str,
        role: str,
        content: str,
        *,
        session_id: Optional[int] = None,
    ) -> None:
        """Persist one conversational turn optionally scoped to a session."""
        if role not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        async with self._lock:
            await self._db.execute(
                "INSERT INTO chat_history (user_id, role, content, session_id) VALUES (?, ?, ?, ?)",
                (user_id, role, content, session_id),
            )
            await self._db.commit()

    async def get_session_chat_history(
        self,
        user_id: str,
        session_id: int,
        limit: int = 100,
    ) -> List[Dict[str, str]]:
        """Return chat turns scoped to a specific session in chronological order."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT role, content FROM ("
                "  SELECT id, role, content FROM chat_history "
                "  WHERE user_id = ? AND session_id = ? ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC",
                (user_id, int(session_id), limit),
            )
            rows = await cursor.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    # ─────────────────────────────────────────────────────────────
    # Moral Audit helpers
    # ─────────────────────────────────────────────────────────────

    async def get_recent_moral_rejections(
        self,
        user_id: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return up to *limit* most recent moral audit entries where is_approved is False."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT moral_decision_json FROM moral_audit_log "
                "WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
                (user_id, limit * 5),
            )
            rows = await cursor.fetchall()
        rejections: List[Dict[str, Any]] = []
        for row in rows:
            if len(rejections) >= limit:
                break
            try:
                decision = json.loads(row["moral_decision_json"])
            except Exception:
                continue
            if not decision.get("is_approved"):
                rejections.append(decision)
        return rejections

    # ─────────────────────────────────────────────────────────────
    # Synthesis helpers
    # ─────────────────────────────────────────────────────────────

    async def count_synthesis_failures_fuzzy(
        self,
        tool_name: str,
        *,
        window_hours: int = 24,
    ) -> int:
        """Count synthesis runs in terminal failure states whose tool name fuzzy-matches
        *tool_name* (token overlap on underscore-split tokens ≥ 3 chars) within the given
        time window."""
        query_tokens = {t.lower() for t in tool_name.split("_") if len(t) >= 3}
        if not query_tokens:
            return 0
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT suggested_tool_name FROM synthesis_runs "
                "WHERE status IN ('blocked', 'rejected', 'failed') "
                "AND created_at >= datetime('now', ? || ' hours')",
                (f"-{int(window_hours)}",),
            )
            rows = await cursor.fetchall()
        count = 0
        for row in rows:
            run_tokens = {t.lower() for t in str(row["suggested_tool_name"] or "").split("_") if len(t) >= 3}
            if query_tokens & run_tokens:
                count += 1
        return count

    async def count_synthesis_runs_in_window(
        self,
        user_id: str,
        *,
        hours: int = 1,
    ) -> int:
        """Return the number of synthesis runs started by *user_id* within the given hours window."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT COUNT(*) AS cnt FROM synthesis_runs "
                "WHERE user_id = ? AND created_at >= datetime('now', ? || ' hours')",
                (user_id, f"-{int(hours)}"),
            )
            row = await cursor.fetchone()
        return int(row["cnt"] or 0) if row else 0

    async def retire_tools(self, tool_names: List[str]) -> int:
        """Set status='retired' for each named tool that is currently approved. Returns count updated."""
        if not tool_names:
            return 0
        count = 0
        async with self._lock:
            for name in tool_names:
                cursor = await self._db.execute(
                    "UPDATE tool_registry SET status='retired' WHERE name=? AND status='approved'",
                    (str(name),),
                )
                count += cursor.rowcount
            await self._db.commit()
        return count

    async def get_system_states_by_prefix(self, prefix: str) -> Dict[str, str]:
        """Return all system_state rows whose key starts with *prefix* as a {key: value} dict."""
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT key, value FROM system_state WHERE key LIKE ?",
                (f"{prefix}%",),
            )
            rows = await cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}
