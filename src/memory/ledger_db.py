"""
Ledger Database Module for Short-Term Memory and Task Management.

This module provides structured storage for operational state and system events
using SQLite. It maintains a task queue for pending operations and a system
log for audit trails and debugging.
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

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


class LedgerMemory:
    """
    A SQLite-backed ledger for short-term memory and task management.

    This class manages two tables:
    - task_queue: Stores pending and historical tasks
    - system_logs: Stores system events and audit trail
    """

    def __init__(self, db_path: str = "data/ledger.db") -> None:
        """
        Initialize the LedgerMemory instance.

        Args:
            db_path: Path where the SQLite database will be stored.
                    Defaults to "data/ledger.db".

        Raises:
            sqlite3.Error: If database initialization fails.
        """
        self.db_path = db_path

        try:
            self.connection = sqlite3.connect(db_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
            self._initialize_tables()
            logger.info(f"LedgerMemory initialized with database at {db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize LedgerMemory: {e}")
            raise

    def _initialize_tables(self) -> None:
        """
        Create tables if they don't exist.

        This method idempotently initializes the schema.

        Raises:
            sqlite3.Error: If table creation fails.
        """
        cursor = self.connection.cursor()

        try:
            # Create task_queue table
            cursor.execute("""
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

            # Create system_logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    context TEXT
                )
            """)

            # Create indices for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_status
                ON task_queue(status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_priority
                ON task_queue(priority)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_log_timestamp
                ON system_logs(timestamp)
            """)

            self.connection.commit()
            logger.debug("Database tables initialized")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize tables: {e}")
            self.connection.rollback()
            raise

    def add_task(
        self,
        task_description: str,
        priority: int = 5,
        status: str = "PENDING"
    ) -> int:
        """
        Add a task to the task queue.

        Args:
            task_description: Description of the task to be performed.
            priority: Priority level (1-10, where 1 is highest). Defaults to 5.
            status: Initial status of the task. Defaults to "PENDING".

        Returns:
            int: The ID of the newly created task.

        Raises:
            ValueError: If task_description is empty or priority is out of range.
            sqlite3.Error: If database insertion fails.
        """
        if not task_description or not isinstance(task_description, str):
            raise ValueError("Task description must be a non-empty string")

        if not 1 <= priority <= 10:
            raise ValueError("Priority must be between 1 and 10")

        if status not in [s.value for s in TaskStatus]:
            raise ValueError(f"Status must be one of {[s.value for s in TaskStatus]}")

        cursor = self.connection.cursor()

        try:
            cursor.execute("""
                INSERT INTO task_queue (priority, task_description, status)
                VALUES (?, ?, ?)
            """, (priority, task_description, status))

            self.connection.commit()
            task_id = cursor.lastrowid

            logger.info(f"Task {task_id} added: {task_description}")
            self.log_event(
                LogLevel.INFO,
                f"Task created: {task_description}",
                {"task_id": task_id, "priority": priority}
            )

            return task_id
        except sqlite3.Error as e:
            logger.error(f"Failed to add task: {e}")
            self.connection.rollback()
            raise

    def get_pending_tasks(self, order_by_priority: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve all pending tasks from the queue.

        Args:
            order_by_priority: If True, order by priority (ascending). Defaults to True.

        Returns:
            List of dictionaries representing pending tasks with keys:
            - id, priority, task_description, status, created_at, updated_at

        Raises:
            sqlite3.Error: If query fails.
        """
        cursor = self.connection.cursor()

        try:
            query = """
                SELECT id, priority, task_description, status, created_at, updated_at
                FROM task_queue
                WHERE status = 'PENDING'
            """

            if order_by_priority:
                query += " ORDER BY priority ASC, created_at ASC"
            else:
                query += " ORDER BY created_at ASC"

            cursor.execute(query)
            rows = cursor.fetchall()

            tasks = [dict(row) for row in rows]
            logger.info(f"Retrieved {len(tasks)} pending tasks")
            return tasks
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve pending tasks: {e}")
            raise

    def update_task_status(
        self,
        task_id: int,
        new_status: str
    ) -> bool:
        """
        Update the status of a task.

        Args:
            task_id: ID of the task to update.
            new_status: New status value.

        Returns:
            bool: True if task was found and updated, False otherwise.

        Raises:
            ValueError: If new_status is invalid.
            sqlite3.Error: If update fails.
        """
        if new_status not in [s.value for s in TaskStatus]:
            raise ValueError(f"Status must be one of {[s.value for s in TaskStatus]}")

        cursor = self.connection.cursor()

        try:
            cursor.execute("""
                UPDATE task_queue
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (new_status, task_id))

            self.connection.commit()

            if cursor.rowcount > 0:
                logger.info(f"Task {task_id} status updated to {new_status}")
                return True
            else:
                logger.warning(f"Task {task_id} not found")
                return False
        except sqlite3.Error as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            self.connection.rollback()
            raise

    def log_event(
        self,
        log_level: LogLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Log a system event to the ledger.

        Args:
            log_level: The log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            message: The log message.
            context: Optional dictionary with additional context information.

        Returns:
            int: The ID of the log entry.

        Raises:
            ValueError: If message is empty.
            sqlite3.Error: If insertion fails.
        """
        if not message or not isinstance(message, str):
            raise ValueError("Message must be a non-empty string")

        context_str = str(context) if context else None

        cursor = self.connection.cursor()

        try:
            cursor.execute("""
                INSERT INTO system_logs (log_level, message, context)
                VALUES (?, ?, ?)
            """, (log_level.value, message, context_str))

            self.connection.commit()
            log_id = cursor.lastrowid

            logger.debug(f"Log entry {log_id} created: {message}")
            return log_id
        except sqlite3.Error as e:
            logger.error(f"Failed to log event: {e}")
            self.connection.rollback()
            raise

    def get_logs(
        self,
        log_level: Optional[LogLevel] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve system logs.

        Args:
            log_level: Optional filter by log level.
            limit: Maximum number of logs to retrieve. Defaults to 100.

        Returns:
            List of dictionaries representing log entries.

        Raises:
            sqlite3.Error: If query fails.
        """
        cursor = self.connection.cursor()

        try:
            if log_level:
                cursor.execute("""
                    SELECT id, log_level, message, timestamp, context
                    FROM system_logs
                    WHERE log_level = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (log_level.value, limit))
            else:
                cursor.execute("""
                    SELECT id, log_level, message, timestamp, context
                    FROM system_logs
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))

            rows = cursor.fetchall()
            logs = [dict(row) for row in rows]
            logger.info(f"Retrieved {len(logs)} log entries")
            return logs
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve logs: {e}")
            raise

    def close(self) -> None:
        """
        Close the database connection.

        This should be called when the application is shutting down.
        """
        try:
            self.connection.close()
            logger.info("LedgerMemory connection closed")
        except sqlite3.Error as e:
            logger.error(f"Error closing database connection: {e}")
