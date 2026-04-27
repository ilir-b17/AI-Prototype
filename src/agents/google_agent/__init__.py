from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from src.agents.base_domain_agent import BaseAgent
from src.memory.ledger_db import LedgerMemory


class GoogleAgentTaskError(RuntimeError):
    """Raised when a Google domain task cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        action_taken: str,
        recipient: str = "",
        subject: str = "",
        attachment_paths: Optional[List[str]] = None,
    ) -> None:
        super().__init__(message)
        self.action_taken = action_taken
        self.recipient = recipient
        self.subject = subject
        self.attachment_paths = list(attachment_paths or [])


class GoogleAgent(BaseAgent):
    MIN_CHILD_POLL_INTERVAL_SECONDS = 0.05

    allowed_tool_names = [
        "read_inbox",
        "send_email",
        "extract_pdf_text",
        "update_objective_status",
        "search_archival_memory",
    ]
    agent_domain = "google"
    own_energy_budget = 1

    def __init__(
        self,
        *,
        child_poll_interval_seconds: float = 0.1,
        child_timeout_seconds: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.child_poll_interval_seconds = max(
            self.MIN_CHILD_POLL_INTERVAL_SECONDS,
            float(child_poll_interval_seconds),
        )
        self.child_timeout_seconds = max(self.child_poll_interval_seconds, float(child_timeout_seconds))

    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_type = self._infer_task_type(task)
        if task_type == "send_email":
            return await self._handle_send_email_task(task)
        if task_type == "email_poll":
            return await self._handle_email_poll_task(task)
        if task_type == "email_request":
            return await self._handle_email_request_task(task)
        raise GoogleAgentTaskError(
            f"Unsupported task type: {task_type}",
            action_taken="unsupported_task_type",
        )

    async def _write_failed_result(self, ledger: LedgerMemory, task_id: int, payload: Dict[str, Any]) -> None:
        structured = {
            "action_taken": str(payload.get("action_taken") or "failed"),
            "recipient": str(payload.get("recipient") or ""),
            "subject": str(payload.get("subject") or ""),
            "attachment_paths": self._as_str_list(payload.get("attachment_paths")),
            "error": str(payload.get("error") or "unknown_error"),
        }
        await ledger.write_task_failure(task_id, structured)
        await ledger.update_objective_status(task_id, "blocked")

    @staticmethod
    def _as_str_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if str(item).strip()]

    @staticmethod
    def _task_text(task: Dict[str, Any]) -> str:
        title = str(task.get("title") or "")
        criteria = str(task.get("acceptance_criteria") or "")
        return f"{title}\n{criteria}".lower()

    def _infer_task_type(self, task: Dict[str, Any]) -> str:
        title = str(task.get("title") or "").strip().lower()
        criteria = str(task.get("acceptance_criteria") or "").strip().lower()
        combined = self._task_text(task)

        if "email_request" in combined or title.startswith("email_request"):
            return "email_request"
        if "email_poll" in combined or title.startswith("email_poll"):
            return "email_poll"
        if "send_email" in combined or title.startswith("send_email"):
            return "send_email"
        return "unknown"

    @staticmethod
    def _parse_json_blob(raw_text: str) -> Dict[str, Any]:
        text = str(raw_text or "").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    async def _tool_json(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raw = await self.execute_tool(tool_name, arguments)
        parsed = self._parse_json_blob(raw)
        if parsed:
            return parsed
        if str(raw).startswith("Error:"):
            return {"status": "error", "message": str(raw)}
        return {"status": "success", "raw": str(raw)}

    async def _handle_send_email_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        details = self._parse_json_blob(str(task.get("acceptance_criteria") or ""))
        recipient = str(details.get("recipient") or details.get("to") or "").strip()
        subject = str(details.get("subject") or "").strip()
        body = str(details.get("body") or details.get("content") or "").strip()
        attachment_path = str(details.get("attachment_path") or "").strip()
        attachment_paths = [attachment_path] if attachment_path else []

        if not recipient or not subject or not body:
            raise GoogleAgentTaskError(
                "send_email task missing recipient/subject/body",
                action_taken="send_email_failed",
                recipient=recipient,
                subject=subject,
                attachment_paths=attachment_paths,
            )

        send_result = await self._tool_json(
            "send_email",
            {
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "attachment_path": attachment_path,
            },
        )
        if str(send_result.get("status") or "").lower() != "success":
            raise GoogleAgentTaskError(
                str(send_result.get("message") or "send_email failed"),
                action_taken="send_email_failed",
                recipient=recipient,
                subject=subject,
                attachment_paths=attachment_paths,
            )

        return {
            "action_taken": "send_email",
            "recipient": recipient,
            "subject": subject,
            "attachment_paths": attachment_paths,
            "error": "",
        }

    async def _handle_email_poll_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        inbox = await self._tool_json("read_inbox", {"max_emails": 10, "mark_as_read": True})
        if str(inbox.get("status") or "").lower() != "success":
            raise GoogleAgentTaskError(
                str(inbox.get("message") or "read_inbox failed"),
                action_taken="email_poll_failed",
            )

        emails = inbox.get("emails")
        if not isinstance(emails, list):
            emails = []

        ledger = await self._resolve_ledger()
        created = 0
        for email in emails:
            if not isinstance(email, dict):
                continue
            sender = str(email.get("sender") or "").strip()
            subject = str(email.get("subject") or "").strip()
            body = str(email.get("body") or "").strip()
            attachment_paths = self._as_str_list(email.get("attachment_paths"))
            title = f"email_request: {subject or 'No subject'}"
            acceptance_criteria = json.dumps(
                {
                    "sender": sender,
                    "subject": subject,
                    "body": body,
                    "attachment_paths": attachment_paths,
                },
                ensure_ascii=False,
            )
            task_id = await ledger.add_objective(
                tier="Task",
                title=title,
                estimated_energy=8,
                origin="GoogleAgent",
                parent_id=task.get("parent_id"),
                acceptance_criteria=acceptance_criteria,
            )
            await ledger.set_task_agent_domain(task_id, "google")
            created += 1

        return {
            "action_taken": "email_poll",
            "recipient": "",
            "subject": "",
            "attachment_paths": [],
            "error": "",
            "objectives_created": created,
        }

    async def _handle_email_request_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_context = self._parse_json_blob(str(task.get("acceptance_criteria") or ""))
        inbox = await self._tool_json("read_inbox", {"max_emails": 10, "mark_as_read": True})
        if str(inbox.get("status") or "").lower() != "success":
            raise GoogleAgentTaskError(
                str(inbox.get("message") or "read_inbox failed"),
                action_taken="email_request_failed",
            )

        selected_email = self._select_email(inbox.get("emails"), task_context)
        if not selected_email:
            raise GoogleAgentTaskError(
                "No matching email found for request",
                action_taken="email_request_failed",
            )

        sender = str(selected_email.get("sender") or task_context.get("sender") or "").strip()
        subject = str(selected_email.get("subject") or task_context.get("subject") or "").strip()
        attachment_paths = self._as_str_list(selected_email.get("attachment_paths"))
        generation_input = await self._build_generation_input(selected_email, task_context)
        child_task_id = await self._spawn_aiden_child_task(task, generation_input, sender, subject)
        child_result = await self._wait_for_child_result(child_task_id)
        body = self._extract_child_output_text(child_result)
        if not body:
            raise GoogleAgentTaskError(
                "Child task produced no response content",
                action_taken="email_request_failed",
                recipient=sender,
                subject=subject,
                attachment_paths=attachment_paths,
            )

        send_result = await self._tool_json(
            "send_email",
            {
                "recipient": sender,
                "subject": f"Re: {subject}" if subject else "Re: (no subject)",
                "body": body,
            },
        )
        if str(send_result.get("status") or "").lower() != "success":
            raise GoogleAgentTaskError(
                str(send_result.get("message") or "send_email failed"),
                action_taken="email_request_failed",
                recipient=sender,
                subject=subject,
                attachment_paths=attachment_paths,
            )

        return {
            "action_taken": "email_request",
            "recipient": sender,
            "subject": subject,
            "attachment_paths": attachment_paths,
            "error": "",
            "child_task_id": child_task_id,
        }

    @staticmethod
    def _select_email(emails: Any, task_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(emails, list):
            return None
        wanted_sender = str(task_context.get("sender") or "").strip().lower()
        wanted_subject = str(task_context.get("subject") or "").strip().lower()
        for item in emails:
            if not isinstance(item, dict):
                continue
            sender = str(item.get("sender") or "").strip().lower()
            subject = str(item.get("subject") or "").strip().lower()
            sender_ok = not wanted_sender or wanted_sender in sender
            subject_ok = not wanted_subject or wanted_subject in subject
            if sender_ok and subject_ok:
                return item
        return None

    async def _build_generation_input(
        self,
        email_payload: Dict[str, Any],
        task_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        attachment_paths = self._as_str_list(email_payload.get("attachment_paths"))
        extracted_documents: List[Dict[str, Any]] = []
        for path in attachment_paths:
            if not str(path).lower().endswith(".pdf"):
                continue
            extracted = await self._tool_json("extract_pdf_text", {"file_path": path, "full_context": False})
            if str(extracted.get("status") or "").lower() == "success":
                extracted_documents.append(
                    {
                        "path": path,
                        "text": str(extracted.get("text") or ""),
                    }
                )

        return {
            "sender": str(email_payload.get("sender") or task_context.get("sender") or ""),
            "subject": str(email_payload.get("subject") or task_context.get("subject") or ""),
            "body": str(email_payload.get("body") or task_context.get("body") or ""),
            "attachment_paths": attachment_paths,
            "extracted_documents": extracted_documents,
            "instruction": "Generate complete email response content for sender.",
        }

    async def _spawn_aiden_child_task(
        self,
        parent_task: Dict[str, Any],
        generation_input: Dict[str, Any],
        sender: str,
        subject: str,
    ) -> int:
        ledger = await self._resolve_ledger()
        child_title = f"aiden_content_generation: {subject or 'email response'}"
        child_acceptance = json.dumps(generation_input, ensure_ascii=False)
        child_task_id = await ledger.add_objective(
            tier="Task",
            title=child_title,
            estimated_energy=10,
            origin="GoogleAgent",
            parent_id=parent_task.get("parent_id"),
            acceptance_criteria=child_acceptance,
        )
        await ledger.set_task_agent_domain(child_task_id, "aiden")
        return child_task_id

    async def _wait_for_child_result(self, child_task_id: int) -> Dict[str, Any]:
        ledger = await self._resolve_ledger()
        deadline = asyncio.get_running_loop().time() + self.child_timeout_seconds

        while asyncio.get_running_loop().time() <= deadline:
            row = await ledger.get_task_row(child_task_id)
            if row:
                status = str(row.get("status") or "").strip().lower()
                if status == "completed":
                    result = await ledger.get_task_result(child_task_id)
                    if result is not None:
                        return result
                    raise GoogleAgentTaskError(
                        "Child task completed without result payload",
                        action_taken="email_request_failed",
                    )
                if status in {"failed", "blocked", "suspended"}:
                    result = await ledger.get_task_result(child_task_id)
                    message = "Child task did not complete successfully"
                    if isinstance(result, dict) and result.get("error"):
                        message = str(result.get("error"))
                    raise GoogleAgentTaskError(message, action_taken="email_request_failed")
            await asyncio.sleep(self.child_poll_interval_seconds)

        raise GoogleAgentTaskError(
            f"Timed out waiting for child task result (task_id={child_task_id})",
            action_taken="email_request_failed",
        )

    @staticmethod
    def _extract_child_output_text(child_result: Dict[str, Any]) -> str:
        for key in ("final_answer", "body", "content", "message", "result"):
            value = child_result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""


__all__ = ["GoogleAgent"]
