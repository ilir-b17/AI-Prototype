"""Memory operations mixin for Orchestrator."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.orchestrator_constants import (
    MEMORY_SAVE_THRESHOLD,
    MEMORY_CONSOLIDATION_INTERVAL,
    _EXPLICIT_MEMORY_REQUEST_RE,
    _CONSOLIDATION_TRIGGER_TURNS,
    _CONSOLIDATION_TURN_COUNT_MAX_USERS,
    _PENDING_STATE_TTL_SECONDS,
    _RECENT_CHAT_HISTORY_LIMIT,
)
from src.core import cloud_redaction
from src.core.state_model import AgentState, normalize_state

logger = logging.getLogger(__name__)


class _MemoryOpsMixin:
    """Memory operation methods. Mixed into Orchestrator."""

    @staticmethod
    def _normalize_display_name(raw_name: str) -> str:
        cleaned = re.sub(r"\s+", " ", raw_name).strip(" ,.!?")
        return " ".join(part.capitalize() for part in cleaned.split())

    @staticmethod
    def _extract_user_profile_updates(user_message: str) -> Dict[str, Any]:
        updates: Dict[str, Any] = {}
        text = (user_message or "").strip()
        lowered = text.lower()

        # Matches "my name is <Name>"; spaces are allowed in the name, and trailing punctuation or end-of-string is required. capitalize() is used, so mixed-case surnames like "McCaffrey" normalize to "Mccaffrey" (known limitation).
        name_match = re.search(
            r"\bmy name is\s+([a-z][a-z' -]{0,40})(?=,|\.|!|\?|$)",
            text,
            flags=re.IGNORECASE,
        )
        if name_match:
            raw_name = Orchestrator._normalize_display_name(name_match.group(1))
            if raw_name:
                updates["name"] = raw_name

        # Matches "<Name> is me"; question forms are intentionally not matched.
        identity_match = re.search(
            r"(?:^|[.!?,]\s*)([a-z][a-z' -]{0,60})\s+is me\b(?=,|\.|!|\?|$)",
            text,
            flags=re.IGNORECASE,
        )
        if identity_match and "name" not in updates:
            raw_name = Orchestrator._normalize_display_name(identity_match.group(1))
            if raw_name:
                updates["name"] = raw_name

        age_match = re.search(r"\b(?:i am|i'm)\s+(\d{1,3})\s*years?\s*old\b", lowered)
        if age_match:
            updates["age"] = int(age_match.group(1))

        if re.search(r"\bi am(?: also)? your developer\b", lowered):
            updates["relationship"] = "developer"

        return updates

    @staticmethod
    def _extract_assistant_identity_update(user_message: str) -> Optional[str]:
        text = (user_message or "").strip()
        if not text:
            return None

        # Polite request patterns that may include "?" - must be checked before the "?" guard.
        # e.g. "Can I call you Aiden?", "I'll call you Aiden", "Let's call you Aiden"
        call_patterns = (
            r"\b(?:can i|may i|i(?:'ll| will| want to)|let(?:'s| us)) call you\s+([a-z][a-z' -]{0,40})(?:\?|,|\.|!|$)",
            r"\bi(?:'ll| will) refer to you as\s+([a-z][a-z' -]{0,40})(?:\?|,|\.|!|$)",
        )
        for pattern in call_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return Orchestrator._normalize_display_name(match.group(1))

        # Statement patterns must not be questions to avoid false positives like "Is your name X?"
        if "?" in text:
            return None

        statement_patterns = (
            r"\b(?:just for your information,\s*)?your name is\s+([a-z][a-z' -]{0,40})(?=,|\.|!|$)",
            r"\bi am informing you(?: that)? your name is\s+([a-z][a-z' -]{0,40})(?=,|\.|!|$)",
        )
        for pattern in statement_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return Orchestrator._normalize_display_name(match.group(1))
        return None

    async def _remember_user_profile(self, user_id: str, user_message: str) -> bool:
        if not user_id or not hasattr(self, "core_memory"):
            return False

        updates = self._extract_user_profile_updates(user_message)
        if not updates:
            return False

        core_state = await self.core_memory.get_all()
        profiles = dict(core_state.get("user_profiles", {}) or {})
        profile = dict(profiles.get(user_id, {}) or {})
        changed = False

        for key, value in updates.items():
            if profile.get(key) != value:
                profile[key] = value
                changed = True

        if not changed:
            return False

        profiles[user_id] = profile
        await self.core_memory.update("user_profiles", profiles)
        return True

    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        if not user_id or not hasattr(self, "core_memory"):
            return {}
        core_state = await self.core_memory.get_all()
        profiles = dict(core_state.get("user_profiles", {}) or {})
        return dict(profiles.get(user_id, {}) or {})

    async def _remember_assistant_identity(self, user_message: str) -> Optional[str]:
        if not hasattr(self, "core_memory"):
            return None

        assistant_name = self._extract_assistant_identity_update(user_message)
        if not assistant_name:
            return None

        core_state = await self.core_memory.get_all()
        assistant_profile = dict(core_state.get("assistant_profile", {}) or {})
        if assistant_profile.get("name") == assistant_name:
            return assistant_name

        assistant_profile["name"] = assistant_name
        await self.core_memory.update("assistant_profile", assistant_profile)
        return assistant_name

    async def _get_assistant_name(self) -> str:
        if not hasattr(self, "core_memory"):
            return "AIDEN"

        core_state = await self.core_memory.get_all()
        assistant_profile = dict(core_state.get("assistant_profile", {}) or {})
        assistant_name = str(assistant_profile.get("name", "")).strip()
        return assistant_name or "AIDEN"

    @staticmethod
    def _format_profile_update_ack(updates: Dict[str, Any]) -> str:
        parts = []
        if "name" in updates:
            parts.append(f"your name as {updates['name']}")
        if "age" in updates:
            parts.append(f"your age as {updates['age']}")
        if "relationship" in updates:
            parts.append(f"your role as {updates['relationship']}")
        detail = " and ".join(parts) if parts else "that information"
        return f"Noted. I've recorded {detail} in my memory."

    @staticmethod
    def _format_user_profile_response(lowered: str, profile: Dict[str, Any]) -> str:
        name = profile.get("name")
        age = profile.get("age")

        if "age" in lowered or "old am i" in lowered:
            if age:
                return f"You told me that you are {age} years old."
            return "You have not told me your age yet."

        if name and age:
            return f"Yes. Your name is {name}, and you told me that you are {age} years old."
        if name:
            return f"Yes. Your name is {name}."
        return "Not yet. You have not told me your name."

    async def _get_archival_context(
        self,
        query: str,
        session_id: Optional[int] = None,
        epic_id: Optional[int] = None,
    ) -> str:
        """Retrieve top archival memory snippets with two-stage retrieval.

        Stage 1: Broad cosine recall (n_results * multiplier candidates).
        Stage 2: LLM reranking to select the most relevant subset.
        Falls back to cosine-only if reranker is disabled or fails.
        """
        if not query:
            return ""

        _max_chunk = int(os.getenv("MAX_ARCHIVAL_CHUNK_CHARS", "2000"))
        _max_total = int(os.getenv("MAX_ARCHIVAL_TOTAL_CHARS", "6000"))
        _final_n = 3  # final number of results to inject into prompt
        _reranker = getattr(self, "memory_reranker", None)
        _reranker_enabled = bool(getattr(_reranker, "enabled", False))

        try:
            # Build where clause for session scoping (from Direction 1)
            where: Optional[Dict[str, Any]] = None
            try:
                from src.skills.search_archival_memory import _build_where_clause
                where = _build_where_clause(session_id, epic_id)
            except ImportError:
                pass

            # Stage 1: broad cosine recall
            broad_n = (
                _reranker.broad_n_results(_final_n)
                if _reranker_enabled
                else _final_n
            )
            try:
                candidates = await self.vector_memory.query_memory_async(
                    query, n_results=broad_n, where=where
                )
            except Exception:
                candidates = []

            # Fallback: if scoped returns < 2 results, go global
            if where is not None and len(candidates) < 2:
                try:
                    candidates = await self.vector_memory.query_memory_async(
                        query, n_results=broad_n
                    )
                except Exception:
                    candidates = []

            if not candidates:
                return ""

            # Stage 2: rerank if we have more candidates than needed
            if (
                _reranker_enabled
                and len(candidates) > _final_n
            ):
                results = await self._rerank_memories(query, candidates, _final_n)
            else:
                results = sorted(
                    candidates,
                    key=lambda c: float(c.get("distance") or 1.0),
                )[:_final_n]

            # Build the archival context XML block
            lines = ["<Archival_Memory>"]
            total = 0
            for item in results:
                snippet = (item.get("document", "") or "")[:_max_chunk]
                if not snippet:
                    continue
                if total + len(snippet) > _max_total:
                    break

                # Annotate with session scope and rerank score when available
                meta = item.get("metadata", {}) or {}
                mem_session = int(meta.get("session_id") or 0)
                scope_tag = (
                    " [session]"
                    if mem_session and mem_session == session_id
                    else ""
                )

                # Include rerank score annotation when reranking was applied
                rerank_annotation = ""
                if item.get("reranked"):
                    rerank_annotation = (
                        f" [relevance={item.get('combined_score', 0):.2f}]"
                    )

                tag_attrs = f"{scope_tag}{rerank_annotation}".strip()
                open_tag = f"<Memory{(' ' + tag_attrs) if tag_attrs else ''}>"
                close_tag = "</Memory>"
                lines.append(f"  {open_tag}{snippet}{close_tag}")
                total += len(snippet)

            lines.append("</Archival_Memory>")
            return "\n".join(lines)

        except Exception as e:
            logger.warning("Archival memory lookup failed: %s", e)
            return ""

    async def _rerank_memories(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        n_results: int,
    ) -> List[Dict[str, Any]]:
        """Two-stage retrieval: rerank VectorMemory candidates with System 1.

        This is the callable registered in runtime_context for use by skills.
        Fails safely - always returns a valid list (cosine-ordered on error).
        """
        _reranker = getattr(self, "memory_reranker", None)
        if not bool(getattr(_reranker, "enabled", False)):
            return candidates[:n_results]
        try:
            return await _reranker.rerank(
                query=query,
                candidates=candidates,
                n_results=n_results,
                route_to_system_1=self._route_memories_to_system_1,
            )
        except Exception as exc:
            logger.warning(
                "_rerank_memories failed (cosine fallback): %s", exc
            )
            return sorted(
                candidates,
                key=lambda c: float(c.get("distance") or 1.0),
            )[:n_results]

    async def _memory_consolidation_loop(self) -> None:
        """Background task: periodically consolidate chat history into long-term memory."""
        logger.info("Memory consolidation loop started.")
        while True:
            await asyncio.sleep(MEMORY_CONSOLIDATION_INTERVAL)
            try:
                await self._sweep_expired_pending_state()
                user_ids = await self.ledger_memory.get_recent_user_ids(limit=20)
                if not user_ids:
                    continue
                for user_id in user_ids:
                    await self._consolidate_memory(user_id)
                # Periodic DB pruning - delete old chat rows and vector memories
                try:
                    deleted = await self.ledger_memory.prune_old_chat_history(
                        days=int(os.getenv("CHAT_HISTORY_MAX_DAYS", "90")),
                        keep_minimum=int(os.getenv("CHAT_HISTORY_MIN_ROWS", "20")),
                    )
                    if deleted:
                        logger.info("Consolidation pruning: removed %d chat_history rows.", deleted)
                except Exception as _p_err:
                    logger.warning("Periodic chat history pruning failed: %s", _p_err)
                try:
                    await asyncio.to_thread(
                        self.vector_memory.prune_old_memories,
                        int(os.getenv("VECTOR_MEMORY_MAX_DAYS", "180")),
                    )
                except Exception as _vp_err:
                    logger.warning("Periodic vector memory pruning failed: %s", _vp_err)
            except Exception as e:
                logger.warning(f"Memory consolidation loop error: {e}", exc_info=True)

    async def _save_memory_async(
        self,
        text: str,
        session_id: Optional[int] = None,
        epic_id: Optional[int] = None,
    ) -> None:
        """Fire-and-forget memory storage with optional session/epic tagging."""
        try:
            if len(text) < MEMORY_SAVE_THRESHOLD:
                return
            metadata: Dict[str, Any] = {
                "type": "conversation",
                "timestamp": datetime.now().isoformat(),
                # ChromaDB requires scalar values; 0 means "no session/epic"
                "session_id": int(session_id) if session_id else 0,
                "epic_id": int(epic_id) if epic_id else 0,
            }
            await self.vector_memory.add_memory_async(text=text, metadata=metadata)
            if session_id:
                increment_memory_count = getattr(
                    self.ledger_memory,
                    "increment_session_memory_count",
                    None,
                )
                if callable(increment_memory_count) and inspect.iscoroutinefunction(increment_memory_count):
                    self._fire_and_forget(increment_memory_count(session_id))
        except Exception as e:
            logger.warning(f"Async memory save failed: {e}")

    @staticmethod
    def _has_explicit_memory_intent(text: str) -> bool:
        return bool(_EXPLICIT_MEMORY_REQUEST_RE.search(str(text or "")))

    async def _save_explicit_memory_request(self, text: str) -> None:
        try:
            await self.vector_memory.add_memory_async(
                text=text,
                metadata={"type": "explicit_memory", "source": "user_request"},
            )
        except Exception as e:
            logger.warning(f"Explicit memory save failed: {e}")

    async def _apply_text_memory_hooks(self, user_id: str, user_message: str) -> None:
        profile_updated = await self._remember_user_profile(user_id, user_message)
        await self._remember_assistant_identity(user_message)
        if not profile_updated and self._has_explicit_memory_intent(user_message):
            await self._save_explicit_memory_request(user_message)

    async def _consolidate_memory(self, user_id: str) -> None:
        """Run deep nocturnal consolidation pipeline (extract -> filter -> score -> write-back)."""
        try:
            turns = await self.ledger_memory.get_chat_history(user_id, limit=20)
            if len(turns) < 6:
                return

            recent_logs = await self.ledger_memory.get_logs(limit=20)
            critic_logs = [
                entry for entry in recent_logs
                if "critic" in str(entry.get("message", "")).lower()
                or str(entry.get("context", "")).lower().find("critic") != -1
            ]
            try:
                blueprint_entries = await self.vector_memory.query_memory_async(
                    "System 2 Reasoning Blueprint",
                    n_results=5,
                )
            except Exception:
                blueprint_entries = []

            filtered_candidates = await self.nocturnal_consolidation.extract_and_filter_candidates(
                user_id=user_id,
                chat_history=turns,
                worker_outputs={},
                critic_feedback="",
                blueprint_entries=blueprint_entries,
                ledger_logs=critic_logs,
                vector_memory=self.vector_memory,
            )

            if not filtered_candidates:
                await self.ledger_memory.trim_chat_history(user_id, keep_last=5)
                return

            scored_candidates = await self.nocturnal_consolidation.score_candidates_with_system2(
                filtered_candidates,
                route_to_system_2=self._score_nocturnal_candidates,
                redactor=cloud_redaction.redact_text_for_cloud,
                threshold=float(os.getenv("NOCTURNAL_Q_THRESHOLD", "3.0")),
            )
            write_result = await self.nocturnal_consolidation.write_back_scored_candidates(
                scored_candidates,
                core_memory=self.core_memory,
                vector_memory=self.vector_memory,
                ledger_memory=self.ledger_memory,
                threshold=float(os.getenv("NOCTURNAL_Q_THRESHOLD", "3.0")),
            )

            await self.ledger_memory.trim_chat_history(user_id, keep_last=5)
            if write_result.get("stored_total", 0) > 0:
                logger.info(
                    "Nocturnal consolidation stored memories for %s: %s",
                    user_id,
                    write_result,
                )
        except asyncio.TimeoutError:
            logger.warning("Memory consolidation skipped: LLM did not respond within 90s")
        except Exception as e:
            logger.warning(f"Memory consolidation failed: {type(e).__name__}: {e}", exc_info=True)

    def _schedule_response_memory_save(
        self,
        user_message: str,
        final_resp: str,
        session_id: Optional[int] = None,
        epic_id: Optional[int] = None,
    ) -> None:
        self._fire_and_forget(
            self._save_memory_async(
                f"User: {user_message}\nAssistant: {final_resp}",
                session_id=session_id,
                epic_id=epic_id,
            )
        )

    async def _load_active_session_from_core(self) -> Optional[Dict[str, Any]]:
        """Fast path: read active session from core memory (no DB query)."""
        try:
            state = await self.core_memory.get_all()
            session = state.get("active_session")
            if not isinstance(session, dict):
                return None
            if not session.get("id"):
                return None
            return session
        except Exception as exc:
            logger.warning("Could not read active session from core memory: %s", exc)
            return None

    async def _sync_active_session_to_core(
        self, session_row: Optional[Dict[str, Any]]
    ) -> None:
        """Write active session summary to core memory for fast reads."""
        try:
            if session_row is None:
                await self.core_memory.update("active_session", None)
                return
            summary = {
                "id": int(session_row.get("id") or 0),
                "name": str(session_row.get("name") or ""),
                "description": str(session_row.get("description") or ""),
                "epic_id": session_row.get("epic_id"),
                "turn_count": int(session_row.get("turn_count") or 0),
                "memory_count": int(session_row.get("memory_count") or 0),
                "started_at": str(session_row.get("created_at") or ""),
            }
            await self.core_memory.update("active_session", summary)
        except Exception as exc:
            logger.warning("Could not sync active session to core memory: %s", exc)

    async def _get_active_session(self) -> Optional[Dict[str, Any]]:
        """Return active session from core memory cache (fast, no DB call)."""
        return await self._load_active_session_from_core()

    async def _get_session_epic_rollup(
        self, epic_id: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        """Return Epic title and story/task counts for session context block."""
        if not epic_id:
            return None
        try:
            rollups = await self.ledger_memory.get_objective_hierarchy_rollup(epic_id)
            epic_rollup = next(
                (r for r in rollups if r.get("tier") == "Epic" and r.get("id") == epic_id),
                None,
            )
            if not epic_rollup:
                tree = await self.ledger_memory.get_active_objective_tree(epic_id)
                epic_node = next(
                    (n for n in tree if n.get("tier") == "Epic"), None
                )
                if epic_node:
                    return {
                        "title": str(epic_node.get("title") or ""),
                        "status": str(epic_node.get("status") or ""),
                        "total_tasks": 0,
                        "completed_tasks": 0,
                        "pending_tasks": 0,
                        "active_tasks": 0,
                    }
                return None
            return {
                "title": str(epic_rollup.get("title") or ""),
                "status": str(epic_rollup.get("status") or ""),
                "total_tasks": int(epic_rollup.get("total_tasks") or 0),
                "completed_tasks": int(epic_rollup.get("completed_tasks") or 0),
                "pending_tasks": int(epic_rollup.get("pending_tasks") or 0),
                "active_tasks": int(epic_rollup.get("active_tasks") or 0),
            }
        except Exception as exc:
            logger.warning("Could not fetch epic rollup for session: %s", exc)
            return None

    async def _activate_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Activate a session in the DB and sync to core memory."""
        row = await self.ledger_memory.activate_session(session_id)
        if row is not None:
            await self._sync_active_session_to_core(row)
            logger.info("Session activated: id=%s name=%r", row["id"], row["name"])
        return row

    async def _deactivate_session(self) -> None:
        """Deactivate all sessions and clear core memory cache."""
        await self.ledger_memory.deactivate_all_sessions()
        await self._sync_active_session_to_core(None)
        logger.info("Session deactivated (no active session)")

    async def _load_state_impl(
        self,
        user_id: str,
        user_message: str,
        *,
        user_prompt: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a state dict: resume HITL if pending, else create fresh."""
        if user_id in self.pending_hitl_state:
            _hitl_entry = self.pending_hitl_state[user_id]
            _age = time.time() - _hitl_entry.get("_hitl_created_at", 0)
            if _age > _PENDING_STATE_TTL_SECONDS:
                self.pending_hitl_state.pop(user_id, None)
                try:
                    await self.ledger_memory.clear_hitl_state(user_id)
                except Exception:
                    pass
                logger.warning(
                    "Expired stale pending_hitl_state for %s (age %.0fs); creating fresh state.",
                    user_id,
                    _age,
                )
            else:
                state = normalize_state(self.pending_hitl_state.pop(user_id))
                try:
                    await self.ledger_memory.clear_hitl_state(user_id)
                except Exception as e:
                    logger.warning("Failed to clear HITL state from DB for %s: %s", user_id, e)
                state["admin_guidance"] = user_message
                state["user_input"] += f"\n[ADMIN GUIDANCE: {user_message}]"
                state["user_prompt"] = dict(user_prompt or {})
                state["current_plan"] = []
                state["_energy_gate_cleared"] = False
                state["_resumed_from_hitl"] = True
                heartbeat_task_id = self._extract_heartbeat_origin_task_id_from_state(state)
                if heartbeat_task_id is not None:
                    state["_heartbeat_origin_task_id"] = heartbeat_task_id
                state["hitl_count"] = state.get("hitl_count", 0) + 1
                state["energy_remaining"] = min(state.get("energy_remaining", 0) + 50, 75)
                self._apply_hitl_resume_limits(state, user_id)
                return normalize_state(state)

        state = AgentState.new(user_id=user_id, user_input=user_message, user_prompt=user_prompt).to_dict()
        state["energy_remaining"] = await self._get_predictive_energy_budget_remaining()
        state["user_prompt"] = dict(user_prompt or {})
        if user_id != "heartbeat":
            try:
                active_session = await self._get_active_session()
                get_session_history = getattr(
                    self.ledger_memory,
                    "get_session_chat_history",
                    None,
                )
                if (
                    active_session
                    and active_session.get("id")
                    and callable(get_session_history)
                    and inspect.iscoroutinefunction(get_session_history)
                ):
                    state["chat_history"] = (
                        await get_session_history(
                            user_id=user_id,
                            session_id=int(active_session["id"]),
                            limit=max(_RECENT_CHAT_HISTORY_LIMIT, 20),
                        )
                    )
                    state["active_session"] = active_session
                else:
                    state["chat_history"] = await self.ledger_memory.get_chat_history(
                        user_id,
                        limit=_RECENT_CHAT_HISTORY_LIMIT,
                    )
                    state["active_session"] = None
            except Exception as e:
                logger.warning("Failed to load chat history for %s: %s", user_id, e)
                state["active_session"] = None
        return normalize_state(state)

    async def _save_turn_and_maybe_consolidate(
        self,
        user_id: str,
        user_message: str,
        final_resp: str,
        turn_failed: bool = False,
        session_id: Optional[int] = None,
    ) -> None:
        if user_id == "heartbeat" or turn_failed:
            return
        try:
            save_with_session = getattr(
                self.ledger_memory,
                "save_chat_turn_with_session",
                None,
            )
            if callable(save_with_session) and inspect.iscoroutinefunction(save_with_session):
                await save_with_session(
                    user_id,
                    "user",
                    user_message,
                    session_id=session_id,
                )
                await save_with_session(
                    user_id,
                    "assistant",
                    final_resp,
                    session_id=session_id,
                )
            else:
                await self.ledger_memory.save_chat_turn(user_id, "user", user_message)
                await self.ledger_memory.save_chat_turn(user_id, "assistant", final_resp)
            if session_id:
                increment_turn_count = getattr(
                    self.ledger_memory,
                    "increment_session_turn_count",
                    None,
                )
                if callable(increment_turn_count) and inspect.iscoroutinefunction(increment_turn_count):
                    self._fire_and_forget(increment_turn_count(session_id))
        except Exception as e:
            logger.warning("Failed to save chat turn for %s: %s", user_id, e)

        counts = self._coerce_consolidation_turn_counts()
        trigger_turns_raw = _CONSOLIDATION_TRIGGER_TURNS
        orchestrator_module = sys.modules.get("src.core.orchestrator")
        if orchestrator_module is not None:
            trigger_turns_raw = int(
                getattr(orchestrator_module, "_CONSOLIDATION_TRIGGER_TURNS", trigger_turns_raw)
            )
        trigger_turns = max(1, trigger_turns_raw)
        new_count = int(counts.get(user_id, 0)) + 1
        counts[user_id] = new_count
        counts.move_to_end(user_id)
        self._evict_ordered_mapping_to_size(
            counts,
            _CONSOLIDATION_TURN_COUNT_MAX_USERS,
            label="Consolidation turn counts",
        )
        self._persist_consolidation_turn_count_async(user_id, new_count)

        if new_count >= trigger_turns:
            counts[user_id] = 0
            counts.move_to_end(user_id)
            self._persist_consolidation_turn_count_async(user_id, 0)
            self._fire_and_forget(self._consolidate_memory(user_id))

    async def _persist_chat_turns_impl(
        self,
        user_id: str,
        user_message: str,
        final_resp: str,
        turn_failed: bool = False,
        session_id: Optional[int] = None,
    ) -> None:
        await self._save_turn_and_maybe_consolidate(
            user_id,
            user_message,
            final_resp,
            turn_failed=turn_failed,
            session_id=session_id,
        )

    async def _finalize_user_response_impl(
        self,
        user_id: str,
        user_message: str,
        response: str,
        state: Optional[Dict[str, Any]] = None,
        session_id: Optional[int] = None,
        epic_id: Optional[int] = None,
    ) -> str:
        final_resp = self.cognitive_router.sanitize_response(response)
        resolved_state: Optional[Dict[str, Any]] = state if isinstance(state, dict) else None
        if resolved_state is None:
            hinted_state = getattr(self, "_finalize_state_hint", None)
            if isinstance(hinted_state, dict):
                resolved_state = hinted_state

        turn_failed = bool((resolved_state or {}).get("_turn_failed", False))
        if resolved_state is None:
            turn_failed = bool(getattr(self, "_finalizing_turn_failed", False))

        resolved_session_id = session_id
        resolved_epic_id = epic_id
        if isinstance(resolved_state, dict):
            active_session = resolved_state.get("active_session")
            if isinstance(active_session, dict):
                if resolved_session_id is None:
                    raw_session_id = active_session.get("id")
                    try:
                        resolved_session_id = int(raw_session_id) if raw_session_id else None
                    except (TypeError, ValueError):
                        resolved_session_id = None
                if resolved_epic_id is None:
                    raw_epic_id = active_session.get("epic_id")
                    try:
                        resolved_epic_id = int(raw_epic_id) if raw_epic_id else None
                    except (TypeError, ValueError):
                        resolved_epic_id = None

        if (
            user_id != "heartbeat"
            and not turn_failed
            and isinstance(resolved_state, dict)
            and not bool(resolved_state.get("_supervisor_decision_logged"))
        ):
            plan_for_log = list(resolved_state.get("current_plan", []) or [])
            reasoning_for_log = str(
                resolved_state.get("worker_outputs", {}).get("supervisor_context", "")
                or ""
            )
            if not plan_for_log:
                direct_preview = str(final_resp or "")[:300]
                reasoning_for_log = reasoning_for_log or direct_preview

            log_supervisor_decision = getattr(
                self.ledger_memory,
                "log_supervisor_decision",
                None,
            )
            if callable(log_supervisor_decision):
                self._fire_and_forget(
                    log_supervisor_decision(
                        user_id=str(user_id),
                        user_input=str(user_message or "")[:500],
                        plan_json=json.dumps(
                            [
                                {
                                    "agent": step.get("agent", ""),
                                    "task": str(step.get("task", ""))[:200],
                                    "reason": str(step.get("reason", ""))[:200],
                                }
                                for step in plan_for_log
                                if isinstance(step, dict)
                            ]
                        ),
                        is_direct=not bool(plan_for_log),
                        reasoning=reasoning_for_log[:1000],
                        energy_before=int(resolved_state.get("energy_remaining", 0)),
                        worker_count=len(plan_for_log),
                        session_id=resolved_session_id,
                    )
                )
                resolved_state["_supervisor_decision_logged"] = True

        await self._persist_chat_turns(
            user_id,
            user_message,
            final_resp,
            turn_failed=turn_failed,
            session_id=resolved_session_id,
        )
        self._schedule_response_memory_save(
            user_message,
            final_resp,
            session_id=resolved_session_id,
            epic_id=resolved_epic_id,
        )
        return final_resp


# Compatibility alias for extracted static helpers that reference Orchestrator explicitly.
Orchestrator = _MemoryOpsMixin
