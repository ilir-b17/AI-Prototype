from __future__ import annotations

import json
import logging
import os as _os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)

_LOW_SIGNAL_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|cool|great|sounds good|got it|understood|yes|no)[!. ]*$",
    flags=re.IGNORECASE,
)
_TRANSIENT_LOG_MARKERS = (
    "routing to system",
    "response sent to user",
    "memory added with id",
    "query returned",
    "system 1 response received",
    "system 2 response received",
    "heartbeat:",
)
_VOICE_NOTE_PLACEHOLDER_RE = re.compile(r"^\[Voice note · \d+ bytes · [^\]]+\]$")
_SEMANTIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "i", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "then",
    "there", "this", "to", "was", "were", "will", "with", "you", "your", "we", "they",
    "he", "she", "them", "our", "but", "if", "so", "do", "does", "did", "can", "could",
}
_QUALITY_FACTORS = (
    "novelty",
    "actionability",
    "confidence",
    "recurrence",
    "charter_alignment",
    "contradiction_risk",
    "staleness",
)
_SYSTEM2_SCORING_SYSTEM_PROMPT = (
    "You evaluate candidate memory facts for long-term retention. "
    "Return ONLY strict JSON and do not include markdown or prose. "
    "For each candidate id, provide integer scores 1-5 for: "
    "novelty, actionability, confidence, recurrence, charter_alignment, contradiction_risk, staleness."
)
_CORE_MEMORY_HINTS = (
    "prefer",
    "preference",
    "my name is",
    "i am",
    "identity",
    "user profile",
    "timezone",
    "current focus",
    "system state",
    "host os",
)
_LEDGER_HINTS = (
    "charter",
    "compliance",
    "governance",
    "audit",
    "policy",
    "critic",
    "moral",
    "violation",
)
_NOCTURNAL_CORE_FACTS_MAX = int(_os.getenv("MAX_NOCTURNAL_CORE_FACTS", "60"))


@dataclass
class ConsolidationCandidate:
    source: str
    text: str
    user_id: str = ""
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredCandidate:
    candidate_id: str
    candidate: ConsolidationCandidate
    factor_scores: Dict[str, int]
    q_score: float
    passed_threshold: bool


class NocturnalConsolidationSlice1:
    """Slice 1 + Slice 2 pipeline: extraction, filtering, and quality scoring."""

    def __init__(
        self,
        *,
        min_chars: int = 30,
        max_candidates_per_source: int = 20,
        batch_semantic_jaccard_threshold: float = 0.82,
        vector_distance_threshold: float = 0.08,
    ) -> None:
        self.min_chars = max(1, int(min_chars))
        self.max_candidates_per_source = max(1, int(max_candidates_per_source))
        self.batch_semantic_jaccard_threshold = max(0.0, min(1.0, float(batch_semantic_jaccard_threshold)))
        self.vector_distance_threshold = max(0.0, min(1.0, float(vector_distance_threshold)))

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in _SEMANTIC_STOPWORDS
        }

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _is_error_payload(text: str) -> bool:
        lower = text.lower()
        return lower.startswith("error:") or lower.startswith("[system 1 - error]") or lower.startswith("[system 2 - error]")

    @staticmethod
    def _is_transient_log(text: str) -> bool:
        lower = text.lower()
        return any(marker in lower for marker in _TRANSIENT_LOG_MARKERS)

    @staticmethod
    def _is_low_signal(text: str) -> bool:
        return bool(_LOW_SIGNAL_RE.match(text))

    @staticmethod
    def _normalize_key(text: str) -> str:
        normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def extract_candidates(
        self,
        *,
        user_id: str,
        chat_history: Optional[Sequence[Dict[str, Any]]] = None,
        worker_outputs: Optional[Dict[str, str]] = None,
        critic_feedback: str = "",
        blueprint_entries: Optional[Sequence[Dict[str, Any]]] = None,
        ledger_logs: Optional[Sequence[Dict[str, Any]]] = None,
        now_iso: Optional[str] = None,
    ) -> List[ConsolidationCandidate]:
        timestamp = now_iso or datetime.now().isoformat()
        extracted: List[ConsolidationCandidate] = []

        extracted.extend(self._extract_chat_candidates(user_id, chat_history or [], timestamp))
        extracted.extend(self._extract_worker_candidates(user_id, worker_outputs or {}, timestamp))
        extracted.extend(self._extract_critic_candidates(user_id, critic_feedback, timestamp))
        extracted.extend(self._extract_blueprint_candidates(user_id, blueprint_entries or [], timestamp))
        extracted.extend(self._extract_log_candidates(user_id, ledger_logs or [], timestamp))

        return extracted

    def _extract_chat_candidates(
        self,
        user_id: str,
        chat_history: Sequence[Dict[str, Any]],
        timestamp: str,
    ) -> List[ConsolidationCandidate]:
        candidates: List[ConsolidationCandidate] = []
        for turn in chat_history:
            role = str(turn.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = self._clean_text(str(turn.get("content") or ""))
            if not content:
                continue
            if _VOICE_NOTE_PLACEHOLDER_RE.match(content):
                continue
            candidates.append(
                ConsolidationCandidate(
                    source=f"chat_{role}",
                    text=content,
                    user_id=user_id,
                    timestamp=timestamp,
                    metadata={"role": role},
                )
            )
            if len(candidates) >= self.max_candidates_per_source:
                break
        return candidates

    def _extract_worker_candidates(
        self,
        user_id: str,
        worker_outputs: Dict[str, str],
        timestamp: str,
    ) -> List[ConsolidationCandidate]:
        candidates: List[ConsolidationCandidate] = []
        for worker_name, raw_output in worker_outputs.items():
            if worker_name == "supervisor_context":
                continue
            content = self._clean_text(str(raw_output or ""))
            if not content:
                continue
            candidates.append(
                ConsolidationCandidate(
                    source="worker_output",
                    text=f"{worker_name}: {content}",
                    user_id=user_id,
                    timestamp=timestamp,
                    metadata={"worker": worker_name},
                )
            )
            if len(candidates) >= self.max_candidates_per_source:
                break
        return candidates

    def _extract_critic_candidates(
        self,
        user_id: str,
        critic_feedback: str,
        timestamp: str,
    ) -> List[ConsolidationCandidate]:
        text = self._clean_text(critic_feedback)
        if not text or text.upper() == "PASS":
            return []
        return [
            ConsolidationCandidate(
                source="critic_feedback",
                text=text,
                user_id=user_id,
                timestamp=timestamp,
                metadata={"critic": "fail"},
            )
        ]

    def _extract_blueprint_candidates(
        self,
        user_id: str,
        blueprint_entries: Sequence[Dict[str, Any]],
        timestamp: str,
    ) -> List[ConsolidationCandidate]:
        candidates: List[ConsolidationCandidate] = []
        for entry in blueprint_entries:
            metadata = dict(entry.get("metadata") or {})
            memory_type = str(metadata.get("type") or "").strip().lower()
            doc = self._clean_text(str(entry.get("document") or entry.get("text") or ""))
            if not doc:
                continue
            if memory_type != "system_2_learned_pattern" and "blueprint" not in doc.lower():
                continue
            candidates.append(
                ConsolidationCandidate(
                    source="system_2_blueprint",
                    text=doc,
                    user_id=user_id,
                    timestamp=timestamp,
                    metadata=metadata,
                )
            )
            if len(candidates) >= self.max_candidates_per_source:
                break
        return candidates

    def _extract_log_candidates(
        self,
        user_id: str,
        ledger_logs: Sequence[Dict[str, Any]],
        timestamp: str,
    ) -> List[ConsolidationCandidate]:
        candidates: List[ConsolidationCandidate] = []
        for entry in ledger_logs:
            level = str(entry.get("log_level") or "").upper()
            if level not in {"WARNING", "ERROR", "CRITICAL"}:
                continue
            message = self._clean_text(str(entry.get("message") or ""))
            if not message:
                continue
            candidates.append(
                ConsolidationCandidate(
                    source="ledger_log",
                    text=f"{level}: {message}",
                    user_id=user_id,
                    timestamp=timestamp,
                    metadata={"log_level": level},
                )
            )
            if len(candidates) >= self.max_candidates_per_source:
                break
        return candidates

    def deterministic_filter(self, candidates: Sequence[ConsolidationCandidate]) -> List[ConsolidationCandidate]:
        filtered: List[ConsolidationCandidate] = []
        seen: Set[str] = set()

        for candidate in candidates:
            text = self._clean_text(candidate.text)
            if len(text) < self.min_chars:
                continue
            if self._is_low_signal(text):
                continue
            if self._is_error_payload(text):
                continue
            if self._is_transient_log(text):
                continue

            key = self._normalize_key(text)
            if not key or key in seen:
                continue

            seen.add(key)
            filtered.append(
                ConsolidationCandidate(
                    source=candidate.source,
                    text=text,
                    user_id=candidate.user_id,
                    timestamp=candidate.timestamp,
                    metadata=dict(candidate.metadata),
                )
            )

        return filtered

    async def semantic_deduplicate(
        self,
        candidates: Sequence[ConsolidationCandidate],
        *,
        vector_memory: Optional[Any] = None,
    ) -> List[ConsolidationCandidate]:
        deduped_batch: List[ConsolidationCandidate] = []
        kept_token_sets: List[Set[str]] = []

        for candidate in candidates:
            tokens = self._tokenize(candidate.text)
            if not tokens:
                deduped_batch.append(candidate)
                kept_token_sets.append(set())
                continue

            is_duplicate = any(
                self._jaccard(tokens, prior_tokens) >= self.batch_semantic_jaccard_threshold
                for prior_tokens in kept_token_sets
                if prior_tokens
            )
            if is_duplicate:
                continue

            deduped_batch.append(candidate)
            kept_token_sets.append(tokens)

        if vector_memory is None:
            return deduped_batch

        unique_candidates: List[ConsolidationCandidate] = []
        for candidate in deduped_batch:
            try:
                matches = await vector_memory.query_memory_async(candidate.text, n_results=1)
            except Exception as exc:
                logger.warning("Semantic dedup query failed; keeping candidate: %s", exc)
                unique_candidates.append(candidate)
                continue

            if not matches:
                unique_candidates.append(candidate)
                continue

            distance = matches[0].get("distance")
            if isinstance(distance, (int, float)) and float(distance) <= self.vector_distance_threshold:
                continue

            unique_candidates.append(candidate)

        return unique_candidates

    async def extract_and_filter_candidates(
        self,
        *,
        user_id: str,
        chat_history: Optional[Sequence[Dict[str, Any]]] = None,
        worker_outputs: Optional[Dict[str, str]] = None,
        critic_feedback: str = "",
        blueprint_entries: Optional[Sequence[Dict[str, Any]]] = None,
        ledger_logs: Optional[Sequence[Dict[str, Any]]] = None,
        vector_memory: Optional[Any] = None,
        now_iso: Optional[str] = None,
    ) -> List[ConsolidationCandidate]:
        extracted = self.extract_candidates(
            user_id=user_id,
            chat_history=chat_history,
            worker_outputs=worker_outputs,
            critic_feedback=critic_feedback,
            blueprint_entries=blueprint_entries,
            ledger_logs=ledger_logs,
            now_iso=now_iso,
        )
        deterministic = self.deterministic_filter(extracted)
        return await self.semantic_deduplicate(deterministic, vector_memory=vector_memory)

    @staticmethod
    def _apply_redactor(
        text: str,
        redactor: Optional[Callable[..., str]],
    ) -> str:
        if redactor is None:
            return text
        try:
            return str(redactor(text, allow_sensitive_context=False, max_chars=1400))
        except TypeError:
            return str(redactor(text))

    def build_scoring_payload(
        self,
        candidates: Sequence[ConsolidationCandidate],
        *,
        redactor: Optional[Callable[..., str]] = None,
    ) -> Dict[str, Any]:
        payload_candidates: List[Dict[str, str]] = []
        for index, candidate in enumerate(candidates, start=1):
            payload_candidates.append(
                {
                    "id": f"cand_{index}",
                    "source": str(candidate.source),
                    "text": self._apply_redactor(str(candidate.text), redactor),
                }
            )

        return {
            "task": "score_nocturnal_candidates",
            "output_format": {
                "scores": {
                    "cand_1": {
                        "novelty": 1,
                        "actionability": 1,
                        "confidence": 1,
                        "recurrence": 1,
                        "charter_alignment": 1,
                        "contradiction_risk": 1,
                        "staleness": 1,
                    }
                }
            },
            "rules": [
                "Every factor must be an integer between 1 and 5.",
                "Include every candidate id exactly once.",
                "Return JSON only. No markdown, no comments, no prose.",
            ],
            "candidates": payload_candidates,
        }

    def build_system2_scoring_messages(
        self,
        candidates: Sequence[ConsolidationCandidate],
        *,
        redactor: Optional[Callable[..., str]] = None,
    ) -> List[Dict[str, str]]:
        payload = self.build_scoring_payload(candidates, redactor=redactor)
        return [
            {"role": "system", "content": _SYSTEM2_SCORING_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]

    @staticmethod
    def _extract_json_text(raw_response: str) -> str:
        text = str(raw_response or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()
        return text

    @staticmethod
    def _coerce_factor_score(value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("Boolean values are invalid for factor scores")
        score = int(value)
        if score < 1 or score > 5:
            raise ValueError(f"Factor score {score} is out of bounds (1-5)")
        return score

    def parse_system2_score_response(
        self,
        response_content: Any,
        *,
        candidate_ids: Sequence[str],
    ) -> Dict[str, Dict[str, int]]:
        if isinstance(response_content, dict):
            decoded = response_content
        else:
            decoded = json.loads(self._extract_json_text(str(response_content or "")))

        score_map = decoded.get("scores") if isinstance(decoded, dict) else None
        if score_map is None and isinstance(decoded, dict):
            score_map = decoded
        if not isinstance(score_map, dict):
            raise ValueError("System 2 response must contain a JSON object of scores")

        parsed: Dict[str, Dict[str, int]] = {}
        for candidate_id in candidate_ids:
            raw_factors = score_map.get(candidate_id)
            if not isinstance(raw_factors, dict):
                raise ValueError(f"Missing factor scores for candidate id '{candidate_id}'")

            parsed[candidate_id] = {
                factor: self._coerce_factor_score(raw_factors.get(factor))
                for factor in _QUALITY_FACTORS
            }

        return parsed

    @staticmethod
    def compute_quality_score(factor_scores: Dict[str, int]) -> float:
        return (
            0.30 * factor_scores["novelty"]
            + 0.25 * factor_scores["actionability"]
            + 0.20 * factor_scores["confidence"]
            + 0.15 * factor_scores["recurrence"]
            + 0.10 * factor_scores["charter_alignment"]
            - 0.20 * factor_scores["contradiction_risk"]
            - 0.15 * factor_scores["staleness"]
        )

    def apply_quality_threshold(
        self,
        candidates: Sequence[ConsolidationCandidate],
        scores_by_candidate_id: Dict[str, Dict[str, int]],
        *,
        threshold: float = 3.0,
    ) -> List[ScoredCandidate]:
        scored: List[ScoredCandidate] = []
        for index, candidate in enumerate(candidates, start=1):
            candidate_id = f"cand_{index}"
            factor_scores = scores_by_candidate_id[candidate_id]
            q_score = self.compute_quality_score(factor_scores)
            scored.append(
                ScoredCandidate(
                    candidate_id=candidate_id,
                    candidate=candidate,
                    factor_scores=dict(factor_scores),
                    q_score=q_score,
                    passed_threshold=q_score >= float(threshold),
                )
            )
        return scored

    async def score_candidates_with_system2(
        self,
        candidates: Sequence[ConsolidationCandidate],
        *,
        route_to_system_2: Callable[[List[Dict[str, str]]], Awaitable[Any]],
        redactor: Optional[Callable[..., str]] = None,
        threshold: float = 3.0,
    ) -> List[ScoredCandidate]:
        if not candidates:
            return []

        messages = self.build_system2_scoring_messages(candidates, redactor=redactor)
        route_result = await route_to_system_2(messages)

        response_content: Any
        if hasattr(route_result, "status"):
            if getattr(route_result, "status", "") != "ok":
                raise ValueError(f"System 2 scoring failed with status: {getattr(route_result, 'status', 'unknown')}")
            response_content = getattr(route_result, "content", "")
        else:
            response_content = route_result

        candidate_ids = [f"cand_{index}" for index in range(1, len(candidates) + 1)]
        parsed_scores = self.parse_system2_score_response(
            response_content,
            candidate_ids=candidate_ids,
        )
        return self.apply_quality_threshold(
            candidates,
            parsed_scores,
            threshold=threshold,
        )

    @staticmethod
    def route_storage_bucket(scored_candidate: ScoredCandidate) -> str:
        source = str(scored_candidate.candidate.source or "").lower()
        text = str(scored_candidate.candidate.text or "").lower()

        if source in {"critic_feedback", "ledger_log"}:
            return "ledger"
        if any(hint in text for hint in _LEDGER_HINTS):
            return "ledger"
        if any(hint in text for hint in _CORE_MEMORY_HINTS):
            return "core"
        if source in {"chat_user", "chat_assistant"} and ("prefer" in text or "identity" in text):
            return "core"
        return "vector"

    @staticmethod
    def _merge_unique(existing: Sequence[str], incoming: Sequence[str]) -> List[str]:
        merged: List[str] = list(existing or [])
        for item in incoming:
            normalized = str(item or "").strip()
            if normalized and normalized not in merged:
                merged.append(normalized)
        return merged

    async def _write_core_memory(
        self,
        core_candidates: Sequence[ScoredCandidate],
        *,
        core_memory: Any,
    ) -> int:
        if not core_candidates or core_memory is None:
            return 0

        state = await core_memory.get_all()
        existing_facts = state.get("nocturnal_core_facts", [])
        if isinstance(existing_facts, str):
            existing_facts = [existing_facts]
        if not isinstance(existing_facts, list):
            existing_facts = []

        new_fact_texts = [scored.candidate.text for scored in core_candidates]
        merged_facts = self._merge_unique(existing_facts, new_fact_texts)

        # Keep the most recently appended facts and evict stale head entries.
        if len(merged_facts) > _NOCTURNAL_CORE_FACTS_MAX:
            evicted = len(merged_facts) - _NOCTURNAL_CORE_FACTS_MAX
            merged_facts = merged_facts[evicted:]
            logger.info(
                "nocturnal_core_facts evicted %d stale facts (cap=%d, remaining=%d)",
                evicted, _NOCTURNAL_CORE_FACTS_MAX, len(merged_facts),
            )

        await core_memory.update("nocturnal_core_facts", merged_facts)

        preference_lines = [
            text for text in new_fact_texts
            if "prefer" in text.lower() or "preference" in text.lower()
        ]
        if preference_lines:
            existing_preferences = str(state.get("user_preferences", "") or "").strip()
            preference_update = " | ".join(self._merge_unique([], preference_lines))
            if existing_preferences:
                merged_preferences = self._merge_unique(
                    [segment.strip() for segment in existing_preferences.split("|") if segment.strip()],
                    [preference_update],
                )
                await core_memory.update("user_preferences", " | ".join(merged_preferences))
            else:
                await core_memory.update("user_preferences", preference_update)

        return len(new_fact_texts)

    async def _write_vector_memory(
        self,
        vector_candidates: Sequence[ScoredCandidate],
        *,
        vector_memory: Any,
    ) -> int:
        if not vector_candidates or vector_memory is None:
            return 0

        inserted = 0
        for scored in vector_candidates:
            await vector_memory.add_memory_async(
                text=scored.candidate.text,
                metadata={
                    "type": "nocturnal_insight",
                    "source": scored.candidate.source,
                    "q_score": round(scored.q_score, 4),
                    "candidate_id": scored.candidate_id,
                    "timestamp": scored.candidate.timestamp or datetime.now().isoformat(),
                },
            )
            inserted += 1
        return inserted

    async def _write_ledger_memory(
        self,
        ledger_candidates: Sequence[ScoredCandidate],
        *,
        ledger_memory: Any,
    ) -> int:
        if not ledger_candidates or ledger_memory is None:
            return 0

        from src.memory.ledger_db import LogLevel

        inserted = 0
        for scored in ledger_candidates:
            await ledger_memory.log_event(
                LogLevel.INFO,
                f"Nocturnal governance memory retained: {scored.candidate.text[:220]}",
                {
                    "candidate_id": scored.candidate_id,
                    "source": scored.candidate.source,
                    "q_score": round(scored.q_score, 4),
                    "factor_scores": dict(scored.factor_scores),
                },
            )
            inserted += 1
        return inserted

    async def write_back_scored_candidates(
        self,
        scored_candidates: Sequence[ScoredCandidate],
        *,
        core_memory: Any,
        vector_memory: Any,
        ledger_memory: Any,
        threshold: float = 3.0,
    ) -> Dict[str, int]:
        passing = [candidate for candidate in scored_candidates if candidate.q_score >= float(threshold)]
        if not passing:
            return {
                "passed": 0,
                "stored_core": 0,
                "stored_vector": 0,
                "stored_ledger": 0,
                "stored_total": 0,
            }

        routed = {"core": [], "vector": [], "ledger": []}
        for candidate in passing:
            routed[self.route_storage_bucket(candidate)].append(candidate)

        stored_core = await self._write_core_memory(routed["core"], core_memory=core_memory)
        stored_vector = await self._write_vector_memory(routed["vector"], vector_memory=vector_memory)
        stored_ledger = await self._write_ledger_memory(routed["ledger"], ledger_memory=ledger_memory)

        return {
            "passed": len(passing),
            "stored_core": stored_core,
            "stored_vector": stored_vector,
            "stored_ledger": stored_ledger,
            "stored_total": stored_core + stored_vector + stored_ledger,
        }
