from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_GOAL_PLANNER_SYSTEM_PROMPT = (
    "You are GoalPlanner. Decompose high-level objectives into an execution-ready "
    "Epic/Story/Task hierarchy. "
    "Return ONLY strict JSON and do not include markdown, prose, or commentary. "
    "You are planning only. Never attempt to execute tools or solve tasks."
)


@dataclass
class PlannedTask:
    task_key: str
    title: str
    acceptance_criteria: str = ""
    estimated_energy: int = 10
    priority: int = 5
    dependency_refs: List[str] = field(default_factory=list)


@dataclass
class PlannedStory:
    story_key: str
    title: str
    acceptance_criteria: str = ""
    estimated_energy: int = 20
    priority: int = 5
    tasks: List[PlannedTask] = field(default_factory=list)


@dataclass
class PlannedEpic:
    title: str
    acceptance_criteria: str = ""
    estimated_energy: int = 35
    priority: int = 5


@dataclass
class PlannedHierarchy:
    epic: PlannedEpic
    stories: List[PlannedStory] = field(default_factory=list)


@dataclass
class PlanningResult:
    epic_id: int
    epic_title: str
    story_ids: List[int] = field(default_factory=list)
    task_ids: List[int] = field(default_factory=list)

    @property
    def story_count(self) -> int:
        return len(self.story_ids)

    @property
    def task_count(self) -> int:
        return len(self.task_ids)


class GoalPlanner:
    """Pure planning utility: System 2 JSON decomposition -> ledger persistence."""

    def __init__(self, *, max_context_chars: int = 1600) -> None:
        self.max_context_chars = max(200, int(max_context_chars))

    @staticmethod
    def _extract_json_text(raw_response: str) -> str:
        text = str(raw_response or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()
        return text

    @staticmethod
    def _apply_redactor(
        text: str,
        redactor: Optional[Callable[..., str]],
    ) -> str:
        if redactor is None:
            return str(text or "")
        try:
            return str(redactor(text, allow_sensitive_context=False, max_chars=2000))
        except TypeError:
            return str(redactor(text))

    @staticmethod
    def _coerce_int(value: Any, *, default: int, minimum: int = 0, maximum: int = 1000) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _normalize_ref_key(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _coerce_dependency_refs(raw_value: Any) -> List[str]:
        if raw_value is None:
            return []

        values: Sequence[Any]
        if isinstance(raw_value, (list, tuple, set)):
            values = list(raw_value)
        elif isinstance(raw_value, str):
            stripped = raw_value.strip()
            if not stripped:
                return []
            values = [part.strip() for part in stripped.split(",") if part.strip()]
        else:
            values = [raw_value]

        normalized: List[str] = []
        for item in values:
            ref = str(item).strip()
            if ref and ref not in normalized:
                normalized.append(ref)
        return normalized

    def build_planning_payload(
        self,
        objective: str,
        *,
        context: str = "",
        redactor: Optional[Callable[..., str]] = None,
    ) -> Dict[str, Any]:
        safe_objective = self._apply_redactor(str(objective or ""), redactor).strip()
        safe_context = self._apply_redactor(str(context or ""), redactor).strip()
        if len(safe_context) > self.max_context_chars:
            safe_context = safe_context[: self.max_context_chars].rstrip() + "\n[TRUNCATED_CONTEXT]"

        return {
            "task": "decompose_goal_hierarchy",
            "objective": safe_objective,
            "relevant_context": safe_context,
            "output_format": {
                "epic": {
                    "title": "Epic title",
                    "acceptance_criteria": "Epic completion definition",
                    "priority": 5,
                    "estimated_energy": 35,
                },
                "stories": [
                    {
                        "story_id": "story_1",
                        "title": "Story title",
                        "acceptance_criteria": "Story done definition",
                        "priority": 5,
                        "estimated_energy": 20,
                        "tasks": [
                            {
                                "task_id": "task_a",
                                "title": "Task title",
                                "acceptance_criteria": "Task done definition",
                                "priority": 5,
                                "estimated_energy": 10,
                                "depends_on_ids": [],
                            }
                        ],
                    }
                ],
            },
            "rules": [
                "Return valid JSON only.",
                "Every Story must contain at least one Task.",
                "Task depends_on_ids must reference prior Task ids (e.g., task_a) when dependencies exist.",
                "Only include fields shown in output_format.",
            ],
        }

    def build_system2_planning_messages(
        self,
        objective: str,
        *,
        context: str = "",
        redactor: Optional[Callable[..., str]] = None,
    ) -> List[Dict[str, str]]:
        payload = self.build_planning_payload(objective, context=context, redactor=redactor)
        return [
            {"role": "system", "content": _GOAL_PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ]

    def _parse_task(
        self,
        task_raw: Dict[str, Any],
        *,
        story_key: str,
        task_index: int,
    ) -> PlannedTask:
        title = str(task_raw.get("title") or "").strip()
        if not title:
            raise ValueError(f"Task #{task_index} in '{story_key}' is missing a title")

        task_key = str(task_raw.get("task_id") or task_raw.get("id") or f"{story_key}_task_{task_index}").strip()
        depends_on_raw = task_raw.get("depends_on_ids", task_raw.get("depends_on", []))
        return PlannedTask(
            task_key=task_key,
            title=title,
            acceptance_criteria=str(task_raw.get("acceptance_criteria") or "").strip(),
            estimated_energy=self._coerce_int(task_raw.get("estimated_energy"), default=10, minimum=1, maximum=1000),
            priority=self._coerce_int(task_raw.get("priority"), default=5, minimum=1, maximum=9),
            dependency_refs=self._coerce_dependency_refs(depends_on_raw),
        )

    def _parse_story(self, story_raw: Dict[str, Any], *, story_index: int) -> PlannedStory:
        title = str(story_raw.get("title") or "").strip()
        if not title:
            raise ValueError(f"Story #{story_index} is missing a title")

        story_key = str(story_raw.get("story_id") or story_raw.get("id") or f"story_{story_index}").strip()
        tasks_raw = story_raw.get("tasks")
        if not isinstance(tasks_raw, list) or not tasks_raw:
            raise ValueError(f"Story '{story_key}' must include at least one task")

        tasks: List[PlannedTask] = []
        for task_index, task_raw in enumerate(tasks_raw, start=1):
            if not isinstance(task_raw, dict):
                raise ValueError(f"Task #{task_index} in '{story_key}' must be a JSON object")
            tasks.append(self._parse_task(task_raw, story_key=story_key, task_index=task_index))

        return PlannedStory(
            story_key=story_key,
            title=title,
            acceptance_criteria=str(story_raw.get("acceptance_criteria") or "").strip(),
            estimated_energy=self._coerce_int(story_raw.get("estimated_energy"), default=20, minimum=1, maximum=1000),
            priority=self._coerce_int(story_raw.get("priority"), default=5, minimum=1, maximum=9),
            tasks=tasks,
        )

    def parse_system2_plan_response(
        self,
        response_content: Any,
        *,
        fallback_objective_title: str,
    ) -> PlannedHierarchy:
        if isinstance(response_content, dict):
            decoded = response_content
        else:
            decoded = json.loads(self._extract_json_text(str(response_content or "")))

        if not isinstance(decoded, dict):
            raise ValueError("GoalPlanner response must be a JSON object")

        epic_raw = decoded.get("epic") if isinstance(decoded.get("epic"), dict) else {}
        epic_title = str(epic_raw.get("title") or fallback_objective_title or "").strip()
        if not epic_title:
            raise ValueError("GoalPlanner response must include an epic title")

        stories_raw = decoded.get("stories")
        if not isinstance(stories_raw, list) or not stories_raw:
            raise ValueError("GoalPlanner response must include a non-empty stories array")

        stories: List[PlannedStory] = []
        for story_index, story_raw in enumerate(stories_raw, start=1):
            if not isinstance(story_raw, dict):
                raise ValueError(f"Story #{story_index} must be a JSON object")
            stories.append(self._parse_story(story_raw, story_index=story_index))

        epic = PlannedEpic(
            title=epic_title,
            acceptance_criteria=str(epic_raw.get("acceptance_criteria") or "").strip(),
            estimated_energy=self._coerce_int(epic_raw.get("estimated_energy"), default=35, minimum=1, maximum=2000),
            priority=self._coerce_int(epic_raw.get("priority"), default=5, minimum=1, maximum=9),
        )
        return PlannedHierarchy(epic=epic, stories=stories)

    def _resolve_dependency_ids(
        self,
        task: PlannedTask,
        task_id_by_ref: Dict[str, int],
    ) -> Tuple[List[int], List[str]]:
        resolved_ids: List[int] = []
        unresolved_refs: List[str] = []

        for raw_ref in task.dependency_refs:
            if not raw_ref:
                continue

            stripped = str(raw_ref).strip()
            if stripped.isdigit():
                dep_id = int(stripped)
                if dep_id > 0 and dep_id not in resolved_ids:
                    resolved_ids.append(dep_id)
                continue

            normalized = self._normalize_ref_key(stripped)
            dep_id = task_id_by_ref.get(normalized)
            if dep_id is None:
                unresolved_refs.append(stripped)
                continue
            if dep_id not in resolved_ids:
                resolved_ids.append(dep_id)

        return resolved_ids, unresolved_refs

    async def persist_hierarchy(
        self,
        hierarchy: PlannedHierarchy,
        *,
        ledger_memory: Any,
        origin: str,
        parent_epic_id: Optional[int] = None,
    ) -> PlanningResult:
        epic_id = parent_epic_id
        if epic_id is None:
            epic_id = await ledger_memory.add_objective(
                tier="Epic",
                title=hierarchy.epic.title,
                estimated_energy=hierarchy.epic.estimated_energy,
                origin=origin,
                priority=hierarchy.epic.priority,
                acceptance_criteria=hierarchy.epic.acceptance_criteria,
            )

        story_ids: List[int] = []
        pending_tasks: List[Tuple[int, PlannedTask]] = []
        for story in hierarchy.stories:
            story_id = await ledger_memory.add_objective(
                tier="Story",
                title=story.title,
                estimated_energy=story.estimated_energy,
                origin=origin,
                priority=story.priority,
                parent_id=epic_id,
                acceptance_criteria=story.acceptance_criteria,
            )
            story_ids.append(story_id)
            for task in story.tasks:
                pending_tasks.append((story_id, task))

        task_ids: List[int] = []
        task_id_by_ref: Dict[str, int] = {}

        while pending_tasks:
            progressed = False
            next_pending: List[Tuple[int, PlannedTask]] = []
            unresolved_snapshot: List[Tuple[str, List[str]]] = []

            for story_id, task in pending_tasks:
                depends_on_ids, unresolved = self._resolve_dependency_ids(task, task_id_by_ref)
                if unresolved:
                    next_pending.append((story_id, task))
                    unresolved_snapshot.append((task.task_key, unresolved))
                    continue

                task_id = await ledger_memory.add_objective(
                    tier="Task",
                    title=task.title,
                    estimated_energy=task.estimated_energy,
                    origin=origin,
                    priority=task.priority,
                    parent_id=story_id,
                    depends_on_ids=depends_on_ids,
                    acceptance_criteria=task.acceptance_criteria,
                )
                task_ids.append(task_id)
                task_id_by_ref[self._normalize_ref_key(task.task_key)] = task_id
                task_id_by_ref[self._normalize_ref_key(task.title)] = task_id
                progressed = True

            if not progressed:
                unresolved_fragments = [
                    f"{task_key}: {', '.join(refs)}"
                    for task_key, refs in unresolved_snapshot
                ]
                detail = "; ".join(unresolved_fragments)
                raise ValueError(f"GoalPlanner could not resolve task dependencies ({detail})")

            pending_tasks = next_pending

        return PlanningResult(
            epic_id=int(epic_id),
            epic_title=hierarchy.epic.title,
            story_ids=story_ids,
            task_ids=task_ids,
        )

    async def plan_goal(
        self,
        objective: str,
        *,
        context: str,
        route_to_system_2: Callable[[List[Dict[str, str]]], Awaitable[Any]],
        ledger_memory: Any,
        redactor: Optional[Callable[..., str]] = None,
        origin: str = "System",
        parent_epic_id: Optional[int] = None,
    ) -> PlanningResult:
        objective_text = str(objective or "").strip()
        if not objective_text:
            raise ValueError("GoalPlanner objective must be a non-empty string")

        messages = self.build_system2_planning_messages(
            objective_text,
            context=context,
            redactor=redactor,
        )
        route_result = await route_to_system_2(messages)

        if hasattr(route_result, "status"):
            if getattr(route_result, "status", "") != "ok":
                raise ValueError(
                    f"GoalPlanner System 2 call failed with status: {getattr(route_result, 'status', 'unknown')}"
                )
            response_content = getattr(route_result, "content", "")
        else:
            response_content = route_result

        hierarchy = self.parse_system2_plan_response(
            response_content,
            fallback_objective_title=objective_text,
        )
        return await self.persist_hierarchy(
            hierarchy,
            ledger_memory=ledger_memory,
            origin=origin,
            parent_epic_id=parent_epic_id,
        )
