"""
SkillRegistry — Dynamic plugin loader for AIDEN's tool system.

Scans src/skills/ at boot, reads each SKILL.md for the JSON schema,
and dynamically imports each __init__.py for the async callable.
Broken skill folders are skipped with a warning; healthy ones load normally.
"""

import os
import json
import re
import asyncio
import inspect
import logging
import importlib.util
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillManifest:
    name: str
    description: str
    source_path: str
    has_frontmatter: bool
    raw_body: str
    is_executable: bool


class SkillRegistry:
    """
    Loads skills from the skills directory.

        Supported skill types:
            - Executable: SKILL.md + __init__.py + JSON schema block
            - Instructional: SKILL.md only (frontmatter + markdown body)

    Usage:
        registry = SkillRegistry()
        schemas  = registry.get_schemas()       # → list of tool dicts for LLM
        result   = await registry.execute("web_search", {"query": "..."})
    """

    def __init__(self, skills_dir: Optional[str] = None) -> None:
        if skills_dir is None:
            skills_dir = os.path.join(os.path.dirname(__file__), "..", "skills")
        self.skills_dir = os.path.abspath(skills_dir)
        self._skills: Dict[str, Dict[str, Any]] = {}  # name → {fn, schema}
        self._manifests: Dict[str, SkillManifest] = {}
        self._load_errors: List[Tuple[str, str]] = []
        self._dynamic_tool_worker = None
        self._load_all()

    def set_dynamic_tool_worker(self, worker: Any) -> None:
        """Set the isolated worker used for runtime-synthesised tools."""
        self._dynamic_tool_worker = worker

    def _load_all(self) -> None:
        self._load_errors = []
        if not os.path.isdir(self.skills_dir):
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return

        for entry in sorted(os.listdir(self.skills_dir)):
            skill_path = os.path.join(self.skills_dir, entry)
            if not os.path.isdir(skill_path) or entry.startswith("_"):
                continue
            try:
                self._load_skill(entry, skill_path)
                logger.debug(f"Loaded skill: {entry}")
            except Exception as exc:
                self._load_errors.append((entry, str(exc)))
                logger.warning(f"Skipping skill '{entry}': {exc}")

        failed_names = ", ".join(name for name, _error in self._load_errors)
        logger.info(
            "%d skills loaded, %d failed: [%s].",
            len(self._skills),
            len(self._load_errors),
            failed_names,
        )
        logger.info(
            f"SkillRegistry ready — {len(self._skills)} skills: "
            f"{', '.join(sorted(self._skills))}"
        )

    def _load_skill(self, name: str, path: str) -> None:
        md_path   = os.path.join(path, "SKILL.md")
        init_path = os.path.join(path, "__init__.py")

        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Missing SKILL.md in {path}")

        manifest = self._parse_manifest(md_path, name)
        if os.path.exists(init_path):
            schema = self._parse_schema(md_path, name)
            tool_name = str(schema.get("name") or name).strip()
            if not tool_name:
                raise ValueError(f"Schema for '{name}' is missing a valid tool name")

            spec   = importlib.util.spec_from_file_location(f"src.skills.{name}", init_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            fn: Optional[Callable] = getattr(module, name, None)
            if fn is None:
                raise AttributeError(
                    f"__init__.py for skill '{name}' must define a function named '{name}'"
                )

            if tool_name in self._skills:
                raise ValueError(f"Duplicate tool name '{tool_name}' from skill folder '{name}'")

            self._skills[tool_name] = {"fn": fn, "schema": schema, "dynamic": False}
            self._register_manifest(tool_name, manifest, is_executable=True)
            return

        # Instructional skill: valid SKILL.md with no executable payload.
        if not manifest.has_frontmatter:
            raise ValueError(
                f"Missing __init__.py for legacy executable skill in {path}. "
                "Instructional skills must use YAML frontmatter."
            )
        self._register_manifest(manifest.name, manifest, is_executable=False)

    def _register_manifest(self, name: str, manifest: SkillManifest, *, is_executable: bool) -> None:
        canonical_name = str(name or "").strip()
        if not canonical_name:
            raise ValueError("Skill manifest has no valid canonical name")
        if canonical_name in self._manifests:
            raise ValueError(f"Duplicate skill manifest name '{canonical_name}'")

        self._manifests[canonical_name] = SkillManifest(
            name=canonical_name,
            description=manifest.description,
            source_path=manifest.source_path,
            has_frontmatter=manifest.has_frontmatter,
            raw_body=manifest.raw_body,
            is_executable=is_executable,
        )

    @classmethod
    def _parse_manifest(cls, md_path: str, skill_name: str) -> SkillManifest:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        front_matter = cls._extract_front_matter(content, md_path)
        raw_body = cls._extract_markdown_body(content)
        if front_matter is not None:
            metadata = cls._parse_front_matter(front_matter, md_path)
            description = str(metadata.get("description") or "").strip()
            if not description:
                raise ValueError(
                    f"Frontmatter description is required in {md_path}. "
                    "Skipping skill to avoid low-quality routing context."
                )
            manifest_name = str(metadata.get("name") or skill_name).strip() or skill_name
            return SkillManifest(
                name=manifest_name,
                description=description,
                source_path=md_path,
                has_frontmatter=True,
                raw_body=raw_body,
                is_executable=False,
            )

        # Legacy path: no frontmatter, derive catalog metadata from JSON schema.
        schema = cls._parse_schema(md_path, skill_name)
        description = str(schema.get("description") or "").strip()
        if not description:
            raise ValueError(
                f"No skill description found in frontmatter or JSON schema for {md_path}"
            )
        return SkillManifest(
            name=str(schema.get("name") or skill_name),
            description=description,
            source_path=md_path,
            has_frontmatter=False,
            raw_body=raw_body,
            is_executable=True,
        )

    @staticmethod
    def _extract_front_matter(content: str, md_path: str) -> Optional[str]:
        lines = content.splitlines()
        if not lines or lines[0].strip() != "---":
            return None

        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return "\n".join(lines[1:index])

        raise ValueError(f"Malformed frontmatter in {md_path}: missing closing '---'")

    @staticmethod
    def _extract_markdown_body(content: str) -> str:
        lines = content.splitlines()
        if not lines:
            return ""

        if lines[0].strip() != "---":
            return content.strip()

        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return "\n".join(lines[index + 1:]).strip()

        # Malformed frontmatter path is handled by _extract_front_matter.
        return ""

    @classmethod
    def _parse_front_matter(cls, front_matter: str, md_path: str) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        lines = front_matter.splitlines()
        index = 0

        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue

            key, value = cls._parse_front_matter_line(line, md_path)
            if value in {">", "|"}:
                block_lines, index = cls._consume_indented_block(lines, index + 1)
                parsed[key] = cls._format_block_value(block_lines, value)
                continue
            if value == "":
                parsed[key], index = cls._consume_list_items(lines, index + 1)
                continue

            parsed[key] = cls._coerce_scalar(value)
            index += 1

        return parsed

    @staticmethod
    def _parse_front_matter_line(line: str, md_path: str) -> tuple[str, str]:
        if ":" not in line:
            raise ValueError(f"Malformed frontmatter in {md_path}: invalid line {line!r}")
        key, raw_value = line.split(":", 1)
        return key.strip(), raw_value.strip()

    @staticmethod
    def _consume_indented_block(lines: List[str], index: int) -> tuple[List[str], int]:
        block_lines: List[str] = []
        while index < len(lines):
            block_line = lines[index]
            if block_line.startswith("  ") or block_line.startswith("\t"):
                block_lines.append(block_line.strip())
                index += 1
                continue
            if not block_line.strip():
                block_lines.append("")
                index += 1
                continue
            break
        return block_lines, index

    @staticmethod
    def _format_block_value(block_lines: List[str], operator: str) -> str:
        if operator == ">":
            return " ".join(part for part in block_lines if part).strip()
        return "\n".join(block_lines).strip()

    @staticmethod
    def _consume_list_items(lines: List[str], index: int) -> tuple[List[str], int]:
        list_items: List[str] = []
        while index < len(lines):
            stripped = lines[index].strip()
            if not stripped:
                index += 1
                continue
            if not stripped.startswith("- "):
                break
            list_items.append(stripped[2:].strip())
            index += 1
        return list_items, index

    @staticmethod
    def _coerce_scalar(value: str) -> Any:
        stripped = value.strip()
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
            stripped = stripped[1:-1]
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        return stripped

    @staticmethod
    def _parse_schema(md_path: str, skill_name: str) -> Dict[str, Any]:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        code_block = re.search(r"```json\s*", content)
        if not code_block:
            raise ValueError(f"No ```json schema block found in {md_path}")
        decoder = json.JSONDecoder()
        try:
            schema, _ = decoder.raw_decode(content, code_block.end())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON schema in {md_path}: {exc}") from exc
        schema.setdefault("name", skill_name)
        return schema

    def get_schemas(self) -> List[Dict[str, Any]]:
        """All loaded tool schemas, ready to pass to Ollama / Groq."""
        return [s["schema"] for s in self._skills.values()]

    def get_load_errors(self) -> List[Tuple[str, str]]:
        """Return skill folders that failed to load and their error messages."""
        return list(self._load_errors)

    def get_skill_catalog(self) -> List[Dict[str, str]]:
        """Ultra-lean Level-1 catalog: tool name + description only."""
        catalog: List[Dict[str, str]] = []
        for name in sorted(self._manifests):
            manifest = self._manifests[name]
            catalog.append({"name": manifest.name, "description": manifest.description})
        return catalog

    def get_executable_skill_catalog(self) -> List[Dict[str, str]]:
        catalog: List[Dict[str, str]] = []
        for name in sorted(self._manifests):
            manifest = self._manifests[name]
            if manifest.is_executable:
                catalog.append({"name": manifest.name, "description": manifest.description})
        return catalog

    def is_executable_skill(self, name: str) -> bool:
        manifest = self._manifests.get(name)
        return bool(manifest and manifest.is_executable)

    def get_skill_body(self, name: str) -> str:
        manifest = self._manifests.get(name)
        return manifest.raw_body if manifest else ""

    def get_function(self, name: str) -> Optional[Callable]:
        skill = self._skills.get(name)
        return skill["fn"] if skill else None

    @staticmethod
    async def _touch_dynamic_tool_usage(tool_name: str) -> None:
        try:
            from src.core.runtime_context import get_ledger

            ledger = get_ledger()
            touch_fn = getattr(ledger, "touch_tool_last_used", None)
            if callable(touch_fn):
                await touch_fn(tool_name)
        except Exception as exc:
            logger.debug("Could not update last_used_at for dynamic tool %s: %s", tool_name, exc)

    async def _execute_dynamic_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        worker = self._dynamic_tool_worker
        if worker is None:
            return f"Error: Dynamic tool worker is not available for '{tool_name}'."
        response = await worker.call_tool(tool_name, arguments or {})
        await self._touch_dynamic_tool_usage(tool_name)
        if not response.get("ok"):
            return f"Error: Dynamic tool '{tool_name}' failed — {response.get('error', 'unknown worker error')}"
        return str(response.get("result", ""))

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a tool call. Returns a plain string result."""
        alias_map = {
            "extract_text_from_file": "extract_pdf_text",
            "extract_text": "extract_pdf_text",
            "extract_pdf_text_from_file": "extract_pdf_text",
            "read_pdf": "extract_pdf_text",
            "pdf_extract": "extract_pdf_text",
            "search_web": "web_search",
            "google_search": "web_search",
            "run_python": "execute_python_sandbox",
            "python_execute": "execute_python_sandbox",
            "run_command": "run_terminal_command",
            "execute_command": "run_terminal_command",
            "file_system": "manage_file_system",
            "read_file": "manage_file_system",
            "write_file": "manage_file_system",
            "stock_price": "get_stock_price",
            "fetch_stock": "get_stock_price",
            "analyze_csv": "analyze_table_file",
            "analyze_excel": "analyze_table_file",
            "extract_article": "extract_web_article",
            "fetch_article": "extract_web_article",
            "scrape_web": "extract_web_article",
        }
        tool_name = alias_map.get(tool_name, tool_name)

        skill = self._skills.get(tool_name)
        if skill is None:
            return f"Error: Unknown tool '{tool_name}'."
        if skill.get("dynamic"):
            return await self._execute_dynamic_tool(tool_name, arguments)
        if tool_name == "manage_file_system":
            if "path" in arguments and "file_path" not in arguments:
                arguments["file_path"] = arguments.pop("path")
        try:
            result = skill["fn"](**arguments)
            if asyncio.iscoroutine(result):
                return await result
            if inspect.isasyncgen(result):
                # Drain async generator and concatenate string chunks
                parts = []
                async for chunk in result:
                    parts.append(str(chunk))
                return "\n".join(parts)
            return result
        except TypeError as exc:
            return f"Error: Bad arguments for '{tool_name}': {exc}"
        except Exception as exc:
            logger.error(f"Tool '{tool_name}' raised: {exc}", exc_info=True)
            return f"Error: Tool '{tool_name}' failed — {exc}"

    def register_dynamic(self, name: str, schema: Dict[str, Any]) -> None:
        """Register a runtime-synthesised tool schema backed by the worker."""
        self._skills[name] = {"fn": None, "schema": schema, "dynamic": True}
        self._manifests[name] = SkillManifest(
            name=name,
            description=str(schema.get("description") or "Runtime dynamic tool"),
            source_path="[runtime_dynamic]",
            has_frontmatter=False,
            raw_body="",
            is_executable=True,
        )
        logger.info(f"Dynamic skill '{name}' registered in SkillRegistry")

    def __len__(self) -> int:
        return len(self._skills)

    def get_skill_names(self) -> List[str]:
        """Return a sorted list of all registered skill names."""
        return sorted(self._skills.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._skills
