"""
SkillRegistry — Dynamic plugin loader for AIDEN's tool system.

Scans src/skills/ at boot, reads each SKILL.md for the JSON schema,
and dynamically imports each __init__.py for the async callable.
Broken skill folders are skipped with a warning; healthy ones load normally.
"""

import os
import json
import re
import logging
import importlib.util
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SkillRegistry:
    """
    Loads skills from the skills directory.

    Each skill folder must contain:
      - __init__.py  : defines an async function named after the folder
      - SKILL.md     : contains a ```json schema block

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
        self._load_all()

    # ─────────────────────────────────────────────────────────────
    # Loading
    # ─────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
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
                logger.warning(f"Skipping skill '{entry}': {exc}")

        logger.info(
            f"SkillRegistry ready — {len(self._skills)} skills: "
            f"{', '.join(sorted(self._skills))}"
        )

    def _load_skill(self, name: str, path: str) -> None:
        md_path   = os.path.join(path, "SKILL.md")
        init_path = os.path.join(path, "__init__.py")

        if not os.path.exists(md_path):
            raise FileNotFoundError(f"Missing SKILL.md in {path}")
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"Missing __init__.py in {path}")

        schema = self._parse_schema(md_path, name)

        spec   = importlib.util.spec_from_file_location(f"src.skills.{name}", init_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        fn: Optional[Callable] = getattr(module, name, None)
        if fn is None:
            raise AttributeError(
                f"__init__.py for skill '{name}' must define a function named '{name}'"
            )

        self._skills[name] = {"fn": fn, "schema": schema}

    @staticmethod
    def _parse_schema(md_path: str, skill_name: str) -> Dict[str, Any]:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if not match:
            raise ValueError(f"No ```json schema block found in {md_path}")
        schema = json.loads(match.group(1))
        schema.setdefault("name", skill_name)
        return schema

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def get_schemas(self) -> List[Dict[str, Any]]:
        """All loaded tool schemas, ready to pass to Ollama / Groq."""
        return [s["schema"] for s in self._skills.values()]

    def get_function(self, name: str) -> Optional[Callable]:
        skill = self._skills.get(name)
        return skill["fn"] if skill else None

    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a tool call. Returns a plain string result."""
        skill = self._skills.get(tool_name)
        if skill is None:
            return f"Error: Unknown tool '{tool_name}'."
        try:
            return await skill["fn"](**arguments)
        except TypeError as exc:
            return f"Error: Bad arguments for '{tool_name}': {exc}"
        except Exception as exc:
            logger.error(f"Tool '{tool_name}' raised: {exc}", exc_info=True)
            return f"Error: Tool '{tool_name}' failed — {exc}"

    def register_dynamic(self, name: str, fn: Callable, schema: Dict[str, Any]) -> None:
        """Register a runtime-synthesised tool (System 2 capability synthesis)."""
        self._skills[name] = {"fn": fn, "schema": schema}
        logger.info(f"Dynamic skill '{name}' registered in SkillRegistry")

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
