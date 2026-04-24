from pathlib import Path

import pytest

from src.core.skill_manager import SkillRegistry


def _write_executable_skill(skill_dir: Path, skill_name: str, markdown: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(markdown, encoding="utf-8")
    (skill_dir / "__init__.py").write_text(
        (
            f"async def {skill_name}(**kwargs):\n"
            "    return \"ok\"\n"
        ),
        encoding="utf-8",
    )



def _write_instructional_skill(skill_dir: Path, markdown: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(markdown, encoding="utf-8")



def test_instructional_skill_is_catalogued_but_not_executable(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_instructional_skill(
        skills_dir / "cost_optimization",
        """---
name: cost-optimization
description: Suggests cloud cost optimization strategies.
---

This is instructional context only.
""",
    )

    registry = SkillRegistry(skills_dir=str(skills_dir))

    assert registry.get_schemas() == []
    assert registry.get_executable_skill_catalog() == []
    assert registry.get_skill_catalog() == [
        {
            "name": "cost-optimization",
            "description": "Suggests cloud cost optimization strategies.",
        }
    ]
    assert registry.is_executable_skill("cost-optimization") is False


@pytest.mark.asyncio
async def test_instructional_skill_cannot_be_executed_as_tool(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_instructional_skill(
        skills_dir / "cost_optimization",
        """---
name: cost-optimization
description: Suggests cloud cost optimization strategies.
---

This is instructional context only.
""",
    )

    registry = SkillRegistry(skills_dir=str(skills_dir))
    response = await registry.execute("cost-optimization", {})
    assert response.startswith("Error: Unknown tool")



def test_hybrid_skill_is_executable_and_keeps_frontmatter_catalog(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_executable_skill(
        skills_dir / "web_search",
        "web_search",
        """---
name: web-search
description: Frontmatter catalog text for executable web search.
---

## Schema
```json
{
  "name": "web_search",
  "description": "Schema tool description.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"}
    },
    "required": ["query"]
  }
}
```
""",
    )

    registry = SkillRegistry(skills_dir=str(skills_dir))

    assert registry.is_executable_skill("web_search") is True
    assert [schema["name"] for schema in registry.get_schemas()] == ["web_search"]
    assert registry.get_executable_skill_catalog() == [
        {
            "name": "web_search",
            "description": "Frontmatter catalog text for executable web search.",
        }
    ]
