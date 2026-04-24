from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.orchestrator import Orchestrator
from src.core.skill_manager import SkillRegistry


def _write_skill(skill_dir: Path, skill_name: str, skill_markdown: str) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(skill_markdown, encoding="utf-8")
    (skill_dir / "__init__.py").write_text(
        (
            f"async def {skill_name}(**kwargs):\n"
            "    return \"ok\"\n"
        ),
        encoding="utf-8",
    )


def test_parser_reads_google_style_frontmatter_and_keeps_legacy_schema(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(
        skills_dir / "web_search",
        "web_search",
        """---
name: web_search
description: Frontmatter routing description for catalog discovery.
---

# web_search

## Schema

```json
{
  "name": "web_search",
  "description": "Schema-level description retained for tool call interfaces.",
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

    schemas = registry.get_schemas()
    catalog = registry.get_skill_catalog()

    assert len(registry) == 1
    assert schemas == [
        {
            "name": "web_search",
            "description": "Schema-level description retained for tool call interfaces.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]
    assert catalog == [
        {
            "name": "web_search",
            "description": "Frontmatter routing description for catalog discovery.",
        }
    ]


def test_parser_rejects_malformed_frontmatter_without_crashing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(
        skills_dir / "broken_skill",
        "broken_skill",
        """---
name broken_skill
description: This should be skipped because frontmatter is malformed.
---

## Schema
```json
{
  "name": "broken_skill",
  "description": "Schema is valid but should not load due to malformed frontmatter.",
  "parameters": {"type": "object", "properties": {}, "required": []}
}
```
""",
    )

    caplog.set_level("WARNING")
    registry = SkillRegistry(skills_dir=str(skills_dir))

    assert len(registry) == 0
    assert registry.get_schemas() == []
    assert registry.get_skill_catalog() == []
    assert any("Skipping skill 'broken_skill'" in record.getMessage() for record in caplog.records)


def test_parser_rejects_frontmatter_missing_description(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(
        skills_dir / "missing_description_skill",
        "missing_description_skill",
        """---
name: missing_description_skill
---

## Schema
```json
{
  "name": "missing_description_skill",
  "description": "Schema description exists but frontmatter description is mandatory when frontmatter is present.",
  "parameters": {"type": "object", "properties": {}, "required": []}
}
```
""",
    )

    caplog.set_level("WARNING")
    registry = SkillRegistry(skills_dir=str(skills_dir))

    assert len(registry) == 0
    assert registry.get_skill_catalog() == []
    assert any("Frontmatter description is required" in record.getMessage() for record in caplog.records)


def test_capabilities_string_uses_ultra_lean_name_description_catalog() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.cognitive_router = MagicMock()
    orchestrator.cognitive_router.registry = MagicMock()
    orchestrator.cognitive_router.registry.get_skill_catalog.return_value = [
        {"name": "web_search", "description": "Search current web sources."},
        {"name": "get_system_info", "description": "Return local date/time and platform details."},
    ]
    orchestrator.cognitive_router.registry.get_schemas.return_value = [
        {
            "name": "schema_only_fallback",
            "description": "Schema description",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]

    capabilities = Orchestrator._get_capabilities_string(orchestrator)

    assert "web_search: Search current web sources." in capabilities
    assert "get_system_info: Return local date/time and platform details." in capabilities
    assert "schema_only_fallback" not in capabilities
    assert "parameters" not in capabilities
    assert "required" not in capabilities
    assert '"type": "object"' not in capabilities
