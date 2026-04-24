from pathlib import Path

from scripts.skill_audit import (
    CATEGORY_HYBRID,
    CATEGORY_INSTRUCTIONAL,
    CATEGORY_LEGACY_JSON,
    audit_skills,
)


def _write_skill_md(path: Path, content: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(content, encoding="utf-8")



def test_skill_audit_categorizes_legacy_hybrid_and_instructional(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"

    # Legacy JSON
    legacy = skills_root / "legacy_web"
    _write_skill_md(
        legacy,
        """# legacy_web

## Schema
```json
{
  "name": "legacy_web",
  "description": "Legacy schema only",
  "parameters": {"type": "object", "properties": {}, "required": []}
}
```
""",
    )
    (legacy / "__init__.py").write_text(
        "async def legacy_web(**kwargs):\n    return \"ok\"\n",
        encoding="utf-8",
    )

    # Hybrid
    hybrid = skills_root / "hybrid_web"
    _write_skill_md(
        hybrid,
        """---
name: hybrid-web
description: Frontmatter + executable.
---

## Schema
```json
{
  "name": "hybrid_web",
  "description": "Schema description",
  "parameters": {"type": "object", "properties": {}, "required": []}
}
```
""",
    )
    (hybrid / "__init__.py").write_text(
        "async def hybrid_web(**kwargs):\n    return \"ok\"\n",
        encoding="utf-8",
    )

    # Instructional
    instructional = skills_root / "cost_guide"
    _write_skill_md(
        instructional,
        """---
name: cost-guide
description: Instructional guidance only.
---

Use this as reference context.
""",
    )

    report = audit_skills(skills_root)

    assert report[CATEGORY_LEGACY_JSON] == ["legacy_web"]
    assert report[CATEGORY_HYBRID] == ["hybrid_web"]
    assert report[CATEGORY_INSTRUCTIONAL] == ["cost_guide"]
