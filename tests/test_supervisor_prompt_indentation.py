from src.core.prompt_config import build_supervisor_prompt


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def test_supervisor_prompt_preserves_flush_left_template_indentation() -> None:
    charter_xml = """<charter>
<principle id=\"1\">Protect operator data.</principle>
<principle id=\"2\">Prefer truthful tool outputs.</principle>
</charter>"""

    prompt = build_supervisor_prompt(
        charter_text=charter_xml,
        core_mem_str="Known preferences: concise answers",
        archival_block="Recent archival lookup",
        capabilities_str="- web_search: search the web",
        agent_descriptions="- research_agent: gathers evidence",
        sensory_context="CPU: 5%",
        os_name="Windows",
        downloads_dir="downloads",
    )

    assert prompt.startswith("<system_identity>")

    lines = prompt.splitlines()
    assert any(line == "<core_directives>" for line in lines)

    workers_lines = [line for line in lines if "WORKERS:" in line]
    assert workers_lines
    assert all(line == line.lstrip(" ") for line in workers_lines)

    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if not stripped or in_code_block:
            continue
        assert _leading_spaces(line) <= 4
