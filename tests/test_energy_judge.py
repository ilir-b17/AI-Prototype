import asyncio

import pytest

from src.core.energy_judge import EnergyJudge
from src.core.llm_router import RouterResult


def _sample_context():
    task = {
        "id": 12,
        "title": "Implement task execution worker",
        "status": "pending",
        "estimated_energy": 15,
        "depends_on_ids": [10],
        "acceptance_criteria": "Worker executes one task and persists output.",
    }
    story = {
        "id": 7,
        "title": "Execution pipeline story",
        "status": "active",
        "acceptance_criteria": "All execution tasks complete and validated.",
    }
    epic = {
        "id": 3,
        "title": "Autonomous orchestration epic",
        "status": "active",
        "acceptance_criteria": "End-to-end orchestration is production ready.",
    }
    return task, story, epic


def test_build_system1_messages_enforce_bounds_and_parent_goal_alignment():
    judge = EnergyJudge()
    task, story, epic = _sample_context()

    messages = judge.build_system1_messages(
        task=task,
        story=story,
        epic=epic,
        additional_context="Keep responses compact.",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "integers from 1 to 10" in messages[0]["content"]

    user_payload = messages[1]["content"]
    assert "Parent Story" in user_payload
    assert "Parent Epic" in user_payload
    assert "expected_value" in user_payload
    assert "direct and significant contribution" in user_payload


def test_parse_system1_response_with_valid_json_scores():
    judge = EnergyJudge()

    result = judge.parse_system1_response(
        {"estimated_effort": 4, "expected_value": 8}
    )

    assert result.estimated_effort == 4
    assert result.expected_value == 8
    assert result.used_fallback is False


def test_parse_system1_response_accepts_fenced_json():
    judge = EnergyJudge()

    raw = """```json
    {"estimated_effort": 3, "expected_value": 9}
    ```"""
    result = judge.parse_system1_response(raw)

    assert result.estimated_effort == 3
    assert result.expected_value == 9
    assert result.used_fallback is False


def test_parse_system1_response_out_of_bounds_uses_conservative_fallback():
    judge = EnergyJudge()

    result = judge.parse_system1_response(
        {"estimated_effort": 0, "expected_value": 11}
    )

    assert result.used_fallback is True
    assert result.estimated_effort == 5
    assert result.expected_value == 5


def test_parse_system1_response_invalid_json_uses_conservative_fallback():
    judge = EnergyJudge()

    result = judge.parse_system1_response("not json")

    assert result.used_fallback is True
    assert result.estimated_effort == 5
    assert result.expected_value == 5


@pytest.mark.asyncio
async def test_evaluate_with_system1_uses_fallback_on_non_ok_router_status():
    judge = EnergyJudge()
    task, story, epic = _sample_context()

    async def fake_route(_messages):
        await asyncio.sleep(0)
        return RouterResult(status="hitl_required", content="")

    result = await judge.evaluate_with_system1(
        task=task,
        story=story,
        epic=epic,
        route_to_system_1=fake_route,
    )

    assert result.used_fallback is True
    assert result.estimated_effort == 5
    assert result.expected_value == 5


@pytest.mark.asyncio
async def test_evaluate_with_system1_parses_router_result_content():
    judge = EnergyJudge()
    task, story, epic = _sample_context()

    async def fake_route(_messages):
        await asyncio.sleep(0)
        return RouterResult(
            status="ok",
            content='{"estimated_effort": 6, "expected_value": 7}',
        )

    result = await judge.evaluate_with_system1(
        task=task,
        story=story,
        epic=epic,
        route_to_system_1=fake_route,
    )

    assert result.used_fallback is False
    assert result.estimated_effort == 6
    assert result.expected_value == 7
