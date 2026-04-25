# AIDEN Hardening Report

## Scope
This report summarizes hardening work completed across Slice 1, Slice 2, and Slice 3.

## Slice 1 - Deadlocks and Data Loss
### Fix summary
- Resolved tool-approval lock re-entrancy deadlock by extracting lock-held turn execution and preventing recursive lock acquisition during approved synthesized tool retry.
- Added explicit shutdown drain for orchestrator fire-and-forget background tasks before resource closure, with timeout warning behavior.

### Files touched
- src/core/orchestrator.py
- src/interfaces/telegram_bot.py
- tests/test_orchestrator_lock_reentrancy.py
- tests/test_shutdown_drains_background_tasks.py

### Tests added (count)
- 3 tests total
- tests/test_orchestrator_lock_reentrancy.py: 1
- tests/test_shutdown_drains_background_tasks.py: 2

## Slice 2 - State Routing and Prompt Integrity
### Fix summary
- Routed heartbeat-origin HITL/MFA state to admin user context, preserved heartbeat-origin task ids, and finalized resumed heartbeat tasks via explicit completion/failure handling.
- Refactored supervisor prompt construction to use a module-level dedented template with format placeholders, preventing post-interpolation indentation drift.

### Files touched
- src/core/orchestrator.py
- src/core/prompt_config.py
- tests/test_heartbeat_hitl_routing.py
- tests/test_supervisor_prompt_indentation.py

### Tests added (count)
- 2 tests total
- tests/test_heartbeat_hitl_routing.py: 1
- tests/test_supervisor_prompt_indentation.py: 1

## Slice 3 - Logic Foot-guns
### Fix summary
- Hardened inline tool-call extraction so only dominant-content JSON (raw or fenced JSON) is executable, rejecting prose-embedded and wrapper-embedded examples.
- Added inline execution guardrails:
  - per-router streak tracking (`_inline_tool_call_streak`),
  - per-route hard cap of one inline tool execution,
  - streak reset on normal text path.
- Replaced per-turn predictive energy replenishment with wall-clock replenishment based on `ENERGY_REPLENISH_PER_HOUR`, preserving fractional accumulation over time.
- Added deprecated compatibility handling for `ENERGY_REPLENISH_PER_TURN` as no-op with one-time startup warning.

### Files touched
- src/core/llm_router.py
- src/core/orchestrator.py
- tests/test_inline_tool_call_guards.py
- tests/test_energy_replenishment_walltime.py
- tests/test_router_interceptor.py
- tests/test_hardening.py

### Tests added (count)
- 9 tests total
- tests/test_inline_tool_call_guards.py: 4
- tests/test_energy_replenishment_walltime.py: 5

## Adjacent issues noticed and deferred
- No additional adjacent production code issues were deferred during these slices.

## Backward compatibility notes
- `ENERGY_REPLENISH_PER_TURN` remains readable from environment for backward compatibility but is now deprecated and treated as a no-op.
- Operators should use `ENERGY_REPLENISH_PER_HOUR` (default `30`) for predictive energy replenishment tuning.
