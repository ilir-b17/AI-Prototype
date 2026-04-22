# AIDEN → Claude Dispatch Model: Alignment Roadmap

**Status**: Planning  
**Reference architecture**: Anthropic Claude Agent SDK (claude-agent-sdk)  
**Goal**: Evolve AIDEN's static two-worker orchestration into a fully dynamic, context-isolated, parallel subagent dispatch system — while preserving the Charter, local-first directive, and MFA/HITL safety rails, and without overrunning local hardware.

---

## Current Architecture Snapshot

```
User message
     │
     ▼
supervisor_node          ← System 1 (Ollama/gemma4:e4b), falls back to System 2 (Groq)
     │                     Parses WORKERS: ["research_agent", "coder_agent"] via regex
     ▼
execute_workers_node     ← Hardcoded: "research" → research_agent(), "coder" → coder_agent()
     │                     Sequential. Shared AgentState passed to every worker.
     ▼
critic_node              ← Always called, even for trivial replies. Uses System 2 (Groq).
     │
     ▼
Final response → Telegram
```

**Key constraints to keep at all times:**
- Charter Directive 1.3: Local model (System 1) must be tried first in every LLM call
- Charter Directive 1.2: MFA required before any write/delete/modify core operation
- Charter Objective 2.4: Critic mandate (3-strike self-correction before HITL)
- Charter Objective 2.5: Capability gap → tool synthesis → Admin approval
- All sensitive data must stay local; System 2 (cloud) only handles orchestration reasoning

---

## Feasibility Verdict

This roadmap is beneficial and should materially improve the current system if implemented with hardware-aware guardrails.

**Highest-value improvements for the current system:**
- Context isolation reduces prompt size and lowers local inference time on every delegated worker call.
- Selective Critic removes unnecessary Groq round-trips on trivial/direct replies.
- Dynamic agent registration removes brittle hardcoded worker routing and makes the orchestration surface easier to extend.
- Model-per-agent routing lets high-value reasoning paths use cloud selectively while keeping the default path local-first.

**Primary constraint:**
- Local parallelism is not truly free. Even with `gemma4:e4b`, multiple concurrent Ollama inferences on consumer hardware are likely to serialize internally or thrash RAM/VRAM.
- Phase 3 must therefore include a System 1 concurrency gate so `asyncio.gather()` does not melt the machine while trying to speed it up.

**Pragmatic recommendation:**
- Pull Context Isolation forward for immediate performance gains.
- Land Selective Critic before true worker parallelism.
- Treat parallel execution as cloud-parallel / local-serialized unless hardware proves otherwise.

---

## Gap Analysis vs Claude Agent SDK Dispatch Model

| Dimension | Current AIDEN | Target (Dispatch Model) |
|-----------|--------------|------------------------|
| Worker selection | Regex parse of `WORKERS: [...]`; only 2 names recognized | LLM auto-selects subagent by matching task to agent `description` fields |
| Worker count | 2 fixed (research, coder) | Unlimited; loaded from filesystem at startup |
| Context isolation | Workers receive full `AgentState` including all chat history | Each subagent gets a fresh conversation; only its final answer returns to parent |
| Parallelism | Sequential `for task in plan` | `asyncio.gather()` for independent agents |
| Agent definition | Hardcoded prompt strings in `orchestrator.py` | `AgentDefinition(name, description, system_prompt, allowed_tools, model)` loaded from `AGENT.md` files |
| Tool ↔ Agent distinction | Skills = tools only (atomic Python functions) | Two layers: skills = atomic tools; agents = LLM-driven workers with own context |
| Critic invocation | Forced on every turn regardless of content | Selective — only when agents produced output; skipped for direct supervisor replies |
| Model per agent | All workers share same System 1 or System 2 | Each agent can specify preferred model (local vs cloud), cost-tiered |
| Session resumption | None | Agents can be resumed by ID across requests |
| MCP | None | Native MCP server registration per agent |

---

## Hardware Reality Check

The roadmap is architecturally sound, but the machine profile matters:

- The local-first path is still the default and should remain so for Charter compliance.
- `gemma4:e4b` is a more realistic default System 1 model for constrained local hardware than `gemma4:26b`.
- Local worker parallelism must be treated as a scheduling problem, not as guaranteed throughput.

**Required mitigation for Phase 3:**

```python
# src/core/llm_router.py
self._system_1_semaphore = asyncio.Semaphore(
    int(os.getenv("SYSTEM_1_MAX_CONCURRENCY", "1"))
)

async def route_to_system_1(...):
    async with self._system_1_semaphore:
        ...
```

This preserves architectural parallelism at the orchestrator level while ensuring local System 1 requests queue politely. Cloud System 2 requests can still run concurrently.

---

## Phase 1 — Dynamic Agent Registry

**Effort**: 1 sprint (~1 week)  
**Risk**: Low — additive; existing workers continue to function as before  
**Goal**: Break out of the 2-worker hardcoded model. Make workers first-class loadable definitions, not functions baked into `orchestrator.py`.

### What to Build

#### 1.1 — `AgentDefinition` dataclass (`src/core/agent_definition.py`)

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class AgentDefinition:
    name: str                          # e.g. "research_agent"
    description: str                   # When should supervisor delegate here?
    system_prompt: str                 # Full system prompt for this agent
    allowed_tools: List[str]           # Skill names this agent may call
    preferred_model: str = "system_1"  # "system_1" | "system_2" | "local:<model>"
    max_tool_calls: int = 5
    energy_cost: int = 15
```

#### 1.2 — `AgentRegistry` loader (`src/core/agent_registry.py`)

- Scan `src/agents/*/AGENT.md` at startup (mirrors how `SkillRegistry` scans `src/skills/*/SKILL.md`)
- Parse YAML front-matter from each `AGENT.md` for `name`, `description`, `allowed_tools`, `preferred_model`
- The body of the markdown becomes `system_prompt`
- Register built-in agents (research, coder) as fallback if no files found

**`AGENT.md` format:**

```markdown
---
name: research_agent
description: >
  Use this agent when the task requires searching memory, gathering background
  information, or retrieving facts before acting. Best for: "look up", "find",
  "what do you know about", "search for".
allowed_tools:
  - search_archival_memory
  - web_search
  - extract_web_article
preferred_model: system_1
max_tool_calls: 4
energy_cost: 15
---

You are the Research Agent for AIDEN. Your role is to gather and surface
relevant information from archival memory and the web.

Rules:
- Search archival memory first before going to the web
- Summarise findings concisely; do not pad with unnecessary detail
- Never modify memory or ledger — read only
- Report exactly what you found; do not fabricate
```

#### 1.3 — Supervisor prompt update

Replace the hardcoded `WORKERS: ["research_agent", "coder_agent"]` instruction with a dynamic block that lists all registered agents and their descriptions:

```python
agent_descriptions = "\n".join(
    f"- {a.name}: {a.description}"
    for a in self.agent_registry.all()
)
# Inject into system prompt under <available_agents>
```

The supervisor is then instructed: *"If delegation is needed, output `WORKERS: [\"agent_name\"]`. Use only the agent names listed above."*

#### 1.4 — `execute_workers_node` refactor

Replace the hardcoded `if "research" in task: ... elif "coder" in task: ...` with:

```python
for task in state["current_plan"]:
    agent_def = self.agent_registry.get(task)
    if agent_def:
        state = await self._run_agent(agent_def, state)
    else:
        logger.warning(f"Unknown agent requested: {task!r}")
```

#### 1.5 — `_run_agent()` generic runner

Extract the common pattern from `research_agent()` and `coder_agent()` into a single `_run_agent(agent_def, state)` method. Both existing methods become thin wrappers or are deleted.

### Files to Create
- `src/core/agent_definition.py`
- `src/core/agent_registry.py`
- `src/agents/research_agent/AGENT.md`
- `src/agents/coder_agent/AGENT.md`

### Files to Modify
- `src/core/orchestrator.py` — `execute_workers_node`, `_get_capabilities_string`, `__init__`
- `src/core/prompt_config.py` — add agent descriptions to supervisor prompt builder

### Tests to Add
- `tests/test_agent_registry.py` — loader, fallback, unknown agent name

### Trade-off Note

This phase improves extensibility, but it does not produce the biggest near-term latency win. Prompt size will also grow with the number of registered agents, so once the agent count climbs beyond roughly 8–10, the supervisor should eventually retrieve a smaller candidate set instead of listing every agent every time.

---

## Phase 2 — Context Isolation

**Effort**: 1 sprint  
**Risk**: Medium — changes how every worker receives context  
**Goal**: Workers stop seeing each other's outputs and the full chat history. Each agent call starts with a clean context, receiving only what the supervisor explicitly passes.

### The Problem

Right now `_run_agent` passes `state["chat_history"]` to every worker. As the conversation grows, each worker call consumes more tokens, slows down, and workers see other workers' reasoning. This is the biggest token-cost and quality degradation source.

### Recommended rollout

Before the full dynamic registry is complete, apply the same handoff discipline to the existing `research_agent` and `coder_agent` paths. That produces immediate speed and quality gains without waiting for the full registry refactor.

### What to Build

#### 2.1 — Isolated message builder

Each agent call builds its own isolated `messages` list containing **only**:
1. The agent's own `system_prompt` (from `AgentDefinition`)
2. A context handoff block from the supervisor:
   ```
   <supervisor_handoff>
   User request: {user_input}
   Relevant context: {summary_of_prior_worker_outputs_if_any}
   Your task: {specific_task_extracted_from_plan}
   </supervisor_handoff>
   ```
3. No chat history, no other workers' outputs, no `AgentState` dump

```python
async def _run_agent(self, agent_def: AgentDefinition, state: dict) -> dict:
    handoff = self._build_handoff(agent_def.name, state)
    messages = [
        {"role": "system", "content": agent_def.system_prompt},
        {"role": "user",   "content": handoff},
    ]
    result = await self._route_agent_request(messages, agent_def)
    state["worker_outputs"][agent_def.name] = result.content if result else "Error"
    return state
```

#### 2.2 — `_build_handoff()` method

Constructs a minimal, targeted context string for a worker. Rules:
- Always includes the original user request
- Includes prior worker outputs **only if** the current agent has a declared data dependency (expressed in `AGENT.md` as `depends_on: [research_agent]`)
- Never includes raw `chat_history` — only the `conversation_summary` from `core_memory` if needed

#### 2.3 — `AgentDefinition` update

Add optional `depends_on: List[str]` field. The executor checks this before building the handoff:
- If `depends_on: [research_agent]` and `research_agent` has already run, its output is included
- If the dependency hasn't run yet, it is queued first (enforces execution order for dependent agents)

### Files to Modify
- `src/core/orchestrator.py` — `_run_agent`, new `_build_handoff`
- `src/core/agent_definition.py` — add `depends_on` field
- `src/core/state_model.py` — no schema changes needed

### Expected Impact
- ~60–80% reduction in tokens sent per worker call
- Workers produce higher-quality output (no cross-contamination)
- Enables Phase 3 (parallelism) because isolated agents have no shared mutable read dependencies

---

## Phase 3 — Parallel Worker Execution

**Effort**: 1 sprint  
**Risk**: Low (once Phase 2 is done — isolation is a prerequisite)  
**Goal**: Independent workers run concurrently with `asyncio.gather`, cutting multi-worker response time to the duration of the slowest single worker.

**Important qualification:** for local System 1 workers, "parallel" means orchestrator-level parallel scheduling guarded by a local concurrency semaphore. Only cloud workers should be expected to execute truly concurrently by default.

### What to Build

#### 3.0 — System 1 concurrency gate

Add a router-level semaphore around local model calls:

```python
# CognitiveRouter.__init__
self._system_1_semaphore = asyncio.Semaphore(
    int(os.getenv("SYSTEM_1_MAX_CONCURRENCY", "1"))
)

# CognitiveRouter.route_to_system_1
async with self._system_1_semaphore:
    ...
```

This ensures:
- local workers do not compete for the same VRAM/RAM at once
- `asyncio.gather()` stays safe even when multiple workers prefer System 1
- System 2 agents can still run in parallel where it actually helps

#### 3.1 — Dependency graph from `depends_on`

Before execution, build a simple two-tier execution plan:

```python
def _build_execution_plan(self, agent_names: List[str]) -> tuple[List[str], List[str]]:
    """Returns (independent_agents, dependent_agents) based on depends_on declarations."""
    definitions = [self.agent_registry.get(n) for n in agent_names]
    independent = [d for d in definitions if not d.depends_on]
    dependent   = [d for d in definitions if d.depends_on]
    return independent, dependent
```

#### 3.2 — Parallel executor in `execute_workers_node`

```python
independent, dependent = self._build_execution_plan(state["current_plan"])

# Run independent agents concurrently
if independent:
    results = await asyncio.gather(
        *[self._run_agent(a, state) for a in independent],
        return_exceptions=True,
    )
    for agent_def, result in zip(independent, results):
        if isinstance(result, Exception):
            logger.error(f"Agent {agent_def.name} failed: {result}")
            state["worker_outputs"][agent_def.name] = f"Error: {result}"
        else:
            state = result  # merge outputs

# Run dependent agents sequentially after independents complete
for agent_def in dependent:
    state = await self._run_agent(agent_def, state)
```

#### 3.2a — Partial-failure state handling

Parallel execution must not discard successful worker output if another worker returns `RequiresHITLError`, `RequiresMFAError`, or another blocked result. The node should merge successful results first, then persist the blocked state transition.

#### 3.3 — Energy budget for parallel agents

When agents run in parallel, deduct combined energy upfront before dispatching:

```python
total_cost = sum(a.energy_cost for a in independent)
state = self._deduct_energy(state, total_cost, f"parallel: {[a.name for a in independent]}")
```

### Files to Modify
- `src/core/orchestrator.py` — `execute_workers_node`, new `_build_execution_plan`

### Performance Impact
- Two independent cloud-capable agents can approach the slowest single-agent duration instead of summing linearly.
- Two local System 1 agents will still queue behind the semaphore, but the orchestration layer remains ready for mixed local/cloud batches and future hardware upgrades.
- At 3+ agents, gains depend heavily on the ratio of local vs cloud workers.

---

## Phase 4 — Selective Critic

**Effort**: 0.5 sprint  
**Risk**: Low  
**Goal**: Eliminate the forced Groq API call on every turn. The Critic should only run when there is substantive agent output to evaluate.

### Current Problem

The Critic always runs via `route_to_system_2` regardless of whether:
- The supervisor answered directly (no workers ran)
- The response was trivial (e.g., "Hi there!")
- The worker output was already known-good (e.g., a tool result)

This wastes ~1–2s and one Groq API call per message.

### What to Build

#### 4.1 — Skip condition in `critic_node`

```python
async def critic_node(self, state):
    output_to_eval = self._get_output_to_evaluate(state)

    # Skip if no meaningful worker output exists
    if not output_to_eval or not state.get("worker_outputs"):
        state["critic_feedback"] = "PASS"
        return state

    # Skip if output is short (direct supervisor reply to simple query)
    if len(output_to_eval) < 150 and not state.get("current_plan"):
        state["critic_feedback"] = "PASS"
        return state

    # ... existing critic logic ...
```

#### 4.2 — Critic as an opt-in agent (future)

Longer term, the Critic can be expressed as an `AgentDefinition` in `src/agents/critic_agent/AGENT.md` with:
- `preferred_model: system_2` (Groq for fast structured output)
- `allowed_tools: []` (no tool access — text-only evaluation)
- The supervisor explicitly includes `"critic_agent"` in `WORKERS: [...]` only when evaluating complex/high-stakes output

This aligns with the Claude Dispatch pattern where the evaluator is a specialized subagent, not a mandatory node.

#### 4.3 — Charter compliance check

The Critic currently checks against a simplified charter excerpt. Upgrade it to use a structured evaluation rubric stored in `data/core_memory.json` under `critic_rubric`, so the criteria can evolve without a code change.

### Files to Modify
- `src/core/orchestrator.py` — `critic_node`
- `src/core/prompt_config.py` — add critic evaluation prompt builder

---

## Phase 5 — Model-per-Agent Routing

**Effort**: 1 sprint  
**Risk**: Low  
**Goal**: Each agent declares its preferred model tier. Fast/simple agents use local Ollama; complex/reasoning agents use cloud. This matches the Claude Dispatch pattern of per-subagent model overrides.

### What to Build

#### 5.1 — Model routing in `_run_agent`

```python
async def _route_agent_request(self, messages, agent_def: AgentDefinition):
    if agent_def.preferred_model == "system_2":
        # Use cloud (Groq) directly for this agent
        return await self.cognitive_router.route_to_system_2(
            messages, allowed_tools=agent_def.allowed_tools
        )
    else:
        # Default: try local first, fall back per Charter 1.3
        result = await self.cognitive_router.route_to_system_1(
            messages, allowed_tools=agent_def.allowed_tools
        )
        if self._is_system_1_error(result):
            result = await self.cognitive_router.route_to_system_2(
                messages, allowed_tools=agent_def.allowed_tools
            )
        return result
```

#### 5.2 — AGENT.md model tiers

- `system_1` — always try Ollama first (default, required by Charter 1.3)
- `system_2` — always use cloud (for critic, synthesizer, or reasoning-heavy tasks only)
- `auto` — supervisor decides at runtime based on query complexity score

#### 5.3 — Cost tracking in ledger

Log which model tier each agent invocation used to `system_logs` table so you can track cloud API usage over time:

```python
await self.ledger_memory.log_event(
    LogLevel.INFO,
    f"Agent {agent_def.name} used {model_tier}. Input tokens ~{token_estimate}."
)
```

### Files to Modify
- `src/core/agent_definition.py` — `preferred_model` field (already planned in Phase 1)
- `src/core/orchestrator.py` — `_route_agent_request`
- `src/memory/ledger_db.py` — optional: add `model_tier` column to `system_logs`

---

## Phase 6 — MCP Integration (Future / Optional)

**Effort**: 2–3 sprints  
**Risk**: Medium (new dependency surface)  
**Goal**: Surface external data sources and APIs as MCP tool namespaces that agents can subscribe to, without hardcoding them as Python skills.

### What This Enables

- A `market_data_agent` subscribes to an MCP stock/crypto feed — no Python skill required
- A `news_agent` subscribes to an MCP RSS/search server
- New integrations added by dropping an MCP server config, not by writing Python code

### High-Level Approach

1. Add `mcp_servers: List[str]` to `AgentDefinition`
2. On agent startup, initialize the declared MCP client connections
3. Translate MCP tool schemas into the same `allowed_tools` format the skill system already uses
4. The `SkillRegistry` gains a `register_mcp_tool()` method alongside the existing `register_dynamic()`

### Charter Constraint

All MCP servers must be local or self-hosted unless explicitly approved by Admin. Cloud MCP connections must go through the existing capability synthesis → MFA approval flow (Charter 1.2 + 2.2).

---

## Implementation Sequence (Recommended Sprint Order)

```
Sprint 7:  Phase 2A — Context Isolation on the existing research/coder workers
Sprint 8:  Phase 4 — Selective Critic
Sprint 9:  Phase 1 — Dynamic Agent Registry
Sprint 10: Phase 2B — Full Context Isolation via AgentDefinition handoffs
Sprint 11: Phase 3 — Parallel Worker Execution with System 1 semaphore
Sprint 12: Phase 5 — Model-per-Agent Routing
Sprint 13+: Phase 6 — MCP Integration
```

This sequence prioritizes immediate responsiveness improvements before extensibility work. Phase 3 still depends on isolation, but it no longer assumes unsafe local concurrency.

---

## Non-Negotiable Invariants (Must Survive Every Phase)

These must be verified by a test assertion after each phase is merged:

1. **Charter 1.3**: Every agent call attempts System 1 (local) before System 2 (cloud), unless `preferred_model: system_2` is explicitly declared
2. **MFA gate**: `request_core_update` and destructive operations always block with `RequiresMFAError`
3. **HITL on 3 critic failures**: `RequiresHITLError` must be raised after 3 rejections — this is `critic_node` logic that must not be simplified away
4. **No cloud data leakage**: Chat history, environment variables, and database content must never appear in a System 2 payload
5. **Graceful degradation**: If all agents fail, a human-readable error message reaches the user — never a raw exception or `None`
6. **No unsafe local parallelism**: Concurrent System 1 agent calls must pass through the router-level concurrency cap

---

## Key Files Reference

| File | Role | Modified In |
|------|------|-------------|
| `src/core/orchestrator.py` | Main state graph, worker dispatch | All phases |
| `src/core/llm_router.py` | System 1/2 routing, tool execution, local concurrency guard | Phase 3, 5 |
| `src/core/skill_manager.py` | Skill/tool registry | Phase 6 |
| `src/core/agent_definition.py` | AgentDefinition dataclass | Phase 1 (new) |
| `src/core/agent_registry.py` | Filesystem agent loader | Phase 1 (new) |
| `src/core/prompt_config.py` | Prompt builders | Phase 1, 4 |
| `src/core/state_model.py` | AgentState schema | Phase 2 (minor) |
| `src/core/workflow_graph.py` | LangGraph topology | Phase 3 |
| `src/agents/*/AGENT.md` | Agent definitions | Phase 1 (new) |
| `src/memory/ledger_db.py` | Audit log | Phase 5 |

---

*Last updated: 2026-04-22*  
*Reference: [Anthropic — Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) | [Claude Agent SDK Subagents](https://code.claude.com/docs/en/agent-sdk/subagents)*
