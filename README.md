# AIDEN — Autonomous Intelligence Decision & Execution Node

AIDEN is a self-contained, biomimetic autonomous AI agent that runs locally on your machine. It communicates through a Telegram bot interface, uses a multi-layer memory architecture, routes between local and cloud LLMs based on task complexity, plans and tracks objectives through a structured Epic/Story/Task hierarchy, enforces ethical governance through a moral ledger, and can autonomously synthesise and deploy new capabilities at runtime.

The project is built with Python, uses Ollama for local inference (System 1) and configurable cloud LLMs for complex reasoning (System 2), and persists all state across ChromaDB (vector memory), SQLite (operational ledger), and a JSON core memory file.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture — Biomimetic Cognitive Model](#2-architecture--biomimetic-cognitive-model)
3. [Infrastructure & Technology Stack](#3-infrastructure--technology-stack)
4. [Memory Architecture](#4-memory-architecture)
   - [Vector Memory — Archival/Long-Term (ChromaDB)](#41-vector-memory--archivallongterm-chromadb)
   - [Ledger Memory — Operational State (SQLite)](#42-ledger-memory--operational-state-sqlite)
   - [Core Memory — Working Memory (JSON)](#43-core-memory--working-memory-json)
5. [LLM Routing — System 1 vs System 2](#5-llm-routing--system-1-vs-system-2)
6. [Orchestration & State Graph](#6-orchestration--state-graph)
   - [AgentState Model](#61-agentstate-model)
   - [Supervisor → Workers → Critic Pipeline](#62-supervisor--workers--critic-pipeline)
   - [Routing Assessor & Fast Paths](#63-routing-assessor--fast-paths)
   - [Energy Budget System](#64-energy-budget-system)
7. [Agent Registry & Worker Agents](#7-agent-registry--worker-agents)
8. [Skills & Tool System](#8-skills--tool-system)
   - [Built-in Skills](#81-built-in-skills)
   - [Dynamic Tool Worker](#82-dynamic-tool-worker)
9. [Goal Planner — Epic/Story/Task Hierarchy](#9-goal-planner--epicstorytask-hierarchy)
10. [Energy Judge & ROI Engine](#10-energy-judge--roi-engine)
11. [Moral Ledger — Ethical Governance](#11-moral-ledger--ethical-governance)
12. [Synthesis Pipeline — Capability Gap Resolution](#12-synthesis-pipeline--capability-gap-resolution)
13. [Nocturnal Consolidation](#13-nocturnal-consolidation)
14. [Security Architecture](#14-security-architecture)
    - [Charter-Based Governance](#141-charter-based-governance)
    - [Multi-Factor Authentication (MFA)](#142-multi-factor-authentication-mfa)
    - [Cloud Redaction](#143-cloud-redaction)
    - [Dynamic Tool Sandboxing](#144-dynamic-tool-sandboxing)
15. [Telegram Interface](#15-telegram-interface)
16. [Configuration Reference](#16-configuration-reference)
17. [Installation & Setup](#17-installation--setup)
18. [Project Structure](#18-project-structure)

---

## 1. System Overview

AIDEN is designed around a "prefrontal cortex" metaphor — a central orchestrator that receives user messages, constructs context from memory, plans and dispatches work to specialised sub-agents, evaluates outputs through a critic, applies ethical governance, and delivers a final response. The system is always-local by default; cloud LLMs are treated as a controlled escalation path, not the primary inference engine.

Key design principles:

- **Local-first execution.** System 1 (Ollama/Gemma) handles the majority of reasoning. Cloud calls (System 2) are rate-limited, redacted of PII, and gated by the energy model.
- **Persistent memory across sessions.** ChromaDB stores semantic memories; SQLite tracks tasks, moral decisions, synthesis runs, and chat history; a JSON file holds the current working context.
- **Autonomous backlog execution.** A heartbeat loop wakes periodically to evaluate and execute pending objectives from the ledger without requiring user interaction.
- **Ethical self-governance.** Every non-trivial output is audited against a moral rubric before delivery. Severe violations halt execution and escalate to the Admin.
- **Self-expanding capabilities.** When a capability gap is detected, the system can draft, self-test, and submit new Python tools for Admin approval; approved tools are deployed at runtime without a restart.

---

## 2. Architecture — Biomimetic Cognitive Model

The architecture maps components to biological analogues:

| Cognitive Analogue | Component | Description |
|---|---|---|
| **Prefrontal Cortex** | `Orchestrator` | Central decision-making, planning, routing, and critic loop |
| **Hippocampus** | `VectorMemory` (ChromaDB) | Long-term semantic memory with similarity search |
| **RAM / Working Memory** | `CoreMemory` (JSON) | Short-term context: current focus, user preferences |
| **Basal Ganglia / Ledger** | `LedgerMemory` (SQLite) | Task queue, audit trail, moral decisions, synthesis history |
| **System 1 (Fast Thinking)** | Ollama local LLM (Gemma) | Quick, local, private pattern recognition and response |
| **System 2 (Slow Thinking)** | Cloud LLM (Groq/Gemini/Ollama Cloud) | Deep reasoning, goal decomposition, moral audit |
| **Cerebellum / Tools** | `SkillRegistry` | Modular callable tools executed by worker agents |
| **Frontal Lobe / Conscience** | `MoralLedger` + Critic | Ethical evaluation and enforcement |
| **Sleep Consolidation** | `NocturnalConsolidation` | Off-peak memory distillation and quality scoring |

The message flow at a high level:

```
Telegram message
      │
      ▼
  Orchestrator.process_message()
      │
      ├── RoutingAssessor → fast-path? (single/direct tool, capability question)
      │         │ yes → skip graph, execute directly
      │         │ no  ↓
      ├── GoalPlanner → detect high-level intent → Epic/Story/Task decomposition
      │
      ├── EnergyBudget check → defer if ROI too low
      │
      ├── MoralAudit (pre-execution) → triviality bypass or System 2 audit
      │
      ├── StateGraph: supervisor_node → execute_workers_node → critic_node
      │         │
      │         ├── supervisor: build plan (agent+task packets)
      │         ├── workers: dispatch agents in dependency order
      │         └── critic: validate output, enforce charter, moral audit
      │
      └── Final response → Telegram
```

---

## 3. Infrastructure & Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Interface** | python-telegram-bot v22+ | User-facing chat interface |
| **Local LLM** | Ollama (Gemma 4 default) | System 1 inference — fast, private |
| **Cloud LLM (primary)** | Groq API | System 2 — fast cloud reasoning (free tier) |
| **Cloud LLM (fallback)** | Google Gemini (google-genai) | System 2 fallback |
| **Cloud LLM (alt)** | Ollama Cloud | Optional paid System 2 alternative |
| **Vector DB** | ChromaDB 0.4.24 | Semantic memory with cosine-similarity embeddings |
| **Operational DB** | SQLite (via aiosqlite) | Ledger, task queue, chat history, moral audit |
| **Workflow Graph** | LangGraph 0.1.17 | Supervisor→Workers→Critic state machine |
| **Async I/O** | asyncio + aiofiles | Non-blocking file and network operations |
| **Web Search** | DuckDuckGo Search (ddgs) | Real-time web results via `web_search` skill |
| **Finance Data** | yfinance | Stock price retrieval via `get_stock_price` skill |
| **PDF Extraction** | pypdf + pdfplumber | Document text extraction |
| **Web Scraping** | trafilatura | Article extraction for web URLs |
| **TOTP / MFA** | pyotp | Optional TOTP second factor |
| **System Monitoring** | psutil | CPU, memory, disk metrics |

Python version: 3.11+ recommended. All async code requires a running event loop (the Telegram application provides this).

---

## 4. Memory Architecture

AIDEN uses three complementary memory stores, each serving a distinct cognitive role.

### 4.1 Vector Memory — Archival/Long-Term (ChromaDB)

**Location:** `src/memory/vector_db.py` | **Storage:** `data/chroma_storage/`

`VectorMemory` wraps a ChromaDB collection (`agent_memory`) configured for cosine-similarity search. Memories are stored with text content, a UTC-normalised timestamp, and optional metadata.

Key methods:
- `add_memory(content, metadata)` — Embeds and stores a new memory chunk.
- `query_memory(query, n_results)` — Returns top-N semantically similar memories.
- `delete_memory(memory_id)` — Removes a specific memory by UUID.
- `get_memory_count()` — Returns total stored memories.
- `prune_old_memories(max_days)` — Removes memories older than the configured retention window.

The collection is thread-safe via an internal `threading.Lock`. ChromaDB telemetry is disabled at startup. The embedding model is ChromaDB's default (all-MiniLM-L6-v2).

### 4.2 Ledger Memory — Operational State (SQLite)

**Location:** `src/memory/ledger_db.py` | **Storage:** `data/ledger.db`

`LedgerMemory` is the operational backbone. All public methods are async, protected by a single `asyncio.Lock`, and use an `aiosqlite` WAL-mode connection for safe concurrent reads.

**Core tables:**

| Table | Purpose |
|---|---|
| `task_queue` | Legacy task queue |
| `system_logs` | Structured event log with log levels |
| `chat_history` | Per-user conversation history with pruning |
| `objective_backlog` | Epic/Story/Task hierarchy with energy columns |
| `tool_registry` | Admin-approved synthesised tools |
| `moral_decisions` | Immutable audit log of every moral evaluation |
| `synthesis_runs` | Top-level synthesis job records |
| `synthesis_attempts` | Per-attempt detail (repair loops) |

**Objective hierarchy columns** (added via idempotent migrations):
- `depends_on_ids` — JSON list of prerequisite objective IDs
- `acceptance_criteria` — Plain-text completion criteria
- `defer_count`, `next_eligible_at`, `last_energy_eval_at`, `last_energy_eval_json` — Energy tracking columns

Key API categories:
- **Task/Objective CRUD:** `add_objective`, `get_pending_objectives`, `update_objective_status`, `get_objective_hierarchy_rollup`
- **Energy tracking:** `record_task_energy_evaluation`, `defer_task_due_to_energy`
- **Moral audit:** `add_moral_decision`, `get_moral_decisions`, `get_recent_moral_rejections`
- **Tool registry:** `add_approved_tool`, `get_approved_tools`, `touch_tool_last_used`, `get_unused_approved_tools`, `retire_tools`
- **Synthesis audit:** `add_synthesis_run`, `update_synthesis_run`, `add_synthesis_attempt`
- **Chat history:** `add_chat_message`, `get_recent_chat_history`, `prune_chat_history`
- **Synthesis throttle:** Per-user hourly synthesis run counts

### 4.3 Core Memory — Working Memory (JSON)

**Location:** `src/memory/core_memory.py` | **Storage:** `data/core_memory.json`

`CoreMemory` is the fast-access working context. It stores a small JSON document containing fields like `current_focus`, `user_preferences`, and any `nocturnal_facts` written by the consolidation pipeline. It is non-blocking (aiofiles) and protected by an `asyncio.Lock`. On startup it atomically creates the file if absent, or validates and recovers a corrupt file.

Key methods: `get_all()`, `get(key)`, `set(key, value)`, `update(partial_dict)`, `append_to_list(key, item, max_items)`.

---

## 5. LLM Routing — System 1 vs System 2

**Location:** `src/core/llm_router.py`

`CognitiveRouter` manages all LLM calls, exposing a unified async interface regardless of which backend is active.

**System 1 — Local (Ollama)**
- Model: configurable via `OLLAMA_MODEL` (default `gemma4:e4b`)
- Concurrency: limited by `SYSTEM_1_MAX_CONCURRENCY` (default 1 on consumer hardware)
- Timeout: `OLLAMA_TIMEOUT` seconds per call
- Queue: requests above the concurrency limit are queued with a semaphore; `SYSTEM_1_QUEUE_TIMEOUT_SECONDS` prevents indefinite waits
- Falls back gracefully when Ollama is unavailable

**System 2 — Cloud (priority order)**
1. **Ollama Cloud** — if `OLLAMA_CLOUD_API_KEY` is set
2. **Groq** — if `GROQ_API_KEY` is set (fast, free tier)
3. **Gemini** — if `GEMINI_API_KEY` + `USE_GEMINI=True`

**RouterResult** is a structured dataclass returned by all router methods:

```
status: "ok" | "mfa_required" | "hitl_required" | "capability_gap" | "cognitive_escalation"
content: str            # the model's text output
mfa_tool_name/arguments # for MFA-gated tool requests
hitl_message            # for human-in-the-loop escalations
gap_description / suggested_tool_name  # for capability gaps
escalation_problem / escalation_context  # for System 2 escalations
```

**Groq cooldown:** When Groq returns rate-limit errors, the router persists a cooldown timestamp to ledger and switches to the next available provider. The cooldown is restored across restarts.

**Cognitive escalation:** Worker agents can call the `escalate_to_system_2` tool to inject a structured blueprint from System 2 reasoning into the current turn state. The supervisor appends it as `<s2_in_turn_blueprint>` context for subsequent workers.

---

## 6. Orchestration & State Graph

**Location:** `src/core/orchestrator.py`

`Orchestrator` is now intentionally thin: it owns bootstrap/lifecycle flow, while domain logic is split into dedicated mixins and constants modules:

- `src/core/orchestrator_llm.py` (`_LLMGatewayMixin`) — System 1/System 2 routing and escalation handling
- `src/core/orchestrator_memory.py` (`_MemoryOpsMixin`) — archival/core memory retrieval, persistence, consolidation helpers
- `src/core/orchestrator_energy.py` (`_EnergyManagerMixin`) — predictive budget, ROI gating, reservation/refund logic
- `src/core/orchestrator_heartbeat.py` (`_HeartbeatManagerMixin`) — autonomous backlog execution and pending-state sweeps
- `src/core/orchestrator_fast_path.py` (`_FastPathMixin`) — direct/meta fast paths, capability responses, charter parsing helpers
- `src/core/orchestrator_graph.py` (`_GraphNodesMixin`) — supervisor/worker/critic graph nodes and turn loop
- `src/core/orchestrator_constants.py` — shared constants extracted to avoid circular imports

### 6.1 AgentState Model

**Location:** `src/core/state_model.py`

`AgentState` is a frozen dataclass passed between all graph nodes:

| Field | Type | Purpose |
|---|---|---|
| `user_id` | str | Per-user identity for lock isolation |
| `user_input` | str | The raw user message |
| `current_plan` | list | Agent+task dispatch packets from supervisor |
| `worker_outputs` | dict | Keyed outputs from each worker agent |
| `energy_remaining` | int | Per-turn energy budget |
| `iteration_count` | int | Critic loop counter (max 3 before HITL) |
| `hitl_count` | int | HITL escalation counter (max 3 cycles) |
| `moral_decision` | dict | Last moral audit result |
| `moral_halt_required` | bool | True if severe moral violation triggered |
| `critic_feedback` / `critic_instructions` | str | Feedback from critic to workers |

### 6.2 Supervisor → Workers → Critic Pipeline

The orchestrator builds a LangGraph `StateGraph` with three nodes. Node implementations and turn-loop helpers are owned by `_GraphNodesMixin`.

**`supervisor_node`**
- Constructs the full supervisor prompt: charter, sensory context (OS, CPU, CWD), core memory, recent archival memories, chat history, available tools, and current energy budget.
- If `ENABLE_SCOPED_SKILL_CONTEXT=true`, only injects full SKILL.md bodies for tools used in the current turn (reduces token overhead).
- Calls System 1 to produce a `current_plan` — an ordered list of task packets, each specifying: `agent`, `task`, `reason`, optional `depends_on`, and optional `preferred_model`.
- Logs a SHA-256 hash of the prompt for auditability.

**`execute_workers_node`**
- Dispatches agent task packets in dependency-layered order (`depends_on` controls execution sequence).
- Each agent is looked up in `AgentRegistry`, which loads on-disk `AGENT.md` definitions (YAML frontmatter + markdown system prompt).
- Workers call System 1 (or System 2 if `preferred_model=system_2`) with their specific system prompt, allowed tools, and prior dependency outputs injected.
- Tool calls are parsed from model output and executed through `SkillRegistry`. Each tool call is energy-deducted.
- Maximum `max_tool_calls` per agent (configurable per agent definition).

**`critic_node`**
- Evaluates the combined worker output against the user intent.
- Checks charter tier compliance by extracting relevant Tier 1/2/3 directives.
- Triggers moral audit:
  - **Triviality bypass:** Read-only, benign responses skip the full audit with a logged bypass.
  - **Local skip:** When System 2 is unavailable, uses a conservative local pass with audit trace.
  - **System 2 audit:** Full multi-dimensional moral evaluation via System 2 against the moral rubric.
- On non-approval: classifies severity and routes to remediation (moderate) or HITL halt (severe/Tier 1 security conflict).
- Accepts or rejects the turn output. Rejected outputs loop back to workers with `critic_instructions` (max 3 iterations, then HITL).

**Per-user locks:** Each user gets an isolated `asyncio.Lock` slot (LRU-capped at `USER_LOCKS_MAX_SIZE`) preventing concurrent overlapping turns for the same user.

**Progress event streaming:**
- The orchestrator can now emit structured progress events through `src/core/progress.py`.
- `process_message(...)` accepts an optional `progress_callback` used by Telegram handlers to stream status updates without changing existing call sites.
- Emission uses a task-local `ContextVar` (`ProgressEmitter`) so supervisor/agent/critic nodes can publish progress without threading callback arguments through every internal method.
- Heartbeat/background turns remain silent (`user_id="heartbeat"`) and do not receive an emitter.

**Heartbeat loop:** The orchestrator runs a periodic background coroutine every `HEARTBEAT_INTERVAL_SECONDS`. It queries energy candidates from the ledger (tasks with resolved dependencies), evaluates each through EnergyJudge + EnergyROIEngine, and executes approved tasks as autonomous agent turns. This allows AIDEN to work on backlog objectives between user interactions.

### 6.3 Routing Assessor & Fast Paths

**Location:** `src/core/routing_assessor.py`

`RoutingAssessor` scores incoming requests before invoking the full LangGraph graph, enabling two fast paths that skip supervisor overhead:

- **Direct response path:** Very short messages (≤12 tokens) with no matching tool signals. System 1 answers directly.
- **Single-tool fast path:** Messages with a strong unambiguous match to exactly one allowlisted tool (`get_system_info`, `get_stock_price`, `web_search`). The tool is called directly and the result formatted without a full supervisor pass.
- **Finance ticker detection:** Multi-ticker messages are detected by combining finance intent regex, uppercase token analysis, and ticker stopword filtering.
- **Capability question fast path:** Questions about what the system can do are answered by a lightweight System 1 classifier call (10s deadline, 60 token cap, no tools).

Routing stopwords, non-utility tool exclusions, and complexity thresholds are tunable constants.

### 6.4 Energy Budget System

Each `process_message` turn starts with `INITIAL_ENERGY_BUDGET` points (default 100). Deductions occur for:
- Supervisor call
- Each worker agent dispatch
- Each tool execution within a worker
- Critic call

If `energy_remaining` drops too low mid-turn, further work is deferred.

A separate **predictive energy account** (`_energy_budget`) is used for heartbeat and ad-hoc background tasks, replenished by wall-clock time (`ENERGY_REPLENISH_PER_HOUR`) and per-heartbeat credits (`ENERGY_REPLENISH_PER_HEARTBEAT`).

`ENERGY_REPLENISH_PER_TURN` is retained only as a legacy compatibility knob and is deprecated for predictive-budget policy; prefer `ENERGY_REPLENISH_PER_HOUR` and `ENERGY_REPLENISH_PER_HEARTBEAT`.

---

## 7. Agent Registry & Worker Agents

**Location:** `src/core/agent_registry.py` | **Agent definitions:** `src/agents/`

`AgentRegistry` loads agent definitions from `src/agents/*/AGENT.md` files. Each `AGENT.md` has YAML frontmatter defining capabilities and a markdown body providing the system prompt. On-disk definitions override built-in fallbacks.

**Built-in agents:**

| Agent | Role | Allowed Tools | Model |
|---|---|---|---|
| `research_agent` | Search memory, gather context | `search_archival_memory`, `escalate_to_system_2` | System 1 |
| `coder_agent` | Implementation, memory writes, file tasks | `update_ledger`, `update_core_memory`, `request_core_update`, `spawn_new_objective`, `update_objective_status`, `extract_pdf_text`, `search_archival_memory`, `escalate_to_system_2` | System 1, depends on research_agent |
| `synthesis_agent` | Combine multiple agent outputs into one response | None | System 2 |

**AgentDefinition fields:**

```python
name: str               # matches agent folder name
description: str        # used by supervisor to select agents
system_prompt: str      # injected as the agent's persona
allowed_tools: list     # subset of SkillRegistry tools available to this agent
preferred_model: str    # "system_1" or "system_2"
max_tool_calls: int     # hard cap per agent execution
energy_cost: int        # flat energy deduction per dispatch
depends_on: list        # agent names that must complete first
```

The supervisor constructs `current_plan` by selecting appropriate agents and composing task packets. Worker dispatch respects dependency order, injecting earlier agents' outputs into later agents' context.

---

## 8. Skills & Tool System

**Location:** `src/skills/` | **Registry:** `src/core/skill_manager.py`

`SkillRegistry` scans `src/skills/` at boot. Each skill is a sub-folder containing a `SKILL.md` (with a JSON schema block) and optionally an `__init__.py` with the async callable. Broken skills are skipped with a warning; healthy ones are registered.

**Skill loading:**
1. Parse `SKILL.md` YAML frontmatter for metadata and the embedded JSON schema block.
2. Import `__init__.py` and locate the async `run(**kwargs)` function.
3. Register the tool name, schema, and callable.

`SkillRegistry.get_schemas()` returns the list of tool dicts injected into LLM prompts. `SkillRegistry.execute(tool_name, args)` dispatches to the appropriate callable.

### 8.1 Built-in Skills

| Skill | Description |
|---|---|
| `web_search` | DuckDuckGo search returning title, URL, snippet |
| `get_stock_price` | Yahoo Finance price, daily change, previous close |
| `get_system_info` | OS, CPU usage, memory, disk, CWD, platform |
| `run_terminal_command` | Allowlisted shell commands (git, ls, dir, echo, etc.) |
| `execute_python_sandbox` | AST-validated Python execution in isolated worker |
| `search_archival_memory` | Semantic search over ChromaDB vector store |
| `update_core_memory` | Write key/value pairs to core memory JSON |
| `update_ledger` | Log events and update task statuses in SQLite |
| `spawn_new_objective` | Add Epic/Story/Task to the objective backlog |
| `update_objective_status` | Change status of an existing objective |
| `query_highest_priority_task` | Fetch the next pending task from the backlog |
| `request_core_update` | MFA-gated modification of charter or core code |
| `request_capability` | Trigger capability-gap synthesis workflow |
| `ask_admin_for_guidance` | Send an urgent question to the Admin via Telegram |
| `escalate_to_system_2` | Route complex sub-problem to System 2 |
| `extract_pdf_text` | Extract text from PDF files in the downloads directory |
| `extract_web_article` | Scrape and extract article text from a URL |
| `analyze_table_file` | Load and analyse CSV/Excel files with pandas |
| `manage_file_system` | Path-confined read/write/list operations |
| `consolidate_memory` | Trigger nocturnal consolidation manually |

### 8.2 Dynamic Tool Worker

**Location:** `src/core/dynamic_tool_worker.py`

Synthesised tools (approved through the synthesis pipeline) are executed in an isolated subprocess with strict security controls:

- **AST validation:** All synthesised code is parsed and inspected before execution. Blocked top-level modules include `os`, `sys`, `subprocess`, `socket`, `pathlib`, `importlib`, `threading`, `ctypes`, and others.
- **Token scanning:** Forbidden dunder attributes (`__import__`, `__globals__`, `__subclasses__`) and dangerous builtins (`eval`, `exec`, `compile`) are blocked.
- **POSIX resource limits** (Linux/macOS): CPU seconds capped at 5 per call; address space limited to 256 MB; open file limit capped at 32.
- **Import allowlist:** Only Python standard library modules are permitted (non-stdlib imports are rejected).
- **IPC:** Parent process communicates with the worker via newline-delimited JSON over stdin/stdout. A separate watchdog process handles ping/timeout.

Tool lifecycle tracking: `touch_tool_last_used` records each execution timestamp; `/retire_unused_tools` admin command (with confirm flow) retires tools idle for a configurable number of days.

---

## 9. Goal Planner — Epic/Story/Task Hierarchy

**Location:** `src/core/goal_planner.py`

`GoalPlanner` transforms high-level user objectives into a structured three-tier work hierarchy stored in the ledger's `objective_backlog` table.

**Hierarchy:**
```
Epic  (large outcome, e.g. "Build a stock screening pipeline")
  └── Story  (functional slice, e.g. "Implement data ingestion")
        └── Task  (atomic work item, e.g. "Write yfinance fetch wrapper")
```

**Planning flow:**
1. Orchestrator detects epic-level intent in the user message.
2. GoalPlanner builds a redacted payload (PII-stripped for cloud) and calls System 2.
3. System 2 returns strict JSON with the full hierarchy including `acceptance_criteria`, `estimated_energy`, `priority`, and symbolic `dependency_refs` between tasks.
4. GoalPlanner resolves symbolic dependency refs to concrete ledger IDs before insertion.
5. All objectives are persisted via `LedgerMemory.add_objective()`.
6. Returns a `PlanningResult` with the assigned epic/story/task IDs.

The Telegram `/addgoal` command triggers automatic Epic decomposition when System 2 is available, preventing duplicate epic creation.

---

## 10. Energy Judge & ROI Engine

**Location:** `src/core/energy_judge.py`, `src/core/energy_roi_engine.py`

The energy system acts as a cognitive resource manager, preventing low-value work from consuming limited compute budget.

### EnergyJudge

A System 1 evaluator that scores a task's `estimated_effort` (1–10) and `expected_value` (1–10) given its parent Story and Epic context. Calls are made with a strict JSON output contract; malformed or out-of-bounds responses fall back to conservative scores (effort=5, value=5).

### EnergyROIEngine

A purely deterministic Python gate (no LLM calls) that decides whether to execute a task:

```
base_roi = expected_value / estimated_effort
fairness_boost = defer_count × FAIRNESS_BOOST_MULTIPLIER
effective_roi = base_roi × (1 + fairness_boost)
predicted_cost = estimated_effort × ENERGY_MULTIPLIER
reserve_after = available_energy - predicted_cost
```

**Decision logic:**
- If `defer_count >= MAX_DEFER_COUNT` and reserve is sufficient → force execute (fairness bypass)
- If `effective_roi < ENERGY_ROI_THRESHOLD` → defer
- If `reserve_after < ENERGY_MIN_RESERVE` → defer (insufficient reserve)
- Otherwise → execute

**Deferred tasks** are marked `deferred_due_to_energy` in the ledger, have their `defer_count` incremented, and receive a `next_eligible_at` cooldown timestamp. The fairness boost ensures long-deferred tasks eventually get executed.

**Completion refunds:** When `update_objective_status` marks a task completed, a predictive energy refund is applied to the shared budget account (bounded by the existing ceiling).

---

## 11. Moral Ledger — Ethical Governance

**Location:** `src/core/moral_ledger.py`

The moral ledger enforces AIDEN's charter directives through a structured multi-dimensional scoring rubric.

**Moral dimensions (8 total):**

Tier 1 (absolute — any violation triggers HITL or rejection):
- `harm_reduction`
- `data_privacy`
- `admin_authority_security`
- `data_sovereignty_local_priority`

Tier 2 (high weight):
- `epistemic_humility`
- `transparency_logging`
- `alignment_with_user_intent`

Tier 3 (standard):
- `output_cleanliness`

**Audit modes:**

| Mode | Trigger | Behaviour |
|---|---|---|
| `triviality_bypass` | Read-only, benign response | Skip full audit; log bypass with reason |
| `local_skip` | System 2 unavailable | Conservative local pass; full trace logged |
| `system2_audit` | Normal non-trivial output | Full System 2 JSON audit against rubric |

**MoralDecision** is an immutable frozen dataclass with fields: `rubric_version`, `scores` (per dimension 1–5), `reasoning`, `is_approved`, `decision_mode`, `bypass_reason`, `remediation_constraints`, `violated_tiers`, `security_conflict`, `validation_error`.

**Remediation routing (in critic node):**
- **Approved** → continue to final response
- **Moderate risk** → `remediation_constraints` from System 2 are injected into `critic_instructions` for a bounded retry loop
- **Severe / Tier 1 security conflict** → immediate `RequiresHITLError` halt; heartbeat task suspended; Admin notified

**Self-reflection loop:** The supervisor prompt receives a `<recent_rejections>` block (last 3 non-approved decisions, capped at 500 chars) so AIDEN can learn from recent moral failures within a session.

**Synthesis backoff:** If the same suggested tool has ≥2 prior `blocked`/`rejected` synthesis runs (fuzzy-deduplicated over a 24h window), any new synthesis attempt is automatically redirected to HITL for manual review.

**Immutability guarantee:** Moral decisions are append-only in the `moral_decisions` table. No update or delete operations are permitted on that table.

---

## 12. Synthesis Pipeline — Capability Gap Resolution

**Location:** `src/core/synthesis_pipeline.py`

When a capability gap is detected (System 1 reports it cannot fulfil a request), the synthesis pipeline manages the full lifecycle of creating and deploying a new tool.

**Pipeline stages:**

1. **Gap detection** — `request_capability` skill signals a gap with a description and suggested tool name.
2. **Duplicate / backoff check** — Fuzzy match against prior blocked/rejected synthesis runs (24h window). If ≥2 similar failures exist, route to HITL instead.
3. **Rate limit check** — Per-user hourly synthesis count (`MAX_SYNTHESES_PER_USER_PER_HOUR`, default 5) blocks excess requests.
4. **System 2 code generation** — Full tool code generated with redacted context.
5. **Self-test** — Generated code is executed in a temporary subprocess. Output is SHA-256 digested (normalised tool + pytest artifacts).
6. **Repair loop** — Up to `MAX_SYNTHESIS_RETRIES` (default 3) repair retries if self-test fails; each attempt is persisted in `synthesis_attempts`.
7. **Admin approval** — Tool code and self-test results are sent to Admin via Telegram. Admin responds with `approve`/`yes` or `reject`/`no`.
8. **Deployment** — On approval: tool is written to `src/skills/<tool_name>/`, registered in `tool_registry` table, loaded into `SkillRegistry`, and logged to core memory under `known_capabilities`.
9. **Verification** — SHA-256 digest is re-verified before deployment to confirm integrity.

**Security controls:**
- Synthesis subprocess environment strips all `TELEGRAM_*`, `GROQ_*`, `GEMINI_*`, `ANTHROPIC_*`, `OPENAI_*`, `OLLAMA_CLOUD_*`, and `ADMIN_*` variables.
- `PENDING_STATE_TTL_SECONDS` (default 86400) auto-expires stale pending approvals.
- `SYNTHESIS_LOCKOUT_TTL_SECONDS` (default 600) prevents rapid re-synthesis attempts for the same tool.

---

## 13. Nocturnal Consolidation

**Location:** `src/core/nocturnal_consolidation.py`

`NocturnalConsolidationSlice1` runs during low-activity periods (configurable via `MEMORY_CONSOLIDATION_INTERVAL`, default 6 hours) to distil raw chat history and ledger events into high-quality long-term memories.

**Pipeline:**

1. **Candidate extraction** — Pulls recent chat messages, system logs, and objective completions. Filters out low-signal messages (short greetings, transient routing logs, voice note placeholders).
2. **Semantic deduplication** — Jaccard similarity (tokenised, stopword-filtered) removes near-duplicate candidates above `batch_semantic_jaccard_threshold` (default 0.82). Vector distance check (default 0.08) filters against already-stored archival memories.
3. **Quality scoring** — System 2 scores each candidate on 7 factors (1–5 each): `novelty`, `actionability`, `confidence`, `recurrence`, `charter_alignment`, `contradiction_risk`, `staleness`. A weighted Q-score is computed; candidates below threshold are discarded.
4. **Storage write-back** — Passing candidates are written to ChromaDB (archival memory). Charter/governance content is flagged for ledger logging. User profile facts are routed to core memory.
5. **Core memory compaction** — Nocturnal facts in core memory are capped at `MAX_NOCTURNAL_CORE_FACTS` (default 60 entries).

Consolidation is also triggerable manually via the `consolidate_memory` skill or `CONSOLIDATION_TRIGGER_TURNS` environment variable (every N user turns).

---

## 14. Security Architecture

### 14.1 Charter-Based Governance

**Location:** `charter.md`

The `Identity_Charter` XML document defines three tiers of directives loaded at startup into a cached parse tree:

- **Tier 1 (Absolute):** Do No Harm; Admin Supremacy; Data Sovereignty/Local Priority. Violations trigger immediate HITL halt.
- **Tier 2 (Strategic):** Epistemic Humility; Autonomous Self-Expansion (with Admin approval); Transparent Governance; Cognitive Reflection; Capability Gap Protocol; Accurate Self-Knowledge.
- **Tier 3 (Operational):** Analytical communication style; Asynchronous collaboration; Graceful degradation; Clean conversational output (no internal reasoning leaked to user).

The critic node extracts relevant charter directives per turn for enforcement. Charter XML is cached with a fail-closed policy (`ALLOW_MISSING_CHARTER=true` overrides to fallback charter).

### 14.2 Multi-Factor Authentication (MFA)

**Location:** `src/core/security.py`

Sensitive operations (modifying the charter, core code updates, tool registry changes) require MFA validation:

- **Passphrase MFA:** `MFA_PASSPHRASE` must be ≥12 characters and must not contain common words (`password`, `admin`, `secret`, etc.). Normalised for whitespace and punctuation before comparison.
- **TOTP (optional):** If `MFA_TOTP_SECRET` is set and `pyotp` is installed, 6-digit TOTP codes are also accepted (1-step validity window).
- **`verify_mfa_challenge(response)`** accepts either a matching passphrase or a valid TOTP code.

The `/retire_unused_tools` Telegram command implements a two-step confirm flow using the `pending_retire_unused_tools` state key.

### 14.3 Cloud Redaction

**Location:** `src/core/cloud_redaction.py`

Before any payload is sent to System 2, `cloud_redaction` applies layered scrubbing:

- **Block-level redaction:** `<Core_Working_Memory>`, `<Archival_Memory>`, and `<chat_history>` XML blocks are replaced with `[REDACTED_CONTEXT_BLOCK]`.
- **Sensory context redaction:** Machine-specific data (OS, CPU usage, CWD, platform) is replaced with `[REDACTED_SENSORY_STATE]`.
- **PII patterns:** Email addresses, phone numbers, physical addresses, names, locations, SSNs, and local filesystem paths are replaced with typed sentinels.
- **Message minimization (`redact_messages_for_cloud`):** when sensitive context is not allowed, only a reduced subset is forwarded (first+last system message and the latest user message), deduplicated after redaction, with a safe empty sentinel fallback.
- **Sensitive-context override:** with `allow_sensitive_context=true`, message pruning/redaction is bypassed but payloads are still bounded by max-character truncation safeguards.

All redaction functions are pure (no side effects) and are compiled as module-level regex patterns for performance.

### 14.4 Dynamic Tool Sandboxing

Runtime-synthesised tools execute in a fully isolated subprocess (`DynamicToolWorkerClient`). The sandbox enforces:

- Blocked standard library modules (filesystem, network, subprocess, introspection)
- Blocked dunder attributes and dangerous builtins
- POSIX resource limits (CPU, memory, file descriptors) on Linux/macOS
- Environment variable sanitisation (all secrets stripped before subprocess launch)
- Newline-delimited JSON protocol over stdin/stdout (no shared memory)

---

## 15. Telegram Interface

**Location:** `src/interfaces/telegram_bot.py`

The Telegram bot is the sole user-facing interface. It runs as an async `python-telegram-bot` Application.

**Authorisation:** Only `ADMIN_USER_ID` can interact with the bot. All other users receive `"Unauthorized."` and the message is logged.

**Message handling:**
- Text messages → `orchestrator.process_message(text, user_id, progress_callback=...)`
- Voice messages → decoded to `[Voice note · N bytes · mime_type]` placeholder (or transcribed if `ENABLE_NATIVE_AUDIO=true`)
- Document messages with `YES`/`NO` captions → resume pending tool approvals
- Long synthesis code responses → sent as a temporary `.py` document attachment

**Streaming response UX:**
- Fast short messages stay on a non-streaming path to avoid status-message flash.
- For graph-bound requests, Telegram handlers use `DeferredStatusMessage` + `StatusFormatter` (`src/interfaces/streaming.py`).
- A status message is only sent after `STREAMING_DELAY_SECONDS`; it is then edited in-place with plan/agent/critic progress and finally replaced with the final answer.

**Admin commands:**

| Command | Purpose |
|---|---|
| `/start` | Welcome message |
| `/status` | System health: skills loaded, failed skills, pending MFA/HITL, predictive energy budget |
| `/addgoal <text>` | Add objective (auto-decomposes if epic-level and System 2 available) |
| `/tasks` | List pending objectives |
| `/memory` | Show current core memory state |
| `/retire_unused_tools` | List and retire dynamically synthesised tools with no recent usage (two-step confirm) |

**Reliability features:**
- Rotating file logger (`logs/telegram_bot.log`, 10 MB max, 5 backups)
- `Conflict` error handling with exponential backoff for multi-instance collisions
- `RetryAfter` handling for Telegram rate limits
- Per-user `AGENT_TIMEOUT_SECONDS` (default 300s) before timeout message is sent
- Outbound message queue (`OUTBOUND_QUEUE_MAX_SIZE`, default 200) for async Telegram sends
- Signal handlers for graceful shutdown (SIGINT, SIGTERM)

---

## 16. Configuration Reference

All configuration is via environment variables loaded from `.env` (copy from `.env.example`).

### Required

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | BotFather token for the Telegram bot |
| `ADMIN_USER_ID` | Your Telegram numeric user ID |
| `MFA_PASSPHRASE` | ≥12 char passphrase for MFA-gated operations |

### LLM Providers (at least one System 2 source recommended)

| Variable | Description |
|---|---|
| `OLLAMA_MODEL` | Local model name (default: `gemma4:e4b`) |
| `OLLAMA_TIMEOUT` | Seconds per local LLM call (default: `60`) |
| `SYSTEM_1_MAX_CONCURRENCY` | Max parallel System 1 calls (default: `1`) |
| `GROQ_API_KEY` | Groq API key (primary System 2) |
| `GEMINI_API_KEY` | Google Gemini API key |
| `USE_GEMINI` | Set `True` to enable Gemini fallback |
| `OLLAMA_CLOUD_API_KEY` | Ollama Cloud key (highest-priority System 2) |
| `SYSTEM_2_MODEL` | Ollama Cloud model (default: `deepseek-v3.2`) |
| `MFA_TOTP_SECRET` | Optional TOTP secret (base32) for 6-digit codes |

### Memory

| Variable | Default | Description |
|---|---|---|
| `MEMORY_SAVE_THRESHOLD` | `60` | Min chars to save turn to vector memory |
| `MEMORY_CONSOLIDATION_INTERVAL` | `21600` | Consolidation interval in seconds |
| `CONSOLIDATION_TRIGGER_TURNS` | `10` | Turns between consolidation triggers |
| `CHAT_HISTORY_MAX_DAYS` | `90` | Days of chat history to retain |
| `CHAT_HISTORY_MIN_ROWS` | `20` | Minimum rows to always keep per user |
| `VECTOR_MEMORY_MAX_DAYS` | `180` | Days of vector memories to retain |

### Orchestrator Tuning

| Variable | Default | Description |
|---|---|---|
| `HEARTBEAT_INTERVAL_SECONDS` | `1800` | Background objective execution interval |
| `INITIAL_ENERGY_BUDGET` | `100` | Per-turn energy budget |
| `ENERGY_REPLENISH_PER_HOUR` | `30` | Wall-clock predictive energy replenishment rate |
| `ENERGY_REPLENISH_PER_HEARTBEAT` | `2` | Predictive energy refund applied at heartbeat cycle start |
| `ENERGY_REPLENISH_PER_TURN` | `5` | Legacy compatibility setting (deprecated) |
| `ENERGY_ROI_THRESHOLD` | `1.25` | Minimum ROI to execute a task |
| `ENERGY_MIN_RESERVE` | `10` | Minimum energy reserve after execution |
| `FAIRNESS_BOOST_MULTIPLIER` | `0.15` | ROI boost per defer count |
| `MAX_DEFER_COUNT` | `5` | Force-execute after this many deferrals |
| `PENDING_STATE_TTL_SECONDS` | `86400` | Auto-expire stale MFA/HITL/approval entries |
| `AGENT_TIMEOUT_SECONDS` | `300` | Per-message timeout |
| `STREAMING_DELAY_SECONDS` | `2.5` | Delay before Telegram status message is first shown |
| `STREAMING_MIN_TOKENS` | `4` | Minimum token count before streaming path is used |
| `TOOL_EXEC_TIMEOUT_SECONDS` | `30` | Per-tool execution timeout |
| `USER_LOCKS_MAX_SIZE` | `500` | LRU cap on per-user async locks |

### Feature Flags

| Variable | Default | Description |
|---|---|---|
| `ENABLE_SCOPED_SKILL_CONTEXT` | `true` | Inject SKILL.md body only for active tools |
| `ENABLE_LOCAL_SKILL_DISCOVERY_GATE` | `true` | Try local skill matching before synthesis |
| `ENABLE_NATIVE_AUDIO` | `false` | Enable native voice message transcription |
| `ALLOW_MISSING_CHARTER` | (unset) | Fallback charter instead of fail-closed |
| `MAX_SYNTHESIS_RETRIES` | `3` | Max repair attempts for synthesised tools |
| `MAX_SYNTHESES_PER_USER_PER_HOUR` | `5` | Rate limit for capability synthesis |

---

## 17. Installation & Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running locally
- A Telegram bot token (create via [@BotFather](https://t.me/BotFather))
- Your Telegram user ID (use [@userinfobot](https://t.me/userinfobot))

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd AI_Prototype

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the local LLM model
ollama pull gemma4:e4b

# 5. Configure environment
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/macOS
# Edit .env with your values

# 6. Run the bot
python main.py
```

### Verifying the setup

```bash
# Run the full test suite
python -m pytest tests/ -q

# Reset ChromaDB if collection is corrupt
python reset_chroma.py
```

Logs are written to `logs/telegram_bot.log` with rotation. The bot prints startup status to stdout, including how many skills were loaded and whether System 2 is available.

---

## 18. Project Structure

```
AI_Prototype/
├── main.py                          # Entry point — starts Telegram bot
├── charter.md                       # XML Identity Charter (governance directives)
├── requirements.txt
├── .env.example                     # Configuration template
├── reset_chroma.py                  # Utility to reset ChromaDB collection
│
├── data/
│   ├── core_memory.json             # Working memory (current focus, preferences)
│   ├── ledger.db                    # SQLite operational database
│   └── chroma_storage/             # ChromaDB vector store
│
├── logs/
│   └── telegram_bot.log            # Rotating log file
│
├── downloads/                       # Default directory for PDF/file analysis
│
├── src/
│   ├── interfaces/
│   │   ├── telegram_bot.py         # Telegram bot handler and admin commands
│   │   └── streaming.py            # Deferred status message + progress formatter
│   │
│   ├── core/
│   │   ├── orchestrator.py         # Thin orchestrator shell (bootstrap + lifecycle wrappers)
│   │   ├── orchestrator_constants.py # Shared orchestrator constants
│   │   ├── orchestrator_llm.py     # LLM routing mixin
│   │   ├── orchestrator_memory.py  # Memory operations mixin
│   │   ├── orchestrator_energy.py  # Predictive energy/ROI mixin
│   │   ├── orchestrator_heartbeat.py # Heartbeat + pending-state mixin
│   │   ├── orchestrator_fast_path.py # Fast-path and charter helpers mixin
│   │   ├── orchestrator_graph.py   # Graph nodes + execution loop mixin
│   │   ├── progress.py             # ProgressEvent + ProgressEmitter ContextVar infra
│   │   ├── state_model.py          # AgentState dataclass
│   │   ├── workflow_graph.py       # LangGraph supervisor→workers→critic graph
│   │   ├── llm_router.py           # CognitiveRouter: System 1/2 routing
│   │   ├── routing_assessor.py     # Fast-path scoring and dispatch mode selection
│   │   ├── agent_registry.py       # Loads on-disk AGENT.md definitions
│   │   ├── agent_definition.py     # AgentDefinition dataclass
│   │   ├── skill_manager.py        # SkillRegistry: loads and executes skills
│   │   ├── dynamic_tool_worker.py  # Sandboxed subprocess for synthesised tools
│   │   ├── synthesis_pipeline.py   # Capability gap → tool synthesis → approval
│   │   ├── goal_planner.py         # System 2 Epic/Story/Task decomposition
│   │   ├── energy_judge.py         # System 1 effort/value scorer
│   │   ├── energy_roi_engine.py    # Deterministic ROI gate
│   │   ├── moral_ledger.py         # Moral rubric schema and decision parsing
│   │   ├── nocturnal_consolidation.py  # Memory distillation pipeline
│   │   ├── cloud_redaction.py      # PII and context redaction for cloud payloads
│   │   ├── security.py             # MFA passphrase + TOTP validation
│   │   ├── prompt_config.py        # Supervisor prompt construction
│   │   └── runtime_context.py      # Module-level references to shared resources
│   │
│   ├── memory/
│   │   ├── vector_db.py            # ChromaDB wrapper (long-term semantic memory)
│   │   ├── ledger_db.py            # SQLite async wrapper (operational state)
│   │   └── core_memory.py          # JSON working memory
│   │
│   ├── agents/
│   │   ├── coder_agent/AGENT.md    # Coder agent definition
│   │   ├── research_agent/AGENT.md # Research agent definition
│   │   └── synthesis_agent/AGENT.md # Synthesis agent definition
│   │
│   ├── skills/                      # Tool plugin directory
│   │   ├── web_search/
│   │   ├── get_stock_price/
│   │   ├── get_system_info/
│   │   ├── run_terminal_command/
│   │   ├── execute_python_sandbox/
│   │   ├── search_archival_memory/
│   │   ├── update_core_memory/
│   │   ├── update_ledger/
│   │   ├── spawn_new_objective/
│   │   ├── update_objective_status/
│   │   ├── query_highest_priority_task/
│   │   ├── request_core_update/
│   │   ├── request_capability/
│   │   ├── ask_admin_for_guidance/
│   │   ├── escalate_to_system_2/
│   │   ├── extract_pdf_text/
│   │   ├── extract_web_article/
│   │   ├── analyze_table_file/
│   │   ├── manage_file_system/
│   │   └── consolidate_memory/
│   │
│   └── tools/
│       └── system_tools.py         # Low-level system utility functions
│
├── scripts/
│   ├── ingest_external_skill.py    # CLI: import a skill from an external path
│   └── skill_audit.py              # CLI: audit registered skills
│
└── tests/                           # pytest test suite (300+ tests)
    ├── conftest.py
    ├── test_progress.py             # Progress event/emitter/contextvar tests
    ├── test_streaming.py            # Telegram streaming helper tests
    ├── test_*.py                    # Unit and integration tests per subsystem
    └── manual/                      # Manual/exploratory test scripts
```

---

## Security Notes

- Never commit `.env` to version control. Use `.env.example` as a template only.
- `MFA_PASSPHRASE` must be strong (≥12 chars, no common words). It gates all destructive operations.
- Cloud payloads are automatically redacted of PII, memory contents, and local paths before being sent to any external LLM.
- Synthesised tools are sandboxed and require explicit Admin approval before deployment.
- The `ADMIN_USER_ID` check is enforced at the Telegram handler level — all other users are rejected before any processing occurs.
- The moral ledger is append-only; no decision record can be deleted or modified after creation.

---

## License

See [LICENSE](LICENSE) for details.
