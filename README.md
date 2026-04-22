# Autonomous Biomimetic AI Agent

## Project Overview

This project implements an autonomous, locally-hosted AI agent featuring a sophisticated biomimetic cognitive architecture. The system mimics biological processes through its state-graph orchestration, continuous memory subsystems, proactive heartbeats, and dynamic capability synthesis.

### Core AI System Functions

The agent operates through several distinct cognitive systems and memory layers:

#### Dual-System Cognitive Engine
- **System 1 (Local LLM)**: Fast, private, and always available. Powered by local models (e.g., `gemma4:e4b` via Ollama). Used for routine pattern matching, tool calling, and initial responses. System 1 actively detects its own capability gaps.
- **System 2 (Cloud LLM)**: Deep reasoning, complex problem solving, and tool synthesis. Prioritises Groq (`llama-3.3-70b-versatile` - fast, free inference) and falls back to Gemini API via Google GenAI. Called when System 1 fails or encounters an explicit capability gap.

#### Orchestrator & State Graph
The core engine operates on a multi-node **State Graph Architecture** with a built-in energy budget, mimicking human cognitive fatigue and multi-step reasoning.
- **Supervisor Node**: Initial planner. It breaks down the user request and delegates to specialized workers.
- **Worker Agents (Research & Coder)**: Execute tasks (e.g., querying vector memory, calling system tools, updating the ledger).
- **Critic Node**: A strict evaluation stage checking all outputs against the AI's core charter. It can force re-iterations (up to 3 times) before falling back to Admin guidance.
- **Energy Budget**: Each operation (supervisor, worker, critic) costs "energy". If the 100-point budget depletes, the system pauses and demands human-in-the-loop prioritization.
- **Proactive Heartbeat**: Every 30 minutes, the agent wakes autonomously, checks its Objective Backlog, picks a prioritized task, executes it, and sends the Admin an update.

#### Dynamic Capability Synthesis
When System 1 encounters an explicit gap in its toolset, it raises a signal to the Orchestrator. The Orchestrator routes the request to System 2, which dynamically writes Python code for a new tool to bridge the gap.
The system uses a **Human-in-the-Loop (HITL)** prompt to ask the Admin to approve the synthesized code before it is actively deployed and registered in the database.

#### Memory Subsystems
- **Core Memory (Short-Term)**: A lightweight JSON memory store tracking immediate environment variables, active focus, user preferences, and known capabilities.
- **Hippocampus / Vector Database (Long-Term)**: An asynchronous chroma-db driven embedding storage capturing semantic concepts, past conversations, and experiences.
- **Ledger Memory (Episodic & Operational)**: A SQLite-backed database maintaining objective backlogs, tool registries, and conversational histories (chat turns).

### Security & Interactions

- **Telegram Interface**: The primary UI interface. Restricted via strict `ADMIN_USER_ID` checks.
- **MFA Intercept**: Sensitive actions (e.g., core system updates) are intercepted by the Orchestrator and challenged with multi-factor authentication phrases before proceeding.
- **Human-In-The-Loop (HITL)**: Used across the system. Halts current processes and saves active states when the Critic fails repeatedly, energy is depleted, or tools need approval, awaiting Admin intervention.

---

## Project Structure

```
AI_Prototype/
├── src/
│   ├── core/                # Orchestrator, LLM Router, Security, Skill Manager
│   ├── interfaces/          # Telegram Bot & other communication nodes
│   ├── memory/              # Core JSON, Vector DB, SQLite Ledger
│   ├── skills/              # Statically implemented tools
│   └── tools/               # Dynamic / Misc tool definitions
├── logs/                    # System logs
├── data/                    # SQLite DB, Chroma storage, JSON state
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── charter.md               # Core directives for the AI Critic
└── .env.example             # Configuration template
```

## Prerequisites

- Python 3.10+
- `pip`
- [Ollama](https://ollama.com/) running locally (ensure `gemma4:e4b` or fallback model is pulled)
- Telegram Bot Token & Admin User ID
- API Keys for Groq (recommended) or Gemini.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in the necessary credentials:
- `TELEGRAM_BOT_TOKEN`: From `@BotFather`
- `ADMIN_USER_ID`: From `@userinfobot`
- `GROQ_API_KEY`: Groq API Key (Preferred System 2)
- `GEMINI_API_KEY`: Gemini API Key (Fallback System 2)
- `USE_GEMINI`: `True` if you want to use Gemini.
- `AIDEN_DOWNLOADS_DIR`: Optional downloads directory used by file/PDF/table skills (default: `downloads`)
- `TOOL_EXEC_TIMEOUT_SECONDS`: Per-tool execution timeout guard (default: `30`)

### 3. Start Local System 1
Ensure Ollama is running and has the required model available:
```bash
ollama run gemma4:e4b
```

### 4. Run the Agent
```bash
python main.py
```

## Next Steps / Future Enhancements (Sprint 7+)

- Goal-driven autonomous exploration.
- Expanded sensor capabilities (filesystem monitoring, active web browsing).
- Deeper reinforcement learning on Critic feedback.
- Self-optimising Core Memory indexing to prevent drift.
