Architectural Blueprint: AIDEN Multi-Agent Federation, Workspace Integration & Autonomous Optimization
1. Vision & Architectural Paradigm
To handle large-scale, asynchronous workflows—such as compiling 100-page policy briefs, managing stakeholder communications, and iteratively updating massive cloud repositories—AIDEN is evolving from a monolithic orchestrator into a Federated Hub-and-Spoke Model driven by a Blackboard Pattern.

The Hub (AIDEN Core): Retains exclusive control of the SQLite LedgerMemory, EnergyBudget, and MoralCharter. It executes no external APIs. It acts solely as the Director, breaking down incoming Epics and delegating them to specialized sub-agents.

The Spokes (Worker Agents): Modular, stateless agents with isolated toolsets (e.g., Google Workspace, Gemini Deep Research).

The Blackboard (Communication Matrix): Sub-agents never speak directly to one another. All inputs (emails, PDFs) and outputs (research plans, final docs) are written to AIDEN's central SQLite database. Agents communicate by reading and writing structured TypedDict JSON payloads to this shared Blackboard, ensuring zero data loss and perfect traceability.

2. The Federated Agent Roster
The ecosystem is populated by specialized agents, each governed by specific System Prompts and isolated tool boundaries.

A. The Communications Agent (ops_agent)
Role: Intake Coordinator and Temporal Manager.

Tools: read_inbox, parse_calendar_invite, send_email (with strict Telegram HITL intercept).

Duties: Triages incoming emails. Extracts requests and attachments, dropping them onto the Blackboard via the spawn_new_objective tool. Monitors the calendar to schedule predictive tasks (e.g., drafting talking points 30 minutes prior to a meeting).

B. The Expert Archiver Agent (librarian_agent)
Role: Cloud Infrastructure & Indexing Manager.

Tools: gdrive_create_folder, gdrive_move_file, vectorize_document.

Duties: Organizes raw attachments into Google Drive using a rigid taxonomy. Chunks and embeds PDFs into ChromaDB, ensuring AIDEN's local semantic memory stays perfectly synced with the cloud Drive.

C. The Deep Research Agent (analyst_agent)
Role: Heavy Compute & Synthesis Engine.

Tools: gemini_deep_research_api (routing to Gemini 3.1 Pro), read_vector_db.

Duties: Executes multi-hour research tasks. It follows a strict methodology: Stakeholder Mapping → Policy Gap Analysis → Perspective Debate → Citation Enforcement. Operates strictly in two phases (Planning vs. Execution) to prevent runaway compute costs.

D. The Workspace Publisher (publisher_agent)
Role: Formatter and Distributor.

Tools: gdocs_create_document, gdocs_append, gdocs_replace_text (Delta updates), gdrive_modify_permissions.

Duties: Translates raw Markdown from the analyst_agent into professional, paginated Google Docs. Applies corporate formatting and modifies IAM permissions to share final artifacts.

E. The Quality Assurance Agent (evaluator_agent) [NEW]
Role: Diagnostics and Continuous Optimization Loop.

Tools: query_ledger_pairs, update_agent_guidelines.

Duties: Operates asynchronously (e.g., during the Nocturnal Consolidation phase or post-Epic). It reads the Blackboard to compare the Input Request/Source Docs against the Final Output Artifact. It scores consistency, identifies recurring failure patterns, and writes corrective protocols to the system's dynamic memory.

3. Lifecycle of a Complex Workflow (Approval-Gated)
To prevent unguided compute expenditure and support long-term project lifecycles, research operates on an approval-gated, iterative methodology.

Phase 1: Intake, Planning & Approval Gate (Low Compute)
Intake: Admin emails a prompt and 3 PDFs. The ops_agent files them and creates the Epic on the Blackboard.

Scoping: AIDEN dispatches the analyst_agent using a fast, low-cost model. The agent performs a shallow read of the PDFs and creates a structured ResearchPlan (Chapters, Objectives, Data Gaps, Expected Token Cost).

The Intercept: The agent writes {"status": "requires_admin_approval", "artifact": "ResearchPlan"} to the Blackboard. AIDEN suspends the Epic and sends the plan to the Admin via Telegram.

Authorization: The Admin reviews, optionally adjusts ("Expand Chapter 2"), and replies YES.

Phase 2: Deep Research Execution (High Compute)
Execution: AIDEN re-activates the Epic and dispatches the analyst_agent using the Gemini 3.1 Pro API. The agent executes the approved plan iteratively.

Drafting: The publisher_agent takes the finalized Markdown, creates the Google Doc, and sends the URL to the Admin via Telegram.

Phase 3: Iterative Updates (Living Documents)
New Data: Two weeks later, the Admin emails a new PDF: "Update the report with this data."

Delta Analysis: The analyst_agent reads the existing Google Doc and the new PDF. It generates a "Delta" (a specific patch of revised text).

Targeted Injection: The publisher_agent uses gdocs_replace_text to insert the Delta into the exact relevant chapter, leaving the rest of the 100-page document untouched.

4. The Autonomous Evaluation & Optimization Loop
To ensure the multi-agent system becomes smarter over time, the evaluator_agent leverages the Blackboard as an evaluation metric.

The Diagnostic Cycle
Every time an Epic is completed, the Blackboard contains a perfect "Input/Output Pair":

Input: The initial email prompt + the indexed PDFs.

Output: The final Google Doc text + API execution logs.

The evaluator_agent analyzes these pairs using a strict grading rubric:

Fulfillment: Did the output answer the specific implicit and explicit constraints of the prompt?

Citation Fidelity: Are the citations real, or did the analyst_agent hallucinate a source?

Formatting: Did the publisher_agent follow the correct header structures?

The Optimization Injection
If the evaluator_agent detects a recurring failure pattern (e.g., "The analyst_agent consistently fails to address economic impacts when researching EU policy"), it triggers the Optimization Loop.

It calls the update_agent_guidelines tool.

This tool writes a new directive to AIDEN's CoreMemory (e.g., analyst_agent_dynamic_guidelines: "CRITICAL: Always include an economic impact subsection for EU policy requests.").

The next time AIDEN's Orchestrator dispatches the analyst_agent, it natively injects this new guideline into the System Prompt.

Result: The system continuously profiles its own weaknesses and writes permanent, dynamic corrections without requiring the Admin to modify the underlying Python code.

5. Development Milestones
Phase I: The Hub Refactor & Blackboard Setup. Deconstruct orchestrator.py. Enforce LangGraph TypedDict payloads. Ensure all agent handoffs pass through the SQLite Ledger.

Phase II: Playground Authorization. Generate Google Cloud Service Account credentials. Build gdocs_create and gdrive_upload using the langchain-google-community library.

Phase III: Spoke Deployment. Instantiate the ops_agent (Gmail IMAP polling) and librarian_agent (G-Drive / ChromaDB syncing).

Phase IV: Deep Research Integration. Connect the analyst_agent to the gemini-3.1-pro-preview API endpoint. Implement the Planning vs. Execution approval gate.

Phase V: Autonomous QA. Deploy the evaluator_agent to run chronologically after Epic completion, initiating the continuous optimization loop.