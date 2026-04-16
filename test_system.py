"""
System Integration Test
Verifies all components work end-to-end without running the Telegram bot.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
results = []

def report(label, ok, detail=""):
    icon = PASS if ok else FAIL
    line = f"  [{icon}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    results.append(ok)


# ─────────────────────────────────────────────
# 1. ENV VARS
# ─────────────────────────────────────────────
print("\n[1] Environment")
report("TELEGRAM_BOT_TOKEN", bool(os.getenv("TELEGRAM_BOT_TOKEN")))
report("ADMIN_USER_ID",      bool(os.getenv("ADMIN_USER_ID")))
report("GROQ_API_KEY",       bool(os.getenv("GROQ_API_KEY")), os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
report("GEMINI_API_KEY",     bool(os.getenv("GEMINI_API_KEY")), f"USE_GEMINI={os.getenv('USE_GEMINI','False')}")


# ─────────────────────────────────────────────
# 2. IMPORTS
# ─────────────────────────────────────────────
print("\n[2] Imports")
try:
    from src.core.llm_router import CognitiveRouter, RequiresHITLError, RequiresMFAError
    report("llm_router", True)
except Exception as e:
    report("llm_router", False, str(e)); sys.exit(1)

try:
    from src.core.orchestrator import Orchestrator
    report("orchestrator", True)
except Exception as e:
    report("orchestrator", False, str(e)); sys.exit(1)

try:
    from src.memory.vector_db import VectorMemory
    from src.memory.ledger_db import LedgerMemory
    from src.memory.core_memory import CoreMemory
    report("memory modules", True)
except Exception as e:
    report("memory modules", False, str(e))


# ─────────────────────────────────────────────
# 3. MEMORY LAYER
# ─────────────────────────────────────────────
print("\n[3] Memory")
try:
    vm = VectorMemory(persist_dir="data/chroma_storage")
    report("VectorMemory init", True)
    mid = vm.add_memory("Test memory entry", {"type": "test"})
    report("VectorMemory write", True, f"id={mid[:8]}...")
    hits = vm.query_memory("test entry", n_results=1)
    report("VectorMemory query", len(hits) > 0, f"{len(hits)} result(s)")
except Exception as e:
    report("VectorMemory", False, str(e))

try:
    lm = LedgerMemory(db_path="data/ledger.db")
    tid = lm.add_task("Test task from integration test", priority=3)
    report("LedgerMemory write", bool(tid), f"task_id={tid}")
    tasks = lm.get_pending_tasks(limit=5)
    report("LedgerMemory read", isinstance(tasks, list), f"{len(tasks)} pending tasks")
except Exception as e:
    report("LedgerMemory", False, str(e))

try:
    cm = CoreMemory(memory_file_path="data/core_memory.json")
    cm.update("current_focus", "integration_test")
    ctx = cm.get_context_string()
    report("CoreMemory", "integration_test" in ctx)
except Exception as e:
    report("CoreMemory", False, str(e))


# ─────────────────────────────────────────────
# 4. LLM ROUTER
# ─────────────────────────────────────────────
print("\n[4] LLM Router")

async def test_router():
    router = CognitiveRouter()
    report("CognitiveRouter init", True,
           f"System2={'Groq' if router.groq_client else 'Gemini' if router.gemini_model else 'None'}")

    # System 2 test (Groq)
    if router.get_system_2_available():
        try:
            r = await asyncio.wait_for(
                router.route_to_system_2([
                    {"role": "user", "content": "Reply with exactly: PING"}
                ]),
                timeout=15.0
            )
            report("System 2 (Groq)", bool(r), repr(r[:60]))
        except Exception as e:
            report("System 2 (Groq)", False, str(e))
    else:
        report("System 2", False, "No provider configured")

    # System 1 test (Gemma via Ollama)
    try:
        r = await asyncio.wait_for(
            router.route_to_system_1([
                {"role": "user", "content": "Reply with exactly: PONG"}
            ]),
            timeout=60.0
        )
        report("System 1 (Gemma)", bool(r) and not r.startswith("[System 1 - Error]"), repr(r[:60]))
    except Exception as e:
        report("System 1 (Gemma)", False, str(e))

asyncio.run(test_router())


# ─────────────────────────────────────────────
# 5. ORCHESTRATOR — FULL FLOW
# ─────────────────────────────────────────────
print("\n[5] Orchestrator (full message flow)")

async def test_orchestrator():
    orch = Orchestrator()

    # Test 1: simple chat
    try:
        resp = await asyncio.wait_for(
            orch.process_message("Hello! Just a simple greeting.", user_id="test_user"),
            timeout=90.0
        )
        ok = bool(resp) and resp not in (
            "I don't have a plan for that.",
            "No valid response could be generated.",
            "An internal error occurred.",
        )
        report("Simple chat response", ok, repr(resp[:80]))
    except Exception as e:
        report("Simple chat response", False, str(e))

    # Test 2: energy budget is tracked
    try:
        state = orch._new_state("test", "hello")
        assert state["energy_remaining"] == 100
        state = await orch._deduct_energy(state, 10, "test")
        assert state["energy_remaining"] == 90
        report("Energy budget", True, "100 -> 90 after -10")
    except Exception as e:
        report("Energy budget", False, str(e))

    # Test 3: WORKERS tag parsing (supervisor prompt)
    import re, json
    response = "Hello! I can help you.\n\nWORKERS: []"
    lines = response.strip().split('\n')
    m = re.search(r'WORKERS:\s*(\[.*?\])', lines[-1])
    answer = '\n'.join(lines[:-1]).strip() if m else response
    plan = json.loads(m.group(1)) if m else None
    report("WORKERS tag parsing", m is not None and plan == [] and "Hello" in answer)

    orch.close()

asyncio.run(test_orchestrator())


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
total = len(results)
passed = sum(results)
failed = total - passed
print(f"\n{'-'*40}")
print(f"  {passed}/{total} checks passed", "All good!" if failed == 0 else f"  {failed} failed -- check above")
print(f"{'-'*40}\n")
sys.exit(0 if failed == 0 else 1)
