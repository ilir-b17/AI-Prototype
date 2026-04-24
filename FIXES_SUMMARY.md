# AI System Fixes Summary

## Slice 1: Security Hardening

- Hardened dynamic-tool AST validation by blocking reflection builtins (`getattr`, `setattr`, `delattr`, `vars`, `globals`, `locals`) and blocked module-name string literals.
- Removed reflection helpers (`getattr`, `vars`, `type`, `object`, `dir`) from dynamic tool safe builtins before `exec`.
- Added tokenizer-based defense-in-depth scanning before dynamic tool registration for `__subclasses__`, `__mro__`, `__globals__`, `__builtins__`, `__import__`, and `sys.modules` outside comments.
- Made MFA fail closed on startup unless `MFA_PASSPHRASE` is configured, at least 12 characters long, and free of common authorization words.
- Removed the old user-facing MFA hint and added optional `MFA_TOTP_SECRET` support for 6-digit TOTP authorization codes.
- Added focused security tests under `tests/security/` for sandbox bypass attempts, runtime token scanning, safe builtin removal, MFA validation, and TOTP verification.

## Slice 2: Concurrency & State Integrity

- Replaced per-user strong lock storage with weakref-backed lock leases and in-flight counters so LRU eviction never removes an active user lock.
- Made parallel worker batches re-raise `RequiresHITLError` and `RequiresMFAError` before merging successful sibling outputs.
- Preserved cumulative critic rejection counts across HITL resume and documented the HITL retry contract in the orchestrator module docstring.
- Refunded unused sibling-agent energy for blocked parallel batches, including capped predictive-budget bookkeeping when graph dispatch had already reserved energy.

## Slice 3: Cloud Redaction Correctness

- Narrowed cloud redaction so sensitive memory and sensory sub-blocks are stripped while non-sensitive tool and agent catalogs remain available for System 2 planning.
- Removed keyword-based sensitive-context escalation from untrusted free text; System 2 paths now use explicit orchestrator-owned redaction decisions.
- Added `cloud_payload_audit` ledger rows for every redacted System 2 payload, recording message counts, purpose, sensitivity flag, and outgoing payload SHA-256.
- Added focused tests for catalog preservation, supervisor fallback visibility, audit logging, and static enforcement of literal sensitivity decisions.

## Slice 4: Memory Pipeline & Persistence

- Surfaced deduplicated `nocturnal_core_facts` in core prompt context so consolidated insights can influence future supervisor planning.
- Added explicit-memory intent persistence for short user requests such as "remember that", "please remember", and "don't forget" without weakening the normal conversation-length threshold.
- Replaced fragile failed-response prefix filtering with an explicit `_turn_failed` state flag while preserving legitimate assistant text that starts with formerly reserved phrases.
- Normalized legacy vector-memory timestamps on local ChromaDB startup so timestamp-based pruning behaves consistently across old and new records.

## Slice 5: Supervisor Routing Accuracy

- Tightened multi-ticker fast-path routing so it requires at least two uppercase ticker candidates, finance intent, and no lowercase duplicate token elsewhere in the message.
- Added a cached, low-budget System 1 intent classifier for capability-query versus task/profile/casual routing, with regex fallback on timeout or malformed output.
- Moved volatile sensory and archival context out of the supervisor system prompt and into the current user-turn context block for more stable KV-cache reuse.
- Tightened age extraction so only first-person age statements update profile memory.

## Slice 6: Operational Robustness

- Added skill-load failure accounting and `/status` visibility for loaded skills, failed skill folders, pending MFA/HITL counts, and predictive energy budget.
- Fixed heartbeat dependency filtering so deferred tasks with unresolved dependencies are excluded from execution candidates.
- Routed Groq cooldown persistence through orchestrator-managed background tasks and surfaced startup restore failures to the admin notification queue.
- Rejected disabled Telegram voice input before fetching files and added document-caption approval handling for long tool-synthesis reviews.
- Replaced charter tier regex extraction with XML parsing, added malformed-charter diagnostics, and disabled critic/moral audits when only the fallback charter is active.
- Added a shared heartbeat task prefix constant and moved long synthesized tool code into temporary Telegram document attachments when approval text would exceed the safe message limit.

## Issues Identified

### 1. **Timeout Errors** (Critical)
The local model (gemma4:e4b) was timing out during processing because:
- Model responses were taking 15-20 seconds
- Timeouts were set too short (20-30 seconds)
- The critic_node was consistently timing out

### 2. **ChromaDB Telemetry Errors** (Non-critical)
- Posthog telemetry API mismatch causing warning messages
- These errors don't affect functionality but clutter the logs

### 3. **Inefficient Prompts**
- System prompts were verbose and caused slower processing
- More tokens = longer processing time

## Fixes Applied

### 1. Increased Timeouts
**Files Modified:** `src/core/orchestrator.py`, `src/core/llm_router.py`

- **Supervisor node**: 25s → 60s
- **Research agent**: 30s → 60s  
- **Coder agent**: 30s → 60s
- **Critic node**: 20s → 60s (this was the main culprit)
- **Salience evaluation**: 30s → 60s
- **LLM Router**: All Ollama calls 30s → 60s

**Why:** The gemma4:e4b model is a 9.6GB model that takes ~20 seconds per request. The 60-second timeout provides a safety buffer.

### 2. Suppressed ChromaDB Telemetry Errors
**Files Modified:** `src/interfaces/telegram_bot.py`, `.env`

- Added logging filter to suppress chromadb.telemetry errors
- Added environment variables to disable telemetry
- These errors don't affect functionality

### 3. Optimized Prompts
**File Modified:** `src/core/orchestrator.py`

**Before:**
```
"You are the Supervisor Node.\n{charter}\n{core_mem}\nAnalyze the user input and decide the next steps. Output a plan as a JSON list of worker nodes to call, e.g. ['research_agent'] or ['coder_agent'] or ['research_agent', 'coder_agent']. If it's a simple conversation, you can just output an empty list [] and provide a final response directly."
```

**After:**
```
"You are the Supervisor. Analyze the user input and decide next steps.\n{charter}\n{core_mem}\n\nOutput a JSON array of workers to call: ['research_agent'], ['coder_agent'], or []. For simple questions, output [] and provide a direct answer."
```

**Result:** ~30-40% fewer tokens in prompts = faster processing

### 4. Updated Dependencies
**File Modified:** `requirements.txt`

- Removed strict version pinning for packages that don't need it
- This allows pip to resolve compatible versions automatically

## Testing

To test the system, run:
```bash
python main.py
```

Then send a simple message like "Hello" or "Test" to your Telegram bot.

## Expected Behavior

✅ **Normal operation:**
- ChromaDB warnings should be suppressed
- No timeout errors (unless model takes >60s)
- System should respond within 30-60 seconds for simple queries
- Longer queries may take up to 120 seconds (multiple model calls)

⚠️ **If you still get timeouts:**
1. Consider using a smaller/faster model like `gemma2:2b`
2. Increase timeouts further if needed
3. Check Ollama is running: `ollama list`

## Model Performance Notes

Your current model **gemma4:e4b** (9.6GB):
- **Response time:** 15-25 seconds per call
- **Quality:** High (4-bit quantized Gemma 4)
- **Good for:** Complex reasoning, detailed responses

Alternative faster models:
- **gemma2:2b** - ~3-5 seconds, lighter reasoning
- **gemma2** - ~8-12 seconds, balanced

To switch models, update the `.env` or modify `orchestrator.py` line 41:
```python
local_model: str = "gemma2:2b"  # Change this
```

## Files Changed

1. `src/core/orchestrator.py` - Increased timeouts, optimized prompts
2. `src/core/llm_router.py` - Increased timeouts
3. `src/interfaces/telegram_bot.py` - Suppressed telemetry logging
4. `.env` - Added ChromaDB telemetry disable flags
5. `requirements.txt` - Relaxed version constraints

## Next Steps

1. Test the system with simple messages
2. If it works well, try more complex queries
3. Monitor response times and adjust timeouts if needed
4. Consider switching to a faster model if response time is critical
