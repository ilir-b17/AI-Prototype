import asyncio
import json
import logging
import os

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a memory consolidation agent. Analyze the conversation history "
    "and extract facts worth remembering permanently:\n"
    "  - Persistent user preferences\n"
    "  - Recurring topics or interests\n"
    "  - Rules/constraints the user has stated\n"
    "  - Important facts about the user's environment or goals\n\n"
    "Return ONLY a JSON object: {\"insights\": [\"...\", \"...\"]}\n"
    "Each insight must be under 100 characters. Return no more than 10."
)


async def consolidate_memory() -> str:
    """Long-term cognitive compression: distil recent chat history into core memory."""
    logger.info("consolidate_memory: starting compression pass")

    try:
        from src.core.runtime_context import get_core_memory, get_ledger
        from src.memory.core_memory import CoreMemory
        from src.memory.ledger_db import LedgerMemory

        ledger = get_ledger()
        owns_ledger = False
        if ledger is None:
            ledger = LedgerMemory()
            await ledger.initialize()
            owns_ledger = True

        try:
            user_ids = await ledger.get_recent_user_ids(limit=10)
            histories = await asyncio.gather(
                *(ledger.get_chat_history(user_id, limit=5) for user_id in user_ids)
            ) if user_ids else []
            history = [turn for user_history in histories for turn in user_history]
        finally:
            if owns_ledger:
                await ledger.close()

        if not history:
            return json.dumps({
                "status": "success",
                "message": "No chat history to consolidate.",
                "insights": []
            })

        history_text = "\n".join(
            f"{t['role'].upper()}: {t['content'][:200]}" for t in history
        )

        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            return json.dumps({
                "status": "error",
                "message": "Consolidation skipped: GROQ_API_KEY not configured.",
                "suggestion": "Configure GROQ_API_KEY in the .env file to enable LLM-driven memory consolidation."
            })

        import groq as groq_lib
        client = groq_lib.AsyncGroq(api_key=groq_key)

        try:
            timeout_seconds = float(os.getenv("TOOL_EXEC_TIMEOUT_SECONDS", "30.0"))
        except ValueError:
            timeout_seconds = 30.0

        async def _call_llm():
            return await client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": f"Chat history:\n{history_text}"},
                ],
                max_tokens=500,
                temperature=0.3,
            )

        try:
            resp = await asyncio.wait_for(_call_llm(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return json.dumps({
                "status": "error",
                "message": f"Consolidation LLM call timed out after {timeout_seconds}s.",
                "suggestion": "The LLM API might be slow. Try again later."
            })

        raw = resp.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        data = json.loads(raw)
        insights: list = data.get("insights", [])

        if not insights:
            return json.dumps({
                "status": "success",
                "message": "Consolidation complete: no new insights extracted.",
                "insights": []
            })

        core = get_core_memory() or CoreMemory()
        state = await core.get_all()
        existing = state.get("consolidated_insights", "")
        new_block = " | ".join(insights)
        merged = f"{existing} | {new_block}".strip(" |") if existing else new_block
        await core.update("consolidated_insights", merged)

        logger.info(f"consolidate_memory: wrote {len(insights)} insights to core memory")
        return json.dumps({
            "status": "success",
            "message": f"Memory consolidation complete. Extracted {len(insights)} insights.",
            "insights": insights
        }, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            "status": "error",
            "message": "Consolidation failed",
            "details": f"Could not parse LLM response: {str(e)}",
            "suggestion": "The LLM returned malformed JSON. This is usually transient, try again."
        })
    except Exception as e:
        logger.error(f"consolidate_memory error: {e}", exc_info=True)
        return json.dumps({
            "status": "error",
            "message": "Consolidation failed",
            "details": str(e)
        })
