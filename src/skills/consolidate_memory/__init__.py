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


def _sync_consolidate(
    ledger_db_path: str = "data/ledger.db",
    core_memory_path: str = "data/core_memory.json",
) -> str:
    try:
        from src.memory.ledger_db import LedgerMemory
        from src.memory.core_memory import CoreMemory

        ledger = LedgerMemory(db_path=ledger_db_path)
        cursor = ledger.connection.cursor()
        cursor.execute("""
            SELECT role, content FROM (
                SELECT id, role, content
                FROM chat_history
                WHERE user_id != 'heartbeat'
                ORDER BY id DESC
                LIMIT 50
            ) ORDER BY id ASC
        """)
        history = [{"role": r["role"], "content": r["content"]} for r in cursor.fetchall()]
        ledger.close()

        if not history:
            return "No chat history to consolidate."

        history_text = "\n".join(
            f"{t['role'].upper()}: {t['content'][:200]}" for t in history
        )

        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            return "Consolidation skipped: GROQ_API_KEY not configured."

        import groq as groq_lib
        client = groq_lib.Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Chat history:\n{history_text}"},
            ],
            max_tokens=500,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        data = json.loads(raw)
        insights: list = data.get("insights", [])

        if not insights:
            return "Consolidation complete: no new insights extracted."

        core = CoreMemory(memory_file_path=core_memory_path)
        existing = core.get_all().get("consolidated_insights", "")
        new_block = " | ".join(insights)
        merged = f"{existing} | {new_block}".strip(" |") if existing else new_block
        core.update("consolidated_insights", merged)

        logger.info(f"consolidate_memory: wrote {len(insights)} insights to core memory")
        return f"Memory consolidation complete. Extracted {len(insights)} insights: {new_block[:200]}"

    except json.JSONDecodeError as e:
        return f"Consolidation failed: could not parse LLM response ({e})."
    except Exception as e:
        logger.error(f"consolidate_memory error: {e}", exc_info=True)
        return f"Consolidation failed: {e}"


async def consolidate_memory() -> str:
    """Long-term cognitive compression: distil recent chat history into core memory."""
    logger.info("consolidate_memory: starting compression pass")
    return await asyncio.to_thread(_sync_consolidate)
