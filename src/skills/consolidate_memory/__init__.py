import json
import logging
import os

import aiosqlite

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

    ledger_db_path = "data/ledger.db"
    core_memory_path = "data/core_memory.json"

    try:
        # Read chat history directly via aiosqlite (cross-user, no user_id filter)
        async with aiosqlite.connect(ledger_db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT role, content FROM ("
                "  SELECT id, role, content FROM chat_history"
                "  WHERE user_id != 'heartbeat'"
                "  ORDER BY id DESC LIMIT 50"
                ") ORDER BY id ASC"
            )
            rows = await cursor.fetchall()

        history = [{"role": r["role"], "content": r["content"]} for r in rows]
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

        from src.memory.core_memory import CoreMemory
        core = CoreMemory(memory_file_path=core_memory_path)
        state = await core.get_all()
        existing = state.get("consolidated_insights", "")
        new_block = " | ".join(insights)
        merged = f"{existing} | {new_block}".strip(" |") if existing else new_block
        await core.update("consolidated_insights", merged)

        logger.info(f"consolidate_memory: wrote {len(insights)} insights to core memory")
        return f"Memory consolidation complete. Extracted {len(insights)} insights: {new_block[:200]}"

    except json.JSONDecodeError as e:
        return f"Consolidation failed: could not parse LLM response ({e})."
    except Exception as e:
        logger.error(f"consolidate_memory error: {e}", exc_info=True)
        return f"Consolidation failed: {e}"
