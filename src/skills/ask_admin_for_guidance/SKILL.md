# ask_admin_for_guidance

Pauses the state graph and escalates to the Admin for strategic input when
the agent faces genuine ambiguity it cannot resolve autonomously.
**Triggers HITL (Human-in-the-Loop)** — the orchestrator intercepts this call,
stores the pending state, and sends the question to the Admin via Telegram.
Execution resumes when the Admin replies.

## Schema

```json
{
  "name": "ask_admin_for_guidance",
  "description": "Pauses execution and asks the Admin for strategic input when facing ambiguity.",
  "parameters": {
    "type": "object",
    "properties": {
      "context_summary": {
        "type": "string",
        "description": "Summary of the current situation."
      },
      "specific_question": {
        "type": "string",
        "description": "The specific question for the Admin."
      }
    },
    "required": ["context_summary", "specific_question"]
  }
}
```
