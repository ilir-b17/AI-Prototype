# Autonomous Biomimetic AI Agent - Sprint 1

## Project Overview

This is the foundational infrastructure for an autonomous, locally-hosted AI agent featuring:
- **Dual-system cognitive engine**: Local LLM (System 1) + Gemini API (System 2)
- **Long-term vector memory**: Hippocampus module
- **Central routing logic**: Prefrontal Cortex (coming soon)
- **Telegram interface**: Primary communication channel (current phase)

## Sprint 1 Objectives

Sprint 1 focuses on **Infrastructure and Interface** setup. The Telegram Bot API serves as the communication pipeline, with security controls to ensure only authorized access.

### Current Deliverables
- ✅ Modular project structure
- ✅ Environment configuration system
- ✅ Asynchronous Telegram bot with security controls
- ✅ Echo handler for pipeline verification
- ✅ Logging infrastructure

## Project Structure

```
AI_Prototype/
├── src/
│   ├── interfaces/          # Telegram bot implementation
│   ├── core/                # Future: Prefrontal cortex (routing logic)
│   └── memory/              # Future: Vector store & databases
├── logs/                    # System logs
├── main.py                  # Entry point
├── requirements.txt         # Python dependencies
├── .env.example             # Environment configuration template
└── README.md               # This file
```

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- A Telegram Bot Token
- Your Telegram User ID

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Obtain Telegram Bot Token

1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the prompts
3. Copy the bot token provided

### 3. Get Your Telegram User ID

1. Search for `@userinfobot` in Telegram
2. Send any message and it will reply with your User ID
3. Copy this ID

### 4. Configure Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in the following:
   ```
   TELEGRAM_BOT_TOKEN=<your_bot_token>
   ADMIN_USER_ID=<your_user_id>
   GEMINI_API_KEY=<placeholder_for_future_use>
   ```

### 5. Start the Bot

```bash
python main.py
```

You should see:
```
INFO - Telegram bot initialized and starting polling...
```

### 6. Test the Bot

In Telegram:
1. Send `/start` to your bot
2. You should receive: "System online. Awaiting input."
3. Send any text message
4. The bot will echo it back: "Echo: <your_message>"

## Security Features

- **User Authorization**: The bot verifies `ADMIN_USER_ID` for all interactions
- **Unauthorized Rejection**: Non-admin users receive an "Unauthorized." response
- **Logging**: All interactions are logged to `logs/telegram_bot.log` and console

## Architecture Notes

### Current Flow
```
Telegram User → Telegram API → telegram_bot.py → Echo Handler
                                      ↓
                              logs/telegram_bot.log
```

### Future Enhancements (Sprint 2+)
- Integration with local LLM (System 1) for fast processing
- Integration with Gemini API (System 2) for complex reasoning
- Vector memory storage (Hippocampus)
- Central executive routing (Prefrontal Cortex)
- Additional handlers for specific cognitive tasks

## Logs

Bot logs are saved to `logs/telegram_bot.log` and displayed in the console. Logs include:
- Authorization attempts
- Incoming messages
- System initialization status
- Error messages

## Troubleshooting

### Bot doesn't respond to messages
- Verify `ADMIN_USER_ID` in `.env` matches your Telegram User ID
- Check that the bot token is correct
- Ensure the bot is running: `python main.py`

### "TELEGRAM_BOT_TOKEN environment variable not set"
- Verify `.env` file exists and contains the token
- Run `python main.py` from the project root directory

### Module import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version is 3.10 or higher: `python --version`

## Code Standards

All code follows **PEP-8** standards:
- 4-space indentation
- Maximum line length: 88 characters (with some flexibility for readability)
- Docstrings for all modules, classes, and functions
- Type hints where applicable

## Next Steps

1. **Sprint 2**: Integrate local LLM for System 1 processing
2. **Sprint 3**: Integrate Gemini API for System 2 processing
3. **Sprint 4**: Implement vector memory with Hippocampus
4. **Sprint 5**: Build Prefrontal Cortex for intelligent routing and decision-making
5. **Sprint 6**: Add advanced features (context awareness, multi-turn conversations, etc.)

## Support & Contributing

For questions or issues, refer to:
- Python Telegram Bot Documentation: https://docs.python-telegram-bot.org/
- Telegram Bot API: https://core.telegram.org/bots/api
- Project guidelines will be added as development progresses

---

**Created**: April 14, 2026
**Status**: Sprint 1 - Infrastructure Phase
