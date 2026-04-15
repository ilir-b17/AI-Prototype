"""
Main entry point for the Autonomous Biomimetic AI Agent.

This script initializes and runs the Telegram bot interface, which serves as the
primary communication channel for the AI agent.
"""

import sys
from dotenv import load_dotenv

# Load environment variables BEFORE importing any modules
load_dotenv()

from src.interfaces.telegram_bot import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting bot: {e}")
        sys.exit(1)
