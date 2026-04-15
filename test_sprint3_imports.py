#!/usr/bin/env python
"""Quick test to verify Sprint 3 imports work."""

try:
    from src.core.llm_router import CognitiveRouter
    print("✓ CognitiveRouter imported successfully")
except Exception as e:
    print(f"✗ Failed to import CognitiveRouter: {e}")
    exit(1)

try:
    from src.core.executive import ExecutiveAgent
    print("✓ ExecutiveAgent imported successfully")
except Exception as e:
    print(f"✗ Failed to import ExecutiveAgent: {e}")
    exit(1)

try:
    from src.interfaces.telegram_bot import main
    print("✓ Telegram bot imports successfully")
except Exception as e:
    print(f"✗ Failed to import telegram_bot: {e}")
    exit(1)

print("\nAll Sprint 3 imports verified successfully!")
