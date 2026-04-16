import logging

logger = logging.getLogger(__name__)


async def ask_admin_for_guidance(context_summary: str, specific_question: str) -> str:
    """
    Intercepted by the router before execution — raises RequiresHITLError,
    which pauses the state graph and routes the question to the Admin via Telegram.
    This function body is never reached during normal operation.
    """
    logger.info(f"ask_admin_for_guidance: {specific_question}")
    return f"Guidance requested: {specific_question}"
