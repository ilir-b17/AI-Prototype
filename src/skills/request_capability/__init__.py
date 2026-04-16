import logging

logger = logging.getLogger(__name__)


async def request_capability(gap_description: str, suggested_tool_name: str) -> str:
    """
    Intercepted by the router before execution — raises RequiresCapabilitySynthesisError,
    which triggers System 2 to synthesise a new tool for Admin approval.
    This function body is never reached during normal operation.
    """
    logger.info(f"request_capability: {gap_description}")
    return f"Capability gap registered: {gap_description}"
