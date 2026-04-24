import re
from typing import Callable
from unittest.mock import MagicMock

import pytest

from src.core.llm_router import CognitiveRouter


def _schema_json(tool_name: str) -> str:
    return (
        '{"name":"%s","description":"Security test tool.",'
        '"parameters":{"type":"object","properties":{},"required":[]}}'
        % tool_name
    )


def _router_with_mock_registry() -> CognitiveRouter:
    router = CognitiveRouter.__new__(CognitiveRouter)
    router.registry = MagicMock()
    return router


def _disable_ast_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    def _noop(_code: str, _tool_name: str) -> None:
        return None

    monkeypatch.setattr(CognitiveRouter, "_validate_tool_code_ast", staticmethod(_noop))


def _disable_runtime_token_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    def _noop(_code: str, _tool_name: str) -> None:
        return None

    monkeypatch.setattr(CognitiveRouter, "_validate_dynamic_tool_token_scan", staticmethod(_noop))


@pytest.mark.parametrize(
    ("builtin_name", "call_source"),
    [
        ("getattr", "getattr(str, 'upper')"),
        ("setattr", "setattr(str, 'x', 1)"),
        ("delattr", "delattr(str, 'x')"),
        ("vars", "vars()"),
        ("globals", "globals()"),
        ("locals", "locals()"),
    ],
)
def test_tool_ast_gate_rejects_reflection_builtins(builtin_name: str, call_source: str) -> None:
    code = (
        "async def reflection_tool() -> object:\n"
        f"    return {call_source}\n"
    )

    with pytest.raises(ValueError, match=f"blocked builtin '{builtin_name}'"):
        CognitiveRouter._validate_tool_code_ast(code, "reflection_tool")


def test_tool_ast_gate_blocks_getattr_to_subclasses_bypass() -> None:
    code = (
        "async def subclasses_tool() -> object:\n"
        "    return getattr(object, '__subclasses__')()\n"
    )

    with pytest.raises(ValueError, match="blocked builtin 'getattr'"):
        CognitiveRouter._validate_tool_code_ast(code, "subclasses_tool")


def test_tool_ast_gate_blocks_import_module_name_string_literal() -> None:
    code = (
        "async def import_string_tool() -> str:\n"
        "    importer_name = '__im' + 'port__'\n"
        "    module_name = 'os'\n"
        "    return importer_name + module_name\n"
    )

    with pytest.raises(ValueError, match="blocked module string literal 'os'"):
        CognitiveRouter._validate_tool_code_ast(code, "import_string_tool")


def test_tool_ast_gate_blocks_type_mro_walk() -> None:
    code = (
        "async def mro_tool() -> object:\n"
        "    return type('Synthetic', (), {}).__mro__\n"
    )

    with pytest.raises(ValueError, match="__mro__"):
        CognitiveRouter._validate_tool_code_ast(code, "mro_tool")


def test_pytest_ast_gate_blocks_blocked_module_string_literal() -> None:
    pytest_code = (
        "def test_generated_module_literal() -> None:\n"
        "    assert 'os'\n"
    )

    with pytest.raises(ValueError, match="blocked module string literal 'os'"):
        CognitiveRouter._validate_pytest_code_ast(pytest_code, "safe_tool")


@pytest.mark.parametrize(
    ("source_line", "blocked_token"),
    [
        ("    return __subclasses__\n", "__subclasses__"),
        ("    return __mro__\n", "__mro__"),
        ("    return __globals__\n", "__globals__"),
        ("    return __builtins__\n", "__builtins__"),
        ("    return __import__\n", "__import__"),
        ("    return sys.modules\n", "sys.modules"),
    ],
)
def test_register_dynamic_tool_token_scan_blocks_non_comment_tokens(
    monkeypatch: pytest.MonkeyPatch,
    source_line: str,
    blocked_token: str,
) -> None:
    _disable_ast_validation(monkeypatch)
    router = _router_with_mock_registry()
    code = "async def scan_tool() -> object:\n" + source_line

    with pytest.raises(ValueError, match=re.escape(blocked_token)):
        router.register_dynamic_tool("scan_tool", code, _schema_json("scan_tool"))


def test_register_dynamic_tool_token_scan_ignores_comments(monkeypatch: pytest.MonkeyPatch) -> None:
    _disable_ast_validation(monkeypatch)
    router = _router_with_mock_registry()
    code = (
        "# __subclasses__ __mro__ __globals__ __builtins__ __import__ sys.modules\n"
        "async def scan_tool() -> str:\n"
        "    return 'ok'\n"
    )

    router.register_dynamic_tool("scan_tool", code, _schema_json("scan_tool"))

    router.registry.register_dynamic.assert_called_once()


@pytest.mark.parametrize(
    ("helper_name", "source_factory"),
    [
        ("getattr", lambda: "VALUE = getattr('abc', 'upper')\n"),
        ("vars", lambda: "VALUE = vars()\n"),
        ("type", lambda: "VALUE = type('Synthetic', (), {})\n"),
        ("object", lambda: "VALUE = object()\n"),
        ("dir", lambda: "VALUE = dir('abc')\n"),
    ],
)
def test_register_dynamic_tool_safe_builtins_omit_reflection_helpers(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    source_factory: Callable[[], str],
) -> None:
    _disable_ast_validation(monkeypatch)
    _disable_runtime_token_scan(monkeypatch)
    router = _router_with_mock_registry()
    code = source_factory() + "async def builtin_tool() -> str:\n    return 'ok'\n"

    with pytest.raises(NameError, match=helper_name):
        router.register_dynamic_tool("builtin_tool", code, _schema_json("builtin_tool"))
