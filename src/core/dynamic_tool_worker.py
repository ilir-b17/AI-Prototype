"""Isolated subprocess runtime for approved dynamic tools.

This module is intentionally clean-room: it imports only Python standard library
modules so the worker process cannot import AIDEN runtime objects by accident.
It also contains the small parent-side client used by the orchestrator to talk
to the worker over newline-delimited JSON on stdin/stdout.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import io
import json
import logging
import math
import os
import sys
import tempfile
import tokenize
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional

try:
    import resource
except ImportError:  # pragma: no cover - Windows path
    resource = None


logger = logging.getLogger(__name__)

DEFAULT_WORKER_CALL_TIMEOUT_SECONDS = 30.0
DEFAULT_WORKER_REGISTER_TIMEOUT_SECONDS = 10.0
DEFAULT_WORKER_PING_INTERVAL_SECONDS = 60.0
POSIX_CPU_SECONDS_PER_CALL = 5
POSIX_AS_LIMIT_BYTES = 256 * 1024 * 1024
POSIX_NOFILE_LIMIT = 32

_BLOCKED_TOP_LEVEL_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib", "socket",
    "importlib", "builtins", "ctypes", "multiprocessing", "threading",
    "signal", "pty", "popen", "pexpect", "atexit", "gc",
    "concurrent", "runpy", "code", "codeop", "compileall", "tempfile",
    "mmap", "pickle", "marshal", "shelve", "ast", "sqlite3", "dbm",
    "fileinput", "linecache", "io", "src",
}
_BLOCKED_DUNDER_ATTRIBUTES = {
    "__import__", "__loader__", "__builtins__", "__code__",
    "__globals__", "__subclasses__", "__mro__", "__dict__",
}
_BLOCKED_TOOL_BUILTINS = {
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "vars",
}
_BLOCKED_DYNAMIC_TOOL_TOKENS = {
    "__subclasses__", "__mro__", "__globals__", "__builtins__", "__import__",
}
_BLOCKED_DYNAMIC_TOOL_DOTTED_TOKENS = {"sys.modules"}
_DYNAMIC_TOOL_IGNORED_TOKEN_TYPES = frozenset({
    tokenize.COMMENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NL,
    tokenize.NEWLINE,
})
_BLOCKED_ASYNCIO_CALLS = {
    "create_subprocess_exec",
    "create_subprocess_shell",
}

_WORKER_TEMP_ROOT = os.path.realpath(
    os.environ.get("AIDEN_DYNAMIC_TOOL_TMPDIR") or tempfile.mkdtemp(prefix="aiden_dynamic_worker_")
)
_WORKER_TOOLS: Dict[str, Dict[str, Any]] = {}
_EXEC_USER_PYTHON_TOOL_NAME = "__exec_user_python__"


class DynamicToolWorkerProcessError(RuntimeError):
    """Raised by the parent-side client when the worker pipe/process fails."""


def _check_blocked_import(module_name: str, tool_name: str) -> None:
    base = module_name.split(".")[0]
    if base in _BLOCKED_TOP_LEVEL_MODULES:
        raise ValueError(
            f"Synthesised tool '{tool_name}' imports blocked module '{module_name}'."
        )
    stdlib_names = getattr(sys, "stdlib_module_names", set())
    if stdlib_names and base not in stdlib_names:
        raise ValueError(
            f"Synthesised tool '{tool_name}' imports non-stdlib module '{module_name}'."
        )


def _validate_ast_import_node(node: ast.AST, tool_name: str) -> bool:
    if isinstance(node, ast.Import):
        for alias in node.names:
            _check_blocked_import(alias.name, tool_name)
        return True
    if isinstance(node, ast.ImportFrom) and node.module:
        _check_blocked_import(node.module, tool_name)
        return True
    return False


def _validate_ast_attribute_node(node: ast.AST, tool_name: str) -> bool:
    if not isinstance(node, ast.Attribute):
        return False
    if node.attr in _BLOCKED_DUNDER_ATTRIBUTES:
        raise ValueError(
            f"Synthesised tool '{tool_name}' uses blocked dunder attribute access '{node.attr}'."
        )
    return True


def _validate_ast_asyncio_call(node: ast.AST, tool_name: str) -> bool:
    if not isinstance(node, ast.Attribute):
        return False
    if node.attr in _BLOCKED_ASYNCIO_CALLS:
        if isinstance(node.value, ast.Name) and node.value.id == "asyncio":
            raise ValueError(
                f"Synthesised tool '{tool_name}' calls blocked asyncio function '{node.attr}'."
            )
    return True


def _validate_ast_call_node(node: ast.AST, tool_name: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if isinstance(fn, ast.Name) and fn.id in _BLOCKED_TOOL_BUILTINS:
        raise ValueError(
            f"Synthesised tool '{tool_name}' calls blocked builtin '{fn.id}'."
        )
    return True


def _validate_ast_string_constant_node(node: ast.AST, tool_name: str) -> bool:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return False
    if node.value in _BLOCKED_TOP_LEVEL_MODULES:
        raise ValueError(
            f"Synthesised tool '{tool_name}' references blocked module string literal '{node.value}'."
        )
    return True


def validate_tool_code_ast(code: str, tool_name: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Synthesised code for '{tool_name}' has a syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if _validate_ast_import_node(node, tool_name):
            continue
        _validate_ast_asyncio_call(node, tool_name)
        if _validate_ast_attribute_node(node, tool_name):
            continue
        if _validate_ast_string_constant_node(node, tool_name):
            continue
        _validate_ast_call_node(node, tool_name)


def _dynamic_tool_significant_tokens(code: str, tool_name: str) -> List[tokenize.TokenInfo]:
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(code.encode("utf-8")).readline))
    except tokenize.TokenError as exc:
        raise ValueError(f"Synthesised code for '{tool_name}' could not be tokenized: {exc}") from exc
    return [token for token in tokens if token.type not in _DYNAMIC_TOOL_IGNORED_TOKEN_TYPES]


def _string_token_literal_value(token: tokenize.TokenInfo) -> Optional[str]:
    if token.type != tokenize.STRING:
        return None
    try:
        literal_value = ast.literal_eval(token.string)
    except (SyntaxError, ValueError):
        return None
    return literal_value if isinstance(literal_value, str) else None


def _validate_dynamic_tool_bare_token(token: tokenize.TokenInfo, tool_name: str) -> None:
    token_value = token.string
    literal_value = _string_token_literal_value(token)
    blocked_value = token_value if token_value in _BLOCKED_DYNAMIC_TOOL_TOKENS else literal_value
    if blocked_value in _BLOCKED_DYNAMIC_TOOL_TOKENS:
        raise ValueError(
            f"Synthesised tool '{tool_name}' contains blocked runtime token '{blocked_value}'."
        )


def _validate_dynamic_tool_dotted_token(
    tokens: List[tokenize.TokenInfo],
    index: int,
    tool_name: str,
) -> None:
    if tokens[index].string != "sys" or index + 2 >= len(tokens):
        return
    dotted = "".join(next_token.string for next_token in tokens[index:index + 3])
    if dotted in _BLOCKED_DYNAMIC_TOOL_DOTTED_TOKENS:
        raise ValueError(
            f"Synthesised tool '{tool_name}' contains blocked runtime token '{dotted}'."
        )


def validate_dynamic_tool_token_scan(code: str, tool_name: str) -> None:
    significant_tokens = _dynamic_tool_significant_tokens(code, tool_name)
    for index, token in enumerate(significant_tokens):
        _validate_dynamic_tool_bare_token(token, tool_name)
        _validate_dynamic_tool_dotted_token(significant_tokens, index, tool_name)


def _safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    if level != 0:
        raise ImportError("Relative imports are not allowed in dynamic tools.")
    base = str(name or "").split(".")[0]
    if base in _BLOCKED_TOP_LEVEL_MODULES:
        raise ImportError(f"Import of module '{name}' is blocked in dynamic tools.")
    stdlib_names = getattr(sys, "stdlib_module_names", set())
    if stdlib_names and base not in stdlib_names:
        raise ImportError(f"Only Python standard library imports are allowed, got '{name}'.")
    return __import__(name, globals_, locals_, fromlist, level)


def _map_worker_temp_path(file: Any) -> str:
    raw = os.fspath(file)
    if not isinstance(raw, str):
        raise PermissionError("Dynamic tools may only open text paths inside the worker temp directory.")

    normalized = raw.replace("\\", "/")
    if normalized == "/tmp" or normalized.startswith("/tmp/"):
        relative = normalized[5:].lstrip("/")
        candidate = os.path.join(_WORKER_TEMP_ROOT, relative)
    elif os.path.isabs(raw):
        candidate = raw
    else:
        candidate = os.path.join(_WORKER_TEMP_ROOT, raw)

    root = os.path.realpath(_WORKER_TEMP_ROOT)
    resolved = os.path.realpath(candidate)
    if resolved != root and not resolved.startswith(root + os.sep):
        raise PermissionError("Dynamic tool file access is limited to the worker temp directory.")
    return resolved


def _safe_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
    resolved = _map_worker_temp_path(file)
    if any(flag in mode for flag in ("w", "a", "x", "+")):
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
    return open(resolved, mode, *args, **kwargs)


def _safe_builtins() -> Dict[str, Any]:
    return {
        "None": None,
        "True": True,
        "False": False,
        "Exception": Exception,
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "bytes": bytes,
        "chr": chr,
        "dict": dict,
        "divmod": divmod,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hasattr": hasattr,
        "hash": hash,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "iter": iter,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "next": next,
        "open": _safe_open,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__build_class__": __build_class__,
        "__import__": _safe_import,
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _response_ok(result: Any = "") -> Dict[str, Any]:
    return {"ok": True, "result": _json_safe(result)}


def _response_error(error: Any) -> Dict[str, Any]:
    return {"ok": False, "error": str(error)}


def _format_execution_output(stdout_text: str, stderr_text: str) -> str:
    output = ""
    if stdout_text:
        output += f"--- STDOUT ---\n{stdout_text}\n"
    if stderr_text:
        output += f"--- STDERR ---\n{stderr_text}\n"
    return output or "Script executed successfully with no output."


async def __exec_user_python__(code_string: str) -> str:
    code = str(code_string or "")
    if not code.strip():
        raise ValueError("execute_python_sandbox requires a non-empty code_string")

    validate_tool_code_ast(code, _EXEC_USER_PYTHON_TOOL_NAME)
    validate_dynamic_tool_token_scan(code, _EXEC_USER_PYTHON_TOOL_NAME)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    sandbox_globals: Dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "__name__": "__main__",
    }

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(compile(code, "<user_python>", "exec"), sandbox_globals, sandbox_globals)
    except Exception as exc:
        raise RuntimeError(traceback.format_exc()) from exc

    return _format_execution_output(stdout_buffer.getvalue(), stderr_buffer.getvalue())


def _register_builtin_tools() -> None:
    _WORKER_TOOLS.setdefault(
        _EXEC_USER_PYTHON_TOOL_NAME,
        {
            "fn": __exec_user_python__,
            "schema": {
                "name": _EXEC_USER_PYTHON_TOOL_NAME,
                "description": (
                    "Builtin worker-only Python execution helper with AST validation, "
                    "blocked imports, and worker temp-dir confinement."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code_string": {
                            "type": "string",
                            "description": "Python source code to execute inside the worker sandbox.",
                        }
                    },
                    "required": ["code_string"],
                },
            },
        },
    )


def _apply_posix_start_limits() -> None:
    if resource is None:
        return
    for limit_name, value in (
        ("RLIMIT_AS", POSIX_AS_LIMIT_BYTES),
        ("RLIMIT_NOFILE", POSIX_NOFILE_LIMIT),
    ):
        limit = getattr(resource, limit_name, None)
        if limit is None:
            continue
        try:
            _soft, hard = resource.getrlimit(limit)
            new_hard = value if hard == resource.RLIM_INFINITY else min(hard, value)
            new_soft = min(value, new_hard)
            resource.setrlimit(limit, (new_soft, new_hard))
        except Exception:
            traceback.print_exc(file=sys.stderr)


def _apply_cpu_limit_for_call() -> None:
    if resource is None or not hasattr(resource, "RLIMIT_CPU"):
        return
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        used_seconds = int(math.ceil(float(usage.ru_utime) + float(usage.ru_stime)))
        target_soft = used_seconds + POSIX_CPU_SECONDS_PER_CALL
        _soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
        if hard != resource.RLIM_INFINITY:
            target_soft = min(target_soft, hard)
        resource.setrlimit(resource.RLIMIT_CPU, (target_soft, hard))
    except Exception:
        traceback.print_exc(file=sys.stderr)


def _validate_register_payload(message: Dict[str, Any]) -> tuple[str, str, Dict[str, Any]]:
    tool_name = str(message.get("tool_name") or "").strip()
    code = str(message.get("code") or "")
    schema = message.get("schema")
    if not tool_name:
        raise ValueError("register requires tool_name")
    if not code.strip():
        raise ValueError(f"register for '{tool_name}' requires code")
    if not isinstance(schema, dict):
        raise ValueError(f"register for '{tool_name}' requires schema object")
    schema_name = str(schema.get("name") or tool_name).strip()
    if schema_name != tool_name:
        raise ValueError(f"schema name '{schema_name}' does not match tool_name '{tool_name}'")
    return tool_name, code, dict(schema)


def _register_tool(message: Dict[str, Any]) -> Dict[str, Any]:
    tool_name, code, schema = _validate_register_payload(message)
    if tool_name == _EXEC_USER_PYTHON_TOOL_NAME:
        raise ValueError(f"Tool name '{tool_name}' is reserved")
    validate_tool_code_ast(code, tool_name)
    validate_dynamic_tool_token_scan(code, tool_name)

    module_globals: Dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "__name__": f"dynamic_tool_{tool_name}",
    }
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, f"<dynamic:{tool_name}>", "exec"), module_globals)

    fn = module_globals.get(tool_name)
    if not callable(fn):
        raise RuntimeError(f"Synthesised code does not define callable '{tool_name}'")

    _WORKER_TOOLS[tool_name] = {"fn": fn, "schema": schema}
    return _response_ok({"registered": tool_name})


async def _call_tool(message: Dict[str, Any]) -> Dict[str, Any]:
    tool_name = str(message.get("tool_name") or "").strip()
    arguments = message.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError(f"call for '{tool_name}' requires arguments object")
    entry = _WORKER_TOOLS.get(tool_name)
    if entry is None:
        raise KeyError(f"Unknown dynamic tool '{tool_name}'")

    _apply_cpu_limit_for_call()
    fn = entry["fn"]
    with contextlib.redirect_stdout(io.StringIO()):
        result = fn(**arguments)
        if inspect.isawaitable(result):
            result = await result
        if inspect.isasyncgen(result):
            parts = []
            async for chunk in result:
                parts.append(str(chunk))
            result = "\n".join(parts)
    return _response_ok(result)


async def _handle_message(message: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    op = str(message.get("op") or "").strip().lower()
    try:
        if op == "ping":
            return _response_ok("pong"), False
        if op == "register":
            return _register_tool(message), False
        if op == "call":
            return await _call_tool(message), False
        if op == "shutdown":
            return _response_ok("shutdown"), True
        return _response_error(f"Unknown op '{op}'"), False
    except Exception as exc:
        return _response_error(exc), False


def _run_worker_loop() -> int:
    os.makedirs(_WORKER_TEMP_ROOT, exist_ok=True)
    os.chdir(_WORKER_TEMP_ROOT)
    _apply_posix_start_limits()
    _register_builtin_tools()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    should_exit = False
    try:
        for line in sys.stdin:
            if not line:
                break
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("Request must be a JSON object")
                response, should_exit = loop.run_until_complete(_handle_message(message))
            except Exception as exc:
                response = _response_error(exc)
                should_exit = False
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            if should_exit:
                break
    finally:
        loop.close()
    return 0


class DynamicToolWorkerClient:
    """Parent-side JSON-RPC client for the dynamic tool worker subprocess."""

    def __init__(
        self,
        *,
        worker_path: Optional[str] = None,
        call_timeout_seconds: float = DEFAULT_WORKER_CALL_TIMEOUT_SECONDS,
        register_timeout_seconds: float = DEFAULT_WORKER_REGISTER_TIMEOUT_SECONDS,
        ping_interval_seconds: float = DEFAULT_WORKER_PING_INTERVAL_SECONDS,
    ) -> None:
        self.worker_path = worker_path or os.path.abspath(__file__)
        self.call_timeout_seconds = float(call_timeout_seconds)
        self.register_timeout_seconds = float(register_timeout_seconds)
        self.ping_interval_seconds = float(ping_interval_seconds)
        self.process: Optional[asyncio.subprocess.Process] = None
        self._request_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._stderr_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._restart_callback: Optional[Callable[[], Awaitable[None]]] = None
        self._restart_callback_running = False

    @property
    def process_id(self) -> Optional[int]:
        return self.process.pid if self.process is not None else None

    def set_restart_callback(self, callback: Optional[Callable[[], Awaitable[None]]]) -> None:
        self._restart_callback = callback

    def _build_worker_env(self, temp_dir: str) -> Dict[str, str]:
        env: Dict[str, str] = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "",
            "AIDEN_DYNAMIC_TOOL_TMPDIR": temp_dir,
            "TMPDIR": temp_dir,
            "TEMP": temp_dir,
            "TMP": temp_dir,
        }
        for key in ("PATH", "SYSTEMROOT", "WINDIR"):
            value = os.environ.get(key)
            if value:
                env[key] = value
        return env

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self.process is not None and self.process.returncode is None:
                if self._ping_task is None or self._ping_task.done():
                    self._ping_task = asyncio.create_task(self._ping_loop())
                return
            if self.process is not None:
                await self._stop_process(kill=True)

            self._temp_dir = tempfile.TemporaryDirectory(prefix="aiden_dynamic_worker_parent_")
            temp_dir = self._temp_dir.name
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                self.worker_path,
                cwd=temp_dir,
                env=self._build_worker_env(temp_dir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._stderr_task = asyncio.create_task(self._drain_stderr(self.process))
            if self._ping_task is None or self._ping_task.done():
                self._ping_task = asyncio.create_task(self._ping_loop())
            logger.info("Dynamic tool worker started pid=%s", self.process.pid)

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        if process.stderr is None:
            return
        while True:
            line = await process.stderr.readline()
            if not line:
                return
            logger.debug(
                "dynamic-tool-worker[%s]: %s",
                process.pid,
                line.decode("utf-8", errors="replace").rstrip(),
            )

    async def _ping_loop(self) -> None:
        while True:
            await asyncio.sleep(self.ping_interval_seconds)
            response = await self.ping(reload_on_failure=True)
            if not response.get("ok"):
                logger.critical("Dynamic tool worker ping failed: %s", response.get("error"))

    async def _request_once(self, message: Dict[str, Any]) -> Dict[str, Any]:
        await self.start()
        process = self.process
        if process is None or process.stdin is None or process.stdout is None:
            raise DynamicToolWorkerProcessError("worker process is not available")
        if process.returncode is not None:
            raise DynamicToolWorkerProcessError(f"worker exited with code {process.returncode}")

        payload = json.dumps(message, separators=(",", ":")) + "\n"
        try:
            process.stdin.write(payload.encode("utf-8"))
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise DynamicToolWorkerProcessError(f"worker pipe broke: {exc}") from exc

        line = await process.stdout.readline()
        if not line:
            returncode = process.returncode
            if returncode is None:
                returncode = await process.wait()
            raise DynamicToolWorkerProcessError(f"worker exited before responding, code={returncode}")
        try:
            response = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise DynamicToolWorkerProcessError(f"worker returned invalid JSON: {exc}") from exc
        if not isinstance(response, dict):
            raise DynamicToolWorkerProcessError("worker returned non-object response")
        return response

    async def _request(
        self,
        message: Dict[str, Any],
        *,
        timeout_seconds: float,
        reload_on_failure: bool = True,
    ) -> Dict[str, Any]:
        failure: Optional[str] = None
        async with self._request_lock:
            try:
                return await asyncio.wait_for(self._request_once(message), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                failure = f"worker request timed out after {timeout_seconds:.1f}s"
            except DynamicToolWorkerProcessError as exc:
                failure = str(exc)

        await self._restart_after_failure(failure or "unknown worker failure", reload_on_failure=reload_on_failure)
        return _response_error(f"Dynamic tool worker unavailable: {failure}")

    async def _restart_after_failure(self, reason: str, *, reload_on_failure: bool) -> None:
        logger.critical("Dynamic tool worker failure detected: %s. Respawning worker.", reason)
        await self._stop_process(kill=True)
        await self.start()
        if reload_on_failure:
            await self._invoke_restart_callback()

    async def _invoke_restart_callback(self) -> None:
        callback = self._restart_callback
        if callback is None or self._restart_callback_running:
            return
        self._restart_callback_running = True
        try:
            await callback()
        finally:
            self._restart_callback_running = False

    async def register_tool(self, tool_name: str, code: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request(
            {"op": "register", "tool_name": tool_name, "code": code, "schema": schema},
            timeout_seconds=self.register_timeout_seconds,
        )

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return await self._request(
            {"op": "call", "tool_name": tool_name, "arguments": arguments or {}},
            timeout_seconds=self.call_timeout_seconds,
        )

    async def ping(self, *, reload_on_failure: bool = True) -> Dict[str, Any]:
        return await self._request(
            {"op": "ping"},
            timeout_seconds=min(5.0, max(1.0, self.call_timeout_seconds)),
            reload_on_failure=reload_on_failure,
        )

    async def _stop_process(self, *, kill: bool) -> None:
        process = self.process
        self.process = None
        if process is None:
            await self._cleanup_process_handles()
            return
        if process.returncode is None:
            if kill:
                process.kill()
            else:
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        await self._cleanup_process_handles()

    async def _cleanup_process_handles(self) -> None:
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stderr_task
            self._stderr_task = None
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    async def shutdown(self) -> None:
        if self._ping_task is not None:
            self._ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ping_task
            self._ping_task = None

        process = self.process
        if process is not None and process.returncode is None:
            async with self._request_lock:
                try:
                    await asyncio.wait_for(self._request_once({"op": "shutdown"}), timeout=3.0)
                except Exception:
                    pass
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        self.process = None
        await self._cleanup_process_handles()


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(_run_worker_loop())