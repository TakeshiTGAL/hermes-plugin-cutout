"""Remember the plan the public pre_tool_call hook showed. Hermes asks the person."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time

HOST_PROCESS_ENV = "HERMES_PLUGIN_HOST_PROCESS"
APPROVAL_LIMIT = 1500
TTL_S = 600
HOST_BLOCK = (
    "BLOCKED: plugins.isolation is host, so nothing was written."
)
NOT_SENT = (
    "The approval line would not fit in one message, so it was not sent and nothing was written."
)
DENIED = "The action was not approved, so nothing was written."
MISMATCH = "The plan changed after the preview, so nothing was written."
EXPIRED = "The approval is too old, so nothing was written."
UNATTENDED_MESSAGE = (
    "This run has no person to approve it, so nothing was written."
)

_ON = frozenset({"1", "true", "yes", "on"})
_UNATTENDED_PLATFORMS = frozenset({"webhook", "msgraph_webhook", "api_server"})
_LOCK = threading.Lock()
_STASH: dict[tuple[str, str, str], tuple[str, float]] = {}
_CTX = None


def in_plugin_host_process() -> bool:
    return bool(os.environ.get(HOST_PROCESS_ENV, "").strip())


def line_is_sendable(line: str) -> bool:
    if not line or "\n" in line or "\r" in line or "`" in line:
        return False
    return len(line) <= APPROVAL_LIMIT


def canonical_args(args: dict) -> str:
    return json.dumps(args, sort_keys=True, default=str, separators=(",", ":"))


def rule_key(args: dict) -> str:
    digest = hashlib.sha256(canonical_args(args).encode("utf-8")).hexdigest()[:16]
    return f"cutout_png:{digest}"


def _key(args: dict, session_id: str, task_id: str) -> tuple[str, str, str]:
    return (canonical_args(args), str(session_id or ""), str(task_id or ""))


def _drop_room(session_id: str, task_id: str) -> None:
    room = (str(session_id or ""), str(task_id or ""))
    for key in [key for key in _STASH if key[1:] == room]:
        del _STASH[key]


def clear() -> None:
    with _LOCK:
        _STASH.clear()


def clear_room(session_id: str = "", task_id: str = "") -> None:
    """Drop saved plans for this session and task. Other rooms keep theirs."""
    with _LOCK:
        _drop_room(session_id, task_id)


def remember(args: dict, session_id: str, task_id: str, line: str) -> None:
    """Drop saved plans for this session and task, then store this one."""
    key = _key(args, session_id, task_id)
    with _LOCK:
        _drop_room(session_id, task_id)
        _STASH[key] = (line, time.monotonic())


def pop(args: dict, session_id: str, task_id: str) -> tuple[str, float] | None:
    key = _key(args, session_id, task_id)
    with _LOCK:
        return _STASH.pop(key, None)


def consume(args: dict, session_id: str, task_id: str, line: str) -> tuple[str, str]:
    """Pop the saved plan. approved, denied, mismatch, or expired."""
    found = pop(args, session_id, task_id)
    if found is None:
        return "denied", DENIED
    stored, made = found
    if time.monotonic() - made > TTL_S:
        return "expired", EXPIRED
    if stored != line:
        return "mismatch", MISMATCH
    return "approved", ""


class SessionUnreadable(Exception):
    """The loaded session reader did not answer."""


def session_value(name: str) -> str:
    """Session value from a module Hermes already loaded, else the process environment.

    This does not import that module. ``get_session_env`` reads a ContextVar
    before the process environment, which is how a cron run carries
    ``HERMES_CRON_SESSION`` without exporting it. When the reader is present
    and raises, the caller does not write.
    """
    module = sys.modules.get("gateway.session_context")
    reader = getattr(module, "get_session_env", None) if module is not None else None
    if not callable(reader):
        return os.environ.get(name, "") or ""
    try:
        return str(reader(name, "") or "")
    except Exception as exc:
        raise SessionUnreadable(name) from exc


def session_source() -> str:
    """Desktop source only when this task bound it.

    An inherited ``HERMES_SESSION_SOURCE`` is not that session. When the
    gateway module is absent, the process environment is the only signal.
    """
    module = sys.modules.get("gateway.session_context")
    if module is None:
        return os.environ.get("HERMES_SESSION_SOURCE", "") or ""
    var_map = getattr(module, "_VAR_MAP", None)
    unset = getattr(module, "_UNSET", None)
    var = var_map.get("HERMES_SESSION_SOURCE") if isinstance(var_map, dict) else None
    if var is None or unset is None:
        return os.environ.get("HERMES_SESSION_SOURCE", "") or ""
    try:
        value = var.get()
    except Exception as exc:
        raise SessionUnreadable("HERMES_SESSION_SOURCE") from exc
    if value is unset:
        return ""
    return str(value or "")


def remember_context(ctx) -> None:
    """Keep the ctx from register. The cron check reads it later."""
    global _CTX
    _CTX = ctx


def _cron_identity() -> bool:
    """``ctx.current_cron_execution()`` when this Hermes has that reader.

    A missing reader means this process has no cron identity to read.
    A reader that raises did not answer, so the caller does not write.
    """
    reader = getattr(_CTX, "current_cron_execution", None)
    if not callable(reader):
        return False
    try:
        return reader() is not None
    except Exception:
        return True


def _flag(name: str) -> bool:
    return session_value(name).strip().lower() in _ON


def unattended() -> bool:
    try:
        if _flag("HERMES_CRON_SESSION") or _flag("HERMES_SINGLE_QUERY_SESSION"):
            return True
        platform = session_value("HERMES_SESSION_PLATFORM").strip().lower()
    except SessionUnreadable:
        return True
    if platform in _UNATTENDED_PLATFORMS:
        return True
    return _cron_identity()
