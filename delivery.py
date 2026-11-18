"""Show a finished PNG in a chat by a MEDIA line on the reply.

Hermes attaches a local file when the assistant reply contains a line
``MEDIA:/absolute/path``. The gateway copies that line out of a tool result
only for its own image and speech tools, so a plugin result that only carries
the line inside JSON does not arrive. ``transform_llm_output`` adds a line for
each PNG this turn wrote that the reply left out. ``pre_llm_call`` runs once at
the start of a turn and forgets what an earlier turn did not deliver, because
Hermes skips ``transform_llm_output`` when a reply is empty. A ``/btw`` answer
and a background review share the session and do not take those lines. The
turn reply does, including when compression continues that turn under a new
session id. A line counts as already present only when Hermes would attach it.

CLI, TUI, SMS, ACP, an empty platform name, and the local gateway keep the
path and do not gain the line. Printing the line there would show the tag as
text. ``api_server``, ``webhook``, and ``msgraph_webhook`` are refused in
``gates.unattended`` before a write, so this module does not add a line for
them. Hermes can inline a ``MEDIA:`` image as a data URL on the ``api_server``
chat, completions, and responses endpoints. This plugin does not write there.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time

# cli and tui print a MEDIA line as text (platform hints in Hermes main and 0.23.2).
# sms send is a text form. acp has no MEDIA extractor. An empty name is not a
# chat that can attach a file. local keeps the gateway text and does not attach
# a file. api_server, webhook, and msgraph_webhook are refused before a write
# (gates._UNATTENDED_PLATFORMS). They stay in this set so a reply does not gain
# a MEDIA line. api_server itself can inline an image MEDIA tag on chat,
# completions, and responses (gateway/platforms/api_server.py).
NO_ATTACHMENT_PLATFORMS = frozenset({
    "",
    "cli",
    "tui",
    "acp",
    "sms",
    "local",
    "api_server",
    "webhook",
    "msgraph_webhook",
})

# Desktop binds HERMES_SESSION_SOURCE and leaves HERMES_SESSION_PLATFORM empty.
# Other sources (cli, tui, oneshot, tool, kanban, a2a, and any --source value)
# are not a chat that can attach a file.
SOURCE_CHATS = frozenset({"desktop"})

# Hermes turns a Telegram raster over 1 MiB into a JPEG on white before
# sendPhoto. [[as_document]] sends the original PNG through send_document.
FILE_PLATFORMS = frozenset({"telegram"})

PENDING_TTL_SECONDS = 30 * 60
_MAX_SESSIONS = 40
_TURN_HOME_MAX = 40

_pending: dict[str, list[tuple[float, str]]] = {}  # bounded: drop entries after 30 min, cap 40 sessions, start_turn clears that session
_order: list[str] = []  # bounded: drop a session id after 30 min, cap 40 sessions, start_turn clears that session
_turn_home: dict[str, str] = {}
_turn_order: list[str] = []
_lock = threading.Lock()


def platform_name(platform: str) -> str:
    return (platform or "").strip().lower()


def can_attach(platform: str) -> bool:
    return platform_name(platform) not in NO_ATTACHMENT_PLATFORMS


def sends_as_file(platform: str) -> bool:
    return platform_name(platform) in FILE_PLATFORMS


def media_block(path: str, platform: str) -> str:
    line = "MEDIA:" + path
    if sends_as_file(platform):
        return "[[as_document]]\n" + line
    return line


def _drop_locked(session_id: str) -> None:
    _pending.pop(session_id, None)
    _order[:] = [sid for sid in _order if sid != session_id]


def clear(session_id: str | None = None) -> None:
    with _lock:
        if session_id is None:
            _pending.clear()
            _order.clear()
            _turn_home.clear()
            _turn_order.clear()
            return
        _drop_locked(session_id)


def _touch_locked(session_id: str, now: float) -> None:
    stale = [
        sid
        for sid in _order
        if sid != session_id and not any(now - stamp < PENDING_TTL_SECONDS for stamp, _path in _pending.get(sid, []))
    ]
    for sid in stale:
        _drop_locked(sid)
    _order[:] = [sid for sid in _order if sid != session_id]
    _order.append(session_id)
    while len(_order) > _MAX_SESSIONS:
        _drop_locked(_order[0])


def remember(session_id: str, path: str) -> None:
    if not session_id or not path:
        return
    now = time.time()
    with _lock:
        items = [(stamp, item) for stamp, item in _pending.get(session_id, []) if now - stamp < PENDING_TTL_SECONDS]
        items.append((now, path))
        _pending[session_id] = items
        _touch_locked(session_id, now)


def take(session_id: str) -> list[str]:
    now = time.time()
    with _lock:
        items = _pending.pop(session_id, [])
        _order[:] = [sid for sid in _order if sid != session_id]
    return [path for stamp, path in items if now - stamp < PENDING_TTL_SECONDS]


def start_turn(session_id: str = "", turn_id: str = "", **_kwargs) -> None:
    """Forget PNGs an earlier turn of this session did not deliver.

    The first session id of a turn stays the home of PNGs written before
    compression continues that same turn under a new session id.
    """
    if session_id:
        clear(session_id)
    if not session_id or not isinstance(turn_id, str) or not turn_id:
        return
    with _lock:
        if turn_id in _turn_home:
            return
        _turn_home[turn_id] = session_id
        _turn_order.append(turn_id)
        while len(_turn_order) > _TURN_HOME_MAX:
            _turn_home.pop(_turn_order.pop(0), None)


# The user turn binds this write origin. ``foreground`` is the unbound default.
# A fork names itself (``side_question``, ``background_review``, ``plugin_fork:…``).
_TURN_WRITE_ORIGINS = frozenset({"foreground", "assistant_tool"})


def _whitelist_is_set() -> bool:
    """True when this thread is inside a fork's tool whitelist.

    Hermes also runs ``transform_llm_output`` on a worker, which does not see
    that thread-local. The write origin below is what the worker does see.
    """
    module = sys.modules.get("hermes_cli.plugins")
    local = getattr(module, "_thread_tool_whitelist", None) if module is not None else None
    return getattr(local, "allowed", None) is not None


def _write_origin() -> str:
    module = sys.modules.get("tools.skill_provenance")
    reader = getattr(module, "get_current_write_origin", None) if module is not None else None
    if not callable(reader):
        try:
            from tools.skill_provenance import get_current_write_origin
        except Exception:
            return ""
        reader = get_current_write_origin
    try:
        origin = reader()
    except Exception:
        return ""
    return origin if isinstance(origin, str) else ""


def _fork_reply() -> bool:
    """True for a /btw answer, a background review, or a plugin session fork."""
    if _whitelist_is_set():
        return True
    origin = _write_origin()
    return bool(origin) and origin not in _TURN_WRITE_ORIGINS


def _pop_turn_home(turn_id: str) -> str:
    if not isinstance(turn_id, str) or not turn_id:
        return ""
    with _lock:
        home = _turn_home.pop(turn_id, "")
        if home:
            _turn_order[:] = [item for item in _turn_order if item != turn_id]
        return home


def _take_turn(session_id: str, turn_id: str) -> list[str]:
    """PNGs for this turn, home session first, then the session id on the reply."""
    paths: list[str] = []
    seen: list[str] = []
    for sid in (_pop_turn_home(turn_id), session_id):
        if sid and sid not in seen:
            seen.append(sid)
            paths.extend(take(sid))
    return paths


def _hermes_media_paths(text: str) -> set[str] | None:
    """Paths Hermes would attach, or None when that extractor cannot be loaded."""
    module = sys.modules.get("gateway.platforms.base")
    extract = None
    if module is not None:
        extract = getattr(getattr(module, "BasePlatformAdapter", None), "extract_media", None)
    if not callable(extract):
        try:
            from gateway.platforms.base import BasePlatformAdapter
        except Exception:
            return None
        extract = BasePlatformAdapter.extract_media
    try:
        found, _cleaned = extract(text or "")
    except Exception:
        return set()
    paths: set[str] = set()
    for item in found or ():
        raw = item[0] if isinstance(item, (tuple, list)) and item else item
        if isinstance(raw, str) and raw:
            paths.add(os.path.normpath(raw))
    return paths


def _names(text: str, path: str, extracted: set[str] | None) -> bool:
    """True when Hermes would attach this path. A longer file name does not count."""
    if extracted is None:
        return re.search(re.escape("MEDIA:" + path) + r"(?![\w/-]|\.\w)", text) is not None
    return os.path.normpath(path) in extracted


def append_missing_media(
    response_text: str, session_id: str = "", platform: str = "", turn_id: str = ""
) -> str | None:
    """Return the reply with a line for each PNG it left out, or None to leave it."""
    if _fork_reply():
        return None
    paths = _take_turn(session_id, turn_id)
    if not paths or not can_attach(platform):
        return None
    text = response_text or ""
    extracted = _hermes_media_paths(text)
    missing = [path for path in dict.fromkeys(paths) if not _names(text, path, extracted)]
    needs_file = sends_as_file(platform) and "[[as_document]]" not in text
    if not missing and not needs_file:
        return None
    lines = ["[[as_document]]"] if needs_file else []
    lines.extend("MEDIA:" + path for path in missing)
    return text.rstrip() + "\n\n" + "\n".join(lines)


def attach_reply(
    response_text: str = "", session_id: str = "", platform: str = "", turn_id: str = "", **_kwargs
) -> str | None:
    return append_missing_media(
        response_text or "",
        session_id or "",
        platform or "",
        turn_id if isinstance(turn_id, str) else "",
    )
