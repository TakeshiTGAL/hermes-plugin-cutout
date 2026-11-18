"""Read a local file without following a symlink."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

if __package__:
    from .safe_write import WriteRefused, _close_all, _open_dir
else:
    from safe_write import WriteRefused, _close_all, _open_dir


class LocalRefused(Exception):
    def __init__(self, kind: str, detail: str):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}")


def hermes_home() -> Path:
    getter = None
    try:
        from hermes_constants import get_hermes_home
    except Exception:
        get_hermes_home = None
    getter = get_hermes_home
    if getter is not None:
        try:
            return Path(getter())
        except Exception:
            pass
    raw = os.environ.get("HERMES_HOME", "").strip()
    if raw:
        return Path(raw)
    return Path.home() / ".hermes"


def _map_open(exc: WriteRefused) -> LocalRefused:
    return LocalRefused(exc.kind, exc.detail)


def lstat_child(directory: Path, name: str):
    """lstat of one child. None when the name is absent. Symlinks are not followed."""
    try:
        parent_fd, fds = _open_dir(directory, create=False)
    except WriteRefused as exc:
        raise _map_open(exc) from exc
    try:
        try:
            return os.lstat(name, dir_fd=parent_fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise LocalRefused("unreadable", name) from exc
    finally:
        _close_all(fds)


def read_child(directory: Path, name: str, limit: int) -> bytes:
    """Read one regular file. Refuse a symlink, a huge file, and a short read past the cap."""
    try:
        parent_fd, fds = _open_dir(directory, create=False)
    except WriteRefused as exc:
        if exc.kind == "symlink":
            _role, _, component = (exc.detail or "").partition(":")
            raise LocalRefused("symlink", "ancestor:" + component) from exc
        raise _map_open(exc) from exc
    fd = -1
    try:
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            fd = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK, errno.ENOTDIR):
                raise LocalRefused("symlink", "leaf:" + name) from exc
            if exc.errno == errno.ENOENT:
                raise LocalRefused("missing", name) from exc
            raise LocalRefused("unreadable", name) from exc
        info = os.fstat(fd)
        if stat.S_ISLNK(info.st_mode):
            raise LocalRefused("symlink", "leaf:" + name)
        if not stat.S_ISREG(info.st_mode):
            raise LocalRefused("not_image", name)
        if info.st_size > limit:
            raise LocalRefused("too_large", name)
        chunks: list[bytes] = []
        left = limit + 1
        while left > 0:
            block = os.read(fd, min(left, 1024 * 1024))
            if not block:
                break
            chunks.append(block)
            left -= len(block)
        data = b"".join(chunks)
        if len(data) > limit:
            raise LocalRefused("too_large", name)
        return data
    finally:
        if fd >= 0:
            os.close(fd)
        _close_all(fds)
