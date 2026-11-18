"""Pin u2netp, hash it, and store it under the Hermes home.

The photo never leaves this machine. The download runs only when the pinned
file is absent. A file that fails the hash is left in place and is not loaded.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

if __package__:
    from .errors import MODEL_FETCH, SHA256_MISMATCH, SYMLINK, symlink_place
    from .paths import LocalRefused, hermes_home, read_child
    from .safe_write import WriteRefused, write_bytes
else:
    from errors import MODEL_FETCH, SHA256_MISMATCH, SYMLINK, symlink_place
    from paths import LocalRefused, hermes_home, read_child
    from safe_write import WriteRefused, write_bytes

MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_VERSION = "v0.0.0"
MODEL_SHA256 = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"
MODEL_BYTES = 4574861
MAX_MODEL_BYTES = 8 * 1024 * 1024
RELATIVE = "models/cutout/u2netp.onnx"
USER_AGENT = "hermes-plugin-cutout"
ALLOWED_HOSTS = frozenset({"github.com", "release-assets.githubusercontent.com"})
DEADLINE_S = 90
TIMEOUT_S = 60

DISK_MISMATCH_MESSAGE = (
    "The model file at HERMES_HOME/models/cutout/u2netp.onnx does not match the pinned sha256, "
    "so it was not loaded and nothing was written."
)
DISK_MISMATCH_NEXT = "Delete that file and call cutout_png again. The photo was not uploaded."
BODY_MISMATCH_MESSAGE = (
    "The download did not match the pinned sha256, so it was not saved and nothing was written."
)
BODY_MISMATCH_NEXT = (
    "Call cutout_png again. Nothing was saved, so there is no model file to delete. "
    "The photo was not uploaded."
)
FETCH_MESSAGE = (
    "The pinned model could not be downloaded, so nothing was written. The photo was not uploaded."
)
FETCH_NEXT = "Check the network and call cutout_png again. A file that fails the hash is not saved."
SYMLINK_MESSAGE = (
    "The model path is a symlink, so it was not followed and nothing was written."
)
SYMLINK_NEXT = "Remove the symlink at HERMES_HOME/models/cutout/u2netp.onnx and call again."
ANCESTOR_MESSAGE = (
    "A parent directory of the model file is a symlink, so it was not followed and nothing was written."
)
ANCESTOR_NEXT = "Point HERMES_HOME at a real directory and call again."


class ModelProblem(Exception):
    def __init__(self, kind: str, message: str, next_step: str):
        self.kind = kind
        self.message = message
        self.next_step = next_step
        super().__init__(message)


class _Hop(Exception):
    def __init__(self, url: str):
        self.url = url


class _StopRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _Hop(newurl)


def _opener():
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _StopRedirect(),
    )


def _allowed(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        return False
    host = (parsed.hostname or "").lower()
    return host in ALLOWED_HOSTS


def _matches(data: bytes) -> bool:
    if len(data) != MODEL_BYTES:
        return False
    digest = hashlib.sha256(data).hexdigest()
    return hmac.compare_digest(digest, MODEL_SHA256)


def _mismatch(on_disk: bool) -> ModelProblem:
    """Saved bytes ask for a delete. A download that was not saved does not."""
    if on_disk:
        return ModelProblem(SHA256_MISMATCH, DISK_MISMATCH_MESSAGE, DISK_MISMATCH_NEXT)
    return ModelProblem(SHA256_MISMATCH, BODY_MISMATCH_MESSAGE, BODY_MISMATCH_NEXT)


def fetch_pinned(opener_factory=None) -> bytes:
    """Download the pinned URL. One HTTPS hop, and only to the two allowed hosts."""
    factory = opener_factory or _opener
    url = MODEL_URL
    deadline = time.monotonic() + DEADLINE_S
    hops = 0
    while True:
        if time.monotonic() > deadline:
            raise ModelProblem(MODEL_FETCH, FETCH_MESSAGE, FETCH_NEXT)
        if hops == 0 and url != MODEL_URL:
            raise ModelProblem(MODEL_FETCH, FETCH_MESSAGE, FETCH_NEXT)
        if not _allowed(url):
            raise ModelProblem(MODEL_FETCH, FETCH_MESSAGE, FETCH_NEXT)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ModelProblem(MODEL_FETCH, FETCH_MESSAGE, FETCH_NEXT)
        request = urllib.request.Request(
            url,
            method="GET",
            headers={"User-Agent": USER_AGENT},
        )
        opener = factory()
        try:
            with opener.open(request, timeout=min(TIMEOUT_S, remaining)) as response:
                length = response.headers.get("Content-Length")
                if length is not None:
                    try:
                        announced = int(length)
                    except ValueError:
                        announced = MAX_MODEL_BYTES + 1
                    if announced > MAX_MODEL_BYTES:
                        raise ModelProblem(MODEL_FETCH, FETCH_MESSAGE, FETCH_NEXT)
                data = response.read(MAX_MODEL_BYTES + 1)
        except ModelProblem:
            raise
        except _Hop as hop:
            hops += 1
            if hops > 1:
                raise ModelProblem(MODEL_FETCH, FETCH_MESSAGE, FETCH_NEXT) from hop
            url = hop.url
            continue
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise ModelProblem(MODEL_FETCH, FETCH_MESSAGE, FETCH_NEXT) from exc
        if len(data) > MAX_MODEL_BYTES:
            raise ModelProblem(MODEL_FETCH, FETCH_MESSAGE, FETCH_NEXT)
        return data


def _model_symlink(detail: str) -> ModelProblem:
    role, _component = symlink_place(detail)
    if role == "ancestor":
        return ModelProblem(SYMLINK, ANCESTOR_MESSAGE, ANCESTOR_NEXT)
    return ModelProblem(SYMLINK, SYMLINK_MESSAGE, SYMLINK_NEXT)


def _model_path(home: Path) -> tuple[Path, str]:
    path = home / "models" / "cutout" / "u2netp.onnx"
    return path, path.name


def _read_existing(home: Path) -> bytes | None:
    directory = home / "models" / "cutout"
    try:
        return read_child(directory, "u2netp.onnx", MAX_MODEL_BYTES)
    except LocalRefused as exc:
        if exc.kind == "missing":
            return None
        if exc.kind == "symlink":
            problem = _model_symlink(exc.detail)
            raise ModelProblem(problem.kind, problem.message, problem.next_step) from exc
        if exc.kind == "too_large":
            raise _mismatch(True) from exc
        raise ModelProblem(MODEL_FETCH, FETCH_MESSAGE, FETCH_NEXT) from exc


def ensure_model() -> bytes:
    """Return the pinned bytes. Re-hash every call. Do not replace a mismatch."""
    home = hermes_home()
    if not home.is_absolute():
        raise ModelProblem(MODEL_FETCH, FETCH_MESSAGE, FETCH_NEXT)
    existing = _read_existing(home)
    if existing is not None:
        if not _matches(existing):
            raise _mismatch(True)
        return existing
    body = fetch_pinned()
    if not _matches(body):
        raise _mismatch(False)
    try:
        write_bytes(home, RELATIVE, body, create=True, exclusive=True)
    except WriteRefused as exc:
        if exc.kind != "exists":
            if exc.kind == "symlink":
                problem = _model_symlink(exc.detail)
                raise ModelProblem(problem.kind, problem.message, problem.next_step) from exc
            raise ModelProblem(MODEL_FETCH, FETCH_MESSAGE, FETCH_NEXT) from exc
        raced = _read_existing(home)
        if raced is not None and _matches(raced):
            return raced
        raise _mismatch(raced is not None) from exc
    stored = _read_existing(home)
    if stored is not None and _matches(stored):
        return stored
    raise _mismatch(stored is not None)
