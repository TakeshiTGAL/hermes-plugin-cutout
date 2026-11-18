"""Typed tool failures. The next step is part of every refusal."""

from __future__ import annotations

NOT_IMAGE = "not_image"
TOO_LARGE = "too_large"
MODEL_FETCH = "model_fetch"
SHA256_MISMATCH = "sha256_mismatch"
EXISTS = "exists"
SYMLINK = "symlink"
OUTSIDE = "outside"
HARDLINK = "hardlink"
MISSING = "missing"
PARTIAL = "partial"
UNREADABLE = "unreadable"
APPROVAL_BLOCKED = "approval_blocked"
APPROVAL_DENIED = "approval_denied"
APPROVAL_REQUIRED = "approval_required"
APPROVAL_NOT_SENT = "approval_not_sent"
UNATTENDED = "unattended"
BAD_ARGUMENT = "bad_argument"
MODEL_RUNTIME = "model_runtime"
DEPENDENCY = "dependency"


def symlink_place(detail: str) -> tuple[str, str]:
    """Split a symlink refusal into leaf or ancestor, plus the path component."""
    role, sep, component = (detail or "").partition(":")
    if sep != ":" or role not in ("leaf", "ancestor"):
        return "ancestor", ""
    if any(mark in component for mark in ("/", "\\", "\x00", "\n", "\r")):
        return role, ""
    return role, component


def fail(kind: str, message: str, next_step: str) -> dict:
    return {
        "ok": False,
        "error_type": kind,
        "message": message,
        "next_step": next_step,
        "wrote": False,
        "isError": True,
    }
