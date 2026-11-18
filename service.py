"""Plan a cutout, ask through pre_tool_call, then write one new PNG."""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import unicodedata
from pathlib import Path

if __package__:
    from . import delivery, engine, gates
    from .errors import (
        APPROVAL_BLOCKED,
        APPROVAL_DENIED,
        APPROVAL_NOT_SENT,
        APPROVAL_REQUIRED,
        BAD_ARGUMENT,
        DEPENDENCY,
        EXISTS,
        HARDLINK,
        MISSING,
        MODEL_RUNTIME,
        NOT_IMAGE,
        OUTSIDE,
        PARTIAL,
        SYMLINK,
        TOO_LARGE,
        UNATTENDED,
        UNREADABLE,
        fail,
        symlink_place,
    )
    from .model_store import MODEL_SHA256, MODEL_VERSION, ModelProblem, ensure_model
    from .paths import LocalRefused, lstat_child, read_child
    from .safe_write import WriteRefused, write_bytes
else:
    import delivery
    import engine
    import gates
    from errors import (
        APPROVAL_BLOCKED,
        APPROVAL_DENIED,
        APPROVAL_NOT_SENT,
        APPROVAL_REQUIRED,
        BAD_ARGUMENT,
        DEPENDENCY,
        EXISTS,
        HARDLINK,
        MISSING,
        MODEL_RUNTIME,
        NOT_IMAGE,
        OUTSIDE,
        PARTIAL,
        SYMLINK,
        TOO_LARGE,
        UNATTENDED,
        UNREADABLE,
        fail,
        symlink_place,
    )
    from model_store import MODEL_SHA256, MODEL_VERSION, ModelProblem, ensure_model
    from paths import LocalRefused, lstat_child, read_child
    from safe_write import WriteRefused, write_bytes

MAX_FILE_BYTES = 33554432
MAX_PIXELS = 24000000
_ACCEPTED = ("image", "folder", "name")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,60}\.png$")
_WRITE_KIND = {
    "exists": EXISTS,
    "symlink": SYMLINK,
    "outside": OUTSIDE,
    "hardlink": HARDLINK,
    "partial": PARTIAL,
    "missing": MISSING,
}


def accepted_args(args: dict) -> dict:
    kept = {"image": args["image"], "folder": args["folder"]}
    if "name" in args:
        kept["name"] = args["name"]
    return kept


def describe(image: str, dest: str) -> str:
    return (
        "cutout_png will read "
        + image
        + " again and write one new PNG at "
        + dest
        + ". An existing file is not overwritten. The local u2netp file is checked with sha256 "
        + MODEL_SHA256
        + ". The pinned file is downloaded only when HERMES_HOME/models/cutout/u2netp.onnx is absent. "
        + "The photo is not uploaded."
    )


def _bad_text(value: object) -> bool:
    if not isinstance(value, str) or value == "" or value != value.strip():
        return True
    if "\\" in value or value.endswith("/"):
        return True
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in value)


def _absolute(value: str) -> Path | None:
    path = Path(value)
    if not path.is_absolute():
        return None
    if any(part in ("", ".", "..") for part in path.parts):
        return None
    return path


def default_name(image: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", image.stem).strip(".-_")[:40]
    if not cleaned or not cleaned[0].isalnum():
        cleaned = "photo"
    name = cleaned + "-cutout.png"
    if ".." in name or _NAME.fullmatch(name) is None:
        return "photo-cutout.png"
    return name


def _load_frame(data: bytes):
    try:
        from PIL import Image
        from PIL import UnidentifiedImageError
    except Exception as exc:
        raise LocalRefused("dependency", "pillow") from exc
    old = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    bomb = Image.DecompressionBombError
    try:
        try:
            image = Image.open(io.BytesIO(data))
        except bomb as exc:
            raise LocalRefused("too_large", "pixels") from exc
        except (UnidentifiedImageError, OSError) as exc:
            raise LocalRefused("not_image", "bytes") from exc
        try:
            frames = int(getattr(image, "n_frames", 1) or 1)
            image.seek(0)
            frame = image.convert("RGB")
            width, height = frame.size
        except bomb as exc:
            raise LocalRefused("too_large", "pixels") from exc
        finally:
            image.close()
    finally:
        Image.MAX_IMAGE_PIXELS = old
    if width < 1 or height < 1:
        raise LocalRefused("not_image", "pixels")
    if width * height > MAX_PIXELS:
        raise LocalRefused("too_large", "pixels")
    return {"image": frame, "frames": frames, "width": width, "height": height}


_NAME_LIMIT = 80
_HIDDEN_NAME = "that directory"


def _name_row(component: str) -> str:
    """show or hide. The rows are the kinds of one path component."""
    if not isinstance(component, str):
        return "hide"
    if component.strip() in ("", ".", ".."):
        return "hide"
    if len(component) > _NAME_LIMIT:
        return "hide"
    for ch in component:
        if ch in "/\\`":
            return "hide"
        category = unicodedata.category(ch)
        if category[0] == "C" or category in ("Zl", "Zp"):
            return "hide"
    return "show"


def _shown_component(component: str) -> str:
    """Name for the symlink reply.

    Letters outside ASCII are shown. A control character, a format character,
    a backtick, a slash, a blank name, or a name over 80 characters would
    break or spoof the sentence, so those stay hidden.
    """
    if _name_row(component) != "show":
        return _HIDDEN_NAME
    return component


def _symlink_fail(scope: str, detail: str) -> dict:
    role, component = symlink_place(detail)
    if scope == "photo" and role == "leaf":
        return fail(
            SYMLINK,
            "The photo is a symlink, so it was not followed and nothing was written.",
            "Pass the real photo file.",
        )
    if scope == "folder" and role == "leaf":
        return fail(
            SYMLINK,
            "The folder is a symlink, so it was not followed and nothing was written.",
            "Pass a real folder.",
        )
    return fail(
        SYMLINK,
        "A parent directory on the path is a symlink, so nothing was written.",
        "Pass a real directory in place of " + _shown_component(component) + ".",
    )


def _refuse_read(exc: LocalRefused) -> dict:
    if exc.kind == "too_large":
        return fail(
            TOO_LARGE,
            "The photo is over the limit of 33554432 bytes or 24000000 pixels, so nothing was written.",
            "Pass a smaller photo.",
        )
    if exc.kind == "not_image":
        return fail(
            NOT_IMAGE,
            "The file is not an image, so nothing was written.",
            "Pass a photo file.",
        )
    if exc.kind == "missing":
        return fail(
            MISSING,
            "The photo was not found, so nothing was written.",
            "Pass the absolute path of a photo that exists.",
        )
    if exc.kind == "symlink":
        return _symlink_fail("photo", exc.detail)
    if exc.kind == "dependency":
        return fail(
            DEPENDENCY,
            "The image library is not installed, so nothing was written.",
            "Install the plugin dependencies and call cutout_png again.",
        )
    if exc.kind == "unreadable":
        return fail(
            UNREADABLE,
            "The photo could not be read, so nothing was written.",
            "Check the file permissions and call again.",
        )
    return fail(
        OUTSIDE,
        "The path is outside the folder you named, so nothing was written.",
        "Pass an absolute folder and a single PNG file name.",
    )


def plan_write(args: dict) -> dict:
    if not isinstance(args, dict):
        return fail(BAD_ARGUMENT, "The arguments are not an object, so nothing was written.", "Pass image and folder.")
    unknown = [key for key in args if key not in _ACCEPTED]
    if unknown:
        return fail(
            BAD_ARGUMENT,
            "An unknown argument was passed, so nothing was written.",
            "Pass image, folder, and an optional png name.",
        )
    if "image" not in args or "folder" not in args:
        return fail(
            BAD_ARGUMENT,
            "image and folder are required, so nothing was written.",
            "Pass both absolute paths.",
        )
    image_text = args["image"]
    folder_text = args["folder"]
    if _bad_text(image_text) or _bad_text(folder_text):
        return fail(
            BAD_ARGUMENT,
            "A path is empty, relative, or has odd characters, so nothing was written.",
            "Pass absolute paths with no spaces around them.",
        )
    image = _absolute(image_text)
    folder = _absolute(folder_text)
    if image is None or folder is None:
        return fail(
            OUTSIDE,
            "The path is not absolute, so nothing was written.",
            "Pass an absolute photo path and an absolute folder.",
        )
    if "name" in args:
        name = args["name"]
        if _bad_text(name) or ".." in name or _NAME.fullmatch(name) is None:
            return fail(
                BAD_ARGUMENT,
                "The PNG name is not a single safe file name, so nothing was written.",
                "Use letters, digits, dot, underscore, or hyphen, and end with .png.",
            )
    else:
        name = default_name(image)
    try:
        info = lstat_child(folder, name)
    except LocalRefused as exc:
        if exc.kind == "missing":
            return fail(
                MISSING,
                "The folder does not exist, so nothing was written.",
                "Create the folder and call again. The plugin does not create it.",
            )
        if exc.kind == "symlink":
            return _symlink_fail("folder", exc.detail)
        return _refuse_read(exc)
    if info is not None:
        if stat.S_ISLNK(info.st_mode):
            return fail(
                SYMLINK,
                "The destination is a symlink, so nothing was written.",
                "Choose a new PNG name that is not a symlink.",
            )
        if stat.S_ISREG(info.st_mode) and info.st_nlink >= 2:
            return fail(
                HARDLINK,
                "The destination has another link, so nothing was written.",
                "Choose a new name that is not a hard link.",
            )
        return fail(
            EXISTS,
            "That PNG name already exists, so nothing was written.",
            "Choose a new name. An existing file is not overwritten.",
        )
    try:
        data = read_child(image.parent, image.name, MAX_FILE_BYTES)
        frame = _load_frame(data)
    except LocalRefused as exc:
        return _refuse_read(exc)
    dest = str(folder / name)
    return {
        "ok": True,
        "image": str(image),
        "folder": str(folder),
        "name": name,
        "dest": dest,
        "width": frame["width"],
        "height": frame["height"],
        "frame_count": frame["frames"],
        "line": describe(str(image), dest),
    }


def _alpha_counts(data: bytes) -> tuple[int, int, int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        hist = image.getchannel("A").histogram()
    return width, height, hist[0], hist[255]


def _finish(folder: str, name: str, rgba, frame_count: int) -> dict:
    buffer = io.BytesIO()
    rgba.save(buffer, format="PNG")
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    try:
        write_bytes(Path(folder), name, data, create=False, exclusive=True)
    except WriteRefused as exc:
        kind = _WRITE_KIND.get(exc.kind, PARTIAL)
        return fail(
            kind,
            "The new PNG was not written.",
            "Choose a new name in a real folder and call again.",
        )
    try:
        stored = read_child(Path(folder), name, max(len(data), 1) + 1)
    except LocalRefused as exc:
        return fail(
            PARTIAL,
            "The new PNG could not be read back, so this result is not a success.",
            "Delete that name if you do not want it, and call again.",
        )
    if hashlib.sha256(stored).hexdigest() != digest or len(stored) != len(data):
        return fail(
            PARTIAL,
            "The bytes read back did not match the PNG just written, so this result is not a success.",
            "Delete that name if you do not want it, and call again.",
        )
    width, height, alpha_zero, alpha_full = _alpha_counts(stored)
    dest = str(Path(folder) / name)
    return {
        "ok": True,
        "wrote": True,
        "isError": False,
        "path": dest,
        "width": width,
        "height": height,
        "bytes": len(stored),
        "sha256": digest,
        "alpha_zero": alpha_zero,
        "alpha_full": alpha_full,
        "model": "u2netp",
        "model_version": MODEL_VERSION,
        "model_sha256": MODEL_SHA256,
        "frames_used": 1,
        "frame_count": frame_count,
        "message": (
            "Wrote one new transparent PNG at "
            + dest
            + ". alpha_zero and alpha_full count pixels in that file."
        ),
    }


def invoke(args: dict, session_id: str = "", task_id: str = "") -> dict:
    if gates.unattended():
        gates.clear_room(session_id, task_id)
        return fail(
            UNATTENDED,
            gates.UNATTENDED_MESSAGE,
            "Run cutout_png from a chat you can answer.",
        )
    if gates.in_plugin_host_process():
        gates.clear_room(session_id, task_id)
        return fail(
            APPROVAL_BLOCKED,
            gates.HOST_BLOCK,
            "Run the plugin outside plugins.isolation host.",
        )
    plan = plan_write(args)
    if plan.get("ok") is not True:
        return plan
    status, detail = gates.consume(accepted_args(args), session_id, task_id, plan["line"])
    if status != "approved":
        text = detail or gates.DENIED
        return fail(APPROVAL_DENIED, text, "Call cutout_png again and approve that card.")
    if gates.unattended() or gates.in_plugin_host_process():
        return fail(
            UNATTENDED,
            gates.UNATTENDED_MESSAGE,
            "Run cutout_png from a chat you can answer.",
        )
    try:
        model = ensure_model()
    except ModelProblem as exc:
        return fail(exc.kind, exc.message, exc.next_step)
    try:
        data = read_child(Path(plan["image"]).parent, Path(plan["image"]).name, MAX_FILE_BYTES)
        frame = _load_frame(data)
    except LocalRefused as exc:
        return _refuse_read(exc)
    try:
        rgba = engine.cut_rgba(frame["image"], model)
    except engine.EngineError:
        return fail(
            MODEL_RUNTIME,
            "The local model did not run, so nothing was written. The photo was not uploaded.",
            "Call again with the pinned u2netp file. Another model is not selected.",
        )
    except Exception:
        return fail(
            MODEL_RUNTIME,
            "The local model did not run, so nothing was written. The photo was not uploaded.",
            "Call again with the pinned u2netp file. Another model is not selected.",
        )
    if rgba.size != (frame["width"], frame["height"]):
        return fail(
            MODEL_RUNTIME,
            "The result size did not match the stored photo, so nothing was written.",
            "Call again. Resize belongs to another plugin.",
        )
    return _finish(plan["folder"], plan["name"], rgba, frame["frames"])


def on_pre_tool_call(tool_name="", args=None, session_id="", task_id="", **_kwargs):
    """Approve cutout_png only when this call would write a new PNG."""
    if tool_name != "cutout_png":
        return None
    gates.clear_room(session_id or "", task_id or "")
    if not isinstance(args, dict):
        return None
    plan = plan_write(args)
    if plan.get("ok") is not True:
        return None
    if gates.unattended():
        return {"action": "block", "message": gates.UNATTENDED_MESSAGE}
    if gates.in_plugin_host_process():
        return {"action": "block", "message": gates.HOST_BLOCK}
    line = plan["line"]
    if not gates.line_is_sendable(line):
        return {"action": "block", "message": gates.NOT_SENT}
    kept = accepted_args(args)
    gates.remember(kept, session_id or "", task_id or "", line)
    return {"action": "approve", "message": line, "rule_key": gates.rule_key(kept)}


def cli_preview(image: str, folder: str, name: str | None) -> dict:
    """Show the plan and do not write. A chat approval of cutout_png writes."""
    args = {"image": image, "folder": folder}
    if name is not None:
        args["name"] = name
    directive = on_pre_tool_call("cutout_png", args, session_id="cli", task_id="cli")
    gates.clear_room("cli", "cli")
    if directive is None:
        return plan_write(args)
    if gates.unattended():
        return fail(
            UNATTENDED,
            gates.UNATTENDED_MESSAGE,
            "Run cutout_png from a chat you can answer.",
        )
    if gates.in_plugin_host_process():
        return fail(
            APPROVAL_BLOCKED,
            gates.HOST_BLOCK,
            "Run the plugin outside plugins.isolation host.",
        )
    if directive.get("action") == "block":
        return fail(
            APPROVAL_NOT_SENT,
            gates.NOT_SENT,
            "Use a shorter folder path and call again.",
        )
    return fail(
        APPROVAL_REQUIRED,
        "This command prints the plan and does not write.",
        "Approve cutout_png in chat to write one new PNG.",
    )


def _chat_platform() -> str:
    """The gateway platform, else a Desktop source this task bound.

    An inherited ``HERMES_SESSION_SOURCE`` is not a Desktop session. When the
    gateway module is absent, the process environment is the only signal.
    """
    try:
        platform = gates.session_value("HERMES_SESSION_PLATFORM")
        if platform.strip():
            return platform
        source = gates.session_source().strip().lower()
    except gates.SessionUnreadable:
        return ""
    return source if source in delivery.SOURCE_CHATS else ""


def _with_delivery(payload: dict, session_id: str) -> dict:
    """Remember a written PNG. Add a media line only on a chat that can attach."""
    if payload.get("ok") is not True or payload.get("wrote") is not True:
        return payload
    path = payload.get("path")
    if not isinstance(path, str) or path == "":
        return payload
    delivery.remember(session_id, path)
    platform = _chat_platform()
    if not delivery.can_attach(platform):
        return payload
    noted = dict(payload)
    noted["media"] = delivery.media_block(path, platform)
    if delivery.sends_as_file(platform):
        note = (
            " The media line sends that PNG as a file. A photo send of a large image "
            "would fill transparent pixels with white."
        )
    else:
        note = " Put the media line on its own line in the reply so the chat can show the PNG."
    noted["message"] = payload["message"] + note
    return noted


def cutout_png(args, session_id="", task_id="", **_kwargs) -> str:
    try:
        if not isinstance(args, dict):
            payload = fail(
                BAD_ARGUMENT,
                "The arguments are not an object, so nothing was written.",
                "Pass image and folder.",
            )
        else:
            payload = invoke(args, session_id=session_id or "", task_id=task_id or "")
    except Exception:
        payload = fail(
            UNREADABLE,
            "The tool hit an unexpected error, so nothing was written.",
            "Try the call again.",
        )
    else:
        payload = _with_delivery(payload, session_id or "")
    return json.dumps(payload, ensure_ascii=False)
