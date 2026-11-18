"""The PNG reaches a chat as a MEDIA line, and nowhere else."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import delivery
import engine
import gates
import service
from pngutil import png

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def room(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_PLUGIN_HOST_PROCESS", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    folder = tmp_path / "out"
    folder.mkdir()
    image = tmp_path / "photo.png"
    image.write_bytes(png(4, 4))
    gates.clear()
    gates.remember_context(None)
    yield {"home": home, "image": image, "folder": folder}
    gates.clear()
    gates.remember_context(None)


def _paint(image, _model):
    rgba = image.convert("RGBA")
    pixels = rgba.load()
    width, height = rgba.size
    for y in range(height):
        for x in range(width):
            red, green, blue, _alpha = pixels[x, y]
            alpha = 0 if x == 0 or y == 0 else 255
            pixels[x, y] = (red, green, blue, alpha)
    return rgba


def _files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


def _succeed(room, monkeypatch, platform: str) -> tuple[str, dict]:
    pytest.importorskip("PIL")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
    monkeypatch.setattr(service, "ensure_model", lambda: b"pinned")
    monkeypatch.setattr(engine, "cut_rgba", _paint)
    service.on_pre_tool_call("cutout_png", {"image": str(room["image"]), "folder": str(room["folder"])}, "s", "t")
    raw = service.cutout_png(
        {"image": str(room["image"]), "folder": str(room["folder"])},
        session_id="s",
        task_id="t",
    )
    return raw, json.loads(raw)


def test_desktop_success_is_one_media_line(room, monkeypatch):
    raw, body = _succeed(room, monkeypatch, "desktop")
    path = body["path"]
    assert body["wrote"] is True
    assert body["media"] == "MEDIA:" + path
    assert "[[as_document]]" not in raw
    reply = delivery.attach_reply(response_text="Done.", session_id="s", platform="desktop")
    assert reply == "Done.\n\nMEDIA:" + path
    assert delivery.attach_reply(response_text="Next.", session_id="s", platform="desktop") is None


def test_discord_is_an_image_line(room, monkeypatch):
    _raw, body = _succeed(room, monkeypatch, "discord")
    assert body["media"] == "MEDIA:" + body["path"]
    assert body["media"].startswith("MEDIA:")


def test_telegram_sends_the_png_as_a_file(room, monkeypatch):
    raw, body = _succeed(room, monkeypatch, "telegram")
    path = body["path"]
    assert body["media"] == "[[as_document]]\nMEDIA:" + path
    assert "sends that PNG as a file" in body["message"]
    reply = delivery.attach_reply(response_text="Done.", session_id="s", platform="telegram")
    assert reply == "Done.\n\n[[as_document]]\nMEDIA:" + path
    already = delivery.remember("s2", path)
    assert already is None
    delivery.remember("s2", path)
    assert delivery.attach_reply(response_text="[[as_document]]\nMEDIA:" + path, session_id="s2", platform="telegram") is None


def test_telegram_adds_the_file_marker_when_the_reply_has_only_the_path(room, monkeypatch):
    _raw, body = _succeed(room, monkeypatch, "telegram")
    path = body["path"]
    reply = delivery.attach_reply(response_text="See MEDIA:" + path, session_id="s", platform="telegram")
    assert reply == "See MEDIA:" + path + "\n\n[[as_document]]"
    assert reply.count("MEDIA:" + path) == 1


def test_hook_leaves_a_reply_that_already_has_the_line(room, monkeypatch):
    _raw, body = _succeed(room, monkeypatch, "slack")
    path = body["path"]
    assert delivery.attach_reply(response_text="here\nMEDIA:" + path, session_id="s", platform="slack") is None


def test_terminal_returns_the_path_and_does_not_pretend(room, monkeypatch):
    pytest.importorskip("PIL")
    for platform in ("cli", "tui", ""):
        delivery.clear()
        if platform:
            monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
        else:
            monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        monkeypatch.setattr(service, "ensure_model", lambda: b"pinned")
        monkeypatch.setattr(engine, "cut_rgba", _paint)
        name = f"cut-{platform or 'empty'}.png"
        args = {"image": str(room["image"]), "folder": str(room["folder"]), "name": name}
        service.on_pre_tool_call("cutout_png", args, "s", "t")
        raw = service.cutout_png(args, session_id="s", task_id="t")
        body = json.loads(raw)
        assert body["wrote"] is True
        assert body["path"].endswith(name)
        assert "media" not in body
        assert "MEDIA:" not in raw
        assert "[[as_document]]" not in raw
        assert delivery.attach_reply(response_text="wrote the file", session_id="s", platform=platform) is None


def test_failure_does_not_emit_a_media_line(room, monkeypatch):
    pytest.importorskip("PIL")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    room["image"].write_bytes(b"this is not a photo")
    service.on_pre_tool_call(
        "cutout_png",
        {"image": str(room["image"]), "folder": str(room["folder"])},
        "s",
        "t",
    )
    raw = service.cutout_png(
        {"image": str(room["image"]), "folder": str(room["folder"])},
        session_id="s",
        task_id="t",
    )
    body = json.loads(raw)
    assert body["ok"] is False
    assert "MEDIA:" not in raw
    assert "[[as_document]]" not in raw
    assert delivery.attach_reply(response_text="no", session_id="s", platform="telegram") is None


def test_denied_call_does_not_emit_a_media_line(room, monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "desktop")
    raw = service.cutout_png(
        {"image": str(room["image"]), "folder": str(room["folder"])},
        session_id="s",
        task_id="t",
    )
    assert "MEDIA:" not in raw
    assert delivery.attach_reply(response_text="no", session_id="s", platform="desktop") is None


def test_success_does_not_write_outside_the_folder(room, monkeypatch, tmp_path):
    before = _files(tmp_path)
    raw, body = _succeed(room, monkeypatch, "telegram")
    after = _files(tmp_path)
    dest = Path(body["path"])
    assert dest.parent == room["folder"]
    assert dest.is_file()
    assert "MEDIA:" + str(dest) in raw
    added = after - before
    assert added == {dest.relative_to(tmp_path)}
    assert not (room["home"] / "cache").exists()


def test_register_adds_the_attach_hook():
    namespace = {
        "__package__": "",
        "__name__": "cutout_register_delivery",
        "__file__": str(ROOT / "__init__.py"),
    }
    exec(compile((ROOT / "__init__.py").read_text(encoding="utf-8"), str(ROOT / "__init__.py"), "exec"), namespace)

    class Ctx:
        def __init__(self):
            self.hooks = []

        def register_tool(self, **_kwargs):
            return None

        def register_skill(self, *_args, **_kwargs):
            return None

        def register_hook(self, name, callback):
            self.hooks.append((name, callback))

        def register_cli_command(self, **_kwargs):
            return None

    ctx = Ctx()
    namespace["register"](ctx)
    names = [name for name, _callback in ctx.hooks]
    assert names == ["pre_tool_call", "pre_llm_call", "transform_llm_output"]
    hook = dict(ctx.hooks)["transform_llm_output"]
    delivery.remember("reg", "/tmp/cutout-reg.png")
    assert hook(response_text="ok", session_id="reg", platform="desktop") == "ok\n\nMEDIA:/tmp/cutout-reg.png"


def _write_with_source(room, monkeypatch, source: str, name: str) -> dict:
    pytest.importorskip("PIL")
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setenv("HERMES_SESSION_SOURCE", source)
    monkeypatch.setattr(service, "ensure_model", lambda: b"pinned")
    monkeypatch.setattr(engine, "cut_rgba", _paint)
    args = {"image": str(room["image"]), "folder": str(room["folder"]), "name": name}
    service.on_pre_tool_call("cutout_png", args, "s", "t")
    return json.loads(service.cutout_png(args, session_id="s", task_id="t"))


def test_desktop_named_only_by_the_session_source_gets_the_media_line(room, monkeypatch):
    import gateway.session_context as session_context

    session_context.reset_session_vars()
    session_context.set_session_vars(source="desktop")
    try:
        body = _write_with_source(room, monkeypatch, "desktop", "desk.png")
        assert body["wrote"] is True
        assert body["media"] == "MEDIA:" + body["path"]
    finally:
        session_context.reset_session_vars()


@pytest.mark.parametrize("source", ["cli", "tui", "oneshot", "acp", "tool", "kanban", "cron", "a2a", "webhook"])
def test_any_other_session_source_gets_no_media_line(room, monkeypatch, source):
    body = _write_with_source(room, monkeypatch, source, f"{source}.png")
    assert body["wrote"] is True
    assert "media" not in body


def test_the_session_platform_wins_over_the_session_source(room, monkeypatch):
    pytest.importorskip("PIL")
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "cli")
    assert service._chat_platform() == "cli"


@pytest.mark.parametrize("platform", ["api_server", "webhook", "msgraph_webhook"])
def test_a_desktop_source_does_not_open_an_unattended_platform(room, monkeypatch, platform):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
    args = {"image": str(room["image"]), "folder": str(room["folder"])}
    directive = service.on_pre_tool_call("cutout_png", args, "s", "t")
    assert directive == {"action": "block", "message": gates.UNATTENDED_MESSAGE}
    assert list(room["folder"].iterdir()) == []


@pytest.mark.parametrize("name", ["README.md", "CHANGELOG.md"])
def test_public_text_describes_the_unattended_refusal_as_it_happens(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert "error `unattended`" not in text
    assert "refused before a write, so nothing is written and no path is returned" in text


def test_a_new_turn_forgets_a_png_the_last_turn_did_not_deliver():
    delivery.remember("s", "/p/old.png")
    delivery.remember("other", "/p/kept.png")
    assert delivery.start_turn(session_id="s", turn_id="t2") is None
    assert delivery.attach_reply(response_text="Second turn.", session_id="s", platform="discord") is None
    kept = delivery.attach_reply(response_text="Hi.", session_id="other", platform="discord")
    assert kept == "Hi.\n\nMEDIA:/p/kept.png"


def test_manifest_declares_every_hook_the_plugin_registers():
    text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    for hook in ("pre_tool_call", "pre_llm_call", "transform_llm_output"):
        assert f"  - {hook}\n" in text


def test_reply_that_names_one_of_two_pngs_gains_the_other():
    delivery.remember("s", "/p/a.png")
    delivery.remember("s", "/p/b.png")
    reply = delivery.attach_reply(response_text="Done\nMEDIA:/p/a.png", session_id="s", platform="discord")
    assert reply == "Done\nMEDIA:/p/a.png\n\nMEDIA:/p/b.png"


def test_telegram_reply_that_names_one_of_two_pngs_gains_the_other_and_the_marker():
    delivery.remember("s", "/p/a.png")
    delivery.remember("s", "/p/b.png")
    reply = delivery.attach_reply(response_text="MEDIA:/p/a.png", session_id="s", platform="telegram")
    assert reply == "MEDIA:/p/a.png\n\n[[as_document]]\nMEDIA:/p/b.png"


def test_reply_that_names_every_png_is_left_alone():
    delivery.remember("s", "/p/a.png")
    delivery.remember("s", "/p/b.png")
    text = "MEDIA:/p/a.png\nMEDIA:/p/b.png"
    assert delivery.attach_reply(response_text=text, session_id="s", platform="discord") is None


def test_a_longer_path_in_the_reply_does_not_count_as_the_shorter_one():
    delivery.remember("s", "/p/a.png")
    text = "MEDIA:/p/a.png.png"
    reply = delivery.attach_reply(response_text=text, session_id="s", platform="discord")
    assert reply == text + "\n\nMEDIA:/p/a.png"


def test_a_sentence_end_after_the_path_still_counts_as_named():
    delivery.remember("s", "/p/a.png")
    assert delivery.attach_reply(response_text="See MEDIA:/p/a.png.", session_id="s", platform="discord") is None


def test_every_png_of_a_long_turn_is_attached():
    paths = [f"/p/{index}.png" for index in range(12)]
    for path in paths:
        delivery.remember("s", path)
    reply = delivery.attach_reply(response_text="Done.", session_id="s", platform="discord")
    assert reply == "Done.\n\n" + "\n".join("MEDIA:" + path for path in paths)


def _hold_fork_whitelist(allowed: set[str]):
    from hermes_cli.plugins import clear_thread_tool_whitelist, set_thread_tool_whitelist

    set_thread_tool_whitelist(
        allowed,
        deny_msg_fmt="Side question (/btw) denied tool call: {tool_name}.",
    )
    return clear_thread_tool_whitelist


def test_a_btw_answer_does_not_take_the_turn_png():
    delivery.remember("s", "/p/a.png")
    release = _hold_fork_whitelist(set())
    try:
        side = delivery.attach_reply(response_text="side answer.", session_id="s", platform="discord", turn_id="t")
    finally:
        release()
    assert side is None
    reply = delivery.attach_reply(response_text="Main reply.", session_id="s", platform="discord", turn_id="t")
    assert reply == "Main reply.\n\nMEDIA:/p/a.png"


def test_a_btw_answer_does_not_gain_a_line_the_reply_already_wrote():
    delivery.remember("s", "/p/a.png")
    release = _hold_fork_whitelist(set())
    try:
        side = delivery.attach_reply(response_text="side answer.", session_id="s", platform="discord", turn_id="t")
    finally:
        release()
    assert side is None
    assert delivery.attach_reply(
        response_text="Main reply.\nMEDIA:/p/a.png", session_id="s", platform="discord", turn_id="t"
    ) is None


def _hold_write_origin(origin: str):
    from tools.skill_provenance import reset_current_write_origin, set_current_write_origin

    token = set_current_write_origin(origin)
    return lambda: reset_current_write_origin(token)


@pytest.mark.parametrize("origin", ["side_question", "background_review", "plugin_fork:cutout"])
def test_a_fork_write_origin_does_not_take_the_turn_png(origin):
    delivery.remember("s", "/p/a.png")
    release = _hold_write_origin(origin)
    try:
        side = delivery.attach_reply(response_text="side answer.", session_id="s", platform="discord", turn_id="t")
    finally:
        release()
    assert side is None
    reply = delivery.attach_reply(response_text="Main reply.", session_id="s", platform="discord", turn_id="t")
    assert reply == "Main reply.\n\nMEDIA:/p/a.png"


def test_the_user_turn_origin_still_attaches_the_png():
    delivery.remember("s", "/p/a.png")
    release = _hold_write_origin("assistant_tool")
    try:
        reply = delivery.attach_reply(response_text="Main reply.", session_id="s", platform="discord", turn_id="t")
    finally:
        release()
    assert reply == "Main reply.\n\nMEDIA:/p/a.png"


def test_a_review_fork_does_not_take_the_turn_png():
    delivery.remember("s", "/p/a.png")
    release = _hold_fork_whitelist({"memory"})
    try:
        side = delivery.attach_reply(response_text="side answer.", session_id="s", platform="discord")
    finally:
        release()
    assert side is None
    reply = delivery.attach_reply(response_text="Main reply.", session_id="s", platform="discord")
    assert reply == "Main reply.\n\nMEDIA:/p/a.png"


def test_a_rotated_session_keeps_the_png_on_the_same_turn():
    delivery.start_turn(session_id="old", turn_id="t1")
    delivery.remember("old", "/p/a.png")
    reply = delivery.attach_reply(
        response_text="Done after compression.", session_id="new", platform="discord", turn_id="t1"
    )
    assert reply == "Done after compression.\n\nMEDIA:/p/a.png"


def test_the_next_turn_does_not_inherit_a_png_left_on_the_rotated_parent():
    delivery.start_turn(session_id="old", turn_id="t1")
    delivery.remember("old", "/p/a.png")
    delivery.start_turn(session_id="new", turn_id="t2")
    assert delivery.attach_reply(response_text="Next.", session_id="new", platform="discord", turn_id="t2") is None


def test_thirty_three_pngs_in_one_turn_all_attach():
    paths = [f"/p/{index}.png" for index in range(33)]
    for path in paths:
        delivery.remember("s", path)
    reply = delivery.attach_reply(response_text="Done.", session_id="s", platform="discord")
    assert reply == "Done.\n\n" + "\n".join("MEDIA:" + path for path in paths)


@pytest.mark.parametrize(
    "text",
    [
        "Example:\n```\nMEDIA:/p/a.png\n```",
        "Type `MEDIA:/p/a.png` in a reply.",
        "> MEDIA:/p/a.png",
        '{"note": "MEDIA:/p/a.png"}',
    ],
)
def test_a_media_line_hermes_does_not_extract_is_not_already_written(text):
    delivery.remember("s", "/p/a.png")
    reply = delivery.attach_reply(response_text=text, session_id="s", platform="discord")
    assert reply == text + "\n\nMEDIA:/p/a.png"


def test_a_real_line_beside_a_fenced_copy_is_left_alone():
    delivery.remember("s", "/p/a.png")
    text = "```\nMEDIA:/p/a.png\n```\nMEDIA:/p/a.png"
    assert delivery.attach_reply(response_text=text, session_id="s", platform="discord") is None


def test_a_terminal_that_inherited_desktop_source_has_no_media_line(room, monkeypatch):
    import gateway.session_context as session_context

    session_context.reset_session_vars()
    try:
        body = _write_with_source(room, monkeypatch, "desktop", "term.png")
        assert body["wrote"] is True
        assert "media" not in body
    finally:
        session_context.reset_session_vars()
