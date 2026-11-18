"""Behavior of cutout_png, the pinned file, and the approval hook."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import engine
import gates
import model_store
import service
from errors import MODEL_FETCH, SHA256_MISMATCH
from model_store import MODEL_URL, ModelProblem
from pngutil import png

ROOT = Path(__file__).resolve().parents[1]
PIN = "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8"


def _plugin_cli():
    """Load this plugin's cli.py. The Hermes tree also has a top-level cli module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("cutout_cli_under_test", ROOT / "cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def room(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in (
        "HERMES_CRON_SESSION",
        "HERMES_SINGLE_QUERY_SESSION",
        "HERMES_SESSION_PLATFORM",
        "HERMES_PLUGIN_HOST_PROCESS",
        "HERMES_YOLO_MODE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "ftp_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    folder = tmp_path / "out"
    folder.mkdir()
    image = tmp_path / "photo.png"
    image.write_bytes(png(4, 4))
    gates.clear()
    gates.remember_context(None)
    yield {"home": home, "image": image, "folder": folder}
    gates.clear()
    gates.remember_context(None)


def _args(room, **extra):
    body = {"image": str(room["image"]), "folder": str(room["folder"])}
    body.update(extra)
    return body


def _names(folder: Path):
    return sorted(path.name for path in folder.iterdir())


def test_pinned_constants_match_the_downloaded_file():
    assert model_store.MODEL_SHA256 == PIN
    assert model_store.MODEL_BYTES == 4574861
    assert model_store.MODEL_VERSION == "v0.0.0"
    assert model_store.MODEL_URL == (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
    )
    sample = Path("/tmp/cutout-verify/u2netp.onnx")
    if sample.is_file():
        data = sample.read_bytes()
        assert len(data) == 4574861
        assert hashlib.sha256(data).hexdigest() == PIN
        assert hashlib.md5(data).hexdigest() == "8e83ca70e441ab06c318d82300c84806"


def test_source_does_not_request_other_models():
    banned = ("bria", "huggingface.co", "withoutbg", "atlascloud", "new_session", "remove(")
    for path in ROOT.rglob("*.py"):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for word in banned:
            assert word not in text, f"{path.name} contains {word}"


def test_fresh_import_does_not_load_the_runtime():
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import service\n"
        "assert 'rembg' not in sys.modules\n"
        "assert 'onnxruntime' not in sys.modules\n"
        "assert 'PIL' not in sys.modules\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_rembg_import_is_inside_the_cut_function():
    tree = ast.parse((ROOT / "engine.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            names = [alias.name for alias in node.names]
            assert "rembg" not in module
            assert "onnxruntime" not in names
            assert "PIL" not in names


def test_not_image(room):
    pytest.importorskip("PIL")
    room["image"].write_bytes(b"this is not a photo")
    before = _names(room["folder"])
    directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert directive is None
    assert body["ok"] is False
    assert body["error_type"] == "not_image"
    assert body["wrote"] is False
    assert "background" not in body["message"].lower()
    assert body["next_step"]
    assert _names(room["folder"]) == before


def test_too_large_file_uses_the_real_cap(room):
    huge = room["image"]
    fd = os.open(huge, os.O_WRONLY | os.O_TRUNC)
    try:
        os.ftruncate(fd, service.MAX_FILE_BYTES + 1)
    finally:
        os.close(fd)
    assert service.MAX_FILE_BYTES == 33554432
    assert service.MAX_PIXELS == 24000000
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["error_type"] == "too_large"
    assert body["wrote"] is False
    assert _names(room["folder"]) == []


def test_too_large_pixels_use_the_real_cap(room):
    pytest.importorskip("PIL")
    import struct
    import zlib
    from PIL import Image

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 10000, 10000, 8, 2, 0, 0, 0)
    room["image"].write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b"")
    )
    old = Image.MAX_IMAGE_PIXELS
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["error_type"] == "too_large"
    assert Image.MAX_IMAGE_PIXELS == old
    assert _names(room["folder"]) == []


def test_pixel_boundary(room, monkeypatch):
    pytest.importorskip("PIL")
    monkeypatch.setattr(service, "MAX_PIXELS", 4)
    room["image"].write_bytes(png(2, 2))
    plan = service.plan_write(_args(room))
    assert plan["ok"] is True
    room["image"].write_bytes(png(3, 2))
    refused = service.plan_write(_args(room))
    assert refused["error_type"] == "too_large"


def test_sha256_mismatch_does_not_load_or_replace(room, monkeypatch):
    pytest.importorskip("PIL")
    model = room["home"] / "models" / "cutout" / "u2netp.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"not-the-model")
    before = model.read_bytes()

    def explode(*_args, **_kwargs):
        raise AssertionError("download started")

    def no_cut(*_args, **_kwargs):
        raise AssertionError("model started")

    monkeypatch.setattr(model_store, "fetch_pinned", explode)
    monkeypatch.setattr(engine, "cut_rgba", no_cut)
    directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    assert directive["action"] == "approve"
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["ok"] is False
    assert body["error_type"] == "sha256_mismatch"
    assert body["next_step"].startswith("Delete that file")
    assert body["wrote"] is False
    assert model.read_bytes() == before
    assert _names(room["folder"]) == []


def test_model_fetch_failure_writes_nothing(room, monkeypatch):
    pytest.importorskip("PIL")
    class Boom:
        def open(self, *_args, **_kwargs):
            raise urllib.error.URLError("offline")

    monkeypatch.setattr(model_store, "_opener", lambda: Boom())
    service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["error_type"] == "model_fetch"
    assert body["wrote"] is False
    assert not (room["home"] / "models" / "cutout" / "u2netp.onnx").exists()
    assert _names(room["folder"]) == []


def test_bad_download_is_not_saved(room, monkeypatch):
    class Body:
        headers: dict = {}

        def read(self, _n=-1):
            return b"zzzz"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Opener:
        def open(self, *_args, **_kwargs):
            return Body()

    monkeypatch.setattr(model_store, "_opener", lambda: Opener())
    with pytest.raises(ModelProblem) as caught:
        model_store.ensure_model()
    assert caught.value.kind == SHA256_MISMATCH
    assert caught.value.next_step == model_store.BODY_MISMATCH_NEXT
    assert not caught.value.next_step.startswith("Delete")
    assert not (room["home"] / "models" / "cutout" / "u2netp.onnx").exists()


def test_second_call_does_not_download(room, monkeypatch):
    body = b"abc"
    monkeypatch.setattr(model_store, "MODEL_BYTES", len(body))
    monkeypatch.setattr(model_store, "MODEL_SHA256", hashlib.sha256(body).hexdigest())
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return body

    monkeypatch.setattr(model_store, "fetch_pinned", fetch)
    assert model_store.ensure_model() == body
    assert model_store.ensure_model() == body
    assert calls["n"] == 1
    assert (room["home"] / "models" / "cutout" / "u2netp.onnx").read_bytes() == body


def test_symlink_model_is_not_followed(room, monkeypatch):
    other = room["home"] / "elsewhere.onnx"
    other.write_bytes(b"secret-weights")
    model = room["home"] / "models" / "cutout" / "u2netp.onnx"
    model.parent.mkdir(parents=True)
    os.symlink(other, model)

    def explode(*_args, **_kwargs):
        raise AssertionError("download started")

    monkeypatch.setattr(model_store, "fetch_pinned", explode)
    with pytest.raises(ModelProblem) as caught:
        model_store.ensure_model()
    assert caught.value.kind == "symlink"
    assert "model path is a symlink" in caught.value.message
    assert other.read_bytes() == b"secret-weights"
    assert stat.S_ISLNK(os.lstat(model).st_mode)


def test_no_overwrite_symlink_hardlink_or_dotdot(room):
    pytest.importorskip("PIL")
    existing = room["folder"] / "photo-cutout.png"
    existing.write_bytes(b"keep")
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["error_type"] == "exists"
    assert existing.read_bytes() == b"keep"

    existing.unlink()
    keeper = room["folder"] / "keeper.png"
    keeper.write_bytes(b"linked")
    linked = room["folder"] / "linked.png"
    os.link(keeper, linked)
    body = json.loads(service.cutout_png(_args(room, name="linked.png"), session_id="s", task_id="t"))
    assert body["error_type"] == "hardlink"
    assert keeper.read_bytes() == b"linked"

    os.symlink(keeper, room["folder"] / "link.png")
    body = json.loads(service.cutout_png(_args(room, name="link.png"), session_id="s", task_id="t"))
    assert body["error_type"] == "symlink"
    assert "destination is a symlink" in body["message"]
    assert keeper.read_bytes() == b"linked"

    parent = room["folder"].parent
    dotted = str(parent / "nope" / ".." / room["folder"].name)
    body = json.loads(service.cutout_png({"image": str(room["image"]), "folder": dotted}, session_id="s", task_id="t"))
    assert body["error_type"] == "outside"
    assert keeper.read_bytes() == b"linked"


def test_ancestor_symlink_is_refused(room, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    nested = real / "box"
    nested.mkdir()
    body = service.plan_write({"image": str(room["image"]), "folder": str(link / "box")})
    assert body["error_type"] == "symlink"
    assert "parent directory" in body["message"]
    assert body["next_step"] == "Pass a real directory in place of link."
    assert _names(nested) == []


def test_path_kind_table(room, tmp_path):
    """One row per path kind. A symlink is refused and is not followed."""
    pytest.importorskip("PIL")
    folder = room["folder"]
    photo = room["image"]

    regular = service.plan_write({"image": str(photo), "folder": str(folder)})
    assert regular["ok"] is True
    assert regular["name"] == "photo-cutout.png"
    assert _names(folder) == []

    leaf = tmp_path / "leaf.png"
    os.symlink(photo, leaf)
    body = service.plan_write({"image": str(leaf), "folder": str(folder)})
    assert body["error_type"] == "symlink"
    assert body["message"] == "The photo is a symlink, so it was not followed and nothing was written."
    assert photo.read_bytes() == png(4, 4)

    real_dir = tmp_path / "real-dir"
    real_dir.mkdir()
    folder_link = tmp_path / "folder-link"
    os.symlink(real_dir, folder_link)
    body = service.plan_write({"image": str(photo), "folder": str(folder_link)})
    assert body["error_type"] == "symlink"
    assert body["message"] == "The folder is a symlink, so it was not followed and nothing was written."
    assert _names(real_dir) == []

    nested_real = tmp_path / "nested-real"
    nested_real.mkdir()
    parent_link = tmp_path / "parent-link"
    os.symlink(nested_real, parent_link)
    held = nested_real / "box"
    held.mkdir()
    body = service.plan_write({"image": str(photo), "folder": str(parent_link / "box")})
    assert body["error_type"] == "symlink"
    assert body["next_step"] == "Pass a real directory in place of parent-link."
    assert _names(held) == []

    pics = tmp_path / "pics"
    pics.mkdir()
    pic_link = tmp_path / "pic-link"
    os.symlink(pics, pic_link)
    shot = pics / "shot.png"
    shot.write_bytes(png(3, 3))
    body = service.plan_write({"image": str(pic_link / "shot.png"), "folder": str(folder)})
    assert body["error_type"] == "symlink"
    assert body["next_step"] == "Pass a real directory in place of pic-link."
    assert shot.read_bytes() == png(3, 3)

    os.symlink(photo, folder / "dest.png")
    body = service.plan_write({"image": str(photo), "folder": str(folder), "name": "dest.png"})
    assert body["error_type"] == "symlink"
    assert "destination is a symlink" in body["message"]

    mac_root = tmp_path / "mac"
    private_tmp = mac_root / "private" / "tmp"
    private_tmp.mkdir(parents=True)
    os.symlink("private/tmp", mac_root / "tmp")
    mac_photo = private_tmp / "shot.png"
    mac_photo.write_bytes(png(5, 5))
    body = service.plan_write({"image": str(mac_root / "tmp" / "shot.png"), "folder": str(folder)})
    assert body["error_type"] == "symlink"
    assert body["next_step"] == "Pass a real directory in place of tmp."
    assert body["wrote"] is False
    direct = service.plan_write({"image": str(mac_photo), "folder": str(folder)})
    assert direct["ok"] is True
    assert _names(folder) == ["dest.png"]

    if os.path.islink("/tmp"):
        real_root = Path("/tmp").resolve()
        live = real_root / ("cutout-g2-" + str(os.getpid()))
        try:
            live.mkdir()
            live_photo = live / "photo.png"
            live_photo.write_bytes(png(5, 5))
            via_tmp = service.plan_write({"image": "/tmp/" + live.name + "/photo.png", "folder": str(folder)})
            assert via_tmp["error_type"] == "symlink"
            assert via_tmp["next_step"] == "Pass a real directory in place of tmp."
            via_real = service.plan_write({"image": str(live_photo), "folder": str(folder)})
            assert via_real["ok"] is True
        finally:
            if live.exists():
                for child in live.iterdir():
                    child.unlink()
                live.rmdir()

    body = service.plan_write({"image": str(tmp_path / "missing.png"), "folder": str(folder)})
    assert body["error_type"] == "missing"
    body = service.plan_write({"image": str(photo), "folder": str(tmp_path / "no-such-folder")})
    assert body["error_type"] == "missing"
    body = service.plan_write({"image": "photo.png", "folder": str(folder)})
    assert body["error_type"] == "outside"
    body = service.plan_write({"image": "https://example.com/a.png", "folder": str(folder)})
    assert body["error_type"] == "outside"
    body = service.plan_write({"image": str(photo) + " ", "folder": str(folder)})
    assert body["error_type"] == "bad_argument"
    body = service.plan_write({"image": str(photo) + "/", "folder": str(folder)})
    assert body["error_type"] == "bad_argument"
    dotted = str(folder.parent / "nope" / ".." / folder.name)
    body = service.plan_write({"image": str(photo), "folder": dotted})
    assert body["error_type"] == "outside"
    assert _names(folder) == ["dest.png"]


def test_default_name_rows(room, tmp_path):
    pytest.importorskip("PIL")
    rows = (
        ("photo.png", "photo-cutout.png"),
        ("my photo.png", "my-photo-cutout.png"),
        ("写真.png", "photo-cutout.png"),
        ("ｐｈｏｔｏ.png", "photo-cutout.png"),
        ("José.png", "Jos-cutout.png"),
        ("Jose.png", "Jose-cutout.png"),
        ("a..b.png", "photo-cutout.png"),
    )
    for filename, expected in rows:
        assert service.default_name(Path(filename)) == expected
    first = tmp_path / "写真.png"
    second = tmp_path / "画像.png"
    first.write_bytes(png(6, 6))
    second.write_bytes(png(8, 8))
    one = service.plan_write({"image": str(first), "folder": str(room["folder"])})
    assert one["ok"] is True
    assert one["name"] == "photo-cutout.png"
    (room["folder"] / "photo-cutout.png").write_bytes(b"keep")
    two = service.plan_write({"image": str(second), "folder": str(room["folder"])})
    assert two["error_type"] == "exists"
    assert two["wrote"] is False
    assert (room["folder"] / "photo-cutout.png").read_bytes() == b"keep"


def test_model_ancestor_symlink_is_not_followed(room, tmp_path, monkeypatch):
    real = tmp_path / "realhome"
    real.mkdir()
    link = tmp_path / "linkhome"
    os.symlink(real, link)
    monkeypatch.setenv("HERMES_HOME", str(link))

    def explode(*_args, **_kwargs):
        raise AssertionError("download started")

    monkeypatch.setattr(model_store, "fetch_pinned", explode)
    with pytest.raises(ModelProblem) as caught:
        model_store.ensure_model()
    assert caught.value.kind == "symlink"
    assert "parent directory" in caught.value.message
    assert not (real / "models").exists()


def test_unknown_arguments_are_refused(room):
    for extra in ({"model": "bria-rmbg"}, {"providers": "CUDA"}, {"session": "u2net"}):
        body = service.plan_write(_args(room, **extra))
        assert body["error_type"] == "bad_argument"
        assert body["wrote"] is False
    assert service.on_pre_tool_call("cutout_png", _args(room, model="bria-rmbg"), "s", "t") is None


def test_hook_message_names_the_paths(room):
    pytest.importorskip("PIL")
    directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    assert directive["action"] == "approve"
    assert str(room["image"]) in directive["message"]
    assert "photo-cutout.png" in directive["message"]
    assert PIN in directive["message"]
    assert "not uploaded" in directive["message"]
    assert "\n" not in directive["message"]
    assert "`" not in directive["message"]
    assert directive["rule_key"].startswith("cutout_png:")
    assert len(directive["rule_key"].split(":", 1)[1]) == 16
    renamed = service.on_pre_tool_call("cutout_png", _args(room, name="sprite.png"), "s", "t")
    assert renamed["rule_key"] != directive["rule_key"]


def test_overlong_or_backtick_line_is_not_sent(room, monkeypatch):
    pytest.importorskip("PIL")
    original = service.describe
    monkeypatch.setattr(service, "describe", lambda image, dest: "x" * 1501)
    directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    assert directive["action"] == "block"
    assert directive["message"] == gates.NOT_SENT
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["error_type"] == "approval_denied"
    assert _names(room["folder"]) == []

    odd = room["folder"].parent / "pho`to.png"
    odd.write_bytes(png(2, 2))
    monkeypatch.setattr(service, "describe", original)
    directive = service.on_pre_tool_call(
        "cutout_png",
        {"image": str(odd), "folder": str(room["folder"])},
        "s",
        "t",
    )
    assert directive["action"] == "block"


def test_unattended_blocks_even_after_a_saved_plan(room, monkeypatch):
    pytest.importorskip("PIL")
    directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    assert directive["action"] == "approve"
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    blocked = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    assert blocked["action"] == "block"
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["error_type"] == "unattended"
    assert body["wrote"] is False
    assert _names(room["folder"]) == []


def test_handler_refuses_unattended_without_a_hook(room, monkeypatch):
    pytest.importorskip("PIL")
    plan = service.plan_write(_args(room))
    gates.remember(service.accepted_args(_args(room)), "s", "t", plan["line"])
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "yes")
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["error_type"] == "unattended"
    assert _names(room["folder"]) == []


def test_yolo_does_not_add_a_plugin_block(room, monkeypatch):
    pytest.importorskip("PIL")
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    assert directive["action"] == "approve"


def test_other_tools_leave_other_rooms_alone(room):
    gates.remember({"image": "a", "folder": "b"}, "room", "task", "saved-line")
    assert service.on_pre_tool_call("file_read", {"path": "a"}, "room", "task") is None
    found = gates.pop({"image": "a", "folder": "b"}, "room", "task")
    assert found is not None
    assert found[0] == "saved-line"


def test_cli_does_not_write(room, capsys):
    pytest.importorskip("PIL")
    cli = _plugin_cli()

    code = cli.handle(
        type("Args", (), {"command": "call", "image": str(room["image"]), "folder": str(room["folder"]), "name": None})()
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["error_type"] == "approval_required"
    assert payload["wrote"] is False
    assert _names(room["folder"]) == []
    assert not (room["home"] / "models" / "cutout" / "u2netp.onnx").exists()


def test_direct_handler_without_approval_writes_nothing(room):
    pytest.importorskip("PIL")
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["error_type"] == "approval_denied"
    assert _names(room["folder"]) == []


def test_success_counts_come_from_the_written_file(room, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    def paint(image, _model):
        rgba = image.convert("RGBA")
        pixels = rgba.load()
        width, height = rgba.size
        for y in range(height):
            for x in range(width):
                red, green, blue, _alpha = pixels[x, y]
                alpha = 0 if x == 0 or y == 0 else 255
                pixels[x, y] = (red, green, blue, alpha)
        return rgba

    monkeypatch.setattr(service, "ensure_model", lambda: b"pinned")
    monkeypatch.setattr(engine, "cut_rgba", paint)
    service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["ok"] is True
    assert body["wrote"] is True
    assert body["model"] == "u2netp"
    assert body["frames_used"] == 1
    assert body["width"] == 4 and body["height"] == 4
    path = Path(body["path"])
    raw = path.read_bytes()
    assert len(raw) == body["bytes"]
    assert hashlib.sha256(raw).hexdigest() == body["sha256"]
    with Image.open(path) as saved:
        hist = saved.getchannel("A").histogram()
    assert body["alpha_zero"] == hist[0]
    assert body["alpha_full"] == hist[255]
    assert body["alpha_zero"] > 0 and body["alpha_full"] > 0
    assert "background is gone" not in body["message"]
    again = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert again["error_type"] == "exists"
    assert path.read_bytes() == raw


def test_readback_mismatch_is_not_success(room, monkeypatch):
    pytest.importorskip("PIL")
    real = service.read_child

    def swapped(directory, name, limit):
        data = real(directory, name, limit)
        if name.endswith("-cutout.png"):
            return b"different"
        return data

    def paint(image, _model):
        return image.convert("RGBA")

    monkeypatch.setattr(service, "ensure_model", lambda: b"pinned")
    monkeypatch.setattr(engine, "cut_rgba", paint)
    monkeypatch.setattr(service, "read_child", swapped)
    service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["ok"] is False
    assert body["error_type"] == "partial"
    assert body["wrote"] is False


def test_proxy_handler_ignores_proxy_env(monkeypatch):
    names = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "NO_PROXY",
        "ftp_proxy",
        "FTP_PROXY",
    )
    for name in names:
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.setattr(
        urllib.request,
        "getproxies",
        lambda: {"http": "http://127.0.0.1:9", "https": "http://127.0.0.1:9"},
    )
    opener = model_store._opener()
    used = [item.proxies for item in opener.handlers if getattr(item, "proxies", None)]
    assert used == []
    plain = urllib.request.build_opener()
    poisoned = [item.proxies for item in plain.handlers if getattr(item, "proxies", None)]
    assert poisoned and "127.0.0.1" in str(poisoned)


def test_redirect_stops_at_the_allowlist():
    class Script:
        def __init__(self):
            self.seen = []

        def __call__(self):
            return self

        def open(self, request, timeout=None):
            self.seen.append(request.full_url)
            assert request.get_header("Authorization") is None
            assert request.get_header("User-agent") == "hermes-plugin-cutout"
            if request.full_url == MODEL_URL:
                raise model_store._Hop("https://169.254.169.254/latest")
            raise AssertionError(request.full_url)

    script = Script()
    with pytest.raises(ModelProblem) as caught:
        model_store.fetch_pinned(script)
    assert caught.value.kind == MODEL_FETCH
    assert script.seen == [MODEL_URL]

    class Second:
        def __call__(self):
            return self

        def open(self, request, timeout=None):
            if request.full_url == MODEL_URL:
                raise model_store._Hop("https://release-assets.githubusercontent.com/u2netp.onnx")
            raise model_store._Hop("https://192.0.2.1/next")

    with pytest.raises(ModelProblem) as caught:
        model_store.fetch_pinned(Second())
    assert caught.value.kind == MODEL_FETCH

    class Plain:
        def __call__(self):
            return self

        def open(self, request, timeout=None):
            raise model_store._Hop("http://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx")

    with pytest.raises(ModelProblem):
        model_store.fetch_pinned(Plain())

    class Zero:
        def __call__(self):
            return self

        def open(self, request, timeout=None):
            raise model_store._Hop("https://0.0.0.0/u2netp.onnx")

    with pytest.raises(ModelProblem):
        model_store.fetch_pinned(Zero())


def test_dotenv_is_not_read(room, capsys):
    secret = "CUTOUT_SHOULD_NOT_APPEAR_IN_OUTPUT"
    (room["folder"] / ".env").write_text(f"HTTP_PROXY=http://127.0.0.1:9\nTOKEN={secret}\n", encoding="utf-8")
    cli = _plugin_cli()

    cli.handle(
        type("Args", (), {"command": "call", "image": str(room["image"]), "folder": str(room["folder"]), "name": None})()
    )
    assert secret not in capsys.readouterr().out
    for path in ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        assert "dotenv" not in path.read_text(encoding="utf-8").lower()


def test_missing_folder_and_relative_path(room):
    body = service.plan_write({"image": str(room["image"]), "folder": str(room["folder"] / "missing")})
    assert body["error_type"] == "missing"
    body = service.plan_write({"image": "photo.png", "folder": str(room["folder"])})
    assert body["error_type"] == "outside"
    body = service.plan_write({"image": str(room["image"]), "folder": str(room["folder"]), "name": "../x.png"})
    assert body["error_type"] == "bad_argument"


def test_webhook_platform_is_unattended(room, monkeypatch):
    import types

    module = types.ModuleType("gateway.session_context")

    def reader(name, default=""):
        if name == "HERMES_SESSION_PLATFORM":
            return "api_server"
        return default

    module.get_session_env = reader
    monkeypatch.setitem(sys.modules, "gateway.session_context", module)
    assert gates.unattended() is True
    directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    assert directive is None or directive.get("action") == "block"


def test_host_process_blocks(room, monkeypatch):
    pytest.importorskip("PIL")
    monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")
    directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    assert directive["action"] == "block"
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["error_type"] == "approval_blocked"
    assert _names(room["folder"]) == []


def test_approval_expires(room, monkeypatch):
    clock = {"t": 10.0}
    monkeypatch.setattr(gates.time, "monotonic", lambda: clock["t"])
    gates.remember({"image": "a", "folder": "b"}, "s", "t", "line")
    clock["t"] = 10.0 + gates.TTL_S + 1
    status, _message = gates.consume({"image": "a", "folder": "b"}, "s", "t", "line")
    assert status == "expired"


def test_consume_is_once():
    gates.clear()
    gates.remember({"image": "a", "folder": "b"}, "s", "t", "line")
    assert gates.consume({"image": "a", "folder": "b"}, "s", "t", "line")[0] == "approved"
    assert gates.consume({"image": "a", "folder": "b"}, "s", "t", "line")[0] == "denied"
    gates.clear()


def test_first_frame_only(room, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    frames = [Image.new("RGB", (3, 2), (255, 0, 0)), Image.new("RGB", (3, 2), (0, 0, 255))]
    gif = room["image"].with_suffix(".gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=40, loop=0)
    seen = {}

    def paint(image, _model):
        seen["size"] = image.size
        return image.convert("RGBA")

    monkeypatch.setattr(service, "ensure_model", lambda: b"pinned")
    monkeypatch.setattr(engine, "cut_rgba", paint)
    args = {"image": str(gif), "folder": str(room["folder"]), "name": "frame.png"}
    service.on_pre_tool_call("cutout_png", args, "s", "t")
    body = json.loads(service.cutout_png(args, session_id="s", task_id="t"))
    assert body["ok"] is True
    assert body["frame_count"] >= 2
    assert body["frames_used"] == 1
    assert body["width"] == 3 and body["height"] == 2
    assert seen["size"] == (3, 2)


def test_exif_orientation_is_not_applied(room, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    image = Image.new("RGB", (2, 5), (0, 80, 200))
    exif = Image.Exif()
    exif[274] = 6
    jpeg = room["folder"].parent / "turned.jpg"
    image.save(jpeg, format="JPEG", exif=exif)
    seen = {}

    def paint(opened, _model):
        seen["size"] = opened.size
        return opened.convert("RGBA")

    monkeypatch.setattr(service, "ensure_model", lambda: b"pinned")
    monkeypatch.setattr(engine, "cut_rgba", paint)
    args = {"image": str(jpeg), "folder": str(room["folder"]), "name": "turned.png"}
    service.on_pre_tool_call("cutout_png", args, "s", "t")
    body = json.loads(service.cutout_png(args, session_id="s", task_id="t"))
    assert body["ok"] is True
    assert seen["size"] == (2, 5)
    assert body["width"] == 2 and body["height"] == 5


def test_hermes_dispatcher_approve(room, monkeypatch):
    pytest.importorskip("PIL")
    hermes_src = os.environ.get("HERMES_SRC", "")
    if not hermes_src or not (Path(hermes_src) / "hermes_cli" / "plugins.py").is_file():
        pytest.skip("set HERMES_SRC to a Hermes source tree")
    monkeypatch.syspath_prepend(hermes_src)
    sys.modules.pop("hermes_cli.plugins", None)
    try:
        import hermes_cli.plugins as plugins
    except Exception as exc:
        pytest.skip("Hermes source could not be imported: " + str(exc))

    manager = plugins.PluginManager(scope_key=str(room["home"]))
    manager._discovered = True
    manager._hooks["pre_tool_call"] = [service.on_pre_tool_call]
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    details = plugins._get_pre_tool_call_directive_details(
        "cutout_png", _args(room), task_id="t", session_id="s"
    )
    assert details.action == "approve"
    assert str(room["image"]) in details.message
    silent = plugins._get_pre_tool_call_directive_details(
        "file_read", {"path": "x"}, task_id="t", session_id="s"
    )
    assert silent.action is None


_NAME_ROWS = (
    ("ascii", "parent-link", "parent-link", True),
    ("dot_underscore", "pic.link_1", "pic.link_1", True),
    ("japanese", "写真親", "写真親", True),
    ("fullwidth", "ｐｈｏｔｏ", "ｐｈｏｔｏ", True),
    ("accent", "José", "José", True),
    ("space", "my photos", "my photos", True),
    ("question", "a?b", "a?b", True),
    ("quote", 'a"b', 'a"b', True),
    ("emoji", "写真📷", "写真📷", True),
    ("backtick", "pho`to", "that directory", True),
    ("line_sep", "a\u2028b", "that directory", False),
    ("para_sep", "a\u2029b", "that directory", False),
    ("zwsp", "a\u200bb", "that directory", False),
    ("mixed_jp_nl", "写真\n親", "that directory", False),
    ("newline", "a\nb", "that directory", False),
    ("cr", "a\rb", "that directory", False),
    ("tab", "a\tb", "that directory", False),
    ("delete", "a\x7fb", "that directory", False),
    ("bidi", "ab\u202ecd", "that directory", False),
    ("private_use", "ab\ue000cd", "that directory", False),
    ("nul", "a\x00b", "that directory", False),
    ("empty", "", "that directory", False),
    ("dot", ".", "that directory", False),
    ("dotdot", "..", "that directory", False),
    ("slash", "a/b", "that directory", False),
    ("backslash", "a\\b", "that directory", False),
    ("spaces", "   ", "that directory", False),
    ("dot_padded", " . ", "that directory", False),
    ("long", "p" * 81, "that directory", True),
    ("limit", "q" * 80, "q" * 80, True),
)


def test_parent_reply_name_table(room, tmp_path):
    """Every component kind. A creatable symlink parent uses the same reply."""
    pytest.importorskip("PIL")
    for kind, component, expected, try_disk in _NAME_ROWS:
        assert service._name_row(component) == ("show" if expected == component else "hide"), kind
        assert service._shown_component(component) == expected, kind
        if not try_disk:
            continue
        real = tmp_path / ("real-" + kind)
        real.mkdir()
        held = real / "box"
        held.mkdir()
        link = tmp_path / ("wrap-" + kind) / component
        link.parent.mkdir()
        os.symlink(real, link)
        body = service.plan_write({"image": str(room["image"]), "folder": str(link / "box")})
        assert body["error_type"] == "symlink", kind
        assert body["ok"] is False, kind
        assert body["wrote"] is False, kind
        assert body["next_step"] == "Pass a real directory in place of " + expected + ".", kind
        assert _names(held) == [], kind


def test_mismatch_message_table(room, monkeypatch):
    """Saved bytes ask for a delete. A body that was not saved does not."""
    good = b"pinned-bytes"
    monkeypatch.setattr(model_store, "MODEL_BYTES", len(good))
    monkeypatch.setattr(model_store, "MODEL_SHA256", hashlib.sha256(good).hexdigest())
    monkeypatch.setattr(model_store, "fetch_pinned", lambda: good)

    def run_with(reads):
        state = {"n": 0}

        def read_existing(_home):
            item = reads[min(state["n"], len(reads) - 1)]
            state["n"] += 1
            return item

        monkeypatch.setattr(model_store, "_read_existing", read_existing)
        with pytest.raises(ModelProblem) as caught:
            model_store.ensure_model()
        return caught.value

    download = run_with([None])
    assert download.kind == SHA256_MISMATCH
    assert download.next_step == model_store.BODY_MISMATCH_NEXT
    assert not download.next_step.startswith("Delete")

    monkeypatch.setattr(model_store, "fetch_pinned", lambda: b"zzzz")
    monkeypatch.setattr(model_store, "MODEL_BYTES", 4)
    monkeypatch.setattr(model_store, "MODEL_SHA256", hashlib.sha256(b"abcd").hexdigest())
    fetched = run_with([None])
    assert fetched.next_step == model_store.BODY_MISMATCH_NEXT

    monkeypatch.setattr(model_store, "fetch_pinned", lambda: good)
    monkeypatch.setattr(model_store, "MODEL_BYTES", len(good))
    monkeypatch.setattr(model_store, "MODEL_SHA256", hashlib.sha256(good).hexdigest())
    existing = run_with([b"not-the-model"])
    assert existing.next_step == model_store.DISK_MISMATCH_NEXT
    assert existing.next_step.startswith("Delete")

    readback = run_with([None, b"not-the-model"])
    assert readback.next_step == model_store.DISK_MISMATCH_NEXT

    missing_after_write = run_with([None, None])
    assert missing_after_write.next_step == model_store.BODY_MISMATCH_NEXT

    def exists(*_args, **_kwargs):
        raise model_store.WriteRefused("exists", "u2netp.onnx")

    monkeypatch.setattr(model_store, "write_bytes", exists)
    raced_bad = run_with([None, b"not-the-model"])
    assert raced_bad.next_step == model_store.DISK_MISMATCH_NEXT
    raced_gone = run_with([None, None])
    assert raced_gone.next_step == model_store.BODY_MISMATCH_NEXT


def test_context_cron_blocks_without_an_environment_flag(room, monkeypatch):
    pytest.importorskip("PIL")
    hermes_src = os.environ.get("HERMES_SRC", "")
    if not hermes_src or not (Path(hermes_src) / "gateway" / "session_context.py").is_file():
        pytest.skip("set HERMES_SRC to a Hermes source tree")
    monkeypatch.syspath_prepend(hermes_src)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    from gateway.session_context import get_session_env, reset_session_vars, set_session_vars

    set_session_vars(cron_session="1")
    try:
        assert os.environ.get("HERMES_CRON_SESSION") in (None, "")
        assert get_session_env("HERMES_CRON_SESSION") == "1"
        assert gates.unattended() is True
        directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
        assert directive["action"] == "block"
        body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
        assert body["error_type"] == "unattended"
        assert body["wrote"] is False
        assert _names(room["folder"]) == []
    finally:
        reset_session_vars()


def test_public_cron_reader_present_missing_and_raising(room, monkeypatch):
    pytest.importorskip("PIL")
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    namespace = {
        "__package__": "",
        "__name__": "cutout_register_under_test",
        "__file__": str(ROOT / "__init__.py"),
    }
    exec(compile((ROOT / "__init__.py").read_text(encoding="utf-8"), str(ROOT / "__init__.py"), "exec"), namespace)
    register = namespace["register"]

    class _Hooks:
        def register_tool(self, **_kwargs):
            return None

        def register_skill(self, *_args, **_kwargs):
            return None

        def register_hook(self, *_args, **_kwargs):
            return None

        def register_cli_command(self, **_kwargs):
            return None

    class Present(_Hooks):
        @staticmethod
        def current_cron_execution():
            return object()

    class Missing(_Hooks):
        pass

    class Raising(_Hooks):
        @staticmethod
        def current_cron_execution():
            raise RuntimeError("identity down")

    try:
        register(Present())
        assert gates.unattended() is True
        directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
        assert directive["action"] == "block"
        assert directive["message"] == gates.UNATTENDED_MESSAGE
        body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
        assert body["error_type"] == "unattended"
        assert body["wrote"] is False
        assert _names(room["folder"]) == []

        register(Missing())
        assert gates.unattended() is False
        directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
        assert directive["action"] == "approve"
        assert gates.UNATTENDED_MESSAGE not in directive["message"]

        register(Raising())
        assert gates.unattended() is True
        directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
        assert directive["action"] == "block"
        body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
        assert body["error_type"] == "unattended"
        assert body["wrote"] is False
        assert _names(room["folder"]) == []
    finally:
        gates.remember_context(None)


def test_session_reader_exception_blocks_the_write(room, monkeypatch):
    pytest.importorskip("PIL")
    module = type(sys)("gateway.session_context")

    def reader(_name, _default=""):
        raise RuntimeError("session down")

    module.get_session_env = reader
    monkeypatch.setitem(sys.modules, "gateway.session_context", module)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    assert gates.unattended() is True
    directive = service.on_pre_tool_call("cutout_png", _args(room), "s", "t")
    assert directive["action"] == "block"
    body = json.loads(service.cutout_png(_args(room), session_id="s", task_id="t"))
    assert body["error_type"] == "unattended"
    assert _names(room["folder"]) == []


def test_allowlist_reaches_a_new_process(room, tmp_path):
    """Hermes writes one plugin_rule line. A new process honors that line only."""
    pytest.importorskip("PIL")
    hermes_src = os.environ.get("HERMES_SRC", "")
    if not hermes_src or not (Path(hermes_src) / "tools" / "approval.py").is_file():
        pytest.skip("set HERMES_SRC to a Hermes source tree")
    other = tmp_path / "out2"
    other.mkdir()
    args = _args(room)
    other_args = {"image": args["image"], "folder": str(other)}
    key = gates.rule_key(service.accepted_args(args))
    other_key = gates.rule_key(service.accepted_args(other_args))
    assert key != other_key
    plan = service.plan_write(args)
    assert plan["ok"] is True
    home = room["home"]
    (home / "config.yaml").write_text("approvals:\n  mode: manual\n", encoding="utf-8")
    grant = tmp_path / "grant.py"
    grant.write_text(
        "import os, sys\n"
        "sys.path.insert(0, os.environ['HERMES_SRC'])\n"
        "from tools.approval import request_tool_approval\n"
        "result = request_tool_approval(\n"
        "    'cutout_png', os.environ['PLAN'], rule_key=os.environ['RULE'],\n"
        "    approval_callback=lambda *_a, **_k: 'always')\n"
        "print('GRANT', bool(result.get('approved')))\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["HOME"] = str(tmp_path / "user")
    env["HERMES_SRC"] = hermes_src
    env["HERMES_INTERACTIVE"] = "1"
    env["RULE"] = key
    env["PLAN"] = plan["line"]
    env.pop("HERMES_CRON_SESSION", None)
    env.pop("HERMES_YOLO_MODE", None)
    env.pop("HERMES_SINGLE_QUERY_SESSION", None)
    env.pop("HERMES_SESSION_PLATFORM", None)
    (tmp_path / "user").mkdir()
    first = subprocess.run(
        [sys.executable, str(grant)],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=90,
        stdin=subprocess.DEVNULL,
    )
    assert first.returncode == 0, first.stderr
    assert "GRANT True" in first.stdout
    stored = (home / "config.yaml").read_text(encoding="utf-8")
    assert "plugin_rule:" + key in stored
    assert str(room["image"]) not in stored
    assert other_key not in stored
    ask = tmp_path / "ask.py"
    ask.write_text(
        "import os, sys\n"
        "sys.path.insert(0, os.environ['HERMES_SRC'])\n"
        "from tools.approval import request_tool_approval\n"
        "calls = {'n': 0}\n"
        "def cb(*_a, **_k):\n"
        "    calls['n'] += 1\n"
        "    return 'deny'\n"
        "same = request_tool_approval('cutout_png', 'plan', rule_key=os.environ['RULE'], approval_callback=cb)\n"
        "print('SAME', bool(same.get('approved')), calls['n'])\n"
        "calls['n'] = 0\n"
        "diff = request_tool_approval('cutout_png', 'plan', rule_key=os.environ['OTHER'], approval_callback=cb)\n"
        "print('DIFF', bool(diff.get('approved')), calls['n'])\n",
        encoding="utf-8",
    )
    env.pop("HERMES_INTERACTIVE", None)
    env["OTHER"] = other_key
    second = subprocess.run(
        [sys.executable, str(ask)],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=90,
        stdin=subprocess.DEVNULL,
    )
    assert second.returncode == 0, second.stderr
    assert "SAME True 0" in second.stdout
    assert "DIFF False" in second.stdout
    cron_script = tmp_path / "cron_same.py"
    cron_script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, os.environ['PLUGIN'])\n"
        "sys.path.insert(0, os.environ['HERMES_SRC'])\n"
        "os.environ.pop('HERMES_CRON_SESSION', None)\n"
        "import engine, service\n"
        "service.ensure_model = lambda: b'pinned'\n"
        "engine.cut_rgba = lambda image, _model: image.convert('RGBA')\n"
        "from gateway.session_context import get_session_env, set_session_vars\n"
        "import hermes_cli.plugins as plugins\n"
        "set_session_vars(cron_session='1')\n"
        "print('ENV', os.environ.get('HERMES_CRON_SESSION'))\n"
        "print('VAR', get_session_env('HERMES_CRON_SESSION'))\n"
        "manager = plugins.PluginManager(scope_key=os.environ['HERMES_HOME'])\n"
        "manager._discovered = True\n"
        "manager._hooks['pre_tool_call'] = [service.on_pre_tool_call]\n"
        "plugins._plugin_manager = manager\n"
        "args = json.loads(os.environ['ARGS'])\n"
        "block, _modified = plugins._dispatch_pre_tool_call_hooks(\n"
        "    'cutout_png', args, session_id='cron', task_id='cron')\n"
        "print('BLOCK', 'NONE' if block is None else 'SET')\n"
        "body = json.loads(service.cutout_png(args, session_id='cron', task_id='cron'))\n"
        "print('KIND', body.get('error_type'), body.get('wrote'))\n"
        "print('FILES', sorted(p.name for p in Path(args['folder']).iterdir()))\n",
        encoding="utf-8",
    )
    apply_script = tmp_path / "apply_same.py"
    apply_script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, os.environ['PLUGIN'])\n"
        "sys.path.insert(0, os.environ['HERMES_SRC'])\n"
        "os.environ.pop('HERMES_CRON_SESSION', None)\n"
        "import engine, service\n"
        "service.ensure_model = lambda: b'pinned'\n"
        "engine.cut_rgba = lambda image, _model: image.convert('RGBA')\n"
        "import hermes_cli.plugins as plugins\n"
        "from tools.terminal_tool import set_approval_callback\n"
        "calls = {'n': 0}\n"
        "def cb(*_a, **_k):\n"
        "    calls['n'] += 1\n"
        "    return 'deny'\n"
        "set_approval_callback(cb)\n"
        "manager = plugins.PluginManager(scope_key=os.environ['HERMES_HOME'])\n"
        "manager._discovered = True\n"
        "manager._hooks['pre_tool_call'] = [service.on_pre_tool_call]\n"
        "plugins._plugin_manager = manager\n"
        "args = json.loads(os.environ['ARGS'])\n"
        "block, _modified = plugins._dispatch_pre_tool_call_hooks(\n"
        "    'cutout_png', args, session_id='person', task_id='person')\n"
        "print('BLOCK', 'NONE' if block is None else 'SET')\n"
        "print('CALLS', calls['n'])\n"
        "body = json.loads(service.cutout_png(args, session_id='person', task_id='person'))\n"
        "print('WROTE', body.get('ok'), body.get('wrote'))\n"
        "print('FILES', sorted(p.name for p in Path(args['folder']).iterdir()))\n",
        encoding="utf-8",
    )
    env["PLUGIN"] = str(ROOT)
    env["ARGS"] = json.dumps(args)
    env["HERMES_INTERACTIVE"] = "1"
    cron = subprocess.run(
        [sys.executable, str(cron_script)],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=90,
        stdin=subprocess.DEVNULL,
    )
    assert cron.returncode == 0, cron.stderr
    assert "ENV None" in cron.stdout
    assert "VAR 1" in cron.stdout
    assert "BLOCK SET" in cron.stdout
    assert "KIND unattended False" in cron.stdout
    assert "FILES []" in cron.stdout
    person = subprocess.run(
        [sys.executable, str(apply_script)],
        env=env,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=90,
        stdin=subprocess.DEVNULL,
    )
    assert person.returncode == 0, person.stderr
    assert "BLOCK NONE" in person.stdout
    assert "CALLS 0" in person.stdout
    assert "WROTE True True" in person.stdout
    assert "FILES ['photo-cutout.png']" in person.stdout
    assert _names(room["folder"]) == ["photo-cutout.png"]
    assert _names(other) == []


def test_public_copy_matches_the_gates():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    skill = (ROOT / "skills" / "cutout" / "SKILL.md").read_text(encoding="utf-8")
    for text in (readme, skill):
        assert "plugin_rule" in text
        assert "command_allowlist" in text
        assert "different folder" in text
        assert "even after always" in text
        assert "letters outside ASCII" in text
        assert "same plan" in text or "same arguments" in text
    assert "one example from 2026-11-10" in readme
    assert "does not ask you to delete a file" in readme
    assert "no file to delete" in skill
    assert "80 characters" in readme
    assert "HERMES_HOME/installs" in readme
    assert "HERMES_HOME/installs" in skill
