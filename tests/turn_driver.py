"""Run the plugin inside a real Hermes AIAgent against a local scripted model.

Usage: turn_driver.py <plugin root> <work dir>. Prints one JSON object.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT, WORK = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, os.environ["HERMES_SRC"])
sys.path.insert(0, str(ROOT / "tests"))

SCRIPT = {"calls": [], "final": "", "repeat": None}


class Model(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return None

    def do_GET(self):
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        messages = body.get("messages") or [{}]
        # Hermes nudges a model that went silent after a tool. ``repeat`` answers each nudge with the
        # same call, so the turn really ends with an empty reply and the output hook does not fire.
        again = SCRIPT["repeat"] if messages[-1].get("role") != "tool" else None
        if SCRIPT["calls"] or again:
            args = SCRIPT["calls"].pop(0) if SCRIPT["calls"] else again
            call = {
                "id": "call_" + uuid.uuid4().hex[:8],
                "type": "function",
                "function": {"name": "cutout_png", "arguments": json.dumps(args)},
            }
            message, finish = {"role": "assistant", "content": None, "tool_calls": [call]}, "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": SCRIPT["final"]}, "stop"
        usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        head = {"id": "c", "created": 1, "model": "mock"}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            delta = dict(message)
            if delta.get("tool_calls"):
                delta["tool_calls"] = [dict(delta["tool_calls"][0], index=0)]
            chunks = (
                {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage},
            )
            for chunk in chunks:
                chunk.update(head, object="chat.completion.chunk")
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        choice = {"index": 0, "message": message, "finish_reason": finish}
        out = json.dumps(dict(head, object="chat.completion", choices=[choice], usage=usage)).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
threading.Thread(target=server.serve_forever, daemon=True).start()

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager
from run_agent import AIAgent

from pngutil import png

# Hermes owns the top-level name ``cli``, so the plugin loads as a package, as Hermes loads it.
spec = importlib.util.spec_from_file_location(
    "cutout_under_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
manager = get_plugin_manager()
plugin.register(PluginContext(PluginManifest(name="cutout"), manager))
engine = importlib.import_module("cutout_under_test.engine")
service = importlib.import_module("cutout_under_test.service")

EVENTS: list[str] = []
manager._hooks.setdefault("pre_llm_call", []).append(lambda **_kwargs: EVENTS.append("turn_start"))
manager._hooks.setdefault("pre_tool_call", []).append(lambda **_kwargs: EVENTS.append("tool"))


def paint(image, _model):
    rgba = image.convert("RGBA")
    rgba.putpixel((0, 0), (0, 0, 0, 0))
    return rgba


service.ensure_model = lambda: b"pinned"
engine.cut_rgba = paint

photo = WORK / "photo.png"
photo.write_bytes(png(4, 4))
folder = WORK / "out"
folder.mkdir()


def plan(name: str) -> dict:
    return {"image": str(photo), "folder": str(folder), "name": name}


def agent_for(session: str, platform: str) -> AIAgent:
    return AIAgent(
        base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="mock",
        provider="custom",
        model="mock",
        platform=platform,
        session_id=session,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        enabled_toolsets=["cutout"],
        max_iterations=8,
    )


def turn(agent: AIAgent, calls: list[dict], final: str, repeat=None, history=None) -> dict:
    SCRIPT.update(calls=list(calls), final=final, repeat=repeat)
    EVENTS.clear()
    result = agent.run_conversation("cut the background", conversation_history=history)
    results = [m.get("content") or "" for m in result.get("messages", []) if m.get("role") == "tool"]
    return {
        "events": list(EVENTS),
        "reply": result.get("final_response") or "",
        "tools": results,
        "messages": result.get("messages"),
    }


def public(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "messages"}


def scene(session: str, bind: dict, platform: str, steps) -> list[dict]:
    tokens = set_session_vars(session_id=session, session_key=session, cron_session="", **bind)
    try:
        agent = agent_for(session, platform)
        rows, history = [], None
        for calls, final, repeat in steps:
            row = turn(agent, calls, final, repeat, history)
            history = row["messages"]
            rows.append(public(row))
        return rows
    finally:
        clear_session_vars(tokens)


a, b, c, d = (str(folder / name) for name in ("a.png", "b.png", "c.png", "d.png"))
report = {
    "two_tools": scene("two", {"platform": "discord"}, "discord", [([plan("a.png"), plan("b.png")], "Done.", None)]),
    "names_one": scene(
        "one", {"platform": "discord"}, "discord", [([plan("c.png"), plan("d.png")], "Done.\nMEDIA:" + c, None)]
    ),
    "stale": scene(
        "stale",
        {"platform": "discord"},
        "discord",
        [([plan("e.png")], "", plan("e.png")), ([], "Second turn.", None)],
    ),
    "desktop": scene("desk", {"source": "desktop"}, "desktop", [([plan("f.png")], "Done.", None)]),
    "paths": {"a": a, "b": b, "c": c, "d": d, "e": str(folder / "e.png"), "f": str(folder / "f.png")},
}
print("REPORT " + json.dumps(report, ensure_ascii=False))
