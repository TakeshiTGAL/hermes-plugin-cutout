"""One turn attaches the PNGs that turn wrote, on a real Hermes loop."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    pytest.importorskip("PIL")
    hermes_src = os.environ.get("HERMES_SRC", "")
    if not hermes_src or not (Path(hermes_src) / "run_agent.py").is_file():
        pytest.skip("set HERMES_SRC to a Hermes source tree")
    work = tmp_path_factory.mktemp("turns")
    home = work / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        'approvals:\n  mode: "off"\ntools:\n  tool_search:\n    enabled: "off"\n',
        encoding="utf-8",
    )
    (work / "user").mkdir()
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(work / "user"),
        "HERMES_HOME": str(home),
        "HERMES_SRC": hermes_src,
        "HERMES_YOLO_MODE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(work),
    }
    done = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "turn_driver.py"), str(ROOT), str(work)],
        env=env,
        cwd=str(work),
        capture_output=True,
        text=True,
        timeout=300,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("REPORT ")]
    assert lines, (done.stdout + done.stderr)[-3000:]
    return json.loads(lines[-1][len("REPORT "):])


def test_turn_start_fires_once_and_before_every_tool(report):
    for name in ("two_tools", "names_one"):
        events = report[name][0]["events"]
        assert events == ["turn_start", "tool", "tool"], name


def test_each_turn_of_a_session_starts_once(report):
    assert [row["events"].count("turn_start") for row in report["stale"]] == [1, 1]


def test_both_pngs_of_one_turn_reach_the_reply(report):
    reply = report["two_tools"][0]["reply"]
    assert reply.count("MEDIA:" + report["paths"]["a"]) == 1
    assert reply.count("MEDIA:" + report["paths"]["b"]) == 1


def test_reply_that_names_one_png_gains_the_other_once(report):
    reply = report["names_one"][0]["reply"]
    assert reply.count("MEDIA:" + report["paths"]["c"]) == 1
    assert reply.count("MEDIA:" + report["paths"]["d"]) == 1


def test_an_empty_reply_does_not_leak_its_png_into_the_next_turn(report):
    assert Path(report["paths"]["e"]).name in report["stale"][0]["tools"][0]
    assert "MEDIA:" not in report["stale"][1]["reply"]


def test_desktop_result_and_reply_carry_the_line(report):
    row = report["desktop"][0]
    line = "MEDIA:" + report["paths"]["f"]
    assert json.loads(row["tools"][0])["media"] == line
    assert row["reply"].count(line) == 1
