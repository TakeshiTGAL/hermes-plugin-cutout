"""One check for each row of the build checklist."""

from pathlib import Path

import pytest

from schemas import SCHEMA

ROOT = Path(__file__).resolve().parents[1]


def _sentence(text: str) -> str:
    head = text.split(". ")[0]
    if not head.endswith("."):
        head += "."
    return head


def test_row_06_skill_and_readme_name_the_card():
    skill = (ROOT / "skills" / "cutout" / "SKILL.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Before approval, write in the message the photo path, the folder, the new PNG name" in skill
    assert "existing file is not overwritten" in skill
    assert "photo path" in readme
    assert "sha256" in readme


def test_row_07_readme_commands():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'hermes plugins install "file://$PWD" --enable --yes-deps' in readme
    assert 'hermes cutout call --image "$PWD/photo.png" --folder "$PWD/out"' in readme
    assert "exits 2" in readme
    assert "not a hermes command" in readme
    assert "pre_tool_call" in readme
    assert "hermes tools disable cutout --platform cli" in readme
    assert "no platform named `gateway`" in readme
    skill = (ROOT / "skills" / "cutout" / "SKILL.md").read_text(encoding="utf-8")
    assert "pre_tool_call" in skill
    assert "hermes tools disable cutout --platform cli" in skill


def test_row_08_ordinary_words_in_the_tool():
    text = SCHEMA["description"].lower()
    for word in (
        "background",
        "transparent",
        "photo",
        "png",
        "sprite",
        "backdrop",
        "foreground",
        "subject",
        "knocked",
        "extracted",
        "prepared",
        "asset",
    ):
        assert word in text
    assert "i need it gone" not in text
    assert "make a sprite of the subject." not in text
    assert "prepare the asset." not in text


def test_row_10_lengths():
    sentence = _sentence(SCHEMA["description"])
    assert len(sentence) <= 60
    assert not sentence.lower().startswith("read-only")
    skill = (ROOT / "skills" / "cutout" / "SKILL.md").read_text(encoding="utf-8")
    description = skill.split("---", 2)[1].split("description:", 1)[1].strip()
    assert len(description) <= 223
    assert "tool_search" in skill.split("---", 2)[2]
    assert "before web_search" in skill.split("---", 2)[2]


def test_row_11_dependencies_are_bounded():
    text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    assert "rembg>=2.0.87,<3" in text
    assert "onnxruntime>=1.31.1,<2" in text
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "pip install rembg" not in readme
    assert "pip install onnxruntime" not in readme


def test_row_13_public_files_are_english():
    import re

    japanese = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
    for path in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md")) + list(ROOT.rglob("*.yaml")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            assert japanese.search(line) is None, f"{path.name}: {line}"


def test_row_05_there_is_no_read_tool():
    text = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    assert "provides_tools:" in text
    assert text.count("cutout_png") >= 1
    assert "cutout_read" not in text
