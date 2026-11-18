import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import delivery


@pytest.fixture(autouse=True)
def _clear_cutout_delivery():
    delivery.clear()
    yield
    delivery.clear()
