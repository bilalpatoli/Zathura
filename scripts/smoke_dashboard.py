"""Headless render of all five tabs, including an isolated empty-data case."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

import zathura.dashboard as dashboard
from zathura.storage import load_records

ROOT = Path(__file__).resolve().parents[1]


def check(label):
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=90)
    if app.exception:
        raise AssertionError([e.message for e in app.exception])
    assert not app.json, "Judge-facing tabs must not dump raw JSON"
    assert [tab.label for tab in app.tabs] == [
        "Mission Control", "Planet Detection", "AI Research Timeline", "Experimental Results", "Sponsor Evidence"]
    for tab in app.tabs:
        print(f"PASS {label}: {tab.label}")
    print(f"PASS {label}: {len(app.get('plotly_chart'))} plots; no render exceptions")
    return app


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with patch.object(dashboard, "ROOT", root), \
             patch.object(dashboard, "cached_targets", return_value=[]), \
             patch.object(dashboard.storage, "load_records", side_effect=lambda: load_records(root / "missing.jsonl")), \
             patch.object(dashboard.LocalBackend, "recall_memory", return_value=[]):
            check("empty data")
    check("existing data")


if __name__ == "__main__":
    main()
