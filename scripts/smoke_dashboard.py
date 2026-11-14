"""Headless render of all four pages, including an isolated empty-data case."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

import zathura.dashboard as dashboard
from zathura.storage import load_records

ROOT = Path(__file__).resolve().parents[1]


PAGES = {"Mission Control": "app.mission_page()", "Planet Detection": "app.planets_page()",
         "AI Research Timeline": "app.timeline_page()", "Experimental Results": "app.results_page()"}


def check(label):
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=90)
    if app.exception:
        raise AssertionError([e.message for e in app.exception])
    plots = 0
    for page, call in PAGES.items():
        script = f"import sys; sys.path.insert(0, {str(ROOT)!r}); import app; app.setup(); {call}"
        page_app = AppTest.from_string(script).run(timeout=90)
        if page_app.exception:
            raise AssertionError([page, [e.message for e in page_app.exception]])
        plots += len(page_app.get("plotly_chart"))
        print(f"PASS {label}: {page}")
    print(f"PASS {label}: {plots} plots; no render exceptions")
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
