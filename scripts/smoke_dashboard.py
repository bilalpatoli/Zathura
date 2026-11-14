"""Headless render of the story and all engineering tabs, with empty-data coverage."""
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
    assert not app.json and not app.dataframe and not app.table, "Story must not dump JSON or tables"
    assert app.title[0].value == "Zathura: An autonomous AI astronomer"
    assert not app.tabs
    with patch.object(dashboard.Mission, "launch", return_value=True) as launch:
        app.button(key="story_begin").click().run(timeout=90)
        assert not app.exception
        launch.assert_called_once_with(3)
    print(f"PASS {label}: Begin Mission invokes the existing worker")
    for title in ("The Signal", "The Shadow", "The Scientist Thinks", "The Skeptic Pushes Back", "It Remembers", "Planet or Impostor?"):
        assert any(title in markdown.value for markdown in app.markdown), title
        print(f"PASS {label}: story · {title}")
    if label == "existing data":
        assert any("Planet candidate" in m.value for m in app.markdown)
        assert len(app.get("plotly_chart")) >= 4
        app.radio(key="story_category").set_value("Eclipsing binary").run(timeout=90)
        assert not app.exception
        assert any("Eclipsing binary" in m.value for m in app.markdown)
        print(f"PASS {label}: real planet and EB finale; no render exceptions")
    app.toggle[0].set_value(True).run(timeout=90)
    if app.exception:
        raise AssertionError([e.message for e in app.exception])
    assert not app.json, "Engineering tabs must not dump raw JSON"
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
