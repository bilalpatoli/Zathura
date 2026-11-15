"""Headless render of the story and all engineering pages, with empty-data coverage."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

import zathura.dashboard as dashboard
from zathura.storage import load_records

ROOT = Path(__file__).resolve().parents[1]


PAGES = ("home_page", "explore_page", "research_page", "how_page", "report_page")


def check(label):
    with patch.object(dashboard.Mission, "launch", return_value=True) as launch:
        app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=90)
        if app.exception:
            raise AssertionError([e.message for e in app.exception])
        assert any("An AI scientist searching for planets." in m.value for m in app.markdown)
        assert not app.tabs
        launch.assert_not_called()
        print(f"PASS {label}: Home renders without starting research")
        plots = 0
        for page in PAGES:
            script = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import app; app.setup(); app.build_pages(); app.{page}()")
            page_app = AppTest.from_string(script).run(timeout=90)
            if page_app.exception:
                raise AssertionError([page, [e.message for e in page_app.exception]])
            plots += len(page_app.get("plotly_chart"))
            if page == "research_page" and label == "empty data":
                page_app.button(key="research_start").click().run(timeout=90)
                assert not page_app.exception
                launch.assert_called_once_with(dashboard.DEFAULT_ROUNDS)
                print(f"PASS {label}: Start research launches {dashboard.DEFAULT_ROUNDS} rounds only on click")
            print(f"PASS {label}: {page}")
        launch.assert_called_once() if label == "empty data" else launch.assert_not_called()
    print(f"PASS {label}: {plots} plots; no render exceptions")


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
