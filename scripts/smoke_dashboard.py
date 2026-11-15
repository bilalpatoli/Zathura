"""Render the five-page dashboard with empty and populated mission records."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

import zathura.dashboard as dashboard
from zathura.runs import run_list, run_view
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
            if page == "explore_page" and label == "existing data":
                select = page_app.selectbox(key="explore_target")
                assert any("known eclipsing binary" in option for option in select.options)
                select.select("KIC 10661783").run(timeout=90)
                assert not page_app.exception
                assert any("Known eclipsing binary" in m.value for m in page_app.markdown)
                assert len(page_app.get("plotly_chart")) == 4
                print(f"PASS {label}: Explore includes labelled EB, source photometry and BLS charts")
            if page == "research_page" and label == "existing data":
                state = dashboard.snapshot()
                for run in run_list(state["events"]):
                    view = run_view(state, run["loop_id"])
                    if (view["baseline"] or {}).get("data_source") == "placeholder":
                        page_app.selectbox(key="research_run").select(run["loop_id"]).run(timeout=90)
                        assert not page_app.exception
                        assert any("Stand-in engine (placeholder) run" in w.value for w in page_app.warning)
                        assert not any("Benchmark transits are synthetic injections" in c.value for c in page_app.caption)
                        print(f"PASS {label}: historical placeholder run is labelled honestly")
                        break
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
