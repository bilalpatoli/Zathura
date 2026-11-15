"""Render all four navigation routes with empty and populated mission records."""
import re
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

import zathura.dashboard as dashboard
from zathura.storage import load_records
from zathura.engine.vetting import vetting_targets
from zathura.simple import PITCH

ROOT = Path(__file__).resolve().parents[1]


PAGES = ["Find", "Improve", "Results", "Technical details"]


def plain(app):
    assert not app.json and not app.dataframe and not app.table, "Pitch pages must not dump data"
    text = " ".join(element.value for group in (app.markdown, app.caption, app.info) for element in group)
    assert not re.search(r"\b[0-9a-f]{32}\b|runtime|confidence interval", text, re.I), text


@patch.object(dashboard.Mission, "launch", return_value=True)
def check(label, launch):
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=90)
    assert not app.exception, [e.message for e in app.exception]
    assert [p["page_name"] for p in app._registered_pages.values()] == PAGES
    assert app.title[0].value == "Find"
    for page in PAGES:
        # AppTest.switch_page supports file pages only; use the registered callable route hash.
        app._page_hash = next(key for key, info in app._registered_pages.items() if info["page_name"] == page)
        app.run(timeout=90)
        assert not app.exception, [page, [e.message for e in app.exception]]
        assert any(PITCH in m.value for m in app.markdown)
        if page != "Technical details":
            assert app.title[0].value == page
            plain(app)
        if page == "Find":
            if label == "existing data":
                assert app.selectbox(key="find_star").value == 6665695
                assert any(">Planet candidate</div>" in m.value for m in app.markdown)
                assert len(app.get("plotly_chart")) == 1
                assert len(app.expander) == 1 and app.expander[0].label == "More checks"
                binary = next(t["kepid"] for t in vetting_targets() if t["truth"] == "EB")
                app.selectbox(key="find_star").set_value(binary).run(timeout=90)
                assert not app.exception
                assert any("Impostor: two stars" in m.value for m in app.markdown)
                app.toggle(key="find_raw").set_value(True).run(timeout=90)
                assert not app.exception and len(app.get("plotly_chart")) == 2
                plain(app)
                print(f"PASS {label}: Find · planet, two stars, extra checks and original brightness")
            else:
                assert not app.get("plotly_chart")
        elif page == "Improve":
            launch.reset_mock()
            app.button(key="simple_start").click().run(timeout=90)
            assert not app.exception
            launch.assert_called_once_with(3)
            plain(app)
            print(f"PASS {label}: Start AI research invokes the existing worker")
        elif page == "Results":
            assert len(app.get("plotly_chart")) == (1 if label == "existing data" else 0)
            assert len(app.metric) == (1 if label == "existing data" else 0)
        else:
            assert [tab.label for tab in app.tabs] == ["Mission Control", "Planet Detection", "Timeline",
                "Experimental Results", "Sponsor Evidence", "Vetting validation", "Research Story"]
            for title in ("The Signal", "The Shadow", "The Scientist Thinks", "The Skeptic Pushes Back", "It Remembers", "Planet or Impostor?"):
                assert any(title in m.value for m in app.markdown), title
            assert sum("Benchmark transits are synthetic injections into real Kepler photometry" in c.value
                       for c in app.caption) >= 5
            assert any(h.value == "Vetting confusion table" for h in app.subheader)
            if label == "existing data":
                assert any(e.label == "Source photometry and BLS periodogram" for e in app.expander)
            launch.reset_mock()
            app.button(key="story_begin").click().run(timeout=90)
            assert not app.exception
            launch.assert_called_once_with(3)
            assert not app.json
        print(f"PASS {label}: {page}")
    print(f"PASS {label}: all four sidebar routes; no render exceptions")


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
