import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from zathura.dashboard import Mission, read_jsonl, snapshot
from zathura.engine import cached_targets
from zathura.engine.data import load_catalog


class DashboardTests(unittest.TestCase):
    def test_service_configuration_is_not_live_evidence(self):
        import app

        keys = {"OPENAI_API_KEY": "test", "EXA_API_KEY": "test", "PASTDEV_API_KEY": "test"}
        for env, expected in (({}, ("OpenAI: MOCK", "Exa: MOCK", "past.dev: LOCAL JSONL")),
                              (keys, ("OpenAI: API configured", "Exa: API configured",
                                      "past.dev: REMOTE-FIRST configured")),
                              (keys | {"ZATHURA_MOCK": "1"},
                               ("OpenAI: MOCK", "Exa: MOCK", "past.dev: REMOTE-FIRST configured"))):
            with self.subTest(env=list(env)), patch.dict(app.os.environ, env, clear=True), \
                 patch.object(app, "st") as ui:
                ui.columns.return_value = [MagicMock(), MagicMock()]
                app.render_sponsors({"latest": [], "memories": []})
                captions = " ".join(c.args[0] for c in ui.caption.call_args_list)
                for label in expected:
                    self.assertIn(label, captions)
                self.assertNotIn("LIVE", captions)

    def test_explore_source_and_periodogram_charts(self):
        import app

        time = np.arange(10, dtype=float)
        curve = {"time": time, "flux": time + 100, "flat_flux": time + 1,
                 "phase": time - 5, "periods": time + 2, "power": time * 2,
                 "in_transit": time < 2, "data_source": "real", "curve_origin": "real",
                 "known_periods": [3.0], "detected": True, "period_days": 3.0,
                 "depth": 0.01, "duration_days": 0.1, "epoch": 0.0, "sde": 8.0, "quarter": 1, "product": "PDCSAP"}
        ui = MagicMock()
        ui.columns.side_effect = lambda n, **kw: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
        ui.selectbox.return_value = "KIC 7"
        ui.session_state = {"explore_target": "KIC 7"}
        ui.button.return_value = False
        with patch.object(app, "st", ui), patch.dict(app.PAGES, {"how": MagicMock()}), \
             patch.object(app, "vet_candidate", side_effect=ValueError), \
             patch.object(app, "target_diagnostic", return_value=curve), patch.object(app, "chart_card") as charts:
            app.render_explore({"targets": [{"target_id": "KIC 7"}]})
            figures = {call.args[0]: call.args[2] for call in charts.call_args_list}
            self.assertEqual(set(figures), {"curve", "fold", "bls", "raw"})
            np.testing.assert_array_equal(figures["raw"].data[0].x, curve["time"])
            np.testing.assert_array_equal(figures["raw"].data[0].y, curve["flux"])
            np.testing.assert_array_equal(figures["bls"].data[0].x, curve["periods"])
            np.testing.assert_array_equal(figures["bls"].data[0].y, curve["power"])

    def test_explore_diagnostics_empty_and_error(self):
        import app

        ui = MagicMock()
        ui.columns.side_effect = lambda n, **kw: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
        ui.session_state = {"explore_target": "KIC 7"}
        ui.selectbox.return_value = "KIC 7"
        ui.button.return_value = False
        with patch.object(app, "st", ui), patch.object(app, "target_diagnostic") as diagnostic, \
             patch.object(app, "chart_card") as charts:
            app.render_explore({"targets": []})
            diagnostic.assert_not_called()
            ui.info.assert_called_once()
            for error in (ValueError, OSError, RuntimeError):
                diagnostic.side_effect = error("sensitive details")
                app.render_explore({"targets": [{"target_id": "KIC 7"}]})
                ui.warning.assert_called_with(f"This star's diagnostic is unavailable: {error.__name__}")
            charts.assert_not_called()

    def test_mission_host_count_and_explicit_holdout_metrics(self):
        import app

        start = {"balanced_accuracy": 0.7, "recovery_rate": 0.5, "false_alarm_rate": 0.1, "runtime_s": 3.2}
        final = {"balanced_accuracy_ci_low": 0.6, "balanced_accuracy_ci_high": 0.8,
                 "runtime_s": 3.2, "balanced_accuracy": 0.7, "false_alarm_rate": 0.1, "recovery_rate": 0.5}
        view = {"events": [], "current": [], "summary": None,
                "baseline": {"data_source": "real", "provenance": {"host_count": 40}, "measurements": start},
                "holdout": {"data_source": "real", "measurements": final}}
        with patch.object(app, "st") as ui, patch.object(app, "chart_card"):
            app.render_run_results(view, {"loop_id": "test"}, "complete")
            ui.caption.assert_any_call("Stars analyzed: 40")
            frame = ui.dataframe.call_args_list[1].args[0]
            row = frame.iloc[0]
            self.assertEqual(row["Test signals recovered"], "— (50%)")
            self.assertEqual(row["False alarms on quiet stars"], "— (10%)")
            self.assertEqual(row["Runtime"], "3.2 s")
            ui.caption.assert_any_call("Final test, selected method: balanced accuracy 0.7000, 95% interval "
                                       "[0.6000, 0.8000] (bootstrap over stars).")
            view["baseline"]["provenance"] = {}
            view["holdout"] = {"data_source": "placeholder", "measurements": {}}
            ui.dataframe.reset_mock()
            app.render_run_results(view, {"loop_id": "test"}, "complete")
            ui.caption.assert_any_call("Stars analyzed: 0")
            row = ui.dataframe.call_args_list[1].args[0].iloc[0]
            self.assertEqual(list(row)[1:], ["—", "—", "—"])

    def test_research_selected_run_provenance(self):
        import app

        events = [{"loop_id": source, "step": "baseline", "timestamp": f"2026-01-0{i}T00:00:00+00:00",
                   "payload": {"data_source": source, "measurements": {"balanced_accuracy": 0.5}}}
                  for i, source in enumerate(("real", "placeholder"), 1)]
        state = {"events": events, "records": [], "holdout": []}
        for source in ("real", "placeholder"):
            for section in ("Overview", "Activity", "Results"):
                ui = MagicMock()
                ui.session_state = {"research_run": source}
                ui.selectbox.return_value = source
                ui.columns.side_effect = lambda n, **kw: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
                ui.segmented_control.return_value = section
                worker = MagicMock()
                worker.status.return_value = {"running": False}
                with self.subTest(source=source, section=section), patch.object(app, "st", ui), \
                     patch.object(app, "render_overview"), patch.object(app, "render_activity"), \
                     patch.object(app, "render_run_results"):
                    app.render_research(state, worker)
                    captions = " ".join(c.args[0] for c in ui.caption.call_args_list)
                    if source == "placeholder":
                        ui.warning.assert_any_call("Stand-in engine (placeholder) run")
                        self.assertNotIn(app.BENCHMARK_DISCLOSURE, captions)
                        self.assertNotIn("signals added to real Kepler observations", str(ui.markdown.call_args_list))
                    else:
                        ui.caption.assert_any_call(app.BENCHMARK_DISCLOSURE)
        with patch.object(app, "st") as ui:
            app.render_run_source({"baseline": None})
            ui.caption.assert_called_with("Data source not recorded for this run.")

    def test_explore_known_binaries_and_source_guards(self):
        import app

        targets = app.explore_targets({"targets": [{"target_id": "KIC 7"}]})
        for kepid in (10661783, 4851217, 4544587):
            self.assertIn("known eclipsing binary", app.target_label(targets[f"KIC {kepid}"]))
        self.assertEqual(app.explore_targets({"targets": []}), {})
        with patch.object(app, "cache_path") as cache:
            cache.return_value.exists.return_value = False
            self.assertEqual(list(app.explore_targets({"targets": [{"target_id": "KIC 7"}]})), ["KIC 7"])
        with patch.object(app, "read_curve", return_value={"origin": "synthetic"}):
            with self.assertRaisesRegex(ValueError, "real source photometry"):
                app.target_diagnostic.__wrapped__("KIC 10661783")
        with patch.object(app, "read_curve", return_value={"origin": "real", "time": np.array([]), "flux": np.array([])}):
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                app.target_diagnostic.__wrapped__("KIC 10661783")

    def test_jsonl_missing_partial_and_valid_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            self.assertEqual(read_jsonl(path), [])
            path.write_text(json.dumps({"step": "baseline"}) + '\n{"partial":')
            self.assertEqual(read_jsonl(path), [{"step": "baseline"}])

    def test_snapshot_invalid_storage_and_memory(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("zathura.dashboard.ROOT", Path(directory)), \
             patch("zathura.dashboard.cached_targets", return_value=[]), \
             patch("zathura.dashboard.storage.load_records", side_effect=ValueError), \
             patch("zathura.dashboard.LocalBackend.recall_memory", side_effect=ValueError):
            state = snapshot()
        self.assertEqual(state["records"], [])
        self.assertEqual(state["memories"], [])
        self.assertIsNotNone(state["warning"])

    def test_mission_thread_guard_callback_and_error_redaction(self):
        entered, finish = threading.Event(), threading.Event()

        def loop(iterations, on_event):
            self.assertEqual(iterations, 2)
            on_event({"payload": {"iteration": 1}})
            entered.set()
            finish.wait(5)
            raise RuntimeError("sensitive transport details")

        worker = Mission()
        with patch("zathura.dashboard.run_research_loop", side_effect=loop):
            self.assertTrue(worker.launch(2))
            self.assertTrue(entered.wait(5))
            self.assertFalse(worker.launch(2))
            self.assertEqual(worker.status()["iteration"], 1)
            threads = [t for t in threading.enumerate() if t.name == "zathura-mission"]
            finish.set()
            for thread in threads:
                thread.join(5)
        self.assertEqual(worker.status(), {"running": False, "iteration": 1, "rounds": 2, "error": "RuntimeError"})

    def test_cached_targets_empty_without_catalog(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("zathura.engine.data.CATALOG_PATH", Path(directory) / "missing.json"):
            load_catalog.cache_clear()
            try:
                self.assertEqual(cached_targets(), [])
            finally:
                load_catalog.cache_clear()

if __name__ == "__main__":
    unittest.main()
