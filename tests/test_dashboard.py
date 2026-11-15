import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from zathura.dashboard import Mission, read_jsonl, snapshot
from zathura.engine import cached_targets
from zathura.engine.data import load_catalog


class DashboardTests(unittest.TestCase):
    def test_mission_host_count_and_explicit_holdout_metrics(self):
        import app

        state = {"records": [], "current": [], "targets": [{}] * 96, "latest": [],
                 "baseline": {"data_source": "real", "provenance": {"host_count": 40},
                              "measurements": {"balanced_accuracy": 0.7, "recovery_rate": 0.5,
                                               "false_alarm_rate": 0.1, "runtime_s": 3.2}},
                 "holdout": [{"data_source": "real", "measurements": {
                     "balanced_accuracy_ci_low": 0.6, "balanced_accuracy_ci_high": 0.8,
                     "runtime_s": 3.2, "balanced_accuracy": 0.7, "false_alarm_rate": 0.1,
                     "recovery_rate": 0.5}}]}
        def columns(n, **kwargs):
            result = [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
            for col in result:
                col.__enter__.return_value = col
            return result

        ui = MagicMock()
        ui.columns.side_effect = columns
        ui.button.return_value = False
        worker = MagicMock()
        worker.status.return_value = {"running": False, "iteration": 0, "error": None}
        with patch.object(app, "st", ui), patch.object(app, "metric_cards") as cards:
            app.render_mission(state, worker)
            self.assertIn(("Stars analyzed", 40), cards.call_args_list[0].args[0])
            app.render_results(state)
            self.assertEqual(cards.call_args.args[0], [
                ("Recovery rate", "0.5000"), ("False alarm rate", "0.1000"),
                ("Balanced accuracy", "0.7000"), ("Runtime s", "3.2000")])
            ui.caption.assert_any_call("Balanced accuracy 95% confidence interval: [0.6000, 0.8000]")
            state["baseline"]["provenance"] = {}
            cards.reset_mock()
            app.render_mission(state, worker)
            self.assertIn(("Stars analyzed", 0), cards.call_args_list[0].args[0])
            state["holdout"] = [{"data_source": "placeholder", "measurements": {}}]
            app.render_results(state)
            self.assertTrue(all(value == "—" for _, value in cards.call_args.args[0]))

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
