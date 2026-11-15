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
