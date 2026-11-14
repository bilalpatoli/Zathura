import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

import zathura.agents.memory as memory
from zathura.agents.exa_search import search_papers
from zathura.agents.llm import complete_json
from zathura.agents.scientist import research_hypothesis
from zathura.schema import BASELINE_CONFIG, DetectionConfig, ExperimentRecord
from zathura.storage import load_records, save_record


class SliceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"ZATHURA_MOCK": "1", "PASTDEV_API_KEY": ""})
        self.env.start()
        self.memory = patch.object(memory, "MEMORY_PATH", self.root / "memory.jsonl")
        self.memory.start()

    def tearDown(self):
        self.env.stop()
        self.memory.stop()
        self.temp.cleanup()

    def test_config_bounds(self):
        self.assertEqual(len(DetectionConfig.model_fields), 6)
        for invalid in ({"detrend_window": 102}, {"detrend_window": 101.0}, {"period_grid_size": 999},
                        {"outlier_sigma": 11}, {"detection_threshold": float("nan")},
                        {"duration_min_days": 0.4, "duration_max_days": 0.3}, {"extra": 1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                DetectionConfig(**invalid)

    def test_scientist_empty_history_and_untried(self):
        first = research_hypothesis([], {"recovery_rate": 0.5})
        config = first["proposed_config"].model_dump()
        self.assertEqual(sum(config[k] != v for k, v in BASELINE_CONFIG.model_dump().items()), 1)
        self.assertTrue(all(s["mock"] for s in first["cited_sources"]))
        second = research_hypothesis([{"proposed_config": config, "accepted": False}], {})
        self.assertNotEqual(second["proposed_config"].model_dump(), config)
        best = BASELINE_CONFIG.model_dump() | {"outlier_sigma": 8.0}
        third = research_hypothesis([], {"current_best_config": best})
        self.assertEqual(third["proposed_config"].outlier_sigma, 8)
        narrow = BASELINE_CONFIG.model_dump() | {"duration_min_days": 0.4, "duration_max_days": 0.5}
        fourth = research_hypothesis([], {"current_best_config": narrow})
        self.assertEqual(fourth["proposed_config"].duration_min_days, 0.4)
        accepted = [{"proposed_config": best, "accepted": True, "measurements": {"balanced_accuracy": 0.9}}]
        fifth = research_hypothesis(accepted, {"balanced_accuracy": 0.5})
        self.assertEqual(fifth["proposed_config"].outlier_sigma, 8)

    def test_client_and_scientist_errors(self):
        with self.assertRaises(ValueError):
            search_papers("", 3)
        with self.assertRaises(ValueError):
            search_papers("transits", 0)
        with self.assertRaises(ValueError):
            complete_json({})
        with patch("zathura.agents.scientist.complete_json", return_value={"proposed_config": BASELINE_CONFIG.model_dump()}):
            with self.assertRaises(ValueError):
                research_hypothesis([], {})

    def test_record_storage_round_trip(self):
        hypothesis = research_hypothesis([], {"recovery_rate": 0.5})
        record = ExperimentRecord(
            experiment_id="a" * 32, hypothesis=hypothesis["hypothesis"], cited_sources=hypothesis["cited_sources"],
            proposed_config=BASELINE_CONFIG, reviewer_decision={"decision": "revise", "rationale": "not reviewed"},
            measurements={"balanced_accuracy": 0.5}, runtime_s=1.0,
            evaluation_result={"improved_vs_baseline": False, "improved_vs_best": False, "delta": 0.0},
            accepted=False, data_source="real")
        log = self.root / "experiments.jsonl"
        self.assertEqual(load_records(log), [])
        save_record(record, log)
        self.assertEqual(load_records(log), [record])
        log.write_text("invalid json\n")
        with self.assertRaises(ValidationError):
            load_records(log)

if __name__ == "__main__":
    unittest.main()
