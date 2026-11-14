import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

import zathura.engine as engine
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

    def test_engine_fallback_cache_evaluation_storage(self):
        with patch.object(engine, "CACHE", self.root / "cache"), patch.object(engine, "RUNS", self.root / "runs"), \
             patch.object(engine.subprocess, "run", side_effect=subprocess.TimeoutExpired("download", 25)) as download:
            with self.assertWarns(UserWarning):
                targets = engine.load_targets()
            self.assertEqual(download.call_count, 6)
            self.assertEqual(len(targets), 6)
            engine.load_targets()
            self.assertEqual(download.call_count, 6)
            curve = engine.get_lightcurve("Kepler-69")
            self.assertEqual(curve["curve_origin"], "synthetic_placeholder")
            self.assertNotIn("injected", curve)
            with self.assertRaises(ValueError):
                engine.get_lightcurve("../bad")
            with self.assertRaises(ValueError):
                engine.run_experiment(BASELINE_CONFIG, "bad")
            with self.assertRaises(ValueError):
                engine.get_experiment_results("../bad")
            with self.assertRaises(FileNotFoundError):
                engine.get_experiment_results("a" * 32)
            baseline_id = engine.run_baseline("dev")
            baseline = engine.get_experiment_results(baseline_id)
            run_id = engine.run_experiment(BASELINE_CONFIG, "dev")
            result = engine.get_experiment_results(run_id)
            for key in ("recovery_rate", "false_alarm_rate", "balanced_accuracy"):
                self.assertEqual(result["measurements"][key], baseline["measurements"][key])
            self.assertLess(result["measurements"]["runtime_s"], 30)
            self.assertEqual(result["measurements"]["positive_count"], 24)
            self.assertEqual(result["measurements"]["negative_count"], 12)
            evaluation = engine.evaluate_experiment(run_id)
            self.assertEqual(evaluation["delta"], 0)
            self.assertFalse(evaluation["improved_vs_best"])
            holdout_id = engine.run_experiment(BASELINE_CONFIG, "holdout")
            with self.assertRaises(ValueError):
                engine.evaluate_experiment(holdout_id)
            self.assertNotIn("true_period", json.dumps(engine.get_experiment_results(holdout_id)))
            hypothesis = research_hypothesis([], result["measurements"])
            record = ExperimentRecord(
                experiment_id=run_id, hypothesis=hypothesis["hypothesis"], cited_sources=hypothesis["cited_sources"],
                proposed_config=BASELINE_CONFIG, reviewer_decision={"decision": "revise", "rationale": "not reviewed"},
                measurements=result["measurements"], runtime_s=result["measurements"]["runtime_s"],
                evaluation_result=evaluation, accepted=False, data_source="placeholder")
            log = self.root / "experiments.jsonl"
            self.assertEqual(load_records(log), [])
            save_record(record, log)
            self.assertEqual(load_records(log), [record])
            log.write_text("invalid json\n")
            with self.assertRaises(ValidationError):
                load_records(log)


if __name__ == "__main__":
    unittest.main()
