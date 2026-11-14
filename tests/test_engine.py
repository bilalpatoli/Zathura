import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import zathura.engine as engine
from zathura.engine import benchmark, data
from zathura.schema import BASELINE_CONFIG


def clear_caches():
    for cached in (data.load_catalog, data._read, benchmark.build, engine._data_provenance):
        cached.cache_clear()


class CommittedDataTests(unittest.TestCase):
    def test_catalog_splits_are_disjoint_real_kepler_quarters(self):
        clear_caches()
        catalog = data.load_catalog()
        dev, holdout = set(catalog["hosts"]["dev"]), set(catalog["hosts"]["holdout"])
        validation = {v["kepid"] for v in catalog["validation"]}
        self.assertGreaterEqual(len(dev), 30)
        self.assertEqual(len(dev), len(holdout))
        self.assertFalse(dev & holdout or dev & validation or holdout & validation)
        self.assertGreaterEqual(len(validation), 8)
        for kepid in dev | holdout | validation:
            curve = data.read_curve(kepid)
            self.assertEqual(curve["origin"], "real")
            self.assertGreater(len(curve["time"]), 2000)
            self.assertGreater(np.ptp(curve["time"]), 60)
            self.assertTrue(np.all(np.isfinite(curve["flux"])))
        for planet in catalog["validation"]:
            self.assertTrue(1 <= planet["period_days"] <= 10)
            self.assertIn(planet["period_days"], planet["host_koi_periods"])


class EngineTests(unittest.TestCase):
    """Runs on a small synthetic catalog so tests stay fast and offline."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        cache, catalog_path = root / "cache", root / "catalog.json"
        cache.mkdir()
        rng = np.random.default_rng(3)
        time = np.arange(0, 89, data.CADENCE_DAYS) + 260
        for kepid in range(1, 8):
            flux = 1 + 2e-4 * np.sin(time / 4 + kepid) + rng.normal(0, 1e-4, len(time))
            if kepid == 7:
                flux *= benchmark.transit_model(time, 3.0, 261.0, 0.12, 0.002, 0.1)
            np.savez_compressed(cache / f"kic_{kepid}.npz", time=time, flux=flux.astype(np.float32),
                                flux_err=np.full(len(time), 1e-4, np.float32), quarter=np.int16(3),
                                kepid=np.int64(kepid), origin="real", product="test.fits")
        catalog_path.write_text(json.dumps({"version": "test", "hosts": {"dev": [1, 2, 3], "holdout": [4, 5, 6]},
            "validation": [{"kepid": 7, "koi": "K1", "name": "Test-1 b", "period_days": 3.0, "epoch_bkjd": 261.0,
                            "duration_days": 0.12, "depth_ppm": 2000, "koi_model_snr": 100,
                            "host_koi_periods": [3.0]}]}))
        self.patches = [patch.object(data, "CACHE", cache), patch.object(data, "CATALOG_PATH", catalog_path),
                        patch.object(engine, "CATALOG_PATH", catalog_path), patch.object(engine, "RUNS", root / "runs"),
                        patch.object(engine, "_workers", return_value=1)]
        for p in self.patches:
            p.start()
        clear_caches()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        clear_caches()
        self.temp.cleanup()

    def test_benchmark_is_deterministic_and_well_formed(self):
        examples, truth = benchmark.build("dev")
        self.assertEqual(len(examples), 3 * (benchmark.INJECTIONS_PER_HOST + benchmark.CONTROLS_PER_HOST))
        first = benchmark.fingerprint("dev")
        benchmark.build.cache_clear()
        self.assertEqual(benchmark.fingerprint("dev"), first)
        self.assertNotEqual(benchmark.fingerprint("holdout"), first)
        for info in truth:
            if info["injected"]:
                self.assertTrue(benchmark.SNR_RANGE[0] <= info["snr"] <= benchmark.SNR_RANGE[1])
                self.assertGreaterEqual(info["transits"], benchmark.MIN_TRANSITS)
                self.assertTrue(benchmark.PERIOD_RANGE[0] <= info["period"] <= benchmark.PERIOD_RANGE[1])
        self.assertEqual({t["kepid"] for t in benchmark.build("holdout")[1]}, {4, 5, 6})

    def test_scoring_requires_period_and_epoch(self):
        truth = {"injected": True, "period": 4.0, "epoch": 1.0, "duration": 0.1}
        hit = {"detected": True, "period_days": 4.02, "epoch": 9.03}
        self.assertTrue(benchmark.score(hit, truth)["recovered"])
        self.assertFalse(benchmark.score(hit | {"epoch": 3.0}, truth)["recovered"])
        self.assertFalse(benchmark.score(hit | {"detected": False}, truth)["recovered"])
        double = benchmark.score(hit | {"period_days": 8.0}, truth)
        self.assertFalse(double["recovered"])
        self.assertTrue(double["harmonic"])
        self.assertTrue(benchmark.score({"detected": True}, {"injected": False})["detected"])

    def test_runs_evaluation_provenance_and_truth_isolation(self):
        with self.assertRaises(ValueError):
            engine.run_experiment(BASELINE_CONFIG, "bad")
        with self.assertRaises(ValueError):
            engine.get_experiment_results("../bad")
        with self.assertRaises(FileNotFoundError):
            engine.get_experiment_results("a" * 32)
        baseline_id = engine.run_baseline("dev")
        baseline = engine.get_experiment_results(baseline_id)
        self.assertTrue(baseline["is_baseline"])
        self.assertEqual(baseline["data_source"], "real")
        repeat_id = engine.run_experiment(BASELINE_CONFIG, "dev")
        repeat = engine.get_experiment_results(repeat_id)
        for key in ("recovery_rate", "false_alarm_rate", "balanced_accuracy", "harmonic_rate"):
            self.assertEqual(repeat["measurements"][key], baseline["measurements"][key])
        m = repeat["measurements"]
        self.assertEqual((m["positive_count"], m["negative_count"]), (12, 3))
        self.assertLessEqual(m["balanced_accuracy_ci_low"], m["balanced_accuracy"])
        self.assertGreaterEqual(m["balanced_accuracy_ci_high"], m["balanced_accuracy"])
        self.assertTrue(all(isinstance(v, float) for v in m.values()))
        self.assertTrue(any(k.startswith("recovery_snr_") for k in m))
        provenance = repeat["provenance"]
        for key in ("code_commit", "packages", "catalog_sha256", "curves_sha256", "truth_sha256", "seed"):
            self.assertIn(key, provenance)
        self.assertEqual(provenance["truth_sha256"], benchmark.fingerprint("dev"))
        evaluation = engine.evaluate_experiment(repeat_id)
        self.assertEqual(evaluation["delta"], 0)
        self.assertFalse(evaluation["improved_vs_best"])
        self.assertFalse(evaluation["significant_vs_baseline"])
        self.assertEqual(evaluation["delta_vs_baseline_ci_low"], 0)
        blind_id = engine.run_experiment(BASELINE_CONFIG.model_copy(update={"detection_threshold": 100.0}), "dev")
        blind = engine.evaluate_experiment(blind_id)
        self.assertEqual(engine.get_experiment_results(blind_id)["measurements"]["recovery_rate"], 0)
        self.assertLessEqual(blind["delta_vs_baseline_ci_high"], blind["delta_vs_baseline_ci_low"] + 1)
        self.assertFalse(blind["improved_vs_best"])
        text = json.dumps(engine.get_experiment_results(blind_id))
        for secret in ("snr\"", "epoch", "injected", "\"period\""):
            self.assertNotIn(secret, text)
        holdout_id = engine.run_experiment(BASELINE_CONFIG, "holdout")
        with self.assertRaises(ValueError):
            engine.evaluate_experiment(holdout_id)

    def test_short_curve_diagnostic(self):
        curve = engine.get_lightcurve("KIC 7")
        for count in (0, 2, 9):
            with self.subTest(count=count), patch.object(engine, "get_lightcurve", return_value=curve | {
                    "time": curve["time"][:count], "flux": curve["flux"][:count]}):
                with self.assertRaisesRegex(ValueError, "Insufficient finite"):
                    engine.inspect_target("KIC 7")

    def test_known_planet_validation_and_targets(self):
        report = engine.validate_known_planets()
        self.assertEqual((report["recovered"], report["total"]), (1, 1))
        self.assertAlmostEqual(report["planets"][0]["found_period_days"], 3.0, delta=0.03)
        targets = engine.load_targets()
        self.assertEqual([t["role"] for t in targets].count("validation"), 1)
        self.assertEqual(len(engine.cached_targets()), 7)
        with self.assertRaises(ValueError):
            engine.get_lightcurve("KIC 99")
        with self.assertRaises(ValueError):
            engine.get_lightcurve("../bad")
        result = engine.inspect_target("KIC 7")
        self.assertAlmostEqual(result["period_days"], 3.0, delta=0.03)
        self.assertEqual(result["known_periods"], [3.0])
        self.assertTrue(result["in_transit"].any())
        self.assertEqual(len(result["flat_flux"]), len(result["time"]))


if __name__ == "__main__":
    unittest.main()
