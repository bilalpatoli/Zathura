import importlib.util
import unittest
from pathlib import Path

import numpy as np

from zathura.engine.benchmark import canonical_truth

spec = importlib.util.spec_from_file_location("reproduce", Path(__file__).resolve().parents[1] / "scripts" / "reproduce.py")
reproduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduce)


def report(sha="abc", recovery=0.5, ci=0.4, sde=20.0, detected=True):
    run = {"truth_sha256": sha, "curves_sha256": "c", "measurements": {"recovery_rate": recovery, "balanced_accuracy_ci_low": ci}}
    return {"benchmark_version": "v", "configs": {"baseline": {"config": {"x": 1}, "dev": run, "holdout": run}},
            "known_planets": {"baseline": {"planets": [{"target_id": "KIC 1", "sde": sde, "detected": detected}]}}}


class ReproducibilityTests(unittest.TestCase):
    def test_truth_hash_ignores_last_bit_float_noise_but_not_real_changes(self):
        row = {"host": np.int64(1), "kepid": 7, "injected": np.bool_(True), "period": 3.123456789012, "transits": 5}
        noisy = row | {"period": np.nextafter(row["period"], 4.0)}
        self.assertEqual(canonical_truth([row]), canonical_truth([noisy]))
        self.assertNotEqual(canonical_truth([row]), canonical_truth([row | {"period": 3.1235}]))
        self.assertIs(canonical_truth([row])[0]["injected"], True)

    def test_exact_fields_stay_exact_and_floats_use_tolerance(self):
        self.assertEqual(reproduce.compare(report(ci=0.4 + 1e-12, sde=20.0 + 3.6e-9), report()), [])
        self.assertTrue(reproduce.compare(report(recovery=0.5 + 1e-12), report()))
        self.assertTrue(reproduce.compare(report(sha="other"), report()))
        self.assertTrue(reproduce.compare(report(detected=False), report()))
        self.assertTrue(reproduce.compare(report(sde=20.1), report()))

    def test_missing_reference_label_fails(self):
        reference = report()
        reference["configs"] = {}
        self.assertTrue(reproduce.compare(report(), reference))


if __name__ == "__main__":
    unittest.main()
