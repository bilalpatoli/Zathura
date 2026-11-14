import unittest
from unittest.mock import patch

import numpy as np

from zathura.engine.vetting import VERDICTS, vet_candidate


class VettingTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(0, 90, .0204)
        self.phase = (self.time - 1 + 1.5) % 3 - 1.5
        self.noise = np.random.default_rng(81).normal(0, 1e-5, len(self.time))
        self.flux = 1 + self.noise - .001 * (np.abs(self.phase) < .12)
        self.curve = {"time": self.time, "flux": self.flux, "origin": "real", "quarter": 3}
        self.result = {"time": self.time, "flat_flux": self.flux, "period_days": 3., "epoch": 1.,
                       "duration_days": .24, "depth": .001, "sde": 12., "detected": True}
        self.metadata = {"kepid": 6665695, "stellar_radius_solar": 1., "truth": "planet", "catalog_periods_days": [3.]}

    def screen(self, result=None, curve=None, metadata=None):
        with patch("zathura.engine.vetting.read_curve", return_value=self.curve if curve is None else curve), \
             patch("zathura.engine.vetting.search", return_value=self.result if result is None else result), \
             patch("zathura.engine.vetting.vetting_targets", return_value=[self.metadata if metadata is None else metadata]):
            return vet_candidate("KIC 6665695")

    def test_planet_and_truth_blindness(self):
        result = self.screen()
        self.assertEqual(result["verdict"], VERDICTS[0])
        self.assertEqual(len(result["tests"]), 5)
        self.assertTrue(all(t["passed"] is True for t in result["tests"]))
        for test in result["tests"]:
            self.assertTrue({"value", "threshold", "passed", "explanation"} <= test.keys())
        changed = self.metadata | {"truth": "EB", "catalog_periods_days": [99.]}
        self.assertEqual(self.screen(metadata=changed)["tests"], result["tests"])

    def test_each_binary_flag_and_weak_signal_precedence(self):
        cycles = np.rint((self.time - 1) / 3).astype(int)
        alternating = self.flux - .003 * ((cycles % 2 == 1) & (np.abs(self.phase) < .12))
        secondary = self.flux - .001 * (np.abs((self.time - 2.5 + 1.5) % 3 - 1.5) < .12)
        for name, change in (("odd-even", {"flat_flux": alternating}), ("secondary", {"flat_flux": secondary}), ("size", {"depth": .08})):
            with self.subTest(name=name):
                result = self.screen(result=self.result | change)
                self.assertEqual(result["verdict"], VERDICTS[1])
                self.assertEqual(self.screen(result=self.result | change | {"detected": False, "sde": 1.})["verdict"], VERDICTS[2])

    def test_v_shape_alone_is_inconclusive_and_solar_radius_is_labelled(self):
        triangle = 1 + self.noise - .001 * np.maximum(0, 1 - np.abs(self.phase) / .12)
        result = self.screen(result=self.result | {"flat_flux": triangle}, metadata=self.metadata | {"stellar_radius_solar": None})
        self.assertEqual(result["verdict"], VERDICTS[2])
        self.assertIs(result["tests"][3]["passed"], False)
        self.assertTrue(result["radius_assumed"])
        self.assertIn("assumed", result["tests"][4]["explanation"])

    def test_insufficient_odd_even_coverage_is_not_a_pass(self):
        keep = self.time < 8
        result = self.result | {"time": self.time[keep], "flat_flux": self.flux[keep]}
        screened = self.screen(result=result)
        self.assertIsNone(screened["tests"][1]["passed"])
        self.assertEqual(screened["verdict"], VERDICTS[2])

    def test_empty_short_nonfinite_unknown_and_nonreal_errors(self):
        for time in (np.array([]), np.arange(10), np.full(30, np.nan)):
            with self.subTest(time=len(time)), self.assertRaisesRegex(ValueError, "Insufficient"):
                self.screen(curve=self.curve | {"time": time, "flux": np.ones(len(time))})
        with self.assertRaisesRegex(ValueError, "real source"):
            self.screen(curve=self.curve | {"origin": "synthetic"})
        with self.assertRaisesRegex(ValueError, "Unknown"):
            vet_candidate("KIC 123")
        with self.assertRaisesRegex(ValueError, "identifier"):
            vet_candidate("not-a-star")

    def test_real_cached_planet_and_binary(self):
        for target, verdict in (("KIC 6665695", VERDICTS[0]), ("KIC 10661783", VERDICTS[1])):
            with self.subTest(target=target):
                result = vet_candidate(target)
                self.assertEqual(result["data_source"], "real")
                self.assertEqual(result["verdict"], verdict)
                self.assertGreater(result["cadences"], 1000)


if __name__ == "__main__":
    unittest.main()
