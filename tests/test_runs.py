import unittest

from zathura.runs import counts, describe, outcome, plain_result, run_list, run_view, selected_measurements


def event(loop, step, payload=None, ts="2026-01-01T00:00:00+00:00"):
    return {"loop_id": loop, "step": step, "payload": payload or {}, "timestamp": ts}


BASE = {"balanced_accuracy": 0.5, "recovery_rate": 0.4, "false_alarm_rate": 0.4, "positive_count": 160.0, "negative_count": 40.0}
BETTER = BASE | {"balanced_accuracy": 0.6}


def state(events, records=(), holdout=()):
    return {"events": events, "records": list(records), "holdout": list(holdout)}


class RunTests(unittest.TestCase):
    def test_runs_are_separated_and_statuses_come_from_recorded_events(self):
        events = [event("a", "targets", ts="2026-01-01T00:00:00+00:00"), event("a", "iteration_complete", {"iteration": 1}),
                  event("a", "loop_complete", ts="2026-01-01T00:05:00+00:00"),
                  event("b", "targets", ts="2026-01-02T00:00:00+00:00"), event("b", "error", {"iteration": 1})]
        runs = run_list(events)
        self.assertEqual([r["loop_id"] for r in runs], ["b", "a"])
        self.assertEqual([r["status"] for r in runs], ["stopped", "complete"])
        self.assertEqual(runs[1]["rounds"], 1)

    def test_outcomes(self):
        record = {"experiment_id": "x", "accepted": True, "measurements": BETTER}
        events = [event("a", "baseline", {"measurements": BASE}),
                  event("a", "iteration_complete", {"iteration": 1, "record": {"experiment_id": "x"}})]
        view = run_view(state(events, [record], [{"loop_id": "a", "measurements": BETTER, "evaluation_result": {"delta": 0.1}}]), "a")
        self.assertEqual(outcome(view, "complete")[0], "improved")
        view["holdout"]["evaluation_result"]["delta"] = -0.02
        self.assertEqual(outcome(view, "complete")[0], "worse")
        self.assertEqual(outcome(view, "stopped")[0], "stopped")
        self.assertEqual(outcome(view, "running")[0], "running")
        view["current"][0]["accepted"] = False
        self.assertEqual(outcome(view, "complete")[0], "unchanged")

    def test_selected_method_never_below_start(self):
        worse = {"experiment_id": "x", "accepted": False, "measurements": BASE | {"balanced_accuracy": 0.3}}
        events = [event("a", "baseline", {"measurements": BASE}),
                  event("a", "iteration_complete", {"iteration": 1, "record": {"experiment_id": "x"}})]
        self.assertIs(selected_measurements(run_view(state(events, [worse]), "a")), BASE)

    def test_counts_use_recorded_denominators(self):
        self.assertEqual(counts(BASE), {"recovery_rate": (64, 160), "false_alarm_rate": (16, 40)})
        self.assertEqual(plain_result(BASE), "Found 64 of 160 test signals, with 16 false alarms on 40 quiet stars.")
        self.assertEqual(plain_result({"recovery_rate": 0.5, "false_alarm_rate": 0.1}),
                         "Found 50% of test signals, with a 10% false-alarm rate.")
        self.assertEqual(plain_result({}), "No measurement recorded.")

    def test_descriptions_are_based_on_recorded_events(self):
        title, _ = describe(event("a", "hypothesis", {"iteration": 2, "current_best_config": {"detrend_window": 301},
                                                      "proposed_config": {"detrend_window": 101}}))
        self.assertEqual(title, "Round 2 · Proposed a change: detrend_window: 301 → 101")
        self.assertEqual(describe(event("a", "iteration_complete", {"iteration": 1, "measurements": {}}))[0], "Round 1 · Not tested")
        self.assertIsNone(describe(event("a", "unknown_step")))


if __name__ == "__main__":
    unittest.main()
