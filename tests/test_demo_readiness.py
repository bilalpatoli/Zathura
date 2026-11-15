"""Readiness regressions on the merged five-page frontend."""
import unittest
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

import app
from zathura.runs import describe, outcome


class Worker:
    def __init__(self, accept=True):
        self.running = False
        self.accept = accept
        self.calls = []

    def status(self):
        return {"running": self.running, "iteration": 0, "rounds": 3, "error": None}

    def launch(self, rounds):
        self.calls.append(rounds)
        if self.running or not self.accept:
            return False
        self.running = True
        return True


class ReadinessTests(unittest.TestCase):
    def test_launch_disables_button_and_duplicate_is_reported(self):
        def page(worker):
            import app
            app.setup_area(worker, {"targets": []})

        worker = Worker()
        ui = AppTest.from_function(page, args=(worker,)).run()
        ui.button(key="research_start").click().run()
        self.assertFalse(ui.exception)
        self.assertTrue(ui.button(key="research_start").disabled)
        self.assertEqual(worker.calls, [3])
        ui.run()
        self.assertEqual(worker.calls, [3])
        blocked = Worker(accept=False)
        ui = AppTest.from_function(page, args=(blocked,)).run()
        ui.button(key="research_start").click().run()
        self.assertTrue(any("did not start another run" in w.value for w in ui.warning))

    def test_incomplete_run_is_interrupted_but_active_run_is_running(self):
        run = {"loop_id": "one", "status": "incomplete"}
        self.assertEqual(app.run_status(run, {"running": False}, [run]), "interrupted")
        self.assertEqual(app.run_status(run, {"running": True}, [run]), "running")
        self.assertEqual(outcome({"holdout": None}, "interrupted")[1], "Interrupted")

    def test_significance_labels_use_recorded_evaluation(self):
        for delta in (.006, -.006):
            view = {"current": [{"accepted": True}], "holdout": {
                "evaluation_result": {"delta": delta, "significant_vs_baseline": False}}}
            self.assertIn("within noise", outcome(view, "complete")[1])
            self.assertIn("not statistically significant", outcome(view, "complete")[2])
        event = {"step": "iteration_complete", "payload": {"iteration": 1, "accepted": True,
                 "measurements": {"balanced_accuracy": .5},
                 "record": {"evaluation_result": {"significant_vs_best": False}}}}
        self.assertIn("within noise", describe(event)[0])
        event["payload"]["record"]["evaluation_result"]["significant_vs_best"] = True
        self.assertNotIn("within noise", describe(event)[0])

    def test_final_test_uses_status_and_significance(self):
        view = {"baseline": {"measurements": {"balanced_accuracy": .5}},
                "current": [], "summary": None, "holdout": None}
        with patch.object(app, "st") as ui, patch.object(app, "chart_card"):
            app.render_run_results(view, {"loop_id": "one"}, "interrupted")
            ui.write.assert_any_call("This run was interrupted before the final test; no held-out result was recorded.")
            view["holdout"] = {"measurements": {"balanced_accuracy": .506},
                               "evaluation_result": {"delta": .006, "significant_vs_baseline": False}}
            app.render_run_results(view, {"loop_id": "one"}, "complete")
            ui.caption.assert_any_call("The difference is within noise (not statistically significant).")


if __name__ == "__main__":
    unittest.main()
