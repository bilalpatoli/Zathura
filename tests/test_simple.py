import unittest
from unittest.mock import MagicMock, patch

from zathura import simple


def event(step, payload, loop_id="current"):
    return {"step": step, "payload": payload, "loop_id": loop_id}


def state():
    return {"baseline": {"measurements": {"balanced_accuracy": .4}, "data_source": "real"},
            "latest": [], "holdout": [], "targets": []}


class SimpleTests(unittest.TestCase):
    def test_rejected_attempts_have_no_score_and_best_never_falls(self):
        data = state()
        for number, score in enumerate((.45, None, .42), 1):
            data["latest"] += [event("hypothesis", {"iteration": number}),
                               event("iteration_complete", {"iteration": number,
                                     "measurements": {} if score is None else {"balanced_accuracy": score}})]
        self.assertEqual(simple.score_history(data), (.4, [.45, None, .42], [.45, .45, .45]))
        self.assertEqual([r["before"] for r in simple.attempt_rows(data)], [.4, .45, .45])

    def test_revisions_are_one_attempt_with_the_final_review(self):
        data = state()
        data["latest"] = [event("hypothesis", {"iteration": 1}),
                          event("review", {"iteration": 1, "decision": "reject"}),
                          event("revision", {"iteration": 1}),
                          event("review_revision", {"iteration": 1, "decision": "approve"}),
                          event("iteration_complete", {"iteration": 1, "measurements": {"balanced_accuracy": .5}})]
        rows = simple.attempt_rows(data)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review"]["decision"], "approve")
        self.assertEqual(rows[0]["after"], .5)

    def test_empty_and_missing_baseline(self):
        data = state() | {"baseline": None}
        self.assertEqual(simple.score_history(data), (None, [], []))
        self.assertIsNone(simple.latest_holdout(data))
        data["latest"] = [event("hypothesis", {"iteration": 1})]
        self.assertEqual(simple.score_history(data), (None, [None], [None]))

    def test_never_use_a_previous_missions_holdout(self):
        data = state()
        old = {"loop_id": "old", "measurements": {"balanced_accuracy": .9}}
        current = {"loop_id": "current", "measurements": {"balanced_accuracy": .3}}
        data["holdout"] = [current, old]
        self.assertIsNone(simple.latest_holdout(data))
        data["latest"] = [event("baseline", {})]
        self.assertIs(simple.latest_holdout(data), current)
        data["latest"] = [event("baseline", {}, "new")]
        self.assertIsNone(simple.latest_holdout(data))

    def test_find_missing_cache_and_diagnostic_error(self):
        ui = MagicMock()
        with patch.object(simple, "st", ui), patch.object(simple, "candidate_view") as candidate:
            simple.render_find(state())
            candidate.assert_not_called()
            with patch.object(simple, "vetting_targets", return_value=[]):
                simple.render_find(state() | {"targets": [{}]})
                candidate.assert_not_called()
            with patch.object(simple, "vetting_targets", return_value=[{"kepid": 7, "truth": "planet", "name": "Example"}]), \
                 patch.object(simple, "cache_path") as path:
                path.return_value.exists.return_value = True
                for error in (ValueError, OSError, RuntimeError):
                    candidate.side_effect = error("private details")
                    simple.render_find(state() | {"targets": [{}]})
                    ui.info.assert_called_with("We could not read enough brightness data for this star. Try another example.")

    def test_memory_claim_requires_a_deterministic_veto(self):
        for explanation, highlighted in (("Deterministic memory veto changed the first choice", True),
                                          ("The model says memory helped", False)):
            data = state()
            data["latest"] = [event("hypothesis", {"iteration": 1, "memory_influence": {
                "memory_ids": ["not-for-display"], "explanation": explanation}})]
            ui, worker = MagicMock(), MagicMock()
            worker.status.return_value = {"running": False, "error": None}
            ui.button.return_value = False
            ui.columns.return_value = [MagicMock() for _ in range(4)]
            with self.subTest(explanation=explanation), patch.object(simple, "st", ui):
                simple.render_improve(data, worker)
                self.assertEqual(ui.success.called, highlighted)

    def test_validation_missing_or_invalid_report(self):
        with patch.object(simple, "st") as ui, patch.object(simple, "ROOT") as root:
            for value in ("not json", "{}"):
                root.__truediv__.return_value.read_text.return_value = value
                simple.render_validation()
                ui.info.assert_called_with("The recorded vetting validation report is unavailable.")


if __name__ == "__main__":
    unittest.main()
