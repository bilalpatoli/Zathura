import unittest

from zathura.story import measured_chain, memory_veto, safe


class StoryTests(unittest.TestCase):
    def test_only_recorded_deterministic_vetoes_are_presented(self):
        proposal = {"memory_influence": {"memory_ids": ["prior"], "explanation": "Deterministic memory veto changed the first choice"}}
        claimed = {"memory_influence": {"memory_ids": ["claim"], "explanation": "The model says memory helped"}}
        events = [{"step": "hypothesis", "payload": proposal}, {"step": "revision", "payload": claimed}]
        self.assertIs(memory_veto(events), proposal)
        self.assertIsNone(memory_veto(events[1:]))

    def test_empty_and_unrelated_payloads(self):
        self.assertIsNone(memory_veto([]))
        self.assertIsNone(memory_veto([{"step": "targets", "payload": []}]))
        self.assertIsNone(memory_veto([{"step": "hypothesis", "payload": {}}]))

    def test_story_text_escapes_markup_and_hides_other_ids(self):
        self.assertEqual(safe("<b>" + "a" * 32 + "</b>"), "&lt;b&gt;an earlier experiment&lt;/b&gt;")

    def test_debate_uses_latest_measurement_without_scoring_rejections(self):
        measured = {"completion": {"measurements": {"balanced_accuracy": .4}}}
        rejected = {"completion": {"measurements": {}}}
        self.assertIs(measured_chain([measured, rejected]), measured)
        self.assertIs(measured_chain([rejected]), rejected)
        self.assertIsNone(measured_chain([]))


if __name__ == "__main__":
    unittest.main()
