import unittest

from zathura.agents.memory import memory_receipts
from zathura.evidence import config_change, memory_activity, proposal_chains


class EvidenceTests(unittest.TestCase):
    def test_empty_and_legacy_receipts(self):
        self.assertEqual(proposal_chains([]), [])
        self.assertEqual(memory_activity([], []), [])
        self.assertEqual(memory_receipts([]), [])
        receipt = memory_receipts([{"memory_id": "old", "agent_id": "skeptic"}])[0]
        self.assertEqual(receipt["read_via"], "not_recorded")
        self.assertIsNone(receipt["read_at"])
        self.assertEqual(config_change({}, {}), "Configuration change not recorded")

    def test_revision_keeps_search_chain_separate(self):
        events = [{"step": "hypothesis", "payload": {"iteration": 1, "hypothesis": "first"}},
                  {"step": "revision", "payload": {"iteration": 1, "hypothesis": "second"}},
                  {"step": "iteration_complete", "payload": {"iteration": 1, "decision": "reject", "measurements": {}}}]
        first, second = proposal_chains(events)
        self.assertTrue(first["superseded"])
        self.assertIsNone(first["completion"])
        self.assertTrue(second["revision"])
        self.assertEqual(second["completion"]["decision"], "reject")
        self.assertEqual(config_change({"sigma": 5}, {"sigma": 4}), "sigma: 5 → 4")

    def test_pending_or_failed_iteration_has_no_invented_result(self):
        events = [{"step": "hypothesis", "payload": {"iteration": 2}},
                  {"step": "error", "payload": {"iteration": 2, "error_type": "ValueError"}}]
        self.assertIsNone(proposal_chains(events)[0]["completion"])
        self.assertEqual(memory_activity(events, []), [])

    def test_missing_write_receipt_does_not_attribute_result_to_orchestrator(self):
        events = [{"step": "iteration_complete", "agent": "orchestrator", "timestamp": "01",
                   "payload": {"result_memory_id": "missing"}}]
        row = memory_activity(events, [])[0]
        self.assertEqual(row["Agent"], "Not recorded")
        self.assertEqual(row["Transport"], "Write receipt not recorded")

    def test_chronology_authors_and_transports(self):
        events = [{"step": "hypothesis", "agent": "research-scientist", "timestamp": "02", "payload": {
                    "memory_id": "proposal", "memory_reads": [{"memory_id": "critique", "author": "skeptic",
                    "read_via": "pastdev_mirror", "read_at": "01"}]}},
                  {"step": "review", "agent": "skeptic", "timestamp": "04", "payload": {
                    "memory_id": "new-critique", "memory_reads": [{"memory_id": "result", "author": "research-scientist",
                    "read_via": "pastdev_remote", "read_at": "03"}]}},
                  {"step": "iteration_complete", "agent": "orchestrator", "timestamp": "05", "payload": {"result_memory_id": "result"}}]
        memories = [{"memory_id": "proposal", "agent_id": "research-scientist", "kind": "hypothesis", "pastdev": {"status": "queued"}},
                    {"memory_id": "result", "agent_id": "research-scientist", "kind": "result", "write_via": "local_jsonl"}]
        rows = memory_activity(events, memories)
        self.assertEqual([r["Time"] for r in rows], ["01", "02", "03", "04", "05"])
        self.assertEqual(rows[0]["Memory author"], "Skeptic")
        self.assertEqual(rows[0]["Transport"], "past.dev local mirror")
        self.assertEqual(rows[1]["Transport"], "past.dev accepted write")
        self.assertEqual(rows[2]["Agent"], "Skeptic")
        self.assertEqual(rows[2]["Transport"], "past.dev remote")
        self.assertEqual(rows[3]["Transport"], "Write receipt not recorded")
        self.assertEqual(rows[4]["Agent"], "Scientist")
        self.assertEqual(rows[4]["Transport"], "Local JSONL")
