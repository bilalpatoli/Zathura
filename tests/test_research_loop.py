import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import zathura.agents.memory as memory
import zathura.orchestrator as orchestrator
from zathura.agents import llm
from zathura.agents.exa_search import search_papers
from zathura.agents.scientist import _ground_prose, _normalize, memory_conflicts, research_hypothesis
from zathura.agents.skeptic import review_hypothesis
from zathura.schema import BASELINE_CONFIG


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"ZATHURA_MOCK": "1", "PASTDEV_API_KEY": ""})
        self.env.start()
        self.path = patch.object(memory, "MEMORY_PATH", self.root / "memory.jsonl")
        self.path.start()
        self.mirror = patch.object(memory, "PASTDEV_MIRROR_PATH", self.root / "pastdev-memory.jsonl")
        self.mirror.start()

    def tearDown(self):
        self.path.stop()
        self.mirror.stop()
        self.env.stop()
        self.temp.cleanup()

    def test_local_cross_agent_keyword_recency_and_errors(self):
        self.assertEqual(memory.recall_memory("skeptic", "failure"), [])
        first = memory.write_memory("skeptic", {"failure": "period_grid_size"}, "critique")
        second = memory.write_memory("research-scientist", "unrelated", "hypothesis", [first])
        recalled = memory.recall_memory("research-scientist", "failure", 1)
        self.assertEqual(recalled[0]["agent_id"], "skeptic")
        self.assertEqual(recalled[0]["memory_id"], first)
        self.assertEqual(memory.recall_memory("skeptic", "", 1)[0]["memory_id"], second)
        for agent, k in (("unknown", 5), ("skeptic", 0), ("skeptic", True)):
            with self.assertRaises(ValueError):
                memory.recall_memory(agent, "", k)
        with self.assertRaises(ValueError):
            memory.write_memory("skeptic", "", "critique")
        with self.assertRaises(ValueError):
            memory.write_memory("skeptic", "text", "critique", "not-a-list")
        memory.MEMORY_PATH.write_text("not json\n")
        with self.assertRaises(json.JSONDecodeError):
            memory.recall_memory("skeptic", "")

    def test_pastdev_documented_ingest_poll_and_author_metadata(self):
        backend = memory.PastDevBackend("test-key", self.root / "mirror.jsonl")
        with patch.object(backend, "_request", side_effect=[
            {"status": "queued", "ingestionId": "ingestion"}, {"status": "completed"},
        ]) as request, patch.object(memory.time, "sleep"):
            memory_id = backend.write_memory("skeptic", {"failed": True}, "critique")
            self.assertEqual(request.call_count, 1)
            self.assertTrue(backend.wait_until_ready(memory_id))
        payload = request.call_args_list[0].args[2]
        self.assertEqual(payload["audience"], "project")
        self.assertEqual(payload["identity"], "skeptic")
        self.assertEqual(payload["id"], memory_id)
        entry = payload["metadata"]["zathura"]
        with patch.object(backend, "_request", return_value={"results": [
            {"sources": [{"metadata": {"zathura": entry}}, {"metadata": {"zathura": entry}}]},
            {"sources": [{"metadata": {"foreign": "source"}}]},
        ]}):
            self.assertEqual(backend.recall_remote("research-scientist", "failed"), [entry])
        with patch.object(backend, "_request", side_effect=AssertionError("mirror reads must not wait for network")):
            self.assertEqual(backend.recall_memory("research-scientist", "failed")[0]["memory_id"], memory_id)
        with patch.object(backend, "_request", side_effect=[
            {"status": "queued", "ingestionId": "ingestion"}, {"status": "failed"},
        ]), patch.object(memory.time, "sleep"):
            with self.assertRaises(RuntimeError):
                pending = backend.write_memory("skeptic", "text", "critique")
                backend.wait_until_ready(pending)
        with patch.object(backend, "_request", side_effect=ConnectionError("unavailable")):
            with self.assertRaises(ConnectionError):
                backend.recall_remote("skeptic", "text")
        with patch.dict(os.environ, {"PASTDEV_API_KEY": "test-key"}):
            self.assertIsInstance(memory.get_backend(), memory.PastDevBackend)

    def test_memory_changes_first_choice_deterministically(self):
        failed = BASELINE_CONFIG.model_dump() | {"period_grid_size": 5000}
        memory_id = memory.write_memory("skeptic", {
            "proposed_config": failed, "decision": "reject", "changed_parameter": "period_grid_size", "direction": 1,
        }, "critique")
        hypothesis = research_hypothesis([], {})
        self.assertEqual(hypothesis["first_choice_config"], failed)
        self.assertNotEqual(hypothesis["proposed_config"].period_grid_size, 5000)
        self.assertIn(memory_id, hypothesis["memory_influence"]["memory_ids"])
        self.assertIn("Deterministic memory veto", hypothesis["memory_influence"]["explanation"])
        change = hypothesis["parameter_change"]
        self.assertIn(f"{change['field']} {change['old']} → {change['new']}", hypothesis["hypothesis"])
        self.assertNotIn("2000 → 5000", hypothesis["hypothesis"])
        self.assertEqual(memory_conflicts(hypothesis["proposed_config"].model_dump(), BASELINE_CONFIG.model_dump(),
                                         memory.recall_memory("skeptic", "", 10)), [])

    def test_skeptic_approval_rejection_and_memory(self):
        hypothesis = research_hypothesis([], {})
        approved = review_hypothesis(hypothesis)
        self.assertEqual(approved["decision"], "approve")
        memory.write_memory("research-scientist", {
            "proposed_config": hypothesis["proposed_config"].model_dump(), "failed": True,
        }, "result")
        rejected = review_hypothesis(hypothesis)
        self.assertEqual(rejected["decision"], "reject")
        self.assertTrue(any(m["agent_id"] == "skeptic" for m in memory.recall_memory("research-scientist", "", 10)))
        with patch("zathura.agents.skeptic.complete_json", return_value={"decision": "invalid"}):
            with self.assertRaises(ValueError):
                review_hypothesis(hypothesis)

    def test_live_dispatch_and_citation_normalization_clamping(self):
        source = {"title": "Original", "url": "https://example.org/paper", "snippet": "original", "mock": False}
        config = BASELINE_CONFIG.model_dump() | {"detrend_window": 2001}
        result = {"hypothesis": "A test", "predicted_result": {"recovery": "higher"},
                  "proposed_config": {"detrend_window": 4000},
                  "cited_sources": [{"title": "Rephrased", "url": "https://example.org/paper/", "snippet": "changed"}],
                  "experiment_request": {"dataset_id": "dev", "action": "run_experiment"}}
        normalized = _normalize(result, BASELINE_CONFIG.model_dump(), [config], [source])
        self.assertEqual(normalized["cited_sources"], [source])
        self.assertEqual(normalized["proposed_config"].detrend_window, 2001)
        self.assertTrue(normalized["config_adjustments"])
        with patch.dict(os.environ, {"ZATHURA_MOCK": "0", "OPENAI_API_KEY": "test-key", "EXA_API_KEY": "test-key"}), \
             patch.object(llm, "OpenAI") as client, patch("zathura.agents.exa_search.Exa") as exa:
            client.return_value.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"decision":"approve"}'))])
            self.assertEqual(llm.complete_json({}, purpose="skeptic"), {"decision": "approve"})
            prompt = client.return_value.chat.completions.create.call_args.kwargs["messages"][0]["content"]
            self.assertIn("independent skeptical", prompt)
            exa.return_value.search_and_contents.return_value = SimpleNamespace(results=[
                SimpleNamespace(title="Paper", url="https://example.org", text="Evidence")])
            self.assertFalse(search_papers("transits", 1)[0]["mock"])
            client.return_value.chat.completions.create.return_value.choices[0].message.content = "bad json"
            with self.assertRaises(json.JSONDecodeError):
                llm.complete_json({}, purpose="skeptic")
            client.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps({
                "chosen_candidate_id": "candidate-0", "cited_sources": [source["url"]]})
            parsed = llm.complete_json({"candidate_configs": [config], "sources": [source]})
            self.assertEqual(parsed["proposed_config"], config)
            self.assertEqual(parsed["experiment_request"]["dataset_id"], "dev")
            self.assertEqual(client.return_value.chat.completions.create.call_args.kwargs["response_format"]["type"], "json_schema")

    def test_loop_revision_rejection_events_and_holdout_isolation(self):
        runs = {}
        def baseline(dataset):
            run_id = uuid4().hex
            runs[run_id] = {"run_id": run_id, "dataset_id": dataset,
                            "measurements": {"balanced_accuracy": 0.5, "recovery_rate": 0.5, "false_alarm_rate": 0.5}}
            if dataset == "holdout":
                runs[run_id]["measurements"]["private_holdout_sentinel"] = 123456789
            return run_id
        def experiment(config, dataset):
            run_id = baseline(dataset)
            runs[run_id]["measurements"]["balanced_accuracy"] = 0.6
            return run_id
        critiques = [{"decision": decision, "rationale": "Test reviewer", "concerns": [], "requested_changes": [],
                      "memory_id": uuid4().hex} for decision in ("revise", "approve", "reject", "revise", "reject")]
        events, records = [], []
        with patch.object(orchestrator, "DATA", self.root), \
             patch.object(orchestrator, "load_targets", return_value=[]), \
             patch.object(orchestrator, "run_baseline", side_effect=baseline), \
             patch.object(orchestrator, "run_experiment", side_effect=experiment) as run, \
             patch.object(orchestrator, "get_experiment_results", side_effect=lambda r: runs[r]), \
             patch.object(orchestrator, "evaluate_experiment", return_value={"improved_vs_baseline": True, "improved_vs_best": True, "delta": 0.1}), \
             patch.object(orchestrator, "review_hypothesis", side_effect=critiques) as review, \
             patch.object(orchestrator, "research_hypothesis", wraps=research_hypothesis) as scientist, \
             patch.object(orchestrator, "save_record", side_effect=records.append):
            summary = orchestrator.run_research_loop(3, on_event=events.append)
        self.assertEqual(summary["rejected"], 2)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(review.call_count, 5)
        self.assertEqual(review.call_args_list[1].kwargs, {"final_review": True})
        self.assertEqual(review.call_args_list[4].kwargs, {"final_review": True})
        self.assertEqual([c.args[1] for c in run.call_args_list], ["dev", "holdout"])
        self.assertEqual(len(records), 3)
        self.assertEqual(records[1].measurements, {})
        self.assertEqual(len((self.root / "holdout.jsonl").read_text().splitlines()), 1)
        self.assertTrue(all("123456789" not in json.dumps(c.args, default=str) for c in scientist.call_args_list))
        self.assertTrue(all(c.args[1].get("dataset_id", "dev") == "dev" for c in scientist.call_args_list))
        self.assertEqual(len(events), len((self.root / "events.jsonl").read_text().splitlines()))
        self.assertEqual(events[-1]["step"], "loop_complete")
        self.assertEqual(events[-2]["step"], "holdout_complete")
        for iterations, dataset in ((-1, "dev"), (True, "dev"), (1, "holdout")):
            with self.assertRaises(ValueError):
                orchestrator.run_research_loop(iterations, dataset)

    def test_loop_error_event_does_not_log_error_details(self):
        events = []
        with patch.object(orchestrator, "DATA", self.root), \
             patch.object(orchestrator, "load_targets", return_value=[]), \
             patch.object(orchestrator, "run_baseline", return_value="baseline"), \
             patch.object(orchestrator, "get_experiment_results", return_value={"measurements": {}}), \
             patch.object(orchestrator, "research_hypothesis", side_effect=ValueError("private transport detail")):
            with self.assertRaises(ValueError):
                orchestrator.run_research_loop(1, on_event=events.append)
        self.assertEqual(events[-1]["step"], "error")
        self.assertNotIn("private transport detail", json.dumps(events))

    def test_empty_loop_runs_no_agents(self):
        with patch.object(orchestrator, "DATA", self.root), \
             patch.object(orchestrator, "load_targets", return_value=[]), \
             patch.object(orchestrator, "run_baseline", return_value="baseline"), \
             patch.object(orchestrator, "run_experiment", return_value="holdout"), \
             patch.object(orchestrator, "evaluate_experiment", return_value={}), \
             patch.object(orchestrator, "get_experiment_results", return_value={"measurements": {}}), \
             patch.object(orchestrator, "research_hypothesis") as scientist:
            summary = orchestrator.run_research_loop(0)
        self.assertEqual(summary["iterations"], 0)
        scientist.assert_not_called()

    def test_binary_review_and_unsupported_prior_claims(self):
        hypothesis = research_hypothesis([], {})
        _, schema = llm._response_schema({"final_review": True}, "skeptic")
        self.assertEqual(schema["json_schema"]["schema"]["properties"]["decision"]["enum"], ["approve", "reject"])
        invalid = {"decision": "revise", "rationale": "A concern", "concerns": [], "requested_changes": []}
        with patch("zathura.agents.skeptic.complete_json", return_value=invalid):
            with self.assertRaises(ValueError):
                review_hypothesis(hypothesis, final_review=True)
        unsupported = invalid | {"decision": "reject", "concrete_problem": "duplicate",
                                 "rationale": "Prior experiments failed and yielded limited success.",
                                 "cited_memory_ids": ["nonexistent"], "concerns": ["Previous runs already failed."]}
        with patch("zathura.agents.skeptic.complete_json", return_value=unsupported):
            critique = review_hypothesis(hypothesis, final_review=True)
        self.assertEqual(critique["decision"], "approve")
        self.assertEqual(critique["dropped_memory_ids"], ["nonexistent"])
        self.assertEqual(critique["prior_outcomes"], [])
        self.assertNotIn("Prior experiments", critique["rationale"])
        self.assertEqual(critique["concerns"], [])
        metrics = {"balanced_accuracy": 0.5, "recovery_rate": 0.5, "false_alarm_rate": 0.5}
        evidence_id = memory.write_memory("research-scientist", {
            "proposed_config": hypothesis["proposed_config"].model_dump(), "measurements": metrics, "failed": True,
        }, "result")
        with patch("zathura.agents.skeptic.complete_json", return_value=unsupported | {"cited_memory_ids": [evidence_id]}):
            grounded = review_hypothesis(hypothesis)
        self.assertEqual(grounded["decision"], "reject")
        self.assertEqual(grounded["prior_outcomes"][0]["measurements"], metrics)
        self.assertEqual(grounded["prior_outcomes"][0]["memory_id"], evidence_id)

    def test_stale_prose_is_discarded_for_actual_selected_change(self):
        result = {"proposed_config": BASELINE_CONFIG.model_copy(update={"detection_threshold": 5.0}),
                  "hypothesis": "A smaller period grid size will improve recovery.", "predicted_result": "Old prediction"}
        grounded = _ground_prose(result, BASELINE_CONFIG.model_dump())
        self.assertIn("detection_threshold 6.0 → 5.0", grounded["hypothesis"])
        self.assertNotIn("period grid", grounded["hypothesis"])
        self.assertEqual(grounded["prose_validation"]["discarded_stale_parameters"], ["period_grid_size"])

    def test_unevidenced_model_memory_influence_is_discarded(self):
        memory_id = memory.write_memory("skeptic", "Shared-memory integration check.", "integration-check")
        complete = llm.complete_json

        def unsupported(payload):
            return complete(payload) | {"memory_influence": {
                "memory_ids": [memory_id], "explanation": "Prior experiments failed."
            }}

        with patch("zathura.agents.scientist.complete_json", side_effect=unsupported):
            hypothesis = research_hypothesis([], {})
        self.assertEqual(hypothesis["memory_influence"]["memory_ids"], [])
        self.assertEqual(hypothesis["memory_influence"]["explanation"], "No evidenced memory-driven change.")


if __name__ == "__main__":
    unittest.main()
