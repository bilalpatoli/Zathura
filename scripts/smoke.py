"""Offline-safe demo: python scripts/smoke.py from any working directory."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ZATHURA_MOCK"] = "1"

from zathura.agents.scientist import research_hypothesis
from zathura.engine import evaluate_experiment, get_experiment_results, load_targets, run_baseline, run_experiment
from zathura.schema import ExperimentRecord
from zathura.storage import DEFAULT_LOG, load_records, save_record


def main():
    targets = load_targets()
    print("Targets:", json.dumps(targets))
    baseline = get_experiment_results(run_baseline("dev"))
    print("Baseline:", json.dumps(baseline["measurements"]))
    hypothesis = research_hypothesis(load_records(), baseline["measurements"])
    print("Scientist (deterministic MOCK):", hypothesis["hypothesis"])
    print("Proposed config:", hypothesis["proposed_config"].model_dump_json())
    run_id = run_experiment(hypothesis["proposed_config"], "dev")
    evaluation = evaluate_experiment(run_id)
    results = get_experiment_results(run_id)
    record = ExperimentRecord(
        experiment_id=run_id, hypothesis=hypothesis["hypothesis"], cited_sources=hypothesis["cited_sources"],
        proposed_config=hypothesis["proposed_config"],
        reviewer_decision={"decision": "revise", "rationale": "Not reviewed: skeptic is not implemented in this slice."},
        measurements=results["measurements"], runtime_s=results["measurements"]["runtime_s"],
        evaluation_result=evaluation, accepted=False, data_source=results["data_source"],
    )
    save_record(record)
    print("Experiment:", json.dumps(results["measurements"]))
    print("Evaluation:", json.dumps(evaluation))
    print("Saved record:", DEFAULT_LOG)
    print("Injection-recovery on real Kepler curves. No reviewer acceptance claimed.")


if __name__ == "__main__":
    main()
