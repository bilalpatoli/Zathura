"""Bounded research loop; holdout is evaluated only after agent work finishes."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from zathura.agents.memory import write_memory
from zathura.agents.scientist import research_hypothesis
from zathura.agents.skeptic import review_hypothesis
from zathura.engine import (
    BENCHMARK_DESCRIPTION, BENCHMARK_VERSION, DATA_SOURCE, evaluate_experiment, get_experiment_results,
    load_targets, run_baseline, run_experiment,
)
from zathura.schema import BASELINE_CONFIG, ExperimentRecord
from zathura.storage import save_record

DATA = Path(__file__).resolve().parents[1] / "data"


def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _append(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(_jsonable(value)) + "\n")


def run_research_loop(max_iterations, dataset_id="dev", on_event=None):
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if dataset_id != "dev":
        raise ValueError("Research agents can only tune on dev")
    loop_id = uuid4().hex

    def emit(step, agent, payload):
        event = {"loop_id": loop_id, "step": step, "agent": agent, "payload": _jsonable(payload),
                 "timestamp": datetime.now(timezone.utc).isoformat()}
        _append(DATA / "events.jsonl", event)
        if on_event:
            on_event(event)

    records, history = [], []
    rejected = 0
    emit("targets", "engine", load_targets())
    best_config = BASELINE_CONFIG
    baseline_id = run_baseline(dataset_id)
    best_results = get_experiment_results(baseline_id)
    emit("baseline", "engine", best_results)
    for iteration in range(1, max_iterations + 1):
        started = time.perf_counter()
        metrics = best_results["measurements"] | {"current_best_config": best_config.model_dump(),
            "benchmark_version": BENCHMARK_VERSION,
            "benchmark_description": BENCHMARK_DESCRIPTION,
        }
        try:
            hypothesis = research_hypothesis(history, metrics)
            emit("hypothesis", "research-scientist", {"iteration": iteration, **hypothesis})
            critique = review_hypothesis(hypothesis)
            emit("review", "skeptic", {"iteration": iteration, **critique})
            refs = hypothesis["recalled_memory_ids"] + [hypothesis["memory_id"], critique["memory_id"]]
            if critique["decision"] == "revise":
                prior = hypothesis
                hypothesis = research_hypothesis(history + [{"proposed_config": prior["proposed_config"].model_dump(), "accepted": False}],
                                                 metrics | {"revision_feedback": critique})
                emit("revision", "research-scientist", {"iteration": iteration, **hypothesis})
                critique = review_hypothesis(hypothesis, final_review=True)
                emit("review_revision", "skeptic", {"iteration": iteration, **critique})
                refs += hypothesis["recalled_memory_ids"] + [hypothesis["memory_id"], critique["memory_id"]]
            measurements = {}
            evaluation = {"improved_vs_baseline": False, "improved_vs_best": False, "delta": 0.0}
            accepted = False
            if critique["decision"] == "approve":
                emit("experiment_started", "engine", {"iteration": iteration, "config": hypothesis["proposed_config"]})
                run_id = run_experiment(hypothesis["proposed_config"], dataset_id)
                evaluation = evaluate_experiment(run_id)
                results = get_experiment_results(run_id)
                measurements = results["measurements"]
                accepted = evaluation["improved_vs_best"]
                emit("evaluation", "engine", {"iteration": iteration, **results})
            else:
                rejected += 1
                run_id = uuid4().hex
            record = ExperimentRecord(
                experiment_id=run_id, hypothesis=hypothesis["hypothesis"], cited_sources=hypothesis["cited_sources"],
                proposed_config=hypothesis["proposed_config"],
                reviewer_decision={"decision": critique["decision"], "rationale": critique["rationale"]},
                measurements=measurements, runtime_s=time.perf_counter() - started,
                evaluation_result=evaluation, memory_refs=list(dict.fromkeys(refs)),
                accepted=accepted, data_source=DATA_SOURCE,
            )
            save_record(record)
            records.append(record)
            history.append(record.model_dump(mode="json"))
            config = hypothesis["proposed_config"].model_dump()
            field = next(k for k in config if config[k] != best_config.model_dump()[k])
            content = record.model_dump(mode="json") | {
                "benchmark_version": BENCHMARK_VERSION, "decision": critique["decision"],
                "failed": not accepted, "changed_parameter": field,
                "direction": 1 if config[field] > best_config.model_dump()[field] else -1,
                "memory_influence": hypothesis["memory_influence"],
            }
            memory_id = write_memory("research-scientist", content, "result", refs=record.memory_refs)
            if measurements and measurements["balanced_accuracy"] > best_results["measurements"]["balanced_accuracy"]:
                best_config, best_results = hypothesis["proposed_config"], results
            emit("iteration_complete", "orchestrator", {
                "iteration": iteration, "hypothesis": hypothesis["hypothesis"], "decision": critique["decision"],
                "measurements": measurements, "accepted": accepted, "memory_influence": hypothesis["memory_influence"],
                "record": record, "result_memory_id": memory_id,
            })
        except Exception as exc:
            # Do not leak transport errors/headers or credentials into timeline logs.
            emit("error", "orchestrator", {"iteration": iteration, "error_type": type(exc).__name__})
            raise
    # No agent calls or memory writes occur after entering the holdout phase.
    emit("holdout_started", "engine", {"config": best_config})
    holdout_baseline = run_baseline("holdout")
    holdout_id = run_experiment(best_config, "holdout")
    holdout_evaluation = evaluate_experiment(holdout_id)
    holdout = get_experiment_results(holdout_id) | {"loop_id": loop_id, "baseline_run_id": holdout_baseline,
                                                   "evaluation_result": holdout_evaluation}
    _append(DATA / "holdout.jsonl", holdout)
    emit("holdout_complete", "engine", holdout)
    summary = {"loop_id": loop_id, "iterations": len(records), "rejected": rejected,
               "accepted": sum(r.accepted for r in records), "best_config": best_config.model_dump(),
               "best_measurements": best_results["measurements"], "holdout_run_id": holdout_id}
    emit("loop_complete", "orchestrator", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    def show(event):
        if event["step"] in ("iteration_complete", "loop_complete", "error"):
            print(json.dumps(event), flush=True)
    run_research_loop(args.iterations, on_event=show)


if __name__ == "__main__":
    main()
