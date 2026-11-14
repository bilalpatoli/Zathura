import json
import math
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from zathura.agents.exa_search import search_papers
from zathura.agents.llm import complete_json
from zathura.agents.memory import memory_receipts, recall_memory, write_memory
from zathura.schema import BASELINE_CONFIG, DetectionConfig

AGENT = "research-scientist"

MECHANISMS = {
    "period_grid_size": "period sampling changes the chance of resolving a transit peak",
    "detrend_window": "the flattening timescale changes transit preservation versus residual stellar variability",
    "outlier_sigma": "upper-outlier rejection changes contamination while preserving negative transit dips",
    "detection_threshold": "the SDE cutoff trades sensitivity against false alarms",
    "duration_min_days": "the shortest searched duration changes sensitivity to short transits and noise",
    "duration_max_days": "the longest searched duration changes sensitivity to long transits and variability",
}


def _ground_prose(result, best):
    config = result["proposed_config"].model_dump()
    field = next(k for k in best if config[k] != best[k])
    aliases = {"period_grid_size": ("period grid", "grid size", "period_grid_size"),
               "detrend_window": ("detrend", "flatten", "window"),
               "outlier_sigma": ("outlier", "sigma clipping", "outlier_sigma"),
               "detection_threshold": ("threshold", "sde cutoff"),
               "duration_min_days": ("duration_min", "minimum duration", "shortest duration"),
               "duration_max_days": ("duration_max", "maximum duration", "longest duration")}
    text = result["hypothesis"].lower()
    stale = [name for name, words in aliases.items() if name != field and any(w in text for w in words)]
    prefix = "[MOCK] " if result.get("mock") else ""
    result["hypothesis"] = (f"{prefix}Changing {field} {best[field]} → {config[field]} may improve balanced accuracy: "
                            f"{MECHANISMS[field]}. Test on identical dev examples and measure recovery and false alarms.")
    result["predicted_result"] = (f"{prefix}The {field} change {best[field]} → {config[field]} may improve the "
                                  "recovery/false-alarm tradeoff; improvement is uncertain and requires evaluation.")
    result["prose_validation"] = {"rewritten_for_final_config": True, "discarded_stale_parameters": stale}
    result["parameter_change"] = {"field": field, "old": best[field], "new": config[field]}
    return result


def _url(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def memory_conflicts(config, best, memories):
    """Only structured negative evidence is eligible for deterministic vetoes."""
    changed = [k for k in best if config[k] != best[k]]
    conflicts = []
    for memory in memories:
        content = memory.get("content")
        if not isinstance(content, dict):
            continue
        measured_failure = memory.get("kind") == "result" and bool(content.get("measurements")) and content.get("failed") is True
        grounded_rejection = (memory.get("kind") == "critique"
                              and content.get("review_policy_version") == "grounded-review-v2"
                              and content.get("concrete_problem") in ("duplicate", "runtime", "false_positive_risk")
                              and bool(content.get("blocking_facts"))
                              and content.get("decision") in ("reject", "revise"))
        if not (measured_failure or grounded_rejection):
            continue
        previous = content.get("proposed_config")
        if not isinstance(previous, dict):
            continue
        exact = config == previous
        field, direction = content.get("changed_parameter"), content.get("direction")
        same_direction = ((measured_failure or content.get("concrete_problem") == "runtime")
                          and field in changed and direction in (-1, 1) and (config[field] - best[field]) * direction > 0)
        if exact or same_direction:
            conflicts.append(memory["memory_id"])
    return conflicts


def _normalize(result, best, candidates, sources):
    merged = best | result["proposed_config"]
    adjustments = []
    bounds = {"detrend_window": (101, 2001), "outlier_sigma": (2, 10),
              "period_grid_size": (1000, 50000), "detection_threshold": (0, 100),
              "duration_min_days": (0.02, 0.5), "duration_max_days": (0.02, 0.5)}
    for key, field in DetectionConfig.model_fields.items():
        if isinstance(merged[key], bool):
            raise ValueError("Boolean config values are invalid")
        value = float(merged[key])
        if not math.isfinite(value):
            raise ValueError("Config values must be finite")
        low, high = bounds[key]
        value = max(low, min(high, value))
        if field.annotation is int:
            value = int(round(value))
            if key == "detrend_window" and value % 2 == 0:
                value += 1
        if merged[key] != value:
            adjustments.append({"field": key, "from": merged[key], "to": value})
        merged[key] = value
    config = DetectionConfig.model_validate(merged)
    if config.model_dump() not in candidates:
        raise ValueError("Scientist must propose an untried single-parameter candidate from current best")
    by_url = {_url(s["url"]): s for s in sources}
    citations = []
    for citation in result["cited_sources"]:
        url = citation if isinstance(citation, str) else citation["url"]
        if _url(url) not in by_url:
            raise ValueError("Scientist must cite supplied search source URLs")
        canonical = by_url[_url(url)]
        if canonical not in citations:
            citations.append(canonical)
    if not citations:
        raise ValueError("Scientist must cite at least one search source")
    if result["experiment_request"] != {"dataset_id": "dev", "action": "run_experiment"}:
        raise ValueError("Scientist may only request dev experiments")
    if isinstance(result.get("predicted_result"), dict):
        result["predicted_result"] = json.dumps(result["predicted_result"])
    for key in ("hypothesis", "predicted_result"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"Missing {key}")
    return result | {"proposed_config": config, "cited_sources": citations, "config_adjustments": adjustments}


def research_hypothesis(history, metrics):
    history = [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in history]
    best = DetectionConfig.model_validate(metrics.get("current_best_config", BASELINE_CONFIG.model_dump()))
    accepted = [r for r in history if r.get("accepted")]
    if accepted and "current_best_config" not in metrics:
        winner = max(accepted, key=lambda r: r["measurements"]["balanced_accuracy"])
        if winner["measurements"]["balanced_accuracy"] >= metrics.get("balanced_accuracy", 0):
            best = DetectionConfig.model_validate(winner["proposed_config"])
    best = best.model_dump()
    weakness = "recovery of shallow transits" if metrics.get("recovery_rate", 0) < 0.9 else "false-alarm rejection"
    memories = recall_memory(AGENT, "failed rejected revise skeptic experiment results " + " ".join(best), k=200)
    memories = [m for m in memories if not isinstance(m.get("content"), dict)
                or not m["content"].get("benchmark_version")
                or m["content"]["benchmark_version"] == metrics.get("benchmark_version")]
    query = f"Kepler box least squares {weakness} detrending period grid"
    sources = search_papers(query, 3)
    memories = [m for m in memories if isinstance(m.get("content"), dict) and (
        (m.get("kind") == "result" and m["content"].get("measurements"))
        or (m.get("kind") == "critique" and m["content"].get("review_policy_version") == "grounded-review-v2"
            and m["content"].get("concrete_problem") != "unfair_comparison"))][:12]
    memory_fields = ("proposed_config", "current_best_config", "decision", "rationale", "concerns", "measurements",
                     "failed", "changed_parameter", "direction", "benchmark_version", "review_policy_version",
                     "concrete_problem", "blocking_facts")
    memories = [m | {"content": {k: m["content"][k] for k in memory_fields if k in m["content"]}} for m in memories]
    tried = [DetectionConfig.model_validate(r["proposed_config"]).model_dump() for r in history]
    candidates = []
    changes = {
        "period_grid_size": [5000, 10000, 20000, 50000, 1000],
        "detrend_window": [501, 701, 1001, 2001, 101],
        "outlier_sigma": [4.0, 3.0, 6.0, 8.0, 2.0, 10.0],
        "detection_threshold": [5.0, 7.0, 4.0, 8.0, 10.0],
        "duration_min_days": [0.02, 0.06, 0.08], "duration_max_days": [0.2, 0.4, 0.5],
    }
    for field, values in changes.items():
        for value in values:
            config = best | {field: value}
            if config != best and config not in tried:
                try:
                    candidates.append(DetectionConfig.model_validate(config).model_dump())
                except ValidationError:
                    continue
    payload = {"history": history, "metrics": metrics, "weakness": weakness,
               "sources": sources, "current_best_config": best, "candidate_configs": candidates,
               "recalled_memories": memories, "revision_feedback": metrics.get("revision_feedback")}
    for attempt in range(2):
        try:
            result = _normalize(complete_json(payload), best, candidates, sources)
            break
        except (ValueError, KeyError, TypeError) as exc:
            if attempt:
                raise
            payload["validation_feedback"] = str(exc)
    first_choice = result["proposed_config"].model_dump()
    conflicts = memory_conflicts(first_choice, best, memories)
    if conflicts:
        safe = [c for c in candidates if not memory_conflicts(c, best, memories)]
        if not safe:
            raise ValueError("All candidates conflict with recalled failed or rejected directions")
        payload.update(candidate_configs=safe, memory_veto={"first_choice": first_choice, "memory_ids": conflicts})
        result = _normalize(complete_json(payload), best, safe, sources)
    influence = result.get("memory_influence", {})
    known = set(conflicts)
    ids = influence.get("memory_ids", []) if isinstance(influence, dict) else []
    ids = [i for i in ids if isinstance(i, str) and i in known]
    explanation = influence.get("explanation", "") if isinstance(influence, dict) else ""
    if conflicts:
        ids = list(dict.fromkeys(conflicts + ids))
        explanation = (f"Deterministic memory veto changed the LLM's first-choice config {first_choice}: "
                       f"prior failure/rejection in {conflicts}. Selected {result['proposed_config'].model_dump()} instead.")
    if not ids or not isinstance(explanation, str) or not explanation.strip():
        ids, explanation = [], "No evidenced memory-driven change."
    result["memory_influence"] = {"memory_ids": ids, "explanation": explanation}
    result["recalled_memory_ids"] = [m["memory_id"] for m in memories]
    result["memory_reads"] = memory_receipts(memories)
    result["literature_search"] = {"query": query,
        "call_location": "zathura/agents/exa_search.py:search_papers",
        "sources": [{k: source[k] for k in ("title", "url", "mock") if k in source} for source in sources]}
    result["current_best_config"] = best
    result["benchmark_version"] = metrics.get("benchmark_version")
    result["context_metrics"] = metrics
    result["first_choice_config"] = first_choice
    result = _ground_prose(result, best)
    entry = {k: v for k, v in result.items() if k != "proposed_config"} | {"proposed_config": result["proposed_config"].model_dump()}
    result["memory_id"] = write_memory(AGENT, entry, "hypothesis", refs=ids)
    return result
