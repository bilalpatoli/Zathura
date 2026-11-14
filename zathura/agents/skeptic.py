import re
from typing import Literal

from pydantic import BaseModel, Field

from zathura.agents.llm import complete_json
from zathura.agents.memory import recall_memory, write_memory
from zathura.schema import BASELINE_CONFIG, DetectionConfig

REVIEW_POLICY_VERSION = "grounded-review-v2"


class Critique(BaseModel):
    decision: Literal["approve", "reject", "revise"]
    rationale: str = Field(min_length=1)
    concerns: list[str]
    requested_changes: list[str]
    concrete_problem: Literal["none", "unfair_comparison", "false_positive_risk", "runtime", "duplicate"] = "none"
    cited_memory_ids: list[str] = Field(default_factory=list)


class FinalCritique(Critique):
    decision: Literal["approve", "reject"]


def _grounded_text(text, cited, unknown=()):
    historical = r"\b(previous(?:ly)?|prior|past|historical|already|tested|failed|failure|recalled|yielded)\b"
    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        ids = set(re.findall(r"\b[a-f0-9]{32}\b", sentence))
        if ids - set(cited) or any(i and i in sentence for i in unknown):
            continue
        if re.search(historical, sentence, re.I) and not ids.intersection(cited):
            continue
        kept.append(sentence)
    return " ".join(kept)


def _blocking_facts(config, best, request, evidence):
    facts = {}
    changed = [k for k in best if config[k] != best[k]]
    if len(changed) != 1 or request != {"dataset_id": "dev", "action": "run_experiment"}:
        facts["unfair_comparison"] = [{"changed_parameters": changed, "experiment_request": request}]
    duplicates = [m["memory_id"] for m in evidence if m["content"].get("proposed_config") == config]
    if duplicates:
        facts["duplicate"] = [{"memory_ids": duplicates, "reason": "Exact configuration already measured on this fixed benchmark."}]
    if config["detection_threshold"] <= 0:
        facts["false_positive_risk"] = [{"detection_threshold": config["detection_threshold"],
                                         "reason": "Nonpositive SDE cutoff disables meaningful peak rejection."}]
    for memory in evidence:
        previous = memory["content"].get("proposed_config", {})
        runtime = memory["content"]["measurements"].get("runtime_s")
        if runtime and previous.get("period_grid_size") and all(config[k] == previous.get(k) for k in config if k != "period_grid_size"):
            estimate = runtime * config["period_grid_size"] / previous["period_grid_size"]
            if estimate > 30:
                facts.setdefault("runtime", []).append({"memory_id": memory["memory_id"],
                    "measured_runtime_s": runtime, "estimated_runtime_s": estimate,
                    "reason": "Linear period-grid cost estimate exceeds the 30-second dev budget; not a measured candidate runtime."})
    return facts


def review_hypothesis(hypothesis, final_review=False):
    config = DetectionConfig.model_validate(hypothesis["proposed_config"]).model_dump()
    best = DetectionConfig.model_validate(hypothesis.get("current_best_config", BASELINE_CONFIG.model_dump())).model_dump()
    memories = recall_memory("skeptic", "past failures rejected experiments " + " ".join(config), k=200)
    evidence = [m for m in memories if m.get("kind") == "result" and isinstance(m.get("content"), dict)
                and m["content"].get("measurements")
                and m["content"].get("benchmark_version") == hypothesis.get("benchmark_version")]
    evidence = [{"memory_id": m["memory_id"], "agent_id": m["agent_id"], "kind": "result",
                 "content": {k: m["content"][k] for k in ("proposed_config", "measurements", "benchmark_version") if k in m["content"]}}
                for m in evidence]
    facts = _blocking_facts(config, best, hypothesis.get("experiment_request"), evidence)
    changed = [k for k in best if config[k] != best[k]]
    contract = {"changed_parameters": changed, "bounded_config": True,
                "fixed_dev_examples_and_seed": True, "fair_comparison": "unfair_comparison" not in facts}
    serial = {k: hypothesis[k] for k in ("hypothesis", "experiment_request", "benchmark_version", "context_metrics") if k in hypothesis}
    serial.update(proposed_config=config, current_best_config=best)
    payload = {"hypothesis": serial, "comparison_contract": contract, "blocking_facts": facts,
               "outcome_evidence": evidence, "final_review": final_review}
    by_id = {m["memory_id"]: m for m in evidence}
    model = FinalCritique if final_review else Critique
    attempts = []
    for attempt in range(2):
        raw = model.model_validate(complete_json(payload, purpose="skeptic")).model_dump()
        attempts.append(raw)
        cited = list(dict.fromkeys(i for i in raw["cited_memory_ids"] if i in by_id))
        unknown = [i for i in raw["cited_memory_ids"] if i not in by_id]
        rationale = _grounded_text(raw["rationale"], cited, unknown)
        problem = raw["concrete_problem"]
        valid_problem = problem == "none" or problem in facts
        consistent = (raw["decision"] == "approve") == (problem == "none")
        label_only = raw["rationale"].strip().lower().startswith("current proposal assessment:")
        if rationale.strip() and valid_problem and consistent and not label_only:
            break
        payload["validation_feedback"] = (
            "Regenerate with specific current-proposal reasoning, not an assessment label. "
            "Only cite supplied outcome ids; put an id in each sentence making a historical claim. "
            "When no blocking fact applies, concrete_problem must be none and decision must be approve. "
            "Do not repeat the invalid review: " + str(raw)
        )
    else:
        raise ValueError("Skeptic did not produce a grounded, policy-consistent rationale after correction")
    critique = raw | {"rationale": rationale, "cited_memory_ids": cited,
                      "dropped_memory_ids": [i for i in raw["cited_memory_ids"] if i not in by_id],
                      "model_reviews": attempts, "comparison_contract": contract,
                      "blocking_facts": facts, "review_policy_version": REVIEW_POLICY_VERSION,
                      "final_review": final_review}
    for key in ("concerns", "requested_changes"):
        critique[key] = [clean for text in raw[key] if (clean := _grounded_text(text, cited, unknown))]
    critique["prior_outcomes"] = [{"memory_id": i, "agent_id": by_id[i]["agent_id"],
        "proposed_config": by_id[i]["content"].get("proposed_config"),
        "measurements": by_id[i]["content"]["measurements"]} for i in cited]
    content = critique | {"proposed_config": config, "current_best_config": best,
                          "benchmark_version": hypothesis.get("benchmark_version")}
    if len(changed) == 1:
        field = changed[0]
        content.update(changed_parameter=field, direction=1 if config[field] > best[field] else -1)
    refs = ([hypothesis["memory_id"]] if hypothesis.get("memory_id") else []) + cited
    critique["memory_id"] = write_memory("skeptic", content, "critique", refs=refs)
    critique["memory_refs"] = refs
    return critique
