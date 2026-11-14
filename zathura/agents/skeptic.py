import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from zathura.agents.llm import complete_json
from zathura.agents.memory import recall_memory, write_memory
from zathura.agents.scientist import memory_conflicts
from zathura.schema import BASELINE_CONFIG, DetectionConfig


class Critique(BaseModel):
    decision: Literal["approve", "reject", "revise"]
    rationale: str = Field(min_length=1)
    concerns: list[str]
    requested_changes: list[str]
    concrete_problem: Literal["none", "unfair_comparison", "false_positive_risk", "runtime", "duplicate"] = "none"
    cited_memory_ids: list[str] = Field(default_factory=list)


class FinalCritique(Critique):
    decision: Literal["approve", "reject"]


def _current_text(text):
    historical = r"previous|prior|past\b|histor|already|tested|failed|failure|memory|recalled|yielded|unproductive|has been|have been|associated with|limited success|[a-f0-9]{32}"
    return "" if re.search(historical, text, re.IGNORECASE) else text


def review_hypothesis(hypothesis, final_review=False):
    config = DetectionConfig.model_validate(hypothesis["proposed_config"]).model_dump()
    best = DetectionConfig.model_validate(hypothesis.get("current_best_config", BASELINE_CONFIG.model_dump())).model_dump()
    memories = recall_memory("skeptic", "past failures rejected experiments " + " ".join(config), k=12)
    memories = [m for m in memories if not isinstance(m.get("content"), dict)
                or not m["content"].get("benchmark_version")
                or m["content"]["benchmark_version"] == hypothesis.get("benchmark_version")]
    memories = [m for m in memories if m.get("kind") in ("result", "critique")]
    serial = hypothesis | {"proposed_config": config}
    evidence = [m for m in memories if m.get("kind") == "result" and isinstance(m.get("content"), dict)
                and m["content"].get("measurements")]
    model = FinalCritique if final_review else Critique
    critique = model.model_validate(complete_json({
        "hypothesis": serial, "recalled_memories": memories, "outcome_evidence": evidence,
        "final_review": final_review,
    }, purpose="skeptic")).model_dump()
    by_id = {m["memory_id"]: m for m in evidence}
    cited = list(dict.fromkeys(i for i in critique["cited_memory_ids"] if i in by_id))
    critique["dropped_memory_ids"] = [i for i in critique["cited_memory_ids"] if i not in by_id]
    critique["cited_memory_ids"] = cited
    critique["rationale"] = _current_text(critique["rationale"])
    for key in ("concerns", "requested_changes"):
        critique[key] = [clean for text in critique[key] if (clean := _current_text(text))]
    exact_duplicate = any(m["content"].get("proposed_config") == config for m in evidence)
    if critique["concrete_problem"] == "none" or (critique["concrete_problem"] == "duplicate" and not exact_duplicate):
        critique.update(decision="approve", rationale="Cheap, bounded, single-parameter dev experiment; the evaluator decides whether it improves.",
                        requested_changes=[], concrete_problem="none")
    if not critique["rationale"]:
        critique["rationale"] = f"Current proposal assessment: {critique['concrete_problem']}."
    critique["prior_outcomes"] = [
        {"memory_id": i, "agent_id": by_id[i]["agent_id"],
         "proposed_config": by_id[i]["content"].get("proposed_config"),
         "measurements": by_id[i]["content"]["measurements"]} for i in cited
    ]
    if cited:
        critique["rationale"] += " Verified recalled outcomes: " + json.dumps(critique["prior_outcomes"])
    conflicts = memory_conflicts(config, best, memories)
    changed = [k for k in best if config[k] != best[k]]
    if conflicts:
        critique.update(decision="reject", rationale=f"Previously failed/rejected configuration or direction: {conflicts}",
                        concerns=critique["concerns"] + ["Shared memory already covers this negative direction."],
                        requested_changes=["Choose an untried direction not covered by prior failures."])
    if len(changed) != 1 or hypothesis.get("experiment_request") != {"dataset_id": "dev", "action": "run_experiment"}:
        critique.update(decision="reject", rationale="Comparison must change exactly one parameter on dev.",
                        requested_changes=["Use one bounded parameter change from current best; dev only."])
    content = critique | {"proposed_config": config, "current_best_config": best,
                          "benchmark_version": hypothesis.get("benchmark_version")}
    if len(changed) == 1:
        field = changed[0]
        content.update(changed_parameter=field, direction=1 if config[field] > best[field] else -1)
    refs = [hypothesis["memory_id"]] if hypothesis.get("memory_id") else []
    refs += [m["memory_id"] for m in memories if m["memory_id"] not in refs]
    critique["memory_id"] = write_memory("skeptic", content, "critique", refs=refs)
    critique["memory_refs"] = refs
    critique["final_review"] = final_review
    return critique
