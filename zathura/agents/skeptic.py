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


def review_hypothesis(hypothesis):
    config = DetectionConfig.model_validate(hypothesis["proposed_config"]).model_dump()
    best = DetectionConfig.model_validate(hypothesis.get("current_best_config", BASELINE_CONFIG.model_dump())).model_dump()
    memories = recall_memory("skeptic", "past failures rejected experiments " + " ".join(config), k=12)
    memories = [m for m in memories if not isinstance(m.get("content"), dict)
                or not m["content"].get("benchmark_version")
                or m["content"]["benchmark_version"] == hypothesis.get("benchmark_version")]
    memories = [m for m in memories if m.get("kind") in ("result", "critique")]
    serial = hypothesis | {"proposed_config": config}
    critique = Critique.model_validate(complete_json({"hypothesis": serial, "recalled_memories": memories}, purpose="skeptic")).model_dump()
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
    return critique
