from pydantic import ValidationError

from zathura.agents.exa_search import search_papers
from zathura.agents.llm import complete_json
from zathura.schema import BASELINE_CONFIG, DetectionConfig, Source


def research_hypothesis(history, metrics):
    history = [r.model_dump(mode="json") if hasattr(r, "model_dump") else r for r in history]
    best = DetectionConfig.model_validate(metrics.get("current_best_config", BASELINE_CONFIG.model_dump()))
    accepted = [r for r in history if r.get("accepted")]
    if accepted:
        winner = max(accepted, key=lambda r: r["measurements"]["balanced_accuracy"])
        if winner["measurements"]["balanced_accuracy"] >= metrics.get("balanced_accuracy", 0):
            best = DetectionConfig.model_validate(winner["proposed_config"])
    weakness = "recovery of shallow transits" if metrics.get("recovery_rate", 0) < 0.9 else "false-alarm rejection"
    sources = search_papers(f"Kepler box least squares {weakness} detrending period grid", 3)
    tried = [DetectionConfig.model_validate(r["proposed_config"]).model_dump() for r in history]
    candidates = []
    changes = {
        "period_grid_size": [5000, 10000, 20000, 50000, 1000],
        "detrend_window": [501, 701, 1001, 2001, 101],
        "outlier_sigma": [4.0, 3.0, 6.0, 8.0, 2.0, 10.0],
        "detection_threshold": [5.0, 7.0, 4.0, 8.0, 10.0],
        "duration_min_days": [0.02, 0.06, 0.08],
        "duration_max_days": [0.2, 0.4, 0.5],
    }
    for field, values in changes.items():
        for value in values:
            config = best.model_dump() | {field: value}
            if config != best.model_dump() and config not in tried:
                try:
                    candidates.append(DetectionConfig.model_validate(config).model_dump())
                except ValidationError:
                    continue
    result = complete_json({"history": history, "metrics": metrics, "weakness": weakness,
                            "sources": sources, "current_best_config": best.model_dump(),
                            "candidate_configs": candidates})
    config = DetectionConfig.model_validate(result["proposed_config"])
    if config.model_dump() not in candidates:
        raise ValueError("Scientist must propose an untried single-parameter candidate from current best")
    citations = [Source.model_validate(s).model_dump() for s in result["cited_sources"]]
    if not citations or any(s not in sources for s in citations):
        raise ValueError("Scientist must cite supplied search sources")
    if result["experiment_request"] != {"dataset_id": "dev", "action": "run_experiment"}:
        raise ValueError("Scientist may only request dev experiments")
    for key in ("hypothesis", "predicted_result"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"Missing {key}")
    return result | {"proposed_config": config, "cited_sources": citations}
