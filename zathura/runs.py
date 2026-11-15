"""Plain-language, read-only views of recorded research runs. Nothing here invents progress."""
from datetime import datetime, timezone

from zathura.evidence import config_change

DECISIONS = {"approve": "approved", "reject": "rejected", "revise": "asked for a revision"}


def parse_time(stamp):
    try:
        return datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None


def run_list(events):
    runs = {}
    for event in events:
        loop_id = event.get("loop_id")
        if not loop_id:
            continue
        run = runs.setdefault(loop_id, {"loop_id": loop_id, "started": event.get("timestamp", ""), "ended": None,
                                        "rounds": 0, "status": "incomplete"})
        run["last"] = event.get("timestamp", "")
        step = event.get("step")
        if step == "iteration_complete":
            run["rounds"] += 1
        elif step == "error":
            run["status"] = "stopped"
        elif step == "loop_complete":
            run["status"], run["ended"] = "complete", event.get("timestamp")
    return sorted(runs.values(), key=lambda r: r["started"], reverse=True)


def run_view(state, loop_id):
    events = [e for e in state["events"] if e.get("loop_id") == loop_id]
    baseline = next((e["payload"] for e in reversed(events) if e.get("step") == "baseline"), None)
    ids = {e["payload"]["record"]["experiment_id"] for e in events
           if e.get("step") == "iteration_complete" and "record" in e.get("payload", {})}
    current = [r for r in state["records"] if r["experiment_id"] in ids]
    holdout = next((h for h in reversed(state["holdout"]) if h.get("loop_id") == loop_id), None)
    summary = next((e["payload"] for e in reversed(events) if e.get("step") == "loop_complete"), None)
    return {"events": events, "baseline": baseline, "current": current, "holdout": holdout, "summary": summary}


def selected_measurements(view):
    """The method the run kept: the best measured development result above the starting method."""
    if view["summary"] and view["summary"].get("best_measurements"):
        return view["summary"]["best_measurements"]
    base = (view["baseline"] or {}).get("measurements")
    measured = [r["measurements"] for r in view["current"] if r["measurements"]]
    if not base:
        return None
    return max([base, *measured], key=lambda m: m.get("balanced_accuracy", float("-inf")))


def outcome(view, status):
    if status == "running":
        return "running", "Research in progress", "Results update as each round finishes."
    if status != "complete":
        return ("stopped", "Stopped", "The run ended before finishing. Completed rounds are shown; no final test was recorded."
                if not view["holdout"] else "The run ended before finishing.")
    if not any(r["accepted"] for r in view["current"]):
        return ("unchanged", "Unchanged",
                "No proposed change beat the starting method on development stars, so Zathura kept the starting method.")
    delta = ((view["holdout"] or {}).get("evaluation_result") or {}).get("delta")
    if delta is None:
        return "improved", "Improved on development stars", "The final test result was not recorded."
    if delta > 0:
        return ("improved", "Improved",
                "The selected method beat the starting method on development stars and on the final test.")
    return ("worse", "Worse on the final test",
            "The selected method helped on development stars but did not beat the starting method on unseen stars.")


def counts(measurements):
    """Recovered test signals and false alarms as (found, total) pairs when the totals were recorded."""
    m = measurements or {}
    out = {}
    for key, total in (("recovery_rate", "positive_count"), ("false_alarm_rate", "negative_count")):
        if m.get(key) is not None and m.get(total):
            out[key] = (round(m[key] * m[total]), int(m[total]))
    return out


def plain_result(measurements):
    m = measurements or {}
    if m.get("recovery_rate") is None:
        return "No measurement recorded."
    c = counts(m)
    found = f"{c['recovery_rate'][0]} of {c['recovery_rate'][1]} test signals" if "recovery_rate" in c else f"{m['recovery_rate']:.0%} of test signals"
    alarms = (f"{c['false_alarm_rate'][0]} false alarms on {c['false_alarm_rate'][1]} quiet stars" if "false_alarm_rate" in c
              else f"a {m['false_alarm_rate']:.0%} false-alarm rate")
    return f"Found {found}, with {alarms}."


def describe(event):
    """(title, why it matters) for one recorded event, or None for events not shown to people."""
    step, p = event.get("step"), event.get("payload")
    p = p if isinstance(p, dict) else {}
    label = f"Round {p['iteration']} · " if "iteration" in p else ""
    if step == "targets":
        return "Loaded real Kepler observations", "Every method is tested on the same stars, so results are comparable."
    if step == "baseline":
        return ("Measured the starting method · " + plain_result(p.get("measurements")),
                "This is the score every proposed change has to beat.")
    if step in ("hypothesis", "revision"):
        change = config_change(p.get("current_best_config"), p.get("proposed_config"))
        verb = "Revised the proposal" if step == "revision" else "Proposed a change"
        sources = len(p.get("cited_sources", []))
        return (f"{label}{verb}: {change}",
                f"The Scientist grounded this idea in {sources} published source{'s' * (sources != 1)}." if sources
                else "The Scientist proposed one bounded change so its effect can be measured.")
    if step in ("review", "review_revision"):
        decision = DECISIONS.get(p.get("decision"), p.get("decision", "reviewed"))
        return (f"{label}Skeptic {decision} the proposal",
                "Only approved changes are tested, which avoids spending compute on weak ideas.")
    if step == "experiment_started":
        return f"{label}Testing the change on development stars", "The test uses simulated signals whose answers are known."
    if step == "evaluation":
        return f"{label}Measured the change · " + plain_result(p.get("measurements")), "Compared with the current best method."
    if step == "iteration_complete":
        if not p.get("measurements"):
            return f"{label}Not tested", "The Skeptic did not approve this change, so no score was recorded."
        if p.get("accepted"):
            return f"{label}Kept the change", "It scored higher than the best method so far on development stars."
        return f"{label}Rejected the change", "It did not score higher than the best method so far."
    if step == "error":
        return f"{label}Stopped: {p.get('error_type', 'unknown error')}", "Completed rounds remain recorded."
    if step == "holdout_started":
        return "Started the final test", "The selected method is tested on stars that weren't used to choose the settings."
    if step == "holdout_complete":
        return "Final test · " + plain_result(p.get("measurements")), "This shows whether the improvement holds on unseen stars."
    if step == "loop_complete":
        return "Research finished", "No AI calls happen after this point."
    return None


def elapsed(started, ended=None):
    start = parse_time(started)
    end = parse_time(ended) if ended else datetime.now(timezone.utc)
    if not start or not end:
        return None
    seconds = max(0, int((end - start).total_seconds()))
    return f"{seconds // 60} min {seconds % 60} s" if seconds >= 60 else f"{seconds} s"
