"""Plain-language, read-only pitch views over the existing science and mission records."""
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from zathura.engine.data import ROOT, cache_path
from zathura.engine.vetting import VERDICTS, vetting_targets
from zathura.evidence import proposal_chains
from zathura.story import CYAN, ROSE, VIOLET, candidate_view, plot

PITCH = "Zathura is an AI scientist that hunts for planets in NASA telescope data, then teaches itself to hunt better."
SCORE_NOTE = "Scores use simulated planets added to real Kepler stars."
STEP_ICONS = {
    "Idea": '<path d="M9 18h6M10 22h4M8 14a7 7 0 1 1 8 0l-1 4H9z"/>',
    "Skeptic": '<path d="m12 2 8 3v6c0 5-8 10-8 10S4 16 4 11V5z"/><path d="m8 11 3 3 5-6"/>',
    "Result": '<path d="M9 2h6M10 2v7L4 20h16L14 9V2M7 15h10"/>',
    "Remembered": '<path d="M12 5c-5-7-12 2-7 6-6 4 1 13 7 8 6 5 13-4 7-8 5-4-2-13-7-6zM12 5v14M7 7l3 3M7 16l3-3M17 7l-3 3M17 16l-3-3"/>',
}


def step_header(title):
    st.markdown(f'<div class="simple-step"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
                f'{STEP_ICONS[title]}</svg>{title}</div>', unsafe_allow_html=True)


def source_note(source):
    st.caption("Real Kepler stars" if source == "real" else
               "Placeholder demonstration — not measured on real stars" if source == "placeholder" else
               "Data source not recorded")


def render_find(state):
    st.title("Find")
    st.write("Is that dip a planet crossing a star — or two stars crossing each other?")
    if not state["targets"]:
        st.info("No star data is available yet. Restore the data cache to explore.")
        return
    targets = [t for t in vetting_targets() if cache_path(t["kepid"]).exists()]
    if not targets:
        st.info("The example stars are not available yet. Restore the data cache to explore.")
        return
    lookup = {t["kepid"]: t for t in targets}
    ids = list(lookup)
    binaries = [t["kepid"] for t in targets if t["truth"] == "EB"]
    labels = {key: (f"Two-star example {binaries.index(key) + 1}" if key in binaries else target["name"])
              for key, target in lookup.items()}
    selected = st.selectbox("Pick a star", ids, index=ids.index(6665695) if 6665695 in ids else 0,
                            format_func=labels.get, key="find_star")
    try:
        result = candidate_view(selected)
    except (ValueError, OSError, RuntimeError):
        st.info("We could not read enough brightness data for this star. Try another example.")
        return
    p = result["plots"]
    window = max(3 * result["duration_days"], .3)
    keep = np.abs(p["phase_days"]) < window
    fig = go.Figure(go.Scattergl(x=p["phase_days"][keep] * 24, y=p["flat_flux"][keep],
                                 mode="markers", marker=dict(size=4, color=CYAN), name="Brightness"))
    half = result["duration_days"] * 12
    fig.add_vrect(x0=-half, x1=half, fillcolor=VIOLET, opacity=.25, line_width=0)
    fig.update_xaxes(title="Hours from the dip")
    fig.update_yaxes(title="Brightness")
    verdict = {VERDICTS[0]: "Planet candidate", VERDICTS[1]: "Impostor: two stars", VERDICTS[2]: "No clear signal"}[result["verdict"]]
    color = CYAN if result["verdict"] == VERDICTS[0] else ROSE if result["verdict"] == VERDICTS[1] else VIOLET
    left, right = st.columns([2, 1])
    with left:
        plot(fig, 320)
        st.caption("Real NASA brightness data · repeated dips lined up; shaded area marks the possible crossing.")
    with right:
        st.markdown(f'<div class="simple-verdict" style="color:{color}">{verdict}</div>', unsafe_allow_html=True)
        st.caption("Screening only — not confirmation. This screen can be wrong; see Technical details for its measured mistakes.")
        for label, test in zip(("Strong dip?", "Alternating dips equal?", "No second eclipse?"), result["tests"][:3]):
            passed = test["passed"]
            mark, color = ("✓", "#71e6b0") if passed is True else ("✕", ROSE) if passed is False else ("?", VIOLET)
            st.markdown(f'<div class="simple-check" style="color:{color}">{mark} {label}</div>', unsafe_allow_html=True)
            st.caption("Yes" if passed is True else "No" if passed is False else "Not enough data",
                       help=f'{test["name"]}: {test["threshold"]}. {test["explanation"]}')
    with st.expander("More checks"):
        for label, test in zip(("Does the dip have a planet-like shape?", "Is the companion small enough?"), result["tests"][3:]):
            st.write(("✓ " if test["passed"] is True else "✕ " if test["passed"] is False else "? ") + label)
            st.caption("Yes" if test["passed"] is True else "No" if test["passed"] is False else "Not enough data",
                       help=f'{test["threshold"]}. {test["explanation"]}')
        st.caption("The size estimate assumes a Sun-sized star." if result["radius_assumed"] else
                   "The size estimate uses the star size in the NASA catalog.")
        st.link_button("Known system reference", lookup[selected]["source"])
    if st.toggle("Show original brightness", key="find_raw", help="Raw normalized Kepler PDCSAP photometry before flattening and phase folding."):
        raw = go.Figure(go.Scattergl(x=p["time"], y=p["flux"], mode="markers", marker=dict(size=2, color=CYAN)))
        raw.update_xaxes(title="Time (days)")
        raw.update_yaxes(title="Brightness")
        plot(raw, 230)


def attempt_rows(state):
    chains = [c for c in proposal_chains(state["latest"]) if not c["superseded"]]
    incumbent = (state.get("baseline") or {}).get("measurements", {}).get("balanced_accuracy")
    rows = []
    for chain in chains:
        proposal, completion = chain["proposal"], chain["completion"] or {}
        step = "review_revision" if chain["revision"] else "review"
        review = next((e["payload"] for e in reversed(state["latest"]) if e.get("step") == step
                       and e["payload"].get("iteration") == proposal.get("iteration")), {})
        after = completion.get("measurements", {}).get("balanced_accuracy")
        rows.append({"proposal": proposal, "review": review, "completion": completion,
                     "before": incumbent, "after": after})
        if after is not None:
            incumbent = max(incumbent, after) if incumbent is not None else after
    return rows


def idea(proposal):
    before, after = proposal.get("current_best_config", {}), proposal.get("proposed_config", {})
    changes = [key for key, value in after.items() if key in before and value != before[key]]
    descriptions = {"period_grid_size": "Try a different search spacing to spot repeating dips.",
                    "detrend_window": "Change how slowly changing starlight is smoothed away.",
                    "outlier_sigma": "Change which unusual brightness points are kept.",
                    "detection_threshold": "Change how strong a dip must be to count.",
                    "duration_min_days": "Look for a different range of crossing lengths.",
                    "duration_max_days": "Look for a different range of crossing lengths."}
    return descriptions.get(changes[0], "Test the recorded idea on the same stars.") if changes else "Test the recorded idea on the same stars."


def run_state(state, running):
    """'none' | 'running' | 'complete' | 'stopped' | 'interrupted' for the latest recorded run."""
    latest = state.get("latest") or []
    if not latest:
        return "none"
    if running:
        return "running"
    steps = {event.get("step") for event in latest}
    if "loop_complete" in steps:
        return "complete"
    if "error" in steps:
        return "stopped"
    return "interrupted"


def render_improve(state, worker):
    st.title("Improve")
    st.write("The AI suggests a change. A skeptic challenges it. Only a measured improvement is kept.")
    status = worker.status()
    if st.button("Start AI research", type="primary", key="simple_start", disabled=status["running"]):
        if worker.launch(3):
            st.rerun()
        else:
            st.warning("Research is already running; this click did not start another run.")
    st.caption("Research is running. The attempts below update automatically." if status["running"] else "Uses live services when configured; API calls may incur charges.")
    current_run_state = run_state(state, status["running"])
    if current_run_state == "interrupted":
        st.warning("The last run stopped before it finished (for example, the app restarted). Its recorded attempts are shown below; it has no final exam result.")
    if status["error"]:
        st.warning("Research stopped. Earlier results remain available; see Technical details.")
    elif current_run_state == "stopped":
        st.warning("The last run stopped with an error. Earlier results remain available; see Technical details.")
    rows = attempt_rows(state)
    if not rows:
        st.info("Measuring the starting method; the first attempt appears shortly." if status["running"]
                else "Start AI research to see the first idea, challenge, test and memory.")
        return
    st.caption(SCORE_NOTE)
    source = (state.get("baseline") or {}).get("data_source")
    if source != "real":
        source_note(source)
    for index, row in enumerate(rows, 1):
        proposal, review, completion = row["proposal"], row["review"], row["completion"]
        with st.container(border=True):
            st.subheader(f"Attempt {index}")
            columns = st.columns(4)
            with columns[0]:
                step_header("Idea")
                st.write(idea(proposal))
                sources = proposal.get("cited_sources", [])
                source = next((s for s in sources if s.get("url", "").startswith(("https://", "http://"))), None)
                if source:
                    st.link_button("Read paper", source["url"])
                    if source.get("mock"):
                        st.caption("Mock source · demonstration only")
                else:
                    st.caption("No paper link recorded")
            with columns[1]:
                step_header("Skeptic")
                decision = review.get("decision", completion.get("decision"))
                decision = decision if isinstance(decision, str) else (decision or {}).get("decision")
                st.write("Approved" if decision == "approve" else "Rejected" if decision == "reject" else "Awaiting review")
                rationale = review.get("rationale", "")
                if rationale:
                    st.caption("Too costly to test within the research budget." if decision == "reject" and ("runtime" in rationale.lower() or "budget" in rationale.lower())
                               else "No concrete reason to block this test was found." if decision == "approve"
                               else "The reviewer found a problem; see Technical details for the full review.")
            with columns[2]:
                step_header("Result")
                before, after = row["before"], row["after"]
                if after is not None and before is not None:
                    delta = after - before
                    color = "#71e6b0" if delta > 0 else ROSE if delta < 0 else "#ccd8ed"
                    arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
                    st.markdown(f'<div class="simple-score" style="color:{color}">{before:.1%} → {after:.1%} {arrow}</div>', unsafe_allow_html=True)
                    significant = ((completion.get("record") or {}).get("evaluation_result") or {}).get("significant_vs_best")
                    st.caption(("Kept: higher score" + ("" if significant else ", but within noise (not statistically significant)"))
                               if completion.get("accepted") else "Not kept as an improvement")
                else:
                    st.write("Not tested" if decision == "reject" else "Waiting for a result")
            with columns[3]:
                step_header("Remembered")
                influence = proposal.get("memory_influence", {})
                changed = bool(influence.get("memory_ids")) and influence.get("explanation", "").startswith("Deterministic memory veto")
                if changed:
                    st.success("Memory changed this decision: earlier evidence ruled out the first idea.")
                elif completion.get("result_memory_id"):
                    st.write("This result was saved to guide the next idea.")
                elif proposal.get("memory_id"):
                    st.write("This idea was saved for the next review.")
                else:
                    st.write("No saved memory recorded yet.")


def score_history(state):
    baseline = (state.get("baseline") or {}).get("measurements", {}).get("balanced_accuracy")
    rows = attempt_rows(state)
    values = [row["after"] for row in rows]
    best, running = [], baseline
    for value in values:
        if value is not None:
            running = max(running, value) if running is not None else value
        best.append(running)
    return baseline, values, best


def latest_holdout(state):
    loop_id = state["latest"][-1].get("loop_id") if state["latest"] else None
    return next((r for r in reversed(state["holdout"]) if loop_id and r.get("loop_id") == loop_id), None)


def render_results(state, running=False):
    st.title("Results")
    st.write("Did the AI get better — and did that carry over to stars it had not practiced on?")
    baseline, values, best = score_history(state)
    current_run_state = run_state(state, running)
    left, right = st.columns([2, 1])
    with left:
        if baseline is None:
            st.info("Start AI research to see measured scores.")
        else:
            source_note(state["baseline"].get("data_source"))
            x = list(range(1, len(values) + 1))
            fig = go.Figure(go.Scatter(x=x, y=values, mode="markers+lines", name="Each attempt",
                                       connectgaps=False, marker=dict(size=10, color=VIOLET)))
            fig.add_trace(go.Scatter(x=[0, *x], y=[baseline, *best], mode="lines", name="Best so far", line=dict(color=CYAN, width=3)))
            fig.add_trace(go.Scatter(x=[0, max(len(values), 1)], y=[baseline, baseline], mode="lines", name="Starting score",
                                     line=dict(color="#aeb9cb", dash="dash")))
            fig.update_xaxes(title="Attempt", dtick=1)
            fig.update_yaxes(title="Score", range=[0, 1], tickformat=".0%")
            plot(fig, 320)
            st.caption("Higher is better. Attempts that were not tested have no score.",
                       help="Development balanced accuracy = (injection recovery rate + 1 − false alarm rate) / 2; not statistical evidence of improvement.")
    with right, st.container(border=True):
        st.subheader("Final exam (held-out stars)")
        holdout = latest_holdout(state)
        measured = (holdout or {}).get("measurements", {}).get("balanced_accuracy")
        if measured is None:
            message = {
                "running": "The final exam appears when this research mission finishes.",
                "interrupted": "This run was interrupted before the final exam, so there is no held-out result for it.",
                "stopped": "This run stopped with an error before the final exam.",
            }.get(current_run_state, "The final exam appears when this research mission finishes.")
            st.info(message)
        else:
            source_note(holdout.get("data_source"))
            st.metric("Score on unseen stars", f"{measured:.1%}", help="Balanced accuracy on held-out stars; these stars were not used to select the best configuration.")
            delta = holdout.get("evaluation_result", {}).get("delta")
            if delta is None:
                st.write("Starting score not recorded for this final exam.")
            else:
                direction = "Lower than" if delta < 0 else "Higher than" if delta > 0 else "Equal to"
                st.write(f"{direction} the starting score of {measured - delta:.1%} on the same unseen stars.")
                if delta != 0 and (holdout.get("evaluation_result") or {}).get("significant_vs_baseline") is False:
                    st.caption("The difference is within noise (not statistically significant).")
            st.caption("A better practice score does not guarantee a better final exam.")
    st.caption(SCORE_NOTE)


def render_validation():
    st.subheader("Vetting confusion table")
    path = ROOT / "data/vetting_results.json"
    try:
        report = json.loads(path.read_text())
        rows = [{"Catalog truth": truth, **report["confusion"][key]}
                for key, truth in (("planet", "Planet host"), ("EB", "Eclipsing binary"))]
    except (OSError, ValueError, KeyError, TypeError):
        st.info("The recorded vetting validation report is unavailable.")
        return
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Selected known systems, not a held-out classifier benchmark. Screening can misclassify real planets; no discoveries claimed.")
