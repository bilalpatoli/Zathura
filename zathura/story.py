"""A read-only scientific narrative; only Begin Mission invokes the research worker."""
import html
import json
import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from zathura.engine import get_lightcurve
from zathura.engine.data import ROOT, cache_path
from zathura.engine.vetting import VERDICTS, vet_candidate, vetting_targets
from zathura.evidence import AGENT_NAMES, TRANSPORT_NAMES, config_change, proposal_chains

CYAN, VIOLET, ROSE = "#60e9e1", "#b89aff", "#ff91ab"


def safe(text):
    return html.escape(re.sub(r"\b[0-9a-f]{32}\b", "an earlier experiment", str(text)))


def chapter(number, title, text, vocabulary):
    st.markdown(f'<div class="chapter"><div class="eyebrow">CHAPTER {number}</div>'
                f'<h2>{title}</h2><p>{text}</p><div class="vocabulary">{vocabulary}</div></div>', unsafe_allow_html=True)


def provenance(source):
    if source == "placeholder":
        st.markdown('<span class="placeholder">PLACEHOLDER engine</span>', unsafe_allow_html=True)
    elif source == "real":
        st.caption("REAL KEPLER PHOTOMETRY · synthetic injections only in the tuning benchmark")
    else:
        st.caption("Data provenance not recorded")


def plot(fig, height=320):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=15, r=15, t=35, b=35),
                      font=dict(color="#ccd8ed", size=13),
                      legend=dict(orientation="h", y=1.16, x=0))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#243044", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


@st.cache_data(show_spinner=False)
def source_inventory(target_ids):
    curves = [get_lightcurve(target) for target in target_ids]
    return len(curves), sum(len(curve["time"]) for curve in curves)


@st.cache_data(show_spinner=False)
def source_view(target_id):
    return get_lightcurve(target_id), vet_candidate(target_id)


@st.cache_data(show_spinner=False)
def candidate_view(kepid):
    return vet_candidate(f"KIC {kepid}")


def bubble(role, text, color):
    st.markdown(f'<div class="thought" style="border-left-color:{color}">'
                f'<div class="eyebrow" style="color:{color}">{safe(role)}</div><p>{safe(text)}</p></div>', unsafe_allow_html=True)


def memory_veto(events):
    for event in reversed(events):
        proposal = event.get("payload", {})
        if event.get("step") not in ("hypothesis", "revision") or not isinstance(proposal, dict):
            continue
        influence = proposal.get("memory_influence", {})
        if influence.get("memory_ids") and influence.get("explanation", "").startswith("Deterministic memory veto"):
            return proposal
    return None


def measured_chain(chains):
    return next((c for c in reversed(chains) if (c.get("completion") or {}).get("measurements")),
                chains[-1] if chains else None)


def folded_trace(fig, phase, flux, name, color, row=None, col=None):
    args = {"row": row, "col": col} if row else {}
    fig.add_trace(go.Scattergl(x=phase, y=flux, mode="markers", name=name,
                              marker=dict(size=3, color=color, opacity=.35), showlegend=False), **args)
    if len(phase) < 5:
        return
    edges = np.linspace(float(np.min(phase)), float(np.max(phase)), 45)
    x, y = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        keep = (phase >= lo) & (phase < hi)
        if keep.sum() >= 3:
            x.append(float(np.mean(phase[keep])))
            y.append(float(np.median(flux[keep])))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name,
                            marker=dict(size=4), line=dict(color=color, width=2)), **args)


def render_story(state, worker):
    st.markdown('<div class="eyebrow">REAL PHOTONS. TESTABLE IDEAS. SHARED MEMORY.</div>', unsafe_allow_html=True)
    st.title("Zathura: An autonomous AI astronomer")
    st.markdown("### A faint shadow. A scientific argument. A memory that changes the next experiment.")
    st.caption("Real NASA Kepler light curves. Live AI research. Transparent limitations. No planet discoveries claimed.")
    iterations = st.slider("Mission length · research iterations", 1, 10, 3, key="story_iterations")
    status = worker.status()
    if st.button("Begin Mission", type="primary", width="stretch", key="story_begin", disabled=status["running"]):
        worker.launch(iterations)
        st.info("Mission launched. This story updates live; API calls may incur charges.")
    st.caption("Research running" if status["running"] else "Ready to research · live API calls may incur charges")
    if status["error"]:
        st.error("Mission stopped: " + status["error"] + ". Existing observations remain available below.")

    demo = next((t for t in state["targets"] if t["target_id"] == "KIC 6665695"), None)
    curve = candidate = None
    if demo:
        try:
            curve, candidate = source_view(demo["target_id"])
        except (ValueError, FileNotFoundError):
            st.info("The demonstration curve is unavailable. Rebuild the committed Kepler cache.")
    chapter("01", "The Signal", "A distant star sends a nearly steady stream of light. We begin with an actual Kepler quarter, not an artist’s impression.",
            "Space-based differential photometry · PDCSAP flux · ppm precision estimated from point-to-point scatter")
    if curve:
        dev = tuple(t["target_id"] for t in state["targets"] if t.get("role") == "dev")
        stars, cadences = source_inventory(dev)
        left, right = st.columns([3, 1])
        with left:
            fig = go.Figure(go.Scattergl(x=curve["time"], y=(curve["flux"] - 1) * 1e6,
                mode="markers", marker=dict(size=2, color=CYAN)))
            fig.update_xaxes(title="Time · BKJD (days)")
            fig.update_yaxes(title="Relative brightness · ppm")
            plot(fig)
        with right:
            st.metric("Stars in dev sample", stars)
            st.metric("Source cadences · dev sample", f"{cadences:,}")
            differences = np.diff(curve["flux"])
            scatter = 1.4826 * np.median(np.abs(differences - np.median(differences))) / np.sqrt(2) * 1e6
            st.metric("Point-to-point scatter · ppm", f"{scatter:.1f}")
        st.caption(f"Kepler-601 · quarter {curve['quarter']} · {np.ptp(curve['time']):.2f} days. Source cadences precede detector clipping; scatter is MAD of flux differences / √2, not a red-noise precision guarantee. Benchmark injections use separate quiet stars.")
    else:
        st.info("Cached Kepler data is missing. Restore the data cache to see the signal; a mission is not required to browse it.")

    chapter("02", "The Shadow", "A crossing companion can briefly dim its star. Aligning repeated dips reveals a pattern that a single passage can hide.",
            "Box Least Squares periodogram · transit depth · orbital period · phase folding")
    if candidate:
        p = candidate["plots"]
        duration = candidate["duration_days"]
        center_index = int(np.argmin(np.abs(p["phase_days"])))
        center = p["time"][center_index] - p["phase_days"][center_index]
        window = max(3 * duration, .3)
        keep = np.abs(p["time"] - center) < window
        folded = np.abs(p["phase_days"]) < window
        fig = make_subplots(rows=1, cols=2, subplot_titles=("One passage", "Repeated passages aligned"))
        fig.add_trace(go.Scatter(x=(p["time"][keep] - center) * 24, y=(p["flat_flux"][keep] - 1) * 1e6,
            mode="markers+lines", line=dict(color=CYAN), showlegend=False), row=1, col=1)
        folded_trace(fig, p["phase_days"][folded] * 24, (p["flat_flux"][folded] - 1) * 1e6, "Observed flux", VIOLET, 1, 2)
        for col in (1, 2):
            fig.add_vrect(x0=-duration * 12, x1=duration * 12, fillcolor=VIOLET, opacity=.12, line_width=0, row=1, col=col)
            fig.update_xaxes(title="Hours from transit center", row=1, col=col)
        fig.update_yaxes(title="Brightness · ppm", row=1, col=1)
        plot(fig)
        for col, label, value in zip(st.columns(2), ("Measured period · days", "Measured depth · ppm"),
                                     (candidate["period_days"], candidate["depth_ppm"])):
            col.metric(label, f"{value:.4f}")
        st.caption("Blind baseline BLS peak on an un-injected, catalogued planet host. A periodic dip alone is not proof of a planet.")
    else:
        st.info("Restore the cached planet-host curve to reveal the shadow.")

    chains = proposal_chains(state["latest"])
    latest = chains[-1] if chains else None
    chapter("03", "The Scientist Thinks", "The Research Scientist reads methods from the literature and proposes one bounded change to the detector. A hypothesis earns its place by being testable—not by sounding convincing.",
            "Literature-grounded hypothesis generation via Exa neural search")
    if latest:
        proposal = latest["proposal"]
        bubble("Research Scientist · hypothesis, not a finding", proposal.get("hypothesis", "Proposal not recorded"), CYAN)
        sources = proposal.get("cited_sources", [])
        left, right = st.columns([3, 1])
        with left:
            for source in sources:
                st.link_button(("[MOCK] " if source.get("mock") else "") + source["title"], source["url"], width="stretch")
        with right:
            st.metric("Sources cited", len(sources))
    else:
        st.info("Launch a mission to see a literature-grounded hypothesis.")

    chapter("04", "The Skeptic Pushes Back", "A second agent challenges the proposal before compute is spent. Approved experiments change one variable on identical development examples; the evaluator—not the debate—decides whether it helped.",
            "Adversarial peer review · controlled single-variable ablation · balanced accuracy")
    if latest:
        debate = measured_chain(chains)
        proposal, completion = debate["proposal"], debate["completion"]
        if debate is not latest:
            st.caption(f"Most recent measured experiment · iteration {proposal.get('iteration', '?')}. The newer proposal has no measurement; it is not assigned a score.")
        reviews = [e["payload"] for e in state["latest"] if e.get("step") == ("review_revision" if debate["revision"] else "review")
                   and e["payload"].get("iteration") == proposal.get("iteration")]
        left, right = st.columns(2)
        with left:
            bubble("Scientist", proposal.get("hypothesis", "Awaiting proposal"), CYAN)
        with right:
            review = reviews[-1] if reviews else {}
            bubble("Skeptic · " + review.get("decision", "pending"), review.get("rationale", "Awaiting peer review."), VIOLET)
        before = proposal.get("context_metrics", {}).get("balanced_accuracy")
        after = (completion or {}).get("measurements", {}).get("balanced_accuracy")
        if before is not None and after is not None:
            provenance(completion.get("record", {}).get("data_source"))
            st.markdown(f'<div class="big-result">{before:.4f} <span>→</span> {after:.4f}</div>', unsafe_allow_html=True)
            st.caption("Incumbent → experiment dev balanced accuracy. " +
                       ("Accepted point-estimate improvement; not a significance claim." if completion.get("accepted") else "The experiment did not improve the incumbent."))
        else:
            st.info("No experiment measurement yet. Rejected proposals are not assigned fabricated scores.")
    else:
        st.info("Launch a mission to see the scientific debate and its measured outcome.")

    chapter("05", "It Remembers", "A useful scientist does not start from zero each time. Shared memory can veto a previously unsuccessful direction and change what gets tested next.",
            "Persistent cross-agent episodic memory on past.dev · author-attributed recall")
    proposal = memory_veto(state["latest"])
    historical = proposal is None
    proposal = proposal or memory_veto(state["events"])
    if proposal:
        if historical:
            st.info("No verified memory veto occurred in the latest mission. This is the most recent recorded example from an earlier mission; its original read source is shown below.")
        memory_id = proposal["memory_influence"]["memory_ids"][0]
        entry = next((m for m in state["memories"] if m["memory_id"] == memory_id), {})
        receipt = next((r for r in proposal.get("memory_reads", []) if r["memory_id"] == memory_id), {})
        author = receipt.get("author", entry.get("agent_id", "Author not recorded"))
        with st.container(border=True):
            memory_loop = next((e.get("loop_id") for e in state["events"] if e.get("payload") is proposal), None)
            vetoes = sum(1 for e in state["events"] if memory_loop and e.get("loop_id") == memory_loop and memory_veto([e]))
            heading, count = st.columns([3, 1])
            heading.subheader("Memory changed a decision")
            count.metric("Verified vetoes · this example’s mission", vetoes)
            left, middle, right = st.columns(3)
            with left:
                bubble("Wanted to try", config_change(proposal.get("current_best_config"), proposal.get("first_choice_config")), ROSE)
            with middle:
                content = entry.get("content", {})
                measurement = content.get("measurements", {}).get("balanced_accuracy") if isinstance(content, dict) else None
                remembered = (f"Earlier dev balanced accuracy: {measurement:.4f}; this result was not accepted as an improvement."
                              if measurement is not None else "A recorded prior failure or grounded rejection vetoed this direction.")
                bubble("Remembered · " + AGENT_NAMES.get(author, author), remembered, VIOLET)
                if measurement is not None:
                    provenance(content.get("data_source"))
            with right:
                bubble("Tested instead", config_change(proposal.get("current_best_config"), proposal.get("proposed_config")), CYAN)
            st.code(memory_id, language=None)
            st.caption("Memory author: " + AGENT_NAMES.get(author, author) + " · read via " +
                       TRANSPORT_NAMES.get(receipt.get("read_via"), "not recorded"))
            if receipt.get("fallback_reason"):
                st.caption("Mirror fallback after remote " + receipt["fallback_reason"])
    else:
        st.info("Launch a mission to see whether recalled evidence changes a decision. No verified memory veto is recorded for this mission yet.")

    render_finale(state)


def render_finale(state):
    chapter("FINALE", "Planet or Impostor?", "Not every shadow is a planet. Five transparent checks look for noise, alternating eclipses, a second dip, a pointed shape, or a companion too large to be planetary.",
            "Odd–even eclipse test · phase-0.5 secondary eclipse · morphology · radius from transit depth")
    targets = [t for t in vetting_targets() if cache_path(t["kepid"]).exists()]
    if not targets:
        st.info("Vetting examples are not cached. Run scripts/build_vetting.py to restore the real-source sample.")
    else:
        category = st.radio("Compare real catalog examples", ["Planet host", "Eclipsing binary"], horizontal=True, key="story_category")
        choices = [t for t in targets if t["truth"] == ("planet" if category == "Planet host" else "EB")]
        if choices:
            lookup = {t["kepid"]: t for t in choices}
            ids = list(lookup)
            default = ids.index(6665695) if 6665695 in ids else 0
            labels = {k: lookup[k]["name"] if category == "Planet host" else f"Binary {chr(65 + i)}" for i, k in enumerate(ids)}
            selected = st.selectbox("Choose an observed star", ids, index=default, format_func=labels.get, key="story_target_" + category)
            target = lookup[selected]
            try:
                result = candidate_view(selected)
            except (ValueError, FileNotFoundError):
                st.info("Insufficient source data for this diagnostic.")
            else:
                color = CYAN if result["verdict"] == VERDICTS[0] else ROSE if result["verdict"] == VERDICTS[1] else VIOLET
                st.markdown(f'<div class="verdict" style="color:{color};border-color:{color}">{result["verdict"]}</div>', unsafe_allow_html=True)
                st.write(result["summary"])
                st.link_button("Catalog classification · " + category, target["source"])
                st.caption("Binary letters are demo labels; the source link identifies each real system. Catalog truth does not enter the detector.")
                for test in result["tests"]:
                    mark = "✓" if test["passed"] is True else "✕" if test["passed"] is False else "?"
                    color = CYAN if test["passed"] is True else ROSE if test["passed"] is False else VIOLET
                    value = f"{test['value']:.3f} {test['unit']}" if test["value"] is not None else "Insufficient coverage"
                    st.markdown(f'<div class="evidence-check"><span style="color:{color}">{mark} <b>{test["name"]}</b> · {value}</span>'
                                f'<div class="vocabulary">{safe(test["threshold"])}</div><div>{safe(test["explanation"])}</div></div>', unsafe_allow_html=True)
                p, duration = result["plots"], result["duration_days"]
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Odd vs even passages", "Search for a secondary eclipse"))
                near = np.abs(p["phase_days"]) < duration * 2
                for odd, name, color in ((True, "Odd transits", CYAN), (False, "Even transits", VIOLET)):
                    keep = near & (p["odd"] == odd)
                    folded_trace(fig, p["phase_days"][keep] * 24, p["flat_flux"][keep], name, color, 1, 1)
                folded_trace(fig, p["phase"], p["flat_flux"], "Folded flux", ROSE, 1, 2)
                width = duration / result["period_days"] / 2
                fig.add_vrect(x0=.5 - width, x1=.5 + width, fillcolor=ROSE, opacity=.2, line_width=0, row=1, col=2)
                fig.update_xaxes(title="Hours from primary center", row=1, col=1)
                fig.update_xaxes(title="Orbital phase · secondary window shaded", row=1, col=2)
                fig.update_yaxes(title="Normalized flux", row=1, col=1)
                plot(fig, 360)
                st.caption("Points are measured flattened flux; lines are phase-bin medians. Not a physical binary fit. Red noise, grazing planets and dilution can fool these heuristics.")
    report_path = ROOT / "data/vetting_results.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        st.subheader("How often does this screen get it right?")
        counts = [[report["confusion"][truth][verdict] for verdict in VERDICTS] for truth in ("planet", "EB")]
        fig = go.Figure(go.Heatmap(z=counts, x=["Planet candidate", "Binary", "Inconclusive"], y=["Known planet hosts", "Known binaries"],
                                  text=counts, texttemplate="%{text}", colorscale=[[0, "#14223b"], [1, "#41636e"]], showscale=False))
        plot(fig, 230)
        st.caption("True catalog class (rows) versus blind screening verdict (columns). Selected convenience sample, not held-out classifier validation. Three planet hosts are falsely flagged as binaries; the long-period Kepler-22 host is inconclusive.")
    render_scoreboard(state)


def render_scoreboard(state):
    st.subheader("The scoreboard · measured, not imagined")
    baseline = state.get("baseline")
    if not baseline:
        st.info("Launch a mission to see development and blind held-out performance.")
        return
    base = baseline["measurements"].get("balanced_accuracy")
    values = [r["measurements"]["balanced_accuracy"] for r in state["current"] if r.get("measurements")]
    best = max([base, *values]) if base is not None else None
    provenance(baseline.get("data_source"))
    if best is not None:
        st.markdown(f'<div class="big-result">{base:.4f} <span>→</span> {best:.4f}</div>', unsafe_allow_html=True)
        st.caption("Baseline → AI-selected best development balanced accuracy. Selection on dev is not evidence of generalization.")
    loop_id = state["latest"][-1].get("loop_id") if state["latest"] else None
    holdout = next((r for r in reversed(state["holdout"]) if r.get("loop_id") == loop_id), None) if loop_id else None
    if not holdout:
        st.info("Blind held-out validation appears after this mission completes.")
        return
    m = holdout["measurements"]
    measured = m.get("balanced_accuracy")
    delta = holdout.get("evaluation_result", {}).get("delta")
    provenance(holdout.get("data_source"))
    if measured is not None:
        st.metric("Blind held-out balanced accuracy · selected configuration", f"{measured:.4f}")
    if measured is not None and delta is not None:
        direction = "lower than" if delta < 0 else "higher than" if delta > 0 else "equal to"
        st.write(f"On unseen stars, the selected configuration scored {direction} the holdout baseline ({measured - delta:.4f}).")
    if "balanced_accuracy_ci_low" in m and "balanced_accuracy_ci_high" in m:
        st.caption(f"Star-bootstrap 95% interval: [{m['balanced_accuracy_ci_low']:.4f}, {m['balanced_accuracy_ci_high']:.4f}].")
    st.caption("Disjoint dev / holdout stars; synthetic transit injections into real Kepler photometry. No agent sees held-out labels. Small benchmark; this is detector tuning, not a planet-discovery claim.")
