"""Zathura mission dashboard. Run with: streamlit run app.py."""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from zathura.dashboard import Mission, snapshot
from zathura.engine import inspect_target
from zathura.evidence import AGENT_NAMES, TRANSPORT_NAMES, config_change, memory_activity, proposal_chains


def badge(source):
    if source == "placeholder":
        st.markdown('<span class="placeholder">PLACEHOLDER engine</span>', unsafe_allow_html=True)
    elif source == "real":
        st.markdown('<span class="real-source">REAL Kepler source</span>', unsafe_allow_html=True)


def metric(label, value, source):
    badge(source)
    st.metric(label, value)


def chart(fig, title, source):
    badge(source)
    fig.update_layout(template="plotly_dark", title=title, paper_bgcolor="#10182d",
                      plot_bgcolor="#10182d", margin=dict(l=35, r=15, t=65, b=40), height=300,
                      font=dict(size=13), title_font_size=15,
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, width="stretch")


def measurement_summary(measurements, source):
    badge(source)
    labels = {"balanced_accuracy": "Balanced accuracy", "recovery_rate": "Recovery",
              "false_alarm_rate": "False alarms", "runtime_s": "Runtime (s)"}
    parts = [f"{label}: {measurements[key]:.4f}" for key, label in labels.items() if key in measurements]
    st.write(" · ".join(parts) if parts else "Not run · no measured result")


def change_label(proposal):
    return config_change(proposal.get("current_best_config"), proposal.get("proposed_config"))


@st.cache_resource
def mission():
    return Mission()


@st.cache_data(show_spinner=False)
def target_diagnostic(target):
    return inspect_target(target)


def render_mission(state, worker):
    st.header("Mission Control")
    st.caption("An autonomous AI scientist • Kepler transit research • auditable decisions")
    mock = os.getenv("ZATHURA_MOCK") == "1"
    services = [f"OpenAI: {'LIVE configured' if os.getenv('OPENAI_API_KEY') and not mock else 'MOCK'}",
                f"Exa: {'LIVE configured' if os.getenv('EXA_API_KEY') and not mock else 'MOCK'}",
                f"past.dev: {'LIVE writes / local mirror reads' if os.getenv('PASTDEV_API_KEY') else 'LOCAL JSONL (not past.dev)'}"]
    st.info(" · ".join(services))
    st.caption("Configured does not guarantee service availability. Historical mock output retains its MOCK labels.")
    records = state["current"]
    analyzed = (state["baseline"].get("provenance", {}).get("host_count", 0)
                if state["baseline"] else 0)
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Stars analyzed", "Experiments run", "Accepted improvements", "Reviewer rejections"],
                                 [analyzed, sum(bool(r["measurements"]) for r in records), sum(r["accepted"] for r in records),
                                  sum(r["reviewer_decision"]["decision"] != "approve" for r in records)]):
        col.metric(label, value)
    st.caption(f"Latest mission only · {len(state['targets'])} cached source curves · {len(state['records'])} archived hypotheses. Stars analyzed counts the dev baseline.")
    baseline = state["baseline"]
    if baseline:
        source = baseline.get("data_source")
        base = baseline["measurements"]["balanced_accuracy"]
        best = max([base] + [r["measurements"]["balanced_accuracy"] for r in state["current"] if r["measurements"]])
        left, right = st.columns(2)
        with left:
            metric("Baseline dev balanced accuracy", f"{base:.4f}", source)
        with right:
            metric("Best dev balanced accuracy", f"{best:.4f}", source)
        st.caption("Comparison uses the latest mission only, avoiding mixed benchmark versions.")
    else:
        st.info("No baseline yet. Launch a mission to measure the dev benchmark.")
    hypothesis = next((e["payload"].get("hypothesis") for e in reversed(state["latest"])
                       if e.get("step") in ("hypothesis", "revision")), None)
    st.subheader("Current hypothesis")
    st.write(hypothesis or "Awaiting a research proposal.")
    status = worker.status()
    st.write(f"Research status: {'RUNNING · iteration ' + str(status['iteration']) if status['running'] else 'IDLE'}")
    if status["error"]:
        st.error(f"Mission stopped: {status['error']}. Check service availability and the event timeline.")
    iterations = st.slider("Research iterations", 1, 10, 4)
    if st.button("Launch Research Mission", type="primary", disabled=status["running"]):
        if worker.launch(iterations):
            st.success("Mission launched in a background worker. This page refreshes every five seconds.")
    st.caption("Launch may call paid APIs when configured. One mission per dashboard process; keep this process alive.")


def render_planets(state):
    st.header("Planet Detection")
    st.caption("Read-only diagnostic of un-injected Kepler curves using the baseline detector. Not benchmark recovery truth.")
    if not state["targets"]:
        st.info("No cached light curves. Rebuild them with scripts/build_catalog.py.")
        return
    labels = {t["target_id"]: f"{t.get('name', t['target_id'])} · {t['target_id']} · {t['role']}" for t in state["targets"]}
    target = st.selectbox("Star", list(labels), format_func=labels.get)
    try:
        curve = target_diagnostic(target)
    except (ValueError, OSError, RuntimeError) as exc:
        st.warning(f"Target diagnostic unavailable: {type(exc).__name__}")
        return
    source = curve["data_source"]
    badge(source)
    st.write(f"Source provenance: {curve['curve_origin']}")
    if "synthetic" in curve["curve_origin"].lower():
        st.warning("SIMULATED source curve — not NASA observations.")
    if curve.get("known_periods"):
        periods = ", ".join(f"{p:.4f}" for p in curve["known_periods"])
        st.info(f"Known-planet validation star · KOI periods: {periods} days.")
        st.markdown("Source: [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) KOI cumulative table.")
    else:
        st.info("Benchmark host star with no KOI or TCE; not certified planet-free.")
    st.write("UNVERIFIED transit candidate" if curve["detected"] else "No threshold-crossing candidate; showing the strongest BLS peak only.")
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Estimated period (days)", "Depth (ppm)", "Duration (days)", "Peak SDE"],
                                 [curve["period_days"], curve["depth"] * 1e6, curve["duration_days"], curve["sde"]]):
        with col:
            metric(label, f"{value:.4f}", source)
    left, right = st.columns(2)
    with left:
        chart(go.Figure(go.Scattergl(x=curve["time"], y=curve["flux"], mode="markers", marker=dict(size=3, color="#96a6cb"))),
              "Source flux · time (BKJD)", source)
    fig = go.Figure(go.Scattergl(x=curve["time"], y=curve["flat_flux"], mode="markers", name="Flattened", marker=dict(size=3, color="#39dce8")))
    mask = curve["in_transit"]
    fig.add_trace(go.Scattergl(x=curve["time"][mask], y=curve["flat_flux"][mask], mode="markers",
                              name="Candidate windows", marker=dict(size=5, color="#ad85ff")))
    with right:
        chart(fig, "Flattened flux · candidate windows", source)
    left, right = st.columns(2)
    with left:
        chart(go.Figure(go.Scatter(x=curve["periods"], y=curve["power"], line=dict(color="#ad85ff"))), "BLS power · period (days)", source)
    with right:
        chart(go.Figure(go.Scattergl(x=curve["phase"], y=curve["flat_flux"], mode="markers", marker=dict(size=3, color="#39dce8"))),
              "Phase folded · days from candidate center", source)


def render_timeline(state):
    st.header("AI Research Timeline")
    st.caption("Latest mission · one research thread per iteration · full sponsor provenance in Sponsor Evidence")
    if not state["latest"]:
        st.info("No mission events yet.")
        return
    for chain in proposal_chains(state["latest"]):
        proposal, completion = chain["proposal"], chain["completion"]
        iteration = proposal.get("iteration", "?")
        with st.expander(f"Iteration {iteration}{' · revision' if chain['revision'] else ''} · {change_label(proposal)}", expanded=not chain["superseded"]):
            st.markdown("**Scientist · research & hypothesis**")
            st.write(proposal.get("hypothesis", "Hypothesis not recorded"))
            if proposal.get("memory_influence", {}).get("memory_ids"):
                verified = proposal["memory_influence"].get("explanation", "").startswith("Deterministic memory veto")
                st.warning("Memory changed this decision · see Sponsor Evidence" if verified else "Model-reported influence · UNVERIFIED")
            reviews = [e["payload"] for e in state["latest"] if e.get("step") == ("review_revision" if chain["revision"] else "review")
                       and e["payload"].get("iteration") == iteration]
            for review in reviews:
                st.markdown(f"**Skeptic · {review.get('decision', 'pending').upper()}**")
                st.write(review.get("rationale", "Reasoning not recorded"))
            if chain["superseded"]:
                st.caption("Superseded by a revised proposal · this configuration was not run.")
            elif completion:
                st.markdown("**Engine · experiment & evaluation**")
                measurement_summary(completion.get("measurements", {}), completion.get("record", {}).get("data_source"))
                st.caption("Accepted dev improvement" if completion.get("accepted") else "No accepted improvement")
                refs = [proposal.get("memory_id"), *(r.get("memory_id") for r in reviews), completion.get("result_memory_id")]
                st.caption(f"Shared memory · {sum(bool(ref) for ref in refs)} writes recorded for this thread")
            else:
                st.caption("Awaiting review or execution. If the mission failed, inspect the status below.")
    for event in state["latest"]:
        if event.get("step") == "error":
            st.error(f"Iteration {event['payload'].get('iteration', '?')} stopped: {event['payload'].get('error_type', 'Unknown error')}")


def render_sponsors(state):
    st.header("Sponsor Evidence")
    st.caption("Recorded provenance, not inferred activity · latest mission · no remote requests triggered by this view")
    chains = proposal_chains(state["latest"])
    authors = {m["memory_id"]: m["agent_id"] for m in state["memories"]}
    left, right = st.columns(2)
    with left:
        st.subheader("Exa · research to result")
        st.caption("Exact SDK call · Exa.search_and_contents")
        st.code("zathura/agents/exa_search.py:search_papers", language=None)
        if not chains:
            st.info("Launch a mission to record search queries and citation chains.")
        for index, chain in enumerate(chains):
            proposal, completion = chain["proposal"], chain["completion"]
            with st.expander(f"Iteration {proposal.get('iteration', '?')}{' · revision' if chain['revision'] else ''} · source → result", expanded=index == 0):
                search = proposal.get("literature_search", {})
                st.caption("Exact query sent to Exa")
                st.write(search.get("query", "Not recorded in this legacy event"))
                cited = {s["url"] for s in proposal.get("cited_sources", [])}
                sources = search.get("sources", proposal.get("cited_sources", []))
                st.caption("Returned sources" if search else "Recorded citations only · complete search response not recorded")
                if search and sources:
                    st.caption("MOCK Exa response" if any(s.get("mock") for s in sources) else "Live Exa response")
                for source in sources:
                    label = ("[MOCK] " if source.get("mock") else "") + source["title"]
                    st.link_button(label, source["url"], width="stretch")
                    st.caption("Scientist cited → " + change_label(proposal) if source["url"] in cited else "Returned by Exa · not cited")
                st.markdown("**Hypothesis → experiment**")
                st.write(proposal.get("hypothesis", "Not recorded"))
                if chain["superseded"]:
                    st.caption("Superseded · not run; the revision has its own source chain.")
                elif completion:
                    st.caption("Review: " + completion.get("decision", "unknown"))
                    measurement_summary(completion.get("measurements", {}), completion.get("record", {}).get("data_source"))
                    if completion.get("measurements"):
                        st.caption("Experiment ID · " + completion.get("record", {}).get("experiment_id", "Not recorded"))
                        st.caption("Accepted dev improvement" if completion.get("accepted") else "Did not improve the incumbent")
                else:
                    st.caption("No experiment outcome recorded yet.")
    with right:
        st.subheader("past.dev · shared intelligence")
        st.markdown("**Scientist** `research-scientist` ↔ **Skeptic** `skeptic`")
        st.caption("Accepted writes hit past.dev. Context reads normally use its local mirror; remote recall is labelled separately. Local JSONL is not past.dev.")
        changes = [c["proposal"] for c in chains if c["proposal"].get("memory_influence", {}).get("memory_ids")]
        if not changes:
            st.info("No memory-driven decision change recorded for this mission.")
        for proposal in changes:
            influence = proposal["memory_influence"]
            verified = influence.get("explanation", "").startswith("Deterministic memory veto")
            with st.container(border=True):
                st.warning("Memory changed a decision" if verified else "Model-reported influence · UNVERIFIED")
                st.caption(f"Iteration {proposal.get('iteration', '?')} · evidence-backed veto" if verified else "Not proof of a causal veto")
                st.write("Before · " + config_change(proposal.get("current_best_config"), proposal.get("first_choice_config")))
                st.write("After · " + change_label(proposal))
                reads = {r["memory_id"]: r for r in proposal.get("memory_reads", [])}
                for memory_id in influence["memory_ids"]:
                    author = reads.get(memory_id, {}).get("author", authors.get(memory_id, "Not recorded"))
                    st.caption(f"Caused by {AGENT_NAMES.get(author, author)} · {memory_id}")
                    st.caption("Read via · " + TRANSPORT_NAMES.get(reads.get(memory_id, {}).get("read_via"), "Not recorded"))
                before, after = proposal.get("first_choice_config", {}), proposal.get("proposed_config", {})
                if before and after:
                    with st.expander("Compare full configurations"):
                        st.dataframe(pd.DataFrame([{"Parameter": key, "Before": str(before.get(key, "—")), "After": str(value)}
                                                   for key, value in after.items()]), hide_index=True, width="stretch")
    st.subheader("Scientist ↔ Skeptic · chronological exchange")
    st.caption("Read context means memories supplied to that agent after filtering. Write transport requires a local ingestion receipt; older missing provenance stays unverified.")
    activity = memory_activity(state["latest"], state["memories"])
    if activity:
        exchange = pd.DataFrame(activity)
        exchange["Activity"] = exchange["Agent"] + " · " + exchange["Action"].str.lower()
        exchange["Time"] = exchange["Time"].str[11:19]
        exchange["Transport"] = exchange["Transport"].replace({"past.dev accepted write": "past.dev write",
            "past.dev local mirror": "past.dev mirror", "past.dev remote": "past.dev remote"})
        st.caption("UTC · " + activity[0]["Time"][:10])
        st.dataframe(exchange[["Time", "Activity", "Memory author", "Memory ID", "Transport"]],
                     hide_index=True, width="stretch", height=360)
    else:
        st.info("No exchange recorded yet. Older events may not contain context-read receipts.")


def render_results(state):
    st.header("Experimental Results")
    st.caption("Latest mission comparisons only. Rejected hypotheses remain visible with missing metrics, never fabricated zeros.")
    rows = []
    baseline = state["baseline"]
    if baseline:
        rows.append({"label": "Baseline", "decision": "baseline", "data_source": baseline["data_source"], **baseline["measurements"]})
    for i, record in enumerate(state["current"], 1):
        rows.append({"label": f"Experiment {i}", "decision": record["reviewer_decision"]["decision"],
                     "data_source": record["data_source"], **record["measurements"]})
    measured = [r for r in rows if "balanced_accuracy" in r]
    if measured:
        best = max(measured, key=lambda r: r["balanced_accuracy"])
        rows.append(best | {"label": "Best config"})
        frame = pd.DataFrame(rows)
        frame["data_source"] = frame["data_source"].replace({"placeholder": "PLACEHOLDER engine"})
        fields = {"label": "Run", "decision": "Review", "data_source": "Source", "balanced_accuracy": "Balanced accuracy",
                  "recovery_rate": "Recovery", "false_alarm_rate": "False alarms", "runtime_s": "Runtime (s)"}
        display = frame[list(fields)].copy()
        for field in ("balanced_accuracy", "recovery_rate", "false_alarm_rate", "runtime_s"):
            display[field] = display[field].map(lambda value: "—" if pd.isna(value) else f"{value:.4f}")
        st.dataframe(display.rename(columns=fields), hide_index=True, width="stretch")
        columns = st.columns(2)
        for index, field in enumerate(("balanced_accuracy", "recovery_rate", "false_alarm_rate", "runtime_s")):
            with columns[index % 2]:
                fig = px.bar(frame, x="label", y=field, color="decision", labels=fields,
                             color_discrete_map={"baseline": "#39dce8", "approve": "#ad85ff", "reject": "#ff829f"})
                if field != "runtime_s":
                    fig.update_yaxes(range=[0, 1])
                chart(fig, fields[field], "placeholder" if any(r["data_source"] == "placeholder" for r in rows) else "real")
    else:
        st.info("No measured results yet.")
    st.subheader("Held-out test performance · not used for tuning")
    if not state["holdout"]:
        st.info("Holdout runs after a mission finishes. No held-out result yet.")
    else:
        result = state["holdout"][-1]
        st.caption(f"Latest completed holdout · loop {result.get('loop_id', 'unknown')} · {result.get('timestamp', '')}")
        measurements = result["measurements"]
        for col, key in zip(st.columns(4), ("recovery_rate", "false_alarm_rate", "balanced_accuracy", "runtime_s")):
            with col:
                value = measurements.get(key)
                metric(key.replace("_", " "), f"{value:.4f}" if value is not None else "—", result.get("data_source"))
        low, high = (measurements.get(key) for key in ("balanced_accuracy_ci_low", "balanced_accuracy_ci_high"))
        if low is not None and high is not None:
            badge(result.get("data_source"))
            st.caption(f"Balanced accuracy 95% confidence interval: [{low:.4f}, {high:.4f}]")
    with st.expander("All stored hypotheses (may span incompatible benchmark versions)"):
        archived = [{"Hypothesis": r["hypothesis"], "Review": r["reviewer_decision"]["decision"],
                     "Balanced accuracy": r["measurements"].get("balanced_accuracy"),
                     "Source": "PLACEHOLDER engine" if r["data_source"] == "placeholder" else r["data_source"]}
                    for r in state["records"][-50:]]
        st.caption("Most recent 50 archived hypotheses · not a cross-benchmark leaderboard")
        if archived:
            st.dataframe(pd.DataFrame(archived), hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="Zathura · Mission Control", page_icon="✦", layout="wide")
    st.markdown("""<style>
    .stApp {background:radial-gradient(ellipse at top right,#17183c 0%,#060b18 65%);}
    .block-container {padding-top:2rem;padding-bottom:2rem;}
    h1,h2,h3 {letter-spacing:.025em;} h1 {color:#39dce8!important;}
    h2 {font-size:1.7rem!important;} h3 {font-size:1.2rem!important;}
    [data-testid="stMarkdownContainer"] p {overflow-wrap:anywhere;}
    [data-testid="stLinkButton"] a {white-space:normal;text-align:left;justify-content:flex-start;}
    .real-source {display:inline-block;color:#79e4df;border:1px solid #276567;background:#102a34;
      border-radius:5px;padding:2px 8px;font-size:11px;font-weight:700;margin-bottom:6px;}
    .placeholder {display:inline-block;background:#392958;color:#e5caff;border:1px solid #ad85ff;
      border-radius:5px;padding:2px 8px;font-size:11px;font-weight:700;margin-bottom:6px;}
    [data-testid="stMetric"] {background:#10182dcc;border:1px solid #293554;border-radius:10px;padding:14px;}
    .stButton button[kind="primary"] {color:#060b18;font-weight:700;}
    </style>""", unsafe_allow_html=True)
    st.title("ZATHURA")
    st.caption("SEARCH THE STARS. QUESTION THE SIGNAL. REMEMBER THE SCIENCE.")
    st.caption("Real Kepler photometry · synthetic transit injections for benchmarking · no new planet discoveries claimed. Historical placeholder and mock output stays labelled.")
    live_dashboard()


@st.fragment(run_every=5)
def live_dashboard():
    state = snapshot()
    if state["warning"]:
        st.warning(state["warning"])
    tabs = st.tabs(["Mission Control", "Planet Detection", "AI Research Timeline", "Experimental Results", "Sponsor Evidence"])
    for tab, render in zip(tabs, [lambda s: render_mission(s, mission()), render_planets, render_timeline, render_results, render_sponsors]):
        with tab:
            render(state)


if __name__ == "__main__":
    main()
