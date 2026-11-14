"""Zathura mission dashboard. Run with: streamlit run app.py."""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from zathura.dashboard import Mission, snapshot
from zathura.engine import inspect_target

ICONS = {"research-scientist": "🔬", "skeptic": "🛡️", "engine": "🔭", "orchestrator": "🚀"}


def badge(source):
    if source == "placeholder":
        st.markdown('<span class="placeholder">PLACEHOLDER engine</span>', unsafe_allow_html=True)


def metric(label, value, source):
    badge(source)
    st.metric(label, value)


def chart(fig, title, source):
    badge(source)
    fig.update_layout(template="plotly_dark", title=title, paper_bgcolor="#10182d",
                      plot_bgcolor="#10182d", margin=dict(l=25, r=20, t=50, b=30), height=320)
    st.plotly_chart(fig, width="stretch")


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
    records = state["records"]
    analyzed = next((len(e["payload"]) for e in state["latest"] if e.get("step") == "targets"), 0) if state["baseline"] else 0
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Stars analyzed", "Experiments run", "Accepted improvements", "Reviewer rejections"],
                                 [analyzed, sum(bool(r["measurements"]) for r in records), sum(r["accepted"] for r in records),
                                  sum(r["reviewer_decision"]["decision"] != "approve" for r in records)]):
        col.metric(label, value)
    st.caption(f"Cached source curves: {len(state['targets'])}. Stars analyzed counts the latest completed dev baseline. Total hypotheses: {len(records)}.")
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
            st.success("Mission launched in a background worker. This page refreshes every two seconds.")
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
    chart(go.Figure(go.Scattergl(x=curve["time"], y=curve["flux"], mode="markers", marker=dict(size=3, color="#96a6cb"))),
          "Raw source flux (median normalized) · time in days", source)
    fig = go.Figure(go.Scattergl(x=curve["time"], y=curve["flat_flux"], mode="markers", name="Flattened", marker=dict(size=3, color="#39dce8")))
    mask = curve["in_transit"]
    fig.add_trace(go.Scattergl(x=curve["time"][mask], y=curve["flat_flux"][mask], mode="markers",
                              name="Peak transit windows (unverified)", marker=dict(size=5, color="#ad85ff")))
    chart(fig, "Processed light curve · strongest-peak transit windows highlighted", source)
    chart(go.Figure(go.Scatter(x=curve["periods"], y=curve["power"], line=dict(color="#ad85ff"))), "BLS power vs period (days)", source)
    chart(go.Figure(go.Scattergl(x=curve["phase"], y=curve["flat_flux"], mode="markers", marker=dict(size=3, color="#39dce8"))),
          "Phase folded · days from peak transit center", source)


def render_timeline(state):
    st.header("AI Research Timeline")
    st.caption("Latest mission • live event log • proposals and critiques are not measured discoveries")
    authors = {m["memory_id"]: m["agent_id"] for m in state["memories"]}
    if not state["latest"]:
        st.info("No mission events yet.")
    groups = {}
    for event in state["latest"]:
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        groups.setdefault(payload.get("iteration", "Mission lifecycle"), []).append(event)
    for iteration, events in groups.items():
        with st.expander(f"Iteration {iteration}" if isinstance(iteration, int) else iteration, expanded=True):
            for event in events:
                payload, agent = event["payload"], event.get("agent", "orchestrator")
                if payload.get("cited_sources"):
                    with st.container(border=True):
                        st.markdown("📚 **Literature search · Exa sources**")
                        st.caption("Sources returned with this proposal; mock sources are explicitly labelled.")
                        for source in payload["cited_sources"]:
                            st.link_button(("[MOCK] " if source.get("mock") else "Exa · ") + source["title"], source["url"])
                with st.container(border=True):
                    st.markdown(f"{ICONS.get(agent, '🛰️')} **{event['step'].replace('_', ' ').title()}** · {agent}")
                    st.markdown(f":{'violet' if agent == 'skeptic' else 'blue'}[{agent}]")
                    st.caption(event.get("timestamp", ""))
                    badge(payload.get("data_source", payload.get("record", {}).get("data_source")))
                    influence = payload.get("memory_influence", payload if event["step"] == "memory_influence" else {})
                    if influence.get("memory_ids"):
                        verified = influence.get("explanation", "").startswith("Deterministic memory veto")
                        st.warning("Memory changed this decision" if verified else "Model-reported memory influence · UNVERIFIED")
                        st.write(influence.get("explanation", ""))
                        for memory_id in influence["memory_ids"]:
                            st.code(f"{memory_id} · author: {authors.get(memory_id, 'not available in local mirror')}")
                    for key in ("hypothesis", "decision", "rationale"):
                        if key in payload:
                            st.write(f"{key.title()}: {payload[key]}")
                    if payload.get("measurements"):
                        badge(payload.get("data_source", payload.get("record", {}).get("data_source")))
                        st.json(payload["measurements"], expanded=False)
                    if event["step"] == "experiment_started":
                        st.write(payload.get("config"))
                memory_id = payload.get("result_memory_id", payload.get("memory_id"))
                if memory_id:
                    with st.container(border=True):
                        st.markdown(f"🧠 **Memory write** · {authors.get(memory_id, agent)}")
                        st.code(memory_id)
    st.subheader("past.dev exchange / shared memory")
    st.caption("Most recent 200 entries from the local Past.dev mirror when configured, otherwise local JSONL. No remote read is triggered by rendering.")
    for author in ("research-scientist", "skeptic"):
        with st.expander(f"{ICONS[author]} {author}"):
            entries = [m for m in state["memories"] if m["agent_id"] == author]
            if not entries:
                st.write("No entries yet.")
            for entry in entries:
                st.caption(f"{entry['timestamp']} · {entry['kind']} · {entry['memory_id']}")
                st.write("Past.dev accepted write" if entry.get("pastdev") else "Local entry / remotely recalled envelope")
                badge(entry.get("content", {}).get("data_source") if isinstance(entry.get("content"), dict) else None)
                st.json(entry["content"], expanded=False) if isinstance(entry["content"], dict) else st.write(entry["content"])


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
        st.dataframe(frame, hide_index=True, width="stretch")
        for field in ("balanced_accuracy", "recovery_rate", "false_alarm_rate", "runtime_s"):
            fig = px.bar(frame, x="label", y=field, color="decision", color_discrete_sequence=["#39dce8", "#ad85ff", "#ff829f"])
            chart(fig, field.replace("_", " ").title(), "placeholder" if any(r["data_source"] == "placeholder" for r in rows) else None)
    else:
        st.info("No measured results yet.")
    st.subheader("Held-out test performance · not used for tuning")
    if not state["holdout"]:
        st.info("Holdout runs after a mission finishes. No held-out result yet.")
    else:
        result = state["holdout"][-1]
        st.caption(f"Latest completed holdout · loop {result.get('loop_id', 'unknown')} · {result.get('timestamp', '')}")
        for col, (key, value) in zip(st.columns(4), result["measurements"].items()):
            with col:
                metric(key.replace("_", " "), f"{value:.4f}", result.get("data_source"))
    with st.expander("All stored hypotheses (may span incompatible benchmark versions)"):
        for record in state["records"]:
            badge(record["data_source"])
            st.write(record["hypothesis"])
            st.write(record["reviewer_decision"])
            st.json(record["measurements"], expanded=False)


def main():
    st.set_page_config(page_title="Zathura · Mission Control", page_icon="🪐", layout="wide")
    st.markdown("""<style>
    .stApp {background:radial-gradient(ellipse at top right,#17183c 0%,#060b18 65%);}
    h1,h2,h3 {letter-spacing:.04em;} h1 {color:#39dce8;}
    .placeholder {display:inline-block;background:#392958;color:#e5caff;border:1px solid #ad85ff;
      border-radius:5px;padding:2px 8px;font-size:11px;font-weight:700;margin-bottom:6px;}
    [data-testid="stMetric"] {background:#10182dcc;border:1px solid #293554;border-radius:10px;padding:14px;}
    .stButton button[kind="primary"] {color:#060b18;font-weight:700;}
    </style>""", unsafe_allow_html=True)
    st.title("🪐 ZATHURA")
    st.caption("SEARCH THE STARS. QUESTION THE SIGNAL. REMEMBER THE SCIENCE.")
    st.caption("Benchmarks inject synthetic transits into real Kepler light curves; results marked PLACEHOLDER come from the "
               "earlier placeholder engine. No new planet discoveries are claimed.")
    live_dashboard()


@st.fragment(run_every=2)
def live_dashboard():
    state = snapshot()
    if state["warning"]:
        st.warning(state["warning"])
    tabs = st.tabs(["Mission Control", "Planet Detection", "AI Research Timeline", "Experimental Results"])
    for tab, render in zip(tabs, [lambda s: render_mission(s, mission()), render_planets, render_timeline, render_results]):
        with tab:
            render(state)


if __name__ == "__main__":
    main()
