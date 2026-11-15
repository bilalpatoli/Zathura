"""Zathura mission dashboard. Run with: streamlit run app.py."""
import html
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from zathura.dashboard import Mission, snapshot
from zathura.engine import inspect_target
from zathura.evidence import AGENT_NAMES, TRANSPORT_NAMES, config_change, memory_activity, proposal_chains
from zathura.story import render_story
from zathura import simple

ASSETS = Path(__file__).resolve().parent / "assets"
BENCHMARK_DISCLOSURE = ("Benchmark transits are synthetic injections into real Kepler photometry; "
                        "historical mock/placeholder output stays labelled.")
SVG = {
    "grid": '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>'
            '<rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>',
    "planet": '<circle cx="12" cy="12" r="6.5"/><path d="M6.6 16.2C3.4 19.4 2 21.6 2.7 22.3c1.3 1.3 7.7-2.9 12.6-7.9s9.2-11.3 7.9-12.6'
              'c-.7-.7-2.9.6-6.1 3.8"/>',
    "layers": '<path d="M12 2.5 2.5 7.2 12 12l9.5-4.8z"/><path d="m2.5 12 9.5 4.8 9.5-4.8"/><path d="m2.5 16.8 9.5 4.7 9.5-4.7"/>',
    "flask": '<path d="M10 2v7.5a2 2 0 0 1-.2.9L4.7 20.6a1 1 0 0 0 .9 1.4h12.8a1 1 0 0 0 .9-1.4l-5.1-10.2a2 2 0 0 1-.2-.9V2"/>'
             '<path d="M8.5 2h7"/><path d="M7 16h10"/>',
    "home": '<path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"/><path d="M3 10a2 2 0 0 1 .7-1.5l7-6a2 2 0 0 1 2.6 0l7 6A2 2 0 0 1 21 10'
            'v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "chevron": '<path d="m9 18 6-6-6-6"/>',
    "chevdown": '<path d="m6 9 6 6 6-6"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "file": '<path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5z"/><path d="M14 2v6h6"/><path d="M9 14h4"/>'
            '<path d="M9 17.5h6"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l3.5 2.5"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/>',
    "code": '<path d="m18 16 4-4-4-4"/><path d="m6 8-4 4 4 4"/><path d="m14.5 4-5 16"/>',
    "clipboard": '<rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6'
                 'a2 2 0 0 1 2-2h2"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/>',
}
PLOT_FONT = dict(family="Noto Sans, sans-serif", size=12, color="#d5d9e0")


def icon(name, size=22, stroke=1.5):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{SVG[name]}</svg>')


def mask_var(name):
    svg = icon(name, 24).replace("currentColor", "black")
    return f'--z-ic-{name}:url("data:image/svg+xml,{quote(svg)}");'


def html_block(markup):
    st.markdown(markup, unsafe_allow_html=True)


def badge(source):
    if source == "placeholder":
        st.markdown('<span class="placeholder">PLACEHOLDER engine</span>', unsafe_allow_html=True)
    elif source == "real":
        st.markdown('<span class="real-source">REAL Kepler source</span>', unsafe_allow_html=True)


def page_header(title, subtitle=None, text=None, subtitle_size=20, title_size=38, offset=0, bottom=0, text_gap=5):
    parts = [f'<div class="z-title" style="font-size:{title_size}px">{title}</div>']
    if subtitle:
        parts.append(f'<div class="z-subtitle" style="font-size:{subtitle_size}px">{subtitle}</div>')
    if text:
        parts.append(f'<div class="z-text" style="margin-top:{text_gap}px">{text}</div>')
    html_block(f'<div class="z-head" style="margin:{offset}px 0 {bottom}px">{"".join(parts)}</div>')
    st.caption(BENCHMARK_DISCLOSURE)


def metric_cards(items, compact=False):
    for col, (label, value) in zip(st.columns(len(items), gap="small"), items):
        with col:
            html_block(f'<div class="z-metric{" compact" if compact else ""}"><div class="label">{label}</div>'
                       f'<div class="value">{value}</div></div>')


def card_title(title, caption=None):
    html_block(f'<div class="z-h3">{title}</div>' + (f'<div class="z-caption">{caption}</div>' if caption else ""))


def empty_state(name, lead=None, strong=None, sub=None, size=44, height=None, pad=0, gap=12, sub_gap=8):
    style = f' style="min-height:{height}px;padding-top:{pad}px"' if height else ""
    body = icon(name, size, 1.3)
    body += f'<div class="lead" style="margin-top:{gap}px">{lead}</div>' if lead else ""
    body += f'<div class="strong" style="margin-top:{gap}px">{strong}</div>' if strong else ""
    body += f'<div class="sub" style="margin-top:{sub_gap}px">{sub}</div>' if sub else ""
    html_block(f'<div class="z-empty"{style}>{body}</div>')


def style_figure(fig, height, x_title=None, legend=False):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=PLOT_FONT,
                      margin=dict(l=0, r=0, t=14, b=0), height=height, showlegend=legend,
                      legend=dict(x=1.01, y=1, yanchor="top", bgcolor="rgba(0,0,0,0)", font=dict(size=12)))
    fig.update_xaxes(title=dict(text=x_title, font=dict(size=12.5)), gridcolor="rgba(0,0,0,0)", zeroline=False,
                     showline=True, linecolor="#6b7380", ticks="outside", tickcolor="#6b7380", ticklen=4)
    fig.update_yaxes(gridcolor="#1f2a39", zeroline=False, showline=True, linecolor="#6b7380", ticks="outside",
                     tickcolor="#6b7380", ticklen=4)
    return fig


def chart_card(key, title, fig, source=None):
    with st.container(key=f"chartcard-{key}"):
        badge(source)
        html_block(f'<div class="z-chart-title">{title}</div>')
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


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


def service_line():
    mock = os.getenv("ZATHURA_MOCK") == "1"
    openai = "API configured" if os.getenv("OPENAI_API_KEY") and not mock else "MOCK"
    exa = "API configured" if os.getenv("EXA_API_KEY") and not mock else "MOCK"
    pastdev = "REMOTE-FIRST configured" if os.getenv("PASTDEV_API_KEY") else "LOCAL JSONL"
    return f"OpenAI: {openai} &nbsp;·&nbsp; Exa: {exa} &nbsp;·&nbsp; past.dev: {pastdev}"


@st.fragment(run_every=5)
def topbar(title):
    status = mission().status()
    label = f"RUNNING · ITERATION {status['iteration']}" if status["running"] else "IDLE"
    demo = os.getenv("ZATHURA_MOCK") == "1" or not os.getenv("OPENAI_API_KEY")
    mode = ('<div class="z-pill" style="color:#f3c969;border-color:#8a6d2b" title="Mock AI responses; no paid API calls">'
            'DEMO MODE · MOCK AI</div>' if demo else '<div class="z-pill" title="Live OpenAI/Exa calls may incur charges">LIVE</div>')
    html_block(f'<div class="z-topbar"><div class="z-crumb">{icon("home", 20, 1.5)}{icon("chevron", 15, 1.8)}<span>{title}</span></div>'
               f'<div class="z-status">{mode}<div class="z-pill{" running" if status["running"] else ""}"><i></i>{label}</div>'
               f'<div class="z-divider"></div></div></div>')


def event_summary(event):
    agent = event.get("agent", "orchestrator")
    stamp = event.get("timestamp", "")[11:19]
    return f'{html.escape(event.get("step", "event").replace("_", " ").title())} · {html.escape(agent)} · {stamp}'


def render_mission(state, worker):
    page_header("Mission Control", "Autonomous transit research",
                "An autonomous AI scientist for Kepler transit research, providing auditable decisions and experiment management.",
                offset=7, bottom=6)
    records = state["current"]
    analyzed = (state["baseline"].get("provenance", {}).get("host_count", 0)
                if state["baseline"] else 0)
    metric_cards([("Stars analyzed", analyzed), ("Experiments run", sum(bool(r["measurements"]) for r in records)),
                  ("Accepted improvements", sum(r["accepted"] for r in records)),
                  ("Reviewer rejections", sum(r["reviewer_decision"]["decision"] != "approve" for r in records))])
    html_block(f'<div class="z-banner">{icon("info", 20, 1.6)}<span>Metrics show the latest mission only. Service configuration is not evidence of live use; see Sponsor Evidence for remote/mirror receipts.</span></div>')
    baseline = state["baseline"]
    if baseline:
        base = baseline["measurements"]["balanced_accuracy"]
        best = max([base] + [r["measurements"]["balanced_accuracy"] for r in state["current"] if r["measurements"]])
        badge(baseline.get("data_source"))
        metric_cards([("Baseline dev balanced accuracy", f"{base:.4f}"), ("Best dev balanced accuracy", f"{best:.4f}")], compact=True)
    status = worker.status()
    left, right = st.columns([627, 617], gap="small")
    with left, st.container(key="card-hypothesis", height="stretch"):
        card_title("Current hypothesis")
        hypothesis = next((e["payload"].get("hypothesis") for e in reversed(state["latest"])
                           if e.get("step") in ("hypothesis", "revision")), None)
        if hypothesis:
            st.write(hypothesis)
        else:
            empty_state("file", lead="Awaiting a research proposal", size=49, height=180, pad=35, gap=20)
    with right, st.container(key="card-mission", height="stretch"):
        card_title("Research mission")
        iterations = st.selectbox("Research iterations", list(range(1, 11)), index=3, width=254, disabled=status["running"])
        if st.button("Launch research mission", type="primary", disabled=status["running"]):
            if worker.launch(iterations):
                st.toast("Mission launched in a background worker.")
        if status["error"]:
            st.error(f"Mission stopped: {status['error']}. Check service availability and the event timeline.")
        html_block('<div class="z-caption z-launch-note">Launch may call paid APIs when configured. One mission per dashboard '
                   'process; keep this process alive.</div>')
    left, right = st.columns([627, 617], gap="small")
    with left, st.container(key="card-activity", height="stretch"):
        card_title("Mission activity")
        if state["latest"]:
            rows = "".join(f"<div>{icon('clock', 18)}<span>{event_summary(e)}</span></div>" for e in state["latest"][-5:][::-1])
            html_block(f'<div class="z-list" style="font-size:15px;gap:12px">{rows}</div>')
        else:
            empty_state("clock", lead="No mission events yet", size=42, height=110, pad=13, gap=14)
    with right, st.container(key="card-provenance", height="stretch"):
        card_title("Data provenance")
        html_block(f'<div class="z-list"><div>{icon("database", 24)}<span>Kepler · {len(state["targets"])} cached source curves</span></div>'
                   f'<div>{icon("code", 24)}<span>{service_line()}</span></div></div>')


def render_planets(state):
    page_header("Planet Detection", text="Read-only diagnostic of un-injected Kepler curves using the baseline detector. "
                                         "Not benchmark recovery truth.", title_size=35, offset=-4, text_gap=-2)
    if not state["targets"]:
        st.info("No cached light curves. Rebuild them with scripts/build_catalog.py.")
        return
    labels = {t["target_id"]: " · ".join(filter(None, [t.get("name"), t["target_id"]])) for t in state["targets"]}
    with st.container(horizontal=True, vertical_alignment="center", gap="medium", key="star-row"):
        html_block('<div class="z-star">Star</div>')
        target = st.selectbox("Star", list(labels), format_func=labels.get, label_visibility="collapsed", width=694)
    try:
        curve = target_diagnostic(target)
    except (ValueError, OSError, RuntimeError) as exc:
        st.warning(f"Target diagnostic unavailable: {type(exc).__name__}")
        return
    source = curve["data_source"]
    simulated = "synthetic" in curve["curve_origin"].lower()
    if curve.get("known_periods"):
        periods = ", ".join(f"{p:.4f}" for p in curve["known_periods"])
        banner = ("Known-planet validation", f"Known-planet validation star &nbsp;·&nbsp; KOI periods: {periods} days.")
        link = ("NASA Exoplanet Archive", "https://exoplanetarchive.ipac.caltech.edu/")
    else:
        banner = ("Benchmark host star", "No KOI or TCE; not certified planet-free.")
        link = ("MAST Kepler archive", "https://archive.stsci.edu/missions-and-data/kepler")
    origin = "SIMULATED source curve — not NASA observations." if simulated else "Real source data."
    left, right = st.columns([831, 418], gap="medium")
    with left:
        html_block(f'<div class="z-validation"><span class="big">{banner[0]}</span><span class="bar"></span>'
                   f'<span class="small">{banner[1]}</span></div>')
    with right:
        html_block(f'<div class="z-source" title="Curve origin: {html.escape(curve["curve_origin"])}">Data source<br>'
                   f'<a href="{link[1]}" target="_blank">{link[0]}</a> &nbsp;·&nbsp; {origin}</div>')
    badge(source)
    html_block(f'<div class="z-h3" style="margin:10px 0 -10px">{"Unverified transit candidate" if curve["detected"] else "No threshold-crossing candidate"}</div>'
               + ("" if curve["detected"] else '<div class="z-caption">Showing the strongest BLS peak only.</div>'))
    metric_cards([(label, f"{value:.4f}") for label, value in zip(
        ["Estimated period (days)", "Depth (ppm)", "Duration (days)", "Peak SDE"],
        [curve["period_days"], curve["depth"] * 1e6, curve["duration_days"], curve["sde"]])], compact=True)
    mask = curve["in_transit"]
    fig = go.Figure(go.Scattergl(x=curve["time"], y=curve["flat_flux"], mode="markers", name="Flattened",
                                 marker=dict(size=3, color="#0ae4ea")))
    fig.add_trace(go.Scattergl(x=curve["time"][mask], y=curve["flat_flux"][mask], mode="markers",
                               name="Peak transit windows<br>(unverified)", marker=dict(size=5, color="#ad7dee")))
    chart_card("curve", "Processed light curve &nbsp;·&nbsp; strongest-peak transit windows highlighted",
               style_figure(fig, 193, "Time (days)", legend=True), source)
    fold = go.Figure(go.Scattergl(x=curve["phase"], y=curve["flat_flux"], mode="markers", marker=dict(size=3, color="#0ae4ea")))
    pad = 0.02 * (curve["phase"].max() - curve["phase"].min())
    fold.update_xaxes(range=[curve["phase"].min() - pad, curve["phase"].max() + pad])
    chart_card("fold", "Phase-folded light curve &nbsp;·&nbsp; days from peak transit center",
               style_figure(fold, 193, "Phase (days)"), source)
    with st.expander("Source photometry and BLS periodogram"):
        raw = go.Figure(go.Scattergl(x=curve["time"], y=curve["flux"], mode="markers",
                                     marker=dict(size=3, color="#96a6cb")))
        raw.update_yaxes(title="Normalized source flux")
        chart_card("source", "Raw source photometry &nbsp;·&nbsp; before flattening",
                   style_figure(raw, 280, "Time (BKJD)"), source)
        periodogram = go.Figure(go.Scatter(x=curve["periods"], y=curve["power"],
                                           line=dict(color="#ad7dee")))
        periodogram.update_yaxes(title="BLS power")
        chart_card("periodogram", "BLS periodogram &nbsp;·&nbsp; power versus trial period",
                   style_figure(periodogram, 280, "Period (days)"), source)


def render_events(state):
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


def render_timeline(state):
    page_header("AI Research Timeline", "Latest mission • live event log • proposals and critiques are not measured discoveries.",
                subtitle_size=18, offset=2)
    with st.container(key="card-timeline"):
        card_title("Mission activity")
        if state["latest"]:
            render_events(state)
        else:
            empty_state("clock", lead="No mission events yet.", size=48, height=222, pad=57, gap=17)
    with st.container(key="card-memory"):
        card_title("Shared research memory", "Recorded Scientist and Skeptic exchanges for the latest mission.")
        activity = memory_activity(state["latest"], state["memories"])
        if activity:
            st.dataframe(pd.DataFrame(activity), hide_index=True, width="stretch")
        else:
            st.write("No exchange recorded yet.")
        st.caption("Open Sponsor Evidence for search sources, memory transports, and verified decision changes.")


def render_sponsors(state):
    page_header("Sponsor Evidence", text="Recorded provenance · latest mission · browsing triggers no remote requests.")
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
        st.caption("Accepted writes hit past.dev. Loop reads are remote-first; the mirror is used only after remote errors or timeouts. UI browsing reads the local audit mirror. Historical receipts retain their original transport labels.")
        for event in state["latest"]:
            payload = event.get("payload", {})
            recall = payload.get("memory_recall") if isinstance(payload, dict) else None
            if recall:
                who = AGENT_NAMES.get(event.get("agent"), "Agent")
                via = TRANSPORT_NAMES.get(recall.get("read_via"), "Not recorded")
                st.caption(f"{who} · iteration {payload.get('iteration', '?')} · recall via {via} · {recall['returned_count']} returned before evidence filtering")
                if recall.get("fallback_reason"):
                    st.caption("Remote failure · " + recall["fallback_reason"])
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



def render_results(state, mission_page=None):
    page_header("Experimental Results", "Latest mission comparisons and held-out evaluation for transit research.",
                "Rejected hypotheses remain visible with missing metrics.", subtitle_size=19, offset=3)
    rows = []
    baseline = state["baseline"]
    if baseline:
        rows.append({"label": "Baseline", "decision": "baseline", "data_source": baseline["data_source"], **baseline["measurements"]})
    for i, record in enumerate(state["current"], 1):
        rows.append({"label": f"Experiment {i}", "decision": record["reviewer_decision"]["decision"],
                     "data_source": record["data_source"], **record["measurements"]})
    measured = [r for r in rows if "balanced_accuracy" in r]
    with st.container(key="card-comparisons"):
        card_title("Latest mission comparisons", "Compare recent mission runs and hypothesis variants using benchmark metrics.")
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
            source = "placeholder" if any(r["data_source"] == "placeholder" for r in rows) else "real"
            columns = st.columns(2)
            for index, field in enumerate(("balanced_accuracy", "recovery_rate", "false_alarm_rate", "runtime_s")):
                with columns[index % 2]:
                    fig = px.bar(frame, x="label", y=field, color="decision", labels=fields,
                                 color_discrete_map={"baseline": "#0ae4ea", "approve": "#ad7dee", "reject": "#ff829f"})
                    if field != "runtime_s":
                        fig.update_yaxes(range=[0, 1])
                    chart_card(f"results-{field}", fields[field], style_figure(fig, 260, legend=True), source)
        else:
            empty_state("clipboard", strong="No measured results yet.",
                        sub="Run a mission to compare hypotheses and view benchmark metrics here.", size=62, height=150, pad=2, gap=10, sub_gap=2)
            with st.container(horizontal_alignment="center", key="launch-row"):
                if st.button("Launch research mission", type="primary", key="results-launch") and mission_page:
                    st.switch_page(mission_page)
    with st.container(key="card-holdout"):
        card_title("Held-out test performance", "Evaluate final hypotheses on a held-out test set (not used for tuning).")
        if not state["holdout"]:
            empty_state("flask", strong="No held-out result yet.", sub="Held-out runs appear here after a mission finishes.",
                        size=58, height=137, pad=0, gap=4, sub_gap=2)
        else:
            result = state["holdout"][-1]
            st.caption(f"Latest completed holdout · loop {result.get('loop_id', 'unknown')} · {result.get('timestamp', '')}")
            badge(result.get("data_source"))
            measurements = result["measurements"]
            fields = ("recovery_rate", "false_alarm_rate", "balanced_accuracy", "runtime_s")
            metric_cards([(key.replace("_", " ").capitalize(),
                           f"{measurements[key]:.4f}" if measurements.get(key) is not None else "—")
                          for key in fields], compact=True)
            low, high = (measurements.get(key) for key in ("balanced_accuracy_ci_low", "balanced_accuracy_ci_high"))
            if low is not None and high is not None:
                st.caption(f"Balanced accuracy 95% confidence interval: [{low:.4f}, {high:.4f}]")
    with st.container(key="stored"), st.expander("Stored hypotheses"):
        archived = [{"Hypothesis": r["hypothesis"], "Review": r["reviewer_decision"]["decision"],
                     "Balanced accuracy": r["measurements"].get("balanced_accuracy"),
                     "Source": "PLACEHOLDER engine" if r["data_source"] == "placeholder" else r["data_source"]}
                    for r in state["records"][-50:]]
        st.caption("Most recent 50 archived hypotheses · not a cross-benchmark leaderboard")
        if archived:
            st.dataframe(pd.DataFrame(archived), hide_index=True, width="stretch")

        else:
            st.write("No stored hypotheses yet.")


def live(render):
    @st.fragment(run_every=5)
    def body():
        state = snapshot()
        if state["warning"]:
            st.warning(state["warning"])
        render(state)
    body()


def setup():
    st.set_page_config(page_title="Zathura · Mission Control", page_icon=str(ASSETS / "zathura_mark.png"), layout="wide",
                       initial_sidebar_state="auto")
    st.logo(str(ASSETS / "zathura_logo.png"), size="large", icon_image=str(ASSETS / "zathura_mark.png"))
    variables = "".join(mask_var(name) for name in ("grid", "planet", "layers", "flask", "chevdown", "home", "database"))
    st.html(f"<style>:root{{{variables}}}\n{(ASSETS / 'zathura.css').read_text()}</style>")
    st.html(f"<style>{(ASSETS / 'simple.css').read_text()}</style>")


def mission_page():
    topbar("Mission Control")
    live(lambda state: render_mission(state, mission()))


def planets_page():
    topbar("Planet Detection")
    state = snapshot()
    render_planets(state)


def timeline_page():
    topbar("AI Research Timeline")
    live(render_timeline)


def results_page(control=None):
    topbar("Experimental Results")
    live(lambda state: render_results(state, control))


def story_page():
    topbar("Research Story")
    live(lambda state: render_story(state, mission()))


def sponsors_page():
    topbar("Sponsor Evidence")
    live(render_sponsors)


def find_page():
    topbar("Find")
    simple.render_find(snapshot())


def improve_page():
    topbar("Improve")
    live(lambda state: simple.render_improve(state, mission()))


def simple_results_page():
    topbar("Results")
    live(lambda state: simple.render_results(state, mission().status()["running"]))


def technical_page():
    topbar("Technical details")

    live(render_technical)


def render_technical(state):
    renderers = [lambda: render_mission(state, mission()), lambda: render_planets(state),
                 lambda: render_timeline(state), lambda: render_results(state), lambda: render_sponsors(state),
                 simple.render_validation, lambda: render_story(state, mission())]
    tabs = st.tabs(["Mission Control", "Planet Detection", "Timeline", "Experimental Results", "Sponsor Evidence",
                    "Vetting validation", "Research Story"])
    for tab, render in zip(tabs, renderers):
        with tab:
            render()


def main():
    setup()
    st.markdown(f'<div class="simple-pitch">{simple.PITCH}</div>', unsafe_allow_html=True)
    pages = [st.Page(find_page, title="Find", icon=":material/home:", default=True),
             st.Page(improve_page, title="Improve", icon=":material/grid_view:", url_path="improve"),
             st.Page(simple_results_page, title="Results", icon=":material/public:", url_path="results"),
             st.Page(technical_page, title="Technical details", icon=":material/layers:", url_path="technical-details")]
    st.navigation(pages, position="sidebar").run()


if __name__ == "__main__":
    main()
