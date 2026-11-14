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

ASSETS = Path(__file__).resolve().parent / "assets"
ICONS = {"research-scientist": ":material/person:", "skeptic": ":material/shield:", "engine": ":material/public:",
         "orchestrator": ":material/rocket_launch:"}
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
        html_block('<span class="placeholder">PLACEHOLDER engine</span>')


def page_header(title, subtitle=None, text=None, subtitle_size=20, title_size=38, offset=0, bottom=0, text_gap=5):
    parts = [f'<div class="z-title" style="font-size:{title_size}px">{title}</div>']
    if subtitle:
        parts.append(f'<div class="z-subtitle" style="font-size:{subtitle_size}px">{subtitle}</div>')
    if text:
        parts.append(f'<div class="z-text" style="margin-top:{text_gap}px">{text}</div>')
    html_block(f'<div class="z-head" style="margin:{offset}px 0 {bottom}px">{"".join(parts)}</div>')


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


@st.cache_resource
def mission():
    return Mission()


@st.cache_data(show_spinner=False)
def target_diagnostic(target):
    return inspect_target(target)


def service_line():
    mock = os.getenv("ZATHURA_MOCK") == "1"
    openai = "LIVE" if os.getenv("OPENAI_API_KEY") and not mock else "MOCK"
    exa = "LIVE" if os.getenv("EXA_API_KEY") and not mock else "MOCK"
    return f"OpenAI: {openai} &nbsp;·&nbsp; Exa: {exa}"


@st.fragment(run_every=2)
def topbar(title):
    status = mission().status()
    label = f"RUNNING · ITERATION {status['iteration']}" if status["running"] else "IDLE"
    html_block(f'<div class="z-topbar"><div class="z-crumb">{icon("home", 20, 1.5)}{icon("chevron", 15, 1.8)}<span>{title}</span></div>'
               f'<div class="z-status"><div class="z-pill{" running" if status["running"] else ""}"><i></i>{label}</div>'
               f'<div class="z-divider"></div></div></div>')


def event_summary(event):
    agent = event.get("agent", "orchestrator")
    stamp = event.get("timestamp", "")[11:19]
    return f'{html.escape(event.get("step", "event").replace("_", " ").title())} · {html.escape(agent)} · {stamp}'


def render_mission(state, worker):
    page_header("Mission Control", "Autonomous transit research",
                "An autonomous AI scientist for Kepler transit research, providing auditable decisions and experiment management.",
                offset=7, bottom=6)
    records = state["records"]
    analyzed = next((len(e["payload"]) for e in state["latest"] if e.get("step") == "targets"), 0) if state["baseline"] else 0
    metric_cards([("Stars analyzed", analyzed), ("Experiments run", sum(bool(r["measurements"]) for r in records)),
                  ("Accepted improvements", sum(r["accepted"] for r in records)),
                  ("Reviewer rejections", sum(r["reviewer_decision"]["decision"] != "approve" for r in records))])
    html_block(f'<div class="z-banner">{icon("info", 20, 1.6)}<span>Synthetic transit injections are used for benchmarks.</span></div>')
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


def render_events(state):
    authors = {m["memory_id"]: m["agent_id"] for m in state["memories"]}
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
                        st.markdown(":material/menu_book: **Literature search · Exa sources**")
                        st.caption("Sources returned with this proposal; mock sources are explicitly labelled.")
                        for source in payload["cited_sources"]:
                            st.link_button(("[MOCK] " if source.get("mock") else "Exa · ") + source["title"], source["url"])
                with st.container(border=True):
                    st.markdown(f"{ICONS.get(agent, ':material/satellite_alt:')} **{event['step'].replace('_', ' ').title()}** · {agent}")
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
                        st.json(payload["measurements"], expanded=False)
                    if event["step"] == "experiment_started":
                        st.write(payload.get("config"))
                memory_id = payload.get("result_memory_id", payload.get("memory_id"))
                if memory_id:
                    with st.container(border=True):
                        st.markdown(f":material/psychology: **Memory write** · {authors.get(memory_id, agent)}")
                        st.code(memory_id)


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
        card_title("Shared research memory", "Proposals and critiques are not measured discoveries.")
        for author, label, glyph in (("research-scientist", "Research Scientist", ":material/person:"),
                                     ("skeptic", "Skeptic", ":material/shield:")):
            with st.expander(label, expanded=True, icon=glyph):
                entries = [m for m in state["memories"] if m["agent_id"] == author]
                if not entries:
                    st.write("No entries yet.")
                for entry in entries:
                    st.caption(f"{entry['timestamp']} · {entry['kind']} · {entry['memory_id']}")
                    st.write("Past.dev accepted write" if entry.get("pastdev") else "Local entry / remotely recalled envelope")
                    content = entry["content"]
                    badge(content.get("data_source") if isinstance(content, dict) else None)
                    if isinstance(content, dict):
                        st.json(content, expanded=False)
                    else:
                        st.write(content)


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
            st.dataframe(frame, hide_index=True, width="stretch")
            source = "placeholder" if any(r["data_source"] == "placeholder" for r in rows) else None
            for field in ("balanced_accuracy", "recovery_rate", "false_alarm_rate", "runtime_s"):
                fig = px.bar(frame, x="label", y=field, color="decision", color_discrete_sequence=["#0ae4ea", "#ad7dee", "#ff829f"])
                chart_card(f"results-{field}", field.replace("_", " ").capitalize(), style_figure(fig, 260, legend=True), source)
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
            metric_cards([(key.replace("_", " ").capitalize(), f"{value:.4f}") for key, value in result["measurements"].items()][:4],
                         compact=True)
    with st.container(key="stored"), st.expander("Stored hypotheses"):
        st.caption("May span incompatible benchmark versions.")
        if not state["records"]:
            st.write("No stored hypotheses yet.")
        for record in state["records"]:
            badge(record["data_source"])
            st.write(record["hypothesis"])
            st.write(record["reviewer_decision"])
            st.json(record["measurements"], expanded=False)


def live(render):
    @st.fragment(run_every=2)
    def body():
        state = snapshot()
        if state["warning"]:
            st.warning(state["warning"])
        render(state)
    body()


def setup():
    st.set_page_config(page_title="Zathura · Mission Control", page_icon=str(ASSETS / "zathura_mark.png"), layout="wide",
                       initial_sidebar_state="expanded")
    st.logo(str(ASSETS / "zathura_logo.png"), size="large", icon_image=str(ASSETS / "zathura_mark.png"))
    variables = "".join(mask_var(name) for name in ("grid", "planet", "layers", "flask", "chevdown"))
    st.html(f"<style>:root{{{variables}}}\n{(ASSETS / 'zathura.css').read_text()}</style>")


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


def main():
    setup()
    control = st.Page(mission_page, title="Mission Control", icon=":material/grid_view:", url_path="mission-control", default=True)
    pages = [control,
             st.Page(planets_page, title="Planet Detection", icon=":material/public:", url_path="planet-detection"),
             st.Page(timeline_page, title="AI Research Timeline", icon=":material/layers:", url_path="ai-research-timeline"),
             st.Page(lambda: results_page(control), title="Experimental Results", icon=":material/science:",
                     url_path="experimental-results")]
    st.navigation(pages, position="sidebar").run()


if __name__ == "__main__":
    main()
