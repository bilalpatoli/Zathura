"""Zathura mission dashboard. Run with: streamlit run app.py."""
import html
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from zathura.dashboard import DEFAULT_ROUNDS, ROUNDS_HELP, Mission, snapshot
from zathura.engine import get_experiment_results, inspect_target
from zathura.engine.data import cache_path, read_curve
from zathura.engine.detect import search
from zathura.engine.vetting import vet_candidate, vetting_targets
from zathura.evidence import AGENT_NAMES, TRANSPORT_NAMES, config_change, memory_activity, proposal_chains
from zathura.runs import (counts, describe, elapsed, outcome, plain_result, run_list, run_view,
                          selected_measurements)
from zathura.schema import BASELINE_CONFIG
from zathura.story import render_how_it_works

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


def page_header(title, subtitle=None, text=None, subtitle_size=20, title_size=38, offset=0, bottom=0, text_gap=5,
                benchmark=True):
    parts = [f'<div class="z-title" style="font-size:{title_size}px">{title}</div>']
    if subtitle:
        parts.append(f'<div class="z-subtitle" style="font-size:{subtitle_size}px">{subtitle}</div>')
    if text:
        parts.append(f'<div class="z-text" style="margin-top:{text_gap}px">{text}</div>')
    html_block(f'<div class="z-head" style="margin:{offset}px 0 {bottom}px">{"".join(parts)}</div>')
    if benchmark:
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
    binary = next((t for t in vetting_targets() if t["truth"] == "EB" and target == f'KIC {t["kepid"]}'), None)
    if binary is None:
        return inspect_target(target)
    curve = read_curve(binary["kepid"])
    if curve["origin"] != "real":
        raise ValueError("Binary examples require real source photometry")
    valid = np.isfinite(curve["time"]) & np.isfinite(curve["flux"])
    t, flux = curve["time"][valid], curve["flux"][valid]
    if len(t) < 10 or np.ptp(t) <= 2:
        raise ValueError("Insufficient finite light-curve samples for BLS")
    order = np.argsort(t)
    t, flux = t[order], flux[order] / np.median(flux)
    result = search(t, flux, BASELINE_CONFIG)
    phase = (result["time"] - result["epoch"] + result["period_days"] / 2) % result["period_days"] - result["period_days"] / 2
    return result | {"target_id": target, "flux": np.interp(result["time"], t, flux), "phase": phase,
                     "in_transit": np.abs(phase) < result["duration_days"] / 2, "known_periods": None,
                     "quarter": int(curve["quarter"]), "product": curve["product"],
                     "curve_origin": curve["origin"], "data_source": "real"}


def demo_mode():
    return os.getenv("ZATHURA_MOCK") == "1" or not os.getenv("OPENAI_API_KEY")


def status_label(status):
    if not status["running"]:
        return "Ready"
    if not status["iteration"]:
        return "Starting research"
    return f"Running · round {status['iteration']} of {status['rounds']}" if status.get("rounds") else f"Running · round {status['iteration']}"


@st.fragment(run_every=5)
def topbar(title):
    status = mission().status()
    demo = '<div class="z-pill demo">Demo mode</div>' if demo_mode() else ""
    html_block(f'<div class="z-topbar"><div class="z-crumb">{icon("home", 20, 1.5)}{icon("chevron", 15, 1.8)}<span>{title}</span></div>'
               f'<div class="z-status">{demo}<div class="z-pill{" running" if status["running"] else ""}"><i></i>{status_label(status)}</div>'
               f'<div class="z-divider"></div></div></div>')


def render_sponsors(state):
    page_header("Technical report", text="Sponsor API evidence, memory transports and raw audit records for the latest run. "
                                     "Browsing makes no remote requests.", title_size=36, text_gap=4)
    mock = os.getenv("ZATHURA_MOCK") == "1"
    openai = "API configured" if os.getenv("OPENAI_API_KEY") and not mock else "MOCK"
    exa = "API configured" if os.getenv("EXA_API_KEY") and not mock else "MOCK"
    pastdev = "REMOTE-FIRST configured" if os.getenv("PASTDEV_API_KEY") else "LOCAL JSONL"
    st.caption(f"OpenAI: {openai} · Exa: {exa} · past.dev: {pastdev}")
    chains = proposal_chains(state["latest"])
    authors = {m["memory_id"]: m["agent_id"] for m in state["memories"]}
    left, right = st.columns(2)
    with left:
        st.subheader("Exa · research to result")
        st.caption("Exact SDK call · Exa.search_and_contents")
        st.code("zathura/agents/exa_search.py:search_papers", language=None)
        if not chains:
            st.info("Start research to record search queries and citation chains.")
        for index, chain in enumerate(chains):
            proposal, completion = chain["proposal"], chain["completion"]
            with st.expander(f"Round {proposal.get('iteration', '?')}{' · revision' if chain['revision'] else ''} · source → result", expanded=index == 0):
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
                st.caption(f"{who} · round {payload.get('iteration', '?')} · recall via {via} · {recall['returned_count']} returned before evidence filtering")
                if recall.get("fallback_reason"):
                    st.caption("Remote failure · " + recall["fallback_reason"])
        changes = [c["proposal"] for c in chains if c["proposal"].get("memory_influence", {}).get("memory_ids")]
        if not changes:
            st.info("No memory-driven decision change recorded for this run.")
        for proposal in changes:
            influence = proposal["memory_influence"]
            verified = influence.get("explanation", "").startswith("Deterministic memory veto")
            with st.container(border=True):
                st.warning("Memory changed a decision" if verified else "Model-reported influence · UNVERIFIED")
                st.caption(f"Round {proposal.get('iteration', '?')} · evidence-backed veto" if verified else "Not proof of a causal veto")
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


EXAMPLE_TARGET = "KIC 6665695"


@st.cache_data(show_spinner=False)
def evidence_checks(target):
    return vet_candidate(target)


def lead(text):
    html_block(f'<div class="z-lead">{text}</div>')


def target_label(target):
    catalog = ("known eclipsing binary" if target.get("truth") == "EB" else
               "known planet host" if target.get("known_periods") else "no known planet")
    return " · ".join(filter(None, [target.get("name"), target["target_id"]])) + f"  ({catalog})"


def explore_targets(state):
    targets = {t["target_id"]: t for t in state["targets"]}
    if targets:
        for target in vetting_targets():
            if target["truth"] == "EB" and cache_path(target["kepid"]).exists():
                targets.setdefault(f'KIC {target["kepid"]}', target | {"target_id": f'KIC {target["kepid"]}'})
    return targets


def catalog_match(curve):
    known = curve.get("known_periods") or []
    return curve["detected"] and any(abs(curve["period_days"] - p) / p < 0.01 for p in known)


def brightness_figure(curve, height=230):
    mask = curve["in_transit"]
    fig = go.Figure(go.Scattergl(x=curve["time"], y=curve["flat_flux"], mode="markers", name="Brightness",
                                 marker=dict(size=3, color="#0ae4ea")))
    fig.add_trace(go.Scattergl(x=curve["time"][mask], y=curve["flat_flux"][mask], mode="markers",
                               name="Dips found by<br>the detector", marker=dict(size=6, color="#ad7dee")))
    fig.update_yaxes(title=dict(text="Relative brightness", font=dict(size=13)))
    return style_figure(fig, height, "Time (days)", legend=True)


def folded_figure(curve, height=230):
    fig = go.Figure(go.Scattergl(x=curve["phase"] * 24, y=curve["flat_flux"], mode="markers",
                                 marker=dict(size=3, color="#0ae4ea")))
    span = max(abs(curve["phase"].min()), abs(curve["phase"].max())) * 24
    fig.update_xaxes(range=[-span * 1.02, span * 1.02])
    fig.update_yaxes(title=dict(text="Relative brightness", font=dict(size=13)))
    return style_figure(fig, height, "Hours from dip center")


def go_explore(example):
    st.session_state["explore_target"] = EXAMPLE_TARGET if example else st.session_state.get("explore_target")
    st.session_state["explore_example"] = example
    st.switch_page(PAGES["explore"])


def go_research(view="Overview"):
    st.session_state["research_view"] = view
    st.switch_page(PAGES["research"])


def render_home(state):
    html_block('<div class="z-hero"><div class="z-title">An AI scientist searching for planets.</div>'
               '<div class="z-lead">Zathura analyzes NASA telescope data for signs of planets beyond our solar system. '
               'It checks possible signals and experiments with better ways to detect them.</div></div>')
    with st.container(horizontal=True, gap="small", key="hero-actions"):
        if st.button("Explore a star", type="primary", key="home_explore"):
            go_explore(False)
        if st.button("See an example", key="home_example"):
            go_explore(True)
    example = next((t for t in state["targets"] if t["target_id"] == EXAMPLE_TARGET), None)
    curve = None
    if example:
        try:
            curve = target_diagnostic(EXAMPLE_TARGET)
        except (ValueError, OSError, RuntimeError):
            curve = None
    with st.container(key="card-home-chart"):
        if curve:
            name = example.get("name", EXAMPLE_TARGET).replace(" b", "")
            html_block(f'<div class="z-chart-title">Star brightness over time · {name}</div>'
                       '<div class="z-caption">When a planet passes in front of its star, it can block a tiny amount of light. '
                       'Zathura looks for these repeating dips.</div>')
            st.plotly_chart(brightness_figure(curve, 250), width="stretch", config={"displayModeBar": False}, key="home_curve")
            found = (f"Highlighted points are dips Zathura's detector found, repeating every {curve['period_days']:.2f} days."
                     if curve["detected"] else "The detector found no dip above its threshold in this quarter.")
            html_block(f'<div class="z-caption">{name} ({EXAMPLE_TARGET}) · NASA Kepler telescope, quarter {curve["quarter"]} · '
                       f'real observations · catalog: known planet host. {found}</div>')
        else:
            st.write("The cached Kepler example is unavailable. Rebuild the cache with scripts/build_catalog.py.")
    columns = st.columns(3, gap="large")
    for col, (glyph, title, text) in zip(columns, [
            ("planet", "Search telescope data", "Look for repeating changes in a star’s brightness."),
            ("clipboard", "Check the evidence", "Test whether a signal could come from a planet, another star, or noise."),
            ("flask", "Improve the search", "Let the AI research and test better detection methods.")]):
        with col:
            html_block(f'<div class="z-explain">{icon(glyph, 26, 1.5)}<div><div class="title">{title}</div>'
                       f'<div class="text">{text}</div></div></div>')
    st.divider()
    with st.container(key="card-home-research"):
        html_block('<div class="z-h3">Help Zathura get better at finding planet signals.</div>'
                   '<div class="z-lead small">Watch it read scientific papers, propose changes, and test whether they improve detection.</div>')
        if st.button("Start research", type="primary", key="home_research"):
            go_research()


def render_explore(state):
    page_header("Explore stars", text="Real NASA Kepler observations analyzed by Zathura’s starting detector. "
                                      "No simulated signals are added on this page.", title_size=36, text_gap=4)
    if not state["targets"]:
        st.info("No cached light curves. Rebuild them with scripts/build_catalog.py.")
        return
    targets = explore_targets(state)
    if st.session_state.get("explore_target") not in targets:
        st.session_state["explore_target"] = EXAMPLE_TARGET if EXAMPLE_TARGET in targets else next(iter(targets))
    target = st.selectbox("Star", list(targets), format_func=lambda t: target_label(targets[t]), key="explore_target", width=694,
                          on_change=lambda: st.session_state.update(explore_example=False))
    try:
        curve = target_diagnostic(target)
    except (ValueError, OSError, RuntimeError) as exc:
        st.warning(f"This star's diagnostic is unavailable: {type(exc).__name__}")
        return
    known = curve.get("known_periods") or []
    name = targets[target].get("name", target)
    if st.session_state.get("explore_example") and target == EXAMPLE_TARGET:
        agree = ("The detector, which never sees the catalog, found dips at the same period. Matching a known planet checks "
                 "the detector; it is not a new discovery." if catalog_match(curve) else
                 "The detector's strongest period differs from the catalog, which shows why signals need checking.")
        html_block(f'<div class="z-banner">{icon("info", 20, 1.6)}<span><b>Example:</b> {html.escape(name)} has a planet in NASA’s catalog '
                   f'every {known[0]:.2f} days. {agree}</span></div>')
    observation = (f"Brightness dips repeat about every {curve['period_days']:.2f} days." if curve["detected"]
                   else "No repeating dip stood out above the detection threshold.")
    html_block(f'<div class="z-observation">{observation}</div>')
    if targets[target].get("truth") == "EB":
        catalog = ("Known eclipsing binary", "Catalogued as two stars crossing each other.")
    elif known:
        periods = ", ".join(f"{p:.2f}" for p in known)
        catalog = ("Known planet host", f"NASA lists {'planets' if len(known) > 1 else 'a planet'} every {periods} days.")
    else:
        catalog = ("No known planet", "No planet candidates are listed. That does not prove the star has none.")
    if curve["detected"]:
        match = " The period matches the catalog." if catalog_match(curve) else (" The period does not match the catalog." if known else "")
        analysis = ("Possible transit signal", "A candidate, not a confirmed planet." + match)
    else:
        analysis = ("No signal above the threshold", "The strongest peak is shown below for reference only.")
    origin = ("Simulated curve", "Not NASA observations.") if "synthetic" in curve["curve_origin"].lower() else \
        ("Observed data", f"NASA Kepler, quarter {curve['quarter']}, {np.ptp(curve['time']):.0f} days.")
    for col, (label, (title, text)) in zip(st.columns(3, gap="small"), [("Catalog", catalog), ("This analysis", analysis), ("Data", origin)]):
        with col:
            html_block(f'<div class="z-fact"><div class="label">{label}</div><div class="title">{title}</div><div class="text">{text}</div></div>')
    if targets[target].get("truth") == "EB":
        st.link_button("Catalog classification", targets[target]["source"])
    prefix = "" if curve["detected"] else "Strongest peak · "
    metric_cards([(prefix + "Repeats every", f"{curve['period_days']:.2f} days"),
                  (prefix + "Brightness drop", f"{curve['depth'] * 100:.3f}%"),
                  (prefix + "Dip duration", f"{curve['duration_days'] * 24:.1f} hours")], compact=True)
    chart_card("curve", "Star brightness over time", brightness_figure(curve))
    chart_card("fold", "Repeated dips lined up &nbsp;·&nbsp; lining up every dip makes the pattern easier to see", folded_figure(curve))
    with st.container(key="card-evidence"):
        html_block('<div class="z-h3">Check the evidence</div><div class="z-caption">Screening checks for signs of another star '
                   'or noise. They are heuristics, not confirmation or a probability.</div>')
        try:
            result = evidence_checks(target)
        except (ValueError, OSError, RuntimeError, KeyError):
            st.write("Evidence checks need more data than this curve provides.")
        else:
            html_block(f'<div class="z-verdict">{html.escape(result["verdict"])}</div><div class="z-text">{html.escape(result["summary"])}</div>')
            with st.expander("Evidence checks"):
                for test in result["tests"]:
                    mark = "Passed" if test["passed"] is True else "Failed" if test["passed"] is False else "Not enough data"
                    value = f"{test['value']:.3f} {test['unit']}" if test["value"] is not None else "insufficient coverage"
                    st.markdown(f"**{test['name']}** · {mark} · {value}  \n{test['explanation']} Threshold: {test['threshold']}.")
    with st.expander("Detection details"):
        st.dataframe(pd.DataFrame([
            {"Measurement": "Period", "Value": f"{curve['period_days']:.6f} days"},
            {"Measurement": "Transit epoch", "Value": f"{curve['epoch']:.4f} BKJD"},
            {"Measurement": "Depth", "Value": f"{curve['depth'] * 1e6:.1f} ppm"},
            {"Measurement": "Duration", "Value": f"{curve['duration_days']:.4f} days"},
            {"Measurement": "Signal detection efficiency (SDE)", "Value": f"{curve['sde']:.2f}"},
            {"Measurement": "Detection threshold", "Value": f"SDE ≥ {BASELINE_CONFIG.detection_threshold:g}"},
            {"Measurement": "Source file", "Value": curve["product"]},
        ]), hide_index=True, width="stretch")
        st.caption("SDE measures how far the strongest Box Least Squares peak stands above the other trial periods. "
                   "It is not a probability. Detector settings: " + ", ".join(f"{k} = {v}" for k, v in BASELINE_CONFIG.model_dump().items()) + ".")
        raw = go.Figure(go.Scattergl(x=curve["time"], y=curve["flux"], mode="markers", marker=dict(size=3, color="#96a6cb")))
        chart_card("raw", "Raw brightness before smoothing (median normalized)", style_figure(raw, 200, "Time (days)"))
        power = go.Figure(go.Scatter(x=curve["periods"], y=curve["power"], line=dict(color="#ad7dee")))
        chart_card("bls", "Box Least Squares power vs trial period", style_figure(power, 200, "Period (days)"))
        st.page_link(PAGES["how"], label="How the detector works", icon=":material/info:")


def run_status(run, worker_status, runs):
    if worker_status["running"] and runs and run["loop_id"] == runs[0]["loop_id"]:
        return "running"
    return run["status"]


def start_research(worker):
    if worker.launch(st.session_state.get("research_rounds", DEFAULT_ROUNDS)):
        st.session_state["research_view"] = "Overview"
        st.toast("Research started.")


def setup_area(worker, state, title="Start a research run"):
    roles = [t.get("role") for t in state["targets"]]
    html_block(f'<div class="z-h3">{title}</div><div class="z-text">Each round, the Scientist reads published methods and proposes one '
               'change to the planet detector, the Skeptic reviews it, and approved changes are tested.</div>'
               f'<div class="z-text">Tests use simulated transit signals added to real Kepler observations of {roles.count("dev")} '
               f'development stars, so the right answers are known. A final test on {roles.count("holdout")} different stars runs at the end.</div>')
    mode = ("<b>Demo mode</b> · recorded AI responses, no paid API calls." if demo_mode()
            else "<b>Live mode</b> · calls OpenAI and Exa, which may incur charges.")
    html_block(f'<div class="z-mode">{mode}</div>')
    with st.expander("Advanced settings"):
        rounds = st.selectbox("Research rounds", list(range(1, 11)), index=DEFAULT_ROUNDS - 1, width=254, key="research_rounds")
        st.caption(ROUNDS_HELP)
    st.button("Start research", type="primary", key="research_start", disabled=worker.status()["running"],
              on_click=start_research, args=(worker,))


def method_rows(start, selected):
    rows = []
    for label, m in (("Starting method", start), ("Selected method", selected)):
        if m is None:
            continue
        c = counts(m)
        found = f"{c['recovery_rate'][0]} of {c['recovery_rate'][1]}" if "recovery_rate" in c else "—"
        alarms = f"{c['false_alarm_rate'][0]} of {c['false_alarm_rate'][1]}" if "false_alarm_rate" in c else "—"
        rows.append({"Method": label, "Test signals recovered": f"{found} ({m['recovery_rate']:.0%})" if "recovery_rate" in m else "—",
                     "False alarms on quiet stars": f"{alarms} ({m['false_alarm_rate']:.0%})" if "false_alarm_rate" in m else "—",
                     "Runtime": f"{m['runtime_s']:.1f} s" if m.get("runtime_s") is not None else "—"})
    return pd.DataFrame(rows)


def holdout_baseline(holdout):
    try:
        return get_experiment_results(holdout["baseline_run_id"])["measurements"]
    except (KeyError, OSError, ValueError, TypeError):
        return None


def render_overview(view, run, status, worker, state, runs):
    worker_status = worker.status()
    key, title, text = outcome(view, status)
    with st.container(key=f"card-outcome-{key}"):
        html_block(f'<div class="z-outcome {key}">{title}</div><div class="z-text">{text}</div>')
        if status == "running":
            done = run["rounds"]
            total = worker_status.get("rounds") or "?"
            last = next((describe(e) for e in reversed(view["events"]) if describe(e)), None)
            current = worker_status["iteration"] or 1
            html_block(f'<div class="z-progress"><div><span>Round</span><b>{current} of {total}</b></div>'
                       f'<div><span>Elapsed</span><b>{elapsed(run["started"]) or "—"}</b></div>'
                       f'<div><span>Rounds finished</span><b>{done}</b></div></div>')
            if last:
                html_block(f'<div class="z-text"><b>Now:</b> {html.escape(last[0])}</div>')
            completed = next((e["payload"] for e in reversed(view["events"]) if e.get("step") == "iteration_complete"), None)
            html_block('<div class="z-text"><b>Latest completed result:</b> ' + (
                f"Round {completed['iteration']} · " + describe({"step": "iteration_complete", "payload": completed})[0].split(" · ")[-1]
                + ". " + plain_result(completed.get("measurements")) if completed and completed.get("measurements") else
                f"Round {completed['iteration']} · not tested." if completed else "No round has finished yet.") + "</div>")
        else:
            start = (view["baseline"] or {}).get("measurements")
            selected = selected_measurements(view)
            if start:
                st.dataframe(method_rows(start, selected), hide_index=True, width="stretch")
                html_block(f'<div class="z-caption">Development stars · {run["rounds"]} round{"s" * (run["rounds"] != 1)} · '
                           f'{elapsed(run["started"], run.get("ended") or run.get("last")) or ""}</div>')
            if view["holdout"]:
                final = view["holdout"]["measurements"]
                html_block('<div class="z-text"><b>Final test:</b> ' + plain_result(final) + "</div>")
            st.button("See activity", key="overview_activity", on_click=st.session_state.update, kwargs={"research_view": "Activity"})
    if worker_status["error"] and status != "running":
        st.error(f"Research stopped: {worker_status['error']}. Completed rounds are kept. Check service availability and try again.")
    if status != "running":
        with st.expander("Start another run"):
            setup_area(worker, state, "New research run")


def render_activity(view, state):
    entries = [(e, describe(e)) for e in view["events"]]
    entries = [(e, d) for e, d in entries if d]
    if not entries:
        st.write("Activity will appear when research starts.")
        return
    html_block('<div class="z-caption">Each entry is a recorded event. Expand it for the full proposal, review, sources, settings and evidence.</div>')
    for event, (title, why) in entries:
        p = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        stamp = event.get("timestamp", "")[11:19]
        with st.expander(f"{stamp} · {title}"):
            st.write(why)
            if p.get("hypothesis") and event.get("step") in ("hypothesis", "revision"):
                st.markdown("**Proposal**")
                st.write(p["hypothesis"])
                if p.get("cited_sources"):
                    st.markdown("**Sources**")
                    for source in p["cited_sources"]:
                        st.markdown(f"- [{('[MOCK] ' if source.get('mock') else '') + source['title']}]({source['url']})")
                if p.get("memory_influence", {}).get("memory_ids"):
                    verified = p["memory_influence"].get("explanation", "").startswith("Deterministic memory veto")
                    st.caption(("Shared memory changed this proposal: " if verified else "Model-reported memory influence (unverified): ")
                               + p["memory_influence"].get("explanation", ""))
                if p.get("proposed_config"):
                    st.markdown("**Settings tested**")
                    st.dataframe(pd.DataFrame([{"Setting": k, "Before": str((p.get("current_best_config") or {}).get(k, "—")), "Proposed": str(v)}
                                               for k, v in p["proposed_config"].items()]), hide_index=True, width="stretch")
            if event.get("step") in ("review", "review_revision"):
                st.markdown("**Reviewer comments**")
                st.write(p.get("rationale", "Not recorded"))
            if p.get("measurements"):
                st.markdown("**Recorded evidence**")
                st.dataframe(pd.DataFrame([{"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else v}
                                           for k, v in p["measurements"].items()]), hide_index=True, width="stretch")
    activity = memory_activity(view["events"], state["memories"])
    if activity:
        with st.expander("Shared memory exchanges"):
            st.dataframe(pd.DataFrame(activity), hide_index=True, width="stretch")
    st.page_link(PAGES["report"], label="Sources & methods · Technical report", icon=":material/description:")


def render_run_results(view, run, status):
    start = (view["baseline"] or {}).get("measurements")
    if not start:
        st.write("Results will appear after the starting method is measured.")
        return
    selected = selected_measurements(view)
    with st.container(key="card-dev-results"):
        html_block('<div class="z-h3">Development results</div><div class="z-caption">Every method is measured on the same '
                   'development stars with simulated signals added.</div>')
        st.dataframe(method_rows(start, selected), hide_index=True, width="stretch")
        if selected is start or selected == start:
            html_block('<div class="z-text">No change beat the starting method, so it stays selected.</div>')
    with st.container(key="card-final-test"):
        html_block('<div class="z-h3">Final test</div><div class="z-caption">Tested on stars that weren’t used to choose the settings.</div>')
        if view["holdout"]:
            base = holdout_baseline(view["holdout"])
            st.dataframe(method_rows(base, view["holdout"]["measurements"]) if base else
                         method_rows(None, view["holdout"]["measurements"]), hide_index=True, width="stretch")
        else:
            st.write("The final test runs when research finishes." if status == "running" else "No final test was recorded for this run.")
    rows = [{"Run": "Starting method", "Review": "baseline", **start}]
    rows += [{"Run": f"Round {i}", "Review": r["reviewer_decision"]["decision"], **r["measurements"]}
             for i, r in enumerate(view["current"], 1)]
    with st.expander("Detailed results"):
        host_count = (view["baseline"] or {}).get("provenance", {}).get("host_count", 0)
        st.caption(f"Stars analyzed: {host_count}")
        frame = pd.DataFrame(rows)
        st.dataframe(frame, hide_index=True, width="stretch")
        st.download_button("Download results (CSV)", frame.to_csv(index=False), file_name=f"zathura-{run['loop_id']}.csv",
                           mime="text/csv", key=f"download_{run['loop_id']}")
        for i, r in enumerate(view["current"], 1):
            st.caption(f"Round {i} · {r['hypothesis'][:160]} · review: {r['reviewer_decision']['decision']}")
        measured = frame.dropna(subset=["balanced_accuracy"]) if "balanced_accuracy" in frame else frame.iloc[0:0]
        if len(measured):
            fig = px.bar(measured, x="Run", y="balanced_accuracy", color="Review",
                         color_discrete_map={"baseline": "#0ae4ea", "approve": "#ad7dee", "reject": "#ff829f"})
            fig.update_yaxes(range=[0, 1], title="Balanced accuracy")
            chart_card(f"detail-{run['loop_id']}", "Balanced accuracy by round (development)", style_figure(fig, 240, legend=True))
        for label, m in (("Development, starting method", start), ("Final test, selected method", (view["holdout"] or {}).get("measurements"))):
            if m and m.get("balanced_accuracy_ci_low") is not None:
                st.caption(f"{label}: balanced accuracy {m['balanced_accuracy']:.4f}, 95% interval "
                           f"[{m['balanced_accuracy_ci_low']:.4f}, {m['balanced_accuracy_ci_high']:.4f}] (bootstrap over stars).")
        if view["summary"]:
            st.caption("Selected configuration: " + ", ".join(f"{k} = {v}" for k, v in view["summary"]["best_config"].items()))


def render_run_source(view):
    source = (view["baseline"] or {}).get("data_source")
    if source == "placeholder":
        st.warning("Stand-in engine (placeholder) run")
        st.caption("Stand-in scores, not measurements on real Kepler observations.")
    elif source == "real":
        badge(source)
        st.caption(BENCHMARK_DISCLOSURE)
    else:
        st.caption("Data source not recorded for this run.")


def render_research(state, worker):
    page_header("Research", text="Zathura’s AI proposes changes to its planet detector and tests them.",
                title_size=36, text_gap=4, benchmark=False)
    runs = run_list(state["events"])
    worker_status = worker.status()
    if not runs:
        if worker_status["running"]:
            with st.container(key="card-outcome-running"):
                html_block('<div class="z-outcome running">Starting research</div>'
                           '<div class="z-text">Loading observations and measuring the starting method.</div>')
        else:
            with st.container(key="card-setup"):
                setup_area(worker, state)
            if worker_status["error"]:
                st.error(f"Research stopped: {worker_status['error']}. Check service availability and try again.")
        return
    ids = [r["loop_id"] for r in runs]
    if worker_status["running"] or st.session_state.get("research_run") not in ids:
        st.session_state["research_run"] = ids[0]
    by_id = {r["loop_id"]: r for r in runs}

    def run_label(loop_id):
        r = by_id[loop_id]
        stamp = (r["started"] or "")[:16].replace("T", " ")
        state_label = run_status(r, worker_status, runs).replace("incomplete", "stopped")
        return f"Run started {stamp} UTC · {r['rounds']} round{'s' * (r['rounds'] != 1)} · {state_label}"

    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium", key="run-row"):
        loop_id = st.selectbox("Research run", ids, format_func=run_label, key="research_run", width=520,
                               disabled=worker_status["running"])
        view_name = st.segmented_control("View", ["Overview", "Activity", "Results"], key="research_view",
                                         label_visibility="collapsed") or "Overview"
    run = by_id[loop_id]
    view = run_view(state, loop_id)
    render_run_source(view)
    status = run_status(run, worker_status, runs)
    if view_name == "Activity":
        render_activity(view, state)
    elif view_name == "Results":
        render_run_results(view, run, status)
    else:
        render_overview(view, run, status, worker, state, runs)


STYLE = [""]


def live(render):
    @st.fragment(run_every=5)
    def body():
        if STYLE[0]:
            st.html(STYLE[0])
        state = snapshot()
        if state["warning"]:
            st.warning(state["warning"])
        render(state)
    body()


def setup():
    st.set_page_config(page_title="Zathura · AI planet search", page_icon=str(ASSETS / "zathura_mark.png"), layout="wide",
                       initial_sidebar_state="auto")
    st.logo(str(ASSETS / "zathura_logo.png"), size="large", icon_image=str(ASSETS / "zathura_mark.png"))
    variables = "".join(mask_var(name) for name in ("home", "planet", "flask", "info", "code", "chevdown"))
    STYLE[0] = f"<style>:root{{{variables}}}\n{(ASSETS / 'zathura.css').read_text()}</style>"
    st.html(STYLE[0])


def home_page():
    topbar("Home")
    render_home(snapshot())


def explore_page():
    topbar("Explore stars")
    render_explore(snapshot())


def research_page():
    topbar("Research")
    st.session_state.setdefault("research_view", "Overview")
    live(lambda state: render_research(state, mission()))


def how_page():
    topbar("How it works")
    live(render_how_it_works)


def report_page():
    topbar("Technical report")
    live(render_sponsors)


PAGES = {}


def build_pages():
    PAGES.update(home=st.Page(home_page, title="Home", icon=":material/home:", default=True),
                 explore=st.Page(explore_page, title="Explore stars", icon=":material/public:", url_path="explore"),
                 research=st.Page(research_page, title="Research", icon=":material/science:", url_path="research"),
                 how=st.Page(how_page, title="How it works", icon=":material/info:", url_path="how-it-works"),
                 report=st.Page(report_page, title="Technical report", icon=":material/description:", url_path="technical-report"))
    return PAGES


def main():
    setup()
    build_pages()
    st.navigation({"": [PAGES["home"], PAGES["explore"], PAGES["research"]],
                   "Reference": [PAGES["how"], PAGES["report"]]}, position="sidebar").run()


if __name__ == "__main__":
    main()
