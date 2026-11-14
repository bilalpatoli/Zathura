# Zathura — autonomous transit research

An offline-capable, **placeholder** benchmark for tuning a Box Least Squares
exoplanet transit detector, with a scientist, independent skeptic, shared memory
and a bounded research loop, with a live Streamlit mission dashboard.

## Dashboard

```sh
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The four tabs are Mission Control, Planet Detection,
AI Research Timeline, and Experimental Results. Theme settings live in
`.streamlit/config.toml`; charts use `plotly_dark`. No API credentials are needed
to browse existing results. Missing data produces empty states without automatic
NASA or agent calls. The explicit source-load button may download NASA curves,
falling back to labelled synthetic curves if downloads fail.

Launch Research Mission starts a daemon background worker. A two-second Streamlit
fragment refreshes the timeline and results; worker callbacks never call Streamlit.
Only one mission can run per dashboard process, including across browser sessions.
Closing the browser does not stop it, but stopping/restarting Streamlit does.
Do not concurrently run a CLI mission: JSONL storage has no cross-process locks.
Launching can incur API charges when live credentials are configured. Backend
indicators show configuration, not verified uptime; historical mock output remains
labelled. A failed mission displays only its error type, not transport secrets.

### Architecture and data flow

- `app.py`: four `render_*` functions, Plotly presentation and a polling fragment.
- `zathura/dashboard.py`: read-only snapshots from storage, events, memory mirror
  and held-out results; `Mission.launch` owns the background thread.
- `zathura/engine/__init__.py`: `cached_targets` reads available source metadata
  without downloading; `inspect_target` returns read-only, un-injected light-curve
  diagnostics. The UI never opens benchmark truth or calculates detections itself.
- `zathura/orchestrator.py::run_research_loop`: scientist → skeptic → optional
  single revision/binary review → approved dev experiment → evaluator → storage and
  shared memory → final holdout. Events go to `data/events.jsonl` and the callback.
- **All Exa calls:** `zathura/agents/exa_search.py::search_papers`, specifically
  `Exa(api_key=key).search_and_contents(...)`. The scientist calls this wrapper;
  the dashboard only renders returned source titles and links. Missing keys or
  `ZATHURA_MOCK=1` select clearly labelled mock OpenAI/Exa behavior.
- **past.dev identities:** `research-scientist` writes hypotheses and measured
  results; `skeptic` writes critiques. Both recall project-visible entries written
  by either author. With `PASTDEV_API_KEY`, writes hit Past.dev ingestion first and
  are mirrored to `data/pastdev-memory.jsonl` after acceptance. Loop/UI reads use
  that mirror to avoid indexing latency; explicit `recall_remote` and
  `scripts/check_memory.py` verify server-only exchange. Without a key,
  `data/memory.jsonl` is the local backend, not a simulated Past.dev service.
  The exchange view shows the most recent 200 author-attributed entries.

### Methodology and limitations

Every current detector/benchmark result is `data_source="placeholder"`, visibly
badged **PLACEHOLDER engine**. Real cached NASA observations do not make this a
validated planet detector. Synthetic source fallbacks, mock literature and mock
agent output are separately labelled. Catalog host labels are not validation of
the displayed peak; rounded known periods for Kepler-69/10/22 are reference values
from the NASA Exoplanet Archive. The diagnostic searches only 1–10 days with the
baseline configuration, so long catalog periods cannot be recovered here.

Each dev/holdout split has only 36 fixed examples: four injected depths and two
controls per cached star (24 positives, 12 negatives). Splits use different fixed
seeds but reuse the same six source stars, so they are not independent stellar
populations. Native signals may contaminate negatives. Recovery requires a period
match within 3%; false-alarm rate uses controls; balanced accuracy averages
recovery and specificity. Runtime is measured detector time, not total API latency.
The target plots show source curves, not injected benchmark examples. The strongest
BLS peak is unverified, and below-threshold peaks are explicitly not detections.

Dashboard baseline/best comparisons and charts use only the latest mission to
avoid mixing benchmark versions. Rejections have missing metrics, not zero scores.
Acceptance means strictly improved dev balanced accuracy, not just reviewer
approval. The separate held-out panel shows the latest completed holdout and its
loop ID; during a new mission it may still belong to the previous mission. Agents
never receive holdout results. The Python API boundary is not a security sandbox.
Historical LLM-reported memory influence without deterministic veto evidence is
marked UNVERIFIED, not presented as proof that memory changed a decision.

Headless checks:

```sh
.venv/bin/python scripts/smoke_dashboard.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/streamlit run app.py --server.port 8501 --server.headless true
curl http://localhost:8501/_stcore/health
```

## Run (Python 3.10+)

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/smoke.py
.venv/bin/python -m zathura.orchestrator --iterations 3
.venv/bin/python -m unittest discover -s tests -v
```

The smoke script deliberately forces `ZATHURA_MOCK=1`. Library calls otherwise
use OpenAI when `OPENAI_API_KEY` is set and Exa when `EXA_API_KEY` is set,
independently. Model: `ZATHURA_MODEL` (default `gpt-4o-mini`). Missing keys use
clearly labelled deterministic mocks. Configured live-client errors propagate;
they are never silently replaced with mock research. Real OpenAI + Exa scientist
calls have been checked. Do not commit `.env` or keys.

## Engine contract

Import `load_targets`, `get_lightcurve`, `run_baseline`, `run_experiment`,
`evaluate_experiment`, `get_experiment_results` from `zathura.engine`.
Both run functions return a persistent run ID. Run a baseline before evaluating
an experiment on the same dataset. `evaluate_experiment` returns improvement
against the latest preceding baseline and best preceding run since that baseline
on the same dataset and benchmark version (strict improvement;
delta is balanced accuracy minus baseline). Results include config, aggregate
metrics and provenance, never per-target answers.

`load_targets()` attempts Lightkurve/MAST downloads with a 25-second limit per
target. Compact normalized curves (first 28 days) are cached under `data/cache`.
If unavailable, explicitly marked synthetic curves are cached instead; delete
that target's `.npz` to retry. `curve_origin` distinguishes real source curves
from synthetic placeholders; **all benchmark results remain placeholder**.
Small cache files are committed so subsequent demos need no NASA access.
Source targets include Kepler-69/10/22 and three solar-like oscillation targets
used as quiet-star candidates, not certified planet-free stars.

Datasets `dev` and `holdout` use separate fixed seeds. Each has 36 examples:
four injections per cached star at depths 300, 700, 1500 and 3000 ppm, plus two
non-injected controls per star. Period, epoch, duration and small independent
noise realizations vary; all configurations see the same examples. Source
curves are shared between splits, but injected signals and noise are not.
These are correlated examples, not 36 independent stars. Holdout
truth stays in private engine internals; agents only request dev. This is an
API boundary, not protection against arbitrary code execution in the same
Python process. Controls on real curves can contain native astrophysical
signals; this small benchmark is not a scientifically clean survey.

Flattening uses Lightkurve's `flatten(window_length=...)`; sigma clipping removes
upper outliers while preserving transit dips. BLS searches 1–10 days with four
durations in the configured range. Threshold is SDE = (peak power - mean power)
/ standard deviation of grid power. Recovery requires an above-threshold peak
within 3% of the injected period (no harmonic credit). False alarms are the
fraction of non-injected controls with above-threshold peaks. Balanced accuracy
is `(recovery_rate + 1 - false_alarm_rate) / 2`. Runtime measures experiment
execution, not initial downloads. Larger grids cost more; default dev is small.

JSONL experiment records live at `data/experiments.jsonl`; engine run JSON files
live at `data/runs/`. Both are local, ignored artifacts. Storage is append-only
for single-process use. Smoke records are explicitly unreviewed and unaccepted;
there is no fabricated skeptic approval. The scientist validates configurations,
proposes one untried parameter change from the supplied/current accepted best,
and restricts citations to search results. Mock search snippets are canned,
not claims that the cited papers were retrieved or read.

## Shared memory and review loop

`write_memory(agent_id, content, kind, refs=None)` returns a memory id;
`recall_memory(agent_id, query, k=5)` returns author-attributed entries, including
other agents. Authors are `research-scientist` and `skeptic`. With no
`PASTDEV_API_KEY`, the local backend uses `data/memory.jsonl` ranked by keyword
overlap then recency. Memory content may be structured JSON or text.

Real Past.dev docs were found at <https://past.dev/mcp> and
<https://past.dev/openapi.json>. `PastDevBackend` implements bearer-authenticated
REST ingestion and recall with source metadata
preserving the original author and envelope. Entries are project-visible so both
agents share them. Use a dedicated Zathura project key in `PASTDEV_API_KEY` (the
vendor examples call this `PAST_API_KEY`). Configured-backend errors propagate;
there is no silent write fallback. When set, this key always selects Past.dev.
Every successful remote ingest is immediately mirrored to `data/pastdev-memory.jsonl`.
Loop reads use that separate mirror, so queued ingestion and remote recall latency
do not block research. Empty mirrors bootstrap with remote recall; use
`recall_remote` for explicit synchronization and server-only evidence, and
`wait_until_ready` for bounded ingestion polling outside the loop.
Run `.venv/bin/python scripts/check_memory.py` to verify server-only recall in
both author/reader directions (requires a real key).

The scientist recalls shared critiques/results before proposing and writes each
hypothesis back. Citations are canonicalized by supplied URL rather than exact
LLM-regenerated snippet equality. Live OpenAI calls use strict JSON schemas with
an enumerated candidate id and source URLs; candidate ids resolve to trusted
validated configurations, eliminating invented/multi-parameter configurations.
Numeric LLM output is also bounded/odd-normalized,
logged in `config_adjustments`, then strictly checked against untried candidates;
invalid output gets one corrective retry, not arbitrary parameter substitution.
`memory_influence` contains recalled ids and an explanation only for an evidenced
deterministic veto; unsupported model-reported influence is discarded.
A deterministic safety check detects an actual first-choice configuration or
parameter direction already failed/rejected in recalled structured memory,
requests a safe alternative and records that veto explicitly. No memory benefit
is claimed when there is no evidence. Negative directions are conservatively
avoided; they are not scientifically proven universally bad.
After the final selection, including any memory veto, hypothesis and prediction
prose are regenerated from the actual parameter, old/new values and a parameter-specific
mechanism. Stale LLM prose is discarded, with mismatched parameter mentions recorded
in `prose_validation`. A veto explanation renders the actual final configuration,
not the LLM's potentially stale explanation.

The skeptic uses its own system prompt, recalls cross-agent history, checks the
bounded comparison and false-positive risk, and writes critiques. Per iteration
the loop allows one revision and one re-review. The second review is binary
`approve|reject`, enforced in both the OpenAI JSON schema and Pydantic model.
Cheap bounded untried experiments default to approval unless a concrete issue
exists; uncertain benefits are left to the evaluator. Historical claims in free
prose are dropped. Prior outcomes are rendered only from recalled measured-result
entries identified by valid memory ids; unknown ids are discarded and reported.
Unapproved final proposals count
as rejected, are logged with empty measurements, and never run. Approved proposals
run/evaluate on dev, are stored, and their results enter shared memory. Acceptance
means strict `improved_vs_best`; the tuning incumbent updates on better dev score.
Each loop starts a new baseline comparison session; benchmark versioning excludes
the earlier tiny benchmark from current comparisons.

`on_event` receives `{loop_id, step, agent, payload, timestamp}` dictionaries, also
appended to `data/events.jsonl`. After tuning, best config runs once on holdout
(plus a holdout baseline); results go separately to `data/holdout.jsonl` and engine
runs/events, **never to either agent or shared memory**. Events are for the future
dashboard. All mutable data files are ignored. JSONL backends are single-process.
