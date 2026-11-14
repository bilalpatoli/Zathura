# Zathura — autonomous transit research

An offline-capable injection-recovery benchmark on real NASA Kepler light curves
for tuning a Box Least Squares exoplanet transit detector, with a scientist,
independent skeptic, shared memory, a bounded research loop and a live Streamlit
mission dashboard.

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
NASA or agent calls; the committed cache needs no network access.

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

Benchmark results are `data_source="real"`: injections into real Kepler PDCSAP
curves. Results recorded by older placeholder engines keep their original label.
Mock literature and mock agent output remain separately labelled. Benchmark stars
have no KOI or TCE, but they are not certified planet-free: native signals and
residual stellar variability can still produce "false alarms" in controls.

Injected transits are trapezoids, not limb-darkened physical models, and SNR is
measured against a fixed ~1-day reference detrend, so the absolute recovery rates
are benchmark-specific. Comparisons between configs are the intended use: every
config sees identical examples, and confidence intervals resample whole stars.
Only one quarter per star is used (period search 1-10 days). The known-planet
check validates the detector on real transits, not just injections.

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
`evaluate_experiment`, `get_experiment_results`, `validate_known_planets` and
`inspect_target` from `zathura.engine`. Both run functions return a persistent
run ID. Run a baseline before evaluating an experiment on the same dataset.
`evaluate_experiment` compares against the latest preceding baseline and the best
preceding run since that baseline (same dataset and benchmark version).
`improved_*` are strict point-estimate comparisons; `significant_*` require the
paired 95% bootstrap interval of the balanced-accuracy delta to exclude zero. Use
`significant_*` to claim a method is better. Results include config, aggregate
metrics and provenance, never per-example answers.

### Data

`scripts/build_catalog.py` queries the NASA Exoplanet Archive and downloads one
quality-masked long-cadence PDCSAP quarter per star from MAST (`data/catalog.json`,
`data/cache/kic_*.npz`, ~4 MB, committed):

- 40 dev and 40 holdout benchmark stars, disjoint: quiet FGK dwarfs (Teff
  4800-6500 K, log g > 4, Kp 11-14.5) with no KOI or TCE.
- 16 validation stars hosting confirmed planets with 1-10 day periods, spread in
  catalog SNR. These never enter the benchmark.

Rebuilding queries live archives, so the selection can change; the committed
catalog is the reference.

### Benchmark (`kepler-injection-v4`)

Each benchmark star gives 4 injected transits and 1 un-injected control (160
positives and 40 negatives per split; seeds 42 dev, 9173 holdout). Injections:
log-uniform period 1-10 days with at least 3 observed transits, impact parameter
0-0.9, duration from solar density, SNR 5-25 log-uniform, where SNR = depth /
(noise at the transit timescale / sqrt(transits)). The real photometric noise is
the noise model. The SHA-256 of the private truth table and of the curve files is
recorded with every run.

Preprocessing sorts, normalizes, flattens with Lightkurve (`detrend_window` in
29.4-minute cadences) and clips only upward outliers. BLS uses a grid uniform in
frequency over 1-10 days and 6 log-spaced durations; SDE = (peak power - mean) /
std. Recovery needs an above-threshold peak within 1% of the injected period and
epoch within one transit duration; detections at 1/3, 1/2, 2 or 3 times the period
are reported as `harmonic_rate`, not recovery. Metrics: recovery, false-alarm
rate, balanced accuracy with a star-level bootstrap CI, recovery per SNR bin and
runtime per curve. `ZATHURA_WORKERS` sets the BLS thread count (default
min(8, CPUs)); astropy's BLS kernel releases the GIL.

### Reproducing results

```sh
.venv/bin/python scripts/reproduce.py --config data/candidate_config.json
```

Re-runs baseline and a tuned candidate on dev and holdout, repeats dev to check
determinism, validates known planets, and compares every non-timing number with
`data/reference_results.json` (`--update-reference` rewrites it).

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
The `grounded-review-v2` policy verifies fairness from the actual config/request:
a bounded single-parameter change on fixed dev examples is fair by construction.
Only observable blocking facts (invalid comparison, exact measured duplicate,
disabled cutoff, or a measured-runtime-based estimate above the 30-second dev
budget) can justify a rejection category. Runtime extrapolations are labelled
estimates, not measurements. Uncertain improvement or possible false positives
alone are left to the evaluator. Model reasoning is retained in `model_reviews`;
unsupported historical sentences are removed individually, and an empty,
label-only, or policy-inconsistent response gets one corrective model call rather
than a canned rationale or a silent decision override. Legacy ungrounded critiques
and unexecuted rejected records cannot poison subsequent scientist decisions;
actual measured negative results remain eligible for memory vetoes.
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
