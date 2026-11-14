# Zathura — first slice

An offline-capable, **placeholder** benchmark for tuning a Box Least Squares
exoplanet transit detector, with a scientist, independent skeptic, shared memory
and a bounded research loop. No dashboard yet.

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
