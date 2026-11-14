# Zathura — first slice

An offline-capable, **placeholder** benchmark for tuning a Box Least Squares
exoplanet transit detector. No skeptic, memory service, orchestrator or UI yet.

## Run (Python 3.10+)

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/smoke.py
.venv/bin/python -m unittest discover -s tests -v
```

The smoke script deliberately forces `ZATHURA_MOCK=1`. Library calls otherwise
use OpenAI when `OPENAI_API_KEY` is set and Exa when `EXA_API_KEY` is set,
independently. Model: `ZATHURA_MODEL` (default `gpt-4o-mini`). Missing keys use
clearly labelled deterministic mocks. Configured live-client errors propagate;
they are never silently replaced with mock research. Live API calls have not
been credential-tested. Do not commit `.env` or keys.

## Engine contract

Import `load_targets`, `get_lightcurve`, `run_baseline`, `run_experiment`,
`evaluate_experiment`, `get_experiment_results` from `zathura.engine`.
Both run functions return a persistent run ID. Run a baseline before evaluating
an experiment on the same dataset. `evaluate_experiment` returns improvement
against baseline and the best other run on that dataset (strict improvement;
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

Datasets `dev` and `holdout` use separate fixed seeds, controlled box-transit
injections on half the curves, and non-injected controls on the rest. Holdout
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
