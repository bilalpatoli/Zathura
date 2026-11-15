# Zathura demo readiness

Status as of branch `devin/*-demo-readiness`, based on `main` at `40763b0`.

## 1. Setup, startup, tests, reproduction

Python **3.10.12** (Linux x86_64 verified). Pinned environment: `requirements.lock`
(`requirements.txt` stays the loose, human-edited list).

```sh
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.lock
ZATHURA_MOCK=1 .venv/bin/python -m pytest -q            # unit + regression tests
ZATHURA_MOCK=1 .venv/bin/python scripts/smoke_dashboard.py
.venv/bin/python scripts/reproduce.py --config data/candidate_config.json   # exit 0 = PASS
.venv/bin/python scripts/validate_vetting.py              # screening report, offline
ZATHURA_MOCK=1 .venv/bin/streamlit run app.py --server.port 8501   # demo mode, no paid calls
.venv/bin/streamlit run app.py --server.port 8501                  # live mode (keys required)
```

## 2. Environment variables (names only)

| Name | Purpose |
|---|---|
| `OPENAI_API_KEY` | Scientist and skeptic agents (live). Missing → mock responses. |
| `ZATHURA_MODEL` | Model name, default `gpt-4o-mini`. |
| `EXA_API_KEY` | Literature search for the scientist (live). |
| `PASTDEV_API_KEY` | Shared agent memory on past.dev; without it memory is the local `data/memory.jsonl`. |
| `ZATHURA_MOCK` | `1` forces labelled mock agents/search, even when keys exist. |
| `ZATHURA_WORKERS` | BLS thread count. |

Never commit `.env`; it is gitignored.

## 3. Offline vs live

Works offline (the cached Kepler curves are committed under `data/cache/`):
browsing stars, light curves, BLS detection, vetting, the benchmark, reproduction,
all tests, and a full **mock** research run (labelled mock in the UI and in the records).

Live services are needed only for a real autonomous research run: OpenAI (agents),
Exa (sources), past.dev (shared memory). Mock runs are never evidence of AI behavior.

## 4. Verified scientific results (`kepler-injection-v4`)

Simulated transit signals injected into **real Kepler observations**: 40 host stars per split,
160 injected signals + 40 no-injection controls per split; dev and holdout stars are disjoint;
16 separate known-planet hosts. Balanced accuracy = (recovery + 1 − false-alarm rate) / 2,
with 95% host-level bootstrap intervals. Runtime is wall-clock on the verification VM and is
excluded from reproduction checks.

| Config | Split | Balanced acc. (95% CI) | Recovery | False alarms | Runtime |
|---|---|---|---|---|---|
| Baseline | dev | 0.459 (0.338–0.572) | 63/160 (39.4%) | 19/40 (47.5%) | 3.8 s |
| Baseline | holdout | 0.384 (0.272–0.500) | 51/160 (31.9%) | 22/40 (55.0%) | 4.6 s |
| Candidate | dev | 0.719 (0.666–0.772) | 74/160 (46.3%) | 1/40 (2.5%) | 9.5 s |
| Candidate | holdout | 0.644 (0.556–0.722) | 66/160 (41.3%) | 5/40 (12.5%) | 9.4 s |

Known-planet period-match check (16 hosts): baseline 12/16, candidate 15/16. This checks that the
detector finds the **catalog period**; it is not independent planet confirmation.

### Provenance of each configuration

- **Baseline** (`BASELINE_CONFIG` in `zathura/schema.py`): fixed starting method.
- **Candidate** (`data/candidate_config.json`: `detrend_window` 101, `period_grid_size` 10000,
  `detection_threshold` 9.0): **separately supplied**, hand-chosen during engine development in
  commit `2e679c3`. It was **not** selected by an autonomous AI run. Its holdout numbers were
  computed after it was fixed, but the holdout split has since been viewed repeatedly, so a fresh
  generalization claim for any newly selected config would need a new untouched split.
- **AI-selected configuration (live run, §10):** `outlier_sigma` 3.0, other settings unchanged from the baseline.
  It was selected by the live loop on dev data, but its gain is within noise (see §10).

## 5. Reproducibility

- Commit under test: this branch; benchmark version `kepler-injection-v4`; seeds dev 42, holdout 9173.
- Truth hashes (canonical, 9 significant digits): dev `6ca024ce2c4a4fd9…`, holdout `63debb333963f63f…`.
  Curve checksums: dev `23aa090420ec4207…`, holdout `486df4ba2b4e339c…` (full values in `data/reference_results.json`).
- Root cause of the audited `truth_sha256` mismatch: the old hash serialized full-precision float
  reprs of injection parameters derived from `exp`/`log` and median filters, whose last bits differ across
  CPUs/NumPy builds. The examples were the same; the hash was fragile. The legacy hash is still recorded
  as `truth_sha256_raw` in run provenance, and the previous reference is kept at
  `data/reference_results.legacy-raw-truth-hash.json`. On this VM the raw hash still equals the legacy
  reference for both splits, which shows the examples did not change. All measurements and known-planet
  results in the new reference are identical to the legacy one.
- Root cause of the ~3.6e-9 SDE differences: floating-point accumulation in BLS on a different platform.
  Floats now compare at 1e-6 relative; counts, rates, decisions, configs and checksums stay exact.
- `scripts/reproduce.py` now exits nonzero when the reference is missing, a label or config has no reference, or anything differs.
- Clean-environment check (fresh clone + fresh venv from `requirements.lock`): two consecutive runs both
  PASS with identical output: `docs/evidence/reproduction-clean-env.txt`.
- Limitation: cross-platform equality was not re-tested on a second machine here. If a platform produces
  different *counts*, that is a real failure and must not be absorbed by tolerances.

## 6. Planet-vs-binary screening (offline, `scripts/validate_vetting.py`)

Blind screening of 19 known planet hosts and 3 known eclipsing binaries (`data/vetting_results.json`):
planets → 15 "Planet candidate", **3 "Eclipsing binary"** (Kepler-10, Kepler-25, Kepler-69), 1 inconclusive
(Kepler-22); all 3 binaries flagged correctly. The three misflagged planet hosts are left as they are, with
no exceptions or label-tuning; for example, Kepler-10 trips the secondary-eclipse test. This is a
selected-systems check, not a classifier benchmark. "Possible signal" (BLS above threshold),
"planet candidate" (passed screening), "catalog classification" (NASA archive label) and "confirmed planet"
(not something Zathura establishes) stay distinct.

## 7. Saved runs, sources and audit evidence

- `data/events.jsonl`: every event of every run, keyed by `loop_id`. `data/experiments.jsonl` holds experiment
  records (proposal, review, status, measurements, memory refs). `data/holdout.jsonl` holds final-test results
  tagged with `loop_id`. `data/runs/*.json` holds engine runs with provenance. All of these are local and gitignored,
  because raw logs can contain model text and service responses.
- Sources the scientist cited are stored in each hypothesis event (`cited_sources`); mock sources are flagged `mock`.
- The UI's **Technical details** page shows the recorded diagnostics and audit records.
- Results survive app restarts because they are read from these files.
- Sanitized evidence in the repo: `docs/evidence/reproduction-clean-env.txt` (two clean-env reproduction runs) and
  `docs/evidence/mock-run-sample.json` (one **recorded mock** run: configs, reviews, sources, scores, memory reads/writes,
  final test). This is saved playback of a mock run, not a live AI session.

### Reopening saved results
Start the app again with the same `data/` directory. Improve, Results and Technical details read the latest
run from `data/events.jsonl`, and earlier runs remain in the files.

### Recovering from an interrupted run
If the app stops mid-run, the run has events but no `loop_complete`/`error` event. The UI marks it as
interrupted and shows no final-test result for it. To continue, start a new run; earlier runs remain
readable. Nothing needs deleting. To start completely fresh, move the five files/dirs above aside.

## 8. Known limitations and blockers

- One bounded live run has been verified (§10). It found no statistically significant improvement, and memory was
  read but did not demonstrably change a decision in that run.
- The scientist's hypothesis **text** is rewritten from a template for the parameter the LLM changed
  (`_ground_prose` in `zathura/agents/scientist.py`), so the prose cannot claim mechanisms for parameters it did not change.
  The LLM chooses the configuration; the wording shown is not free LLM prose.
- The candidate config is hand-supplied (see §4), and the holdout split is no longer untouched.
- One Kepler quarter per star, so periods are limited to 1–10 days; the trapezoid injection model means
  absolute recovery rates are specific to this benchmark.
- Acceptance in the research loop is point-estimate based (`improved_vs_best`), unchanged from `main`.
  Significance (`significant_vs_best`, 95% CI on the paired difference) is recorded for every experiment,
  and the Improve page now labels a kept change that is within noise as "not statistically significant".
  Example from a recorded mock run: +0.3 points dev balanced accuracy, CI −2.5 to +3.7 points, kept but not significant.
- Technical details → Mission Control still has its own launch button (default 4 rounds; Improve uses 3).
  Both buttons use the same single worker, so a second launch is refused while a run is active.
- Modal parallelization: not implemented (deferred).

## 9. UI states verified (browser, mock mode, desktop 1672×941 and phone 390×844)

- Find opens on cached real Kepler data, with no research launched and no paid calls. The screening verdict reads
  "Planet candidate" (previously "Planet") with the "screening only, not confirmation" note.
- The top bar shows **DEMO MODE · MOCK AI** when `ZATHURA_MOCK=1` or no OpenAI key is set, and **LIVE** otherwise.
- Start AI research launches one run. The button is disabled while the run is active. Navigating away and back,
  or reloading, does not start another run. A duplicate launch shows "Research is already running".
- A run killed mid-way (simulated by stopping the process) shows as interrupted on Improve, and Results says it has no final exam.
- Results only shows the held-out result whose `loop_id` matches the latest run.
- Narrow screens: the sidebar now collapses (`initial_sidebar_state="auto"` plus a CSS breakpoint at 768 px)
  instead of covering the page.

## 10. Verified live autonomous run (2026-11-14, loop `8e862fba…`)

Services: OpenAI `gpt-4o-mini` (scientist and skeptic), Exa search, past.dev shared memory. Keys were supplied for that
session only and appear in no file. Before the run, `scripts/check_memory.py` passed: both cross-agent recalls came from
past.dev, not the local mirror. Sanitized record: `docs/evidence/live-run-2026-11-14.json`.

| Step | What happened |
|---|---|
| Start | Baseline config, dev balanced accuracy 0.459 |
| Exa | Query `Kepler box least squares … detrending period grid`; 3 real sources (Wōtan detrending paper, TLS docs, TLS talk), cited in every round |
| Round 1 | LLM proposed `outlier_sigma` 5.0 → 3.0; skeptic approved; dev 0.4625 (+0.3 pts, 95% CI −3.1 to +4.1); **kept** (point increase, not significant) |
| Round 2 | Proposed `period_grid_size` 2000 → 20000; skeptic **rejected** (estimated ~61 s over the 30 s budget); not executed, no metrics recorded |
| Round 3 | Proposed `period_grid_size` 2000 → 5000 on top of round 1; approved; dev 0.453 (−0.9 pts vs best); **not kept** |
| Memory | Every round read prior memories (`memory_refs`) and wrote its result to past.dev. Influence recorded as "No evidenced memory-driven change" in all rounds: memory was read but did not change a decision |
| Final test | Selected config (`outlier_sigma` 3.0), run once on holdout after selection: 0.391 vs baseline 0.384 (+0.6 pts, CI 0.0 to +1.6, **not significant**) |

Conclusion: the live loop worked end to end and reported honestly, but it found no real improvement in 3 rounds.
The hand-supplied candidate (§4) is much stronger, but it was not produced by the AI.
