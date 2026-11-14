"""Stable engine API. Injection truth never crosses this boundary.

Benchmarks inject synthetic transits into real Kepler PDCSAP light curves of stars
with no known KOI/TCE; dev and holdout use disjoint stars. The evaluator is a
software boundary, not a security sandbox for Python callers.
"""
import hashlib
import json
import os
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from uuid import uuid4

import numpy as np

from zathura.engine import benchmark
from zathura.engine.benchmark import BENCHMARK_VERSION, SNR_BINS
from zathura.engine.data import CACHE, CATALOG_PATH, ROOT, cache_path, load_catalog, read_curve
from zathura.engine.detect import search
from zathura.schema import BASELINE_CONFIG, DetectionConfig

RUNS = ROOT / "data" / "runs"
DATA_SOURCE = "real"
BOOTSTRAP_RESAMPLES = 2000
BENCHMARK_DESCRIPTION = (
    "Injection-recovery on one real ~90-day Kepler long-cadence PDCSAP quarter per star. "
    "Dev and holdout use disjoint sets of 40 quiet stars (no KOI or TCE). Each star yields 4 injected "
    "trapezoid transits (period 1-10 d log-uniform, solar-density durations, SNR 5-25 against the star's noise at the transit timescale) and 1 "
    "un-injected control: 160 positives and 40 negatives per split. Same examples for every config. "
    "BLS over 1-10 d on a frequency-uniform grid with 6 log-spaced durations; SDE threshold; upper-only "
    "outlier clipping; Lightkurve flatten (window in 29.4-min cadences). Recovery needs an above-threshold "
    "peak within 1% of the injected period and an epoch within one transit duration. Balanced accuracy "
    "averages recovery and specificity; confidence intervals bootstrap over stars. Per-SNR recovery is "
    "reported. Holdout runs only after tuning."
)


def _target_kepid(target_id):
    text = str(target_id).strip()
    kepid = int(text[4:]) if text.upper().startswith("KIC ") else None
    catalog = load_catalog()
    known = {*catalog["hosts"]["dev"], *catalog["hosts"]["holdout"], *(v["kepid"] for v in catalog["validation"])}
    if kepid is None or kepid not in known:
        raise ValueError(f"Unknown target: {target_id}")
    return kepid


def _targets(require_cache):
    catalog = load_catalog()
    validation = {v["kepid"]: v for v in catalog["validation"]}
    rows = [(k, "validation") for k in validation] + [(k, s) for s in ("dev", "holdout") for k in catalog["hosts"][s]]
    targets = []
    for kepid, role in rows:
        path = cache_path(kepid)
        if require_cache and not path.exists():
            continue
        row = {"target_id": f"KIC {kepid}", "role": role, "curve_origin": "real", "data_source": DATA_SOURCE}
        if role == "validation":
            v = validation[kepid]
            row |= {"name": v["name"] or v["koi"], "known_periods": v["host_koi_periods"]}
        targets.append(row)
    return targets


def load_targets():
    """Catalog stars and roles. Curves are committed; rebuild with scripts/build_catalog.py."""
    targets = _targets(require_cache=False)
    missing = [t["target_id"] for t in targets if not cache_path(_target_kepid(t["target_id"])).exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached curves for {missing[:5]}; run scripts/build_catalog.py")
    return targets


def cached_targets():
    """Read cached target metadata without triggering downloads."""
    try:
        return _targets(require_cache=True)
    except FileNotFoundError:
        return []


def get_lightcurve(target_id):
    """Return the un-injected source curve, never benchmark labels."""
    curve = read_curve(_target_kepid(target_id))
    return {"target_id": f"KIC {curve['kepid']}", "time": curve["time"], "flux": curve["flux"].astype(float),
            "quarter": int(curve["quarter"]), "product": curve["product"], "curve_origin": curve["origin"],
            "data_source": DATA_SOURCE}


def _analyze(args):
    t, flux, config = args
    started = time.perf_counter()
    result = search(t, flux, DetectionConfig.model_validate(config))
    keys = ("period_days", "epoch", "duration_days", "depth", "sde", "detected")
    return {k: result[k] for k in keys} | {"elapsed_s": time.perf_counter() - started}


@lru_cache(maxsize=1)
def _workers():
    return max(1, int(os.getenv("ZATHURA_WORKERS", min(8, os.cpu_count() or 1))))


def _map(tasks):
    # Threads, not processes: astropy's BLS kernel releases the GIL, and threads need no
    # __main__ re-import, so this is safe inside Streamlit and unguarded agent scripts.
    workers = _workers()
    if workers == 1:
        return [_analyze(t) for t in tasks]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(_analyze, tasks))


def _sha256_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _data_provenance(dataset_id):
    hosts = load_catalog()["hosts"][dataset_id]
    return {"catalog_version": load_catalog()["version"], "catalog_sha256": _sha256_files([CATALOG_PATH]),
            "curves_sha256": _sha256_files([cache_path(k) for k in hosts]),
            "truth_sha256": benchmark.fingerprint(dataset_id), "seed": benchmark.SEEDS[dataset_id],
            "host_count": len(hosts)}


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _provenance(dataset_id):
    packages = {}
    for name in ("numpy", "scipy", "astropy", "lightkurve", "pydantic"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"code_commit": _git("rev-parse", "HEAD") or None,
            "code_dirty": bool(_git("status", "--porcelain", "--", "zathura")),
            "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "workers": _workers(), **_data_provenance(dataset_id)}


def _metrics(outcomes, truth, wall_s):
    injected = np.array([t["injected"] for t in truth])
    detected = np.array([o["detected"] for o in outcomes])
    recovered = np.array([o["recovered"] for o in outcomes])
    harmonic = np.array([o["harmonic"] for o in outcomes])
    hosts = np.array([t["host"] for t in truth])
    recovery, far = recovered[injected].mean(), detected[~injected].mean()
    weights = benchmark.bootstrap_weights(hosts.max() + 1, BOOTSTRAP_RESAMPLES)
    samples = _balanced_samples(recovered, detected, injected, hosts, weights)
    measurements = {"recovery_rate": float(recovery), "false_alarm_rate": float(far),
                    "balanced_accuracy": float((recovery + 1 - far) / 2),
                    "balanced_accuracy_ci_low": float(np.percentile(samples, 2.5)),
                    "balanced_accuracy_ci_high": float(np.percentile(samples, 97.5)),
                    "harmonic_rate": float(harmonic[injected].mean()),
                    "runtime_s": float(wall_s),
                    "runtime_per_curve_s": float(np.mean([o["elapsed_s"] for o in outcomes])),
                    "positive_count": float(injected.sum()), "negative_count": float((~injected).sum())}
    snr = np.array([t.get("snr", np.nan) for t in truth])
    for low, high in SNR_BINS:
        selected = injected & (snr >= low) & (snr < high)
        if selected.any():
            measurements[f"recovery_snr_{low}_{high}"] = float(recovered[selected].mean())
    return measurements, {"detected": detected, "recovered": recovered, "injected": injected, "hosts": hosts}


def _balanced_samples(recovered, detected, injected, hosts, weights):
    w = weights[:, hosts]
    recovery = (w[:, injected] * recovered[injected]).sum(1) / np.maximum(w[:, injected].sum(1), 1e-12)
    far = (w[:, ~injected] * detected[~injected]).sum(1) / np.maximum(w[:, ~injected].sum(1), 1e-12)
    return (recovery + 1 - far) / 2


def _private_path(run_id):
    return RUNS / "private" / f"{run_id}.npz"


def _read_run(run_id):
    if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Invalid run_id")
    return json.loads((RUNS / f"{run_id}.json").read_text())


def _write_run(record):
    (RUNS / f"{record['run_id']}.json").write_text(json.dumps(record))


def run_experiment(config, dataset_id, is_baseline=False):
    config = DetectionConfig.model_validate(config)
    if dataset_id not in ("dev", "holdout"):
        raise ValueError("dataset_id must be 'dev' or 'holdout'")
    examples, truth = benchmark.build(dataset_id)
    started = time.perf_counter()
    results = _map([(t, f, config.model_dump()) for t, f in examples])
    wall = time.perf_counter() - started
    outcomes = [r | benchmark.score(r, info) for r, info in zip(results, truth)]
    measurements, private = _metrics(outcomes, truth, wall)
    run_id = uuid4().hex
    (RUNS / "private").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_private_path(run_id), **private)
    _write_run({"run_id": run_id, "dataset_id": dataset_id, "config": config.model_dump(),
                "measurements": measurements, "data_source": DATA_SOURCE, "is_baseline": is_baseline,
                "benchmark_version": BENCHMARK_VERSION, "timestamp": datetime.now(timezone.utc).isoformat(),
                "provenance": _provenance(dataset_id)})
    return run_id


def run_baseline(dataset_id):
    return run_experiment(BASELINE_CONFIG, dataset_id, is_baseline=True)


def _paired_delta(run_id, other_id, dataset_id):
    """Bootstrap CI for balanced-accuracy difference on identical examples, resampling stars."""
    try:
        with np.load(_private_path(run_id)) as a, np.load(_private_path(other_id)) as b:
            a, b = dict(a), dict(b)
    except FileNotFoundError:
        return None
    hosts = a["hosts"]
    weights = benchmark.bootstrap_weights(hosts.max() + 1, BOOTSTRAP_RESAMPLES, seed=1)
    delta = (_balanced_samples(a["recovered"], a["detected"], a["injected"], hosts, weights)
             - _balanced_samples(b["recovered"], b["detected"], b["injected"], hosts, weights))
    return float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))


def evaluate_experiment(run_id):
    """Compare with the latest baseline and the best run since it on the same benchmark.

    improved_* are strict point-estimate comparisons; significant_* additionally
    require the 95% paired bootstrap interval of the difference to exclude zero.
    """
    record = _read_run(run_id)
    peers = [json.loads(p.read_text()) for p in RUNS.glob("*.json")]
    truth = record.get("provenance", {}).get("truth_sha256")
    peers = [p for p in peers if p["dataset_id"] == record["dataset_id"] and p["run_id"] != run_id
             and p.get("benchmark_version") == record.get("benchmark_version")
             and p.get("provenance", {}).get("truth_sha256") == truth
             and p.get("timestamp", "") <= record.get("timestamp", "")]
    baselines = [p for p in peers if p["is_baseline"]]
    if not baselines and not record["is_baseline"]:
        raise ValueError("Run a baseline for this dataset before evaluating")
    score = record["measurements"]["balanced_accuracy"]
    latest = max(baselines, key=lambda p: p.get("timestamp", "")) if baselines else record
    peers = [p for p in peers if p.get("timestamp", "") >= latest.get("timestamp", "")]
    baseline = latest["measurements"]["balanced_accuracy"]
    best_run = max(peers, key=lambda p: p["measurements"]["balanced_accuracy"], default=record)
    best = best_run["measurements"]["balanced_accuracy"]
    evaluation = {"improved_vs_baseline": score > baseline, "improved_vs_best": score > best,
                  "delta": score - baseline, "delta_vs_best": score - best,
                  "baseline_run_id": latest["run_id"], "best_run_id": best_run["run_id"]}
    for label, other in (("baseline", latest), ("best", best_run)):
        ci = _paired_delta(run_id, other["run_id"], record["dataset_id"]) if other is not record else (0.0, 0.0)
        if ci is not None:
            evaluation[f"delta_vs_{label}_ci_low"], evaluation[f"delta_vs_{label}_ci_high"] = ci
            evaluation[f"significant_vs_{label}"] = ci[0] > 0
    record["evaluation_result"] = evaluation
    _write_run(record)
    return evaluation


def get_experiment_results(run_id):
    """Aggregate metrics and provenance only; no per-example predictions or injection answers."""
    return _read_run(run_id)


def _period_match(found, periods, tolerance=0.01):
    exact = any(abs(found / p - 1) < tolerance for p in periods)
    harmonic = any(abs(found / (p * h) - 1) < tolerance for p in periods for h in (0.5, 2.0, 1 / 3, 3.0))
    return exact, harmonic and not exact


def validate_known_planets(config=None):
    """Blind BLS search on real stars with confirmed 1-10 day planets; public catalog answers."""
    config = DetectionConfig.model_validate(config or BASELINE_CONFIG)
    planets = load_catalog()["validation"]
    tasks = []
    for planet in planets:
        curve = read_curve(planet["kepid"])
        tasks.append((curve["time"], curve["flux"].astype(float), config.model_dump()))
    rows = []
    for planet, result in zip(planets, _map(tasks)):
        exact, harmonic = _period_match(result["period_days"], planet["host_koi_periods"])
        rows.append({"target_id": f"KIC {planet['kepid']}", "name": planet["name"] or planet["koi"],
                     "catalog_period_days": planet["period_days"], "depth_ppm": planet["depth_ppm"],
                     "found_period_days": result["period_days"], "sde": result["sde"],
                     "detected": result["detected"], "recovered": bool(result["detected"] and exact),
                     "harmonic_only": bool(result["detected"] and harmonic)})
    recovered = sum(r["recovered"] for r in rows)
    return {"config": config.model_dump(), "planets": rows, "recovered": recovered, "total": len(rows),
            "recovery_rate": recovered / len(rows) if rows else float("nan"), "data_source": DATA_SOURCE,
            "catalog_version": load_catalog()["version"]}


def inspect_target(target_id, config=None):
    """Read-only diagnostic on an un-injected source, not a benchmark example."""
    config = DetectionConfig.model_validate(config or BASELINE_CONFIG)
    curve = get_lightcurve(target_id)
    valid = np.isfinite(curve["time"]) & np.isfinite(curve["flux"])
    t, flux = curve["time"][valid], curve["flux"][valid]
    if len(t) < 10 or np.ptp(t) <= 2:
        raise ValueError("Insufficient finite light-curve samples for BLS")
    order = np.argsort(t)
    t, flux = t[order], flux[order] / np.median(flux[order])
    result = search(t, flux, config)
    period, epoch, duration = result["period_days"], result["epoch"], result["duration_days"]
    phase = (result["time"] - epoch + period / 2) % period - period / 2
    known = next((v for v in load_catalog()["validation"] if f"KIC {v['kepid']}" == curve["target_id"]), None)
    return curve | result | {"time": result["time"], "flux": np.interp(result["time"], t, flux),
                             "phase": phase, "in_transit": np.abs(phase) < duration / 2,
                             "known_periods": known["host_koi_periods"] if known else None}
