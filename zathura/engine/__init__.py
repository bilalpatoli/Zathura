"""Stable engine API. Injection labels never cross this boundary.

All benchmark results are placeholder, even when the underlying curve is real.
The evaluator is a software boundary, not a security sandbox for Python callers.
"""
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path
from uuid import uuid4

import numpy as np
from astropy.timeseries import BoxLeastSquares
from lightkurve import LightCurve

from zathura.schema import BASELINE_CONFIG, DetectionConfig

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
RUNS = ROOT / "data" / "runs"
_TARGETS = ["Kepler-69", "Kepler-10", "Kepler-22", "KIC 8006161", "KIC 6106415", "KIC 6603624"]


def _cache_path(target_id):
    if target_id not in _TARGETS:
        raise ValueError(f"Unknown target: {target_id}")
    return CACHE / (target_id.replace(" ", "_") + ".npz")


def load_targets():
    """Download once, or cache explicitly synthetic fallback curves on failure.

    Remove a fallback .npz to retry that target's download later.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    targets = []
    for index, target_id in enumerate(_TARGETS):
        path = _cache_path(target_id)
        if not path.exists():
            try:
                subprocess.run(
                    [sys.executable, "-m", "zathura.engine._download", target_id, str(path)],
                    check=True, capture_output=True, text=True, timeout=25, cwd=ROOT,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                detail = getattr(exc, "stderr", "") or str(exc)
                warnings.warn(f"MAST download failed for {target_id}: {detail[-600:]}; using synthetic placeholder")
                rng = np.random.default_rng(100 + index)
                t = np.arange(0, 28, 0.0204)
                f = 1 + 0.001 * np.sin(t / 3) + rng.normal(0, 0.0005, len(t))
                np.savez_compressed(path, time=t, flux=f, origin="synthetic_placeholder")
        with np.load(path) as cached:
            targets.append({"target_id": target_id, "curve_origin": str(cached["origin"]), "data_source": "placeholder"})
    return targets


def get_lightcurve(target_id):
    """Return the un-injected source curve, never benchmark labels."""
    path = _cache_path(target_id)
    if not path.exists():
        load_targets()
    with np.load(path) as cached:
        return {"target_id": target_id, "time": cached["time"].copy(), "flux": cached["flux"].copy(),
                "curve_origin": str(cached["origin"]), "data_source": "placeholder"}


def _dataset(dataset_id):
    if dataset_id not in ("dev", "holdout"):
        raise ValueError("dataset_id must be 'dev' or 'holdout'")
    rng = np.random.default_rng(42 if dataset_id == "dev" else 9173)
    for i, target in enumerate(_TARGETS):
        curve = get_lightcurve(target)
        t, f = curve["time"], curve["flux"]
        valid = np.isfinite(t) & np.isfinite(f)
        t, f = t[valid], f[valid]
        order = np.argsort(t)
        t, f = t[order], f[order]
        t = t - t.min()
        select = t < 28
        t, f = t[select][::max(1, len(t[select]) // 1800)], f[select][::max(1, len(t[select]) // 1800)]
        f = f / np.median(f)
        injected = i % 2 == 0
        period, epoch, duration = rng.uniform(2, 8), rng.uniform(0, 2), rng.uniform(0.08, 0.18)
        if injected:
            in_transit = np.abs((t - epoch + period / 2) % period - period / 2) < duration / 2
            f[in_transit] -= rng.uniform(0.0008, 0.002)
        yield t, f, injected, period


def _read_run(run_id):
    if not isinstance(run_id, str) or len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Invalid run_id")
    return json.loads((RUNS / f"{run_id}.json").read_text())


def run_experiment(config, dataset_id):
    config = DetectionConfig.model_validate(config)
    if dataset_id not in ("dev", "holdout"):
        raise ValueError("dataset_id must be 'dev' or 'holdout'")
    started = time.perf_counter()
    recovered = false_alarms = positives = negatives = 0
    for t, flux, injected, true_period in _dataset(dataset_id):
        window = min(config.detrend_window, len(flux) // 2 * 2 - 1)
        flat = np.asarray(LightCurve(time=t, flux=flux).flatten(window_length=window).flux.value)
        scale = max(1.4826 * np.median(np.abs(flat - np.median(flat))), 1e-10)
        # Keep negative excursions: these contain the transit signal.
        keep = flat < np.median(flat) + config.outlier_sigma * scale
        bls = BoxLeastSquares(t[keep], flat[keep])
        result = bls.power(np.linspace(1, 10, config.period_grid_size),
                           np.linspace(config.duration_min_days, config.duration_max_days, 4))
        best = int(np.argmax(result.power))
        sde = (result.power[best] - np.mean(result.power)) / max(np.std(result.power), 1e-12)
        detected = sde >= config.detection_threshold
        if injected:
            positives += 1
            recovered += int(detected and abs(result.period[best] / true_period - 1) < 0.03)
        else:
            negatives += 1
            false_alarms += int(detected)
    runtime = time.perf_counter() - started
    recovery, far = recovered / positives, false_alarms / negatives
    measurements = {"recovery_rate": recovery, "false_alarm_rate": far,
                    "balanced_accuracy": (recovery + 1 - far) / 2, "runtime_s": runtime}
    run_id = uuid4().hex
    RUNS.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "dataset_id": dataset_id, "config": config.model_dump(),
              "measurements": measurements, "data_source": "placeholder", "is_baseline": False}
    (RUNS / f"{run_id}.json").write_text(json.dumps(record))
    return run_id


def run_baseline(dataset_id):
    run_id = run_experiment(BASELINE_CONFIG, dataset_id)
    record = _read_run(run_id)
    record["is_baseline"] = True
    (RUNS / f"{run_id}.json").write_text(json.dumps(record))
    return run_id


def evaluate_experiment(run_id):
    record = _read_run(run_id)
    peers = [json.loads(p.read_text()) for p in RUNS.glob("*.json")]
    peers = [p for p in peers if p["dataset_id"] == record["dataset_id"] and p["run_id"] != run_id]
    baselines = [p for p in peers if p["is_baseline"]]
    if not baselines and not record["is_baseline"]:
        raise ValueError("Run a baseline for this dataset before evaluating")
    score = record["measurements"]["balanced_accuracy"]
    baseline = baselines[0]["measurements"]["balanced_accuracy"] if baselines else score
    best = max([p["measurements"]["balanced_accuracy"] for p in peers], default=score)
    evaluation = {"improved_vs_baseline": score > baseline, "improved_vs_best": score > best,
                  "delta": score - baseline}
    record["evaluation_result"] = evaluation
    (RUNS / f"{run_id}.json").write_text(json.dumps(record))
    return evaluation


def get_experiment_results(run_id):
    """Aggregate metrics only; no per-target predictions or injection answers."""
    return _read_run(run_id)
