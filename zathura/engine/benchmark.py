"""Seeded injection-recovery benchmark on real Kepler curves. Truth never leaves the engine."""
import hashlib
import json
from functools import lru_cache

import numpy as np

from zathura.engine.data import load_catalog, read_curve, transit_noise

BENCHMARK_VERSION = "kepler-injection-v4"
SEEDS = {"dev": 42, "holdout": 9173}
INJECTIONS_PER_HOST = 4
CONTROLS_PER_HOST = 1
SNR_RANGE = (5.0, 25.0)
SNR_BINS = ((5, 8), (8, 12), (12, 18), (18, 25))
MIN_TRANSITS = 3
PERIOD_RANGE = (1.0, 10.0)
PERIOD_TOLERANCE = 0.01
SOLAR_DURATION_AT_1YR_DAYS = 13 / 24


def transit_model(time, period, epoch, duration, depth, ingress_fraction):
    """Symmetric trapezoid transit, a simple stand-in for limb-darkened ingress/egress."""
    offset = np.abs((time - epoch + period / 2) % period - period / 2)
    half, ingress = duration / 2, max(duration * ingress_fraction, 1e-6)
    shape = np.clip((half - offset) / ingress, 0, 1)
    return 1 - depth * shape


def _transit_count(time, period, epoch, duration):
    centers = np.arange(np.floor((time.min() - epoch) / period), np.ceil((time.max() - epoch) / period) + 1)
    count = 0
    for n in centers:
        in_window = np.abs(time - (epoch + n * period)) < duration / 2
        if in_window.sum() >= 0.5 * duration / np.median(np.diff(time)):
            count += 1
    return count


def _inject(rng, time, flux):
    for _ in range(50):
        period = float(np.exp(rng.uniform(*np.log(PERIOD_RANGE))))
        impact = rng.uniform(0, 0.9)
        duration = float(SOLAR_DURATION_AT_1YR_DAYS * (period / 365.25) ** (1 / 3) * np.sqrt(1 - impact ** 2))
        epoch = float(time.min() + rng.uniform(0, period))
        transits = _transit_count(time, period, epoch, duration)
        if transits >= MIN_TRANSITS:
            break
    else:
        raise RuntimeError("Could not place an injection with enough observed transits")
    snr = float(np.exp(rng.uniform(*np.log(SNR_RANGE))))
    # SNR = depth / (noise at the transit timescale / sqrt(number of transits)).
    depth = float(snr * transit_noise(time, flux, duration) / np.sqrt(transits))
    model = transit_model(time, period, epoch, duration, depth, 0.1 + 0.3 * impact)
    truth = {"period": period, "epoch": epoch, "duration": duration, "depth": depth, "snr": snr,
             "impact": impact, "transits": transits}
    return flux * model, truth


@lru_cache(maxsize=2)
def build(dataset_id):
    """Return (examples, truth) for a split; examples carry only arrays and an opaque index."""
    if dataset_id not in SEEDS:
        raise ValueError("dataset_id must be 'dev' or 'holdout'")
    catalog = load_catalog()
    rng = np.random.default_rng(SEEDS[dataset_id])
    examples, truth = [], []
    for host_index, kepid in enumerate(catalog["hosts"][dataset_id]):
        curve = read_curve(kepid)
        time, base = curve["time"], curve["flux"].astype(float)
        for kind in ["injection"] * INJECTIONS_PER_HOST + ["control"] * CONTROLS_PER_HOST:
            # Independent noise draws are not added: real photometric noise is the noise model.
            if kind == "injection":
                flux, info = _inject(rng, time, base)
            else:
                flux, info = base.copy(), {}
            examples.append((time, flux))
            truth.append({"host": host_index, "kepid": int(kepid), "injected": kind == "injection", **info})
    return examples, truth


TRUTH_DIGITS = 9


def canonical_truth(truth):
    """Truth rows with floats rounded to TRUTH_DIGITS significant digits.

    Injection parameters come from exp/log and median filters whose last bits can differ across
    CPUs and NumPy builds; hashing full float reprs made identical benchmarks look different.
    Integers, booleans and star assignments stay exact.
    """
    def value(v):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        return float(f"{float(v):.{TRUTH_DIGITS}g}")
    return [{k: value(v) for k, v in row.items()} for row in truth]


def fingerprint(dataset_id):
    """Platform-stable SHA-256 of the private truth table, recorded so runs can prove identical benchmarks."""
    _, truth = build(dataset_id)
    blob = json.dumps(canonical_truth(truth), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def raw_fingerprint(dataset_id):
    """Legacy full-precision hash (pre-canonicalization); platform dependent, kept for audit only."""
    _, truth = build(dataset_id)
    blob = json.dumps(truth, sort_keys=True, default=float).encode()
    return hashlib.sha256(blob).hexdigest()


def score(result, truth):
    """Outcome for one example; recovery needs period and epoch to match the injection."""
    if not truth["injected"]:
        return {"detected": result["detected"], "recovered": False, "harmonic": False}
    ratio = result["period_days"] / truth["period"]
    period_ok = abs(ratio - 1) < PERIOD_TOLERANCE
    phase = (result["epoch"] - truth["epoch"] + truth["period"] / 2) % truth["period"] - truth["period"] / 2
    epoch_ok = abs(phase) < truth["duration"]
    harmonic = any(abs(ratio / h - 1) < PERIOD_TOLERANCE for h in (0.5, 2.0, 1 / 3, 3.0))
    return {"detected": result["detected"], "recovered": bool(result["detected"] and period_ok and epoch_ok),
            "harmonic": bool(result["detected"] and harmonic)}


def balanced_accuracy(recovered, false_alarm, injected, hosts=None, host_weights=None):
    weights = np.ones(len(injected)) if host_weights is None else host_weights[hosts]
    pos, neg = injected, ~injected
    recovery = np.sum(weights[pos] * recovered[pos]) / max(np.sum(weights[pos]), 1e-12)
    far = np.sum(weights[neg] * false_alarm[neg]) / max(np.sum(weights[neg]), 1e-12)
    return (recovery + 1 - far) / 2


def bootstrap_weights(n_hosts, resamples=2000, seed=0):
    """Host-level resampling weights; examples on one star are correlated, so stars are the unit."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_hosts, size=(resamples, n_hosts))
    return np.stack([np.bincount(d, minlength=n_hosts) for d in draws]).astype(float)
