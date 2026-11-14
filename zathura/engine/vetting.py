"""Transparent single-band screening, not planet confirmation or a calibrated classifier.

Blind BLS peak: weak detection -> inconclusive; odd/even >=3 sigma, secondary
>=5 sigma or radius >2 Rj -> EB; otherwise all five checks must pass for a
planet candidate. Shape alone or missing evidence -> inconclusive.
Catalog classifications and ephemerides are never passed to the detector.
"""
import hashlib
import json

import numpy as np
from astropy.constants import R_jup, R_sun

from zathura.engine.data import ROOT, cache_path, load_catalog, read_curve
from zathura.engine.detect import search
from zathura.schema import BASELINE_CONFIG, DetectionConfig

CATALOG = ROOT / "data/vetting_catalog.json"
VERDICTS = ("Planet candidate", "Eclipsing binary", "Inconclusive / noise")


def vetting_targets():
    if CATALOG.exists():
        return json.loads(CATALOG.read_text())["targets"]
    return []


def _depths(time, flux, period, epoch, duration, offset=0):
    cycles = np.rint((time - epoch - offset) / period).astype(int)
    distance = time - epoch - offset - cycles * period
    depths, errors, indices = [], [], []
    scatter = max(1.4826 * np.median(np.abs(flux - np.median(flux))), 1e-10)
    for cycle in np.unique(cycles):
        inside = (cycles == cycle) & (np.abs(distance) < duration / 2)
        outside = (cycles == cycle) & (np.abs(distance) > duration) & (np.abs(distance) < 2 * duration)
        if inside.sum() >= 3 and outside.sum() >= 5:
            depths.append(float(np.mean(flux[outside]) - np.mean(flux[inside])))
            errors.append(scatter * np.sqrt(1 / inside.sum() + 1 / outside.sum()))
            indices.append(cycle)
    return np.asarray(depths), np.asarray(errors), np.asarray(indices)


def _estimate(depths, errors):
    if len(depths) < 2:
        return None, None
    error = max(float(np.std(depths, ddof=1) / np.sqrt(len(depths))),
                float(np.sqrt(np.sum(errors ** 2)) / len(depths)), 1e-12)
    return float(np.mean(depths)), error


def _test(name, value, threshold, passed, explanation, unit=""):
    return {"name": name, "value": None if value is None else float(value), "threshold": threshold,
            "passed": None if passed is None else bool(passed), "explanation": explanation, "unit": unit}


def vet_candidate(target_id, config=None):
    """Screen the blind 1–10 day BLS peak of a cached real Kepler target.

    Errors on depths use the larger of per-event scatter and a point-scatter
    floor; neither accounts fully for red noise. Radius assumes an opaque,
    undiluted transit. A phase-0.5 secondary misses eccentric binaries.
    """
    config = DetectionConfig.model_validate(config or BASELINE_CONFIG)
    try:
        kepid = int(str(target_id).upper().removeprefix("KIC "))
    except ValueError as exc:
        raise ValueError("Use a cached KIC target identifier") from exc
    metadata = next((t for t in vetting_targets() if t["kepid"] == kepid), {})
    catalog = load_catalog()
    allowed = {*catalog["hosts"]["dev"], *catalog["hosts"]["holdout"],
               *(p["kepid"] for p in catalog["validation"])}
    if not metadata and kepid not in allowed:
        raise ValueError(f"Unknown vetting target: {target_id}")
    curve = read_curve(kepid)
    if curve["origin"] != "real":
        raise ValueError("Vetting requires real source photometry")
    t, flux = np.asarray(curve["time"], float), np.asarray(curve["flux"], float)
    if np.isfinite(t * flux).sum() < 20 or np.ptp(t[np.isfinite(t)]) <= 2:
        raise ValueError("Insufficient finite light-curve samples for vetting")
    result = search(t, flux, config)
    t, flat = result["time"], result["flat_flux"]
    period, epoch, duration = result["period_days"], result["epoch"], result["duration_days"]
    phase_days = (t - epoch + period / 2) % period - period / 2
    depths, errors, cycles = _depths(t, flat, period, epoch, duration)
    odd, odd_err = _estimate(depths[cycles % 2 == 1], errors[cycles % 2 == 1])
    even, even_err = _estimate(depths[cycles % 2 == 0], errors[cycles % 2 == 0])
    odd_sigma = abs(odd - even) / np.hypot(odd_err, even_err) if odd is not None and even is not None else None
    secondary, secondary_err = _estimate(*_depths(t, flat, period, epoch, duration, period / 2)[:2])
    secondary_sigma = max(0., secondary / secondary_err) if secondary is not None else None
    inside, core = np.abs(phase_days) < duration / 2, np.abs(phase_days) < duration / 6
    background = float(np.median(flat[~inside]))
    mean_depth = background - float(np.mean(flat[inside]))
    ratio = ((background - float(np.mean(flat[core]))) / mean_depth
             if inside.sum() >= 10 and core.sum() >= 5 and mean_depth > 0 else None)
    radius = metadata.get("stellar_radius_solar")
    assumed = radius is None or not np.isfinite(radius) or radius <= 0
    radius = 1. if assumed else float(radius)
    companion = np.sqrt(max(result["depth"], 0)) * radius * (R_sun / R_jup).value
    tests = [
        _test("Detection strength", result["sde"], f"SDE ≥ {config.detection_threshold:g}", result["detected"],
              "The strongest periodic dip must stand above the background of trial periods.", "SDE"),
        _test("Odd / even depths", odd_sigma, "difference < 3 sigma", None if odd_sigma is None else odd_sigma < 3,
              "Alternating eclipse depths suggest two stars; at least two events in each group are required.", "sigma"),
        _test("Secondary eclipse", secondary_sigma, "dip at phase 0.5 < 5 sigma", None if secondary_sigma is None else secondary_sigma < 5,
              "A significant dip halfway around the orbit suggests a luminous companion.", "sigma"),
        _test("Transit shape", ratio, "central / mean depth ≤ 1.6", None if ratio is None else 0 < ratio <= 1.6,
              "A strongly peaked V-shape is suspicious, but shape alone cannot establish a binary.", "ratio"),
        _test("Implied size", companion, "radius ≤ 2 Jupiter radii", companion <= 2,
              "Depth implies size only for an opaque, undiluted transit; " +
              ("one solar stellar radius is assumed, not measured." if assumed else "stellar radius is from the NASA catalog."), "Rj"),
    ]
    if not result["detected"]:
        verdict = VERDICTS[2]
    elif any(tests[i]["passed"] is False for i in (1, 2, 4)):
        verdict = VERDICTS[1]
    elif all(test["passed"] is True for test in tests):
        verdict = VERDICTS[0]
    else:
        verdict = VERDICTS[2]
    flags = [test["name"] for test in tests if test["passed"] is False]
    return {"target_id": f"KIC {kepid}", "verdict": verdict, "data_source": "real", "tests": tests,
            "summary": ("Screening flags: " + ", ".join(flags) if flags else "No decisive binary flag in this quarter.") +
                       " Heuristic screening, not confirmation or a calibrated probability.",
            "period_days": period, "depth_ppm": result["depth"] * 1e6, "duration_days": duration,
            "stellar_radius_solar": radius, "radius_assumed": assumed,
            "odd_depth_ppm": None if odd is None else odd * 1e6,
            "even_depth_ppm": None if even is None else even * 1e6,
            "secondary_depth_ppm": None if secondary is None else secondary * 1e6,
            "cadences": len(t), "span_days": float(np.ptp(t)), "quarter": int(curve["quarter"]),
            "plots": {"time": t, "flux": np.interp(t, curve["time"], curve["flux"]), "flat_flux": flat,
                      "phase_days": phase_days, "phase": ((t - epoch) / period + .25) % 1 - .25,
                      "odd": (np.rint((t - epoch) / period).astype(int) % 2).astype(bool)}}


def validate_vetting():
    rows = []
    confusion = {truth: {verdict: 0 for verdict in VERDICTS} for truth in ("planet", "EB")}
    for target in vetting_targets():
        result = vet_candidate(f"KIC {target['kepid']}")
        confusion[target["truth"]][result["verdict"]] += 1
        rows.append({k: v for k, v in target.items()} | {"verdict": result["verdict"], "tests": result["tests"],
                     "found_period_days": result["period_days"],
                     "sha256": hashlib.sha256(cache_path(target["kepid"]).read_bytes()).hexdigest()})
    return {"config": BASELINE_CONFIG.model_dump(), "confusion": confusion, "targets": rows,
            "description": "Blind peak screening on selected known hosts and known binaries, not a held-out classifier benchmark."}
