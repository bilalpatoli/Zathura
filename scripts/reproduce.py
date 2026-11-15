"""Re-run the engine's headline results and check them against data/reference_results.json.

  .venv/bin/python scripts/reproduce.py                    # baseline + known-planet validation
  .venv/bin/python scripts/reproduce.py --config best.json # also a tuned config on dev and holdout
  .venv/bin/python scripts/reproduce.py --update-reference # record the current results as reference

Exits nonzero on any genuine failure, including a missing reference. Configurations, counts,
detection decisions and checksums must match exactly; floating-point statistics use a 1e-6
relative tolerance (see FLOAT_RTOL).
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zathura.engine import (  # noqa: E402
    BENCHMARK_VERSION, get_experiment_results, run_baseline, run_experiment, validate_known_planets,
)
from zathura.schema import BASELINE_CONFIG, DetectionConfig  # noqa: E402

REFERENCE = ROOT / "data" / "reference_results.json"
TIMING = ("runtime_s", "runtime_per_curve_s")


def measured(run_id):
    result = get_experiment_results(run_id)
    return {"measurements": {k: v for k, v in result["measurements"].items() if k not in TIMING},
            "runtime_s": result["measurements"]["runtime_s"],
            "truth_sha256": result["provenance"]["truth_sha256"],
            "curves_sha256": result["provenance"]["curves_sha256"], "run_id": run_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="JSON DetectionConfig to evaluate alongside the baseline")
    parser.add_argument("--update-reference", action="store_true")
    args = parser.parse_args()
    configs = {"baseline": BASELINE_CONFIG}
    if args.config:
        configs["candidate"] = DetectionConfig.model_validate_json(args.config.read_text())
    report = {"benchmark_version": BENCHMARK_VERSION, "configs": {}, "known_planets": {}}
    for label, config in configs.items():
        runs = {}
        for dataset in ("dev", "holdout"):
            run_id = run_baseline(dataset) if label == "baseline" else run_experiment(config, dataset)
            runs[dataset] = measured(run_id)
            print(f"{label:9s} {dataset:7s} " + json.dumps(runs[dataset]["measurements"]), flush=True)
        repeat = measured(run_experiment(config, "dev"))
        runs["deterministic"] = repeat["measurements"] == runs["dev"]["measurements"]
        validation = validate_known_planets(config)
        report["configs"][label] = {"config": config.model_dump(), **runs}
        report["known_planets"][label] = validation
        print(f"{label:9s} known planets recovered {validation['recovered']}/{validation['total']}; "
              f"deterministic={runs['deterministic']}", flush=True)
    (ROOT / "data" / "reproduction.json").write_text(json.dumps(report, indent=1, default=float))
    if args.update_reference:
        REFERENCE.write_text(json.dumps(report, indent=1, default=float) + "\n")
        print(f"Reference written to {REFERENCE}")
        return
    if not all(c["deterministic"] for c in report["configs"].values()):
        sys.exit("FAIL: repeated runs differ")
    if not REFERENCE.exists():
        sys.exit(f"FAIL: no reference at {REFERENCE}; nothing was compared")
    failures = compare(report, json.loads(REFERENCE.read_text()))
    if failures:
        sys.exit("FAIL: differs from reference:\n  " + "\n  ".join(failures))
    print("PASS: matched the reference (exact: configs, counts, decisions, checksums; "
          f"floats within rel {FLOAT_RTOL:g}; runtime excluded)")


# Count-derived metrics are exact. Bootstrap CIs and BLS statistics are floating-point results whose last
# bits can vary across CPUs/BLAS builds; 1e-6 relative is far below any scientifically meaningful change.
FLOAT_RTOL = 1e-6
EXACT_MEASUREMENTS = ("positive_count", "negative_count", "recovery_rate", "false_alarm_rate", "balanced_accuracy",
                      "harmonic_rate", "recovery_snr_5_8", "recovery_snr_8_12", "recovery_snr_12_18", "recovery_snr_18_25")


def close(a, b):
    return math.isclose(a, b, rel_tol=FLOAT_RTOL, abs_tol=1e-12)


def compare(report, reference):
    failures = []
    if reference.get("benchmark_version") != report["benchmark_version"]:
        failures.append(f"benchmark_version {report['benchmark_version']} != reference {reference.get('benchmark_version')}")
    for label, current in report["configs"].items():
        expected = reference["configs"].get(label)
        if expected is None:
            failures.append(f"{label}: no reference for this label")
            continue
        if expected["config"] != current["config"]:
            failures.append(f"{label}: config differs from reference")
            continue
        for dataset in ("dev", "holdout"):
            now, then = current[dataset], expected[dataset]
            for key in ("truth_sha256", "curves_sha256"):
                if now[key] != then.get(key):
                    failures.append(f"{label}/{dataset}/{key}")
            if set(now["measurements"]) != set(then["measurements"]):
                failures.append(f"{label}/{dataset}: measurement keys differ")
            for key, value in now["measurements"].items():
                ref = then["measurements"].get(key)
                ok = value == ref if key in EXACT_MEASUREMENTS else ref is not None and close(value, ref)
                if not ok:
                    failures.append(f"{label}/{dataset}/{key}: {value!r} != {ref!r}")
        planets = report["known_planets"][label]["planets"]
        ref_planets = reference["known_planets"].get(label, {}).get("planets")
        if ref_planets is None or len(planets) != len(ref_planets):
            failures.append(f"{label}/known_planets: missing or different count")
            continue
        for now, then in zip(planets, ref_planets):
            for key, value in now.items():
                ref = then.get(key)
                ok = close(value, ref) if isinstance(value, float) and isinstance(ref, float) else value == ref
                if not ok:
                    failures.append(f"{label}/known_planets/{now.get('target_id', '?')}/{key}: {value!r} != {ref!r}")
    return failures


if __name__ == "__main__":
    main()
