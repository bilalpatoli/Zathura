"""Re-run the engine's headline results and check them against data/reference_results.json.

  .venv/bin/python scripts/reproduce.py                    # baseline + known-planet validation
  .venv/bin/python scripts/reproduce.py --config best.json # also a tuned config on dev and holdout
  .venv/bin/python scripts/reproduce.py --update-reference # record the current results as reference

Benchmark metrics are deterministic for fixed code, data and seeds, so anything other
than runtime must match the reference exactly.
"""
import argparse
import json
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
        print("No reference results; run with --update-reference to create one.")
        return
    reference = json.loads(REFERENCE.read_text())
    failures = []
    for label, current in report["configs"].items():
        expected = reference["configs"].get(label)
        if expected is None or expected["config"] != current["config"]:
            print(f"SKIP {label}: no reference for this config")
            continue
        for dataset in ("dev", "holdout"):
            for key in ("measurements", "truth_sha256", "curves_sha256"):
                if current[dataset][key] != expected[dataset][key]:
                    failures.append(f"{label}/{dataset}/{key}")
        if report["known_planets"][label]["planets"] != reference["known_planets"][label]["planets"]:
            failures.append(f"{label}/known_planets")
    if failures:
        sys.exit("FAIL: differs from reference: " + ", ".join(failures))
    print("PASS: reproduced reference results exactly (runtime excluded)")


if __name__ == "__main__":
    main()
