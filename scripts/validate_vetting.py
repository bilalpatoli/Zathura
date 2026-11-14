"""Reproduce the real-source screening confusion matrix without network calls."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zathura.engine.data import ROOT
from zathura.engine.vetting import validate_vetting


if __name__ == "__main__":
    report = validate_vetting()
    (ROOT / "data/vetting_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["confusion"], indent=2))
    for row in report["targets"]:
        print(row["name"], row["truth"], "→", row["verdict"], " · ".join(
            f"{t['name']}: {t['value']}" for t in row["tests"] if t["passed"] is False))
