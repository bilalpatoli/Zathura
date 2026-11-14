"""Cache real vetting examples without modifying the tuning benchmark or its splits."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_catalog import tap
from zathura.engine.data import ROOT, cache_path, download_curve, load_catalog


def main():
    existing = load_catalog()["validation"]
    ids = sorted({v["kepid"] for v in existing} | {11904151, 10593626, 8692861})
    query = ("select kepid, kepler_name, koi_period, koi_srad from cumulative "
             "where koi_disposition='CONFIRMED' and kepid in (" + ",".join(map(str, ids)) + ")")
    frame = tap(query)
    targets = []
    for kepid in ids:
        rows = frame[frame.kepid == kepid]
        if rows.empty:
            raise RuntimeError(f"No confirmed planet catalog entry for {kepid}")
        radius = rows.koi_srad.dropna()
        targets.append({"kepid": kepid, "name": str(rows.iloc[0].kepler_name).rsplit(" ", 1)[0],
                        "truth": "planet", "catalog_periods_days": list(map(float, rows.koi_period)),
                        "stellar_radius_solar": float(radius.iloc[0]) if len(radius) else None,
                        "source": "https://exoplanetarchive.ipac.caltech.edu/", "radius_source": "NASA cumulative koi_srad"})
    for kepid, period in ((10661783, 1.2313633), (4851217, 2.4702800), (4544587, 2.1890973)):
        targets.append({"kepid": kepid, "name": f"KIC {kepid}", "truth": "EB", "catalog_periods_days": [period],
                        "stellar_radius_solar": None, "radius_source": "Solar-radius assumption (not measured)",
                        "source": f"https://keplerebs.villanova.edu/overview/?k={kepid}"})
    for target in targets:
        path = cache_path(target["kepid"])
        if not path.exists():
            download_curve(target["kepid"], path)
        print(f"Cached {target['name']} · {target['truth']} · {path.name}", flush=True)
    catalog = {"description": "Convenience validation examples, not tuning or holdout stars. Truth never used by vet_candidate.",
               "planet_query": query, "photometry": "MAST Kepler quality-masked long-cadence PDCSAP, preferred full quarter",
               "eb_reference": "Kirk et al. 2016, Kepler Eclipsing Binary Stars VII, https://doi.org/10.3847/0004-6256/151/3/68",
               "targets": targets}
    (ROOT / "data/vetting_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
