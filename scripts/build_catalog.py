"""Rebuild data/catalog.json and the committed Kepler light-curve cache.

Queries the NASA Exoplanet Archive for (a) quiet benchmark hosts with no KOI or
TCE and (b) confirmed 1-10 day planets used to validate the detector on real
transits, then downloads one long-cadence PDCSAP quarter per star from MAST.
Selection is seeded, so rerunning reproduces the same star lists while the
archive tables are unchanged.
"""
import argparse
import io
import json
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zathura.engine.data import CACHE, CATALOG_PATH, cache_path, download_curve  # noqa: E402

LK_CACHE = Path.home() / ".lightkurve" / "cache" / "mastDownload" / "Kepler"
TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
HOST_QUERY = (
    "select kepid, kepmag, teff, logg, rrmscdpp03p0 from keplerstellar "
    "where st_delivname='q1_q17_dr25_stellar' and kepmag between 11.5 and 13.5 and logg > 4.2 "
    "and teff between 5000 and 6500 and nkoi=0 and ntce=0 and st_quarters='11111111111111111'"
)
PLANET_QUERY = (
    "select kepid, kepoi_name, kepler_name, koi_period, koi_time0bk, koi_duration, koi_depth, koi_model_snr, "
    "koi_kepmag from cumulative where koi_disposition='CONFIRMED' and koi_period between 1 and 10 "
    "and koi_model_snr between 60 and 5000 and koi_kepmag < 14.5"
)
ALL_KOI_QUERY = "select kepid, koi_period, koi_disposition from cumulative"


def tap(sql):
    response = requests.get(TAP, params={"query": sql, "format": "csv"}, timeout=300)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def fetch(kepid):
    path = cache_path(kepid)
    if path.exists():
        return kepid, True
    for attempt in range(2):
        try:
            download_curve(kepid, path)
            return kepid, True
        except Exception as exc:  # noqa: BLE001 - report and skip unavailable stars
            # Interrupted downloads leave corrupt FITS files in the Lightkurve cache.
            for stale in LK_CACHE.glob(f"kplr{int(kepid):09d}_lc_*"):
                shutil.rmtree(stale, ignore_errors=True)
            error = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    print(f"skip KIC {kepid}: {error}", flush=True)
    return kepid, False


def download_until(candidates, needed, workers):
    kept = []
    with ProcessPoolExecutor(workers) as pool:
        for start in range(0, len(candidates), workers * 2):
            for kepid, ok in pool.map(fetch, candidates[start:start + workers * 2]):
                if ok and len(kept) < needed:
                    kept.append(int(kepid))
            print(f"downloaded {len(kept)}/{needed}", flush=True)
            if len(kept) >= needed:
                return kept
    raise RuntimeError(f"Only {len(kept)} of {needed} stars downloaded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts-per-split", type=int, default=40)
    parser.add_argument("--validation", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    hosts = tap(HOST_QUERY).drop_duplicates("kepid").sort_values("kepid")
    candidates = [int(k) for k in rng.permutation(hosts["kepid"].to_numpy())]
    kept = download_until(candidates, 2 * args.hosts_per_split, args.workers)
    splits = {"dev": sorted(kept[0::2]), "holdout": sorted(kept[1::2])}

    planets = tap(PLANET_QUERY).sort_values("koi_model_snr").reset_index(drop=True)
    koi = tap(ALL_KOI_QUERY)
    # Stratify by full-mission SNR so validation spans easy and marginal real transits.
    snr = planets["koi_model_snr"].to_numpy()
    targets = np.geomspace(snr.min(), snr.max(), args.validation * 2)
    picks = list(dict.fromkeys(int(np.argmin(np.abs(np.log(snr / t)))) for t in targets))
    validation = []
    for _, row in planets.iloc[picks].drop_duplicates("kepid").iterrows():
        if len(validation) >= args.validation:
            break
        if not fetch(int(row.kepid))[1]:
            continue
        others = koi[(koi.kepid == row.kepid) & (koi.koi_disposition != "FALSE POSITIVE")]
        validation.append({
            "kepid": int(row.kepid), "koi": row.kepoi_name, "name": row.kepler_name,
            "period_days": float(row.koi_period), "epoch_bkjd": float(row.koi_time0bk),
            "duration_days": float(row.koi_duration) / 24, "depth_ppm": float(row.koi_depth),
            "koi_model_snr": float(row.koi_model_snr),
            "host_koi_periods": sorted(float(p) for p in others.koi_period.dropna()),
        })

    catalog = {
        "version": "kepler-q-catalog-v1", "created": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed, "source": "NASA Exoplanet Archive TAP + MAST Kepler long-cadence PDCSAP",
        "host_query": HOST_QUERY, "planet_query": PLANET_QUERY, "hosts": splits, "validation": validation,
    }
    CATALOG_PATH.write_text(json.dumps(catalog, indent=1) + "\n")
    used = {*kept, *(v["kepid"] for v in validation)}
    for path in CACHE.glob("*.npz"):
        if not path.stem.startswith("kic_") or int(path.stem[4:]) not in used:
            path.unlink()
    print(f"catalog: {len(splits['dev'])} dev, {len(splits['holdout'])} holdout, {len(validation)} validation")


if __name__ == "__main__":
    main()
