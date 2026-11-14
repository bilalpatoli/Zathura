"""Kepler data access and preprocessing shared by the benchmark and diagnostics."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from lightkurve import LightCurve

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
CATALOG_PATH = ROOT / "data" / "catalog.json"
# Full ~90-day quarters, preferring early ones that predate the module 3 failure.
QUARTER_PREFERENCE = (3, 2, 5, 6, 7, 9, 10, 11, 13, 14, 15)
CADENCE_DAYS = 0.0204336


def cache_path(kepid):
    return CACHE / f"kic_{int(kepid)}.npz"


def download_curve(kepid, destination):
    """Download one quality-masked, normalized long-cadence PDCSAP quarter from MAST."""
    import lightkurve as lk

    result = lk.search_lightcurve(f"KIC {int(kepid)}", mission="Kepler", author="Kepler", exptime=1800)
    if len(result) == 0:
        raise RuntimeError(f"No Kepler long-cadence light curve for KIC {kepid}")
    quarters = list(result.table["mission"])
    for quarter in QUARTER_PREFERENCE:
        matches = [i for i, name in enumerate(quarters) if name == f"Kepler Quarter {quarter:02d}"]
        if matches:
            index = matches[0]
            break
    else:
        raise RuntimeError(f"No full preferred quarter for KIC {kepid}")
    curve = result[index].download(quality_bitmask="default", flux_column="pdcsap_flux")
    curve = curve.remove_nans().normalize()
    time = np.asarray(curve.time.value, dtype=np.float64)
    flux = np.asarray(curve.flux.value, dtype=np.float32)
    flux_err = np.asarray(curve.flux_err.value, dtype=np.float32)
    keep = np.isfinite(time) & np.isfinite(flux)
    if keep.sum() < 2000:
        raise RuntimeError(f"Too few valid cadences for KIC {kepid}: {keep.sum()}")
    np.savez_compressed(destination, time=time[keep], flux=flux[keep], flux_err=flux_err[keep],
                        quarter=np.int16(quarter), kepid=np.int64(kepid), origin="real",
                        product=str(result.table["productFilename"][index]))


@lru_cache(maxsize=1)
def load_catalog():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError("data/catalog.json is missing; run scripts/build_catalog.py")
    return json.loads(CATALOG_PATH.read_text())


@lru_cache(maxsize=256)
def _read(kepid):
    path = cache_path(kepid)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached curve {path.name}; run scripts/build_catalog.py")
    with np.load(path) as cached:
        return {key: cached[key].copy() if cached[key].ndim else cached[key].item() for key in cached.files}


def read_curve(kepid):
    """Cached raw curve (time in BKJD, normalized PDCSAP flux). Returns copies."""
    curve = _read(int(kepid))
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in curve.items()}


def transit_noise(time, flux, duration_days, reference_window=49):
    """Robust scatter of the flux averaged over one transit duration (a CDPP-like noise level).

    Uses a fixed ~1-day reference detrend so the benchmark does not depend on any agent config,
    and captures correlated stellar noise that point-to-point scatter misses."""
    flat = np.asarray(LightCurve(time=time, flux=flux / np.median(flux)).flatten(
        window_length=reference_window).flux.value, float)
    flat = flat[np.isfinite(flat)]
    n = max(1, int(round(duration_days / np.median(np.diff(time)))))
    binned = np.convolve(flat - np.median(flat), np.ones(n) / n, mode="valid")
    return float(1.4826 * np.median(np.abs(binned - np.median(binned))))


def preprocess(time, flux, detrend_window, outlier_sigma):
    """Sort, normalize, flatten and clip upper outliers; transit dips are preserved."""
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    valid = np.isfinite(time) & np.isfinite(flux)
    order = np.argsort(time[valid])
    time, flux = time[valid][order], flux[valid][order]
    flux = flux / np.median(flux)
    window = max(5, min(detrend_window, len(flux) // 2 * 2 - 1))
    flat = np.asarray(LightCurve(time=time, flux=flux).flatten(window_length=window).flux.value, float)
    scale = max(1.4826 * np.median(np.abs(flat - np.median(flat))), 1e-10)
    keep = np.isfinite(flat) & (flat < np.median(flat) + outlier_sigma * scale)
    return time[keep], flat[keep]
