"""Box Least Squares transit search on a preprocessed light curve."""
import numpy as np
from astropy.timeseries import BoxLeastSquares

from zathura.engine.data import preprocess

PERIOD_MIN_DAYS = 1.0
PERIOD_MAX_DAYS = 10.0
DURATION_COUNT = 6


def period_grid(baseline_days, size):
    """Periods uniformly spaced in frequency, which matches BLS peak widths."""
    p_max = min(PERIOD_MAX_DAYS, baseline_days / 2)
    if p_max <= PERIOD_MIN_DAYS:
        raise ValueError("Light curve too short for a 1-day minimum period search")
    return np.sort(1 / np.linspace(1 / p_max, 1 / PERIOD_MIN_DAYS, size))


def search(time, flux, config):
    """Return BLS spectrum and the strongest peak; detection uses the SDE threshold."""
    t, flat = preprocess(time, flux, config.detrend_window, config.outlier_sigma)
    if len(t) < 10:
        raise ValueError("Insufficient finite light-curve samples for BLS")
    periods = period_grid(np.ptp(t), config.period_grid_size)
    durations = np.geomspace(config.duration_min_days, config.duration_max_days, DURATION_COUNT)
    result = BoxLeastSquares(t, flat).power(periods, durations)
    power = np.asarray(result.power)
    peak = int(np.argmax(power))
    sde = float((power[peak] - power.mean()) / max(power.std(), 1e-12))
    return {"time": t, "flat_flux": flat, "periods": periods, "power": power,
            "period_days": float(result.period[peak]), "epoch": float(result.transit_time[peak]),
            "duration_days": float(result.duration[peak]), "depth": float(result.depth[peak]),
            "sde": sde, "detected": sde >= config.detection_threshold}
