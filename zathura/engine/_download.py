"""Isolated download worker so unreachable MAST cannot block the demo."""
import sys

import lightkurve as lk
import numpy as np


def download(target_id, destination):
    result = lk.search_lightcurve(target_id, mission="Kepler", author="Kepler", cadence="long")
    if len(result) == 0:
        raise RuntimeError(f"No Kepler light curve found for {target_id}")
    curve = result[0].download().remove_nans().normalize()
    time = np.asarray(curve.time.value, dtype=float)
    flux = np.asarray(curve.flux.value, dtype=float)
    keep = (time < time.min() + 28) & np.isfinite(flux)
    np.savez_compressed(destination, time=time[keep], flux=flux[keep], origin="real")


if __name__ == "__main__":
    download(sys.argv[1], sys.argv[2])
