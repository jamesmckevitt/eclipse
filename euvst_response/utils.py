"""
Utility functions for coordinate transformations, unit conversions, and general helpers.
"""

from __future__ import annotations
import contextlib
from pathlib import Path
import numpy as np
import astropy.units as u
import astropy.constants as const
import joblib
from tqdm import tqdm


def wl_to_vel(wl: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    """Convert wavelength to line-of-sight velocity."""
    return (wl - wl0) / wl0 * const.c


def vel_to_wl(v: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    """Convert line-of-sight velocity to wavelength."""
    return wl0 * (1 + v / const.c)


def gaussian(wave, peak, centre, sigma, back):
    """Gaussian function for spectral line fitting."""
    return peak * np.exp(-0.5 * ((wave - centre) / sigma) ** 2) + back


def angle_to_distance(angle: u.Quantity) -> u.Quantity:
    """Convert angular size to linear distance at 1 AU."""
    if angle.unit.physical_type != "angle":
        raise ValueError("Input must be an angle")
    return 2 * const.au * np.tan(angle.to(u.rad) / 2)


def distance_to_angle(distance: u.Quantity) -> u.Quantity:
    """Convert linear distance to angular size at 1 AU."""
    if distance.unit.physical_type != "length":
        raise ValueError("Input must be a length")
    return (2 * np.arctan(distance / (2 * const.au))).to(u.arcsec)


def parse_yaml_input(val):
    """Parse YAML input values - handle both single values and lists."""
    if isinstance(val, str):
        return u.Quantity(val)
    elif isinstance(val, (list, tuple)):
        # Handle list of values
        if all(isinstance(v, str) for v in val):
            return [u.Quantity(v) for v in val]
        else:
            return list(val)
    else:
        return val


def ensure_list(val):
    """Ensure input is a list (for parameter sweeps)."""
    if not isinstance(val, (list, tuple)):
        return [val]
    return list(val)


def save_maps(path: str | Path, log_intensity: np.ndarray, v_map: u.Quantity,
              x_pix_size: float, y_pix_size: float) -> None:
    """Save intensity and velocity maps for later comparison."""
    np.savez(
        path,
        log_si=log_intensity,
        v_map=v_map.to(u.km / u.s).value,
        x_pix_size=x_pix_size,
        y_pix_size=y_pix_size,
    )


def load_maps(path: str | Path) -> dict:
    """Load previously saved intensity and velocity maps."""
    dat = np.load(path)
    return dict(
        log_si=dat["log_si"],
        v_map=dat["v_map"],
        x_pix_size=float(dat["x_pix_size"]),
        y_pix_size=float(dat["y_pix_size"]),
    )


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager that patches joblib so it uses the supplied tqdm
    instance to report progress.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):  # type: ignore[attr-defined]
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack  # type: ignore[attr-defined]
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()
