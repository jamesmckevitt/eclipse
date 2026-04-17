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


# Global debug flag - can be set by command line or configuration
DEBUG_MODE = False


def set_debug_mode(enabled: bool):
    """Set global debug mode."""
    global DEBUG_MODE
    DEBUG_MODE = enabled


def debug_break(message: str = "Debug break triggered", locals_dict=None, globals_dict=None):
    """
    Break into IPython debugger if debug mode is enabled.
    
    Usage:
        debug_break("Check values here", locals(), globals())
    or:
        debug_break("Error occurred")
    """
    if not DEBUG_MODE:
        return
        
    print(f"\n=== DEBUG BREAK: {message} ===")
    
    try:
        # Try to import and start IPython
        from IPython import embed
        
        # Prepare namespace for IPython
        user_ns = {}
        if locals_dict:
            user_ns.update(locals_dict)
        if globals_dict:
            user_ns.update(globals_dict)
            
        print("Starting IPython session...")
        print("Available variables:", list(user_ns.keys()) if user_ns else "None provided")
        print("Type 'exit()' or Ctrl+D to continue execution")
        
        # Start IPython with the provided namespace
        embed(user_ns=user_ns)
        
    except ImportError:
        print("IPython not available. Using standard Python debugger...")
        import pdb
        pdb.set_trace()


def debug_on_error(func):
    """
    Decorator to automatically break into debugger on exceptions when debug mode is enabled.
    
    Usage:
        @debug_on_error
        def my_function():
            # your code here
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if DEBUG_MODE:
                print(f"\n=== EXCEPTION IN {func.__name__}: {e} ===")
                # Get the frame where the exception occurred
                import sys
                frame = sys.exc_info()[2].tb_frame
                debug_break(f"Exception in {func.__name__}: {e}", frame.f_locals, frame.f_globals)
            raise
    return wrapper


def wl_to_vel(wl: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    """Convert wavelength to line-of-sight velocity."""
    return (wl - wl0) / wl0 * const.c


def vel_to_wl(v: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    """Convert line-of-sight velocity to wavelength."""
    return wl0 * (1 + v / const.c)


def gaussian(wave, peak, centre, sigma, back):
    """Gaussian function for spectral line fitting."""
    return peak * np.exp(-0.5 * ((wave - centre) / sigma) ** 2) + back


def multi_gaussian(wave, *params, n_components=1):
    """Multi-component Gaussian plus constant background.

    Parameters are ordered as::

        [peak_0, centre_0, sigma_0, peak_1, centre_1, sigma_1, ..., background]

    Total number of parameters = 3 * n_components + 1.
    """
    result = np.zeros_like(wave, dtype=float)
    for i in range(n_components):
        peak = params[3 * i]
        centre = params[3 * i + 1]
        sigma = params[3 * i + 2]
        if sigma == 0:
            continue
        result += peak * np.exp(-0.5 * ((wave - centre) / sigma) ** 2)
    result += params[-1]  # background
    return result


def angle_to_distance(angle: u.Quantity) -> u.Quantity:
    """Convert angular size to linear distance at 1 AU."""
    if angle.unit.physical_type != "angle":
        raise ValueError("Input must be an angle")
    return 2 * const.au * np.tan(angle.to(u.rad) / 2)


def rebin_slit_offchip(cube, n_bin: int):
    """Sum adjacent pixels along the slit axis to simulate off-chip binning.

    Off-chip (ground-based) binning sums already-read-out pixels, so each
    pixel carries its own independent noise (read noise, dark current, etc.).
    The resulting signal increases by *n_bin* while uncorrelated noise adds
    in quadrature, improving SNR by sqrt(n_bin).

    Parameters
    ----------
    cube : NDCube
        Data cube with shape ``(n_scan, n_slit, n_lambda)``.
    n_bin : int
        Number of slit pixels to sum.  Must be >= 1.
        Pixels that don't fill a complete bin at the slit edge are discarded.

    Returns
    -------
    NDCube
        Rebinned cube with shape ``(n_scan, n_slit // n_bin, n_lambda)``.
        WCS is updated so the slit pixel scale (CDELT) is scaled by *n_bin*.
    """
    from ndcube import NDCube

    if n_bin < 1:
        raise ValueError(f"n_bin must be >= 1, got {n_bin}")
    if n_bin == 1:
        return cube

    data = cube.data
    n_scan, n_slit, n_lam = data.shape
    n_keep = (n_slit // n_bin) * n_bin
    trimmed = data[:, :n_keep, :]
    rebinned = trimmed.reshape(n_scan, n_keep // n_bin, n_bin, n_lam).sum(axis=2)

    # Update WCS for the slit axis.
    # Numpy axis 1 (slit) corresponds to WCS axis 1 (HPLT) in the
    # reversed FITS convention (naxis-1-numpy_axis for a 3-axis WCS).
    new_wcs = cube.wcs.deepcopy()
    slit_wcs_axis = 1  # HPLT-TAN
    new_wcs.wcs.cdelt[slit_wcs_axis] *= n_bin
    # Map the original reference pixel to the new grid.
    # FITS crpix is 1-based; the center-preserving mapping for binning
    # anchored at pixel 1 is:  crpix_new = (crpix_old - 0.5) / n_bin + 0.5
    new_wcs.wcs.crpix[slit_wcs_axis] = (
        new_wcs.wcs.crpix[slit_wcs_axis] - 0.5
    ) / n_bin + 0.5

    return NDCube(data=rebinned, wcs=new_wcs, unit=cube.unit,
                  meta=cube.meta)


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
