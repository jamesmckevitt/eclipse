"""
Utility functions for coordinate transformations, unit conversions, and general helpers.
"""

from __future__ import annotations
import contextlib
import dataclasses
import subprocess
from pathlib import Path
import warnings
import numpy as np
import astropy.units as u
import astropy.constants as const
import joblib
from tqdm import tqdm


# Global debug flag - can be set by command line or configuration
DEBUG_MODE = False


def _get_mpi_info():
    """Return (comm, rank, world_size) if MPI is active with multiple ranks.

    MPI is auto-detected: if ``mpi4py`` is importable **and** the MPI world
    contains more than one process (i.e. launched via ``srun`` / ``mpirun``),
    the communicator is returned.  Otherwise falls back to single-process
    mode ``(None, 0, 1)``.
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        if size > 1:
            return comm, comm.Get_rank(), size
    except (ImportError, RuntimeError):
        # ImportError: mpi4py not installed.
        # RuntimeError: mpi4py installed but MPI library not loaded (e.g. on a
        #   login node before 'module load intel-mpi').  Falls back to serial mode.
        pass
    return None, 0, 1


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


def get_git_commit_id() -> str:
    """Get the last git commit ID from the package's git repository."""
    try:
        from importlib.resources import files
        pkg_path = Path(str(files("euvst_response"))).parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=pkg_path, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "unknown (not a git repository)"
    except Exception as e:
        return f"unknown ({e})"


def _get_software_version() -> str:
    """Get the installed package version."""
    try:
        import euvst_response
        return euvst_response.__version__
    except Exception:
        return "unknown"


def deduplicate_list(param_list, param_name):
    """
    Remove duplicates from a parameter list and warn if duplicates were found.

    Parameters
    ----------
    param_list : list
        List of parameter values that may contain duplicates.
    param_name : str
        Name of the parameter for warning messages.

    Returns
    -------
    list
        List with duplicates removed, preserving original order.
    """
    seen = set()
    deduplicated = []
    duplicates_found = False

    for item in param_list:
        if hasattr(item, "unit"):
            try:
                key = float(item.si.value)
            except Exception:
                try:
                    key = float(item.to(u.K, equivalencies=u.temperature()).value)
                except Exception:
                    key = float(item.value)
        else:
            key = item

        if key not in seen:
            seen.add(key)
            deduplicated.append(item)
        else:
            duplicates_found = True

    if duplicates_found:
        warnings.warn(
            f"Duplicate values found in '{param_name}' parameter list. "
            f"Removed duplicates: {len(param_list)} -> {len(deduplicated)} unique values.",
            UserWarning,
        )

    return deduplicated


# List-type dataclass fields: treated as a single fixed value (not a sweep dimension)
# even when the YAML value is a list.
_SECTION_LIST_FIELDS = {
    "simulation": [],
    "detector": [],
    "telescope": ["psf_params"],
    "filter": [],
}

# String-valued dataclass fields.  These bypass parse_yaml_input, which reads
# every string as a Quantity and would raise on "gaussian" or "2012-06-03".
# They can still be swept: a list of strings becomes a sweep dimension.
_SECTION_STRING_FIELDS = {
    # 'instrument' is deliberately absent. It is a Simulation field, but the
    # instrument is chosen by the top-level key and main() builds both
    # Simulation objects from that, so a value here would be parsed, swept and
    # then discarded. main() rejects it rather than letting it look effective.
    "simulation": [],
    "detector": ["material"],
    "telescope": ["psf_type", "calibration", "date"],
    "filter": [],
}


def _parse_section(section_dict: dict, class_name: str) -> tuple:
    """
    Parse a YAML config section into fixed and sweep parameters.

    Any field whose value is a list with more than one element becomes a sweep
    dimension.  Fields listed in ``_SECTION_LIST_FIELDS`` are always treated as
    a single (list-valued) fixed parameter, and fields listed in
    ``_SECTION_STRING_FIELDS`` are taken verbatim rather than parsed as
    quantities.

    Parameters
    ----------
    section_dict : dict
        The YAML section content, e.g. ``config["detector"]``.
    class_name : str
        The section name: ``"simulation"``, ``"detector"``, ``"telescope"``,
        or ``"filter"``.

    Returns
    -------
    fixed_params : dict
        ``{attr: value}`` -single values used for every combination.
    sweep_params : dict
        ``{attr: [values]}`` -lists of values to sweep over.
    """
    list_fields = _SECTION_LIST_FIELDS.get(class_name, [])
    string_fields = _SECTION_STRING_FIELDS.get(class_name, [])
    fixed = {}
    sweep = {}

    for key, val in section_dict.items():
        if key in list_fields:
            parsed = parse_yaml_input(val)
            fixed[key] = parsed if isinstance(parsed, list) else [parsed]
        else:
            if key in string_fields:
                parsed = list(val) if isinstance(val, (list, tuple)) else val
            else:
                parsed = parse_yaml_input(val)
            if isinstance(parsed, list):
                if len(parsed) == 1:
                    fixed[key] = parsed[0]
                else:
                    sweep[key] = deduplicate_list(parsed, f"{class_name}.{key}")
            else:
                fixed[key] = parsed

    return fixed, sweep


def _to_canonical_scalar(val):
    """Convert a parameter value to a canonical scalar for comparison."""
    if hasattr(val, "unit"):
        try:
            return float(val.si.value)
        except Exception:
            try:
                return float(val.to(u.K, equivalencies=u.temperature()).value)
            except Exception:
                return float(val.value)
    return val


def _params_to_key(params: dict) -> tuple:
    """
    Convert a parameters dict to a hashable tuple key.

    All Quantity values are converted to canonical SI scalars so that the key
    is independent of the units used in the YAML file.  Parameters are sorted
    by name to guarantee a deterministic ordering.

    ``simulation.pinhole_sizes`` and ``simulation.pinhole_positions`` are
    excluded from the key because they are not sweep dimensions.
    """
    _skip = {"simulation.pinhole_sizes", "simulation.pinhole_positions"}
    items = {}
    for name, val in params.items():
        if name in _skip:
            continue
        if isinstance(val, (list, tuple)):
            items[name] = tuple(_to_canonical_scalar(v) for v in val)
        else:
            items[name] = _to_canonical_scalar(val)
    return tuple(sorted(items.items()))


def _extract_config_params(obj, section: str) -> dict:
    """
    Extract all user-facing field values from a dataclass instance.

    Skips private fields (name starts with '_'), Path-valued fields (data file
    paths), and fields whose value is itself a dataclass (nested objects are
    stored under their own section).
    """
    params = {}
    for f in dataclasses.fields(obj):
        if f.name.startswith("_"):
            continue
        val = getattr(obj, f.name)
        if isinstance(val, Path):
            continue
        if dataclasses.is_dataclass(val):
            continue
        params[f"{section}.{f.name}"] = val
    return params



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


def _fwhm_to_sigma(fwhm: float) -> float:
    """Convert FWHM to Gaussian sigma: sigma = FWHM / (2 * sqrt(2 * ln2))."""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
