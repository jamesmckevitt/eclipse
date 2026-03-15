"""
Utility functions for coordinate transformations, unit conversions, and general helpers.
"""

from __future__ import annotations
import contextlib
import dataclasses
import subprocess
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
    import warnings
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


def _validate_yaml_keys(config: dict, section_classes: dict,
                        top_level_class) -> None:
    """
    Warn about unrecognized keys in a YAML config.

    Parameters
    ----------
    config : dict
        The full parsed YAML config.
    section_classes : dict
        Mapping of section name to the dataclass class used for that section,
        e.g. ``{"detector": Detector_SWC, "telescope": Telescope_EUVST, ...}``.
    top_level_class : dataclass class
        A dataclass whose field names define the valid non-section top-level
        keys (e.g. ``TopLevelConfig``).  Section names from *section_classes*
        are added automatically.
    """
    import warnings

    top_level_fields = {f.name for f in dataclasses.fields(top_level_class)}
    all_valid_top = top_level_fields | set(section_classes.keys())
    unknown_top = set(config.keys()) - all_valid_top
    if unknown_top:
        warnings.warn(
            f"Unrecognized top-level config key(s): {sorted(unknown_top)}. "
            "These will be ignored. Check for typos.",
            UserWarning,
        )

    for section_name, cls in section_classes.items():
        section = config.get(section_name, {})
        if not section:
            continue
        valid_fields = {f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")}
        unknown = set(section.keys()) - valid_fields
        if unknown:
            warnings.warn(
                f"Unrecognized key(s) in '{section_name}' section: {sorted(unknown)}. "
                f"Valid keys are: {sorted(valid_fields)}. Check for typos.",
                UserWarning,
            )


def _parse_section(section_dict: dict, class_name: str) -> tuple:
    """
    Parse a YAML config section into fixed and sweep parameters.

    Any field whose value is a list with more than one element becomes a sweep
    dimension.  Fields listed in ``_SECTION_LIST_FIELDS`` are always treated as
    a single (list-valued) fixed parameter.

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
    fixed = {}
    sweep = {}

    for key, val in section_dict.items():
        if key in list_fields:
            parsed = parse_yaml_input(val)
            fixed[key] = parsed if isinstance(parsed, list) else [parsed]
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
        if hasattr(val, "unit"):
            try:
                items[name] = float(val.si.value)
            except Exception:
                try:
                    items[name] = float(val.to(u.K, equivalencies=u.temperature()).value)
                except Exception:
                    items[name] = float(val.value)
        elif isinstance(val, (list, tuple)):
            items[name] = tuple(
                float(v.si.value) if hasattr(v, "unit") else v for v in val
            )
        else:
            items[name] = val
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
