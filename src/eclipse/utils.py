"""
Utility functions for coordinate transformations, unit conversions, and general helpers.
"""

from __future__ import annotations
import contextlib
import functools
import difflib
import dataclasses
import subprocess
import sys
from pathlib import Path
import warnings
import numpy as np
import yaml
import astropy.units as u
import astropy.constants as const
import joblib
from scipy import sparse
from tqdm import tqdm
from mendeleev import element


# Global debug flag - can be set by command line or configuration
DEBUG_MODE = False

# Recorded in every synthesised line cube. Cubes without it took the velocity
# from the simulation's component along the integration axis, which is
# opposite in sign to the velocity away from the observer for views along x
# and z, and the same for views along y.
VELOCITY_CONVENTION = "line-of-sight velocity, positive away from the observer"

# Observer side for a view given only by its axis: +1 on the side of
# increasing coordinate, -1 on the other. The observer is above the box (+z)
# for the top view, and at +x and -y for the two side views.
OBSERVER_SIDE = {"x": +1, "y": -1, "z": +1}


def _warn_outside(message: str, category=FutureWarning) -> None:
    """
    Warn, attributed to the first caller outside ECLIPSE.

    Frames in `dataclasses` and `copy` are skipped too, so a warning raised
    while `dataclasses.replace` makes an object names the line that called it.
    """
    frame, level = sys._getframe(1), 2
    package = __name__.split(".")[0] + "."
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if not (module.startswith(package) or module in ("dataclasses", "copy")):
            break
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level)


def view_axis_and_side(view: str) -> tuple[str, int]:
    """
    Axis of a view and the side of the box the observer is on.

    Parameters
    ----------
    view : str
        `"x"`, `"y"` or `"z"`, seen from the side in :data:`OBSERVER_SIDE`,
        or an axis with a sign for the side, such as `"-x"` for the view
        from -x.

    Returns
    -------
    axis : str
        `"x"`, `"y"` or `"z"`.
    side : int
        +1 if the observer is on the side of increasing coordinate, -1
        otherwise.
    """
    if isinstance(view, str) and view in OBSERVER_SIDE:
        return view, OBSERVER_SIDE[view]
    if isinstance(view, str) and len(view) == 2 and view[0] in "+-" and view[1] in OBSERVER_SIDE:
        return view[1], +1 if view[0] == "+" else -1
    raise ValueError(f"A view is 'x', 'y' or 'z', or one of them with + or - for the side "
                     f"of the box to look from, such as '-x'; got {view!r}.")


def view_name(view: str) -> str:
    """
    Name a view is recorded under.

    A view from the default side in :data:`OBSERVER_SIDE` is named by its
    axis alone, as in files written before other sides could be chosen. A
    view from the other side is named by its axis with the sign of its side,
    such as `"-x"`.
    """
    axis, side = view_axis_and_side(view)
    return axis if side == OBSERVER_SIDE[axis] else ("+" if side > 0 else "-") + axis


def has_wrong_velocity_sign(meta) -> bool:
    """
    Whether a synthesised cube's Doppler shifts use the old sign convention.

    A cube that records :data:`VELOCITY_CONVENTION` uses the current one. A
    cube without it predates the switch to velocity away from the observer,
    which reversed the sign for views along x and z but not y. A cube with
    no integration axis predates the side views, so it is a view along z.

    Parameters
    ----------
    meta : dict or None
        Cube metadata.

    Returns
    -------
    bool
    """
    meta = meta or {}
    if meta.get("velocity_convention") == VELOCITY_CONVENTION:
        return False
    return meta.get("integration_axis", "z") != "y"


def _get_mpi_info():
    """Return (comm, rank, world_size) if MPI is active with multiple ranks.

    MPI is used if `mpi4py` imports and the MPI world has more than one
    process (launched with `srun` or `mpirun`). Otherwise this returns
    `(None, 0, 1)` for a single process.
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


def debug_break(message: str = "Debug break triggered", locals_dict=None, globals_dict=None,
                traceback=None):
    """
    Break into IPython debugger if debug mode is enabled.

    Without IPython, pdb opens `traceback` post mortem at the frame that
    raised, if given, and otherwise stops here.
    
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
        # Globals first, so a local of the same name wins.
        user_ns = {}
        if globals_dict:
            user_ns.update(globals_dict)
        if locals_dict:
            user_ns.update(locals_dict)
            
        print("Starting IPython session...")
        print("Available variables:", list(user_ns.keys()) if user_ns else "None provided")
        print("Type 'exit()' or Ctrl+D to continue execution")
        
        # Start IPython with the provided namespace
        embed(user_ns=user_ns)
        
    except ImportError:
        print("IPython not available. Using standard Python debugger...")
        import pdb
        if traceback is not None:
            pdb.post_mortem(traceback)
        else:
            pdb.set_trace()


def debug_on_error(func):
    """
    Decorator to automatically break into debugger on exceptions when debug mode is enabled.
    
    Usage:
        @debug_on_error
        def my_function():
            # your code here
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if DEBUG_MODE:
                print(f"\n=== EXCEPTION IN {func.__name__}: {type(e).__name__}: {e} ===")
                # Open the session, with the exception as `exception`, in the
                # innermost ECLIPSE frame: the decorated function (the second
                # frame of the traceback, after this wrapper) or a function
                # it called. Frames in numpy or astropy say little about the
                # run. Without IPython, pdb opens where the exception was
                # raised and can go up.
                package = __name__.split(".")[0]
                frame = (e.__traceback__.tb_next or e.__traceback__).tb_frame
                entry = e.__traceback__.tb_next
                while entry is not None:
                    if entry.tb_frame.f_globals.get("__name__", "").split(".")[0] == package:
                        frame = entry.tb_frame
                    entry = entry.tb_next
                debug_break(f"Exception in {func.__name__}: {e}",
                            {**frame.f_locals, "exception": e}, frame.f_globals,
                            traceback=e.__traceback__)
            raise
    return wrapper


# Cache for element_data, keyed by symbol, atomic number and the lookup key.
_ELEMENTS: dict = {}


def element_data(symbol_or_number: str | int) -> tuple[int, float]:
    """
    Atomic number and atomic weight (atomic mass units) of an element.

    The element is given by its symbol, such as `"Fe"`, or its atomic
    number. A lookup in mendeleev's database takes a fraction of a second,
    so each element is cached under both its symbol and its atomic number.

    Raises
    ------
    ValueError
        If there is no such element.
    """
    if symbol_or_number not in _ELEMENTS:
        atom = element(symbol_or_number)
        data = (atom.atomic_number, atom.atomic_weight)
        for key in (atom.symbol, atom.atomic_number, symbol_or_number):
            _ELEMENTS[key] = data
    return _ELEMENTS[symbol_or_number]


def wl_to_vel(wl: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    """
    Line-of-sight velocity that shifts a line from `wl0` to `wl`.

    A positive velocity is a redshift, away from the observer.

    Parameters
    ----------
    wl : u.Quantity
        Observed wavelength.
    wl0 : u.Quantity
        Rest wavelength.

    Returns
    -------
    u.Quantity
        Velocity.
    """
    return (wl - wl0) / wl0 * const.c


def vel_to_wl(v: u.Quantity, wl0: u.Quantity) -> u.Quantity:
    """
    Wavelength of a line shifted by a line-of-sight velocity.

    A positive velocity is a redshift, away from the observer.

    Parameters
    ----------
    v : u.Quantity
        Velocity.
    wl0 : u.Quantity
        Rest wavelength.

    Returns
    -------
    u.Quantity
        Shifted wavelength.
    """
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


def pixel_mean_gaussians(wave, *params, n_components=1, pixel=1.0):
    """
    :func:`multi_gaussian` averaged over pixels `pixel` wide centred on `wave`.

    This is what a detector pixel records. For a line narrower than a pixel
    it differs from the Gaussian at the pixel centre. The parameters are as
    for :func:`multi_gaussian`.
    """
    from scipy.special import erf

    result = np.zeros_like(wave, dtype=float)
    half = pixel / 2
    for i in range(n_components):
        peak = params[3 * i]
        centre = params[3 * i + 1]
        sigma = params[3 * i + 2]
        if sigma == 0:
            continue
        scale = np.sqrt(2.0) * sigma
        result += (peak * sigma * np.sqrt(np.pi / 2) / pixel
                   * (erf((wave + half - centre) / scale) - erf((wave - half - centre) / scale)))
    result += params[-1]  # background
    return result


def _bin_edges(centres: np.ndarray) -> np.ndarray:
    """Bin edges halfway between `centres`, the outer ones mirrored about the end centres."""
    inner = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate([[centres[0] - (inner[0] - centres[0])], inner,
                           [centres[-1] + (centres[-1] - inner[-1])]])


def onto_wavelength_bins(spectra: np.ndarray, wavelength: np.ndarray,
                         reference: np.ndarray) -> np.ndarray:
    """
    Spectra averaged onto the wavelength bins of another grid.

    Each sample covers the bin halfway to its neighbours, and each reference
    bin gets the mean of the samples overlapping it, taken as zero outside
    the sampled range. This keeps the integral, so a line narrower than a
    bin is not lost as it would be by interpolation. Spectra already on the
    reference grid are returned unchanged.

    `wavelength` and `reference` are increasing, in the same unit, and
    `spectra` has wavelength on its last axis.
    """
    spectra = np.asarray(spectra, dtype=float)
    if wavelength.shape == reference.shape and np.array_equal(wavelength, reference):
        return spectra
    source, target = _bin_edges(wavelength), _bin_edges(reference)
    # Only overlapping bins are used, to save memory and time and to stop a
    # NaN in one sample spreading to every bin.
    last_bin = reference.size - 1
    first = np.clip(np.searchsorted(target, source[:-1], side="right") - 1, 0, last_bin)
    last = np.clip(np.searchsorted(target, source[1:], side="left") - 1, 0, last_bin)
    count = last - first + 1
    rows = np.repeat(np.arange(wavelength.size), count)
    cols = np.repeat(first, count) + np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    overlap = np.minimum(source[rows + 1], target[cols + 1]) - np.maximum(source[rows], target[cols])
    kept = overlap > 0
    weights = sparse.csr_matrix(
        (overlap[kept] / np.diff(target)[cols[kept]], (rows[kept], cols[kept])),
        shape=(wavelength.size, reference.size))
    flat = spectra.reshape(-1, wavelength.size)
    return np.asarray((weights.T @ flat.T).T).reshape(spectra.shape[:-1] + reference.shape)


def angle_to_distance(angle: u.Quantity) -> u.Quantity:
    """
    Length on the Sun subtended by an angle seen from 1 AU.

    Parameters
    ----------
    angle : u.Quantity
        Angle.

    Returns
    -------
    u.Quantity
        Length.
    """
    if angle.unit.physical_type != "angle":
        raise ValueError("Input must be an angle")
    return 2 * const.au * np.tan(angle.to(u.rad) / 2)


def rebin_slit_offchip(cube, n_bin: int):
    """Sum adjacent pixels along the slit axis to simulate off-chip binning.

    Off-chip (ground-based) binning sums pixels after read-out, so each
    carries its own independent noise (read noise, dark current, etc.). The
    signal grows by `n_bin` and uncorrelated noise adds in quadrature, so
    the SNR improves by sqrt(n_bin).

    Parameters
    ----------
    cube : NDCube
        Data cube with shape `(n_slit, n_scan, n_lambda)`.
    n_bin : int
        Number of slit pixels to sum, >= 1. Pixels at the slit edge that do
        not fill a whole bin are discarded.

    Returns
    -------
    NDCube
        Rebinned cube with shape `(n_slit // n_bin, n_scan, n_lambda)`. The
        slit pixel scale (CDELT) in the WCS is multiplied by `n_bin`.
    """
    from ndcube import NDCube

    if n_bin < 1:
        raise ValueError(f"n_bin must be >= 1, got {n_bin}")
    if n_bin == 1:
        return cube

    data = cube.data
    n_slit, n_scan, n_lam = data.shape
    if n_bin > n_slit:
        raise ValueError(f"offchip_bin_slit {n_bin} bins more rows than the {n_slit} along the "
                         f"slit, so there would be nothing left; bin at most {n_slit}.")
    n_keep = (n_slit // n_bin) * n_bin
    trimmed = data[:n_keep, :, :]
    rebinned = trimmed.reshape(n_keep // n_bin, n_bin, n_scan, n_lam).sum(axis=1)

    # Update WCS for the slit axis.
    # Numpy axis 0 (slit) corresponds to WCS axis 2 (HPLT) in the
    # reversed FITS convention (naxis-1-numpy_axis for a 3-axis WCS).
    new_wcs = cube.wcs.deepcopy()
    slit_wcs_axis = 2  # HPLT-TAN
    new_wcs.wcs.cdelt[slit_wcs_axis] *= n_bin
    # Map the original reference pixel to the new grid.
    # FITS crpix is 1-based; the center-preserving mapping for binning
    # anchored at pixel 1 is:  crpix_new = (crpix_old - 0.5) / n_bin + 0.5
    new_wcs.wcs.crpix[slit_wcs_axis] = (
        new_wcs.wcs.crpix[slit_wcs_axis] - 0.5
    ) / n_bin + 0.5

    # Drop the per-pixel photon metadata: binned_photon_wavelengths works out
    # the binned pixels' photon wavelengths from the photons of their rows.
    meta = cube.meta
    carried = ("photon_wavelength", "photon_rms_wavelength", "photon_filter_transmission",
               "blocked_photon_wavelength", "blocked_photon_rms_wavelength")
    if isinstance(meta, dict) and any(key in meta for key in carried):
        meta = {key: value for key, value in meta.items() if key not in carried}
    return NDCube(data=rebinned, wcs=new_wcs, unit=cube.unit, meta=meta)


def binned_photon_wavelengths(photons, n_bin: int) -> tuple:
    """
    Wavelengths of the mean and RMS photon energies in off-chip binned pixels.

    Parameters
    ----------
    photons : NDCube
        Photons in each pixel, with the wavelengths of the mean and RMS
        photon energies in `meta["photon_wavelength"]` and
        `meta["photon_rms_wavelength"]`, as from `simulate_once`. Without the
        second, each pixel's photons are taken to have a single energy.
    n_bin : int
        Number of pixels summed along the slit, as in
        :func:`rebin_slit_offchip`.

    Returns
    -------
    tuple of u.Quantity
        `(mean, rms)`, each with the shape of the binned pixels. A binned
        pixel with no photons takes the mean-energy wavelength of its first
        row for both.
    """
    from ndcube import NDCube
    from .sampling import _carry_photon_wavelengths

    mean = u.Quantity(photons.meta["photon_wavelength"])
    rms = u.Quantity(photons.meta.get("photon_rms_wavelength", mean)).to(mean.unit)
    if n_bin == 1:
        return mean, rms
    n_keep = (photons.data.shape[0] // n_bin) * n_bin

    def binned(values):
        return rebin_slit_offchip(NDCube(values, wcs=photons.wcs), n_bin).data

    _, binned_mean, binned_rms = _carry_photon_wavelengths(
        binned, photons.data, mean.value, rms.value, mean.value[:n_keep:n_bin])
    return binned_mean * mean.unit, binned_rms * mean.unit


def distance_to_angle(distance: u.Quantity) -> u.Quantity:
    """
    Angle subtended by a length on the Sun seen from 1 AU.

    Parameters
    ----------
    distance : u.Quantity
        Length.

    Returns
    -------
    u.Quantity
        Angle in arcsec.
    """
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
    """
    Get the last git commit ID from the package's git repository.

    Only a git checkout of ECLIPSE, with the package in its `src` folder,
    has one. An environment inside another repository, such as a project's
    .venv, is not part of it, and that repository's commit says nothing
    about ECLIPSE.
    """
    try:
        from importlib.resources import files
        package = Path(str(files("eclipse"))).resolve()
        pkg_path = package.parent.parent
        if pkg_path / "src" / "eclipse" != package:
            return "unknown (not installed from a git checkout of ECLIPSE)"
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=pkg_path, timeout=5,
        )
        if top.returncode != 0:
            return "unknown (not a git repository)"
        if Path(top.stdout.strip()).resolve() != pkg_path.resolve():
            return "unknown (not installed from a git checkout of ECLIPSE)"
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
        import eclipse
        return eclipse.__version__
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
    "detector": ["dispersion", "lit_band", "slit_image_tilt"],
    "telescope": ["psf_params", "fiducial_centres", "fiducial_lengths"],
    "filter": [],
}

# String-valued dataclass fields.  These bypass parse_yaml_input, which reads
# every string as a Quantity and would raise on "gaussian" or "2012-06-03".
# They can still be swept: a list of strings becomes a sweep dimension.
_SECTION_STRING_FIELDS = {
    # 'instrument' is left out on purpose. It is a Simulation field, but the
    # top-level key chooses the instrument and main() builds both Simulation
    # objects from it, so a value here would be parsed, swept and discarded.
    # main() rejects it rather than let it appear to take effect.
    "simulation": ["psf_boundary", "spectral_psf"],
    "detector": ["material"],
    "telescope": ["psf_type", "calibration", "date", "pm_table", "grating_table",
                  "field_of_view", "resolution_budget"],
    "filter": ["al_table", "oxide_table", "c_table"],
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """PyYAML's safe loader, refusing repeated keys in a mapping."""


def _construct_mapping_once(loader, node, deep=False):
    # YAML forbids repeated keys, but PyYAML keeps the last, so a section
    # written twice would lose its first block and fall back to defaults.
    seen = set()
    for key_node, _ in node.value:
        # Skip merge keys ("<<: *anchor"): construct_mapping brings in the
        # anchor's keys, which keys beside it may override, as YAML allows.
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"the key {key!r} is given twice in one mapping", key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                                 _construct_mapping_once)


def load_yaml_config(text: str):
    """Parse configuration YAML as yaml.safe_load does, but refuse repeated keys."""
    return yaml.load(text, Loader=_UniqueKeyLoader)


def _parse_section(section_dict: dict, class_name: str) -> tuple:
    """
    Parse a YAML config section into fixed and sweep parameters.

    Any field whose value is a list of more than one element becomes a sweep
    dimension. Fields in `_SECTION_LIST_FIELDS` are always one fixed
    (list-valued) parameter, and fields in `_SECTION_STRING_FIELDS` are taken
    verbatim rather than parsed as quantities.

    Parameters
    ----------
    section_dict : dict
        YAML section content, e.g. `config["detector"]`.
    class_name : str
        Section name: `"simulation"`, `"detector"`, `"telescope"` or
        `"filter"`.

    Returns
    -------
    fixed_params : dict
        `{attr: value}`: single values used for every combination.
    sweep_params : dict
        `{attr: [values]}`: lists of values to sweep over.
    """
    list_fields = _SECTION_LIST_FIELDS.get(class_name, [])
    string_fields = _SECTION_STRING_FIELDS.get(class_name, [])
    fixed = {}
    sweep = {}

    for key, val in section_dict.items():
        # An empty key would read as None: noise off, or no exposure times and
        # so no run at all, reported as a success.
        if val is None or (isinstance(val, (list, tuple)) and len(val) == 0):
            raise ValueError(f"'{class_name}.{key}' is empty. Give it a value, or leave it "
                             f"out for its default.")
        if key in list_fields:
            parsed = parse_yaml_input(val)
            fixed[key] = parsed if isinstance(parsed, list) else [parsed]
        elif key.endswith("_table") and isinstance(val, (list, tuple)):
            # Result keys leave out the tables, which are files, so a sweep
            # over tables would give every table's results the same key.
            raise ValueError(f"'{class_name}.{key}' names one table; to compare tables, "
                             f"run each in its own configuration.")
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

    Quantities become canonical SI scalars, so the key does not depend on
    the units in the YAML file. Parameters are sorted by name for a fixed
    order.

    `simulation.pinhole_sizes` and `simulation.pinhole_positions` are left
    out because they are not sweep dimensions.
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


# Keys that moved or were renamed, so an old config is told where the setting
# went, not just that the key is unknown.
_RENAMED_CONFIG_KEYS = {
    "aluminium_thickness": "filter.al_thickness",
    "slit_bin_pairs": "offchip_bin_slit",
    "exposure": "simulation.expos",
}


def _describe_key(key: str, where: str) -> str:
    """Phrase naming `key` in section `where` ('' for the top level)."""
    if where:
        return f"'{key}' in the '{where}:' section"
    return f"'{key}' at the top level"


def suggest_config_key(key: str, allowed, elsewhere: dict | None = None):
    """
    Best guess at what an unrecognised config key was meant to be.

    Looks for a rename, then the same name in another section, then a near
    miss. The section lookup matters most in practice: the old flat layout
    put keys such as `expos` and `ccd_temperature` at the top level, where
    they are valid names at the wrong depth.

    Parameters
    ----------
    key : str
        Unrecognised key.
    allowed : iterable of str
        Keys that are valid where this one appeared.
    elsewhere : dict, optional
        `{section_name: valid_keys}` for the other places a key could live.
        A section name of `''` means the top level.

    Returns
    -------
    str or None
        A phrase naming the suggestion, or None if nothing looks close.
    """
    if key in _RENAMED_CONFIG_KEYS:
        return f"'{_RENAMED_CONFIG_KEYS[key]}'"

    for where, fields in (elsewhere or {}).items():
        if key in fields:
            return _describe_key(key, where)

    close = difflib.get_close_matches(key, sorted(allowed), n=1, cutoff=0.7)
    if close:
        return f"'{close[0]}'"

    for where, fields in (elsewhere or {}).items():
        close = difflib.get_close_matches(key, sorted(fields), n=1, cutoff=0.8)
        if close:
            return _describe_key(close[0], where)

    return None


def check_config_keys(provided, allowed, context: str,
                      elsewhere: dict | None = None) -> None:
    """
    Raise if `provided` has any key not in `allowed`.

    An unread key is not a harmless typo: the run uses the default, gives
    plausible output and says nothing. A sweep at the wrong depth is worse,
    as it returns results that are identical for every combination.

    Parameters
    ----------
    provided : iterable of str
        Keys found in the config.
    allowed : iterable of str
        Keys that are valid here.
    context : str
        Where the keys are, for the error message, e.g. `"top-level"`.
    elsewhere : dict, optional
        Passed to :func:`suggest_config_key`.

    Raises
    ------
    ValueError
        If any key is unrecognised, listing each one with a suggestion.
    """
    allowed = set(allowed)
    unknown = [k for k in provided if k not in allowed]
    if not unknown:
        return

    lines = []
    for key in sorted(unknown, key=str):
        guess = suggest_config_key(str(key), allowed, elsewhere)
        if guess:
            lines.append(f"  {key!r}: did you mean {guess}?")
        else:
            lines.append(f"  {key!r}")

    visible = sorted(k for k in allowed if not k.startswith("_"))
    plural = "keys" if len(unknown) > 1 else "key"
    raise ValueError(
        f"Unrecognised {context} config {plural}:\n"
        + "\n".join(lines)
        + f"\n\nECLIPSE never reads these, so the run would have used the "
        f"default for whatever each was meant to set.\n"
        f"Valid {context} keys: {', '.join(visible)}"
    )


# Units in the last place of single precision that an evenly spaced grid may
# be off by. MURaM's float32 heights are up to 1.4 ULP from even (7 m at
# 42 Mm), and edges computed from such centres add their own rounding.
_SINGLE_PRECISION_ULPS = 4


def require_uniform_grid(values, name: str, rtol: float = 1e-6) -> float:
    """
    Check that `values` is a finite, increasing, evenly spaced 1D grid.

    The velocity binning and the wavelength WCS both take one spacing
    and apply it everywhere, so an uneven grid is silently misread. A
    decreasing grid is worse: the bin edges come out in descending order,
    every `>= low & < high` test fails, and the emission measure is zero
    everywhere.

    Parameters
    ----------
    values : np.ndarray or u.Quantity
        1D grid of bin centres.
    name : str
        Name to use in the error message.
    rtol : float, optional
        Largest allowed offset of any value from the evenly spaced grid
        through the first and last, relative to the spacing. The default
        allows the rounding in `np.arange` and `np.linspace` but not a grid
        built unevenly on purpose. Single-precision rounding, common in
        simulation output, is allowed whatever `rtol`, as long as it is
        under half a spacing.

    Returns
    -------
    float
        Mean spacing, in the units of `values`, which single-precision
        rounding in one cell does not skew.
    """
    plain = np.asarray(getattr(values, "value", values), dtype=float)

    if plain.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {plain.ndim} dimensions.")
    if plain.size < 2:
        raise ValueError(f"{name} must have at least 2 elements, "
                         f"got {plain.size}.")

    # Comparisons with NaN are always false, so a NaN or inf in the grid can
    # slip past the spacing checks below and come back as the spacing.
    non_finite = np.flatnonzero(~np.isfinite(plain))
    if non_finite.size:
        first_bad = int(non_finite[0])
        raise ValueError(f"{name} must be finite, got {plain[first_bad]} "
                         f"at index {first_bad}.")

    if plain[1] <= plain[0] or plain[-1] <= plain[0]:
        raise ValueError(
            f"{name} must increase. Bin edges are built by stepping out from "
            f"the first value, so a decreasing grid produces edges in "
            f"descending order and every bin ends up empty."
        )
    step = float(plain[-1] - plain[0]) / (plain.size - 1)

    # Single-precision values are off by up to a few units in the last place,
    # about a part in 1e7 of the largest grid value, however fine the
    # spacing. If that reaches half a spacing, single precision cannot hold
    # the grid, as a value that far off lies in its neighbour's cell. Below
    # that, every value stays in its own cell and the grid still increases.
    rounding = _SINGLE_PRECISION_ULPS * np.finfo(np.float32).eps * np.abs(plain).max()
    tolerance = max(rtol * step, rounding if rounding < step / 2 else 0.0)
    offsets = plain - (plain[0] + np.arange(plain.size) * step)
    worst = int(np.argmax(np.abs(offsets)))
    if abs(offsets[worst]) > tolerance:
        raise ValueError(
            f"{name} must be evenly spaced. Element {worst} is {plain[worst]:.10g}, "
            f"{offsets[worst] / step:+.3g} of a spacing ({step:.6g}) from where an evenly "
            f"spaced grid with the same first and last elements puts it. ECLIPSE uses "
            f"one spacing for every bin edge and for the wavelength CDELT, so an uneven "
            f"grid puts emission in the wrong bins and writes wrong wavelength "
            f"coordinates. Resample onto a uniform grid first."
        )

    return step


def velocity_grid(vel_res: u.Quantity, vel_lim: u.Quantity,
                  names: tuple = ("vel_res", "vel_lim")) -> u.Quantity:
    """
    Velocity bin centres `vel_res` apart, to at least `vel_lim` each way, in cm/s.

    The bins step out from zero both ways, so a static plasma is centred on
    a bin whatever the two values. A grid built from -vel_lim has no bin at
    zero when the limit is not a whole number of steps, and a static line
    would come out Doppler shifted. `names` are the names used in the error
    for a value that is not a positive velocity.
    """
    values = []
    for value, name in zip((vel_res, vel_lim), names):
        value = u.Quantity(value)
        if not value.unit.is_equivalent(u.km / u.s) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive velocity, got {value}.")
        values.append(value.to_value(u.cm / u.s))
    res, lim = values
    # Whole steps to reach lim, rounding up only when lim is not a whole
    # number of steps, whatever rounding error the division has.
    steps = int(np.ceil(np.round(lim / res, 9)))
    return np.arange(-steps, steps + 1) * res * (u.cm / u.s)


def velocity_centers_to_edges(vel_grid: np.ndarray) -> np.ndarray:
    """
    Convert velocity grid centers to bin edges.

    Parameters
    ----------
    vel_grid : np.ndarray
        1D array of velocity centers.  Must be evenly spaced and increasing.

    Returns
    -------
    np.ndarray
        1D array of velocity bin edges (length = len(vel_grid) + 1).
    """
    dv = require_uniform_grid(vel_grid, "vel_grid")

    return np.concatenate([
        [vel_grid[0] - 0.5 * dv],
        vel_grid[:-1] + 0.5 * dv,
        [vel_grid[-1] + 0.5 * dv]
    ])

def require_downsample_divides(shape: tuple[int, ...], downsample: int) -> None:
    """
    Check that `downsample` divides every dimension of `shape`.

    Downsampling keeps every `downsample`-th cell and gives each kept cell
    `downsample` times the voxel size. If a dimension is not a multiple of
    the factor, the last kept cell stands for fewer cells than that, so the
    domain comes out too large, as does the emission measure when that axis
    is the line of sight.

    Parameters
    ----------
    shape : tuple of int
        Cube dimensions, in any order.
    downsample : int
        Downsampling factor.
    """
    if (isinstance(downsample, bool) or not isinstance(downsample, (int, np.integer))
            or downsample < 1):
        raise ValueError(f"The downsampling factor must be a whole number of 1 or "
                         f"more, got {downsample!r}.")
    uneven = [n for n in shape if n % downsample]
    if uneven:
        raise ValueError(
            f"--downsample {downsample} does not divide the cube shape "
            f"{tuple(shape)}: {uneven} not a multiple of {downsample}. Choose "
            f"a factor that divides every dimension."
        )


def _unpickle(handle):
    """
    Object from an older version's pickle, which may name the package's former modules.

    Those load through `euvst_response`, whose warning to change an import is
    not for the reader of a file, so it is not shown here.
    """
    import dill

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The euvst_response package is now called")
        return dill.load(handle)
