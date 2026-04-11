"""
Spectral fitting functions for Gaussian line profile analysis.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import astropy.units as u
import astropy.constants as const
from ndcube import NDCube
from scipy.optimize import curve_fit, OptimizeWarning
from joblib import Parallel, delayed
from tqdm import tqdm
from .utils import gaussian, multi_gaussian, tqdm_joblib


# ---------------------------------------------------------------------------
#  Fit configuration
# ---------------------------------------------------------------------------

@dataclass
class FitComponent:
    """One spectral line component in a multi-Gaussian fit."""
    wavelength: u.Quantity
    tie_center: Optional[int] = None   # index of component whose centre this is tied to
    tie_width: Optional[int] = None    # index of component whose width this is tied to


@dataclass
class FitConfig:
    """Configuration for multi-component Gaussian fitting.

    When *components* is empty (or has one entry with no ties) the fitter
    falls back to the fast single-Gaussian path.

    The ``primary_component`` index selects which component's centre and
    width are used for velocity/width analysis downstream.
    """
    components: List[FitComponent] = field(default_factory=list)
    primary_component: int = 0

    @property
    def n_components(self) -> int:
        return max(len(self.components), 1)

    @property
    def is_single(self) -> bool:
        return len(self.components) <= 1

    @property
    def n_full_params(self) -> int:
        """Total stored parameters: 3 per component + 1 background."""
        return 3 * self.n_components + 1

    # Indices into the full parameter vector for the primary component
    @property
    def idx_peak(self) -> int:
        return 3 * self.primary_component

    @property
    def idx_center(self) -> int:
        return 3 * self.primary_component + 1

    @property
    def idx_sigma(self) -> int:
        return 3 * self.primary_component + 2


# ---------------------------------------------------------------------------
#  Single-component helpers
# ---------------------------------------------------------------------------

def _guess_params(wv: np.ndarray, prof: np.ndarray) -> list:
    """Guess initial parameters for Gaussian fit."""
    back = prof.min()
    prof_c = prof - back
    prof_c[prof_c < 0] = 0
    peak = prof_c.max()
    centre = wv[np.nanargmax(prof_c)]
    if peak == 0:
        sigma = (wv.max() - wv.min()) / 10
    else:
        # Simple FWHM estimate
        half_max = 0.5 * peak
        indices = np.where(prof_c >= half_max)[0]
        if len(indices) > 1:
            fwhm = wv[indices[-1]] - wv[indices[0]]
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma = (wv.max() - wv.min()) / 10
    return [peak, centre, sigma, back]


def _fit_one(wv: np.ndarray, prof: np.ndarray) -> np.ndarray:
    """Fit single spectrum with Gaussian."""
    p0 = _guess_params(wv, prof)
    # Bounds: peak >= 0, others unconstrained
    lower = [0, -np.inf, -np.inf, -np.inf]
    upper = [np.inf, np.inf, np.inf, np.inf]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt, _ = curve_fit(gaussian, wv, prof, p0=p0, bounds=(lower, upper))
            return popt
        except:
            return np.array(p0)


# ---------------------------------------------------------------------------
#  Multi-component helpers
# ---------------------------------------------------------------------------

def _build_multi_model(fit_config: FitConfig):
    """Build a tied multi-Gaussian model suitable for *curve_fit*.

    Returns
    -------
    model_func : callable
        ``model_func(x, *free_params) -> y``
    free_to_full : callable
        ``free_to_full(free_params) -> full_params`` (length 3*N+1)
    n_free : int
        Number of free (optimised) parameters.
    free_indices : list[int]
        Mapping from free-parameter position to full-parameter index.
    """
    nc = fit_config.n_components
    n_full = fit_config.n_full_params  # 3*nc + 1

    # Determine which full-parameter indices are free vs tied
    free_indices: list[int] = []
    # tie_spec[full_idx] = (source_full_idx, offset)  or None
    tie_spec: dict[int, tuple[int, float] | None] = {}

    for i, comp in enumerate(fit_config.components):
        base = 3 * i
        # peak is always free
        free_indices.append(base)
        tie_spec[base] = None

        # centre
        if comp.tie_center is not None:
            src = comp.tie_center
            offset = (comp.wavelength - fit_config.components[src].wavelength).to(u.cm).value
            tie_spec[base + 1] = (3 * src + 1, offset)
        else:
            free_indices.append(base + 1)
            tie_spec[base + 1] = None

        # width
        if comp.tie_width is not None:
            src = comp.tie_width
            tie_spec[base + 2] = (3 * src + 2, 0.0)
        else:
            free_indices.append(base + 2)
            tie_spec[base + 2] = None

    # background is always free
    bg_idx = n_full - 1
    free_indices.append(bg_idx)
    tie_spec[bg_idx] = None

    n_free = len(free_indices)
    # Reverse map: full_idx -> position in free array (for tied referencing)
    full_to_free = {fi: pos for pos, fi in enumerate(free_indices)}

    def free_to_full(free_params):
        full = np.empty(n_full)
        # First fill free slots
        for pos, fi in enumerate(free_indices):
            full[fi] = free_params[pos]
        # Then fill tied slots
        for fi in range(n_full):
            spec = tie_spec.get(fi)
            if spec is not None:
                src_fi, offset = spec
                full[fi] = full[src_fi] + offset
        return full

    def model_func(x, *free_params):
        full = free_to_full(free_params)
        return multi_gaussian(x, *full, n_components=nc)

    return model_func, free_to_full, n_free, free_indices


def _guess_multi_params(wv: np.ndarray, prof: np.ndarray, fit_config: FitConfig,
                        free_indices: list[int]) -> np.ndarray:
    """Generate an initial guess for the *free* parameters of a multi-component fit."""
    nc = fit_config.n_components
    back = prof.min()
    prof_c = prof - back
    prof_c[prof_c < 0] = 0

    # Primary component gets the dominant peak guess
    peak = prof_c.max()
    if peak == 0:
        sigma = (wv.max() - wv.min()) / 10
    else:
        half_max = 0.5 * peak
        indices = np.where(prof_c >= half_max)[0]
        if len(indices) > 1:
            fwhm = wv[indices[-1]] - wv[indices[0]]
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma = (wv.max() - wv.min()) / 10

    # Build full initial guess, then extract free parameters
    full_guess = np.zeros(fit_config.n_full_params)
    for i, comp in enumerate(fit_config.components):
        base = 3 * i
        wl_cm = comp.wavelength.to(u.cm).value
        if i == fit_config.primary_component:
            full_guess[base] = peak
            full_guess[base + 1] = wl_cm
            full_guess[base + 2] = sigma
        else:
            # Secondary component: fraction of primary peak, same sigma
            full_guess[base] = peak * 0.15
            full_guess[base + 1] = wl_cm
            full_guess[base + 2] = sigma
    full_guess[-1] = back

    # Extract only the free parameters
    return full_guess[free_indices]


def _fit_one_multi(wv: np.ndarray, prof: np.ndarray, fit_config: FitConfig,
                   model_func, free_to_full, n_free: int,
                   free_indices: list[int]) -> np.ndarray:
    """Fit a single spectrum with the tied multi-component model.

    Returns the *full* parameter vector (length 3*N+1).
    """
    p0_free = _guess_multi_params(wv, prof, fit_config, free_indices)
    # Build bounds: peak parameters >= 0, others unconstrained
    nc = fit_config.n_components
    peak_full_indices = set(range(0, 3 * nc, 3))
    lower = [-np.inf] * n_free
    for pos, fi in enumerate(free_indices):
        if fi in peak_full_indices:
            lower[pos] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt_free, _ = curve_fit(model_func, wv, prof, p0=p0_free,
                                     bounds=(lower, np.inf), maxfev=5000)
        except Exception:
            popt_free = p0_free
    return free_to_full(popt_free)


# ---------------------------------------------------------------------------
#  Public fitting entry point
# ---------------------------------------------------------------------------

def fit_cube_gauss(signal_cube: NDCube, n_jobs: int = -1,
                   fit_config: FitConfig | None = None) -> tuple[np.ndarray, list[u.Unit]]:
    """
    Fit Gaussian(s) to every (slit x wavelength) spectrum.

    Parameters
    ----------
    signal_cube : NDCube
        Data cube with shape (n_scan, n_slit, n_lambda).
    n_jobs : int
        Joblib parallelism (-1 = all cores).
    fit_config : FitConfig, optional
        Multi-component configuration.  When *None* or single-component,
        the original single-Gaussian fitter is used.

    Returns
    -------
    data_array : ndarray
        Shape ``(n_scan, n_slit, n_params)`` where *n_params* is 4 for a
        single component (``[peak, centre, sigma, background]``) or
        ``3*N+1`` for *N* components
        (``[peak0, centre0, sigma0, ..., background]``).
    units_list : list of Unit
        One unit per parameter.
    """
    n_scan, n_slit, _ = signal_cube.shape
    wv = signal_cube.axis_world_coords(2)[0].cgs  # wavelength axis

    # --- single-component fast path ---
    if fit_config is None or fit_config.is_single:
        def _fit_block(spec_block):
            results = np.empty((spec_block.shape[0], 4))
            for i in range(spec_block.shape[0]):
                results[i] = _fit_one(wv.value, spec_block[i])
            return results

        with tqdm_joblib(tqdm(total=n_scan, desc="Fit chunks", leave=False)):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_block)(signal_cube.data[i]) for i in range(n_scan)
            )

        data_array = np.stack(results, axis=0)
        units_list = [signal_cube.unit, wv.unit, wv.unit, signal_cube.unit]
        return data_array, units_list

    # --- multi-component path ---
    model_func, free_to_full, n_free, free_indices = _build_multi_model(fit_config)
    n_params = fit_config.n_full_params

    def _fit_block_multi(spec_block):
        results = np.empty((spec_block.shape[0], n_params))
        for i in range(spec_block.shape[0]):
            results[i] = _fit_one_multi(wv.value, spec_block[i], fit_config,
                                        model_func, free_to_full, n_free,
                                        free_indices)
        return results

    with tqdm_joblib(tqdm(total=n_scan, desc="Fit chunks (multi)", leave=False)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_block_multi)(signal_cube.data[i]) for i in range(n_scan)
        )

    data_array = np.stack(results, axis=0)

    # Build units list: [signal, wl, wl, signal, wl, wl, ..., signal]
    units_list = []
    for _ in range(fit_config.n_components):
        units_list.extend([signal_cube.unit, wv.unit, wv.unit])
    units_list.append(signal_cube.unit)  # background

    return data_array, units_list


def velocity_from_fit(fit_arr: u.Quantity | np.ndarray, wl0: u.Quantity,
                      n_jobs: int = -1, fit_config: FitConfig | None = None) -> u.Quantity:
    """
    Convert fitted line centres to LOS velocity.
    Works with either a Quantity array or an object-dtype array whose
    elements are Quantities. Uses joblib.Parallel for speed.
    """
    idx = 1 if (fit_config is None or fit_config.is_single) else fit_config.idx_center
    centres_raw = fit_arr[..., idx]  # (n_scan, n_slit)
    # Ensure we have a pure Quantity array
    if isinstance(centres_raw, u.Quantity):
        centres = centres_raw.to(wl0.unit)
    else:  # object array of Quantity scalars
        get_val = np.vectorize(lambda q: q.to_value(wl0.unit))
        centres = u.Quantity(get_val(centres_raw), wl0.unit)

    n_scan = centres.shape[0]

    def _one_row(i):
        return ((centres[i] - wl0) / wl0 * const.c).to(u.cm / u.s).value

    with tqdm_joblib(tqdm(total=n_scan, desc="Velocity calc", leave=False)):
        v_val = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_one_row)(i) for i in range(n_scan)
            )
        )

    v = v_val * (u.cm / u.s)
    return v


def width_from_fit(fit_arr: u.Quantity | np.ndarray, n_jobs: int = -1,
                   fit_config: FitConfig | None = None) -> u.Quantity:
    """
    Extract fitted line widths (sigma) from fit results.
    """
    idx = 2 if (fit_config is None or fit_config.is_single) else fit_config.idx_sigma
    widths_raw = fit_arr[..., idx]  # (n_scan, n_slit)
    # Ensure we have a pure Quantity array
    if isinstance(widths_raw, u.Quantity):
        widths = widths_raw
    else:  # object array of Quantity scalars
        get_val = np.vectorize(lambda q: q.value)
        get_unit = widths_raw.flat[0].unit  # Get unit from first element
        widths = u.Quantity(get_val(widths_raw), get_unit)
    
    return widths


def analyse(fits_all: u.Quantity | np.ndarray, v_true: u.Quantity, wl0: u.Quantity,
            fit_config: FitConfig | None = None) -> dict:
    """
    Monte-Carlo velocity statistics given pre-computed ground truth.
    """
    v_all = velocity_from_fit(fits_all, wl0, fit_config=fit_config)
    w_all = width_from_fit(fits_all, fit_config=fit_config)
    return {
        "v_mean": v_all.mean(axis=0),
        "v_std":  v_all.std(axis=0),
        "v_err":  v_true - v_all.mean(axis=0),
        "v_samples": v_all,
        "v_true":    v_true,
        "w_mean": w_all.mean(axis=0),
        "w_std":  w_all.std(axis=0),
        "w_samples": w_all,
    }
