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
from .extern.mpfit import mpfit


# ---------------------------------------------------------------------------
#  Fit configuration
# ---------------------------------------------------------------------------

@dataclass
class FitComponent:
    """One spectral line component in a multi-Gaussian fit."""
    wavelength: u.Quantity
    tie_center: Optional[int] = None   # index of component whose centre this is tied to
    tie_width: Optional[int] = None    # index of component whose width this is tied to
    amplitude_greater_than: Optional[int] = None  # index of component this must be brighter than


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
    constrain_positive_intensity: bool = False
    backend: str | None = None  # None = auto (scipy unless positive-intensity needed)

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt, _ = curve_fit(gaussian, wv, prof, p0=p0)
            return popt
        except:
            return np.array(p0)


# ---------------------------------------------------------------------------
#  Multi-component helpers  (scipy back-end -- fast, default)
# ---------------------------------------------------------------------------

def _build_scipy_multi(fit_config: FitConfig):
    """Build a tied multi-Gaussian model for *curve_fit*.

    Internally the model works in **Angstrom** for wavelength-related
    parameters (centres, sigmas) so that all free parameters are within
    a few orders of magnitude of each other.  This avoids the severe
    ill-conditioning that occurs when the fitter operates in CGS.

    When no amplitude ordering constraint is active the Levenberg-Marquardt
    (``lm``) method is used (fast, no bounds).  When
    ``amplitude_greater_than`` is set on any component, the constrained
    component's amplitude is reparametrised as a **ratio** element of [0, 1] of
    the parent amplitude and *curve_fit* switches to the Trust Region
    Reflective (``trf``) method which supports bounds.

    Returns
    -------
    model_func : callable
        ``model_func(x_angstrom, *free_params_angstrom) -> y``
    free_to_full_A : callable
        ``free_to_full_A(free_params_A) -> full_params_A`` (length 3*N+1)
    n_free : int
    free_indices : list[int]
    ratio_spec : dict
        ``{free_position: parent_full_amplitude_index}`` for ratio params.
    bounds : tuple
        Bounds suitable for ``curve_fit(..., bounds=...)``.
    """
    nc = fit_config.n_components
    n_full = fit_config.n_full_params  # 3*nc + 1

    # Pre-scan: identify which components have their amplitude constrained
    # to be smaller than another.
    # amp_ratio_child[j] = i means amp_j = amp_i * ratio  (ratio in [0,1]).
    amp_ratio_child: dict[int, int] = {}
    for i, comp in enumerate(fit_config.components):
        if comp.amplitude_greater_than is not None:
            j = comp.amplitude_greater_than
            amp_ratio_child[j] = i

    free_indices: list[int] = []
    # tie_spec[full_idx] = (source_full_idx, offset_angstrom)
    tie_spec: dict[int, tuple[int, float]] = {}
    # ratio_spec[free_position] = parent_full_amplitude_index
    ratio_spec: dict[int, int] = {}

    for i, comp in enumerate(fit_config.components):
        base = 3 * i
        # peak - always free; mark as ratio if constrained
        if i in amp_ratio_child:
            ratio_spec[len(free_indices)] = 3 * amp_ratio_child[i]
        free_indices.append(base)

        # centre
        if comp.tie_center is not None:
            src = comp.tie_center
            offset = float((comp.wavelength
                            - fit_config.components[src].wavelength).to(u.Angstrom).value)
            tie_spec[base + 1] = (3 * src + 1, offset)
        else:
            free_indices.append(base + 1)

        # sigma
        if comp.tie_width is not None:
            src = comp.tie_width
            tie_spec[base + 2] = (3 * src + 2, 0.0)
        else:
            free_indices.append(base + 2)

    # background -- always free
    free_indices.append(n_full - 1)

    n_free = len(free_indices)

    # Build per-parameter bounds (default: unbounded -> LM method).
    # Any non-default bound triggers TRF method.
    lower = np.full(n_free, -np.inf)
    upper = np.full(n_free, np.inf)
    for fp in ratio_spec:
        lower[fp] = 0.0
        upper[fp] = 1.0
    # Enforce positive amplitudes during fitting (not post-hoc clipping)
    if fit_config.constrain_positive_intensity:
        for pos, fi in enumerate(free_indices):
            if fi % 3 == 0 and fi < 3 * nc and pos not in ratio_spec:
                lower[pos] = 0.0
    has_bounds = ratio_spec or fit_config.constrain_positive_intensity
    bounds = (lower, upper) if has_bounds else (-np.inf, np.inf)

    def free_to_full_A(free_params):
        full = np.empty(n_full)
        for pos, fi in enumerate(free_indices):
            full[fi] = free_params[pos]
        # Convert ratio params -> absolute amplitudes
        for fp, parent_idx in ratio_spec.items():
            child_idx = free_indices[fp]
            full[child_idx] = full[parent_idx] * free_params[fp]
        for fi, (src_fi, offset) in tie_spec.items():
            full[fi] = full[src_fi] + offset
        return full

    def model_func(x, *free_params):
        full = free_to_full_A(free_params)
        return multi_gaussian(x, *full, n_components=nc)

    return model_func, free_to_full_A, n_free, free_indices, ratio_spec, bounds


def _fit_one_scipy_multi(wv_cm: np.ndarray, prof: np.ndarray,
                         fit_config: FitConfig,
                         model_func, free_to_full_A,
                         free_indices: list[int],
                         ratio_spec: dict, bounds) -> np.ndarray:
    """Fit one spectrum with scipy curve_fit (multi-component, A scaling).

    *wv_cm* is the wavelength axis in **cm** (CGS).  The fit is performed
    in Angstrom internally, then the result is converted back to cm.
    """
    CM_TO_A = 1e8

    # Initial guess in cm (from shared helper)
    p0_full_cm = _guess_multi_params(wv_cm, prof, fit_config)

    # Convert centres & sigmas to A
    p0_full_A = p0_full_cm.copy()
    for i in range(fit_config.n_components):
        p0_full_A[3 * i + 1] *= CM_TO_A  # centre
        p0_full_A[3 * i + 2] *= CM_TO_A  # sigma

    p0_free_A = p0_full_A[free_indices]

    # Convert absolute amplitudes to ratios for constrained params
    for free_pos, parent_idx in ratio_spec.items():
        parent_val = p0_full_A[parent_idx]
        if parent_val > 0:
            p0_free_A[free_pos] = p0_full_A[free_indices[free_pos]] / parent_val
        else:
            p0_free_A[free_pos] = 0.15

    wv_A = wv_cm * CM_TO_A

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt_free_A, _ = curve_fit(model_func, wv_A, prof, p0=p0_free_A,
                                       bounds=bounds, maxfev=5000)
        except Exception:
            popt_free_A = p0_free_A

    # Reconstruct full A vector, then convert centres & sigmas back to cm
    full_A = free_to_full_A(popt_free_A)
    full_cm = full_A.copy()
    for i in range(fit_config.n_components):
        full_cm[3 * i + 1] /= CM_TO_A  # centre
        full_cm[3 * i + 2] /= CM_TO_A  # sigma

    return full_cm


# ---------------------------------------------------------------------------
#  Multi-component helpers  (mpfit back-end)
# ---------------------------------------------------------------------------

def _build_parinfo(fit_config: FitConfig, p0: np.ndarray,
                   ratio_params: dict | None = None) -> list[dict]:
    """Build mpfit *parinfo* list from a FitConfig and initial guess.

    The full parameter vector has layout
    ``[peak0, centre0, sigma0, peak1, centre1, sigma1, ..., background]``.

    Parameters
    ----------
    fit_config : FitConfig
        Multi-component configuration (ties, constraints ...).
    p0 : np.ndarray
        Initial-guess vector (length ``3*N + 1``).
    ratio_params : dict, optional
        ``{child_full_amp_idx: parent_full_amp_idx}`` identifying
        amplitude parameters that are fitted as a ratio in [0, 1] of
        their parent amplitude.

    Returns
    -------
    parinfo : list[dict]
        One dict per parameter, suitable for ``mpfit(..., parinfo=...)``.
    """
    nc = fit_config.n_components
    if ratio_params is None:
        ratio_params = {}
    parinfo: list[dict] = []

    for i, comp in enumerate(fit_config.components):
        base = 3 * i

        # --- peak (intensity) ---
        if base in ratio_params:
            parent_amp = p0[ratio_params[base]]
            ratio = p0[base] / parent_amp if parent_amp > 0 else 0.15
            peak_info: dict = {"value": ratio,
                               "limited": [1, 1], "limits": [0.0, 1.0]}
        else:
            peak_info = {"value": p0[base]}
            if fit_config.constrain_positive_intensity:
                peak_info["limited"] = [1, 0]
                peak_info["limits"] = [0.0, 0.0]
        parinfo.append(peak_info)

        # --- centre ---
        centre_info: dict = {"value": p0[base + 1]}
        if comp.tie_center is not None:
            src = comp.tie_center
            offset = float((comp.wavelength
                            - fit_config.components[src].wavelength).to(u.cm).value)
            # mpfit tie expression references the parameter array p
            centre_info["tied"] = f"p[{3 * src + 1}] + {offset!r}"
        parinfo.append(centre_info)

        # --- sigma (width) ---
        sigma_info: dict = {"value": p0[base + 2]}
        # Width must be strictly positive (avoid divide-by-zero in Gaussian)
        sigma_info["limited"] = [1, 0]
        sigma_info["limits"] = [1e-30, 0.0]
        if comp.tie_width is not None:
            src = comp.tie_width
            sigma_info["tied"] = f"p[{3 * src + 2}]"
        parinfo.append(sigma_info)

    # --- background ---
    bg_info: dict = {"value": p0[-1]}
    parinfo.append(bg_info)

    return parinfo


def _guess_multi_params(wv: np.ndarray, prof: np.ndarray,
                        fit_config: FitConfig) -> np.ndarray:
    """Generate an initial guess for the *full* parameter vector."""
    back = float(prof.min())
    prof_c = prof - back
    prof_c[prof_c < 0] = 0

    peak = float(prof_c.max())
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

    # Use the peak-intensity pixel as the initial centre for the primary
    peak_wl = wv[np.nanargmax(prof_c)]

    full_guess = np.zeros(fit_config.n_full_params)
    for i, comp in enumerate(fit_config.components):
        base = 3 * i
        wl_cm = comp.wavelength.to(u.cm).value
        if i == fit_config.primary_component:
            full_guess[base] = peak
            full_guess[base + 1] = peak_wl  # peak pixel, not rest wavelength
            full_guess[base + 2] = sigma
        else:
            full_guess[base] = peak * 0.15
            full_guess[base + 1] = wl_cm
            full_guess[base + 2] = sigma
    full_guess[-1] = back
    return full_guess


def _mpfit_residuals(p, fjac=None, x=None, y=None, n_components=1,
                     ratio_params=None):
    """Residual function in the form mpfit expects.

    Must return ``[status, residuals]`` where *status* is 0 for success.
    """
    p_eval = np.array(p, dtype=float)
    if ratio_params:
        for child_idx, parent_idx in ratio_params.items():
            p_eval[child_idx] = p_eval[parent_idx] * p[child_idx]
    model = multi_gaussian(x, *p_eval, n_components=n_components)
    return [0, y - model]


def _fit_one_multi(wv: np.ndarray, prof: np.ndarray,
                   fit_config: FitConfig,
                   parinfo_template: list[dict],
                   ratio_params: dict | None = None) -> np.ndarray:
    """Fit a single spectrum with mpfit.

    Returns the *full* parameter vector (length ``3*N + 1``) with
    absolute amplitudes (ratio parameters are converted back).
    """
    p0 = _guess_multi_params(wv, prof, fit_config)
    p0_orig = p0.copy()

    # Stamp current initial guesses into the parinfo dicts
    parinfo = []
    for i, pi in enumerate(parinfo_template):
        d = dict(pi)
        d["value"] = p0[i]
        parinfo.append(d)

    # Convert absolute amplitudes to ratios for constrained params
    if ratio_params:
        for child_idx, parent_idx in ratio_params.items():
            parent_amp = p0[parent_idx]
            ratio = p0[child_idx] / parent_amp if parent_amp > 0 else 0.15
            parinfo[child_idx]["value"] = ratio
            p0[child_idx] = ratio

    functkw = {"x": wv, "y": prof, "n_components": fit_config.n_components,
               "ratio_params": ratio_params}

    try:
        result = mpfit(_mpfit_residuals, p0, parinfo=parinfo,
                       functkw=functkw, quiet=True, maxiter=200)
        if result.status > 0:
            out = np.asarray(result.params, dtype=float)
            # Convert ratios back to absolute amplitudes
            if ratio_params:
                for child_idx, parent_idx in ratio_params.items():
                    out[child_idx] = out[parent_idx] * result.params[child_idx]
            return out
    except Exception:
        pass

    return p0_orig  # fall back to initial guess (absolute amplitudes)


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
    n_params = fit_config.n_full_params

    # Auto-select backend: scipy (default, fast) or mpfit (explicit only).
    # scipy supports all constraints (positive intensity, amplitude ratio)
    # via TRF bounds.  Use mpfit only when explicitly requested.
    use_mpfit = (fit_config.backend == "mpfit")

    if not use_mpfit:
        # scipy curve_fit -- LM when unconstrained, TRF when bounds active.
        (model_func, free_to_full, n_free,
         free_indices, ratio_spec, bounds) = _build_scipy_multi(fit_config)

        def _fit_block_multi(spec_block):
            results = np.empty((spec_block.shape[0], n_params))
            for i in range(spec_block.shape[0]):
                results[i] = _fit_one_scipy_multi(
                    wv.value, spec_block[i], fit_config,
                    model_func, free_to_full, free_indices,
                    ratio_spec, bounds)
            return results

    else:
        # mpfit with full parinfo (slower, supports hard bounds)
        # Build ratio_params for amplitude ordering constraint
        ratio_params: dict[int, int] = {}
        for i, comp in enumerate(fit_config.components):
            if comp.amplitude_greater_than is not None:
                j = comp.amplitude_greater_than
                ratio_params[3 * j] = 3 * i
        ratio_params = ratio_params or None

        p0_template = _guess_multi_params(wv.value, signal_cube.data[0, 0], fit_config)
        parinfo_template = _build_parinfo(fit_config, p0_template, ratio_params)

        def _fit_block_multi(spec_block):
            results = np.empty((spec_block.shape[0], n_params))
            for i in range(spec_block.shape[0]):
                results[i] = _fit_one_multi(wv.value, spec_block[i], fit_config,
                                            parinfo_template, ratio_params)
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
