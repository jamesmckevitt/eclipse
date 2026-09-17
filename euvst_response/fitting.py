"""
Spectral fitting functions for Gaussian line profile analysis.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Optional, overload
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
    # Label for this component in the results. Defaults to its rest
    # wavelength, e.g. "195.1190 Angstrom".
    name: Optional[str] = None


def default_component_name(wavelength: u.Quantity) -> str:
    """The name a component gets in the results when it is not given one."""
    return f"{wavelength.to_value(u.AA):.4f} Angstrom"


@dataclass
class FitConfig:
    """Configuration for Gaussian fitting.

    With no *components* this configures the single-Gaussian fit, and every
    setting applies except ``primary_component`` and
    ``constrain_positive_intensity``. Otherwise there must be at least two
    components: one on its own is not a blend, and the fitter would ignore its
    wavelength and ties and fit one free Gaussian.

    The ``primary_component`` index selects which component's centre and
    width are used for velocity/width analysis downstream.
    """
    components: List[FitComponent] = field(default_factory=list)
    primary_component: int = 0
    constrain_positive_intensity: bool = False
    backend: str | None = None  # None = auto (scipy)
    # Iterations the optimiser may take before it gives up and returns
    # wherever it has got to. Counted in iterations rather than function
    # evaluations so that it means the same thing whichever backend runs, and
    # so that it does not quietly shrink as components are added. EISPAC uses
    # 2000 for the same job; fits here converge in tens.
    max_iter: int = 1000
    # Divide by n - 1 rather than n in the standard deviation over Monte Carlo
    # iterations (Bessel's correction). Off by default, which keeps results
    # comparable with runs made before it existed.
    bessel_correction: bool = False
    # Keep every iteration's fitted parameters in the results, not only their
    # statistics. The results grow by about n_iter times the fit arrays.
    save_iterations: bool = False

    def __post_init__(self):
        if len(self.components) == 1:
            raise ValueError(
                "fitting.components has one entry. Multi-component fitting "
                "needs at least two; for a single Gaussian leave components "
                "out, and the other fitting settings still apply."
            )
        if self.backend not in (None, "scipy", "mpfit"):
            raise ValueError(
                f"Unknown fitting backend '{self.backend}'. "
                f"Supported values: 'scipy', 'mpfit', or omit for auto."
            )
        if (isinstance(self.max_iter, bool)
                or not isinstance(self.max_iter, (int, np.integer))
                or self.max_iter < 1):
            raise ValueError(
                f"fitting.max_iter must be a positive integer, got "
                f"{self.max_iter!r}."
            )
        for key in ("bessel_correction", "save_iterations"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(
                    f"fitting.{key} must be true or false, got "
                    f"{getattr(self, key)!r}."
                )
        if self.components:
            n = len(self.components)
            if not 0 <= self.primary_component < n:
                raise ValueError(
                    f"fitting.primary_component is {self.primary_component}, "
                    f"but there are {n} components, numbered from 0."
                )
            for idx, comp in enumerate(self.components):
                for key in ("tie_center", "tie_width", "amplitude_greater_than"):
                    other = getattr(comp, key)
                    if other is not None and (
                            isinstance(other, bool)
                            or not isinstance(other, (int, np.integer))
                            or not 0 <= other < n or other == idx):
                        raise ValueError(
                            f"fitting.components[{idx}].{key} is {other!r}. It "
                            f"has to be the index of another component, from "
                            f"0 to {n - 1}."
                        )
                if comp.name is not None and (
                        not isinstance(comp.name, str) or not comp.name.strip()):
                    raise ValueError(
                        f"fitting.components[{idx}].name must be a non-empty "
                        f"string, got {comp.name!r}."
                    )
            names = [default_component_name(c.wavelength) if c.name is None
                     else c.name for c in self.components]
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise ValueError(
                    f"Fitted components need distinct names, because the "
                    f"results are stored by name. Repeated: {repeated}. Give "
                    f"the components a 'name'."
                )
        else:
            # Neither setting has anything to act on without components, so
            # accepting them would look effective and change nothing.
            if self.primary_component != 0:
                raise ValueError(
                    "fitting.primary_component picks one of the components, "
                    "and there are none."
                )
            if self.constrain_positive_intensity:
                raise ValueError(
                    "fitting.constrain_positive_intensity is only applied to "
                    "multi-component fits, and there are no components."
                )

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


def _fit_one_mpfit(wv: np.ndarray, prof: np.ndarray,
                   max_iter: int = FitConfig.max_iter) -> tuple[np.ndarray, bool]:
    """Fit single spectrum with Gaussian, using mpfit.

    The same model, initial guess and fallback as :func:`_fit_one`, with the
    width held positive as on the multi-component mpfit path.  *max_iter* is
    passed to mpfit, which counts iterations directly.

    Returns the parameters and whether the fit succeeded; see
    :func:`_mpfit_succeeded`.
    """
    p0 = np.asarray(_guess_params(wv, prof), dtype=float)
    parinfo = [
        {"value": p0[0]},
        {"value": p0[1]},
        {"value": p0[2], "limited": [1, 0], "limits": [1e-30, 0.0]},
        {"value": p0[3]},
    ]
    functkw = {"x": wv, "y": prof, "n_components": 1}
    try:
        result = mpfit(_mpfit_residuals, p0, parinfo=parinfo,
                       functkw=functkw, quiet=True, maxiter=max_iter)
        if result.status > 0:
            params = np.asarray(result.params, dtype=float)
            return params, _mpfit_succeeded(result.status, params)
    except Exception:
        pass
    return p0, False


def _mpfit_succeeded(status: int, params: np.ndarray) -> bool:
    """Whether an mpfit result counts as a successful fit.

    Status 5 means mpfit ran out of iterations.  scipy raises in that case
    and the fit falls back to its initial guess, so it is counted as failed
    here too, even though mpfit hands back wherever it had got to.
    """
    return status > 0 and status != 5 and bool(np.all(np.isfinite(params)))


def _fit_one(wv: np.ndarray, prof: np.ndarray,
             max_iter: int = FitConfig.max_iter) -> tuple[np.ndarray, bool]:
    """Fit single spectrum with Gaussian.

    *max_iter* is an iteration count.  curve_fit uses lm here, since there are
    no bounds, and lm counts every residual call against maxfev including the
    one per parameter that builds each finite-difference Jacobian, so an
    iteration costs len(p0) + 1 evaluations.

    Returns the parameters and whether the fit succeeded.  A failed fit,
    including one that runs out of iterations, returns the initial guess.
    """
    p0 = _guess_params(wv, prof)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt, _ = curve_fit(gaussian, wv, prof, p0=p0,
                                maxfev=max_iter * (len(p0) + 1))
            return popt, bool(np.all(np.isfinite(popt)))
        except:
            return np.array(p0), False


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
    # tie_spec[full_idx] = (source_full_idx, factor), applied multiplicatively
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
            # Tied centres share one velocity, and one velocity moves a line by
            # an amount proportional to its wavelength, so the tie is the ratio
            # of the two rest wavelengths rather than their separation. Holding
            # the separation fixed instead would hold the shift in wavelength
            # constant across the window, which is a different velocity for
            # every component: over an 8 Angstrom window near 195 Angstrom the
            # end components disagree by (8 / 195) v, some 12 km/s at 300 km/s.
            # The ratio is dimensionless, so it is the same number whether the
            # model is evaluated in Angstrom or in cm.
            factor = float(comp.wavelength.to(u.cm).value
                           / fit_config.components[src].wavelength.to(u.cm).value)
            tie_spec[base + 1] = (3 * src + 1, factor)
        else:
            free_indices.append(base + 1)

        # sigma
        if comp.tie_width is not None:
            src = comp.tie_width
            # Widths stay tied as equals. Thermal broadening does scale with
            # wavelength, but the instrumental width that dominates these
            # windows does not, so this is left as it was.
            tie_spec[base + 2] = (3 * src + 2, 1.0)
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
        for fi, (src_fi, factor) in tie_spec.items():
            full[fi] = full[src_fi] * factor
        return full

    def model_func(x, *free_params):
        full = free_to_full_A(free_params)
        return multi_gaussian(x, *full, n_components=nc)

    return model_func, free_to_full_A, n_free, free_indices, ratio_spec, bounds, has_bounds


def _fit_one_scipy_multi(wv_cm: np.ndarray, prof: np.ndarray,
                         fit_config: FitConfig,
                         model_func, free_to_full_A,
                         free_indices: list[int],
                         ratio_spec: dict, bounds,
                         has_bounds: bool) -> tuple[np.ndarray, bool]:
    """Fit one spectrum with scipy curve_fit (multi-component, A scaling).

    *wv_cm* is the wavelength axis in **cm** (CGS).  The fit is performed
    in Angstrom internally, then the result is converted back to cm.

    Returns the parameters and whether the fit succeeded.  A failed fit
    returns the initial guess.
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

    # Loosen tolerances for both LM and TRF methods.  The spectra are
    # noisy so 1e-4 tolerances introduce negligible velocity error
    # (<0.01 km/s at good S/N, ~0.15 km/s at very faint signals).
    # Cap function evaluations as a safety net (fits converge in ~50).
    # Explicitly select 'trf' when bounds are active, 'lm' otherwise;
    # each method uses a different keyword for max evaluations.
    #
    # fit_config.max_iter is an iteration count, so convert it to whatever
    # each method counts. MINPACK's lm counts every residual call against
    # maxfev, including the n_free calls that build each forward-difference
    # Jacobian, so an iteration costs n_free + 1 and the cap has to scale with
    # the problem or it shrinks as components are added. least_squares' trf
    # counts only its own residual calls and reports Jacobian work separately
    # in njev, so there its cap is already an iteration count.
    n_free = len(free_indices)
    max_iter = fit_config.max_iter
    if has_bounds:
        fit_kwargs: dict = {
            "method": "trf", "max_nfev": max_iter,
            # Amplitudes run to ~1e11 while sigmas are ~0.03 Angstrom, so the
            # free parameters span some thirteen orders of magnitude. Take
            # steps in variables normalised by the Jacobian rather than in
            # the raw ones; lm applies equivalent scaling internally, which
            # is why only the bounded path has to be told. On the blends
            # tested this changed neither the fitted velocity nor the
            # evaluation count, so it is insurance against worse-conditioned
            # windows rather than a fix for an observed failure.
            "x_scale": "jac",
            "ftol": 1e-4, "gtol": 1e-4, "xtol": 1e-4,
        }
    else:
        fit_kwargs: dict = {
            "method": "lm", "maxfev": max_iter * (n_free + 1),
            "ftol": 1e-4, "gtol": 1e-4, "xtol": 1e-4,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt_free_A, _ = curve_fit(model_func, wv_A, prof, p0=p0_free_A,
                                       bounds=bounds, **fit_kwargs)
            succeeded = bool(np.all(np.isfinite(popt_free_A)))
        except Exception:
            popt_free_A = p0_free_A
            succeeded = False

    # Reconstruct full A vector, then convert centres & sigmas back to cm
    full_A = free_to_full_A(popt_free_A)
    full_cm = full_A.copy()
    for i in range(fit_config.n_components):
        full_cm[3 * i + 1] /= CM_TO_A  # centre
        full_cm[3 * i + 2] /= CM_TO_A  # sigma

    return full_cm, succeeded


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
            # Ratio of rest wavelengths, so that the one free centre means one
            # velocity for every tied component; see _build_scipy_multi.
            factor = float(comp.wavelength.to(u.cm).value
                           / fit_config.components[src].wavelength.to(u.cm).value)
            # mpfit tie expression references the parameter array p
            centre_info["tied"] = f"p[{3 * src + 1}] * {factor!r}"
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
        # Width of the feature *containing the peak*, not of everything above
        # half maximum.
        #
        # Taking the first and last pixel above half max measures one line in
        # a single-line window and the span of the whole blend in a
        # multi-component one. Every component then starts several pixels
        # wide, and the fit can settle there: on a seven-component EIS window
        # the widths came out at 89 mA and 16 mA either side of a 28 mA
        # instrumental floor, with a residual 45 percent of the peak and the
        # requested line fitted at effectively zero. Walking outwards from the
        # peak until the profile drops below half maximum measures the peak
        # feature alone, and is identical for a single line.
        half_max = 0.5 * peak
        above = prof_c >= half_max
        peak_idx = int(np.nanargmax(prof_c))
        lo = hi = peak_idx
        while lo > 0 and above[lo - 1]:
            lo -= 1
        while hi < len(above) - 1 and above[hi + 1]:
            hi += 1
        if hi > lo:
            sigma = (wv[hi] - wv[lo]) / (2 * np.sqrt(2 * np.log(2)))
        else:
            # Unresolved: the feature is one pixel wide, so take that as an
            # upper bound on the width rather than a tenth of the window.
            sigma = float(np.median(np.diff(wv))) if len(wv) > 1 else \
                (wv.max() - wv.min()) / 10

    # Estimate ONE global shift of the primary centre, and start every other
    # centre at its rest wavelength scaled by the same factor, which is the
    # relation the tie itself enforces.
    #
    # The peak pixel belongs to whichever component dominates the window,
    # which is not necessarily the primary. Starting the primary's centre on
    # it therefore displaces the primary by the gap between the two, and since
    # every tied centre is a fixed multiple of the primary, the whole comb
    # starts displaced by that gap. The optimiser does not reliably recover: a
    # noiseless 7-component EIS window whose primary sat 0.14 Angstrom from
    # the dominant line returned a 239 km/s centroid for lines that were at
    # rest by construction, and moving the primary to a different component
    # changed the answer to 451 km/s, each time by very nearly the offset
    # between that component and the brightest one.
    #
    # Match the whole comb against the profile instead. Attributing the peak
    # pixel to the nearest rest wavelength would only resolve the shift while
    # it stays below half the spacing to the neighbouring component, which is
    # a small velocity for a close blend: Fe XII 195.119 and 195.179 are
    # 0.060 Angstrom apart, so anything beyond about 46 km/s picks the wrong
    # component and displaces the comb by a whole spacing again, and the
    # synthesis default admits +/- 300 km/s.
    #
    # Scoring every trial shift by the total profile height under the shifted
    # comb uses the *spacing pattern*, which one pixel does not carry. Trial
    # shifts run over the range that keeps the comb inside the observed
    # window, sampled at the wavelength grid itself, so no resolution is
    # invented. With a single component the score is just the profile sampled
    # at each grid point, so the best shift puts the centre on the brightest
    # pixel exactly as before.
    rest_wl = np.array([c.wavelength.to(u.cm).value
                        for c in fit_config.components])
    ref_wl = float(rest_wl[fit_config.primary_component])
    if peak > 0:
        # A trial shift moves the primary centre to ref_wl + shift, and every
        # other component to its own rest wavelength times the same factor. The
        # range is the one that keeps the whole comb inside the observed window;
        # with all components at one wavelength it reduces to shifting the comb
        # bodily, as the additive version did.
        lo = float(wv.min()) * ref_wl / float(rest_wl.min()) - ref_wl
        hi = float(wv.max()) * ref_wl / float(rest_wl.max()) - ref_wl
        step = float(np.median(np.diff(wv))) if len(wv) > 1 else 0.0
        if hi > lo and step > 0:
            trial = np.arange(lo, hi + 0.5 * step, step)
            score = np.zeros(trial.size)
            for r in rest_wl:
                score += np.interp(r * (ref_wl + trial) / ref_wl, wv, prof_c,
                                   left=0.0, right=0.0)
            # Known limit: if the primary component carries no flux, every
            # alias that puts *some* component on the one visible line scores
            # alike, and the data cannot say which component produced it. With
            # amplitudes free, all the flux in one component at one alias fits
            # exactly as well as all of it in another at the next, so no
            # scoring rule resolves it and the comb can start a spacing away.
            # The velocity asked for in that case is the velocity of a line
            # that is not there, so there is nothing to recover.
            shift = float(trial[int(np.argmax(score))])
        else:
            # The comb is wider than the window, so no shift keeps all of it
            # inside. Fall back to the brightest pixel and the component
            # nearest to it, read as a Doppler factor.
            peak_wl = float(wv[np.nanargmax(prof_c)])
            nearest = float(rest_wl[np.argmin(np.abs(rest_wl - peak_wl))])
            shift = ref_wl * (peak_wl / nearest - 1.0)
    else:
        shift = 0.0

    doppler = (ref_wl + shift) / ref_wl
    full_guess = np.zeros(fit_config.n_full_params)
    for i, comp in enumerate(fit_config.components):
        base = 3 * i
        full_guess[base] = peak if i == fit_config.primary_component else peak * 0.15
        full_guess[base + 1] = rest_wl[i] * doppler
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
                   ratio_params: dict | None = None) -> tuple[np.ndarray, bool]:
    """Fit a single spectrum with mpfit.

    Returns the *full* parameter vector (length ``3*N + 1``) with
    absolute amplitudes (ratio parameters are converted back), and whether
    the fit succeeded; see :func:`_mpfit_succeeded`.
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
                       functkw=functkw, quiet=True,
                       maxiter=fit_config.max_iter)
        if result.status > 0:
            out = np.asarray(result.params, dtype=float)
            # Convert ratios back to absolute amplitudes
            if ratio_params:
                for child_idx, parent_idx in ratio_params.items():
                    out[child_idx] = out[parent_idx] * result.params[child_idx]
            return out, _mpfit_succeeded(result.status, out)
    except Exception:
        pass

    return p0_orig, False  # fall back to initial guess (absolute amplitudes)


# ---------------------------------------------------------------------------
#  Public fitting entry point
# ---------------------------------------------------------------------------

@overload
def fit_cube_gauss(signal_cube: NDCube, n_jobs: int = -1,
                   fit_config: FitConfig | None = None, *,
                   return_failed: Literal[False] = False
                   ) -> tuple[np.ndarray, list[u.Unit]]: ...


@overload
def fit_cube_gauss(signal_cube: NDCube, n_jobs: int = -1,
                   fit_config: FitConfig | None = None, *,
                   return_failed: Literal[True]
                   ) -> tuple[np.ndarray, list[u.Unit], np.ndarray]: ...


def fit_cube_gauss(signal_cube: NDCube, n_jobs: int = -1,
                   fit_config: FitConfig | None = None, *,
                   return_failed: bool = False):
    """
    Fit Gaussian(s) to every (slit x wavelength) spectrum.

    Parameters
    ----------
    signal_cube : NDCube
        Data cube with shape (n_slit, n_scan, n_lambda).
    n_jobs : int
        Joblib parallelism (-1 = all cores).
    fit_config : FitConfig, optional
        Fit configuration.  When *None* or without components, a single
        Gaussian is fitted, with the configuration's ``max_iter`` and
        ``backend`` when one is given.
    return_failed : bool, optional
        Also return which fits failed.  Keyword-only.  Default False.

    Returns
    -------
    data_array : ndarray
        Shape ``(n_slit, n_scan, n_params)`` where *n_params* is 4 for a
        single component (``[peak, centre, sigma, background]``) or
        ``3*N+1`` for *N* components
        (``[peak0, centre0, sigma0, ..., background]``).
    units_list : list of Unit
        One unit per parameter.
    failed : ndarray of bool
        Shape ``(n_slit, n_scan)``, True where the fit failed, including where
        the optimiser ran out of ``max_iter``.  Those pixels hold the initial
        guess (or, for mpfit, wherever it had got to).  Only returned when
        *return_failed* is True.
    """
    n_slit, n_scan, _ = signal_cube.shape
    wv = signal_cube.axis_world_coords(2)[0].cgs  # wavelength axis

    # The iteration limit applies to every path. Without a fitting block there
    # is no FitConfig to carry it, so fall back to the same default.
    max_iter = FitConfig.max_iter if fit_config is None else fit_config.max_iter

    # --- single-component fast path ---
    if fit_config is None or fit_config.is_single:
        use_mpfit = fit_config is not None and fit_config.backend == "mpfit"
        fit_one = _fit_one_mpfit if use_mpfit else _fit_one

        def _fit_block(spec_block):
            results = np.empty((spec_block.shape[0], 4))
            succeeded = np.empty(spec_block.shape[0], dtype=bool)
            for i in range(spec_block.shape[0]):
                results[i], succeeded[i] = fit_one(wv.value, spec_block[i], max_iter)
            return results, succeeded

        with tqdm_joblib(tqdm(total=n_slit, desc="Fit chunks", leave=False)):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_block)(signal_cube.data[i]) for i in range(n_slit)
            )

        data_array = np.stack([r[0] for r in results], axis=0)
        failed = ~np.stack([r[1] for r in results], axis=0)
        units_list = [signal_cube.unit, wv.unit, wv.unit, signal_cube.unit]
        if return_failed:
            return data_array, units_list, failed
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
         free_indices, ratio_spec, bounds, has_bounds) = _build_scipy_multi(fit_config)

        def _fit_block_multi(spec_block):
            results = np.empty((spec_block.shape[0], n_params))
            succeeded = np.empty(spec_block.shape[0], dtype=bool)
            for i in range(spec_block.shape[0]):
                results[i], succeeded[i] = _fit_one_scipy_multi(
                    wv.value, spec_block[i], fit_config,
                    model_func, free_to_full, free_indices,
                    ratio_spec, bounds, has_bounds)
            return results, succeeded

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
            succeeded = np.empty(spec_block.shape[0], dtype=bool)
            for i in range(spec_block.shape[0]):
                results[i], succeeded[i] = _fit_one_multi(
                    wv.value, spec_block[i], fit_config,
                    parinfo_template, ratio_params)
            return results, succeeded

    with tqdm_joblib(tqdm(total=n_slit, desc="Fit chunks (multi)", leave=False)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_block_multi)(signal_cube.data[i]) for i in range(n_slit)
        )

    data_array = np.stack([r[0] for r in results], axis=0)
    failed = ~np.stack([r[1] for r in results], axis=0)

    # Build units list: [signal, wl, wl, signal, wl, wl, ..., signal]
    units_list = []
    for _ in range(fit_config.n_components):
        units_list.extend([signal_cube.unit, wv.unit, wv.unit])
    units_list.append(signal_cube.unit)  # background

    if return_failed:
        return data_array, units_list, failed
    return data_array, units_list


# ---------------------------------------------------------------------------
#  Results by component
# ---------------------------------------------------------------------------

def component_names(fit_config: FitConfig | None,
                    rest_wavelength: u.Quantity) -> list[str]:
    """
    Names of the fitted components, in fit order.

    Parameters
    ----------
    fit_config : FitConfig or None
        The fit configuration.  A single-Gaussian fit has one component.
    rest_wavelength : u.Quantity
        Rest wavelength of the line, which names a single-Gaussian fit.

    Returns
    -------
    list of str
    """
    if fit_config is None or fit_config.is_single:
        return [default_component_name(rest_wavelength)]
    return [default_component_name(c.wavelength) if c.name is None else c.name
            for c in fit_config.components]


def spectral_pixel_width(cube: NDCube) -> u.Quantity:
    """Width of one spectral pixel of a cube with wavelength on its last axis."""
    wavelength = u.Quantity(cube.axis_world_coords(2)[0]).to(u.cm)
    return wavelength[1] - wavelength[0]


def fit_quantities(fit_data: np.ndarray, units: list,
                   wavelength_step: u.Quantity, rest_wavelength: u.Quantity,
                   fit_config: FitConfig | None = None) -> dict:
    """
    Intensity, velocity and width of every fitted component, by name.

    Parameters
    ----------
    fit_data : np.ndarray
        Fitted parameters with the parameter axis last, as from
        :func:`fit_cube_gauss`.
    units : list of Unit
        One unit per parameter, as from :func:`fit_cube_gauss`.
    wavelength_step : u.Quantity
        Width of one spectral pixel.
    rest_wavelength : u.Quantity
        Rest wavelength of a single-Gaussian fit.  With components, each uses
        its own.
    fit_config : FitConfig, optional
        The fit configuration; None for a single Gaussian.

    Returns
    -------
    dict
        ``components`` maps each component name to ``intensity``,
        ``velocity`` and ``width``, and ``background`` holds the fitted
        background.  Each is a Quantity shaped like *fit_data* without its
        last axis.  ``intensity`` is the fitted Gaussian summed over spectral
        pixels, so it is in counts, like the total signal maps.  ``velocity``
        is relative to that component's rest wavelength, and ``width`` is the
        Gaussian sigma.
    """
    names = component_names(fit_config, rest_wavelength)
    if fit_config is None or fit_config.is_single:
        rest = [rest_wavelength]
    else:
        rest = [c.wavelength for c in fit_config.components]

    components = {}
    for i, (name, lam0) in enumerate(zip(names, rest)):
        peak = fit_data[..., 3 * i] * units[3 * i]
        centre = fit_data[..., 3 * i + 1] * units[3 * i + 1]
        sigma = fit_data[..., 3 * i + 2] * units[3 * i + 2]
        pixels = (sigma / wavelength_step).to(u.dimensionless_unscaled)
        components[name] = {
            "intensity": np.sqrt(2 * np.pi) * peak * pixels * u.pix,
            "velocity": ((centre - lam0) / lam0 * const.c).to(u.km / u.s),
            "width": sigma.to(u.AA),
        }
    return {"components": components,
            "background": fit_data[..., -1] * units[-1]}


def summarise_fits(fit_data: np.ndarray, failed: np.ndarray, units: list,
                   wavelength_step: u.Quantity, rest_wavelength: u.Quantity,
                   fit_config: FitConfig | None = None) -> dict:
    """
    Statistics of the Monte Carlo fits, per parameter and per component.

    Failed fits are left out of every mean and standard deviation.  A pixel
    whose fits all failed has NaN statistics.  The intensity statistics are
    taken over each iteration's intensity, so they account for the peak and
    width moving together, which the per-parameter statistics cannot.

    Parameters
    ----------
    fit_data : np.ndarray
        Shape ``(n_iter, n_slit, n_scan, n_params)``.
    failed : np.ndarray of bool
        Shape ``(n_iter, n_slit, n_scan)``, True where a fit failed.
    units, wavelength_step, rest_wavelength, fit_config
        As for :func:`fit_quantities`.  ``fit_config.bessel_correction``
        makes the standard deviations divide by n - 1 rather than n, and
        ``fit_config.save_iterations`` keeps *fit_data* and *failed*.

    Returns
    -------
    dict
        ``first_fit_data`` (the first iteration, failed or not),
        ``mean_data``, ``std_data`` and ``units``, per parameter as before,
        and:

        - ``components``: for each component name, its ``rest_wavelength``,
          ``tied`` (the names of the components its ``velocity`` and
          ``width`` are tied to, or None), and ``intensity``, ``velocity``
          and ``width``, each with ``first``, ``mean`` and ``std`` maps.
          ``first`` is NaN where the first fit failed.
        - ``background``: ``first``, ``mean`` and ``std`` maps.
        - ``primary_component``: the primary component's name.
        - ``failed_fits``: the number of failed fits in each pixel.
        - ``n_iterations``: the number of fits in each pixel.
        - ``bessel_correction``: whether the standard deviations divide by
          n - 1.
        - ``iterations``: ``fit_data`` and ``failed``, only when
          ``fit_config.save_iterations`` is set.
    """
    bessel = fit_config is not None and fit_config.bessel_correction
    ddof = 1 if bessel else 0

    def _stats(values, leave_out):
        kept = np.where(leave_out, np.nan, values)
        with warnings.catch_warnings():
            # A pixel with no successful fit, or only one under Bessel's
            # correction, has no statistic, and NaN is the right answer.
            warnings.simplefilter("ignore", RuntimeWarning)
            return (kept[0], np.nanmean(kept, axis=0),
                    np.nanstd(kept, axis=0, ddof=ddof))

    _, mean_data, std_data = _stats(fit_data, failed[..., np.newaxis])

    quantities = fit_quantities(fit_data, units, wavelength_step,
                                rest_wavelength, fit_config)
    names = list(quantities["components"])
    configured = ([None] if fit_config is None or fit_config.is_single
                  else fit_config.components)

    def _summary(q):
        first, mean, std = _stats(q.value, failed)
        return {"first": first * q.unit, "mean": mean * q.unit,
                "std": std * q.unit}

    components = {}
    for name, comp, values in zip(names, configured,
                                  quantities["components"].values()):
        components[name] = {
            "rest_wavelength": rest_wavelength if comp is None else comp.wavelength,
            "tied": {
                "velocity": (None if comp is None or comp.tie_center is None
                             else names[comp.tie_center]),
                "width": (None if comp is None or comp.tie_width is None
                          else names[comp.tie_width]),
            },
            **{key: _summary(values[key])
               for key in ("intensity", "velocity", "width")},
        }

    summary = {
        "first_fit_data": fit_data[0],
        "mean_data": mean_data,
        "std_data": std_data,
        "units": units,
        "components": components,
        "background": _summary(quantities["background"]),
        "primary_component": names[0 if fit_config is None or fit_config.is_single
                                   else fit_config.primary_component],
        "failed_fits": np.count_nonzero(failed, axis=0),
        "n_iterations": fit_data.shape[0],
        "bessel_correction": bessel,
    }
    if fit_config is not None and fit_config.save_iterations:
        summary["iterations"] = {"fit_data": fit_data, "failed": failed}
    return summary


def ground_truth_summary(cube: NDCube, fit_config: FitConfig | None = None,
                         n_jobs: int = -1) -> dict:
    """
    Fit the noiseless cube, and give each component's velocity and width.

    Parameters
    ----------
    cube : NDCube
        The cube before the instrument, with ``rest_wav`` in its metadata.
    fit_config : FitConfig, optional
        The fit configuration; None for a single Gaussian.
    n_jobs : int
        Joblib parallelism.

    Returns
    -------
    dict
        ``fit_truth_data`` and ``fit_truth_units`` as from
        :func:`fit_cube_gauss`, ``failed`` (True where the fit failed), and
        ``components``: for each component name, ``velocity`` and ``width``
        maps, NaN where the fit failed.  There is no intensity, because the
        cube is in the units of the synthesis rather than in counts.
    """
    data, units, failed = fit_cube_gauss(cube, n_jobs=n_jobs,
                                         fit_config=fit_config,
                                         return_failed=True)
    quantities = fit_quantities(data, units, spectral_pixel_width(cube),
                                cube.meta["rest_wav"], fit_config)
    components = {
        name: {key: np.where(failed, np.nan, values[key].value) * values[key].unit
               for key in ("velocity", "width")}
        for name, values in quantities["components"].items()
    }
    return {"fit_truth_data": data, "fit_truth_units": units,
            "failed": failed, "components": components}


def velocity_from_fit(fit_arr: u.Quantity | np.ndarray, wl0: u.Quantity,
                      n_jobs: int = -1, fit_config: FitConfig | None = None) -> u.Quantity:
    """
    Convert fitted line centres to LOS velocity.
    Works with either a Quantity array or an object-dtype array whose
    elements are Quantities. Uses joblib.Parallel for speed.
    """
    idx = 1 if (fit_config is None or fit_config.is_single) else fit_config.idx_center
    centres_raw = fit_arr[..., idx]  # (n_slit, n_scan)
    # Ensure we have a pure Quantity array
    if isinstance(centres_raw, u.Quantity):
        centres = centres_raw.to(wl0.unit)
    else:  # object array of Quantity scalars
        get_val = np.vectorize(lambda q: q.to_value(wl0.unit))
        centres = u.Quantity(get_val(centres_raw), wl0.unit)

    n_rows = centres.shape[0]

    def _one_row(i):
        return ((centres[i] - wl0) / wl0 * const.c).to(u.cm / u.s).value

    with tqdm_joblib(tqdm(total=n_rows, desc="Velocity calc", leave=False)):
        v_val = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_one_row)(i) for i in range(n_rows)
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
    widths_raw = fit_arr[..., idx]  # (n_slit, n_scan)
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
