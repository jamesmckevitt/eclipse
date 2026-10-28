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
from scipy.optimize import curve_fit, least_squares, OptimizeWarning
from joblib import Parallel, delayed
from tqdm import tqdm
from .utils import _fwhm_to_sigma, gaussian, multi_gaussian, pixel_mean_gaussians, tqdm_joblib
from .extern.mpfit import mpfit


def _line_profile(x, *params, n_components=1, pixel=None):
    """
    Fitted model: the Gaussians at wavelengths `x`, or with `pixel`, their
    mean over pixels that wide centred on `x`, as a detector records them.
    """
    if pixel is None:
        return multi_gaussian(x, *params, n_components=n_components)
    return pixel_mean_gaussians(x, *params, n_components=n_components, pixel=pixel)


def _poisson_residuals(counts: np.ndarray, model: np.ndarray) -> np.ndarray:
    """
    Residuals whose sum of squares is the Poisson deviance of counts about a model.

    For counts d and model m, each residual is
    sign(d - m) sqrt(2 (m - d + d ln(d / m))). Their sum of squares is minus
    twice the Poisson log-likelihood plus a constant, so a least-squares fit
    to them is the maximum-likelihood fit. The residual is zero at d = m and
    close to (d - m) / sqrt(d) near it, the residual weighted by the noise.

    A zero model has zero likelihood for counts above zero, so it is raised
    to the smallest positive float to keep the arithmetic finite; the fit
    then moves away from it.
    """
    d = np.asarray(counts, dtype=float)
    m = np.maximum(np.asarray(model, dtype=float), np.finfo(float).tiny)
    positive = d > 0
    log_ratio = np.log(np.where(positive, d, 1.0) / m)
    deviance = 2.0 * (m - d + np.where(positive, d * log_ratio, 0.0))
    return np.sign(d - m) * np.sqrt(np.maximum(deviance, 0.0))


# ---------------------------------------------------------------------------
#  Fit configuration
# ---------------------------------------------------------------------------

@dataclass
class FitComponent:
    """
    One line of a blend, fitted as one Gaussian.

    Parameters
    ----------
    wavelength : u.Quantity
        Rest wavelength of the line.
    tie_center : int, optional
        Index of a component whose velocity this line shares.
    tie_width : int, optional
        Index of a component whose width this line shares.
    amplitude_greater_than : int, optional
        Index of a component this line's amplitude is kept above.
    name : str, optional
        Name in the results. Default the rest wavelength, e.g.
        `"195.1190 Angstrom"`.
    velocity_limits : tuple of u.Quantity, optional
        Lowest and highest velocity the fit may give this line, measured
        from its rest wavelength, such as `(-60 * u.km / u.s, 60 * u.km / u.s)`.
        Not for a line whose centre is tied to another's. Default None, for
        no limits.
    width_limits : tuple of u.Quantity, optional
        Smallest and largest Gaussian sigma the fit may give this line, such
        as `(0.019 * u.AA, 0.051 * u.AA)`. Not for a line whose width is tied
        to another's. Default None, for no limits.
    """
    wavelength: u.Quantity
    tie_center: Optional[int] = None   # index of component whose centre this is tied to
    tie_width: Optional[int] = None    # index of component whose width this is tied to
    amplitude_greater_than: Optional[int] = None  # index of component this must be brighter than
    # Label for this component in the results. Defaults to its rest
    # wavelength, e.g. "195.1190 Angstrom".
    name: Optional[str] = None
    velocity_limits: Optional[tuple] = None
    width_limits: Optional[tuple] = None

    def __post_init__(self):
        # Catch a wavelength with no unit here rather than in the fit.
        if not (isinstance(self.wavelength, u.Quantity)
                and self.wavelength.unit.physical_type == "length"
                and np.isfinite(self.wavelength.value) and self.wavelength.value > 0):
            raise ValueError(f"A fitting component's wavelength must be the wavelength of a "
                             f"line, such as '195.119 AA'; got {self.wavelength!r}.")
        for key, kind, example in (("velocity_limits", u.km / u.s, "[-60 km/s, 60 km/s]"),
                                   ("width_limits", u.AA, "[0.019 AA, 0.051 AA]")):
            value = getattr(self, key)
            if value is None:
                continue
            try:
                low, high = (u.Quantity(v) for v in value)
            except (TypeError, ValueError):
                low = high = None
            if (low is None or not low.unit.is_equivalent(kind)
                    or not high.unit.is_equivalent(kind)
                    or not (np.isfinite(low.value) and np.isfinite(high.value))
                    or not low < high or (key == "width_limits" and low <= 0)):
                raise ValueError(
                    f"A fitting component's {key} must be two {kind.physical_type}s, the "
                    f"lower first{', both above zero' if key == 'width_limits' else ''}, "
                    f"such as {example}; got {value!r}.")
            setattr(self, key, (low.to(kind), high.to(kind)))

    def centre_limits(self) -> Optional[tuple]:
        """Lowest and highest centre in cm that `velocity_limits` allow, or None."""
        if self.velocity_limits is None:
            return None
        rest = self.wavelength.to_value(u.cm)
        c = const.c.to_value(u.km / u.s)
        return tuple(rest * (1 + v.to_value(u.km / u.s) / c) for v in self.velocity_limits)

    def sigma_limits(self) -> Optional[tuple]:
        """Smallest and largest sigma in cm that `width_limits` allow, or None."""
        if self.width_limits is None:
            return None
        return tuple(w.to_value(u.cm) for w in self.width_limits)


def default_component_name(wavelength: u.Quantity) -> str:
    """Name of a component in the results when it is not given one."""
    return f"{wavelength.to_value(u.AA):.4f} Angstrom"


@dataclass
class FitConfig:
    """
    Fit settings: one Gaussian, or several for a blend.

    Parameters
    ----------
    components : list of FitComponent
        Lines of a blend, two or more. Default empty, for one Gaussian.
    primary_component : int
        Index of the blend component whose velocity and width are reported.
        Default 0.
    constrain_positive_intensity : bool
        Keep every blend component's amplitude at zero or above. Default
        False.
    backend : str, optional
        Optimiser, `"scipy"` or `"mpfit"`. Default None, for scipy.
    max_iter : int
        Maximum optimiser iterations per spectrum. Default 1000.
    bessel_correction : bool
        Use n - 1 rather than n in the standard deviations over Monte Carlo
        iterations (Bessel's correction). Default False.
    save_iterations : bool
        Keep every iteration's fit in the results as well as the statistics.
        Default False.
    weighted : bool
        Weight each pixel of the DN spectra by one over its uncertainty
        squared, estimated from its own signal as eispac does, and fit the
        photon spectra by their Poisson likelihood. Default True.
    start : str
        Where each fit of a blend starts: `"search"` (default) slides the
        components together along the spectrum to where they best match it;
        `"rest"` starts every component at its rest wavelength, as eispac's
        templates do.
    """
    components: List[FitComponent] = field(default_factory=list)
    primary_component: int = 0
    constrain_positive_intensity: bool = False
    backend: str | None = None  # None = auto (scipy)
    # Iterations the optimiser may take before it gives up. Counted in
    # iterations, not function evaluations, so it means the same for both
    # backends and does not shrink as components are added. EISPAC uses 2000;
    # fits here converge in tens.
    max_iter: int = 1000
    # Use n - 1 rather than n in the standard deviation over Monte Carlo
    # iterations (Bessel's correction). Off by default, to keep results
    # comparable with earlier runs.
    bessel_correction: bool = False
    # Keep every iteration's fitted parameters in the results, not only their
    # statistics. The results grow by about n_iter times the fit arrays.
    save_iterations: bool = False
    # Weight the DN fits by each pixel's uncertainty and fit the photons by
    # their Poisson likelihood. Unweighted, the velocity scatter of a Gaussian
    # line is 1.24 times the photon-noise limit, because the wings that fix
    # its position count no more than its noisier peak. Results files from
    # before this setting are unweighted.
    weighted: bool = True
    # Where a blend's fit starts. A search can start a faint blend a whole
    # component spacing from the truth; starting at rest, as eispac does,
    # keeps it on the line that is there.
    start: str = "search"

    def __post_init__(self):
        if len(self.components) == 1:
            raise ValueError(
                "fitting.components has one entry. Multi-component fitting "
                "needs at least two; for a single Gaussian leave components "
                "out, and the other fitting settings still apply."
            )
        if self.start not in ("search", "rest"):
            raise ValueError(f"fitting.start must be 'search' or 'rest', got {self.start!r}.")
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
        for key in ("bessel_correction", "save_iterations", "constrain_positive_intensity",
                    "weighted"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(
                    f"fitting.{key} must be true or false, got "
                    f"{getattr(self, key)!r}."
                )
        if self.components:
            n = len(self.components)
            if (isinstance(self.primary_component, bool)
                    or not isinstance(self.primary_component, (int, np.integer))
                    or not 0 <= self.primary_component < n):
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
                for key, tie in (("velocity_limits", "tie_center"), ("width_limits", "tie_width")):
                    if getattr(comp, key) is not None and getattr(comp, tie) is not None:
                        raise ValueError(
                            f"fitting.components[{idx}] has {key} and {tie}. A tied "
                            f"component follows the one it is tied to, so give the limits "
                            f"to that one.")
                if comp.name is not None and (
                        not isinstance(comp.name, str) or not comp.name.strip()):
                    raise ValueError(
                        f"fitting.components[{idx}].name must be a non-empty "
                        f"string, got {comp.name!r}."
                    )
            for key in ("tie_center", "tie_width", "amplitude_greater_than"):
                for start in range(n):
                    index, chain = start, [start]
                    while getattr(self.components[index], key) is not None:
                        index = getattr(self.components[index], key)
                        if index in chain:
                            raise ValueError(
                                f"fitting.components {chain + [index]} make a loop of "
                                f"{key}, which no fit can meet. One component in a chain "
                                f"has to be left free.")
                        chain.append(index)
            fainter = [comp.amplitude_greater_than for comp in self.components
                       if comp.amplitude_greater_than is not None]
            shared = sorted({index for index in fainter if fainter.count(index) > 1})
            if shared:
                raise ValueError(
                    f"fitting.components {shared} are each held fainter than more than one "
                    f"component by amplitude_greater_than. A component can be held fainter "
                    f"than one other; chain them to hold it below several.")
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
            # Both settings need components; without them they would
            # silently do nothing.
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
            if self.start != "search":
                raise ValueError(
                    "fitting.start is only applied to multi-component fits, and there are "
                    "no components."
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


def _tie_root(fit_config: FitConfig, key: str, index: int) -> int:
    """
    Free component at the end of the chain of `key` ties from component `index`.

    Tying to a tied component is the same as tying to the one it is tied to:
    for centres the rest-wavelength ratios multiply out to the ratio with the
    last, and tied widths are all equal. FitConfig refuses loops, so the
    chain ends.
    """
    while getattr(fit_config.components[index], key) is not None:
        index = getattr(fit_config.components[index], key)
    return index


# Fraction of the primary's amplitude that every other component starts at.
_GUESS_FRACTION = 0.15


def _start_ratio(wv: np.ndarray, prof: np.ndarray, guess: np.ndarray,
                 child: int, parent: int) -> float:
    """
    Starting amplitude ratio of a component held fainter than its parent.

    The ratio is read from the spectrum above the guessed background, at the
    two guessed centres, interpolated between the finite pixels. The guess
    itself starts every component but the primary at the same fraction of
    the primary, so two components held one below the other would start
    equal, at the bound, and a primary held below another would start above
    it. If the spectrum makes the child the brighter, `_GUESS_FRACTION` is
    used, strictly inside the bounds. `child` and `parent` index the
    amplitudes in the full `guess`. A centre in NaN or infinite pixels at an
    end of the window has zero height, as background.
    """
    wv, prof = np.asarray(wv), np.asarray(prof)
    finite = np.isfinite(prof)
    if not finite.any():
        return _GUESS_FRACTION
    if len(wv) > 1 and wv[0] > wv[-1]:
        wv, prof, finite = wv[::-1], prof[::-1], finite[::-1]
    left = None if finite[0] else 0.0
    right = None if finite[-1] else 0.0
    wv, prof = _finite_pixels(wv, prof, finite)
    height = prof - guess[-1]

    def level(index):
        return float(np.interp(guess[index + 1], wv, height, left=left, right=right))

    child_level, parent_level = level(child), level(parent)
    if parent_level > 0 and 0 < child_level / parent_level < 1:
        return child_level / parent_level
    return _GUESS_FRACTION


def _finite_pixels(wv: np.ndarray, prof: np.ndarray, finite: np.ndarray) -> tuple:
    """Wavelengths and values where `finite` is True, the arrays themselves if it is everywhere."""
    if finite.all():
        return wv, prof
    return wv[finite], prof[finite]


def _fit_finite(fit, wv: np.ndarray, prof: np.ndarray, sigma: np.ndarray | None,
                kept: np.ndarray, min_pixels: int, start) -> tuple[np.ndarray, bool]:
    """
    Fit only the pixels of a spectrum marked in `kept`.

    `fit(wv, prof, sigma)` fits a spectrum and returns the parameters and
    whether it succeeded. `sigma` is None, or one uncertainty per pixel,
    kept or left out with its pixel. A spectrum with every pixel kept is
    fitted as it is. With fewer than `min_pixels` kept, the fit is not
    tried, and `start` of the whole window, with the pixels left out set to
    NaN, is returned as a failed fit.
    """
    if kept.all():
        return fit(wv, prof, sigma)
    if np.count_nonzero(kept) < min_pixels:
        return np.asarray(start(wv, np.where(kept, prof, np.nan)), dtype=float), False
    return fit(wv[kept], prof[kept], None if sigma is None else sigma[kept])


def _in_dependency_order(fainter_than: dict) -> list:
    """
    `(child, parent)` items of `fainter_than`, each parent before its children.

    A child's amplitude is fitted as a fraction of its parent's, so the
    parent's must be known first. This matters when a parent is itself the
    child of a component listed after it.
    """
    def depth(child):
        steps = 0
        while child in fainter_than:
            child = fainter_than[child]
            steps += 1
        return steps

    return sorted(fainter_than.items(), key=lambda item: depth(item[0]))


# ---------------------------------------------------------------------------
#  Single-component helpers
# ---------------------------------------------------------------------------

def _guess_params(wv: np.ndarray, prof: np.ndarray) -> list:
    """
    Starting peak, centre, width and background for a Gaussian fit.

    The guesses are made robust to noise. The background is the median, as
    most of a window is off the line; the minimum would sit below it by the
    largest noise excursion. The line is found in the spectrum smoothed over
    three pixels, so one noisy pixel cannot take it. Its width is that of
    the run of pixels around it above half its height; the first and last
    pixels above half maximum anywhere in the window would include noise far
    from the line and start the width several times too wide. With those
    cruder guesses, at a peak of two or three times the noise per pixel, the
    fits ending hundreds of km/s away and the scatter of the fitted
    velocities were both twice those from a good start.

    Only the finite pixels are used. A spectrum with no finite pixel starts
    at zero peak and background, at the first wavelength.
    """
    wv = np.asarray(wv)
    prof = np.asarray(prof, dtype=float)
    finite = np.isfinite(prof)
    if not finite.any():
        return [0.0, wv[0], (wv.max() - wv.min()) / 10, 0.0]
    wv, prof = _finite_pixels(wv, prof, finite)
    back = float(np.median(prof))
    height = prof - back
    smooth = np.convolve(height, np.ones(3) / 3, mode="same") if prof.size >= 3 else height
    peak_idx = int(np.nanargmax(smooth))
    peak = max(float(height[peak_idx]), float(smooth[peak_idx]), 0.0)
    centre = wv[peak_idx]
    if smooth[peak_idx] <= 0:
        sigma = (wv.max() - wv.min()) / 10
    else:
        above = smooth >= 0.5 * smooth[peak_idx]
        lo = hi = peak_idx
        while lo > 0 and above[lo - 1]:
            lo -= 1
        while hi < len(above) - 1 and above[hi + 1]:
            hi += 1
        # Every pixel of the run is a pixel wide, ends included, so add one step.
        step = abs(float(np.median(np.diff(wv)))) if len(wv) > 1 else 0.0
        fwhm = abs(float(wv[hi] - wv[lo])) + step
        sigma = (_fwhm_to_sigma(fwhm) if fwhm > 0
                 else (wv.max() - wv.min()) / 10)
    return [peak, centre, sigma, back]


def _single_fit_scales(wv: np.ndarray, prof: np.ndarray) -> tuple[float, float, float]:
    """
    Wavelength offset, wavelength step and signal scale for a single-Gaussian fit.

    In the cgs units of the spectra, a line's peak can be 1e13 and its centre
    2e-6 cm, thousands of widths from zero. The optimiser's convergence tests
    are then set by the largest numbers, and a change in the last bits of the
    data could move a fitted velocity by half a km/s. So the fit uses
    wavelengths in pixels from the middle of the window and the signal
    divided by its largest value, and the result is converted back. The
    model is the same, so the fit is the same, in better-conditioned numbers.
    """
    offset = 0.5 * (float(np.min(wv)) + float(np.max(wv)))
    step = float(np.median(np.abs(np.diff(wv)))) if len(wv) > 1 else 0.0
    if not (np.isfinite(step) and step > 0):
        step = 1.0
    peak = float(np.max(np.abs(prof)))
    scale = peak if np.isfinite(peak) and peak > 0 else 1.0
    return offset, step, scale


def _single_fit_back(params, offset: float, step: float, scale: float) -> np.ndarray:
    """Parameters in the units of :func:`_single_fit_scales`, converted to the spectrum's."""
    params = np.asarray(params, dtype=float)
    return np.array([params[0] * scale, params[1] * step + offset,
                     params[2] * step, params[3] * scale])


def _fit_one_mpfit(wv: np.ndarray, prof: np.ndarray,
                   max_iter: int = FitConfig.max_iter,
                   pixel: float | None = None,
                   sigma: np.ndarray | None = None,
                   poisson: bool = False) -> tuple[np.ndarray, bool]:
    """Fit one spectrum with a Gaussian using mpfit.

    Same model, initial guess and fallback as :func:`_fit_one`, with the
    width held positive as on the multi-component mpfit path. `max_iter` is
    passed to mpfit, which counts iterations directly. `sigma` is each
    pixel's uncertainty; the residuals are divided by it. With `poisson`, the
    spectrum is counts, fitted by their Poisson likelihood
    (:func:`_poisson_residuals`), with the peak and background held at zero
    or above so the model is too.

    Returns the parameters and whether the fit succeeded; see
    :func:`_mpfit_succeeded`.
    """
    offset, step, scale = _single_fit_scales(wv, prof)
    if poisson:
        # Leave counts unscaled, as their likelihood depends on their values.
        scale = 1.0
    x = (np.asarray(wv, dtype=float) - offset) / step
    y = np.asarray(prof, dtype=float) / scale
    p0 = np.asarray(_guess_params(x, y), dtype=float)
    parinfo = [
        {"value": p0[0]},
        {"value": p0[1]},
        {"value": p0[2], "limited": [1, 0], "limits": [1e-30, 0.0]},
        {"value": p0[3]},
    ]
    if poisson:
        for index in (0, 3):
            p0[index] = max(p0[index], 0.0)
            parinfo[index] = {"value": p0[index], "limited": [1, 0], "limits": [0.0, 0.0]}
    functkw = {"x": x, "y": y, "n_components": 1,
               "pixel": None if pixel is None else pixel / step,
               "err": None if sigma is None else np.asarray(sigma, dtype=float) / scale,
               "poisson": poisson}
    try:
        result = mpfit(_mpfit_residuals, p0, parinfo=parinfo,
                       functkw=functkw, quiet=True, maxiter=max_iter)
        if result.status > 0:
            params = _single_fit_back(result.params, offset, step, scale)
            return params, _mpfit_succeeded(result.status, params)
    except Exception:
        pass
    return _single_fit_back(p0, offset, step, scale), False


def _mpfit_succeeded(status: int, params: np.ndarray) -> bool:
    """Whether an mpfit result counts as a successful fit.

    Status 5 means mpfit ran out of iterations. scipy raises in that case
    and the fit falls back to its initial guess, so it counts as failed here
    too, even though mpfit returns its last parameters.
    """
    return status > 0 and status != 5 and bool(np.all(np.isfinite(params)))


def _fit_one_poisson(x: np.ndarray, counts: np.ndarray, p0, max_iter: int,
                     width: float | None, offset: float, step: float) -> tuple[np.ndarray, bool]:
    """
    Maximum Poisson likelihood fit of a single Gaussian; see :func:`_fit_one`.

    `x` is in the pixel units of :func:`_single_fit_scales`, with pixels
    `width` wide, and the counts are unscaled.
    """
    start = np.array(p0, dtype=float)
    start[[0, 3]] = np.maximum(start[[0, 3]], 0.0)
    lower = np.array([0.0, -np.inf, -np.inf, 0.0])

    def residuals(params):
        return _poisson_residuals(counts, _line_profile(x, *params, pixel=width))

    try:
        result = least_squares(residuals, start, bounds=(lower, np.inf), method="trf",
                               x_scale="jac", max_nfev=max_iter)
    except Exception:
        return _single_fit_back(start, offset, step, 1.0), False
    succeeded = result.status > 0 and bool(np.all(np.isfinite(result.x)))
    return _single_fit_back(result.x if succeeded else start, offset, step, 1.0), succeeded


def _fit_one(wv: np.ndarray, prof: np.ndarray,
             max_iter: int = FitConfig.max_iter,
             pixel: float | None = None,
             sigma: np.ndarray | None = None,
             poisson: bool = False) -> tuple[np.ndarray, bool]:
    """Fit one spectrum with a Gaussian.

    `max_iter` is an iteration count. curve_fit uses lm here, as there are no
    bounds. lm counts every residual call against maxfev, including one per
    parameter for each finite-difference Jacobian, so an iteration costs
    len(p0) + 1 evaluations. With `pixel`, the Gaussian is averaged over
    pixels that wide (:func:`_line_profile`). `sigma` is each pixel's
    uncertainty; each pixel is weighted by one over its square.

    With `poisson`, the spectrum is counts, fitted by their Poisson
    likelihood (:func:`_poisson_residuals`). The peak and background are
    held at zero or above so the model is too, which needs least_squares'
    trf method. That counts only its own residual calls, so `max_iter` is
    the cap as it is. Wavelengths are in pixels from the middle of the
    window, as in the other fits, but the counts are not divided by their
    largest value, as the likelihood depends on the counts themselves.

    Returns the parameters and whether the fit succeeded. A failed fit,
    including one that runs out of iterations, returns the initial guess.
    """
    offset, step, scale = _single_fit_scales(wv, prof)
    if poisson:
        # Leave counts unscaled, as their likelihood depends on their values.
        scale = 1.0
    x = (np.asarray(wv, dtype=float) - offset) / step
    y = np.asarray(prof, dtype=float) / scale
    p0 = _guess_params(x, y)
    if poisson:
        return _fit_one_poisson(x, y, p0, max_iter, None if pixel is None else pixel / step,
                                offset, step)
    model = gaussian if pixel is None else (
        lambda x, *params: pixel_mean_gaussians(x, *params, n_components=1,
                                                pixel=pixel / step))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        try:
            popt, _ = curve_fit(model, x, y, p0=p0,
                                sigma=None if sigma is None else np.asarray(sigma) / scale,
                                maxfev=max_iter * (len(p0) + 1))
            params = _single_fit_back(popt, offset, step, scale)
            return params, bool(np.all(np.isfinite(params)))
        except:
            return _single_fit_back(p0, offset, step, scale), False


# ---------------------------------------------------------------------------
#  Multi-component helpers  (scipy back-end -- fast, default)
# ---------------------------------------------------------------------------

def _build_scipy_multi(fit_config: FitConfig, pixel_angstrom: float | None = None):
    """Build a tied multi-Gaussian model for `curve_fit`.

    With `pixel_angstrom`, the Gaussians are averaged over pixels that wide
    (:func:`_line_profile`).

    The model takes centres and sigmas in Angstrom, so all free parameters
    are within a few orders of magnitude of each other. In cgs the fit is
    severely ill-conditioned.

    Without bounds, the Levenberg-Marquardt (`lm`) method is used (fast, no
    bounds). When any component sets `amplitude_greater_than`, the
    constrained component's amplitude is fitted as a ratio in [0, 1] of its
    parent's. That, or `constrain_positive_intensity`, sets bounds, and
    `curve_fit` then uses the Trust Region Reflective (`trf`) method.

    Returns
    -------
    model_func : callable
        `model_func(x_angstrom, *free_params_angstrom) -> y`
    free_to_full_A : callable
        `free_to_full_A(free_params_A) -> full_params_A` (length 3*N+1)
    n_free : int
    free_indices : list[int]
    ratio_spec : dict
        `{free_position: parent_full_amplitude_index}` for ratio params.
    bounds : tuple
        Bounds for `curve_fit(..., bounds=...)`.
    """
    nc = fit_config.n_components
    n_full = fit_config.n_full_params  # 3*nc + 1

    # Components whose amplitude is held below another's.
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
            src = _tie_root(fit_config, "tie_center", i)
            # Tied centres share one velocity, and a velocity shifts a line in
            # proportion to its wavelength, so the tie is the ratio of the rest
            # wavelengths, not their separation. A fixed separation would give
            # each component a different velocity: across an 8 Angstrom window
            # near 195 Angstrom the end components would differ by (8 / 195) v,
            # some 12 km/s at 300 km/s. The ratio is dimensionless, so it is the
            # same in Angstrom or cm.
            factor = float(comp.wavelength.to(u.cm).value
                           / fit_config.components[src].wavelength.to(u.cm).value)
            tie_spec[base + 1] = (3 * src + 1, factor)
        else:
            free_indices.append(base + 1)

        # sigma
        if comp.tie_width is not None:
            src = _tie_root(fit_config, "tie_width", i)
            # Tied widths are equal. Thermal broadening scales with wavelength,
            # but the instrumental width that dominates these windows does not.
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
    # Each component's limits on its centre and width, in Angstrom as the
    # model takes them.
    limited = False
    for pos, fi in enumerate(free_indices):
        if fi < 3 * nc and fi % 3 in (1, 2):
            comp = fit_config.components[fi // 3]
            limits = comp.centre_limits() if fi % 3 == 1 else comp.sigma_limits()
            if limits is not None:
                lower[pos], upper[pos] = (value * 1e8 for value in limits)
                limited = True
    has_bounds = ratio_spec or fit_config.constrain_positive_intensity or limited
    bounds = (lower, upper) if has_bounds else (-np.inf, np.inf)

    # Order the ratios so parents come before their children.
    by_child = {free_indices[fp]: fp for fp in ratio_spec}
    ratio_order = [(by_child[child], parent) for child, parent in
                   _in_dependency_order({free_indices[fp]: parent
                                         for fp, parent in ratio_spec.items()})]

    def free_to_full_A(free_params):
        full = np.empty(n_full)
        for pos, fi in enumerate(free_indices):
            full[fi] = free_params[pos]
        # Convert ratio params -> absolute amplitudes
        for fp, parent_idx in ratio_order:
            child_idx = free_indices[fp]
            full[child_idx] = full[parent_idx] * free_params[fp]
        for fi, (src_fi, factor) in tie_spec.items():
            full[fi] = full[src_fi] * factor
        return full

    def model_func(x, *free_params):
        full = free_to_full_A(free_params)
        return _line_profile(x, *full, n_components=nc, pixel=pixel_angstrom)

    return model_func, free_to_full_A, n_free, free_indices, ratio_spec, bounds, has_bounds


def _n_free_params(fit_config: FitConfig | None) -> int:
    """Number of parameters a fit adjusts: those not tied to another, and the background."""
    if fit_config is None or fit_config.is_single:
        return 4
    return _build_scipy_multi(fit_config)[2]


def _fit_one_scipy_multi(wv_cm: np.ndarray, prof: np.ndarray,
                         fit_config: FitConfig,
                         model_func, free_to_full_A,
                         free_indices: list[int],
                         ratio_spec: dict, bounds,
                         has_bounds: bool,
                         sigma: np.ndarray | None = None,
                         poisson: bool = False) -> tuple[np.ndarray, bool]:
    """Fit one spectrum with several Gaussians using scipy's curve_fit.

    `wv_cm` is the wavelength axis in cm. The fit works in Angstrom and the
    result is converted back to cm. `sigma` is each pixel's uncertainty;
    each pixel is weighted by one over its square. With `poisson`, the
    spectrum is counts, fitted by their Poisson likelihood
    (:func:`_poisson_residuals`) with least_squares' trf method.

    Returns the parameters and whether the fit succeeded. A failed fit
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

    # Start constrained amplitudes as ratios read from the spectrum.
    for free_pos, parent_idx in ratio_spec.items():
        p0_free_A[free_pos] = _start_ratio(wv_cm, prof, p0_full_cm,
                                           free_indices[free_pos], parent_idx)

    wv_A = wv_cm * CM_TO_A

    # Keep scipy's default tolerances. At 1e-4, lm stopped with a blend's
    # centres still over a km/s from convergence: the centres are some 195
    # Angstrom from zero, so a relative step of 1e-4 is 30 km/s. Cap function
    # evaluations as a safety net. Use 'trf' when bounds are active and 'lm'
    # otherwise; each takes a different keyword for the cap.
    #
    # fit_config.max_iter is an iteration count, so convert it to what each
    # method counts. MINPACK's lm counts every residual call against maxfev,
    # including the n_free calls for each forward-difference Jacobian, so an
    # iteration costs n_free + 1 and the cap must scale with the problem or
    # it shrinks as components are added. least_squares' trf counts only its
    # own residual calls (Jacobian work goes in njev), so its cap is already
    # an iteration count.
    n_free = len(free_indices)
    max_iter = fit_config.max_iter
    if has_bounds:
        fit_kwargs: dict = {
            "method": "trf", "max_nfev": max_iter,
            # Amplitudes reach ~1e11 while sigmas are ~0.03 Angstrom, some
            # thirteen orders of magnitude apart, so step in variables scaled
            # by the Jacobian. lm scales the same way internally, so only trf
            # needs telling. On the blends tested this changed neither the
            # fitted velocity nor the evaluation count; it guards against
            # worse-conditioned windows rather than fixing an observed failure.
            "x_scale": "jac",
        }
    else:
        fit_kwargs: dict = {
            "method": "lm", "maxfev": max_iter * (n_free + 1),
        }

    if poisson:
        # Hold the background and every amplitude not fitted as a ratio at
        # zero or above, so the model is too.
        lower = np.array(bounds[0], dtype=float) * np.ones(n_free)
        upper = np.array(bounds[1], dtype=float) * np.ones(n_free)
        n_full = fit_config.n_full_params
        for pos, index in enumerate(free_indices):
            if (index % 3 == 0 and pos not in ratio_spec) or index == n_full - 1:
                lower[pos] = max(lower[pos], 0.0)
        start = np.clip(p0_free_A, lower, upper)
        try:
            result = least_squares(lambda p: _poisson_residuals(prof, model_func(wv_A, *p)),
                                   start, bounds=(lower, upper), method="trf",
                                   x_scale="jac", max_nfev=max_iter)
            succeeded = result.status > 0 and bool(np.all(np.isfinite(result.x)))
            popt_free_A = result.x if succeeded else start
        except Exception:
            popt_free_A = start
            succeeded = False
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            try:
                popt_free_A, _ = curve_fit(model_func, wv_A, prof, p0=p0_free_A,
                                           sigma=sigma, bounds=bounds, **fit_kwargs)
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
    """Build the mpfit `parinfo` list from a FitConfig and initial guess.

    The full parameter vector is
    `[peak0, centre0, sigma0, peak1, centre1, sigma1, ..., background]`.

    Parameters
    ----------
    fit_config : FitConfig
        Multi-component configuration, with ties and constraints.
    p0 : np.ndarray
        Initial guess, length `3*N + 1`.
    ratio_params : dict, optional
        `{child_full_amp_idx: parent_full_amp_idx}` for amplitudes fitted as
        a ratio in [0, 1] of their parent's.

    Returns
    -------
    parinfo : list[dict]
        One dict per parameter, for `mpfit(..., parinfo=...)`.
    """
    nc = fit_config.n_components
    if ratio_params is None:
        ratio_params = {}
    parinfo: list[dict] = []

    for i, comp in enumerate(fit_config.components):
        base = 3 * i

        # --- peak (intensity) ---
        if base in ratio_params:
            # _fit_one_multi sets this for each spectrum; any value inside
            # the bounds will do here.
            ratio = _GUESS_FRACTION
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
            src = _tie_root(fit_config, "tie_center", i)
            # Ratio of rest wavelengths, so every tied component has the same
            # velocity; see _build_scipy_multi.
            factor = float(comp.wavelength.to(u.cm).value
                           / fit_config.components[src].wavelength.to(u.cm).value)
            # mpfit tie expression references the parameter array p
            centre_info["tied"] = f"p[{3 * src + 1}] * {factor!r}"
        elif comp.centre_limits() is not None:
            centre_info["limited"] = [1, 1]
            centre_info["limits"] = list(comp.centre_limits())
        parinfo.append(centre_info)

        # --- sigma (width) ---
        sigma_info: dict = {"value": p0[base + 2]}
        # Width must be strictly positive (avoid divide-by-zero in Gaussian)
        sigma_info["limited"] = [1, 0]
        sigma_info["limits"] = [1e-30, 0.0]
        if comp.sigma_limits() is not None:
            sigma_info["limited"] = [1, 1]
            sigma_info["limits"] = list(comp.sigma_limits())
        if comp.tie_width is not None:
            src = _tie_root(fit_config, "tie_width", i)
            sigma_info["tied"] = f"p[{3 * src + 2}]"
        parinfo.append(sigma_info)

    # --- background ---
    bg_info: dict = {"value": p0[-1]}
    parinfo.append(bg_info)

    return parinfo


def _guess_multi_params(wv: np.ndarray, prof: np.ndarray,
                        fit_config: FitConfig) -> np.ndarray:
    """
    Initial guess for the full parameter vector.

    Only the finite pixels are used. A spectrum with no finite pixel starts
    at zero peak and background.
    """
    wv, prof = np.asarray(wv), np.asarray(prof)
    if len(wv) > 1 and wv[0] > wv[-1]:
        # The width and comb search below need increasing wavelengths.
        wv, prof = wv[::-1], prof[::-1]
    finite = np.isfinite(prof)
    if finite.any():
        wv, prof = _finite_pixels(wv, prof, finite)
        back = float(prof.min())
        prof_c = prof - back
    else:
        back = 0.0
        prof_c = np.zeros(len(wv))
    prof_c[prof_c < 0] = 0

    peak = float(prof_c.max())
    if peak == 0:
        sigma = (wv.max() - wv.min()) / 10
    else:
        # Width of the feature containing the peak, not of everything above
        # half maximum.
        #
        # In a blend, the first and last pixels above half maximum span the
        # whole blend, not one line. Every component then starts several
        # pixels wide, and the fit can settle there: on a seven-component EIS
        # window the widths came out at 89 mA and 16 mA either side of a 28 mA
        # instrumental floor, with a residual 45 percent of the peak and the
        # requested line fitted at effectively zero. Walking out from the peak
        # to half maximum measures the peak feature alone, and for a single
        # line it matches the first and last pixels.
        half_max = 0.5 * peak
        above = prof_c >= half_max
        peak_idx = int(np.nanargmax(prof_c))
        lo = hi = peak_idx
        while lo > 0 and above[lo - 1]:
            lo -= 1
        while hi < len(above) - 1 and above[hi + 1]:
            hi += 1
        if hi > lo:
            sigma = _fwhm_to_sigma(wv[hi] - wv[lo])
        else:
            # Unresolved: the feature is one pixel wide, so take that as an
            # upper bound on the width rather than a tenth of the window.
            sigma = float(np.median(np.diff(wv))) if len(wv) > 1 else \
                (wv.max() - wv.min()) / 10

    # Estimate one shift of the primary centre, and start every other centre
    # at its rest wavelength scaled by the same factor, as the tie enforces.
    #
    # The peak pixel belongs to whichever component dominates the window,
    # which need not be the primary. Starting the primary there displaces it
    # by the gap between the two, and as every tied centre is a fixed multiple
    # of the primary's, the whole comb starts displaced by that gap. The
    # optimiser does not reliably recover: a noiseless 7-component EIS window
    # whose primary sat 0.14 Angstrom from the dominant line gave a 239 km/s
    # centroid for lines at rest by construction, and moving the primary to a
    # different component gave 451 km/s, each time very nearly the offset
    # between that component and the brightest one.
    #
    # So match the whole comb against the profile. Assigning the peak pixel
    # to the nearest rest wavelength only works while the shift is below half
    # the spacing to the neighbouring component, a small velocity for a close
    # blend: Fe XII 195.119 and 195.179 are 0.060 Angstrom apart, so beyond
    # about 46 km/s it picks the wrong component and displaces the comb by a
    # whole spacing, while the synthesis default allows +/- 300 km/s.
    #
    # Each trial shift is scored by the total profile height under the
    # shifted comb, which uses the spacing pattern that one pixel lacks. The
    # trial shifts cover the range that keeps the comb inside the observed
    # window, on the wavelength grid itself, so no resolution is invented.
    # With a single component the score is the profile at each grid point, so
    # the best shift puts the centre on the brightest pixel.
    rest_wl = np.array([c.wavelength.to(u.cm).value
                        for c in fit_config.components])
    ref_wl = float(rest_wl[fit_config.primary_component])
    if fit_config.start == "rest":
        # Every component at its rest wavelength, as eispac's templates start.
        peak_for_search = 0.0
    else:
        peak_for_search = peak
    if peak_for_search > 0:
        # A trial shift moves the primary centre to ref_wl + shift and every
        # other component to its rest wavelength times the same factor. The
        # range keeps the whole comb inside the observed window; with all
        # components at one wavelength it reduces to shifting the comb bodily.
        lo = float(wv.min()) * ref_wl / float(rest_wl.min()) - ref_wl
        hi = float(wv.max()) * ref_wl / float(rest_wl.max()) - ref_wl
        step = float(np.median(np.diff(wv))) if len(wv) > 1 else 0.0
        if hi > lo and step > 0:
            trial = np.arange(lo, hi + 0.5 * step, step)
            score = np.zeros(trial.size)
            for r in rest_wl:
                score += np.interp(r * (ref_wl + trial) / ref_wl, wv, prof_c,
                                   left=0.0, right=0.0)
            # Known limit: if the primary carries no flux, every alias that
            # puts some component on the one visible line scores the same, and
            # the data cannot say which component produced it. With amplitudes
            # free, all the flux in one component at one alias fits exactly as
            # well as all of it in another at the next, so no scoring rule
            # resolves it and the comb can start a spacing away. The velocity
            # asked for is then that of a line that is not there, so there is
            # nothing to recover.
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
    # A start outside a component's limits is moved to the nearest limit, as
    # the optimisers need one inside them.
    for i, comp in enumerate(fit_config.components):
        for offset, limits in ((1, comp.centre_limits()), (2, comp.sigma_limits())):
            if limits is not None:
                full_guess[3 * i + offset] = np.clip(full_guess[3 * i + offset], *limits)
    full_guess[-1] = back
    return full_guess


def _mpfit_residuals(p, fjac=None, x=None, y=None, n_components=1,
                     ratio_params=None, pixel=None, err=None, poisson=False):
    """Residual function in the form mpfit expects.

    Must return `[status, residuals]`, with `status` 0 for success. `err` is
    each pixel's uncertainty; the residuals are divided by it. With
    `poisson`, `y` is counts and the residuals are those whose squares sum
    to the Poisson deviance (:func:`_poisson_residuals`).
    """
    p_eval = np.array(p, dtype=float)
    if ratio_params:
        for child_idx, parent_idx in _in_dependency_order(ratio_params):
            p_eval[child_idx] = p_eval[parent_idx] * p[child_idx]
    model = _line_profile(x, *p_eval, n_components=n_components, pixel=pixel)
    if poisson:
        return [0, _poisson_residuals(y, model)]
    residuals = y - model
    return [0, residuals if err is None else residuals / err]


def _fit_one_multi(wv: np.ndarray, prof: np.ndarray,
                   fit_config: FitConfig,
                   parinfo_template: list[dict],
                   ratio_params: dict | None = None,
                   pixel: float | None = None,
                   sigma: np.ndarray | None = None,
                   poisson: bool = False) -> tuple[np.ndarray, bool]:
    """Fit one spectrum with several Gaussians using mpfit.

    With `pixel`, the Gaussians are averaged over pixels that wide
    (:func:`_line_profile`). `sigma` is each pixel's uncertainty; the
    residuals are divided by it. With `poisson`, the spectrum is counts,
    fitted by their Poisson likelihood (:func:`_poisson_residuals`), with
    every amplitude and the background held at zero or above so the model is
    too.

    Returns the full parameter vector (length `3*N + 1`), with ratio
    parameters converted back to absolute amplitudes, and whether the fit
    succeeded; see :func:`_mpfit_succeeded`.
    """
    p0 = _guess_multi_params(wv, prof, fit_config)
    p0_orig = p0.copy()

    # Stamp current initial guesses into the parinfo dicts
    parinfo = []
    for i, pi in enumerate(parinfo_template):
        d = dict(pi)
        d["value"] = p0[i]
        parinfo.append(d)

    # Convert absolute amplitudes to ratios for constrained params. Use the
    # original guesses, as a parent may already have been converted.
    if ratio_params:
        for child_idx, parent_idx in ratio_params.items():
            ratio = _start_ratio(wv, prof, p0_orig, child_idx, parent_idx)
            parinfo[child_idx]["value"] = ratio
            p0[child_idx] = ratio

    if poisson:
        nc = fit_config.n_components
        held = [3 * i for i in range(nc) if 3 * i not in (ratio_params or {})] + [3 * nc]
        for index in held:
            p0[index] = max(p0[index], 0.0)
            parinfo[index].update({"value": p0[index], "limited": [1, 0], "limits": [0.0, 0.0]})

    functkw = {"x": wv, "y": prof, "n_components": fit_config.n_components,
               "ratio_params": ratio_params, "pixel": pixel, "err": sigma,
               "poisson": poisson}

    try:
        result = mpfit(_mpfit_residuals, p0, parinfo=parinfo,
                       functkw=functkw, quiet=True,
                       maxiter=fit_config.max_iter)
        if result.status > 0:
            out = np.asarray(result.params, dtype=float)
            # Convert ratios back to absolute amplitudes
            if ratio_params:
                for child_idx, parent_idx in _in_dependency_order(ratio_params):
                    out[child_idx] = out[parent_idx] * result.params[child_idx]
            return out, _mpfit_succeeded(result.status, out)
    except Exception:
        pass

    return p0_orig, False  # fall back to initial guess (absolute amplitudes)


# ---------------------------------------------------------------------------
#  Public fitting entry point
# ---------------------------------------------------------------------------
def _unmeasurable(params: np.ndarray, spectra: np.ndarray, wavelength: np.ndarray,
                  n_components: int, kept: np.ndarray | None = None) -> np.ndarray:
    """
    Fits that cannot have measured a line, to be counted as failed.

    Also makes the widths positive in place. A Gaussian does not depend on
    the sign of its width, so a fit with a negative width is kept with the
    sign flipped, which also flips the sign of the intensity computed from
    it. Only the pixels each fit kept are considered: those marked in
    `kept`, by default the finite ones. A flat spectrum, such as one with no
    photons, holds no line to fit, and a component wider than the window of
    kept pixels is background, not a line. A spectrum with no kept pixel
    counts as flat. Where the pixels at an end of the window were left out,
    a centre, tied or not, beyond the edge of the last kept pixel on that
    side was fitted with no data there.
    """
    widths = np.abs(params[..., 2:3 * n_components:3])
    params[..., 2:3 * n_components:3] = widths
    spectra = np.asarray(spectra)
    finite = np.isfinite(spectra) if kept is None else kept
    if not np.issubdtype(spectra.dtype, np.floating):
        if finite.all():
            return (np.all(spectra == spectra[..., :1], axis=-1)
                    | np.any(widths > np.ptp(wavelength), axis=-1))
        spectra = spectra.astype(float)
    wavelength = np.asarray(wavelength)
    first, last = np.min(wavelength), np.max(wavelength)
    half_pixel = 0.5 * np.median(np.abs(np.diff(wavelength))) if wavelength.size > 1 else 0.0
    wavelength = np.broadcast_to(wavelength, spectra.shape)

    def extent(values):
        return (np.max(values, axis=-1, where=finite, initial=-np.inf),
                np.min(values, axis=-1, where=finite, initial=np.inf))

    highest, lowest = extent(spectra)
    flat = highest <= lowest
    longest, shortest = extent(wavelength)
    too_wide = np.any(widths > (longest - shortest)[..., np.newaxis], axis=-1)
    centres = params[..., 1:3 * n_components:3]
    below = ((shortest > first)[..., np.newaxis]
             & (centres < (shortest - half_pixel)[..., np.newaxis]))
    above = ((longest < last)[..., np.newaxis]
             & (centres > (longest + half_pixel)[..., np.newaxis]))
    return flat | too_wide | np.any(below | above, axis=-1)


@overload
def fit_cube_gauss(signal_cube: NDCube, n_jobs: int = -1,
                   fit_config: FitConfig | None = None, *,
                   return_failed: Literal[False] = False, pixel_mean: bool = False,
                   uncertainty: np.ndarray | None = None, poisson: bool = False
                   ) -> tuple[np.ndarray, list[u.Unit]]: ...


@overload
def fit_cube_gauss(signal_cube: NDCube, n_jobs: int = -1,
                   fit_config: FitConfig | None = None, *,
                   return_failed: Literal[True], pixel_mean: bool = False,
                   uncertainty: np.ndarray | None = None, poisson: bool = False
                   ) -> tuple[np.ndarray, list[u.Unit], np.ndarray]: ...


def fit_cube_gauss(signal_cube: NDCube, n_jobs: int = -1,
                   fit_config: FitConfig | None = None, *,
                   return_failed: bool = False, pixel_mean: bool = False,
                   uncertainty: np.ndarray | None = None, poisson: bool = False):
    """
    Fit a Gaussian, or several for a blend, on a flat background to every spectrum.

    By default each Gaussian is evaluated at the centre of each wavelength
    pixel, as observers fit spectra. With `pixel_mean`, it is averaged over
    each pixel instead, which is what the pixel records. The two differ for
    a line not much wider than a pixel.

    By default every pixel of a spectrum has the same weight. With
    `uncertainty`, each is weighted by one over its uncertainty squared, as
    eispac weights its fits. Only the ratios of the uncertainties within a
    spectrum matter, not their scale. With `poisson`, the spectra are counts,
    such as photons, and each is fitted by maximum Poisson likelihood
    instead. This needs no uncertainty, even for a pixel with none.

    Parameters
    ----------
    signal_cube : NDCube
        Spectra, shaped (along the slit, slit positions, wavelength).
    n_jobs : int, optional
        Number of processes; -1 for one per CPU. Default -1.
    fit_config : FitConfig, optional
        Fit configuration. Default None, for one Gaussian.
    return_failed : bool, optional
        Also return which fits failed. Keyword only. Default False.
    pixel_mean : bool, optional
        Fit the Gaussians' mean over each pixel. Keyword only. Default False.
    uncertainty : np.ndarray, optional
        Uncertainty of every value in the cube, same shape as its data or
        broadcastable to it. A pixel whose uncertainty is NaN or positive
        infinity is left out. Where the value is finite, an uncertainty at or
        below zero, negative infinity included, is refused. If there are
        finite values but none has a finite uncertainty, the uncertainty is
        refused. Keyword only. Default None, for an unweighted fit.
    poisson : bool, optional
        Fit the spectra as counts by their Poisson likelihood, with every
        amplitude and the background held at zero or above. Each finite
        count must be zero or more. Not with `uncertainty`. Keyword only.
        Default False.

    Returns
    -------
    data_array : np.ndarray
        Fitted values, shaped (along the slit, slit positions, parameters).
        The parameters are `[peak, centre, sigma, background]` for one
        Gaussian and `[peak0, centre0, sigma0, peak1, ..., background]` for
        several.
    units_list : list of Unit
        Unit of each parameter.
    failed : np.ndarray of bool
        True where the fit failed, shaped (along the slit, slit positions).
        This includes where the optimiser used up `max_iter`, and spectra
        with pixels left out, as given in the Notes. Only with
        `return_failed`.

    Notes
    -----
    A pixel whose value is NaN or infinite is left out, and each spectrum is
    fitted to the pixels kept. With `uncertainty`, so is a pixel whose
    uncertainty is NaN or positive infinity.
    Such a spectrum is a failed fit if fewer pixels are kept than free
    parameters, or if the pixels at an end of the window were left out and
    a fitted centre lies more than half a pixel beyond the outermost kept
    pixel on that side. As for every spectrum, it is also a failed fit if
    its kept pixels are flat or a fitted width is more than their span.
    """
    n_slit, n_scan, _ = signal_cube.shape
    wv = signal_cube.axis_world_coords_values(2)[0].cgs  # wavelength axis

    # A component outside the window has no line to fit, and would come back
    # at its guess with no spread, like a perfect measurement.
    if fit_config is not None and not fit_config.is_single:
        for idx, component in enumerate(fit_config.components):
            if not wv.min() <= component.wavelength.to(wv.unit) <= wv.max():
                raise ValueError(
                    f"fitting.components[{idx}] is at {component.wavelength}, outside the "
                    f"observed window, {wv.min().to(u.AA):.3f} to {wv.max().to(u.AA):.3f}, "
                    f"so there is no line of it to fit. Check its wavelength.")

    # The iteration limit applies to every path. Without a fitting block
    # there is no FitConfig, so use its default.
    max_iter = FitConfig.max_iter if fit_config is None else fit_config.max_iter
    # Pixel width in cm, the unit the fits use for wavelength.
    pixel = spectral_pixel_width(signal_cube).to_value(wv.unit) if pixel_mean else None

    if poisson and uncertainty is not None:
        raise ValueError("A Poisson fit needs no uncertainty: give poisson or uncertainty, "
                         "not both.")
    # Pixels the fits keep: those with a finite value and, if given, a finite uncertainty.
    kept = np.isfinite(signal_cube.data)
    if poisson and np.any(kept & (signal_cube.data < 0)):
        raise ValueError("A Poisson fit is of counts, which are zero or more.")
    if uncertainty is not None:
        try:
            uncertainty = np.broadcast_to(np.asarray(uncertainty, dtype=float),
                                          signal_cube.data.shape)
        except ValueError:
            raise ValueError(
                f"The uncertainty, shaped {np.shape(uncertainty)}, does not match the "
                f"spectra, shaped {signal_cube.data.shape}.") from None
        if np.any(kept & (uncertainty <= 0)):
            raise ValueError("The uncertainty of a finite value must be above zero, or NaN "
                             "to leave that pixel out, as each pixel is weighted by one over "
                             "its square.")
        finite_uncertainty = np.isfinite(uncertainty)
        if kept.any() and not np.any(kept & finite_uncertainty):
            raise ValueError("No finite value has a finite uncertainty, so there is nothing "
                             "to weight the fits by.")
        kept &= finite_uncertainty

    min_pixels = _n_free_params(fit_config)

    def _rows():
        """Each row along the slit, with its uncertainties and the pixels kept."""
        for i in range(n_slit):
            yield (signal_cube.data[i], None if uncertainty is None else uncertainty[i],
                   kept[i])

    if fit_config is None or fit_config.is_single:
        # One Gaussian on a background, with its own fast fit.
        use_mpfit = fit_config is not None and fit_config.backend == "mpfit"
        fit_one = _fit_one_mpfit if use_mpfit else _fit_one

        def _fit(wave, prof, sigma):
            return fit_one(wave, prof, max_iter, pixel, sigma, poisson)

        _start = _guess_params
        n_components, n_params = 1, 4
        description = "Fit chunks"
    else:
        n_components, n_params = fit_config.n_components, fit_config.n_full_params
        description = "Fit chunks (multi)"
        if fit_config.backend != "mpfit":
            # scipy (default, fast) supports all constraints, positive
            # intensity and amplitude ratio, through TRF bounds: LM when
            # unconstrained, TRF when bounds are active.
            (model_func, free_to_full, n_free,
             free_indices, ratio_spec, bounds, has_bounds) = _build_scipy_multi(
                fit_config, None if pixel is None else pixel * 1e8)

            def _fit(wave, prof, sigma):
                return _fit_one_scipy_multi(wave, prof, fit_config, model_func, free_to_full,
                                            free_indices, ratio_spec, bounds, has_bounds,
                                            sigma, poisson)

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

            def _fit(wave, prof, sigma):
                return _fit_one_multi(wave, prof, fit_config, parinfo_template, ratio_params,
                                      pixel, sigma, poisson)

        def _start(wave, prof):
            return _guess_multi_params(wave, prof, fit_config)

    def _fit_block(spec_block, unc_block, kept_block):
        results = np.empty((spec_block.shape[0], n_params))
        succeeded = np.empty(spec_block.shape[0], dtype=bool)
        for i in range(spec_block.shape[0]):
            results[i], succeeded[i] = _fit_finite(
                _fit, wv.value, spec_block[i],
                None if unc_block is None else unc_block[i], kept_block[i], min_pixels,
                _start)
        return results, succeeded

    with tqdm_joblib(tqdm(total=n_slit, desc=description, leave=False)):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_fit_block)(*row) for row in _rows()
        )

    data_array = np.stack([r[0] for r in results], axis=0)
    failed = ~np.stack([r[1] for r in results], axis=0)
    failed |= _unmeasurable(data_array, signal_cube.data, wv.value, n_components, kept)

    # Amplitude, centre and width of each component, then the background.
    units_list = [signal_cube.unit, wv.unit, wv.unit] * n_components + [signal_cube.unit]

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
        Fit configuration. A single-Gaussian fit has one component.
    rest_wavelength : u.Quantity
        Rest wavelength of the line, used to name a single-Gaussian fit.

    Returns
    -------
    list of str
    """
    if fit_config is None or fit_config.is_single:
        return [default_component_name(rest_wavelength)]
    return [default_component_name(c.wavelength) if c.name is None else c.name
            for c in fit_config.components]


def spectral_pixel_width(cube: NDCube) -> u.Quantity:
    """Width of one spectral pixel, above zero, of a cube with wavelength on its last axis."""
    wavelength = u.Quantity(cube.axis_world_coords_values(2)[0]).to(u.cm)
    return abs(wavelength[1] - wavelength[0])


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
        Width of one spectral pixel; its sign is ignored.
    rest_wavelength : u.Quantity
        Rest wavelength of a single-Gaussian fit. With components, each uses
        its own.
    fit_config : FitConfig, optional
        Fit configuration; None for a single Gaussian.

    Returns
    -------
    dict
        `components` maps each component name to `intensity`, `velocity` and
        `width`, and `background` holds the fitted background. Each is a
        Quantity shaped like `fit_data` without its last axis. `intensity` is
        the fitted Gaussian summed over spectral pixels, so it is in counts,
        like the total signal maps. `velocity` is relative to the component's
        rest wavelength, and `width` is the Gaussian sigma.
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
        pixels = (sigma / abs(wavelength_step)).to(u.dimensionless_unscaled)
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

    Failed fits are left out of every mean and standard deviation. A pixel
    whose fits all failed has NaN statistics. The intensity statistics are
    taken over each iteration's intensity, so they account for the peak and
    width moving together, which the per-parameter statistics cannot.

    Parameters
    ----------
    fit_data : np.ndarray
        Shape `(n_iter, n_slit, n_scan, n_params)`.
    failed : np.ndarray of bool
        Shape `(n_iter, n_slit, n_scan)`, True where a fit failed.
    units, wavelength_step, rest_wavelength, fit_config
        As for :func:`fit_quantities`. `fit_config.bessel_correction` makes
        the standard deviations divide by n - 1 rather than n, and
        `fit_config.save_iterations` keeps `fit_data` and `failed`.

    Returns
    -------
    dict
        `first_fit_data` (the first iteration, failed or not), `mean_data`,
        `std_data` and `units`, per parameter, and:

        - `components`: for each component name, its `rest_wavelength`,
          `tied` (names of the components its `velocity` and `width` are
          tied to, or None), and `intensity`, `velocity` and `width`, each
          with `first`, `mean` and `std` maps. `first` is NaN where the
          first fit failed.
        - `background`: `first`, `mean` and `std` maps.
        - `primary_component`: name of the primary component.
        - `failed_fits`: number of failed fits in each pixel.
        - `n_iterations`: number of fits in each pixel.
        - `bessel_correction`: whether the standard deviations divide by
          n - 1.
        - `iterations`: `fit_data` and `failed`, only when
          `fit_config.save_iterations` is set.
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
                         n_jobs: int = -1, *,
                         uncertainty: np.ndarray | None = None) -> dict:
    """
    Fit the noiseless cube and return each component's velocity and width.

    Parameters
    ----------
    cube : NDCube
        Cube before the instrument, with `rest_wav` in its metadata.
    fit_config : FitConfig, optional
        Fit configuration; None for a single Gaussian.
    n_jobs : int
        Joblib parallelism.
    uncertainty : np.ndarray, optional
        Each pixel's uncertainty, to weight the fit like the measured
        spectra, as for :func:`fit_cube_gauss`. Keyword only. Default None,
        for an unweighted fit.

    Returns
    -------
    dict
        `fit_truth_data` and `fit_truth_units` as from
        :func:`fit_cube_gauss`, `failed` (True where the fit failed), and
        `components`: for each component name, `velocity` and `width` maps,
        NaN where the fit failed. There is no intensity, because the cube is
        in the units of the synthesis rather than in counts.
    """
    # The cube holds the spectrum averaged over each pixel, so the truth is
    # fitted with pixel-averaged Gaussians. Fitted at the pixel centres, as
    # the noisy spectra are, a line not much wider than a pixel comes out
    # pulled toward the middle of its pixel and too wide, and the truth would
    # hide the same error in the noisy fits.
    data, units, failed = fit_cube_gauss(cube, n_jobs=n_jobs,
                                         fit_config=fit_config,
                                         return_failed=True, pixel_mean=True,
                                         uncertainty=uncertainty)
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
    Line-of-sight velocity of each fitted line from its centre.

    A positive velocity is a redshift, away from the observer.

    Parameters
    ----------
    fit_arr : np.ndarray or u.Quantity
        Fitted values from `fit_cube_gauss`, with the parameters on the last
        axis.
    wl0 : u.Quantity
        Rest wavelength the velocity is measured from.
    n_jobs : int, optional
        Number of processes; -1 for one per CPU. Default -1.
    fit_config : FitConfig, optional
        Fit configuration of a blend, which gives the parameter holding the
        primary component's centre.

    Returns
    -------
    u.Quantity
        Velocity in each pixel, in cm/s.
    """
    idx = 1 if (fit_config is None or fit_config.is_single) else fit_config.idx_center
    centres_raw = fit_arr[..., idx]  # (n_slit, n_scan)
    # Ensure we have a pure Quantity array
    if isinstance(centres_raw, u.Quantity):
        centres = centres_raw.to(wl0.unit)
    elif np.asarray(centres_raw).dtype != object:
        # fit_cube_gauss fits in the cube's wavelengths in cgs.
        centres = u.Quantity(centres_raw, u.cm).to(wl0.unit)
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
    Width of each fitted line: its Gaussian sigma, in wavelength.

    Parameters
    ----------
    fit_arr : np.ndarray or u.Quantity
        Fitted values from `fit_cube_gauss`, with the parameters on the last
        axis.
    n_jobs : int, optional
        Not used; kept so existing calls still work.
    fit_config : FitConfig, optional
        Fit configuration of a blend, which gives the parameter holding the
        primary component's width.

    Returns
    -------
    u.Quantity
        Width in each pixel: in cm for a plain array, as `fit_cube_gauss`
        returns, or in the unit of `fit_arr` for quantities.
    """
    idx = 2 if (fit_config is None or fit_config.is_single) else fit_config.idx_sigma
    widths_raw = fit_arr[..., idx]  # (n_slit, n_scan)
    # Ensure we have a pure Quantity array
    if isinstance(widths_raw, u.Quantity):
        widths = widths_raw
    elif np.asarray(widths_raw).dtype != object:
        widths = u.Quantity(widths_raw, u.cm)
    else:  # object array of Quantity scalars
        get_val = np.vectorize(lambda q: q.value)
        get_unit = widths_raw.flat[0].unit  # Get unit from first element
        widths = u.Quantity(get_val(widths_raw), get_unit)
    
    return widths


def analyse(fits_all: u.Quantity | np.ndarray, v_true: u.Quantity, wl0: u.Quantity,
            fit_config: FitConfig | None = None) -> dict:
    """
    Mean and scatter of the velocity and width over Monte Carlo iterations.

    Parameters
    ----------
    fits_all : np.ndarray or u.Quantity
        Fits of every iteration, with the iterations on the first axis and
        the parameters on the last.
    v_true : u.Quantity
        True velocity, for comparison.
    wl0 : u.Quantity
        Rest wavelength the velocity is measured from.
    fit_config : FitConfig, optional
        Fit configuration of a blend.

    Returns
    -------
    dict
        `v_mean`, `v_std`, `w_mean` and `w_std`: mean and standard deviation
        of the velocity and width over the iterations. `v_err`: true
        velocity minus the mean. `v_samples` and `w_samples`: every
        iteration's values. `v_true`: the true velocity.
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
