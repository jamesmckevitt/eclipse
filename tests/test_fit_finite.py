"""Fits that leave out pixels whose value or uncertainty is NaN or infinite."""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.fitting import (FitComponent, FitConfig, _fit_finite, _n_free_params,
                                   _unmeasurable, fit_cube_gauss)

C = const.c.to_value(u.km / u.s)
STEP = 0.0169e-8
N_WAVE = 41
START = 195.15e-8 - (N_WAVE - 1) / 2 * STEP
DOPPLER = 1 + 20 / C
BLEND = [FitComponent(195.119 * u.AA),
         FitComponent(195.179 * u.AA, tie_center=0, tie_width=0)]
# Six free parameters: two amplitudes, the second held below the first, two
# widths, one centre and the background.
PARTIAL = [FitComponent(195.119 * u.AA, amplitude_greater_than=1),
           FitComponent(195.179 * u.AA, tie_center=0)]


def _wave(n=N_WAVE, first=0):
    return START + (first + np.arange(n)) * STEP


def _line(wave, single):
    """One line at 20 km/s, or the Fe XII 195.119 and 195.179 blend, 10:3, on a background of 10."""
    prof = 10.0 + 100.0 * np.exp(-0.5 * ((wave - 195.119e-8 * DOPPLER) / 0.03e-8) ** 2)
    if not single:
        prof += 30.0 * np.exp(-0.5 * ((wave - 195.179e-8 * DOPPLER) / 0.03e-8) ** 2)
    return prof


def _cube(rows, first=0, dtype=float, decreasing=False):
    """
    Spectra shaped (rows, n) as a (rows, 1, n) cube starting at pixel `first` of `_wave`.

    With `decreasing`, the axis runs the other way from the same last wavelength.
    """
    rows = np.asarray(rows, dtype=dtype)
    n = rows.shape[-1]
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [-STEP if decreasing else STEP, 0.4, 0.16]
    wcs.wcs.crpix = [1.0, 1.0, 1.0]
    wcs.wcs.crval = [_wave()[first + n - 1 if decreasing else first], 0.0, 0.0]
    return NDCube(rows[:, None, :], wcs=wcs, unit=u.DN / u.pix)


def _config(single, backend, components=BLEND):
    if single:
        return None if backend == "scipy" else FitConfig(backend=backend)
    return FitConfig(components=components, backend=backend)


def _velocity(fits):
    """Velocity of the first component in km/s."""
    return (fits[..., 1] / 195.119e-8 - 1) * C


@pytest.mark.parametrize("pixel_mean", [False, True], ids=["centre", "pixel_mean"])
@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
@pytest.mark.parametrize("single", [True, False], ids=["single", "blend"])
def test_nan_pixels_at_the_ends_fit_as_a_shorter_window(single, backend, pixel_mean):
    """
    NaN in the first 3 and last 2 pixels gives the fit of the window without them.

    The two cubes' wavelengths differ in their last bits, and the optimisers
    stop at a relative tolerance near 1e-8, so the fits agree to about that.
    """
    noisy = _line(_wave(), single) + np.random.default_rng(3).normal(0, 3, N_WAVE)
    with_nan = noisy.copy()
    with_nan[:3] = np.nan
    with_nan[-2:] = np.nan
    config = _config(single, backend)
    data, _, failed = fit_cube_gauss(_cube([with_nan]), n_jobs=1, fit_config=config,
                                     return_failed=True, pixel_mean=pixel_mean)
    short, _, short_failed = fit_cube_gauss(_cube([noisy[3:-2]], first=3), n_jobs=1,
                                            fit_config=config, return_failed=True,
                                            pixel_mean=pixel_mean)
    assert not failed.any() and not short_failed.any()
    assert data == pytest.approx(short, rel=1e-7)
    assert (data[0, 0, 1] / short[0, 0, 1] - 1) * C == pytest.approx(0, abs=1e-3)


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
@pytest.mark.parametrize("single", [True, False], ids=["single", "blend"])
def test_a_few_nan_pixels_inside_the_window_change_the_fit_little(single, backend):
    """Three NaN pixels, one on the line, leave a noise-free line's velocity within 0.01 km/s."""
    clean = _line(_wave(), single)
    with_nan = clean.copy()
    with_nan[[5, int(np.argmax(clean)) + 1, 30]] = np.nan
    data, _, failed = fit_cube_gauss(_cube([clean, with_nan]), n_jobs=1,
                                     fit_config=_config(single, backend), return_failed=True)
    assert not failed.any()
    velocity = (data[:, 0, 1] / 195.119e-8 - 1) * C
    assert velocity == pytest.approx(20, abs=1e-2)


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
@pytest.mark.parametrize("single", [True, False], ids=["single", "blend"])
def test_too_few_finite_pixels_fail_alone(single, backend):
    """
    A spectrum with fewer finite pixels than free parameters fails; the others are fitted.

    One Gaussian has 4 free parameters. The tied blend has 5: two
    amplitudes, one centre, one width and the background.
    """
    wave = _wave()
    rows = np.tile(_line(wave, single), (4, 1))
    rows[0, :] = np.nan
    rows[0, 18:21] = _line(wave, single)[18:21]
    rows[1, :] = np.nan
    rows[3, :] = np.nan
    rows[3, 17:21] = _line(wave, single)[17:21]
    data, _, failed = fit_cube_gauss(_cube(rows), n_jobs=1,
                                     fit_config=_config(single, backend), return_failed=True)
    assert failed[:3, 0].tolist() == [True, True, False]
    assert (data[2, 0, 1] / 195.119e-8 - 1) * C == pytest.approx(20, abs=1e-3)
    if not single:
        # 4 finite pixels are too few for the blend's 5 free parameters.
        assert failed[3, 0]


@pytest.mark.parametrize("poisson", [False, True], ids=["unweighted", "poisson"])
@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
@pytest.mark.parametrize("single", [True, False], ids=["single", "blend"])
def test_integer_counts_are_fitted(single, backend, poisson):
    """Whole-number spectra, such as photon counts, are fitted and not counted as failed."""
    rows = np.rint(10 * np.tile(_line(_wave(), single), (2, 1)))
    _, _, failed = fit_cube_gauss(_cube(rows, dtype=np.int64), n_jobs=1,
                                  fit_config=_config(single, backend), return_failed=True,
                                  poisson=poisson)
    assert not failed.any()


def test_the_free_parameters_are_counted():
    assert _n_free_params(None) == 4
    assert _n_free_params(FitConfig(components=BLEND)) == 5
    assert _n_free_params(FitConfig(components=PARTIAL)) == 6


def test_the_fit_is_tried_from_as_many_finite_pixels_as_free_parameters():
    """
    A spectrum with NaN pixels is fitted from 4 finite pixels, not 3; one with
    every pixel finite is always fitted, as the optimiser decides.
    """
    tried = []

    def fit(wave, prof, sigma):
        tried.append(len(prof))
        return np.zeros(1), True

    def start(wave, prof):
        return np.ones(1)

    wave = _wave()
    prof = np.full(N_WAVE, np.nan)
    prof[:4] = 1.0
    assert _fit_finite(fit, wave, prof, None, np.isfinite(prof), 4, start)[1]
    prof[0] = np.nan
    assert not _fit_finite(fit, wave, prof, None, np.isfinite(prof), 4, start)[1]
    assert _fit_finite(fit, wave[:3], np.ones(3), None, np.ones(3, bool), 4, start)[1]
    assert tried == [4, 3]


def test_the_kept_pixels_go_to_the_fit_with_their_uncertainties():
    seen = []

    def fit(wave, prof, sigma):
        seen.append((wave, prof, sigma))
        return np.zeros(1), True

    wave = _wave(5)
    sigma = np.array([1.0, np.inf, 3.0, 4.0, 5.0])
    prof = np.array([10.0, 20.0, np.nan, 40.0, 50.0])
    _fit_finite(fit, wave, prof, sigma, np.isfinite(prof) & np.isfinite(sigma), 1, None)
    kept_wave, kept_prof, kept_sigma = seen[0]
    assert kept_wave.tolist() == wave[[0, 3, 4]].tolist()
    assert kept_prof.tolist() == [10.0, 40.0, 50.0]
    assert kept_sigma.tolist() == [1.0, 4.0, 5.0]


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
def test_exactly_as_many_finite_pixels_as_free_parameters_are_fitted(backend):
    """Four noise-free pixels around one line determine it, and give its velocity."""
    rows = np.full((1, N_WAVE), np.nan)
    rows[0, 17:21] = _line(_wave(), True)[17:21]
    data, _, failed = fit_cube_gauss(_cube(rows), n_jobs=1, fit_config=_config(True, backend),
                                     return_failed=True)
    assert not failed.any()
    assert _velocity(data[0, 0]) == pytest.approx(20, abs=1e-3)


def test_a_centre_in_nan_pixels_at_the_end_fails():
    """
    A centre beyond the finite pixels was fitted with no data there.

    Spectra: all finite, NaN in the last 10 pixels, NaN in the first 10,
    and NaN in one pixel inside. The centres are in the middle, and 0.4 and
    0.6 pixels past the last (or before the first) finite pixel of the
    spectra with NaN ends. Only a centre in the NaN pixels at an end, more
    than half a pixel past the last finite one, fails. A centre past the
    whole window fails only if the spectrum's pixels at that end are NaN,
    and a centre on a NaN pixel inside is kept.
    """
    wave = _wave()
    after = [wave[20], wave[-11] + 0.4 * STEP, wave[-11] + 0.6 * STEP]
    before = [wave[20], wave[10] - 0.4 * STEP, wave[10] - 0.6 * STEP]
    params = np.array([[[10.0, centre, STEP, 0.0] for centre in after + after + before + after]])
    spectra = np.tile(np.arange(N_WAVE, dtype=float), (1, 12, 1))
    spectra[0, 3:6, -10:] = np.nan
    spectra[0, 6:9, :10] = np.nan
    spectra[0, 9:, 30] = np.nan
    assert _unmeasurable(params, spectra, wave, 1).tolist() == [
        [False, False, False, False, False, True, False, False, True, False, False, False]]

    # Past the long end of the window: kept for a finite spectrum, failed with NaN at that
    # end, kept with NaN only at the other end. On the NaN pixel inside: kept.
    params = np.array([[[10.0, wave[-1] + 2 * STEP, STEP, 0.0]] * 3
                       + [[10.0, wave[30], STEP, 0.0]]])
    assert _unmeasurable(params, spectra[:, [0, 3, 6, 9]], wave, 1).tolist() == [
        [False, True, False, False]]


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
def test_nan_pixels_on_a_decreasing_axis_fit_as_on_an_increasing_one(backend):
    prof = _line(_wave(), False)
    prof[[2, 19, 30]] = np.nan
    rising, _, failed = fit_cube_gauss(_cube([prof]), n_jobs=1,
                                       fit_config=_config(False, backend), return_failed=True)
    falling, _, falling_failed = fit_cube_gauss(_cube([prof[::-1]], decreasing=True), n_jobs=1,
                                                fit_config=_config(False, backend),
                                                return_failed=True)
    assert not failed.any() and not falling_failed.any()
    assert _velocity(falling[0, 0]) == pytest.approx(_velocity(rising[0, 0]), abs=1e-3)
    assert _velocity(rising[0, 0]) == pytest.approx(20, abs=1e-2)


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
@pytest.mark.parametrize("single", [True, False], ids=["single", "blend"])
def test_float32_spectra_with_nan_pixels_are_fitted(single, backend):
    prof = _line(_wave(), single)
    prof[[2, 19, 30]] = np.nan
    data, _, failed = fit_cube_gauss(_cube([prof], dtype=np.float32), n_jobs=1,
                                     fit_config=_config(single, backend), return_failed=True)
    assert not failed.any()
    assert _velocity(data[0, 0]) == pytest.approx(20, abs=0.1)


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
def test_a_partly_tied_blend_with_nan_pixels(backend):
    """With 6 free parameters, NaN pixels are left out, and 5 finite pixels are too few."""
    clean = _line(_wave(), False)
    with_nan = clean.copy()
    with_nan[[2, 19, 30]] = np.nan
    few = np.full(N_WAVE, np.nan)
    few[17:22] = clean[17:22]
    data, _, failed = fit_cube_gauss(_cube([clean, with_nan, few]), n_jobs=1,
                                     fit_config=_config(False, backend, PARTIAL),
                                     return_failed=True)
    assert failed[:, 0].tolist() == [False, False, True]
    assert _velocity(data[1, 0]) == pytest.approx(_velocity(data[0, 0]), abs=1e-2)
    assert _velocity(data[1, 0]) == pytest.approx(20, abs=1e-2)


def _weighting(mode, rows):
    """Keywords for an unweighted, weighted or Poisson fit of `rows`, shaped (rows, n)."""
    if mode == "poisson":
        return {"poisson": True}
    if mode == "uncertainty":
        return {"uncertainty": np.sqrt(np.abs(rows) + 1.0)[:, None, :]}
    return {}


@pytest.mark.parametrize("mode", ["uncertainty", "poisson"])
@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
@pytest.mark.parametrize("single", [True, False], ids=["single", "blend"])
def test_weighted_and_poisson_fits_leave_out_nan_pixels_at_the_ends(single, backend, mode):
    """As for an unweighted fit, the fit is that of the window without them."""
    counts = np.random.default_rng(4).poisson(_line(_wave(), single)).astype(float)
    with_nan = counts.copy()
    with_nan[:3] = np.nan
    with_nan[-2:] = np.inf
    config = _config(single, backend)
    data, _, failed = fit_cube_gauss(_cube([with_nan]), n_jobs=1, fit_config=config,
                                     return_failed=True, **_weighting(mode, with_nan[None]))
    short, _, short_failed = fit_cube_gauss(_cube([counts[3:-2]], first=3), n_jobs=1,
                                            fit_config=config, return_failed=True,
                                            **_weighting(mode, counts[None, 3:-2]))
    assert not failed.any() and not short_failed.any()
    assert data == pytest.approx(short, rel=1e-6)
    assert _velocity(data[0, 0]) == pytest.approx(_velocity(short[0, 0]), abs=1e-3)


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
def test_a_pixel_with_a_nan_uncertainty_is_left_out_of_a_weighted_fit(backend):
    """The data of that pixel, though finite, is not used: a spike there does not move the fit."""
    clean = _line(_wave(), False)
    spiked = clean.copy()
    spiked[25] += 500.0
    uncertainty = np.ones((2, 1, N_WAVE))
    uncertainty[1, 0, 25] = np.nan
    data, _, failed = fit_cube_gauss(_cube([clean, spiked]), n_jobs=1,
                                     fit_config=_config(False, backend), return_failed=True,
                                     uncertainty=uncertainty)
    assert not failed.any()
    assert _velocity(data[:, 0]) == pytest.approx(20, abs=1e-2)


@pytest.mark.parametrize("mode", ["unweighted", "uncertainty", "poisson"])
@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
def test_too_few_kept_pixels_fail_in_every_mode(backend, mode):
    """Three kept pixels are too few for the blend's 5 parameters, so that spectrum fails alone."""
    clean = np.rint(_line(_wave(), False))
    few = np.full(N_WAVE, np.nan)
    few[17:20] = clean[17:20]
    rows = np.stack([clean, few])
    _, _, failed = fit_cube_gauss(_cube(rows), n_jobs=1, fit_config=_config(False, backend),
                                  return_failed=True, **_weighting(mode, rows))
    assert failed[:, 0].tolist() == [False, True]


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
def test_a_nan_value_with_a_finite_uncertainty_is_left_out(backend):
    """The value decides too: a NaN value with an uncertainty of 1 is left out."""
    prof = _line(_wave(), False)
    with_nan = prof.copy()
    with_nan[[4, 25]] = np.nan
    data, _, failed = fit_cube_gauss(_cube([prof, with_nan]), n_jobs=1,
                                     fit_config=_config(False, backend), return_failed=True,
                                     uncertainty=np.ones((2, 1, N_WAVE)))
    assert not failed.any()
    assert _velocity(data[:, 0]) == pytest.approx(20, abs=1e-2)


def test_a_missing_pixel_may_carry_any_uncertainty():
    """Data marked missing with NaN and an uncertainty of -100 is fitted on the other pixels."""
    prof = _line(_wave(), True)
    prof[30] = np.nan
    uncertainty = np.ones(N_WAVE)
    uncertainty[30] = -100.0
    data, _, failed = fit_cube_gauss(_cube([prof]), n_jobs=1, return_failed=True,
                                     uncertainty=uncertainty[None, None])
    assert not failed.any()
    assert _velocity(data[0, 0]) == pytest.approx(20, abs=1e-2)
    with pytest.raises(ValueError, match="above zero"):
        fit_cube_gauss(_cube([_line(_wave(), True)]), n_jobs=1,
                       uncertainty=uncertainty[None, None])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_an_uncertainty_that_leaves_out_every_value_is_refused(bad):
    with pytest.raises(ValueError, match="No finite value has a finite uncertainty"):
        fit_cube_gauss(_cube([_line(_wave(), True)]), n_jobs=1,
                       uncertainty=np.full((1, 1, N_WAVE), bad))


def test_a_negative_infinite_uncertainty_is_refused_and_a_positive_one_left_out():
    uncertainty = np.ones((1, 1, N_WAVE))
    uncertainty[0, 0, 30] = -np.inf
    with pytest.raises(ValueError, match="above zero"):
        fit_cube_gauss(_cube([_line(_wave(), True)]), n_jobs=1, uncertainty=uncertainty)
    uncertainty[0, 0, 30] = np.inf
    data, _, failed = fit_cube_gauss(_cube([_line(_wave(), True)]), n_jobs=1,
                                     return_failed=True, uncertainty=uncertainty)
    assert not failed.any()
    assert _velocity(data[0, 0]) == pytest.approx(20, abs=1e-2)


def test_values_that_are_all_nan_with_nan_uncertainties_fail_without_stopping():
    """As a run gives them: a NaN DN has a NaN uncertainty."""
    rows = np.full((1, N_WAVE), np.nan)
    _, _, failed = fit_cube_gauss(_cube(rows), n_jobs=1, return_failed=True,
                                  uncertainty=np.full((1, 1, N_WAVE), np.nan))
    assert failed.all()


def test_values_that_are_all_nan_fail_in_a_weighted_fit_as_in_an_unweighted_one():
    rows = np.full((1, N_WAVE), np.nan)
    for weighting in ({}, {"uncertainty": np.ones((1, 1, N_WAVE))}):
        _, _, failed = fit_cube_gauss(_cube(rows), n_jobs=1, return_failed=True, **weighting)
        assert failed.all()


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
def test_a_centre_in_end_pixels_left_out_for_their_uncertainty_fails(backend):
    """The end rule holds for pixels left out for a NaN uncertainty, as for a NaN value."""
    wave = _wave()
    prof = _line(wave, True)
    uncertainty = np.ones((1, 1, N_WAVE))
    # The line is near pixel 19; leave out everything from pixel 15 on, so the
    # fit finds the centre from the line's short-wavelength side alone.
    uncertainty[0, 0, 15:] = np.nan
    data, _, failed = fit_cube_gauss(_cube([prof]), n_jobs=1, fit_config=_config(True, backend),
                                     return_failed=True, uncertainty=uncertainty)
    assert data[0, 0, 1] > wave[14] + 0.5 * STEP
    assert failed[0, 0]
    # The same pixels as a window of their own fit to the same centre and do
    # not fail, so the end rule alone failed the fit above.
    short, _, short_failed = fit_cube_gauss(_cube([prof[:15]]), n_jobs=1,
                                            fit_config=_config(True, backend),
                                            return_failed=True,
                                            uncertainty=np.ones((1, 1, 15)))
    assert not short_failed[0, 0]
    assert short[0, 0, 1] == pytest.approx(data[0, 0, 1], rel=1e-9)
