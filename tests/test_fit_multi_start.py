"""Starting values for a multi-component fit: width, wavelength order and NaN pixels."""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.fitting import (FitComponent, FitConfig, _guess_multi_params, _start_ratio,
                                   fit_cube_gauss)
from euvst_response.utils import _fwhm_to_sigma

C = const.c.to_value(u.km / u.s)
STEP = 0.0169e-8
N_WAVE = 41
CENTRE = 195.15e-8
WV = CENTRE + (np.arange(N_WAVE) - (N_WAVE - 1) / 2) * STEP
CONFIG = FitConfig(components=[FitComponent(195.119 * u.AA),
                               FitComponent(195.179 * u.AA, tie_center=0, tie_width=0)])
DOPPLER = 1 + 20 / C


def _blend(wave):
    """Fe XII 195.119 and 195.179 at 20 km/s, 3:1, on a background of 10."""
    return (10.0 + 100.0 * np.exp(-0.5 * ((wave - 195.119e-8 * DOPPLER) / 0.03e-8) ** 2)
            + 30.0 * np.exp(-0.5 * ((wave - 195.179e-8 * DOPPLER) / 0.03e-8) ** 2))



def _box(n_pixels):
    """A flat background with `n_pixels` bright pixels in the middle."""
    prof = np.full(N_WAVE, 10.0)
    start = N_WAVE // 2 - n_pixels // 2
    prof[start:start + n_pixels] = 110.0
    return prof


@pytest.mark.parametrize("n_pixels", [1, 2, 3])
def test_a_feature_n_pixels_wide_starts_n_pixels_wide(n_pixels):
    """Each pixel above half maximum counts a whole pixel wide, ends included."""
    sigma = _guess_multi_params(WV, _box(n_pixels), CONFIG)[2]
    assert sigma == pytest.approx(_fwhm_to_sigma(n_pixels * STEP))


@pytest.mark.parametrize("n_pixels", [1, 2])
def test_a_decreasing_wavelength_axis_starts_as_an_increasing_one(n_pixels):
    increasing = _guess_multi_params(WV, _box(n_pixels), CONFIG)
    decreasing = _guess_multi_params(WV[::-1], _box(n_pixels)[::-1], CONFIG)
    assert decreasing[2] > 0
    assert decreasing == pytest.approx(increasing, rel=1e-12)


def test_one_nan_pixel_leaves_the_start_finite():
    prof = _box(3)
    prof[5] = np.nan
    guess = _guess_multi_params(WV, prof, CONFIG)
    assert np.all(np.isfinite(guess))
    assert guess == pytest.approx(_guess_multi_params(WV, _box(3), CONFIG), rel=1e-12)


def test_a_nan_pixel_on_the_line_leaves_the_start_finite_and_near_the_line():
    prof = _blend(WV)
    prof[np.argmax(prof)] = np.nan
    guess = _guess_multi_params(WV, prof, CONFIG)
    assert np.all(np.isfinite(guess))
    assert abs(guess[1] - 195.119e-8 * DOPPLER) < 2 * STEP


def test_an_all_nan_spectrum_starts_at_zero():
    guess = _guess_multi_params(WV, np.full(N_WAVE, np.nan), CONFIG)
    assert guess[0] == 0 and guess[-1] == 0
    assert np.all(np.isfinite(guess))


def test_a_nan_pixel_counts_as_background_in_the_starting_ratio():
    """A NaN beside the fainter line's centre gives the ratio of a background pixel there."""
    prof = _blend(WV)
    guess = _guess_multi_params(WV, prof, CONFIG)
    beside = int(np.searchsorted(WV, guess[4]))
    with_nan, with_background = prof.copy(), prof.copy()
    with_nan[beside] = np.nan
    with_background[beside] = guess[-1]
    ratio = _start_ratio(WV, with_nan, guess, 3, 0)
    assert 0 < ratio < 1
    assert ratio == pytest.approx(_start_ratio(WV, with_background, guess, 3, 0), rel=1e-12)


def test_a_peak_halfway_between_two_centres_starts_the_same_in_either_order():
    """
    The peak pixel is one pixel from the components either side of it.

    It is given to the one on the brighter side, whose run of pixels above
    half maximum is two pixels wide, whatever the order of the components.
    """
    prof = 10 + 100 * np.exp(-0.5 * ((WV - CENTRE) / STEP) ** 2)
    prof[N_WAVE // 2 + 1] += 30
    far = FitComponent((CENTRE - 10 * STEP) * 1e8 * u.AA)
    either_side = [FitComponent((CENTRE + sign * STEP) * 1e8 * u.AA) for sign in (-1, 1)]
    widths = [_guess_multi_params(WV, prof, FitConfig(components=[far] + side))[2]
              for side in (either_side, either_side[::-1])]
    assert widths[0] == widths[1] == pytest.approx(_fwhm_to_sigma(2 * STEP))


def test_two_components_at_one_wavelength_start_apart():
    """Started identical, a core and a wing would stay identical through the fit."""
    config = FitConfig(components=[FitComponent(195.119 * u.AA),
                                   FitComponent(195.119 * u.AA, name="wing")])
    guess = _guess_multi_params(WV, _box(3), config)
    assert guess[1] == guess[4]
    assert guess[3] < guess[0]


def _cube(data, centre=CENTRE, decreasing=False):
    """Spectra shaped (rows, N_WAVE) as a (rows, 1, N_WAVE) cube, wavelength step `STEP`."""
    sign = -1 if decreasing else 1
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [sign * STEP, 0.4, 0.16]
    wcs.wcs.crpix = [(N_WAVE + 1) / 2, 1.0, 1.0]
    wcs.wcs.crval = [centre, 0.0, 0.0]
    return NDCube(np.asarray(data)[:, None, :], wcs=wcs, unit=u.DN / u.pix)


def _blend_cube(decreasing=False):
    wave = WV[::-1] if decreasing else WV
    return _cube(np.tile(_blend(wave), (3, 1)), decreasing=decreasing)


@pytest.mark.parametrize("settings", [{"backend": "mpfit"}, {}], ids=["mpfit", "scipy"])
def test_a_blend_on_a_decreasing_wavelength_axis_fits_as_on_an_increasing_one(settings):
    """
    Both backends fit it to the same values, at the velocity it was made with.

    Centres are compared as velocities, as a relative tolerance on a centre
    near 195 A would allow 0.3 km/s per 1e-6.
    """
    config = FitConfig(components=CONFIG.components, **settings)
    fits = []
    for decreasing in (False, True):
        data, _, failed = fit_cube_gauss(_blend_cube(decreasing), n_jobs=1, fit_config=config,
                                         return_failed=True)
        assert not failed.any()
        fits.append(data[:, 0])
    velocity = [(fit[:, [1, 4]] / [195.119e-8, 195.179e-8] - 1) * C for fit in fits]
    assert velocity[1] == pytest.approx(velocity[0], abs=1e-3)
    assert velocity[1] == pytest.approx(20, abs=1e-3)
    others = [0, 2, 3, 5, 6]
    assert fits[1][:, others] == pytest.approx(fits[0][:, others], rel=1e-6)


@pytest.mark.parametrize("backend", ["mpfit", "scipy"])
def test_nan_spectra_fail_alone(backend):
    """
    An all-NaN first spectrum and one with a NaN on the line fail, and the rest are fitted.

    mpfit builds its parameter template from the first spectrum, so an
    all-NaN first spectrum must not stop the others.
    """
    on_line = _blend(WV)
    on_line[np.argmax(on_line)] = np.nan
    rows = [np.full(N_WAVE, np.nan), _blend(WV), on_line, _blend(WV)]
    config = FitConfig(components=CONFIG.components, backend=backend)
    data, _, failed = fit_cube_gauss(_cube(rows), n_jobs=1, fit_config=config,
                                     return_failed=True)
    assert failed[:, 0].tolist() == [True, False, True, False]
    velocity = (data[[1, 3], 0, 1] / 195.119e-8 - 1) * C
    assert velocity == pytest.approx(20, abs=1e-3)


def test_a_noisy_blend_of_three_free_lines_fits_the_brightest_where_it_is():
    """
    Lines at 195.05, 195.119 and 195.20 A, peaks 500, 1000 and 300, noise 10.

    The primary is the fainter line at 195.05 A, so the peak pixel is not
    its. Every fit ends within 10 km/s of the brightest line's velocity.
    """
    rng = np.random.default_rng(10)
    rests = (195.05e-8, 195.119e-8, 195.20e-8)
    wave = 195.13e-8 + (np.arange(N_WAVE) - (N_WAVE - 1) / 2) * STEP
    velocity = rng.uniform(-50, 50, 50)
    doppler = 1 + velocity[:, None] / C
    data = 10.0 + sum(peak * np.exp(-0.5 * ((wave - rest * doppler) / 0.025e-8) ** 2)
                      for peak, rest in zip((500.0, 1000.0, 300.0), rests))
    data = data + 10.0 * rng.standard_normal(data.shape)
    config = FitConfig(components=[FitComponent(rest * 1e8 * u.AA) for rest in rests])
    fits, _, failed = fit_cube_gauss(_cube(data, 195.13e-8), n_jobs=1, fit_config=config,
                                     return_failed=True)
    assert not failed.any()
    fitted = (fits[:, 0, 4] / rests[1] - 1) * C
    assert np.abs(fitted - velocity).max() < 10
