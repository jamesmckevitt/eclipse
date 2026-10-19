"""Starting values for a multi-component fit: width, wavelength order and NaN pixels."""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.fitting import FitComponent, FitConfig, _guess_multi_params, fit_cube_gauss

STEP = 0.0169e-8
N_WAVE = 41
CENTRE = 195.15e-8
WV = CENTRE + (np.arange(N_WAVE) - (N_WAVE - 1) / 2) * STEP
FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))
CONFIG = FitConfig(components=[FitComponent(195.119 * u.AA),
                               FitComponent(195.179 * u.AA, tie_center=0, tie_width=0)])


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
    assert sigma == pytest.approx(n_pixels * STEP * FWHM_TO_SIGMA)


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


def test_an_all_nan_spectrum_starts_at_zero():
    guess = _guess_multi_params(WV, np.full(N_WAVE, np.nan), CONFIG)
    assert guess[0] == 0 and guess[-1] == 0
    assert np.all(np.isfinite(guess))


def test_two_components_at_one_wavelength_start_apart():
    """Started identical, a core and a wing would stay identical through the fit."""
    config = FitConfig(components=[FitComponent(195.119 * u.AA),
                                   FitComponent(195.119 * u.AA, name="wing")])
    guess = _guess_multi_params(WV, _box(3), config)
    assert guess[1] == guess[4]
    assert guess[3] < guess[0]


def _blend_cube(decreasing, rows=None):
    """Fe XII 195.119 and 195.179 at 20 km/s, 3:1, in a (len, 1, N_WAVE) cube."""
    sign = -1 if decreasing else 1
    wave = CENTRE + sign * (np.arange(N_WAVE) - (N_WAVE - 1) / 2) * STEP
    doppler = 1 + 20 / const.c.to_value(u.km / u.s)
    prof = (10.0 + 100.0 * np.exp(-0.5 * ((wave - 195.119e-8 * doppler) / 0.03e-8) ** 2)
            + 30.0 * np.exp(-0.5 * ((wave - 195.179e-8 * doppler) / 0.03e-8) ** 2))
    data = np.tile(prof, (3, 1, 1)) if rows is None else rows(prof)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [sign * STEP, 0.4, 0.16]
    wcs.wcs.crpix = [(N_WAVE + 1) / 2, 1.0, 1.0]
    wcs.wcs.crval = [CENTRE, 0.0, 0.0]
    return NDCube(data, wcs=wcs, unit=u.DN / u.pix)


@pytest.mark.parametrize("settings", [{"backend": "mpfit"}, {}], ids=["mpfit", "scipy"])
def test_a_blend_on_a_decreasing_wavelength_axis_fits_as_on_an_increasing_one(settings):
    """Both backends fit it to the same values, at the velocity it was made with."""
    config = FitConfig(components=CONFIG.components, **settings)
    fits = []
    for decreasing in (False, True):
        data, _, failed = fit_cube_gauss(_blend_cube(decreasing), n_jobs=1, fit_config=config,
                                         return_failed=True)
        assert not failed.any()
        fits.append(data[:, 0])
    assert fits[1] == pytest.approx(fits[0], rel=1e-6)
    velocity = (fits[1][:, 1] / 195.119e-8 - 1) * const.c.to_value(u.km / u.s)
    assert velocity == pytest.approx(20, abs=1e-3)


def test_an_all_nan_spectrum_fails_alone():
    """The spectra either side are still fitted."""
    def rows(prof):
        return np.stack([prof, np.full(N_WAVE, np.nan), prof])[:, None, :]

    data, _, failed = fit_cube_gauss(_blend_cube(False, rows), n_jobs=1, fit_config=CONFIG,
                                     return_failed=True)
    assert failed[:, 0].tolist() == [False, True, False]


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
    doppler = 1 + velocity[:, None] / const.c.to_value(u.km / u.s)
    data = 10.0 + sum(peak * np.exp(-0.5 * ((wave - rest * doppler) / 0.025e-8) ** 2)
                      for peak, rest in zip((500.0, 1000.0, 300.0), rests))
    data = data + 10.0 * rng.standard_normal(data.shape)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP, 0.4, 0.16]
    wcs.wcs.crpix = [(N_WAVE + 1) / 2, 1.0, 1.0]
    wcs.wcs.crval = [195.13e-8, 0.0, 0.0]
    config = FitConfig(components=[FitComponent(rest * 1e8 * u.AA) for rest in rests])
    fits, _, failed = fit_cube_gauss(NDCube(data[:, None, :], wcs=wcs, unit=u.DN / u.pix),
                                     n_jobs=1, fit_config=config, return_failed=True)
    assert not failed.any()
    fitted = (fits[:, 0, 4] / rests[1] - 1) * const.c.to_value(u.km / u.s)
    assert np.abs(fitted - velocity).max() < 10
