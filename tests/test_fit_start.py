"""Starting values for a single-Gaussian fit that noise does not mislead."""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.fitting import FitConfig, _fit_one, _guess_params, fit_cube_gauss
from euvst_response.utils import gaussian

C = 299792.458
REST, STEP, SIGMA, BACK = 195.119, 0.0169, 0.03, 5.0
WV = REST + (np.arange(31) - 15) * STEP


def test_a_clean_line_starts_at_its_centre_width_and_background():
    peak, centre, sigma, back = _guess_params(WV, gaussian(WV, 100.0, REST, SIGMA, BACK))
    assert centre == REST
    assert back == pytest.approx(BACK, abs=1.0)
    assert peak == pytest.approx(100.0, rel=0.05)
    assert sigma == pytest.approx(SIGMA, rel=0.25)


def test_a_single_bright_pixel_away_from_the_line_does_not_take_it():
    """Starting at the brightest pixel would put the centre on a spike brighter than the line."""
    spectrum = gaussian(WV, 100.0, REST, SIGMA, BACK)
    spectrum[3] += 130.0
    _, centre, _, _ = _guess_params(WV, spectrum)
    assert centre == REST


def test_noise_above_half_maximum_away_from_the_line_does_not_widen_it():
    """Counting every pixel above half maximum lets one far from the line stretch the width."""
    spectrum = gaussian(WV, 100.0, REST, SIGMA, BACK)
    spectrum[28] += 60.0
    _, _, sigma, _ = _guess_params(WV, spectrum)
    assert sigma == pytest.approx(SIGMA, rel=0.25)


def test_a_spectrum_with_nothing_above_its_background_starts_as_before():
    peak, centre, sigma, back = _guess_params(WV, np.full(WV.size, BACK))
    assert (peak, centre, back) == (0.0, WV[0], BACK)
    assert sigma == pytest.approx((WV[-1] - WV[0]) / 10)


def test_at_a_peak_of_three_times_the_noise_no_fit_ends_far_away():
    """From a simple start, 1 per cent ended over 300 km/s away; the rest scattered 37 km/s."""
    rng = np.random.default_rng(7)
    spectra = gaussian(WV, 30.0, REST, SIGMA, BACK) + 10.0 * rng.standard_normal((500, WV.size))
    fits = [_fit_one(WV, spectrum) for spectrum in spectra]
    velocity = np.array([(params[1] / REST - 1) * C for params, _ in fits])
    assert all(succeeded for _, succeeded in fits)
    assert np.all(np.abs(velocity) < 300)
    assert velocity.std() < 25


def test_a_nan_pixel_on_the_line_counts_as_background():
    spectrum = gaussian(WV, 100.0, REST, SIGMA, BACK)
    spectrum[16] = np.nan
    peak, centre, sigma, back = _guess_params(WV, spectrum)
    assert np.all(np.isfinite([peak, centre, sigma, back]))
    assert centre == pytest.approx(REST, abs=STEP)
    assert back == pytest.approx(BACK, abs=1.0)


def test_an_all_nan_spectrum_starts_at_zero():
    peak, _, sigma, back = _guess_params(WV, np.full(WV.size, np.nan))
    assert (peak, back) == (0.0, 0.0)
    assert sigma == pytest.approx((WV[-1] - WV[0]) / 10)


@pytest.mark.parametrize("backend", [None, "mpfit"], ids=["scipy", "mpfit"])
def test_nan_spectra_fail_alone(backend):
    """An all-NaN first spectrum and a NaN on the line fail; the clean spectrum is fitted."""
    line = gaussian(WV, 100.0, REST, SIGMA, BACK)
    on_line = line.copy()
    on_line[15] = np.nan
    data = np.stack([np.full(WV.size, np.nan), on_line, line])[:, None, :]
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP, 0.4, 0.16]
    wcs.wcs.crpix = [1.0, 1.0, 1.0]
    wcs.wcs.crval = [WV[0], 0.0, 0.0]
    config = None if backend is None else FitConfig(backend=backend)
    fits, _, failed = fit_cube_gauss(NDCube(data, wcs=wcs, unit=u.DN / u.pix), n_jobs=1,
                                     fit_config=config, return_failed=True)
    assert failed[:, 0].tolist() == [True, True, False]
    assert (fits[2, 0, 1] * 1e8 / REST - 1) * C == pytest.approx(0, abs=1e-3)
