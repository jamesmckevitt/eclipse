"""Starting values for a single-Gaussian fit that noise does not mislead.

A simple start is misled by noise. The minimum sits below the background by
the largest noise excursion. The brightest pixel can be a noisy one far from
the line. The span from the first to the last pixel above half maximum
anywhere in the window is stretched several times over by noise far from the
line. From such a start, at a peak of a few times the noise, fits ended
hundreds of km/s away twice as often, and scattered twice as far, as from a
good start.
"""
import numpy as np
import pytest

from euvst_response.fitting import _fit_one, _guess_params
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
