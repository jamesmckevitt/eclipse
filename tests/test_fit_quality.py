"""Fits that converge, and fits that cannot have measured a line counted as failed.

The multi-component scipy fit stopped at tolerances of 1e-4, which for
centres some 195 Angstrom from zero left a blend's velocities over a km/s
short of where the fit converges. A spectrum with no line in it, a width
wider than the window, or a component outside the window came back as a
successful fit with no spread, which reads as a perfect measurement.
"""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.fitting import FitComponent, FitConfig, _unmeasurable, fit_cube_gauss

REST = 195.119 * u.Angstrom
BLEND = 195.179 * u.Angstrom
STEP = 0.0169 * u.Angstrom
N_WAVE = 31
CENTRE = 195.149 * u.Angstrom


def _cube(profile):
    """*profile* in every pixel of a (2, 3, N_WAVE) cube."""
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP.to_value(u.cm), 0.4, 0.16]
    wcs.wcs.crpix = [(N_WAVE + 1) / 2, 1.0, 1.0]
    wcs.wcs.crval = [CENTRE.to_value(u.cm), 0.0, 0.0]
    return NDCube(np.tile(profile, (2, 3, 1)), wcs=wcs, unit=u.DN / u.pix)


def _wave():
    return CENTRE + (np.arange(N_WAVE) - (N_WAVE - 1) / 2) * STEP


def _gaussian(peak, centre, sigma):
    return peak * np.exp(-0.5 * ((_wave() - centre) / sigma).decompose().value ** 2)


def _shifted(rest, velocity):
    return rest * (1 + velocity / const.c)


@pytest.mark.parametrize("backend", [None, "mpfit"])
def test_a_noiseless_blend_is_fitted_to_where_it_is(backend):
    """The default scipy lm stopped over 4 km/s short on this untied pair."""
    velocity = 10 * u.km / u.s
    sigma = 0.03 * u.Angstrom
    profile = (_gaussian(1000.0, _shifted(REST, velocity), sigma)
               + _gaussian(300.0, _shifted(BLEND, velocity), sigma) + 10.0)
    config = FitConfig(components=[FitComponent(REST), FitComponent(BLEND)], backend=backend)
    data, _, failed = fit_cube_gauss(_cube(profile), n_jobs=1, fit_config=config,
                                     return_failed=True)
    assert not failed.any()
    for index, rest in ((1, REST), (4, BLEND)):
        fitted = (data[..., index] * u.cm / rest.to(u.cm) - 1) * const.c
        assert np.abs(fitted - velocity).max() < 1e-3 * u.km / u.s
    assert data[..., 0] == pytest.approx(1000.0, rel=1e-6)
    assert data[..., 3] == pytest.approx(300.0, rel=1e-6)


def test_a_spectrum_with_no_line_is_a_failed_fit_not_a_perfect_one():
    """As a photon spectrum with no photons in it is."""
    profile = _gaussian(50.0, REST, 0.03 * u.Angstrom)
    cube = _cube(profile)
    cube.data[0, 1] = 0.0
    cube.data[1, 2] = 7.0
    _, _, failed = fit_cube_gauss(cube, n_jobs=1, return_failed=True)
    assert failed.tolist() == [[False, True, False], [False, False, True]]


def test_widths_are_positive_and_one_wider_than_the_window_fails():
    wavelength = _wave().to_value(u.cm)
    span = np.ptp(wavelength)
    params = np.array([[[10.0, 1.0, -0.2 * span, 0.0], [10.0, 1.0, 2 * span, 0.0]]])
    spectra = np.arange(2 * N_WAVE, dtype=float).reshape(1, 2, N_WAVE)
    failed = _unmeasurable(params, spectra, wavelength, 1)
    assert params[0, 0, 2] == pytest.approx(0.2 * span)
    assert failed.tolist() == [[False, True]]


def test_a_component_outside_the_window_is_refused():
    """It came back at its guess with no spread, as a perfect measurement."""
    profile = _gaussian(1000.0, REST, 0.03 * u.Angstrom) + 10.0
    config = FitConfig(components=[FitComponent(159.119 * u.Angstrom), FitComponent(BLEND)])
    with pytest.raises(ValueError, match=r"fitting.components\[0\] is at 159.119 Angstrom, "
                                         r"outside the observed window"):
        fit_cube_gauss(_cube(profile), n_jobs=1, fit_config=config)
