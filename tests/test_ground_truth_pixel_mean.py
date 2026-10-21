"""The ground truth is the line whose pixel means match the noiseless cube.

Fitting the Gaussian at pixel centres instead pulls a narrow line toward the
middle of its pixel, by up to 0.9 km/s at a sigma of 0.28 pixels.
"""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube
from scipy.special import erf

from eclipse.fitting import FitComponent, FitConfig, fit_cube_gauss, ground_truth_summary

REST = 195.119 * u.AA
BLEND = 195.179 * u.AA
STEP = 0.0169 * u.AA
N_WAVE = 41
C = const.c.to_value(u.km / u.s)


def _wavelengths():
    return REST + (np.arange(N_WAVE) - (N_WAVE - 1) / 2) * STEP


def _pixel_means(lines, background=1.0):
    """Exact mean over every pixel of the (peak, centre, sigma) Gaussians."""
    edges = _wavelengths()[:, np.newaxis] + np.array([-0.5, 0.5]) * STEP
    total = np.full(N_WAVE, background)
    for peak, centre, sigma in lines:
        scaled = ((edges - centre) / (np.sqrt(2) * sigma)).decompose().value
        total += (peak * sigma * np.sqrt(np.pi / 2) / STEP).decompose().value * (
            erf(scaled[:, 1]) - erf(scaled[:, 0]))
    return total


def _cube(profile):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP.to_value(u.cm), 0.4, 0.16]
    wcs.wcs.crpix = [(N_WAVE + 1) / 2, 1.0, 1.0]
    wcs.wcs.crval = [REST.to_value(u.cm), 0.0, 0.0]
    return NDCube(np.tile(profile, (2, 2, 1)), wcs=wcs, unit=u.erg / (u.s * u.cm**2 * u.sr * u.cm),
                  meta={"rest_wav": REST})


def test_the_truth_of_a_line_narrower_than_a_pixel_is_where_it_is():
    sigma, shift = 0.28 * STEP, 0.3 * STEP
    cube = _cube(_pixel_means([(100.0, REST + shift, sigma)]))
    truth = (shift / REST).decompose().value * C
    summary = ground_truth_summary(cube, n_jobs=1)
    (component,) = summary["components"].values()
    assert not summary["failed"].any()
    assert component["velocity"].to_value(u.km / u.s) == pytest.approx(truth, abs=1e-4)
    # A fit at the pixel centres is off in both centre and width.
    sampled, units = fit_cube_gauss(cube, n_jobs=1)
    sampled_velocity = ((sampled[..., 1] * units[1] - REST) / REST).decompose().value * C
    assert np.all(np.abs(sampled_velocity - truth) > 0.3)
    assert np.all(sampled[..., 2] * units[2] > 1.1 * sigma)


@pytest.mark.parametrize("backend", ["scipy", "mpfit"])
def test_the_truth_of_a_narrow_blend_is_where_its_lines_are(backend):
    sigma, shift = 0.35 * STEP, -0.2 * STEP
    doppler = 1 + (shift / REST).decompose().value
    cube = _cube(_pixel_means([(100.0, REST * doppler, sigma), (40.0, BLEND * doppler, sigma)]))
    config = FitConfig(components=[FitComponent(REST), FitComponent(BLEND, tie_center=0,
                                                                    tie_width=0)],
                       backend=backend)
    summary = ground_truth_summary(cube, fit_config=config, n_jobs=1)
    assert not summary["failed"].any()
    for name, component in summary["components"].items():
        assert component["velocity"].to_value(u.km / u.s) == pytest.approx(
            (doppler - 1) * C, abs=1e-3), name
