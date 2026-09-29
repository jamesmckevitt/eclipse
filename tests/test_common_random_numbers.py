"""With photon_shot_inverse_transform, runs that differ only in photon flux share their random numbers.

The photon draws of the first Monte Carlo iteration were shared, but the
quantum efficiency was a binomial draw, whose sampler uses a number of random
values that depends on the photon count, and the Fano spread was drawn for
the pixels with photons only, how many depending on the flux. After the first
iteration the two runs were out of step, and their noise no longer cancelled.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.config import Detector_SWC
from euvst_response.radiometric import (_binomial_inverse_transform, _vectorized_fano_noise,
                                        sample_photon_arrivals, to_electrons)

REST = 195.119 * u.Angstrom


def _photons(mean, shape=(40, 5, 30)):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [0.0169, 0.2, 0.16]
    wcs.wcs.crval = [REST.to_value(u.Angstrom), 0.0, 0.0]
    return NDCube(np.full(shape, mean), wcs=wcs, unit=u.photon / u.pix, meta={"rest_wav": REST})


def _iterations(mean, n_iter=4):
    """The electrons of each Monte Carlo iteration, from one seed."""
    det = Detector_SWC()
    np.random.seed(11)
    electrons = []
    for _ in range(n_iter):
        arrived = sample_photon_arrivals(_photons(mean), photon_shot_inverse_transform=True)
        electrons.append(to_electrons(arrived, 1 * u.s, det,
                                      photon_shot_inverse_transform=True).data.ravel())
    return electrons


def test_runs_differing_in_flux_stay_in_step_in_every_iteration():
    """At two photons a pixel, which pixels have photons at all changes with the flux."""
    for low, high in zip(_iterations(2.0), _iterations(2.4)):
        assert np.corrcoef(low, high)[0, 1] > 0.9


def test_the_quantum_efficiency_drawn_by_inverse_transform_is_binomial():
    np.random.seed(3)
    detected = _binomial_inverse_transform(np.full(200_000, 40), 0.76)
    assert detected.dtype == np.int64 and detected.min() >= 0 and detected.max() <= 40
    assert detected.mean() == pytest.approx(40 * 0.76, rel=2e-3)
    assert detected.var() == pytest.approx(40 * 0.76 * 0.24, rel=2e-2)


def test_the_fano_spread_drawn_for_every_pixel_is_the_same_spread():
    det = Detector_SWC()
    photons = np.full(200_000, 100.0)
    photons[::2] = 0.0
    spread = {}
    for every_pixel in (False, True):
        np.random.seed(5)
        electrons = _vectorized_fano_noise(photons, REST, det, every_pixel=every_pixel)
        assert np.all(electrons[::2] == 0.0)
        spread[every_pixel] = (electrons[1::2].mean(), electrons[1::2].var())
    assert spread[True][0] == pytest.approx(spread[False][0], rel=1e-4)
    assert spread[True][1] == pytest.approx(spread[False][1], rel=2e-2)
