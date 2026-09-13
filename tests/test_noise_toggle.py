"""With noise off, the chain returns the signal the instrument measures on average.

Every random draw is replaced by its own mean, so two runs agree exactly and
the result sits where the noisy realisations scatter about. Deterministic
detector behaviour stays: DN are still rounded and still clip at the full
well, because those are not noise.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.config import Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.monte_carlo import simulate_once
from euvst_response.radiometric import (
    add_visible_stray_light,
    sample_photon_arrivals,
    to_electrons,
)

REST = 195.119 * u.Angstrom
NSCAN, NSLIT, NWAVE = 2, 3, 8


def _wcs():
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLT-TAN", "HPLN-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [0.0169, 0.16, 0.2]
    wcs.wcs.crpix = [NWAVE / 2.0, NSLIT / 2.0, 1.0]
    wcs.wcs.crval = [REST.to_value(u.Angstrom), 0.0, 0.0]
    return wcs


def _photon_cube(mean=500.0):
    return NDCube(np.full((NSCAN, NSLIT, NWAVE), mean), wcs=_wcs(),
                  unit=u.photon / u.pix, meta={"rest_wav": REST})


def _intensity_cube(peak=2.0e4):
    """A Gaussian line, in the units the synthesis stage emits."""
    lam = np.arange(NWAVE) - NWAVE / 2.0
    profile = peak * np.exp(-0.5 * (lam / 1.5) ** 2)
    data = np.tile(profile, (NSCAN, NSLIT, 1))
    return NDCube(data, wcs=_wcs(),
                  unit=u.erg / (u.cm**2 * u.s * u.sr * u.cm),
                  meta={"rest_wav": REST})


def test_photon_arrivals_return_the_mean_and_stay_continuous():
    """Rounding to whole photons would put quantisation back in."""
    cube = _photon_cube(12.4)
    out = sample_photon_arrivals(cube, noise=False)
    assert np.allclose(out.data, 12.4)
    assert out.data.dtype.kind == "f"


def test_photon_arrivals_are_still_random_by_default():
    np.random.seed(3)
    a = sample_photon_arrivals(_photon_cube()).data
    b = sample_photon_arrivals(_photon_cube()).data
    assert not np.array_equal(a, b)


def test_electrons_are_qe_times_gain_exactly():
    """Both draws collapse to their expectation: N * qe * (E_ph / w(T))."""
    det = Detector_SWC(ccd_temperature=-60 * u.deg_C)
    det.dark_current = 0 * u.electron / (u.pix * u.s)
    det.read_noise_rms = 0 * u.electron / u.pix

    photons = 500.0
    out = to_electrons(_photon_cube(photons), 1 * u.s, det, noise=False)

    temp_k = det.ccd_temperature.to_value(u.K, equivalencies=u.temperature())
    w_ev = 3.71 - 0.0006 * (temp_k - 300.0)
    e_ph_ev = 12398.419843320026 / REST.to_value(u.Angstrom)
    expected = photons * det.qe_euv * (e_ph_ev / w_ev)

    assert np.allclose(out.data, expected, rtol=1e-6)
    # and identical from one call to the next
    again = to_electrons(_photon_cube(photons), 1 * u.s, det, noise=False)
    assert np.array_equal(out.data, again.data)


def test_dark_current_still_contributes_its_expected_electrons():
    """Turning off the shot noise must not turn off the dark current."""
    det = Detector_SWC(ccd_temperature=-60 * u.deg_C)
    det.read_noise_rms = 0 * u.electron / u.pix
    t_exp = 100 * u.s

    dark_only = to_electrons(_photon_cube(0.0), t_exp, det, noise=False)
    expected = (det.dark_current * t_exp).to_value(u.electron / u.pix)
    assert np.allclose(dark_only.data, expected, rtol=1e-12)


def test_read_noise_is_dropped_not_averaged():
    """It is zero-mean, so its expectation adds nothing."""
    det = Detector_SWC(ccd_temperature=-60 * u.deg_C)
    det.dark_current = 0 * u.electron / (u.pix * u.s)
    det.read_noise_rms = 50 * u.electron / u.pix

    quiet = to_electrons(_photon_cube(0.0), 1 * u.s, det, noise=False)
    assert np.allclose(quiet.data, 0.0)


def test_stray_light_contributes_its_mean():
    det = Detector_SWC(ccd_temperature=-60 * u.deg_C)
    sim = Simulation(instrument="SWC",
                     vis_sl=1.0e6 * u.photon / (u.s * u.cm**2))
    electrons = NDCube(np.zeros((NSCAN, NSLIT, NWAVE)), wcs=_wcs(),
                       unit=u.electron / u.pix, meta={"rest_wav": REST})

    out = add_visible_stray_light(electrons, 10 * u.s, det, sim, noise=False)
    assert np.all(out.data > 0)
    again = add_visible_stray_light(electrons, 10 * u.s, det, sim, noise=False)
    assert np.array_equal(out.data, again.data)


def test_a_noiseless_run_is_reproducible_end_to_end():
    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.2 * u.arcsec, noise=False)

    np.random.seed(1)
    first = simulate_once(_intensity_cube(), 40 * u.s, det, tel, sim)[-1]
    np.random.seed(99)
    second = simulate_once(_intensity_cube(), 40 * u.s, det, tel, sim)[-1]

    assert np.array_equal(first.data, second.data)
    assert np.any(first.data > 0)


def test_the_noiseless_signal_sits_where_the_noisy_ones_scatter():
    """The whole claim: noise off gives the mean of noise on.

    Compared against the spread of the noisy runs rather than a fixed
    tolerance, so the test states the statistical claim rather than a number
    that happened to pass.
    """
    det, tel = Detector_SWC(), Telescope_EUVST()
    noisy_sim = Simulation(instrument="SWC", slit_width=0.2 * u.arcsec)
    quiet_sim = Simulation(instrument="SWC", slit_width=0.2 * u.arcsec,
                           noise=False)

    quiet = simulate_once(_intensity_cube(), 40 * u.s, det, tel, quiet_sim)[-1]

    np.random.seed(20260913)
    n_draw = 60
    draws = np.stack([
        simulate_once(_intensity_cube(), 40 * u.s, det, tel, noisy_sim)[-1].data
        for _ in range(n_draw)
    ])

    mean = draws.mean(axis=0)
    sem = draws.std(axis=0) / np.sqrt(n_draw)
    # Allow 5 standard errors, plus half a DN for the rounding in to_dn.
    assert np.all(np.abs(quiet.data - mean) <= 5.0 * sem + 0.5)


def test_noise_defaults_to_on():
    """An existing config that says nothing must keep its noise."""
    assert Simulation().noise is True
    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.2 * u.arcsec)

    np.random.seed(5)
    first = simulate_once(_intensity_cube(), 40 * u.s, det, tel, sim)[-1]
    second = simulate_once(_intensity_cube(), 40 * u.s, det, tel, sim)[-1]
    assert not np.array_equal(first.data, second.data)


def test_dn_are_still_quantised_and_still_clip():
    """Quantisation and saturation are detector behaviour, not noise."""
    det = Detector_SWC(ccd_temperature=-60 * u.deg_C)
    tel = Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.2 * u.arcsec, noise=False)

    # 1e12 erg/(s cm2 sr cm) only reaches a few hundred DN through the real
    # SWC effective area, so the line has to be far brighter than anything
    # solar to drive the detector into its full well.
    dn = simulate_once(_intensity_cube(peak=1.0e16), 40 * u.s, det, tel, sim)[-1]
    assert np.array_equal(dn.data, np.round(dn.data))
    assert dn.data.max() == det.max_dn.to_value(u.DN / u.pix)
