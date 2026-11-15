"""Fits do not depend on how many pieces each row is fitted in."""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from eclipse import fitting
from eclipse.fitting import FitComponent, FitConfig, fit_cube_gauss

STEP = 0.0169e-8
N_WAVE = 41
START = 195.15e-8 - (N_WAVE - 1) / 2 * STEP
BLEND = [FitComponent(195.119 * u.AA),
         FitComponent(195.179 * u.AA, tie_center=0, tie_width=0)]


def _cube(n_slit=3, n_scan=7):
    """Noisy Fe XII 195.119 and 195.179 blends, shaped (n_slit, n_scan, N_WAVE)."""
    wave = START + np.arange(N_WAVE) * STEP
    prof = (10.0 + 100.0 * np.exp(-0.5 * ((wave - 195.119e-8) / 0.03e-8) ** 2)
            + 30.0 * np.exp(-0.5 * ((wave - 195.179e-8) / 0.03e-8) ** 2))
    data = np.random.default_rng(5).poisson(prof, (n_slit, n_scan, N_WAVE)).astype(float)
    data[1, 2, 5] = np.nan
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [STEP, 0.4, 0.16]
    wcs.wcs.crpix = [1.0, 1.0, 1.0]
    wcs.wcs.crval = [START, 0.0, 0.0]
    return NDCube(data, wcs=wcs, unit=u.DN / u.pix)


@pytest.mark.parametrize("fit_config", [None, FitConfig(components=BLEND)])
def test_the_fits_do_not_depend_on_the_pieces_each_row_is_cut_into(fit_config, monkeypatch):
    cube = _cube()
    uncertainty = np.sqrt(np.maximum(cube.data, 1.0))
    fits = []
    for workers in (1, 100):
        # 1 worker gives one piece per row, 100 workers one piece per spectrum.
        monkeypatch.setattr(fitting, "effective_n_jobs", lambda n_jobs: workers)
        values, units, failed = fit_cube_gauss(cube, n_jobs=1, fit_config=fit_config,
                                               return_failed=True, uncertainty=uncertainty)
        fits.append((values, failed))
    assert np.array_equal(fits[0][0], fits[1][0], equal_nan=True)
    assert np.array_equal(fits[0][1], fits[1][1])
    assert fits[0][0].shape[:2] == cube.data.shape[:2]


def _intensity_cube(n_slit=3, n_scan=2, n_wave=16, peak=2.0e4):
    """A line of the same radiance in every pixel, shaped (n_slit, n_scan, n_wave)."""
    step, rest = 0.0169 * u.AA, 195.119 * u.AA
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [step.to_value(u.AA), 0.2, 0.16]
    wcs.wcs.crpix = [n_wave / 2.0, 1.0, n_slit / 2.0]
    wcs.wcs.crval = [rest.to_value(u.AA), 0.0, 0.0]
    lam = np.arange(n_wave) - n_wave / 2.0
    data = np.tile(peak * np.exp(-0.5 * (lam / 1.5) ** 2), (n_slit, n_scan, 1))
    return NDCube(data, wcs=wcs, unit=u.erg / (u.cm**2 * u.s * u.sr * u.cm),
                  meta={"rest_wav": rest})


def test_simulating_ahead_of_the_fits_gives_the_same_monte_carlo(monkeypatch):
    import importlib

    from eclipse.config import Detector_SWC, Simulation, Telescope_EUVST

    # The module, not the function `eclipse.monte_carlo` of the same name.
    mc = importlib.import_module("eclipse.monte_carlo")
    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.2 * u.arcsec, ncpu=1)
    runs = []
    for workers in (1, 2):
        # Two workers make the next iteration be simulated in a thread; the
        # fits themselves still run here, as ncpu is 1.
        monkeypatch.setattr(mc, "effective_n_jobs", lambda n_jobs: workers)
        np.random.seed(11)
        runs.append(mc.monte_carlo(_intensity_cube(), 40 * u.s, det, tel, sim, n_iter=3))
    for serial, ahead in zip(*runs):
        if serial is None:
            assert ahead is None
        elif isinstance(serial, dict):
            name = "195.1190 Angstrom"
            for key in ("mean", "std", "first"):
                assert np.array_equal(serial["components"][name]["velocity"][key].value,
                                      ahead["components"][name]["velocity"][key].value,
                                      equal_nan=True)
            assert np.array_equal(serial["failed_fits"], ahead["failed_fits"])
        else:
            assert np.array_equal(serial.data, ahead.data)
