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
        # 1 worker gives 3 pieces per row of 7 spectra, 100 workers one piece per spectrum.
        monkeypatch.setattr(fitting, "effective_n_jobs", lambda n_jobs: workers)
        values, units, failed = fit_cube_gauss(cube, n_jobs=1, fit_config=fit_config,
                                               return_failed=True, uncertainty=uncertainty)
        fits.append((values, failed))
    assert np.array_equal(fits[0][0], fits[1][0], equal_nan=True)
    assert np.array_equal(fits[0][1], fits[1][1])
    assert fits[0][0].shape[:2] == cube.data.shape[:2]
