"""The rebinned spectral axis has the right wavelengths for any number of pixels.

With an even number of pixels the WCS reference lies between two pixels,
and taking the pixel below it would put every velocity half a pixel blue.
"""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from eclipse.config import Detector_SWC, Simulation
from eclipse.data_processing import (
    rebin_atmosphere,
    resample_ndcube_spectral_axis,
)
from eclipse.fitting import fit_cube_gauss

REST = 171.073 * u.Angstrom
C = const.c.to_value(u.km / u.s)
# The synthesis samples its window at 5 km/s; 121 and 131 samples span an
# even and an odd number of detector pixels at this wavelength.
STEP = (5 * u.km / u.s / const.c * REST).to(u.Angstrom)
PARITIES = [(121, 0), (131, 1)]


def _line_cube(n_fine, centre_offset, sigma, ny=4, nx=4, voxel=0.192):
    """Line cube as the synthesis writes it, with the same Gaussian in every
    column: data axes (y, x, wavelength), WCS axes (WAVE, SOLX, SOLY)."""
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
    wcs.wcs.cunit = ["Angstrom", "Mm", "Mm"]
    wcs.wcs.crpix = [(n_fine + 1) / 2, (nx + 1) / 2, (ny + 1) / 2]
    wcs.wcs.crval = [REST.to_value(u.Angstrom), 0.0, 0.0]
    wcs.wcs.cdelt = [STEP.to_value(u.Angstrom), voxel, voxel]
    wavelength = REST + (np.arange(n_fine) - (n_fine - 1) / 2) * STEP
    centre = REST + centre_offset
    profile = np.exp(-0.5 * ((wavelength - centre) / sigma).decompose().value ** 2)
    data = np.broadcast_to(profile, (ny, nx, n_fine)).copy()
    return NDCube(data, wcs=wcs, unit=u.erg / u.s / u.cm**2 / u.sr / u.cm,
                  meta={"rest_wav": REST})


def _centroid_velocity(cube):
    """Intensity-weighted line centre of every spectrum, as a velocity from rest."""
    wavelength = cube.axis_world_coords(2)[0].to_value(u.Angstrom)
    centre = (cube.data * wavelength).sum(axis=-1) / cube.data.sum(axis=-1)
    return (centre - REST.value) / REST.value * C


def _pixel_count(cube, resolution):
    """Number of detector pixels the resampling produces, counted the same way."""
    wavelength = cube.axis_world_coords(2)[0].to_value(resolution.unit)
    return len(np.arange(wavelength.min(), wavelength.max() + resolution.value,
                         resolution.value))


@pytest.mark.parametrize("n_fine, parity", PARITIES)
def test_the_resampled_axis_keeps_a_line_where_it_is(n_fine, parity):
    """A line off centre by a known amount is still there once resampled."""
    offset = 0.02 * u.Angstrom
    cube = _line_cube(n_fine, centre_offset=offset, sigma=0.012 * u.Angstrom)
    resolution = 0.0169 * u.Angstrom
    assert _pixel_count(cube, resolution) % 2 == parity

    resampled = resample_ndcube_spectral_axis(cube, spectral_axis=2,
                                              output_resolution=resolution, ncpu=1)

    expected = (offset / REST).decompose().value * C
    # Half a pixel here is 15 km/s; the binning itself moves the centroid by
    # far less than a tenth of that.
    assert _centroid_velocity(resampled) == pytest.approx(expected, abs=1.0)


@pytest.mark.parametrize("n_fine, parity", PARITIES)
def test_a_line_at_rest_is_fitted_at_rest_through_the_instrument(n_fine, parity):
    """A line at rest, rebinned to the detector and fitted, has zero velocity."""
    cube = _line_cube(n_fine, centre_offset=0 * u.Angstrom, sigma=0.012 * u.Angstrom,
                      ny=8, nx=8)
    det = Detector_SWC()
    sim = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1)
    assert _pixel_count(cube, det.wvl_res * u.pix) % 2 == parity

    rebinned = rebin_atmosphere(cube, det, sim)
    fit, units = fit_cube_gauss(rebinned, n_jobs=1)

    centre = fit[..., 1] * units[1]
    velocity = ((centre - REST) / REST * const.c).to_value(u.km / u.s)
    assert velocity == pytest.approx(0.0, abs=0.5)
