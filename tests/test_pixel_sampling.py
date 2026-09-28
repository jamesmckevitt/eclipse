"""Each detector pixel holds the mean of the scene over its footprint, through the PSF.

A single snapshot was sampled bilinearly at the pixel centres, so structure
finer than a pixel came out wherever the centres happened to fall; and the
PSF was applied to the pixels, which drags a line narrower than a pixel
toward the middle of the pixel it is in. The scene now goes onto the pixels
through the PSF in one step, from its own cells.
"""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube
from scipy.signal import fftconvolve

from euvst_response.config import Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.data_processing import rebin_atmosphere
from euvst_response.fitting import fit_cube_gauss
from euvst_response.monte_carlo import simulate_once
from euvst_response.radiometric import (apply_focusing_optics_psf, spectral_psf_fwhm,
                                        spectral_psf_margin)
from euvst_response.sampling import centred_edges, pixel_weights
from euvst_response.utils import _fwhm_to_sigma, angle_to_distance

REST = 195.119 * u.AA
RADIANCE = u.erg / (u.s * u.cm**2 * u.sr * u.cm)
C = const.c.to_value(u.km / u.s)


def _blurred_means(cells, pixels, sigma, width):
    """The weights by brute force: each cell blurred on a fine grid, then averaged over each pixel."""
    step = min(sigma if sigma else np.inf, width if width else np.inf,
               np.diff(cells).min(), np.diff(pixels).min()) / 400
    reach = 12 * sigma + width
    x = np.arange(pixels[0] - reach - 1.0, pixels[-1] + reach + 1.0, step)
    kernel_x = np.arange(-reach, reach + step, step)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2) if sigma else (np.abs(kernel_x) < step / 2) * 1.0
    if width:
        kernel = fftconvolve(kernel, (np.abs(kernel_x) <= width / 2) * 1.0, mode="same")
    kernel /= kernel.sum()
    weights = np.empty((pixels.size - 1, cells.size - 1))
    for b in range(cells.size - 1):
        light = fftconvolve(((x >= cells[b]) & (x < cells[b + 1])) * 1.0, kernel, mode="same")
        for p in range(pixels.size - 1):
            inside = (x >= pixels[p]) & (x < pixels[p + 1])
            weights[p, b] = light[inside].mean()
    return weights


def test_with_no_blur_a_pixel_holds_the_share_of_it_each_cell_covers():
    cells = np.arange(0, 3.01, 0.3)
    pixels = np.arange(0, 3.01, 0.4)
    weights = pixel_weights(cells, pixels)
    # The pixel from 0.4 to 0.8 is 0.2 of the second cell and 0.2 of the third.
    assert weights[1, 1] == pytest.approx(0.5) and weights[1, 2] == pytest.approx(0.5)
    assert weights.sum(axis=1) == pytest.approx(1.0, abs=1e-12)
    assert np.count_nonzero(weights[1]) == 2


@pytest.mark.parametrize("sigma, width", [(0.25, 0.0), (0.2, 0.6), (0.0, 0.5)])
def test_the_blurred_weights_are_the_blur_integrated_over_cell_and_pixel(sigma, width):
    cells = np.array([0.0, 0.3, 0.45, 1.0, 1.1])
    pixels = np.array([-0.5, 0.2, 0.6, 1.4])
    assert pixel_weights(cells, pixels, sigma, width) == pytest.approx(
        _blurred_means(cells, pixels, sigma, width), abs=2e-3)


def test_blurring_keeps_every_cells_light():
    cells = np.linspace(0, 1, 11)
    pixels = np.linspace(-3, 4, 36)
    for sigma, width in ((0.3, 0.0), (0.1, 0.5)):
        weights = pixel_weights(cells, pixels, sigma, width)
        assert weights.T @ np.diff(pixels) == pytest.approx(np.diff(cells), rel=1e-12)


def test_a_scene_that_goes_on_past_its_edges_is_as_bright_at_them():
    cells = np.linspace(0, 1, 11)
    pixels = np.linspace(0, 1, 6)
    assert pixel_weights(cells, pixels, 0.2, extend=True).sum(axis=1) == pytest.approx(1.0)
    assert pixel_weights(cells, pixels, 0.2).sum(axis=1)[0] < 0.8


def _scene(data, cell, wavelength_step=0.0169 * u.AA / 10):
    """A cube as the synthesis writes it, with square cells *cell* across."""
    ny, nx, nl = data.shape
    size = angle_to_distance(cell).to_value(u.Mm)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
    wcs.wcs.cunit = ["Angstrom", "Mm", "Mm"]
    wcs.wcs.crpix = [(nl + 1) / 2, (nx + 1) / 2, (ny + 1) / 2]
    wcs.wcs.crval = [REST.to_value(u.AA), 0.0, 0.0]
    wcs.wcs.cdelt = [wavelength_step.to_value(u.AA), size, size]
    return NDCube(data, wcs=wcs, unit=RADIANCE, meta={"rest_wav": REST})


def test_structure_finer_than_a_pixel_is_averaged_over_it():
    """Columns alternately dark and twice as bright, four to a 0.4 arcsec slit: every slit sees the mean."""
    ny, nx = 32, 64
    data = np.ones((ny, nx, 21))
    data[:, ::2] = 0.0
    data[:, 1::2] = 2.0
    out = rebin_atmosphere(_scene(data, 0.1 * u.arcsec), Detector_SWC(),
                           Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1))
    spectral = out.data[:, :, 2:-2]
    assert spectral == pytest.approx(1.0, rel=1e-12)


def _narrow_line(offset_pixels, sigma_pixels=0.28, n_fine=301):
    """A line narrower than a pixel, off the middle of one by a known amount, in every cell."""
    pixel = 0.0169 * u.AA
    step = pixel / 10
    wavelength = REST + (np.arange(n_fine) - (n_fine - 1) / 2) * step
    centre = REST + offset_pixels * pixel
    profile = np.exp(-0.5 * ((wavelength - centre) / (sigma_pixels * pixel)).decompose().value ** 2)
    return _scene(np.broadcast_to(profile, (8, 24, n_fine)).copy(), 0.1 * u.arcsec, step), \
        (offset_pixels * pixel / REST).decompose().value * C


def _fitted_velocity(cube):
    fit, units = fit_cube_gauss(cube, n_jobs=1)
    return float(np.median(((fit[..., 1] * units[1] - REST) / REST).decompose().value * C))


def test_a_narrow_line_keeps_its_place_through_the_psf():
    """Blurred on the pixels, a line 0.28 pixels wide came out 2 km/s from where it was."""
    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.2 * u.arcsec, ncpu=1, psf=True)
    scene, truth = _narrow_line(0.3)
    through = rebin_atmosphere(scene, det, sim, tel=tel)
    assert through.meta["psf_applied"]
    old = apply_focusing_optics_psf(rebin_atmosphere(scene, det, sim), tel, det, sim)
    assert abs(_fitted_velocity(old) - truth) > 1.0
    assert _fitted_velocity(through) == pytest.approx(truth, abs=0.1)


def test_the_psf_moves_light_but_counts_its_photons_at_its_own_wavelength():
    """The run counts a pixel's photons at the pixel's wavelength; the light the PSF brings there keeps its own."""
    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=1.6 * u.arcsec, ncpu=1, psf=True)
    scene, _ = _narrow_line(0.3)
    through = rebin_atmosphere(scene, det, sim, tel=tel)

    def photons_per_radiance(wavelength):
        return np.array([w.to_value(u.cm) * tel.ea_and_throughput(w).cgs.value
                         for w in wavelength])

    fine = scene.axis_world_coords(2)[0]
    pixels = through.axis_world_coords(2)[0]
    step_fine = np.diff(fine.to_value(u.cm))[0]
    step_pixel = (det.wvl_res * u.pix).to_value(u.cm)
    # Everything in the middle column, at the middle of the slit.
    row, column = through.data.shape[0] // 2, through.data.shape[1] // 2
    counted = (through.data[row, column] * photons_per_radiance(pixels)).sum() * step_pixel
    emitted = (scene.data[4, 4] * photons_per_radiance(fine)).sum() * step_fine
    assert counted == pytest.approx(emitted, rel=1e-9)
    # The window grows by the margin a wider slit's PSF needs.
    plain = rebin_atmosphere(scene, det, Simulation(instrument="SWC", slit_width=1.6 * u.arcsec,
                                                    ncpu=1))
    assert through.data.shape[2] == plain.data.shape[2] + 2 * spectral_psf_margin(
        tel, det, 1.6 * u.arcsec)


def test_the_monte_carlo_does_not_blur_a_cube_that_went_through_the_psf():
    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(instrument="SWC", slit_width=0.2 * u.arcsec, ncpu=1, psf=True, noise=False)
    scene, _ = _narrow_line(0.3)
    through = rebin_atmosphere(scene, det, sim, tel=tel)
    steps = simulate_once(through, 1 * u.s, det, tel, sim)
    photons_pixels, photons_focused = steps[3], steps[4]
    assert np.array_equal(photons_pixels.data, photons_focused.data)


def test_the_telescopes_blur_across_the_slit_brings_in_light_from_beside_it():
    """One bright column: without the blur only the slit over it sees it."""
    det = Detector_SWC()
    data = np.zeros((64, 64, 21))
    data[:, 28:32] = 1.0  # under one slit position, 0.4 arcsec
    scene = _scene(data, 0.1 * u.arcsec)
    sim = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, psf=True,
                     psf_boundary="zero")
    sharp = rebin_atmosphere(scene, det, sim, tel=Telescope_EUVST())
    blurred = rebin_atmosphere(scene, det, sim, tel=Telescope_EUVST(psf_across_slit=1.0 * u.arcsec))
    row = sharp.data.shape[0] // 2
    columns = sharp.data[row].sum(axis=-1)
    assert np.count_nonzero(columns > 1e-12 * columns.max()) == 1
    seen = blurred.data[row].sum(axis=-1) / columns.max()
    expected = pixel_weights(centred_edges(64, 0.1), centred_edges(16, 0.4),
                             _fwhm_to_sigma(1.0)) @ ((np.arange(64) >= 28) & (np.arange(64) < 32))
    assert seen == pytest.approx(expected, rel=1e-9, abs=1e-15)
    assert np.count_nonzero(seen > 0.01) >= 3
