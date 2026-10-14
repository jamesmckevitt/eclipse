"""Tests of the slit's ends and fiducials blocking the light.

With `simulation.slit_mask` on, the pixels along the slit average only the
light that gets through, blurred along the slit before the slit.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.config import Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.data_processing import rebin_atmosphere
from euvst_response.sampling import centred_edges, pixel_weights, pixel_weights_through
from euvst_response.utils import _parse_section, angle_to_distance

REST = 195.119 * u.AA
RADIANCE = u.erg / (u.s * u.cm**2 * u.sr * u.cm)


def _open_share(pixels, openings):
    """Share of each pixel the openings cover."""
    low, high = pixels[:-1, np.newaxis], pixels[1:, np.newaxis]
    covered = np.clip(np.minimum(high, openings[:, 1]) - np.maximum(low, openings[:, 0]), 0, None)
    return covered.sum(axis=1) / (pixels[1:] - pixels[:-1])


def test_one_opening_past_every_pixel_gives_the_plain_weights():
    cells = centred_edges(50, 0.3)
    pixels = centred_edges(80, 0.159)
    for sigma in (0.0, 0.2):
        through = pixel_weights_through(cells, pixels, [[-100.0, 100.0]], sigma)
        assert through == pytest.approx(pixel_weights(cells, pixels, sigma), rel=1e-12, abs=1e-15)


def test_a_uniform_scene_shows_the_open_share_of_each_pixel():
    """Blurred, a uniform scene stays uniform, so each pixel holds the share of it that is open."""
    cells = centred_edges(60, 0.25)
    pixels = centred_edges(90, 0.159)
    openings = np.array([[-7.0, 0.33], [1.13, 3.0], [3.8, 7.0]])
    for sigma in (0.0, 0.18):
        seen = pixel_weights_through(cells, pixels, openings, sigma, extend=True) @ np.ones(60)
        assert seen == pytest.approx(_open_share(pixels, openings), rel=1e-10, abs=1e-12)
    assert np.any(seen == 0.0) and np.any((seen > 0.0) & (seen < 1.0))


def test_the_weights_match_a_fine_grid():
    """Compare with blurring cells on a fine grid, cutting out the gap and averaging per pixel."""
    rng = np.random.default_rng(3)
    cells = centred_edges(12, 0.4)
    pixels = centred_edges(30, 0.159)
    openings = np.array([[-3.0, -0.31], [0.49, 3.0]])
    sigma = 0.15
    values = rng.uniform(0.5, 2.0, 12)

    step = 1e-4
    x = np.arange(pixels[0], pixels[-1], step) + step / 2
    from scipy.special import erf
    scale = np.sqrt(2) * sigma
    light = sum(v * 0.5 * (erf((x - lo) / scale) - erf((x - hi) / scale))
                for v, lo, hi in zip(values, cells[:-1], cells[1:]))
    inside = (x[:, np.newaxis] > openings[:, 0]) & (x[:, np.newaxis] < openings[:, 1])
    light *= np.any(inside, axis=1)
    pixel = np.searchsorted(pixels, x, side="right") - 1
    brute = np.bincount(pixel, weights=light, minlength=30) / np.bincount(pixel, minlength=30)

    exact = pixel_weights_through(cells, pixels, openings, sigma) @ values
    # The fine grid puts each edge to within one step, 1/1600 of a pixel.
    assert exact == pytest.approx(brute, rel=2e-3, abs=1e-6)


def test_the_openings_of_each_slit():
    tel = Telescope_EUVST()
    for slit, gap in ((0.2, 0.8), (0.4, 0.8), (0.8, 0.8), (1.6, 1.6)):
        openings = tel.slit_openings(slit * u.arcsec).to_value(u.arcsec)
        expected = [[-150.0, -25.0 - gap / 2], [-25.0 + gap / 2, 70.0 - gap / 2],
                    [70.0 + gap / 2, 150.0]]
        assert openings == pytest.approx(np.array(expected))
    with pytest.raises(ValueError, match="0.2, 0.4, 0.8 and 1.6 arcsec"):
        tel.slit_openings(0.3 * u.arcsec)


@pytest.mark.parametrize("settings, message", [
    (dict(fiducial_centres=[149.5 * u.arcsec]), "inside the"),
    (dict(fiducial_centres=[0 * u.arcsec, 1.0 * u.arcsec]), "apart"),
    (dict(fiducial_centres=[-25 * u.mm]), "angles along the slit"),
    (dict(fiducial_lengths=[0.8 * u.arcsec] * 3), "four lengths"),
    (dict(fiducial_lengths=[0.8 * u.arcsec] * 3 + [0 * u.arcsec]), "four lengths"),
    (dict(slit_length=-300 * u.arcsec), "more than zero"),
])
def test_fiducials_that_cannot_be_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        Telescope_EUVST(**settings)


def test_the_mask_is_off_by_default_and_for_swc_only():
    assert Simulation().slit_mask is False
    with pytest.raises(ValueError, match="slit_mask is for SWC"):
        Simulation(instrument="EIS", slit_width=1 * u.arcsec, slit_mask=True)


def test_the_fiducial_lists_are_one_setting_not_a_sweep():
    fixed, sweep = _parse_section({"fiducial_centres": ["-25 arcsec", "70 arcsec"],
                                   "fiducial_lengths": ["0.8 arcsec", "0.8 arcsec",
                                                        "0.8 arcsec", "1.6 arcsec"]}, "telescope")
    assert not sweep
    assert fixed["fiducial_centres"] == [-25 * u.arcsec, 70 * u.arcsec]
    Telescope_EUVST(**fixed)


def _scene(data, cell):
    ny, nx, nl = data.shape
    size = angle_to_distance(cell).to_value(u.Mm)
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
    wcs.wcs.cunit = ["Angstrom", "Mm", "Mm"]
    wcs.wcs.crpix = [(nl + 1) / 2, (nx + 1) / 2, (ny + 1) / 2]
    wcs.wcs.crval = [REST.to_value(u.AA), 0.0, 0.0]
    wcs.wcs.cdelt = [0.0169 / 10, size, size]
    return NDCube(data, wcs=wcs, unit=RADIANCE, meta={"rest_wav": REST})


@pytest.mark.parametrize("psf", [False, True])
def test_the_rebinned_scene_is_dark_past_the_ends_and_in_the_fiducials(psf):
    """A uniform scene 24 arcsec long through a 20 arcsec slit with gaps at -3 and +5 arcsec."""
    det = Detector_SWC()
    tel = Telescope_EUVST(slit_length=20 * u.arcsec,
                          fiducial_centres=[-3 * u.arcsec, 5 * u.arcsec])
    scene = _scene(np.ones((240, 4, 21)), 0.1 * u.arcsec)
    common = dict(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1, psf=psf)
    plain = rebin_atmosphere(scene, det, Simulation(**common), tel=tel)
    masked = rebin_atmosphere(scene, det, Simulation(**common, slit_mask=True), tel=tel)

    rows = plain.data.shape[0]
    pixels = centred_edges(rows, det.plate_scale_angle.to_value(u.arcsec / u.pix))
    share = _open_share(pixels, tel.slit_openings(0.4 * u.arcsec).to_value(u.arcsec))
    # With the blur along the slit replicating the edges, the plain scene is
    # uniform along it, so the mask scales each row by its open share.
    ratio = masked.data.sum(axis=(1, 2)) / plain.data.sum(axis=(1, 2))
    assert ratio == pytest.approx(share, rel=1e-9, abs=1e-12)
    assert np.count_nonzero(share == 0.0) > 0
