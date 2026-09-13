"""The spatial PSF no longer treats the Sun outside the field as dark.

convolve2d(..., mode="same") fills everything beyond the array with zeros.
Along the wavelength axis that is right, because the grid runs several sigma
past the line. Along the slit it is not: the raster is a window onto a Sun
that carries on, so the rows just inside the edge really do receive PSF
contributions from emission the zero fill throws away.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.config import Simulation, Telescope_EUVST
from euvst_response.radiometric import apply_focusing_optics_psf

REST = 195.119 * u.Angstrom
N_SCAN, N_SLIT, N_WAVE = 2, 24, 32


def _cube(data):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLT-TAN", "HPLN-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [0.0169, 0.16, 0.2]
    wcs.wcs.crpix = [N_WAVE / 2.0, N_SLIT / 2.0, 1.0]
    wcs.wcs.crval = [REST.to_value(u.Angstrom), 0.0, 0.0]
    return NDCube(data, wcs=wcs, unit=u.photon / u.pix,
                  meta={"rest_wav": REST})


def _uniform_slit_cube(value=100.0):
    """Uniform along the slit, a Gaussian in wavelength well inside the grid."""
    lam = np.arange(N_WAVE) - N_WAVE / 2.0
    profile = value * np.exp(-0.5 * (lam / 2.0) ** 2)
    return _cube(np.tile(profile, (N_SCAN, N_SLIT, 1)))


TEL = Telescope_EUVST()


def test_a_field_uniform_along_the_slit_keeps_its_edge_rows():
    """The clearest statement of the bug: this field has no edges to speak of.

    Every row along the slit is identical, so blurring along the slit cannot
    change anything. Under zero fill the outer rows lose flux anyway.
    """
    cube = _uniform_slit_cube()
    replicated = apply_focusing_optics_psf(cube, TEL, boundary="replicate")

    middle = replicated.data[0, N_SLIT // 2, :]
    for row in range(N_SLIT):
        assert np.allclose(replicated.data[0, row, :], middle, rtol=1e-12)


def test_zero_fill_darkens_the_outer_rows_by_the_kernel_weight():
    """Quantifies what the old default cost, against the kernel itself.

    The deficit in row r is the fraction of the Gaussian kernel's weight that
    falls off the end of the array, computed here from the kernel rather than
    quoted, so the numbers cannot drift apart from the PSF they describe.
    """
    cube = _uniform_slit_cube()
    zero_filled = apply_focusing_optics_psf(cube, TEL, boundary="zero")
    replicated = apply_focusing_optics_psf(cube, TEL, boundary="replicate")

    fwhm = TEL.psf_params[0].to_value(u.pixel)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = max(7, int(np.ceil(6 * sigma)))
    half = (half + 1) // 2 if half % 2 == 0 else half // 2
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    for row in range(4):
        lost = kernel[offsets < -row].sum()
        ratio = (zero_filled.data[0, row, :].sum()
                 / replicated.data[0, row, :].sum())
        assert ratio == pytest.approx(1.0 - lost, abs=1e-6)

    # The edge row loses about a third of the kernel, as the issue reports.
    assert kernel[offsets < 0].sum() == pytest.approx(0.32, abs=0.02)


def test_the_interior_is_untouched_by_the_choice():
    """Only rows within a kernel half-width of an edge can differ."""
    rng = np.random.RandomState(20260913)
    cube = _cube(rng.uniform(10.0, 200.0, (N_SCAN, N_SLIT, N_WAVE)))

    zero_filled = apply_focusing_optics_psf(cube, TEL, boundary="zero")
    replicated = apply_focusing_optics_psf(cube, TEL, boundary="replicate")

    interior = slice(4, N_SLIT - 4)
    assert np.allclose(zero_filled.data[:, interior, :],
                       replicated.data[:, interior, :], rtol=1e-12)
    # and the edges really are different, or the test above proves nothing
    assert not np.allclose(zero_filled.data[:, 0, :], replicated.data[:, 0, :])


def test_the_spectral_axis_is_still_zero_filled():
    """Established in #44: there is no flux at the ends of the grid to lose."""
    data = np.zeros((N_SCAN, N_SLIT, N_WAVE))
    data[:, :, 0] = 1000.0     # all the flux against the blue edge
    cube = _cube(data)

    out = apply_focusing_optics_psf(cube, TEL, boundary="replicate")
    assert out.data.sum() < 0.75 * data.sum()


def test_replication_does_not_invent_flux_in_a_dark_field():
    """Continuing a dark edge outward must stay dark."""
    data = np.zeros((N_SCAN, N_SLIT, N_WAVE))
    data[:, N_SLIT // 2, N_WAVE // 2] = 1.0
    out = apply_focusing_optics_psf(_cube(data), TEL, boundary="replicate")

    # A point source well inside the field keeps all of its flux, and none of
    # it appears at the edge rows the replication reaches.
    assert out.data.sum() == pytest.approx(data.sum(), rel=1e-9)
    assert out.data[:, 0, :].sum() == pytest.approx(0.0, abs=1e-12)


def test_the_uniform_intensity_path_is_unaffected():
    """It skips the spatial convolution entirely, so the boundary is moot."""
    cube = _uniform_slit_cube()
    a = apply_focusing_optics_psf(cube, TEL, convolve_spatial=False,
                                  boundary="replicate")
    b = apply_focusing_optics_psf(cube, TEL, convolve_spatial=False,
                                  boundary="zero")
    assert np.array_equal(a.data, b.data)


def test_an_unknown_boundary_is_refused():
    with pytest.raises(ValueError, match="replicate"):
        apply_focusing_optics_psf(_uniform_slit_cube(), TEL, boundary="edge")


def test_the_simulation_default_is_replicate():
    assert Simulation().psf_boundary == "replicate"


def test_the_simulation_rejects_an_unknown_boundary():
    with pytest.raises(ValueError, match="psf_boundary"):
        Simulation(psf_boundary="reflect")


def test_scan_positions_stay_independent():
    """The PSF acts within one detector frame; exposures are taken in turn."""
    data = np.zeros((N_SCAN, N_SLIT, N_WAVE))
    data[0, N_SLIT // 2, N_WAVE // 2] = 1.0
    out = apply_focusing_optics_psf(_cube(data), TEL, boundary="replicate")
    assert out.data[1].sum() == pytest.approx(0.0, abs=1e-12)
