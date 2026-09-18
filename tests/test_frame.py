"""Check that a spectrum lands on the detector rows with the right flux.

The frame builder is the standard radiometric equation with the pixel bandwidth
replaced by an integral between row boundaries:

    photons per second per pixel = I * Omega_pix * A_eff / E_photon

    E_photon  = h c / lam                      erg per photon
    Omega_pix = (w_slit * w_pix) / au^2        sr, the patch of Sun a pixel sees

Every expected value below is written out from that expression with the
constants as literals, and the telescope is a stub with a constant effective
area, so a failure means the frame builder moved rather than that two spellings
of the same helper disagree. One test compares a flat spectrum against the
existing radiometric chain, which is checked separately in
test_radiometric_chain.py.
"""
import astropy.units as u
import numpy as np
import pytest

from euvst_response.frame import (
    apply_spectral_psf,
    photons_from_lines,
    photons_from_spectrum,
    pixel_solid_angle,
    thermal_width,
)
from euvst_response.readout import FocalPlane_SWC

H_ERG_S = 6.62607015e-27
C_CM_S = 2.99792458e10
HC_ERG_CM = H_ERG_S * C_CM_S
AU_CM = 1.495978707e13
ARCSEC_RAD = np.pi / (180.0 * 3600.0)
K_B_ERG_K = 1.380649e-16
U_G = 1.66053906892e-24

PLATE_SCALE = 0.159          # arcsec per pixel along the slit
SLIT_WIDTH = 0.4             # arcsec, science case 2.1.1
EFFECTIVE_AREA = 0.3         # cm^2, the stub telescope


class StubTelescope:
    """A telescope with one effective area, so the arithmetic is by hand."""

    psf_params = [2.66 * u.pixel, 2.54 * u.pixel]

    def ea_and_throughput(self, wavelength):
        return EFFECTIVE_AREA * u.cm**2


def expected_solid_angle():
    return (PLATE_SCALE * ARCSEC_RAD * AU_CM) * (SLIT_WIDTH * ARCSEC_RAD * AU_CM) / AU_CM**2


def expected_photons(intensity, wavelength_angstrom):
    """Photons per second per pixel for a line of this integrated radiance."""
    energy = HC_ERG_CM / (wavelength_angstrom * 1e-8)
    return intensity * expected_solid_angle() * EFFECTIVE_AREA / energy


def row_variance(rows):
    """How spread out a frame's rows are, in rows squared."""
    index = np.arange(rows.size)
    centre = (index * rows).sum() / rows.sum()
    return ((index - centre) ** 2 * rows).sum() / rows.sum()


def test_the_pixel_sees_the_slit_width_by_the_plate_scale():
    fp = FocalPlane_SWC()
    omega = pixel_solid_angle(fp, SLIT_WIDTH * u.arcsec).to_value(u.sr)
    assert omega == pytest.approx(expected_solid_angle(), rel=1e-6)


def test_a_line_keeps_its_flux():
    fp = FocalPlane_SWC()
    rows = photons_from_lines(fp, "left", StubTelescope(), SLIT_WIDTH * u.arcsec,
                              [195.119] * u.Angstrom, [5.0e4] * u.erg / (u.s * u.cm**2 * u.sr),
                              [0.02] * u.Angstrom)
    assert rows.to_value(1 / u.s).sum() == pytest.approx(expected_photons(5.0e4, 195.119), rel=1e-4)


def test_a_line_lands_on_its_own_row():
    fp = FocalPlane_SWC()
    rows = photons_from_lines(fp, "left", StubTelescope(), SLIT_WIDTH * u.arcsec,
                              [195.119] * u.Angstrom, [5.0e4] * u.erg / (u.s * u.cm**2 * u.sr),
                              [0.02] * u.Angstrom)
    _, row = fp.row_of_wavelength(195.119 * u.Angstrom)
    assert int(np.argmax(rows.value)) == pytest.approx(round(row), abs=1)


def test_a_narrow_line_spreads_over_a_few_rows():
    # 0.0169 A is one row at 195 A, so three rows reach 1.5 sigma either side of
    # the middle row's centre. How much of the line that holds depends on where
    # in its row the line sits: erf(1.5 / sqrt 2) = 0.866 on a row centre, and
    # [erf(1 / sqrt 2) + erf(2 / sqrt 2)] / 2 = 0.819 on a row boundary.
    fp = FocalPlane_SWC()
    rows = photons_from_lines(fp, "left", StubTelescope(), SLIT_WIDTH * u.arcsec,
                              [195.119] * u.Angstrom, [1.0] * u.erg / (u.s * u.cm**2 * u.sr),
                              [0.0169] * u.Angstrom)
    peak = int(np.argmax(rows.value))
    total = rows.value.sum()
    assert 0.818 < rows.value[peak - 1:peak + 2].sum() / total < 0.867
    assert rows.value[peak - 6:peak + 7].sum() / total == pytest.approx(1.0, abs=1e-3)


def test_the_baffle_keeps_the_dark_rows_dark():
    # 165 A is on the left CCD but outside the band the baffle passes.
    fp = FocalPlane_SWC()
    arguments = (fp, "left", StubTelescope(), SLIT_WIDTH * u.arcsec,
                 [165.0] * u.Angstrom, [1.0e4] * u.erg / (u.s * u.cm**2 * u.sr),
                 [0.02] * u.Angstrom)
    assert photons_from_lines(*arguments).value.sum() == 0.0
    assert photons_from_lines(*arguments, lit_only=False).value.sum() > 0.0


def test_a_flat_spectrum_fills_each_row_by_the_wavelength_it_covers():
    fp = FocalPlane_SWC()
    grid = np.linspace(190.0, 200.0, 20001) * u.Angstrom
    radiance = np.full(grid.size, 1.0e3) * u.erg / (u.s * u.cm**2 * u.sr * u.Angstrom)
    rows = photons_from_spectrum(fp, "left", StubTelescope(), SLIT_WIDTH * u.arcsec,
                                 grid, radiance).to_value(1 / u.s)
    row = 1861
    width = abs(fp.row_edges("left")[row + 1] - fp.row_edges("left")[row]).to_value(u.Angstrom)
    wavelength = fp.wavelength(row, "left").to_value(u.Angstrom)
    assert rows[row] == pytest.approx(expected_photons(1.0e3 * width, wavelength), rel=2e-3)


def test_the_two_paths_agree_on_the_same_line():
    # The same Gaussian, once as a line and once sampled as a spectrum.
    fp = FocalPlane_SWC()
    centre, sigma, intensity = 195.119, 0.03, 2.0e4
    grid = np.linspace(centre - 1.0, centre + 1.0, 200001)
    profile = (intensity / (sigma * np.sqrt(2 * np.pi))
               * np.exp(-0.5 * ((grid - centre) / sigma) ** 2))
    as_spectrum = photons_from_spectrum(
        fp, "left", StubTelescope(), SLIT_WIDTH * u.arcsec, grid * u.Angstrom,
        profile * u.erg / (u.s * u.cm**2 * u.sr * u.Angstrom)).value
    as_line = photons_from_lines(
        fp, "left", StubTelescope(), SLIT_WIDTH * u.arcsec, [centre] * u.Angstrom,
        [intensity] * u.erg / (u.s * u.cm**2 * u.sr), [sigma] * u.Angstrom).value
    assert as_spectrum.sum() == pytest.approx(as_line.sum(), rel=1e-3)
    assert as_spectrum == pytest.approx(as_line, abs=1e-3 * as_line.max())


def test_a_flat_spectrum_matches_the_radiometric_chain():
    # With evenly spaced rows and a flat effective area, a row of this frame is
    # the same quantity the cube chain produces for one pixel: I / E_ph * A_eff
    # * Omega_pix * dlam_pix, with dlam_pix the row's own width.
    linear = FocalPlane_SWC(dispersion=(198.886, 1.2518518518518519))
    grid = np.linspace(194.0, 196.0, 20001) * u.Angstrom
    radiance = np.full(grid.size, 4.0e3) * u.erg / (u.s * u.cm**2 * u.sr * u.Angstrom)
    rows = photons_from_spectrum(linear, "left", StubTelescope(), SLIT_WIDTH * u.arcsec,
                                 grid, radiance).to_value(1 / u.s)
    _, row = linear.row_of_wavelength(195.119 * u.Angstrom)
    row = int(round(row))
    wavelength = linear.wavelength(row, "left").to_value(u.Angstrom)
    chain = (4.0e3 * 0.0169) * expected_solid_angle() * EFFECTIVE_AREA / (HC_ERG_CM / (wavelength * 1e-8))
    assert rows[row] == pytest.approx(chain, rel=1e-3)


def test_the_spectral_psf_conserves_flux_and_widens_a_line():
    fp = FocalPlane_SWC()
    rows = photons_from_lines(fp, "left", StubTelescope(), SLIT_WIDTH * u.arcsec,
                              [195.119] * u.Angstrom, [1.0] * u.erg / (u.s * u.cm**2 * u.sr),
                              [0.001] * u.Angstrom)
    blurred = apply_spectral_psf(rows, StubTelescope())
    assert blurred.value.sum() == pytest.approx(rows.value.sum(), rel=1e-6)
    assert blurred.value.max() < rows.value.max()
    # Convolution adds variances, so the line comes out wider by the response's
    # own sigma: 2.54 pixels of FWHM is 2.54 / (2 sqrt(2 ln 2)) = 1.078 rows.
    sigma = 2.54 / (2 * np.sqrt(2 * np.log(2)))
    widening = row_variance(blurred.value) - row_variance(rows.value)
    assert widening == pytest.approx(sigma**2, rel=0.01)


def test_thermal_width_is_the_doppler_width():
    # sigma_lam = lam * sqrt(k T / m) / c, for iron at 2 MK.
    width = thermal_width(195.119 * u.Angstrom, 2.0e6 * u.K, 55.845 * u.u)
    speed = np.sqrt(K_B_ERG_K * 2.0e6 / (55.845 * U_G))
    assert width.to_value(u.Angstrom) == pytest.approx(195.119 * speed / C_CM_S, rel=1e-4)


def test_a_non_thermal_speed_adds_in_quadrature():
    hot = thermal_width(195.119 * u.Angstrom, 2.0e6 * u.K, 55.845 * u.u,
                        non_thermal=20.0 * u.km / u.s).to_value(u.Angstrom)
    plain = thermal_width(195.119 * u.Angstrom, 2.0e6 * u.K, 55.845 * u.u).to_value(u.Angstrom)
    extra = 195.119 * 20.0e5 / C_CM_S
    assert hot == pytest.approx(np.sqrt(plain**2 + extra**2), rel=1e-6)


def test_a_line_list_must_be_consistent():
    fp = FocalPlane_SWC()
    with pytest.raises(ValueError, match="wavelength, an intensity and a width"):
        photons_from_lines(fp, "left", StubTelescope(), SLIT_WIDTH * u.arcsec,
                           [195.119, 192.03] * u.Angstrom,
                           [1.0] * u.erg / (u.s * u.cm**2 * u.sr), [0.02] * u.Angstrom)
