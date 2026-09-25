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
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.config import Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.frame import (
    apply_spectral_psf,
    detect,
    digitise,
    expose_with_wavelength,
    photons_from_lines,
    photons_from_spectrum,
    pixel_solid_angle,
    thermal_width,
)
from euvst_response.radiometric import (
    apply_focusing_optics_psf,
    spectral_psf_fwhm,
    spectral_psf_reach,
    to_electrons,
)
from euvst_response.readout import FocalPlane_SWC, ReadoutSequence, expose
from euvst_response.utils import _fwhm_to_sigma

H_ERG_S = 6.62607015e-27
C_CM_S = 2.99792458e10
HC_ERG_CM = H_ERG_S * C_CM_S
AU_CM = 1.495978707e13
ARCSEC_RAD = np.pi / (180.0 * 3600.0)
K_B_ERG_K = 1.380649e-16
U_G = 1.66053906892e-24
ERG_PER_EV = 1.602176634e-12

# Energy to liberate one electron-hole pair in silicon at -60 C, the SWC
# operating temperature: w(T) = 3.71 - 0.0006 (T - 300) eV.
W_EV_AT_MINUS_60 = 3.71 - 0.0006 * (213.15 - 300.0)


def photon_energy_ev(wavelength_angstrom):
    return HC_ERG_CM / (wavelength_angstrom * 1e-8) / ERG_PER_EV

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
    telescope, det, slit = Telescope_EUVST(), Detector_SWC(), SLIT_WIDTH * u.arcsec
    rows = photons_from_lines(fp, "left", StubTelescope(), slit,
                              [195.119] * u.Angstrom, [1.0] * u.erg / (u.s * u.cm**2 * u.sr),
                              [0.001] * u.Angstrom)
    blurred = apply_spectral_psf(rows, telescope, det, slit)
    assert blurred.value.sum() == pytest.approx(rows.value.sum(), rel=1e-6)
    assert blurred.value.max() < rows.value.max()
    # The 0.4 arcsec slit's response is 3.35 rows of FWHM, the optics and the
    # slit's image in quadrature, not the 2.54 of the 0.2 arcsec slit.
    fwhm = spectral_psf_fwhm(telescope, det, slit)
    assert fwhm == pytest.approx(3.35, abs=0.01)
    # Convolution adds variances, so the line comes out wider by the variance
    # of the response itself: that Gaussian at whole rows, out to its reach.
    reach = spectral_psf_reach(telescope, det, slit)
    offsets = np.arange(-reach, reach + 1)
    kernel = np.exp(-0.5 * (offsets / _fwhm_to_sigma(fwhm)) ** 2)
    kernel /= kernel.sum()
    widening = row_variance(blurred.value) - row_variance(rows.value)
    assert widening == pytest.approx(np.sum(offsets**2 * kernel), rel=0.01)


@pytest.mark.parametrize("spectral_psf", ["quadrature", "convolution"])
@pytest.mark.parametrize("slit", [0.2, 0.4, 1.6])
def test_a_frame_is_blurred_as_a_synthesis_is(slit, spectral_psf):
    """The rows of a frame get the spectral response a synthesis through the same slit gets."""
    telescope, det = Telescope_EUVST(), Detector_SWC()
    sim = Simulation(instrument="SWC", slit_width=slit * u.arcsec, spectral_psf=spectral_psf)
    rows = np.zeros(61)
    rows[30] = 1.0
    frame = apply_spectral_psf(rows / u.s, telescope, det, slit * u.arcsec, spectral_psf)
    cube = NDCube(rows[np.newaxis, np.newaxis, :], wcs=WCS(naxis=3), unit=1 / u.s)
    synthesis = apply_focusing_optics_psf(cube, telescope, det, sim, convolve_spatial=False)
    np.testing.assert_allclose(frame.value, synthesis.data[0, 0], rtol=1e-12, atol=1e-15)


def test_light_blurred_off_the_end_of_the_rows_is_lost():
    # Nothing lies beyond the rows given, so an impulse in the first row keeps
    # the half of the response that lands on them, and its centre.
    telescope, det, slit = Telescope_EUVST(), Detector_SWC(), SLIT_WIDTH * u.arcsec
    edge = apply_spectral_psf(np.eye(61)[0] / u.s, telescope, det, slit).value
    centre = apply_spectral_psf(np.eye(61)[30] / u.s, telescope, det, slit).value.max()
    assert edge.sum() == pytest.approx((1.0 + centre) / 2.0, rel=1e-12)


def test_the_edge_rows_get_the_light_from_just_off_the_chip():
    # A line in the gap, two rows past the left CCD's butted edge, still
    # reaches its last rows through the spectral response.
    fp = FocalPlane_SWC()
    telescope, det, slit = Telescope_EUVST(), Detector_SWC(), SLIT_WIDTH * u.arcsec
    reach = spectral_psf_reach(telescope, det, slit)
    line = ([fp.wavelength(fp.n_rows + 1, "left").to_value(u.Angstrom)] * u.Angstrom,
            [1.0] * u.erg / (u.s * u.cm**2 * u.sr), [0.001] * u.Angstrom)

    def on_chip(margin):
        rows = photons_from_lines(fp, "left", StubTelescope(), slit, *line,
                                  lit_only=False, margin=margin)
        return rows, apply_spectral_psf(rows, telescope, det, slit, margin=margin).value

    rows, blurred = on_chip(reach)
    without, _ = on_chip(0)
    assert rows.size == fp.n_rows + 2 * reach
    np.testing.assert_array_equal(rows[reach:-reach].value, without.value)
    assert without.value[-1] < 1e-6 * rows.value.max()
    assert blurred.size == fp.n_rows
    assert blurred[-1] > 0.05 * rows.value.max()
    # Any margin that reaches as far as the response gives the same rows.
    np.testing.assert_allclose(on_chip(3 * reach)[1], blurred, rtol=1e-12, atol=0)


def test_a_margin_has_to_reach_as_far_as_the_response():
    telescope, det, slit = Telescope_EUVST(), Detector_SWC(), SLIT_WIDTH * u.arcsec
    with pytest.raises(ValueError, match="less than the"):
        apply_spectral_psf(np.ones(100) / u.s, telescope, det, slit, margin=1)
    with pytest.raises(ValueError, match="baffle comes after"):
        photons_from_lines(FocalPlane_SWC(), "left", StubTelescope(), slit,
                           [195.119] * u.Angstrom, [1.0] * u.erg / (u.s * u.cm**2 * u.sr),
                           [0.02] * u.Angstrom, margin=5)


def test_each_pixel_gets_the_wavelength_of_its_mean_photon_energy():
    # Two rows lit at different wavelengths. Without a shutter a packet
    # collects from both, and since a photon carries h c / lambda the
    # wavelength of the mean energy is the photon-weighted harmonic mean.
    n_rows = 8
    wavelength = np.linspace(170.0, 210.0, n_rows) * u.Angstrom
    first, second = np.zeros((n_rows, 1)), np.zeros((n_rows, 1))
    first[5], second[2] = 100.0, 40.0
    sequence = ReadoutSequence(shutter=False, dump_rows=n_rows, parallel_overscan_rows=2)
    photons, mean = expose_with_wavelength(first + second, wavelength, 1.0 * u.s, sequence)
    a, b = (expose(rate, 1.0 * u.s, sequence) for rate in (first, second))
    # The smear is convolved by FFT, so pixels holding only smear carry
    # rounding of order 1e-16 of the brightest pixel.
    np.testing.assert_allclose(photons, a + b, rtol=1e-9)
    lam = wavelength.to_value(u.Angstrom)
    expected = (a + b) / (a / lam[5] + b / lam[2])
    np.testing.assert_allclose(mean.to_value(u.Angstrom), expected, rtol=1e-9)
    # With a shutter the lit rows keep their own wavelength, and a pixel with
    # no photons its row's, or the last image row's in the overscan.
    sequence = ReadoutSequence(shutter=True, dump_rows=n_rows, parallel_overscan_rows=2)
    photons, mean = expose_with_wavelength(first + second, wavelength, 1.0 * u.s, sequence)
    np.testing.assert_allclose(mean.to_value(u.Angstrom)[:, 0],
                               np.concatenate([lam, [lam[-1]] * 2]), rtol=1e-12)


def test_the_spectral_psf_refuses_an_unknown_mode():
    with pytest.raises(ValueError, match="quadrature"):
        apply_spectral_psf(np.ones(10) / u.s, Telescope_EUVST(), Detector_SWC(),
                           0.4 * u.arcsec, "gaussian")


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


def test_a_wavelength_the_telescope_cannot_see_is_an_error():
    # NaN from a throughput table would otherwise spread to every row.
    class TelescopeWithTables(StubTelescope):
        def ea_and_throughput(self, wavelength):
            w = wavelength.to_value(u.Angstrom)
            return np.where((w >= 170.0) & (w <= 214.0), EFFECTIVE_AREA, np.nan) * u.cm**2

    fp = FocalPlane_SWC()
    with pytest.raises(ValueError, match="no effective area at 169.9000"):
        photons_from_lines(fp, "left", TelescopeWithTables(), SLIT_WIDTH * u.arcsec,
                           [169.9, 195.119] * u.Angstrom,
                           [1.0, 1.0] * u.erg / (u.s * u.cm**2 * u.sr), [0.02, 0.02] * u.Angstrom)


def test_the_telescope_gives_a_spectrum_the_area_at_each_wavelength():
    # A frame asks for the effective area of a whole spectrum in one call.
    telescope = Telescope_EUVST()
    wavelength = np.linspace(170.5, 211.5, 9) * u.Angstrom
    together = telescope.ea_and_throughput(wavelength).to_value(u.cm**2)
    one_by_one = [telescope.ea_and_throughput(w).to_value(u.cm**2) for w in wavelength]
    assert together.shape == (9,)
    np.testing.assert_allclose(together, one_by_one, rtol=1e-14, atol=0)
    # Every stage answers an array with one value per wavelength.
    assert telescope.throughput(wavelength).unit == u.dimensionless_unscaled
    assert telescope.throughput(wavelength).shape == (9,)
    assert telescope.filter.total_throughput(wavelength).shape == (9,)
    for stage in (telescope.primary_mirror_efficiency, telescope.grating_efficiency,
                  telescope.microroughness_efficiency):
        assert np.shape(stage(wavelength)) == (9,)
        assert np.ndim(stage(wavelength[0])) == 0


def test_the_throughput_tables_are_read_once(monkeypatch):
    # Reading all five again for every wavelength made a spectrum take minutes.
    import pathlib

    from euvst_response import config

    monkeypatch.setattr(config, "_THROUGHPUT_TABLES", {})
    reads = []
    read_text = pathlib.Path.read_text

    def counted(self, *args, **kwargs):
        reads.append(self.name)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", counted)
    telescope = Telescope_EUVST()
    for w in np.linspace(171.0, 211.0, 20) * u.Angstrom:
        telescope.ea_and_throughput(w)
    assert len(reads) == len(set(reads)) == 5


def test_a_line_list_must_be_consistent():
    fp = FocalPlane_SWC()
    with pytest.raises(ValueError, match="wavelength, an intensity and a width"):
        photons_from_lines(fp, "left", StubTelescope(), SLIT_WIDTH * u.arcsec,
                           [195.119, 192.03] * u.Angstrom,
                           [1.0] * u.erg / (u.s * u.cm**2 * u.sr), [0.02] * u.Angstrom)


def test_each_row_converts_photons_at_its_own_energy():
    # A 170 A photon liberates 212 / 170 times the electrons of a 212 A one.
    det = Detector_SWC(ccd_temperature=-60 * u.deg_C)
    det.dark_current = 0 * u.electron / (u.pixel * u.s)
    electrons = detect(np.full((2, 3), 100.0), [170.0, 212.0] * u.Angstrom, 1.0 * u.s,
                       det, noise=False)
    for row, wavelength in enumerate([170.0, 212.0]):
        expected = 100.0 * det.qe_euv * photon_energy_ev(wavelength) / W_EV_AT_MINUS_60
        assert electrons[row] == pytest.approx(expected, rel=1e-6)
    assert electrons[0, 0] / electrons[1, 0] == pytest.approx(212.0 / 170.0, rel=1e-6)


def test_a_pixel_can_carry_its_own_wavelength():
    # Smear puts photons from other rows into a pixel; given per pixel, the
    # wavelength converts each pixel at its own energy.
    det = Detector_SWC(ccd_temperature=-60 * u.deg_C)
    det.dark_current = 0 * u.electron / (u.pixel * u.s)
    wavelengths = np.array([[170.0, 212.0], [190.0, 200.0]])
    electrons = detect(np.full((2, 2), 50.0), wavelengths * u.Angstrom, 1.0 * u.s,
                       det, noise=False)
    expected = 50.0 * det.qe_euv * photon_energy_ev(wavelengths) / W_EV_AT_MINUS_60
    assert electrons == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError, match="per row or per pixel"):
        detect(np.zeros((2, 2)), [190.0, 191.0, 192.0] * u.Angstrom, 1.0 * u.s, det)


def test_each_row_collects_dark_current_for_its_own_time():
    det = Detector_SWC(ccd_temperature=-60 * u.deg_C)
    dark = det.dark_current.to_value(u.electron / (u.pixel * u.s))
    electrons = detect(np.zeros((3, 2)), [190.0, 191.0, 192.0] * u.Angstrom,
                       [1.0, 2.0, 4.0] * u.s, det, noise=False)
    assert electrons[:, 0] == pytest.approx(dark * np.array([1.0, 2.0, 4.0]), rel=1e-9)
    assert electrons[:, 1] == pytest.approx(electrons[:, 0], rel=1e-12)


def test_detect_agrees_with_the_cube_chain_at_one_wavelength():
    # A frame whose rows all record one wavelength is the cube chain's case.
    det = Detector_SWC(ccd_temperature=-60 * u.deg_C)
    photons = np.full((4, 5), 250.0)
    mine = detect(photons, np.full(4, 195.119) * u.Angstrom, 2.0 * u.s, det, noise=False)
    cube = NDCube(photons[np.newaxis], wcs=WCS(naxis=3), unit=u.photon / u.pix,
                  meta={"rest_wav": 195.119 * u.Angstrom})
    chain = to_electrons(cube, 2.0 * u.s, det, noise=False).data[0]
    assert mine == pytest.approx(chain, rel=1e-12)


def test_noise_draws_around_the_same_mean():
    np.random.seed(20260918)
    det = Detector_SWC(ccd_temperature=-60 * u.deg_C)
    photons = np.full((200, 200), 400.0)
    wavelengths = np.full(200, 192.03) * u.Angstrom
    noisy = detect(photons, wavelengths, 3.0 * u.s, det)
    clean = detect(photons, wavelengths, 3.0 * u.s, det, noise=False)
    assert noisy.mean() == pytest.approx(clean.mean(), rel=2e-3)
    assert noisy.std() > 0.0


def test_whole_photons_are_required_with_noise_on():
    det = Detector_SWC()
    with pytest.raises(ValueError, match="whole numbers"):
        detect(np.full((1, 1), 2.5), [192.0] * u.Angstrom, 1.0 * u.s, det)


def test_digitise_divides_by_the_gain_and_clips():
    # 2.78 electrons per DN, and nothing above 65535.
    det = Detector_SWC()
    dn = digitise(np.array([[0.0, 2.78, 2780.0, 1.0e9]]), det)
    assert dn.tolist() == [[0.0, 1.0, 1000.0, 65535.0]]
