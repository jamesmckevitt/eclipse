"""The radiometric chain against the standard equation.

For a pixel at wavelength `lam` viewing a spectral radiance `I`:

    N = (I * t_exp / E_ph) * A_eff * Omega_pix * dlam_pix

    E_ph      = h c / lam                         erg per photon
    Omega_pix = (w_slit * w_pix) / au^2           sr, the patch of Sun a pixel sees
    dlam_pix  = det.wvl_res                       cm per pixel

Expected values use literal constants and, where only the arithmetic is
tested, a stub telescope with a constant effective area.
"""
import inspect

import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from eclipse.config import Detector_EIS, Detector_SWC, Simulation
from eclipse.radiometric import (
    _dark_electrons,
    _detected_photons,
    _digitise,
    _read_noise,
    add_telescope_throughput,
    apply_exposure,
    intensity_to_photons,
    photons_to_pixel_counts,
    sample_photon_arrivals,
    to_dn,
    to_electrons,
)
from eclipse.utils import angle_to_distance

# CODATA values and the IAU astronomical unit, written out rather than taken
# from astropy so the expected values do not depend on the code under test.
H_ERG_S = 6.62607015e-27
C_CM_S = 2.99792458e10
HC_ERG_CM = H_ERG_S * C_CM_S
AU_CM = 1.495978707e13
ARCSEC_RAD = np.pi / (180.0 * 3600.0)
ERG_PER_EV = 1.602176634e-12

N_SCAN, N_SLIT, N_SPEC = 2, 8, 16
REST = 195.119 * u.Angstrom

# Spectral sampling of the input cube, in Angstrom. It differs on purpose from
# both detectors' wvl_res (16.9 and 22.3 mAA): the chain takes the photon
# energy from the cube's wavelength axis but the wavelength a pixel covers
# from det.wvl_res, and equal values would hide a mix-up between the two.
CDELT_ANGSTROM = 0.0200

# Spectral radiance per unit wavelength, in the synthesis output units.
RADIANCE_UNIT = u.erg / (u.cm**2 * u.s * u.sr * u.cm)
RADIANCE = 1.0e4


class ConstantAreaTelescope:
    """Telescope with a fixed effective area times throughput.

    The chain only calls `ea_and_throughput`, so this is the whole interface.
    A fixed value separates the geometry and unit arithmetic from the
    calibration curve, which has its own tests.
    """

    def __init__(self, area_cm2=1.5):
        self.area_cm2 = area_cm2

    def ea_and_throughput(self, wavelength):
        return self.area_cm2 * u.cm**2


def make_radiance_cube(value=RADIANCE, n_spec=N_SPEC, rest=REST):
    """Uniform radiance cube with a wavelength axis centred on `rest`."""
    wcs = WCS(naxis=3)
    # WCS axes are in reverse numpy order, so WCS axis 0 is the last
    # (spectral) data axis.
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [CDELT_ANGSTROM, 1.0, 1.0]
    wcs.wcs.crpix = [n_spec / 2.0, 1.0, N_SLIT / 2.0]
    wcs.wcs.crval = [rest.to_value(u.Angstrom), 0.0, 0.0]
    return NDCube(
        np.full((N_SLIT, N_SCAN, n_spec), float(value)),
        wcs=wcs,
        unit=RADIANCE_UNIT,
        meta={"rest_wav": rest},
    )


# Slit widths each instrument offers. Simulation rejects any other value, and
# the two instruments share none, so a comparison between them must include
# the slit ratio.
SLIT_WIDTHS = {
    Detector_SWC: (0.2, 0.4, 0.8, 1.6),
    Detector_EIS: (1.0, 2.0),
}
INSTRUMENT_SLITS = [(cls, slit)
                    for cls, slits in SLIT_WIDTHS.items()
                    for slit in slits]


def sim_for(detector_cls, slit_arcsec=None):
    """Simulation for `detector_cls` at one of its slit widths."""
    if slit_arcsec is None:
        slit_arcsec = SLIT_WIDTHS[detector_cls][0]
    name = "EIS" if detector_cls is Detector_EIS else "SWC"
    return Simulation(instrument=name, slit_width=slit_arcsec * u.arcsec)


def run_chain(cube, tel, det, sim, t_exp):
    """Photons per pixel from the four deterministic steps."""
    stage = apply_exposure(cube, t_exp)
    stage = intensity_to_photons(stage)
    stage = add_telescope_throughput(stage, tel)
    return photons_to_pixel_counts(
        stage,
        det.wvl_res,
        det.plate_scale_length,
        angle_to_distance(sim.slit_width),
    )


def literal_wavelength_axis_cm(n_spec=N_SPEC, rest=REST):
    """Wavelength axis, in cm, of the cube from make_radiance_cube.

    FITS pixel coordinates are one-based, so the value at numpy index `i` is
    `crval + (i + 1 - crpix) * cdelt`. It is written out rather than read
    from the cube so that test_wavelength_axis_follows_the_fits_convention
    can check the pipeline sees the intended axis.
    """
    crpix = n_spec / 2.0
    i = np.arange(n_spec)
    return (rest.to_value(u.Angstrom)
            + (i + 1 - crpix) * CDELT_ANGSTROM) * 1e-8


def expected_photons_per_pixel(radiance, t_exp, area_cm2, det, sim,
                               lam_cm=None):
    """Photons per pixel from the standard equation, with literal constants.

    `lam_cm` is the wavelength of each pixel, not the rest wavelength, since
    the photon energy is hc / lambda at the pixel's own wavelength. Across a
    16 pixel window at 195 Angstrom this is a 0.13 per cent gradient in
    photon count.
    """
    if lam_cm is None:
        lam_cm = literal_wavelength_axis_cm()
    e_ph = HC_ERG_CM / np.asarray(lam_cm)

    # Linear size at 1 au of the angles a pixel spans. The pipeline uses the
    # exact 2 au tan(theta / 2), not the small-angle form; they differ by
    # theta^2 / 12, two parts in 1e12 at 1 arcsec (see
    # test_small_angle_correction_is_quadratic).
    slit_cm = 2.0 * AU_CM * np.tan(0.5 * sim.slit_width.to_value(u.rad))
    pix_rad = (det.plate_scale_angle * (1 * u.pix)).to_value(u.rad)
    pix_cm = 2.0 * AU_CM * np.tan(0.5 * pix_rad)
    omega_sr = slit_cm * pix_cm / AU_CM**2

    dlam_cm = (det.wvl_res * (1 * u.pix)).to_value(u.cm)

    return radiance * t_exp.to_value(u.s) / e_ph * area_cm2 * omega_sr * dlam_cm


@pytest.mark.parametrize("detector_cls,slit_arcsec", INSTRUMENT_SLITS)
@pytest.mark.parametrize("t_exp_s", [1.0, 60.0])
@pytest.mark.parametrize("area_cm2", [0.25, 1.5])
def test_photons_per_pixel_matches_standard_equation(detector_cls, slit_arcsec,
                                                     t_exp_s, area_cm2):
    """Photons per pixel from the chain match the standard equation.

    Run at every slit width of each instrument, since the slit sets the
    solid angle a pixel sees and enters the result linearly.
    """
    det = detector_cls()
    sim = sim_for(detector_cls, slit_arcsec)
    tel = ConstantAreaTelescope(area_cm2)
    t_exp = t_exp_s * u.s

    got = run_chain(make_radiance_cube(), tel, det, sim, t_exp)
    got_per_pix = got.data * got.unit
    assert got_per_pix.unit.is_equivalent(u.photon / u.pix)

    expected = expected_photons_per_pixel(RADIANCE, t_exp, area_cm2, det, sim)
    actual = got_per_pix.to_value(u.photon / u.pix)

    # atol=0 on purpose: the default 1e-8 exceeds these photon counts at short
    # exposures and would hide a relative error of any size.
    assert np.allclose(actual, expected[np.newaxis, np.newaxis, :],
                       rtol=1e-10, atol=0.0)


def test_wavelength_axis_follows_the_fits_convention():
    """The pipeline reads the wavelength axis this file means to write.

    The other tests build their expected photon counts from
    literal_wavelength_axis_cm. If its one-based FITS convention were wrong,
    the expected values would all be wrong together and could still agree.
    """
    cube = make_radiance_cube()
    from_cube = cube.axis_world_coords(2)[0].to_value(u.cm)
    assert np.allclose(from_cube, literal_wavelength_axis_cm(), rtol=1e-12,
                       atol=0.0)


def test_photons_scale_linearly_with_radiance_exposure_and_area():
    """Each factor in the equation enters exactly once and to first power."""
    det = Detector_SWC()
    sim = Simulation()

    base = run_chain(make_radiance_cube(1.0e4), ConstantAreaTelescope(1.0),
                     det, sim, 1 * u.s)
    base_val = (base.data * base.unit).to_value(u.photon / u.pix)

    doubled_radiance = run_chain(make_radiance_cube(2.0e4),
                                 ConstantAreaTelescope(1.0), det, sim, 1 * u.s)
    doubled_exposure = run_chain(make_radiance_cube(1.0e4),
                                 ConstantAreaTelescope(1.0), det, sim, 2 * u.s)
    doubled_area = run_chain(make_radiance_cube(1.0e4),
                             ConstantAreaTelescope(2.0), det, sim, 1 * u.s)

    for cube in (doubled_radiance, doubled_exposure, doubled_area):
        val = (cube.data * cube.unit).to_value(u.photon / u.pix)
        assert np.allclose(val, 2.0 * base_val, rtol=1e-12, atol=0.0)


def test_photons_scale_with_wavelength_at_fixed_energy_radiance():
    """At fixed energy radiance, photon count goes as lambda.

    Doubling the wavelength halves the energy per photon, so the same energy
    gives twice the photons. This catches an inverted conversion, which a
    single wavelength cannot.
    """
    det = Detector_SWC()
    sim = sim_for(Detector_SWC)
    tel = ConstantAreaTelescope(1.0)

    short = 100.0 * u.Angstrom
    long = 200.0 * u.Angstrom

    n_short = run_chain(make_radiance_cube(rest=short), tel, det, sim,
                        1 * u.s).data[0, 0, :]
    n_long = run_chain(make_radiance_cube(rest=long), tel, det, sim,
                       1 * u.s).data[0, 0, :]

    # Compare pixel by pixel with the ratio of the pixels' wavelengths, not a
    # flat 2.0: each window is 0.30 Angstrom wide, so the ratio runs from
    # 2.0014 at the blue end to 1.9984 at the red end, and a flat comparison
    # would need a tolerance loose enough to hide a real error.
    expected = (literal_wavelength_axis_cm(rest=long)
                / literal_wavelength_axis_cm(rest=short))
    assert np.allclose(n_long / n_short, expected, rtol=1e-10, atol=0.0)
    assert expected.min() < 2.0 < expected.max()


def test_pixel_solid_angle_matches_geometry():
    """Omega_pix is the patch of Sun one pixel sees, in steradian.

    Recomputed from arcseconds and the astronomical unit, using only the
    detector numbers from the pipeline.
    """
    for det in (Detector_SWC(), Detector_EIS()):
        for slit in (0.2, 1.0) * u.arcsec:
            slit_cm = 2.0 * AU_CM * np.tan(0.5 * slit.to_value(u.rad))
            pix_rad = (det.plate_scale_angle * (1 * u.pix)).to_value(u.rad)
            pix_cm = 2.0 * AU_CM * np.tan(0.5 * pix_rad)

            assert angle_to_distance(slit).to_value(u.cm) == pytest.approx(
                slit_cm, rel=1e-12)
            assert (det.plate_scale_length * (1 * u.pix)).to_value(
                u.cm) == pytest.approx(pix_cm, rel=1e-12)

            omega = slit_cm * pix_cm / AU_CM**2
            # A 1 arcsec by 1 arcsec patch is ARCSEC_RAD**2 steradian.
            expected = (slit.to_value(u.arcsec) * ARCSEC_RAD
                        * det.plate_scale_angle.to_value(u.arcsec / u.pix)
                        * ARCSEC_RAD)
            assert omega == pytest.approx(expected, rel=1e-10)


def test_small_angle_correction_is_quadratic():
    """2 au tan(theta / 2) exceeds au theta by theta^2 / 12.

    This shows the size of the term for anyone simplifying
    angle_to_distance: two parts in 1e12 at 1 arcsec, seven parts in 1e9 at
    60 arcsec. It is negligible at every slit width of either instrument, but
    it is a real term, not an identity.
    """
    for arcsec in (0.2, 1.0, 4.0, 60.0):
        theta = arcsec * ARCSEC_RAD
        exact = 2.0 * AU_CM * np.tan(0.5 * theta)
        approx = AU_CM * theta
        assert exact / approx - 1.0 == pytest.approx(theta**2 / 12.0, rel=1e-6)


def test_spectral_pitch_enters_once():
    """Photons per pixel is proportional to the wavelength a pixel covers."""
    det = Detector_SWC()
    sim = Simulation()
    tel = ConstantAreaTelescope(1.0)
    cube = make_radiance_cube()

    narrow = photons_to_pixel_counts(
        add_telescope_throughput(
            intensity_to_photons(apply_exposure(cube, 1 * u.s)), tel),
        det.wvl_res, det.plate_scale_length,
        angle_to_distance(sim.slit_width))
    wide = photons_to_pixel_counts(
        add_telescope_throughput(
            intensity_to_photons(apply_exposure(cube, 1 * u.s)), tel),
        2 * det.wvl_res, det.plate_scale_length,
        angle_to_distance(sim.slit_width))

    assert np.allclose(wide.data, 2.0 * narrow.data, rtol=1e-12)


def test_photon_arrival_sampling_preserves_the_mean():
    """Poisson sampling is unbiased, and returns whole photons."""
    rng = np.random.RandomState(20260912)
    np.random.seed(20260912)

    mean = 400.0
    wcs = make_radiance_cube().wcs
    counts = NDCube(np.full((N_SLIT, N_SCAN, N_SPEC), mean),
                    wcs=wcs, unit=u.photon / u.pix,
                    meta={"rest_wav": REST})

    draws = [sample_photon_arrivals(counts).data for _ in range(40)]
    stack = np.concatenate([d.ravel() for d in draws])

    assert stack.dtype.kind == "i"
    # Standard error on the mean of n Poisson draws of mean m is sqrt(m / n).
    n = stack.size
    assert abs(stack.mean() - mean) < 5.0 * np.sqrt(mean / n)
    # Poisson variance equals its mean.
    assert stack.var() == pytest.approx(mean, rel=0.05)
    del rng


def test_electron_conversion_applies_quantum_efficiency_and_fano_gain():
    """Mean electrons per detected photon is E_ph / w(T).

    w(T) = 3.71 - 0.0006 (T - 300) eV per electron-hole pair, so at 195
    Angstrom a detected photon frees about seventeen electrons. Quantum
    efficiency is a binomial draw, so the mean carries one factor of it.
    """
    np.random.seed(20260912)
    det = Detector_SWC(ccd_temperature=-60 * u.deg_C)

    shape = (4, 64, 64)
    wcs = make_radiance_cube().wcs
    photons = NDCube(np.full(shape, 500, dtype=np.int64), wcs=wcs,
                     unit=u.photon / u.pix, meta={"rest_wav": REST})

    # Isolate the photon path: no dark current, no read noise.
    det.dark_current = 0 * u.electron / (u.pix * u.s)
    det.read_noise_rms = 0 * u.electron / u.pix

    out = to_electrons(photons, 1 * u.s, det)

    temp_k = det.ccd_temperature.to_value(u.K, equivalencies=u.temperature())
    w_ev = 3.71 - 0.0006 * (temp_k - 300.0)
    e_ph_ev = HC_ERG_CM / REST.to_value(u.cm) / ERG_PER_EV
    expected = 500 * det.qe_euv * (e_ph_ev / w_ev)

    assert e_ph_ev == pytest.approx(63.54, rel=1e-3)
    assert out.data.mean() == pytest.approx(expected, rel=0.01)


def test_dn_conversion_divides_by_gain_and_clips_at_max_dn():
    """DN = electrons / gain, rounded, and never above max_dn."""
    det = Detector_EIS()
    wcs = make_radiance_cube().wcs

    electrons = NDCube(
        np.array([[[0.0, 6.3, 63.0, 1e9]]]).repeat(1, axis=0),
        wcs=WCS(naxis=3), unit=u.electron / u.pix,
        meta={"rest_wav": REST})
    # A bare WCS is enough here; to_dn only copies it.
    out = to_dn(electrons, det)

    gain = det.gain_e_per_dn.to_value(u.electron / u.DN)
    assert out.data.ravel()[0] == 0.0
    assert out.data.ravel()[1] == pytest.approx(round(6.3 / gain))
    assert out.data.ravel()[2] == pytest.approx(round(63.0 / gain))
    assert out.data.ravel()[3] == det.max_dn.to_value(u.DN / u.pix)
    assert np.all(out.data <= det.max_dn.to_value(u.DN / u.pix))


def test_swc_pixel_fills_before_the_digitiser():
    """
    The SW CCD full well is 150 ke- (datasheet), below what the FEE can
    digitise.
    """
    det = Detector_SWC()
    assert det.full_well == 150000 * u.electron / u.pix
    assert det.full_well < det.max_dn * det.gain_e_per_dn


def test_swc_full_well_moves_no_constructor_argument():
    """`full_well` is keyword-only, so the positional arguments keep their order."""
    parameters = inspect.signature(Detector_SWC).parameters
    assert parameters["full_well"].kind is inspect.Parameter.KEYWORD_ONLY
    positional = [name for name, p in parameters.items()
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional[positional.index("max_dn") + 1] == "pix_size"


def test_eis_and_swc_differ_only_by_their_numbers():
    """For the same radiance, the EIS to SWC ratio is set by geometry alone.

    At equal effective area the instruments differ only in the sky area a
    pixel sees and the wavelength it covers: EIS has a 1 arcsec pixel and a
    1 arcsec slit against SWC's 0.159 and 0.2, and a 22.3 mAA spectral pitch
    against 16.9. This checks that choosing an instrument changes nothing
    else in the chain.
    """
    tel = ConstantAreaTelescope(1.0)
    swc, eis = Detector_SWC(), Detector_EIS()
    sim_swc, sim_eis = sim_for(Detector_SWC), sim_for(Detector_EIS)

    n_swc = run_chain(make_radiance_cube(), tel, swc, sim_swc,
                      1 * u.s).data.ravel()[0]
    n_eis = run_chain(make_radiance_cube(), tel, eis, sim_eis,
                      1 * u.s).data.ravel()[0]

    def etendue(det, sim):
        slit_cm = 2.0 * AU_CM * np.tan(0.5 * sim.slit_width.to_value(u.rad))
        pix_rad = (det.plate_scale_angle * (1 * u.pix)).to_value(u.rad)
        pix_cm = 2.0 * AU_CM * np.tan(0.5 * pix_rad)
        dlam_cm = (det.wvl_res * (1 * u.pix)).to_value(u.cm)
        return slit_cm * pix_cm * dlam_cm

    expected = etendue(eis, sim_eis) / etendue(swc, sim_swc)
    assert n_eis / n_swc == pytest.approx(expected, rel=1e-10)


def test_an_iteration_raises_no_warnings_about_the_cube_having_no_observer():
    """
    A cube with helioprojective and wavelength axes has no observer, and
    astropy warns each time its wavelengths become SpectralCoord objects.
    eclipse filters these warnings, but a call from Python does not, so each
    Monte Carlo iteration must not raise them.
    """
    import warnings

    from eclipse import Telescope_EUVST, create_uniform_intensity_cube, simulate_once

    det, tel = Detector_SWC(), Telescope_EUVST()
    sim = Simulation(slit_width=0.2 * u.arcsec, expos=1 * u.s, psf=True)
    cube = create_uniform_intensity_cube(5000 * u.erg / (u.s * u.cm**2 * u.sr),
                                         195.119 * u.AA, 20 * u.km / u.s, det, sim, tel=tel)
    np.random.seed(20260929)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        simulate_once(cube, sim.expos, det, tel, sim, uniform_mode=True)
    about_the_observer = [str(w.message) for w in caught
                          if "ICRS" in str(w.message) or "observer" in str(w.message)]
    assert about_the_observer == []


def test_dark_electrons_without_noise_is_a_writable_mean():
    dark = _dark_electrons(3.5, (2, 3), noise=False)
    assert dark.shape == (2, 3) and np.all(dark == 3.5)
    dark += 1.0
    assert np.all(dark == 4.5)


@pytest.mark.parametrize("inverse_transform", [False, True])
def test_dark_electrons_refuses_a_mean_that_does_not_fit_the_pixels(inverse_transform):
    with pytest.raises(ValueError, match="does not broadcast"):
        _dark_electrons(np.ones(3), (2, 4), inverse_transform=inverse_transform)
    with pytest.raises(ValueError, match="does not broadcast"):
        _dark_electrons(np.ones((3, 1)), (1, 4), inverse_transform=inverse_transform)


def test_dark_electrons_takes_a_mean_per_row():
    np.random.seed(0)
    dark = _dark_electrons(np.array([[0.0], [1e6]]), (2, 3))
    assert dark.shape == (2, 3)
    assert np.all(dark[0] == 0) and np.all(dark[1] > 9e5)


def test_dark_electrons_by_inverse_transform_takes_a_mean_per_pixel():
    mean = np.array([[0.0, 5.0], [50.0, 500.0]])
    np.random.seed(0)
    draws = np.array([_dark_electrons(mean, mean.shape, inverse_transform=True)
                      for _ in range(2000)])
    assert np.all(draws[:, 0, 0] == 0)
    np.testing.assert_allclose(draws.mean(axis=0), mean, rtol=0.03, atol=0.2)


def test_digitise_without_rounding_keeps_fractional_dn():
    det = Detector_SWC()
    gain = det.gain_e_per_dn.to_value(u.electron / u.DN)
    max_dn = det.max_dn.to_value(u.DN / u.pixel)
    electrons = np.array([0.3 * gain, 10.6 * gain, 10 * max_dn * gain])
    np.testing.assert_allclose(_digitise(electrons, det, rounded=False), [0.3, 10.6, max_dn])
    np.testing.assert_array_equal(_digitise(electrons, det), [0.0, 11.0, max_dn])


def test_detected_photons_without_noise_match_the_photons_times_the_qe():
    det = Detector_SWC()
    photons = np.array([[10.0, 20.5]], dtype=np.float32)
    detected = _detected_photons(photons, det, noise=False)
    assert detected.dtype == np.float64
    np.testing.assert_array_equal(detected, (photons * det.qe_euv).astype(float))


def test_detected_photons_with_noise_drop_fractions_of_a_photon():
    det = Detector_SWC()
    np.random.seed(0)
    detected = _detected_photons(np.full(20000, 3.9), det)
    # A binomial draw over 3 photons, not 3.9: a mean of 3 qe.
    assert detected.mean() == pytest.approx(3 * det.qe_euv, rel=0.02)


def test_read_noise_has_the_detector_rms():
    det = Detector_SWC()
    np.random.seed(0)
    noise = _read_noise((400, 400), det)
    assert noise.mean() == pytest.approx(0.0, abs=0.1)
    assert noise.std() == pytest.approx(det.read_noise_rms.to_value(u.electron / u.pixel),
                                        rel=0.01)


def test_dn_are_whole_dn_per_pixel_whatever_the_unit_of_the_maximum():
    det = Detector_SWC(max_dn=65.535 * u.DN / u.mpix)
    gain = det.gain_e_per_dn.to_value(u.electron / u.DN)
    # The maximum is 65535 DN per pixel, so 50400.4 DN rounds and 70000 clips.
    dn = _digitise(np.array([50400.4, 70000.0]) * gain, det)
    np.testing.assert_array_equal(dn, [50400.0, 65535.0])


def test_dark_electrons_per_row_draw_as_over_every_pixel():
    mean = np.array([[1.0], [50.0], [400.0]])
    np.random.seed(5)
    per_row = _dark_electrons(mean, (3, 7))
    np.random.seed(5)
    every_pixel = np.random.poisson(np.broadcast_to(mean, (3, 7)))
    np.testing.assert_array_equal(per_row, every_pixel)
