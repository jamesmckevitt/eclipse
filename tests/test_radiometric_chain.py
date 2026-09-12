"""Validate the radiometric chain against the standard equations.

The pipeline converts a spectral radiance into photons per pixel in four steps,
spread over four functions. Written as one equation, for a pixel at wavelength
``lam`` observing a source of spectral radiance ``I``:

    N = (I * t_exp / E_ph) * A_eff * Omega_pix * dlam_pix

    E_ph      = h c / lam                         erg per photon
    Omega_pix = (w_slit * w_pix) / au^2           sr, the patch of Sun a pixel sees
    dlam_pix  = det.wvl_res                       cm per pixel

Every expected value below is built from that expression with the physical
constants written out as literals, so a test failure means the pipeline moved
rather than that two spellings of the same helper disagree. The effective area
is held constant by a stub telescope where the point is the chain arithmetic;
the real wavelength- and date-dependent areas are checked against IDL in
test_eis_effective_area.py.
"""
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.config import Detector_EIS, Detector_SWC, Simulation
from euvst_response.radiometric import (
    add_telescope_throughput,
    apply_exposure,
    intensity_to_photons,
    photons_to_pixel_counts,
    sample_photon_arrivals,
    to_dn,
    to_electrons,
)
from euvst_response.utils import angle_to_distance

# CODATA, and the IAU definition of the astronomical unit. Written out rather
# than imported from astropy so that the expected value in each test is
# independent of the module it is checking.
H_ERG_S = 6.62607015e-27
C_CM_S = 2.99792458e10
HC_ERG_CM = H_ERG_S * C_CM_S
AU_CM = 1.495978707e13
ARCSEC_RAD = np.pi / (180.0 * 3600.0)
ERG_PER_EV = 1.602176634e-12

N_SCAN, N_SLIT, N_SPEC = 2, 8, 16
REST = 195.119 * u.Angstrom

# Spectral radiance per unit wavelength, in the units the synthesis stage emits.
RADIANCE_UNIT = u.erg / (u.cm**2 * u.s * u.sr * u.cm)
RADIANCE = 1.0e4


class ConstantAreaTelescope:
    """Telescope whose effective area times throughput is a fixed number.

    The chain only ever asks a telescope for ``ea_and_throughput``, so this is
    the whole interface. Fixing it isolates the geometry and unit arithmetic
    from the calibration curve, which has its own tests.
    """

    def __init__(self, area_cm2=1.5):
        self.area_cm2 = area_cm2

    def ea_and_throughput(self, wavelength):
        return self.area_cm2 * u.cm**2


def make_radiance_cube(value=RADIANCE, n_spec=N_SPEC, rest=REST):
    """Uniform radiance cube with a wavelength axis centred on *rest*."""
    wcs = WCS(naxis=3)
    # WCS axis order is the reverse of the numpy order, so axis 0 here is the
    # last data axis, which is the spectral one.
    wcs.wcs.ctype = ["WAVE", "HPLT-TAN", "HPLN-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [0.0169, 1.0, 1.0]
    wcs.wcs.crpix = [n_spec / 2.0, N_SLIT / 2.0, 1.0]
    wcs.wcs.crval = [rest.to_value(u.Angstrom), 0.0, 0.0]
    return NDCube(
        np.full((N_SCAN, N_SLIT, n_spec), float(value)),
        wcs=wcs,
        unit=RADIANCE_UNIT,
        meta={"rest_wav": rest},
    )


def sim_for(detector_cls):
    """Simulation with a slit width the instrument actually offers.

    Simulation validates slit width against the instrument: EIS takes 1, 2 or
    4 arcsec and SWC takes 0.2, 0.4, 0.8 or 1.6, so the two share no legal
    value and a comparison between them has to carry the slit ratio explicitly.
    """
    if detector_cls is Detector_EIS:
        return Simulation(instrument="EIS", slit_width=1 * u.arcsec)
    return Simulation(instrument="SWC", slit_width=0.2 * u.arcsec)


def run_chain(cube, tel, det, sim, t_exp):
    """The four deterministic steps that produce photons per pixel."""
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
    """The spectral axis implied by the WCS make_radiance_cube builds.

    FITS pixel coordinates are one-based, so the value at numpy index ``i`` is
    ``crval + (i + 1 - crpix) * cdelt``. Written out rather than read back from
    the cube so that test_wavelength_axis_follows_the_fits_convention can check
    the pipeline sees the axis this file thinks it wrote.
    """
    crpix = n_spec / 2.0
    cdelt = 0.0169
    i = np.arange(n_spec)
    return (rest.to_value(u.Angstrom) + (i + 1 - crpix) * cdelt) * 1e-8


def expected_photons_per_pixel(radiance, t_exp, area_cm2, det, sim,
                               lam_cm=None):
    """The standard equation, written out from literal constants.

    *lam_cm* is the wavelength of each pixel. It is an array, not the rest
    wavelength, because the energy per photon is hc / lambda at the wavelength
    the pixel actually sees: across a 16 pixel window at 195 Angstrom that is a
    0.13 per cent gradient in photon count, small but not zero.
    """
    if lam_cm is None:
        lam_cm = literal_wavelength_axis_cm()
    e_ph = HC_ERG_CM / np.asarray(lam_cm)

    # Linear size at 1 au of the angles a pixel spans. The pipeline uses the
    # exact 2 au tan(theta / 2) rather than the small-angle form; the two differ
    # by theta^2 / 12, which is two parts in 1e12 at 1 arcsec, and which
    # test_small_angle_correction_is_quadratic pins down.
    slit_cm = 2.0 * AU_CM * np.tan(0.5 * sim.slit_width.to_value(u.rad))
    pix_rad = (det.plate_scale_angle * (1 * u.pix)).to_value(u.rad)
    pix_cm = 2.0 * AU_CM * np.tan(0.5 * pix_rad)
    omega_sr = slit_cm * pix_cm / AU_CM**2

    dlam_cm = (det.wvl_res * (1 * u.pix)).to_value(u.cm)

    return radiance * t_exp.to_value(u.s) / e_ph * area_cm2 * omega_sr * dlam_cm


@pytest.mark.parametrize("detector_cls", [Detector_SWC, Detector_EIS])
@pytest.mark.parametrize("t_exp_s", [1.0, 10.0, 60.0])
@pytest.mark.parametrize("area_cm2", [0.25, 1.5])
def test_photons_per_pixel_matches_standard_equation(detector_cls, t_exp_s,
                                                     area_cm2):
    """The headline check: ph/pix out of the chain equals the equation."""
    det = detector_cls()
    sim = sim_for(detector_cls)
    tel = ConstantAreaTelescope(area_cm2)
    t_exp = t_exp_s * u.s

    got = run_chain(make_radiance_cube(), tel, det, sim, t_exp)
    got_per_pix = got.data * got.unit
    assert got_per_pix.unit.is_equivalent(u.photon / u.pix)

    expected = expected_photons_per_pixel(RADIANCE, t_exp, area_cm2, det, sim)
    actual = got_per_pix.to_value(u.photon / u.pix)

    # atol=0 deliberately. The default 1e-8 is larger than these photon counts
    # at short exposures, so it would pass on absolute agreement alone and hide
    # a relative error of any size.
    assert np.allclose(actual, expected[np.newaxis, np.newaxis, :],
                       rtol=1e-10, atol=0.0)


def test_wavelength_axis_follows_the_fits_convention():
    """The pipeline reads the axis this file thinks it wrote.

    Everything else here builds its expected photon count per pixel from
    literal_wavelength_axis_cm, so if the one-based FITS convention were wrong
    the expected values would be wrong in step with each other and agree anyway.
    """
    cube = make_radiance_cube()
    from_cube = cube.axis_world_coords(2)[0].to_value(u.cm)
    assert np.allclose(from_cube, literal_wavelength_axis_cm(), rtol=1e-12,
                       atol=0.0)


def test_photon_energy_is_hc_over_lambda():
    """Independent check of the energy the conversion divides by.

    195.119 Angstrom is 63.54 eV, so a photon count is an energy divided by
    1.018e-10 erg. If this drifts, every photon number here drifts with it.
    """
    e_ph_erg = HC_ERG_CM / REST.to_value(u.cm)
    assert e_ph_erg == pytest.approx(1.0181e-10, rel=1e-3)
    assert e_ph_erg / ERG_PER_EV == pytest.approx(63.54, rel=1e-3)

    # And that this is the energy the chain actually divides by: with area,
    # solid angle, pitch and exposure all unity-scaled out, photons per pixel
    # times E_ph must return the radiance that went in.
    det = Detector_SWC()
    sim = sim_for(Detector_SWC)
    t_exp = 1 * u.s
    got = run_chain(make_radiance_cube(), ConstantAreaTelescope(1.0), det, sim,
                    t_exp)
    actual = (got.data * got.unit).to_value(u.photon / u.pix)[0, 0, :]

    lam_cm = literal_wavelength_axis_cm()
    slit_cm = 2.0 * AU_CM * np.tan(0.5 * sim.slit_width.to_value(u.rad))
    pix_rad = (det.plate_scale_angle * (1 * u.pix)).to_value(u.rad)
    pix_cm = 2.0 * AU_CM * np.tan(0.5 * pix_rad)
    geometry = (slit_cm * pix_cm / AU_CM**2
                * (det.wvl_res * (1 * u.pix)).to_value(u.cm))

    recovered = actual * (HC_ERG_CM / lam_cm) / geometry / t_exp.to_value(u.s)
    assert np.allclose(recovered, RADIANCE, rtol=1e-10, atol=0.0)


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
    """Radiance is energy, so photon count goes as lambda for a fixed radiance.

    Doubling the wavelength halves the energy per photon, so the same energy
    buys twice the photons. This catches an inverted conversion, which a single
    wavelength cannot.
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

    # Compared pixel by pixel against the ratio of those pixels' wavelengths,
    # not against a flat 2.0: the two windows are each 0.27 Angstrom wide, so
    # the ratio runs from 2.0024 at the blue end to 1.9976 at the red end and a
    # flat comparison would need a tolerance loose enough to hide a real error.
    expected = (literal_wavelength_axis_cm(rest=long)
                / literal_wavelength_axis_cm(rest=short))
    assert np.allclose(n_long / n_short, expected, rtol=1e-10, atol=0.0)
    assert expected.min() < 2.0 < expected.max()


def test_pixel_solid_angle_matches_geometry():
    """Omega_pix is the patch of Sun one pixel sees, in steradian.

    Recomputed from arcseconds and the astronomical unit, with no reference to
    the pipeline's helpers beyond the detector numbers themselves.
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

    Recorded so that anyone simplifying angle_to_distance can see the size of
    what they would be changing: two parts in 1e12 at 1 arcsec, rising to seven
    parts in 1e9 at a degree-scale 60 arcsec. Negligible at every slit width
    either instrument offers, but it is a real term rather than an identity.
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
    counts = NDCube(np.full((N_SCAN, N_SLIT, N_SPEC), mean),
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

    w(T) = 3.71 - 0.0006 (T - 300) eV per electron-hole pair, so at 195 Angstrom
    a detected photon liberates about seventeen electrons. Quantum efficiency
    enters as a binomial draw, so the expectation carries one factor of it.
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


def test_dn_conversion_divides_by_gain_and_clips_at_full_well():
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


def test_eis_and_swc_differ_only_by_their_numbers():
    """Same radiance through both instruments, ratio set purely by geometry.

    At equal effective area the two instruments differ only in the area of sky
    a pixel sees and the wavelength it covers: EIS has a 1 arcsec pixel and a
    1 arcsec slit against SWC's 0.159 and 0.2, and a 22.3 mAA spectral pitch
    against 16.9. This is the check that choosing an instrument does not quietly
    change anything else in the chain.
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
