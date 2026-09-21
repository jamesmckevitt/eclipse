"""The uniform-intensity line is integrated across each detector pixel.

create_uniform_intensity_cube evaluated the Gaussian at the pixel centres,
with the line centred on one of them, and each pixel then stood for that value
times the pixel width. That adds up to the line's intensity only when the line
is more than about half a pixel wide (sigma). Several science case lines are
narrower at their formation temperature: Fe IX 171.07 has a sigma of 0.35 SWC
pixels and came out 19 per cent too bright, Fe VIII 185.21 at 0.30 pixels 35
per cent.

The expected photon count below is the radiometric equation written out from
literal constants, as in test_radiometric_chain.py, so a failure means the
pipeline moved rather than that two spellings of one helper disagree.
"""
import astropy.units as u
import numpy as np
import pytest
from scipy.integrate import quad

from euvst_response.config import Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.data_processing import create_uniform_intensity_cube
from euvst_response.monte_carlo import simulate_once

# CODATA, and the IAU definition of the astronomical unit.
HC_ERG_CM = 6.62607015e-27 * 2.99792458e10
C_KM_S = 2.99792458e5
AU_CM = 1.495978707e13

DET = Detector_SWC()
TEL = Telescope_EUVST()
INTENSITY = 1000 * u.erg / (u.s * u.cm**2 * u.sr)
REST = 195.119 * u.Angstrom

# 1-sigma widths from a tenth of an SWC pixel to two pixels at 195 A.
WIDTHS = [3, 8, 15, 30, 60]  # km/s


class FlatTelescope(Telescope_EUVST):
    """The EUVST telescope with an effective area that does not change with wavelength.

    So that the photon count of a line depends only on how much of it reaches
    the detector, not on where the effective area curve bends.
    """

    def ea_and_throughput(self, wl0):
        return 1.0 * u.cm**2


def _cube(width, sim=None, tel=TEL):
    if sim is None:
        sim = Simulation(slit_width=0.2 * u.arcsec)
    return create_uniform_intensity_cube(
        total_intensity=INTENSITY, rest_wavelength=REST,
        thermal_width=width * u.km / u.s, det=DET, sim=sim, tel=tel)


def _sigma_pixels(width):
    dispersion = DET.wvl_res.to_value(u.Angstrom / u.pixel)
    return REST.to_value(u.Angstrom) * width / C_KM_S / dispersion


@pytest.mark.parametrize("width", WIDTHS)
def test_the_pixels_add_up_to_the_line_intensity_whatever_its_width(width):
    cube = _cube(width)
    dlam = (DET.wvl_res * u.pix).to(u.cm)
    total = (cube.data.sum(axis=-1) * cube.unit * dlam).to_value(INTENSITY.unit)
    assert np.allclose(total, INTENSITY.value, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("width", WIDTHS)
def test_each_pixel_holds_the_gaussian_integrated_across_it(width):
    """Pixel by pixel against a numerical integral, with the line on the middle pixel."""
    cube = _cube(width)
    n = cube.data.shape[-1]
    offsets = np.arange(n) - n // 2
    # The WCS puts the rest wavelength on the middle pixel.
    dispersion = DET.wvl_res.to_value(u.Angstrom / u.pixel)
    labelled = (cube.axis_world_coords(2)[0].to_value(u.Angstrom)
                - REST.to_value(u.Angstrom)) / dispersion
    assert np.allclose(labelled, offsets, rtol=0.0, atol=1e-9)

    sigma = _sigma_pixels(width)

    def gaussian(x):
        return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    expected = np.array([quad(gaussian, x - 0.5, x + 0.5, epsabs=1e-15, epsrel=1e-12)[0]
                         for x in offsets])
    dlam = (DET.wvl_res * u.pix).to(u.cm)
    got = (cube.data[0, 0] * cube.unit * dlam).to_value(INTENSITY.unit) / INTENSITY.value
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-14)
    assert np.sum(offsets * got) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("slit", [0.2, 0.4, 0.8])
@pytest.mark.parametrize("width", WIDTHS)
def test_the_detector_receives_the_photons_the_radiometric_equation_gives(width, slit):
    """N = I t A Omega lambda0 / (h c), through the instrument with the noise off.

    Counted after the spectral PSF, which is what the fit sees. The photon
    energy changes linearly across the line, so for a line symmetric about
    lambda0 the energy at lambda0 is exact.
    """
    tel = FlatTelescope()
    sim = Simulation(slit_width=slit * u.arcsec, psf=True, noise=False)
    t_exp = 10.0
    cube = _cube(width, sim=sim, tel=tel)
    photons = simulate_once(cube, t_exp * u.s, DET, tel, sim, uniform_mode=True)[4]

    slit_cm = 2.0 * AU_CM * np.tan(0.5 * (slit * u.arcsec).to_value(u.rad))
    pix_cm = 2.0 * AU_CM * np.tan(0.5 * (DET.plate_scale_angle * u.pix).to_value(u.rad))
    omega_sr = slit_cm * pix_cm / AU_CM**2
    expected = (INTENSITY.value * t_exp * 1.0 * omega_sr
                * REST.to_value(u.cm) / HC_ERG_CM)
    assert photons.data.sum() == pytest.approx(expected, rel=1e-9)
