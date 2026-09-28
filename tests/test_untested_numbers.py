"""The sizes of things no test pinned down.

Each of these could change by a large factor with every test still passing:
the variance of each detector noise term, the rounding of electrons to DN,
the brightness and width of a synthesised line, the filter's attenuation of
the visible stray light, the row edges of a shifted or tilted focal plane,
the effective area and the dark current, and the contribution function
itself. Each is checked here against a value worked out independently.
"""
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.config import Detector_SWC, Simulation, Telescope_EUVST
from euvst_response.radiometric import add_visible_stray_light, to_dn, to_electrons
from euvst_response.readout import MEASURED_SLIT_IMAGE_TILT, FocalPlane_SWC

REST = 195.119 * u.AA
# Energy per electron-hole pair in silicon at -60 C, the SWC operating
# temperature: w(T) = 3.71 - 0.0006 (T - 300) eV.
ELECTRONS_PER_PHOTON = ((const.h * const.c / REST).to_value(u.eV)
                        / (3.71 - 0.0006 * (213.15 - 300.0)))


def _photons(value, shape=(1000, 1000, 1)):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.crval = [REST.to_value(u.cm), 0.0, 0.0]
    wcs.wcs.cdelt = [16.9e-11, 0.2, 0.159]
    return NDCube(np.full(shape, float(value)), wcs=wcs, unit=u.photon / u.pix,
                  meta={"rest_wav": REST})


def _detector(qe=0.76, dark=0.0, read=0.0):
    det = Detector_SWC()
    det.qe_euv = qe
    det.dark_current = dark * u.electron / (u.pix * u.s)
    det.read_noise_rms = read * u.electron / u.pix
    return det


def _variance(photons, det):
    np.random.seed(20260928)
    return to_electrons(_photons(photons), 1 * u.s, det).data.var(ddof=1)


# A million pixels measure a variance to 0.14 per cent; each term below is
# checked to 1.5 per cent, so a noise term off by a tenth of itself fails.
def test_the_fano_spread_of_each_photons_electrons():
    det = _detector(qe=1.0)
    expected = 1000 * det.si_fano * ELECTRONS_PER_PHOTON
    assert _variance(1000, det) == pytest.approx(expected, rel=0.015)


def test_the_quantum_efficiency_is_a_binomial_draw_on_top():
    det = _detector(qe=0.76)
    expected = 1000 * 0.76 * (0.24 * ELECTRONS_PER_PHOTON**2 + det.si_fano * ELECTRONS_PER_PHOTON)
    assert _variance(1000, det) == pytest.approx(expected, rel=0.015)


def test_the_dark_current_is_a_poisson_draw_and_the_read_noise_adds_its_square():
    # A mean of 50 electrons keeps the draws clear of zero.
    assert _variance(0, _detector(dark=50.0)) == pytest.approx(50.0, rel=0.015)
    assert _variance(0, _detector(dark=50.0, read=10.0)) == pytest.approx(150.0, rel=0.015)


def test_electrons_are_rounded_to_the_nearest_dn():
    """Electrons that were whole multiples of the gain could not tell rounding from truncating."""
    det = Detector_SWC()
    gain = det.gain_e_per_dn.to_value(u.electron / u.DN)
    electrons = NDCube(np.array([0.4, 1.4, 1.6, 2.6]).reshape(1, 1, 4) * gain,
                       wcs=_photons(0, (1, 1, 4)).wcs, unit=u.electron / u.pix)
    assert to_dn(electrons, det).data.ravel().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_a_synthesised_line_has_the_brightness_and_the_thermal_width_it_should():
    """G EM / 4 pi, spread over a Gaussian of sigma lambda0 sqrt(kT / m) / c."""
    from euvst_response.synthesis import synthesise_spectra
    log_t = np.array([6.2])
    grid = np.arange(-200, 201) * u.km / u.s
    emission_measure = 1.0e27
    em_tv = np.zeros((1, 1, 1, grid.size))
    em_tv[0, 0, 0, grid.size // 2] = emission_measure
    goft = {"Fe12_195.1190": {"wl0": REST.to(u.cm), "g": np.full((1, 1, 1), 1.0e-24),
                              "atom": 26, "ion": 12}}
    synthesise_spectra(goft, em_tv, grid.to(u.cm / u.s), log_t)
    line = goft["Fe12_195.1190"]
    wavelength = line["wl_grid"].to_value(u.cm)
    spectrum = line["si"][0, 0]
    step = np.diff(wavelength)[0]
    assert spectrum.sum() * step == pytest.approx(1.0e-24 * emission_measure / (4 * np.pi),
                                                  rel=1e-9)
    iron = 55.845 * u.u
    sigma = (REST * np.sqrt(const.k_B * 10**6.2 * u.K / iron) / const.c).to_value(u.cm)
    centre = (spectrum * wavelength).sum() / spectrum.sum()
    width = np.sqrt((spectrum * (wavelength - centre) ** 2).sum() / spectrum.sum())
    assert width == pytest.approx(sigma, rel=1e-6)


def test_the_filter_attenuates_the_visible_stray_light():
    """The only test of it left the telescope out, and so the filter."""
    det = Detector_SWC()
    sim = Simulation(instrument="SWC", slit_width=0.2 * u.arcsec,
                     vis_sl=1.0e12 * u.photon / (u.s * u.cm**2))
    electrons = NDCube(np.zeros((2, 2, 3)), wcs=_photons(0, (2, 2, 3)).wcs,
                       unit=u.electron / u.pix, meta={"rest_wav": REST})
    tel = Telescope_EUVST()
    bare = add_visible_stray_light(electrons, 10 * u.s, det, sim, noise=False).data
    filtered = add_visible_stray_light(electrons, 10 * u.s, det, sim, tel, noise=False).data
    assert np.all(bare > 0)
    assert filtered == pytest.approx(bare * tel.filter.visible_light_throughput(), rel=1e-12)


@pytest.mark.parametrize("ccd", ["left", "right"])
def test_the_row_edges_are_the_wavelengths_half_a_row_either_side_of_each_centre(ccd):
    """With a calibration offset and the slit image's tilt, the edges shift with the centres."""
    fp = FocalPlane_SWC(wavelength_offset=0.1 * u.AA, slit_image_tilt=MEASURED_SLIT_IMAGE_TILT)
    for column in (0, 1023.5, 2047):
        edges = fp.row_edges(ccd, column)
        expected = fp.wavelength(np.arange(fp.n_rows + 1) - 0.5, ccd, column)
        assert u.allclose(edges, expected, rtol=1e-14)
    plain = FocalPlane_SWC()
    shift = fp.row_edges(ccd, 1023.5) - plain.row_edges(ccd)
    assert u.allclose(shift, 0.1 * u.AA, rtol=1e-9)


def test_the_effective_area_is_the_product_of_its_parts():
    """Half the 280 mm aperture, times the mirror, its roughness, the grating and the filter."""
    tel = Telescope_EUVST()

    def table(path):
        data = np.loadtxt(path, skiprows=2)
        return lambda wavelength_nm: np.interp(wavelength_nm, data[:, 0], data[:, 1])

    flt = tel.filter
    for wavelength in (171.073, 195.119, 211.317) * u.AA:
        nm = wavelength.to_value(u.nm)
        filter_throughput = (table(flt.al_table)(nm) ** (1485 / 1000)
                             * table(flt.oxide_table)(nm) ** (95 / 1000)
                             * table(flt.c_table)(nm) ** 0 * 0.8)
        expected = (0.5 * np.pi * (14.0 * u.cm) ** 2 * table(tel.pm_table)(nm)
                    * np.exp(-(4 * np.pi * 0.3 / nm) ** 2) * table(tel.grating_table)(nm)
                    * filter_throughput)
        assert tel.ea_and_throughput(wavelength).to_value(u.cm**2) == pytest.approx(
            expected.to_value(u.cm**2), rel=1e-9)
    assert flt.visible_light_throughput() == pytest.approx(10 ** (-1485 / 170) * 0.8, rel=1e-12)


def test_the_dark_current_at_the_operating_temperature():
    """20000 e/pix/s at 293 K, scaled by 122 T^3 exp(-6400 / T): 2.15 e/pix/s at -60 C."""
    dark = Detector_SWC(ccd_temperature=-60 * u.deg_C).dark_current
    assert dark.to_value(u.electron / (u.pix * u.s)) == pytest.approx(2.1548, rel=1e-4)


# ---------------------------------------------------------------------------
# The contribution function itself, where the CHIANTI database is there
# ---------------------------------------------------------------------------
fiasco = pytest.importorskip("fiasco")


def _database():
    return Path(fiasco.defaults["hdf5_dbase_root"])


chianti = pytest.mark.skipif(not _database().exists(),
                             reason="needs the CHIANTI database fiasco builds")


@chianti
def test_the_contribution_function_is_fiascos_times_the_proton_electron_ratio():
    """G(T, n) for the line, as fiasco gives it per n_e^2, with n_H / n_e, on the right axes."""
    from euvst_response.synthesis import compute_goft_fiasco

    temperature = 10 ** np.array([6.1, 6.3]) * u.K
    density = 10 ** np.array([8.0, 10.0]) / u.cm**3
    goft, log_t, log_n = compute_goft_fiasco(["Fe12_195.1190"], logT_min=6.1, logT_max=6.3,
                                             nT=2, logN_min=8.0, logN_max=10.0, nN=2,
                                             n_workers=1)
    ion = fiasco.Ion("Fe 12", temperature, abundance="sun_coronal_2021_chianti")
    index = int(np.argmin(np.abs(ion.transitions.wavelength[ion.transitions.is_bound_bound]
                                 - REST)))
    direct = (ion.contribution_function(density)[:, :, index]
              * ion.proton_electron_ratio[:, np.newaxis]).to_value(u.erg * u.cm**3 / u.s)
    assert goft["Fe12_195.1190"]["g_tn"] == pytest.approx(direct.T, rel=1e-12)
    assert goft["Fe12_195.1190"]["g_tn"].shape == (log_n.size, log_t.size)


@chianti
def test_the_mass_per_electron_of_coronal_abundances():
    """About 1.16 atomic mass units per free electron, fully ionised."""
    from euvst_response.atmosphere import mass_per_electron
    assert mass_per_electron("sun_coronal_2021_chianti") == pytest.approx(1.158, abs=0.002)
