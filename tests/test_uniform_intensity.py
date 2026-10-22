"""The uniform-intensity line is integrated across each detector pixel.

Taken at pixel centres instead, lines narrower than about half a pixel come
out too bright, such as Fe IX 171.07 (sigma 0.35 SWC pixels) by 19 per
cent. The expected photon count is the radiometric equation with literal
constants. With the PSF on, the line is blurred by the spectral response
before it is integrated, so a narrow line keeps its place within a pixel.
"""
import dataclasses
import importlib
import sys
import warnings

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
import yaml
from ndcube import NDCube
from scipy.integrate import quad
from scipy.special import erf

from eclipse.config import (Detector_EIS, Detector_SWC, Simulation, Telescope_EIS,
                            Telescope_EUVST)
from eclipse.data_processing import create_uniform_intensity_cube
from eclipse.fitting import fit_cube_gauss
from eclipse.monte_carlo import monte_carlo, simulate_once
from eclipse.radiometric import _spectral_response, apply_focusing_optics_psf
from eclipse.sampling import points_onto_pixels

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
    """EUVST telescope with an effective area constant in wavelength.

    The photon count of a line then depends only on how much of it reaches
    the detector, not on the shape of the effective area curve.
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


def _gaussian_between(sigma, lo, hi):
    """Integral of a unit Gaussian of width `sigma` from `lo` to `hi`, done numerically."""
    def gaussian(x):
        return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    return quad(gaussian, lo, hi, epsabs=1e-15, epsrel=1e-12)[0]


@pytest.mark.parametrize("width", WIDTHS)
def test_the_pixels_add_up_to_the_line_intensity_whatever_its_width(width):
    """At the default extent, which leaves out a part in 1e15 of the line."""
    cube = _cube(width)
    dlam = (DET.wvl_res * u.pix).to(u.cm)
    total = (cube.data.sum(axis=-1) * cube.unit * dlam).to_value(INTENSITY.unit)
    assert np.allclose(total, INTENSITY.value, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("width", WIDTHS)
def test_each_pixel_holds_the_gaussian_integrated_across_it(width):
    """Compared pixel by pixel with a numerical integral, with the line on the middle pixel."""
    cube = _cube(width)
    n = cube.data.shape[-1]
    offsets = np.arange(n) - n // 2
    # The WCS puts the rest wavelength on the middle pixel.
    dispersion = DET.wvl_res.to_value(u.Angstrom / u.pixel)
    labelled = (cube.axis_world_coords(2)[0].to_value(u.Angstrom)
                - REST.to_value(u.Angstrom)) / dispersion
    assert np.allclose(labelled, offsets, rtol=0.0, atol=1e-9)

    sigma = _sigma_pixels(width)
    expected = np.array([_gaussian_between(sigma, x - 0.5, x + 0.5) for x in offsets])
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


def test_a_grid_cut_short_holds_exactly_the_part_of_the_line_on_it():
    """A small `n_sigma_extent` ends the grid inside the line, leaving out the tails beyond."""
    width = 60  # km/s, two pixels
    cube = create_uniform_intensity_cube(
        total_intensity=INTENSITY, rest_wavelength=REST,
        thermal_width=width * u.km / u.s, det=DET,
        sim=Simulation(slit_width=0.2 * u.arcsec), n_sigma_extent=1.0)
    # From the line centre to the far edge of the outermost pixel.
    reach = cube.data.shape[-1] // 2 + 0.5
    on_grid = _gaussian_between(_sigma_pixels(width), -reach, reach)
    assert on_grid < 0.9

    dlam = (DET.wvl_res * u.pix).to(u.cm)
    total = (cube.data.sum(axis=-1) * cube.unit * dlam).to_value(INTENSITY.unit)
    assert np.allclose(total, INTENSITY.value * on_grid, rtol=1e-10, atol=0.0)


# The line through the spectral response. A line of a tenth of a pixel, at
# a sub-pixel offset, is pulled toward the pixel's centre by about a quarter
# of a pixel when the PSF blurs the pixels instead of the spectrum.
NARROW = 0.1  # sigma in pixels
INSTRUMENTS = {
    "SWC": (Detector_SWC(), Telescope_EUVST(), 0.2 * u.arcsec),
    # EIS's spectral FWHM is the same for every slit unless psf_slit_width
    # says which slit it was measured with, which the convolution needs.
    "EIS": (Detector_EIS(), Telescope_EIS(psf_slit_width=2 * u.arcsec), 2 * u.arcsec),
}


def _narrow_cube(instrument, sim, offset=0.0):
    """
    A line of `NARROW` pixels from `create_uniform_intensity_cube`, moved
    `offset` pixels from the centre of its pixel.

    The cube always has the line on the middle of a pixel, so for other
    offsets its data are replaced by the line laid on the same pixels the
    same way: blurred by the spectral response, with `sim.psf` on, and
    integrated over each pixel. At offset 0 this gives the cube's data.
    """
    det, tel, _ = INSTRUMENTS[instrument]
    dispersion = det.wvl_res.to_value(u.Angstrom / u.pixel)
    width = NARROW * dispersion / REST.to_value(u.Angstrom) * C_KM_S
    cube = create_uniform_intensity_cube(
        total_intensity=INTENSITY, rest_wavelength=REST, thermal_width=width * u.km / u.s,
        det=det, sim=sim, tel=tel)
    sigma, box = NARROW, 0.0
    if sim.psf:
        sigma, box = _spectral_response(tel, det, sim.slit_width, sim.spectral_psf)
        sigma = np.hypot(NARROW, sigma)
    n_half = cube.data.shape[-1] // 2
    edges = np.arange(2 * n_half + 2) - n_half - 0.5
    share = points_onto_pixels(offset, edges, sigma, box).toarray()[:, 0]
    data = (INTENSITY * share / (det.wvl_res * u.pix)).to_value(cube.unit)
    if offset == 0:
        assert np.allclose(data, cube.data[0, 0], rtol=1e-12, atol=0.0)
    return NDCube(data[None, None, :], wcs=cube.wcs, unit=cube.unit, meta=cube.meta)


def _fitted_offset(instrument, cube, sim):
    """Fitted centre of the noise-free DN, in pixels from `REST`."""
    det, tel, _ = INSTRUMENTS[instrument]
    dn = simulate_once(cube, 10 * u.s, det, tel, sim, uniform_mode=True)[-1]
    values, units = fit_cube_gauss(dn, n_jobs=1)
    centre = (values[0, 0, 1] * units[1]).to_value(u.Angstrom)
    return (centre - REST.to_value(u.Angstrom)) / det.wvl_res.to_value(u.Angstrom / u.pixel)


def _sim(instrument, psf=True, spectral_psf="quadrature"):
    return Simulation(slit_width=INSTRUMENTS[instrument][2], instrument=instrument, psf=psf,
                      spectral_psf=spectral_psf, noise=False)


@pytest.mark.parametrize("offset", [0.0, 0.25, 0.4])
@pytest.mark.parametrize("spectral_psf", ["quadrature", "convolution"])
@pytest.mark.parametrize("instrument", ["SWC", "EIS"])
def test_a_line_narrower_than_a_pixel_keeps_its_fitted_centre(instrument, spectral_psf, offset):
    """
    Blurred before it is integrated over the pixels, the line is fitted
    within 0.01 pixel of where it is. What is left comes from the effective
    area and photon energy changing across the blurred line, about 0.002
    pixel.
    """
    sim = _sim(instrument, spectral_psf=spectral_psf)
    cube = _narrow_cube(instrument, sim, offset)
    assert cube.meta["psf_applied"] is True
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert _fitted_offset(instrument, cube, sim) == pytest.approx(offset, abs=0.01)


@pytest.mark.parametrize("instrument", ["SWC", "EIS"])
def test_a_cube_without_psf_applied_is_blurred_on_its_pixels_and_warns(instrument):
    """
    A cube passed in without `psf_applied` is blurred on its pixels, in
    wavelength only for a uniform intensity, and the line is pulled toward
    the pixel's centre. Each call warns once.
    """
    det, tel, _ = INSTRUMENTS[instrument]
    sim = _sim(instrument)
    cube = _narrow_cube(instrument, _sim(instrument, psf=False), 0.4)
    assert "psf_applied" not in cube.meta

    with pytest.warns(UserWarning, match="pulls a line narrower than a pixel") as seen:
        steps = simulate_once(cube, 10 * u.s, det, tel, sim, uniform_mode=True)
    assert len([w for w in seen if "narrower than a pixel" in str(w.message)]) == 1
    assert seen[0].filename == __file__
    expected = apply_focusing_optics_psf(steps[3], tel, det, sim, convolve_spatial=False,
                                         boundary=sim.psf_boundary)
    assert np.array_equal(steps[4].data, expected.data)

    with pytest.warns(UserWarning, match="pulls a line narrower than a pixel"):
        assert _fitted_offset(instrument, cube, sim) < 0.2

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        monte_carlo(cube, 10 * u.s, det, tel, sim, n_iter=3, uniform_mode=True, fit_signals="dn")
    assert len([w for w in seen if "narrower than a pixel" in str(w.message)]) == 1


@pytest.mark.parametrize("spectral_psf", ["quadrature", "convolution"])
@pytest.mark.parametrize("instrument", ["SWC", "EIS"])
def test_the_response_keeps_the_line_intensity(instrument, spectral_psf):
    """The pixels add up to the line intensity, as without the PSF."""
    det = INSTRUMENTS[instrument][0]
    cube = _narrow_cube(instrument, _sim(instrument, spectral_psf=spectral_psf))
    dlam = (det.wvl_res * u.pix).to(u.cm)
    total = (cube.data.sum(axis=-1) * cube.unit * dlam).to_value(INTENSITY.unit)
    assert np.allclose(total, INTENSITY.value, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("width", WIDTHS)
def test_without_the_psf_each_pixel_holds_the_line_alone(width):
    """With `psf` off, or without `tel`, the line is integrated over the pixels as it is."""
    sigma = (REST.to(u.cm) * width * u.km / u.s / const.c).to(u.cm)
    dlam = DET.wvl_res.to(u.cm / u.pix) * u.pix
    for sim, tel in ((Simulation(slit_width=0.2 * u.arcsec), TEL),
                     (Simulation(slit_width=0.2 * u.arcsec, psf=True), None)):
        cube = _cube(width, sim=sim, tel=tel)
        assert "psf_applied" not in cube.meta
        n_half = cube.data.shape[-1] // 2
        edges = ((np.arange(2 * n_half + 2) - n_half - 0.5) * dlam
                 / (np.sqrt(2.0) * sigma)).decompose().value
        expected = (INTENSITY * 0.5 * np.diff(erf(edges)) / dlam).to_value(cube.unit)
        assert np.array_equal(cube.data[0, 0], expected)


def test_a_run_lays_the_line_through_each_combinations_response(tmp_path, monkeypatch):
    """
    Swept over `psf`, `spectral_psf` and the slit, each combination observes
    the line through its own response, as `create_uniform_intensity_cube`
    lays it, and without the PSF observes it as it is. Noise is off, so the
    first signal is the mean signal.
    """
    main_module = importlib.import_module("eclipse.main")
    saved = {}
    save = main_module.save_results

    def keep(path, payload, **kwargs):
        saved["payload"] = payload
        return save(path, payload, **kwargs)

    monkeypatch.setattr(main_module, "save_results", keep)
    config = {"instrument": "SWC", "n_iter": 2, "ncpu": 1, "fit_signals": "dn",
              "uniform_intensity": "5000 erg / (s cm2 sr)", "thermal_width": "10 km/s",
              "simulation": {"slit_width": ["0.2 arcsec", "0.8 arcsec"], "expos": "10 s",
                             "psf": [False, True], "spectral_psf": ["quadrature", "convolution"],
                             "noise": False}}
    path = tmp_path / "uniform.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(path)])
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*narrower than a pixel")
        main_module.main()

    combinations = saved["payload"]["results"]["all_combinations"].values()
    assert len(combinations) == 8
    for combination in combinations:
        sim = combination["config_objects"]["simulation"]
        cube = create_uniform_intensity_cube(
            total_intensity=5000 * u.erg / (u.s * u.cm**2 * u.sr), rest_wavelength=REST,
            thermal_width=10 * u.km / u.s, det=DET, sim=sim, tel=TEL)
        assert cube.meta.get("psf_applied", False) is sim.psf
        dn = simulate_once(cube, 10 * u.s, DET, TEL, sim, uniform_mode=True)[-1]
        assert np.array_equal(combination["first_dn_signal_data"], dn.data), sim


@pytest.mark.parametrize("change", [{"psf": False}, {"spectral_psf": "convolution"},
                                    {"slit_width": 0.4 * u.arcsec}])
def test_a_line_made_for_another_response_is_refused(change):
    """The line was blurred for one simulation, so observing it with another would be wrong."""
    sim = Simulation(slit_width=0.2 * u.arcsec, psf=True, noise=False)
    cube = _cube(8, sim=sim)
    simulate_once(cube, 1 * u.s, DET, TEL, sim, uniform_mode=True)
    other = dataclasses.replace(sim, **change)
    with pytest.raises(ValueError, match="another PSF, slit width or spectral_psf"):
        simulate_once(cube, 1 * u.s, DET, TEL, other, uniform_mode=True)
    with pytest.raises(ValueError, match="another PSF, slit width or spectral_psf"):
        monte_carlo(cube, 1 * u.s, DET, TEL, other, n_iter=1, uniform_mode=True)
