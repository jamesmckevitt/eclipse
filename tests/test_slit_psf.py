"""The spectral PSF follows the slit (issue #31).

The spectral resolution ECLIPSE was given, 43.00 mA or 2.54 SWC pixels, is
RSC-2022021C's value for the 0.2 arcsec slit: the FWHM of the optics after
the slit, 0.352 arcsec at 212.3 A, added in quadrature to the width of the
slit, giving 0.405 arcsec. The wider slits add more, and were given the
0.2 arcsec value.

By default the PSF stays a Gaussian whose FWHM adds the slit in quadrature,
as the document does. spectral_psf: convolution instead convolves the optics
Gaussian with the slit's rectangular image, which is how the document
defines the line profile.
"""
import sys

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
import yaml
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.analysis import analyse_fit_statistics, load_instrument_response_results
from euvst_response.config import (Detector_EIS, Detector_SWC, Simulation,
                                   Telescope_EIS, Telescope_EUVST)
from euvst_response.data_processing import create_uniform_intensity_cube, pad_spectral_axis
from euvst_response.main import _validate_config_keys
from euvst_response.radiometric import (apply_focusing_optics_psf, slit_image_width,
                                        spectral_line_spread, spectral_optics_fwhm,
                                        spectral_psf_fwhm, spectral_psf_margin)
from euvst_response.synthesis_file import write_line_cubes
from euvst_response.utils import VELOCITY_CONVENTION, _fwhm_to_sigma

TEL = Telescope_EUVST()
DET = Detector_SWC()
REST = 195.119 * u.Angstrom
SLITS = [0.2, 0.4, 0.8, 1.6]  # arcsec, every SWC slit
N_SLIT, N_WAVE = 5, 81


def _frame(slit_width, spectral_psf="quadrature", profile=None):
    """One exposure of a line, one pixel wide unless *profile* is given, uniform along the slit."""
    if profile is None:
        profile = np.zeros(N_WAVE)
        profile[N_WAVE // 2] = 1.0
    data = np.tile(profile, (N_SLIT, 1, 1))
    n_wave = profile.size
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [0.0169, slit_width, 0.159]
    wcs.wcs.crpix = [(n_wave + 1) / 2, 1.0, (N_SLIT + 1) / 2]
    wcs.wcs.crval = [REST.to_value(u.Angstrom), 0.0, 0.0]
    cube = NDCube(data, wcs=wcs, unit=u.photon / u.pix, meta={"rest_wav": REST})
    sim = Simulation(slit_width=slit_width * u.arcsec, spectral_psf=spectral_psf)
    return cube, sim


def _profile(slit_width, spectral_psf="quadrature"):
    """The line after the PSF, along the dispersion."""
    cube, sim = _frame(slit_width, spectral_psf)
    out = apply_focusing_optics_psf(cube, TEL, DET, sim, convolve_spatial=False)
    return out.data[N_SLIT // 2, 0]


def _fwhm(profile):
    """Full width at half maximum of a single-peaked profile, in pixels."""
    half = profile.max() / 2
    above = np.nonzero(profile >= half)[0]
    left, right = above[0], above[-1]
    # Interpolate each crossing between the last pixel above half and the next.
    x_left = left - (profile[left] - half) / (profile[left] - profile[left - 1])
    x_right = right + (profile[right] - half) / (profile[right] - profile[right + 1])
    return x_right - x_left


def _variance(profile):
    x = np.arange(profile.size)
    mean = (x * profile).sum() / profile.sum()
    return ((x - mean) ** 2 * profile).sum() / profile.sum()


# ----------------------------------------------------------------------
# The widths
# ----------------------------------------------------------------------
def test_the_optics_part_is_the_one_rsc_2022021c_gives():
    """0.352 arcsec, and with the 0.2 arcsec slit in quadrature 0.405 arcsec."""
    plate_scale = DET.plate_scale_angle.to_value(u.arcsec / u.pixel)
    # The document's own convolved value is the quadrature sum.
    assert np.hypot(0.352, 0.2) == pytest.approx(0.405, abs=5e-4)
    # What the code separates out agrees with it to the rounding of 43.00 mA
    # to 2.54 pixels and of 0.0118 arcsec per 13.5 micron pixel to 0.159.
    assert spectral_optics_fwhm(TEL, DET) * plate_scale == pytest.approx(0.352, abs=0.002)
    assert slit_image_width(0.2 * u.arcsec, DET) == pytest.approx(0.2 / plate_scale)


def test_the_reference_slit_keeps_the_spectral_fwhm_it_was_given():
    """To the last bit, so that a 0.2 arcsec run is exactly what it was."""
    assert spectral_psf_fwhm(TEL, DET, 0.2 * u.arcsec) == 2.54
    other = Telescope_EUVST(psf_params=[2.66 * u.pixel, 3.1 * u.pixel])
    assert spectral_psf_fwhm(other, DET, 0.2 * u.arcsec) == 3.1


@pytest.mark.parametrize("slit, expected", [(0.4, 3.35), (0.8, 5.49), (1.6, 10.30)])
def test_wider_slits_add_their_width_in_quadrature(slit, expected):
    plate_scale = DET.plate_scale_angle.to_value(u.arcsec / u.pixel)
    optics_squared = 2.54**2 - (0.2 / plate_scale) ** 2
    fwhm = spectral_psf_fwhm(TEL, DET, slit * u.arcsec)
    assert fwhm == pytest.approx(np.sqrt(optics_squared + (slit / plate_scale) ** 2))
    assert fwhm == pytest.approx(expected, abs=0.005)


def test_the_reference_slit_blurs_exactly_as_one_fixed_psf_did():
    """What every slit got before is what the 0.2 arcsec slit still gets."""
    fixed = Telescope_EUVST(psf_slit_width=None)
    for convolve_spatial in (True, False):
        cube, sim = _frame(0.2)
        cube.data[0, 0, 10] = 3.0  # something that is not uniform along the slit
        now = apply_focusing_optics_psf(cube, TEL, DET, sim, convolve_spatial=convolve_spatial)
        before = apply_focusing_optics_psf(cube, fixed, DET, sim,
                                           convolve_spatial=convolve_spatial)
        assert np.array_equal(now.data, before.data)


@pytest.mark.parametrize("slit", [0.8, 1.6])
def test_a_wide_slit_broadens_the_line_to_the_quadrature_width(slit):
    profile = _profile(slit)
    assert profile.sum() == pytest.approx(1.0, rel=1e-12)
    assert _fwhm(profile) == pytest.approx(spectral_psf_fwhm(TEL, DET, slit * u.arcsec),
                                           rel=0.01)


# ----------------------------------------------------------------------
# The exact convolution
# ----------------------------------------------------------------------
@pytest.mark.parametrize("slit", SLITS)
def test_the_convolution_is_the_optics_gaussian_across_the_slit_image(slit):
    """Against a numerical convolution on a grid a thousand times finer."""
    kernel = spectral_line_spread(TEL, DET, slit * u.arcsec)
    sigma = _fwhm_to_sigma(spectral_optics_fwhm(TEL, DET))
    half = 0.5 * slit_image_width(slit * u.arcsec, DET)

    x = np.arange(-(kernel.size // 2), kernel.size // 2 + 1)
    # The Gaussian averaged across the rectangle, at each pixel centre, by the
    # midpoint rule on exactly the rectangle.
    n = 20000
    offsets = -half + (2 * half / n) * (np.arange(n) + 0.5)
    numerical = np.array([np.exp(-0.5 * ((centre - offsets) / sigma) ** 2).sum()
                          for centre in x])
    numerical /= numerical.sum()
    assert kernel.sum() == pytest.approx(1.0, rel=1e-12)
    assert kernel == pytest.approx(numerical, rel=1e-4, abs=1e-9)


def test_a_wide_slit_gives_a_flat_topped_profile():
    """The 1.6 arcsec slit is ten pixels wide against an optics FWHM of 2.2."""
    kernel = spectral_line_spread(TEL, DET, 1.6 * u.arcsec)
    centre = kernel.size // 2
    top = kernel[centre - 2:centre + 3]
    assert top.min() == pytest.approx(top.max(), rel=1e-3)
    # Its FWHM is the slit's image, where the Gaussian is rounder and wider
    # at the top.
    assert _fwhm(_profile(1.6, "convolution")) == pytest.approx(
        slit_image_width(1.6 * u.arcsec, DET), rel=0.03)
    gaussian = _profile(1.6)
    assert gaussian[centre - 2:centre + 3].min() < 0.95 * gaussian.max()


def test_the_convolution_is_narrower_than_quadrature_for_the_narrow_slit():
    """A rectangle adds less to a width than a Gaussian with its FWHM does."""
    convolved = _profile(0.2, "convolution")
    quadrature = _profile(0.2)
    assert convolved.sum() == pytest.approx(1.0, rel=1e-12)
    assert _variance(convolved) < _variance(quadrature)


# ----------------------------------------------------------------------
# EIS and the configuration
# ----------------------------------------------------------------------
@pytest.mark.parametrize("slit", [1, 2, 4])
def test_eis_keeps_one_spectral_psf_for_every_slit(slit):
    assert spectral_psf_fwhm(Telescope_EIS(), Detector_EIS(), slit * u.arcsec) == 3.0


def test_eis_cannot_convolve_a_psf_that_is_not_split_into_optics_and_slit():
    with pytest.raises(ValueError, match="psf_slit_width"):
        spectral_line_spread(Telescope_EIS(), Detector_EIS(), 1 * u.arcsec)


def test_a_spectral_fwhm_narrower_than_its_own_slit_is_refused():
    narrow = Telescope_EUVST(psf_params=[2.66 * u.pixel, 1.0 * u.pixel])
    with pytest.raises(ValueError, match="leaves nothing for the optics"):
        spectral_psf_fwhm(narrow, DET, 0.4 * u.arcsec)


def test_the_configuration_takes_the_new_settings():
    _validate_config_keys({"instrument": "SWC", "uniform_intensity": "5000 erg / (s cm2 sr)",
                           "simulation": {"spectral_psf": "convolution"},
                           "telescope": {"psf_slit_width": "0.2 arcsec"}}, "SWC")
    with pytest.raises(ValueError, match="spectral_psf must be"):
        Simulation(spectral_psf="gaussian")


def test_the_uniform_intensity_grid_holds_a_wide_slit_line():
    """The grid is sized for the line after this slit's PSF.

    Whether or not psf is on, as it was sized for the 0.2 arcsec PSF whether
    or not psf was on, so that a sweep of psf sees one grid per slit.
    """
    widths = []
    for slit in (0.2, 1.6):
        cube = create_uniform_intensity_cube(
            total_intensity=5000 * u.erg / (u.s * u.cm**2 * u.sr),
            rest_wavelength=REST, thermal_width=20 * u.km / u.s, det=DET,
            sim=Simulation(slit_width=slit * u.arcsec), tel=TEL)
        widths.append(cube.data.shape[-1])
    assert widths[1] > 3 * widths[0]


# ----------------------------------------------------------------------
# The wavelength window of a synthesis
# ----------------------------------------------------------------------
@pytest.mark.parametrize("slit, margin", [(0.2, 0), (0.4, 1), (0.8, 4), (1.6, 10)])
def test_a_wider_slit_widens_the_window_by_its_extra_reach(slit, margin):
    assert spectral_psf_margin(TEL, DET, slit * u.arcsec) == margin
    assert spectral_psf_margin(Telescope_EIS(), Detector_EIS(), 4 * u.arcsec) == 0


def test_a_widened_window_keeps_a_doppler_shifted_line():
    """The default synthesis window, 300 km/s either side, is 25 pixels at 195 A.

    A line at +200 km/s sits five pixels from its end. Blurred by the 1.6
    arcsec PSF, about a seventh of it would spread off the end of that
    window; the widened window holds all of it, and the pixels already there
    keep their wavelengths.
    """
    pixel = (DET.wvl_res * u.pixel).to(u.AA)
    shift = (200 * u.km / u.s / const.c * REST / pixel).to_value(u.dimensionless_unscaled)
    sigma = (20 * u.km / u.s / const.c * REST / pixel).to_value(u.dimensionless_unscaled)
    x = np.arange(25) - 12
    cube, sim = _frame(1.6, profile=np.exp(-0.5 * ((x - shift) / sigma) ** 2))
    total = cube.data.sum()

    margin = spectral_psf_margin(TEL, DET, sim.slit_width)
    widened = pad_spectral_axis(cube, margin)
    assert np.array_equal(widened.data[..., margin:-margin], cube.data)
    assert widened.wcs.wcs.crpix[0] == cube.wcs.wcs.crpix[0] + margin

    cut = apply_focusing_optics_psf(cube, TEL, DET, sim, convolve_spatial=False)
    kept = apply_focusing_optics_psf(widened, TEL, DET, sim, convolve_spatial=False)
    assert cut.data.sum() < 0.9 * total
    assert kept.data.sum() == pytest.approx(total, rel=1e-4)


def test_the_window_is_widened_only_along_wavelength():
    """The last data axis, which is the first WCS axis."""
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["HPLN-TAN", "HPLT-TAN", "WAVE"]
    cube = NDCube(np.zeros((4, 3, 2)), wcs=wcs)
    with pytest.raises(ValueError, match="last data axis, which is the first WCS axis"):
        pad_spectral_axis(cube, 1)


def test_the_reference_slit_cannot_be_swept(tmp_path, monkeypatch):
    """The grids are cached by slit width, and sized with it."""
    from euvst_response.main import main

    config = tmp_path / "sweep.yaml"
    config.write_text(yaml.safe_dump({
        "instrument": "SWC", "uniform_intensity": "5000 erg / (s cm2 sr)",
        "n_iter": 1, "ncpu": 1,
        "telescope": {"psf_slit_width": ["0.2 arcsec", "0.4 arcsec"]},
    }))
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="psf_slit_width.*cannot be swept"):
        main()


def test_an_instrument_run_widens_a_synthesis_window_for_a_wide_slit(tmp_path, monkeypatch):
    """From a synthesis file with the default window to the rebinned cubes the run keeps."""
    from euvst_response.main import main

    n_spec, n_cells = 121, 8  # 300 km/s either side at 5 km/s; 1.5 Mm square
    step = (5 * u.km / u.s / const.c * REST).to_value(u.cm)
    sigma = (20 * u.km / u.s / const.c * REST).to_value(u.cm)
    offsets = (np.arange(n_spec) - n_spec // 2) * step
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
    wcs.wcs.cunit = ["cm", "Mm", "Mm"]
    wcs.wcs.cdelt = [step, 0.192, 0.192]
    wcs.wcs.crpix = [(n_spec + 1) / 2, (n_cells + 1) / 2, (n_cells + 1) / 2]
    wcs.wcs.crval = [REST.to_value(u.cm), 0.0, 0.0]
    line = 1e13 * np.exp(-0.5 * (offsets / sigma) ** 2)
    cube = NDCube(np.tile(line, (n_cells, n_cells, 1)), wcs=wcs,
                  unit=u.erg / (u.s * u.cm**3 * u.sr),
                  meta={"rest_wav": REST, "integration_axis": "z",
                        "velocity_convention": VELOCITY_CONVENTION})
    synthesis_file = write_line_cubes({"Fe12_195.1190": cube}, tmp_path / "synthesis.h5")

    config = tmp_path / "window.yaml"
    config.write_text(yaml.safe_dump({
        "instrument": "SWC", "synthesis_file": str(synthesis_file),
        "reference_line": "Fe12_195.1190", "n_iter": 1, "ncpu": 1,
        "simulation": {"slit_width": ["0.2 arcsec", "1.6 arcsec"], "expos": "10 s",
                       "psf": True, "noise": False},
    }))
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    monkeypatch.chdir(tmp_path)
    main()

    cubes = load_instrument_response_results(tmp_path / "run" / "result" / "window.pkl")[
        "cube_reb_dict"]
    narrow, wide = cubes[(0.2, 1)], cubes[(1.6, 1)]
    margin = spectral_psf_margin(TEL, DET, 1.6 * u.arcsec)
    assert wide.data.shape[-1] == narrow.data.shape[-1] + 2 * margin
    assert wide.wcs.wcs.crpix[0] == narrow.wcs.wcs.crpix[0] + margin


def test_an_instrument_run_measures_the_width_each_slit_gives(tmp_path, monkeypatch):
    """From a configuration to fitted widths, with the noise off."""
    from euvst_response.main import main

    config = tmp_path / "slits.yaml"
    config.write_text(yaml.safe_dump({
        "instrument": "SWC",
        "uniform_intensity": "5000 erg / (s cm2 sr)",
        "n_iter": 1,
        "ncpu": 1,
        "simulation": {"slit_width": ["0.2 arcsec", "1.6 arcsec"], "expos": "10 s",
                       "psf": True, "noise": False,
                       "spectral_psf": ["quadrature", "convolution"]},
    }))
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    monkeypatch.chdir(tmp_path)
    main()

    results = load_instrument_response_results(tmp_path / "run" / "result" / "slits.pkl")
    measured = {}
    for combination in results["results"]["all_combinations"].values():
        parameters = combination["parameters"]
        slit = parameters["simulation.slit_width"].to_value(u.arcsec)
        stats = analyse_fit_statistics(combination)
        fitted = np.nanmean(stats["w_mean"].to_value(u.AA))
        true = np.nanmean(stats["w_true"].to_value(u.AA))
        measured[slit, parameters["simulation.spectral_psf"]] = (fitted, true)
    assert len(measured) == 4

    # A Gaussian PSF adds its sigma to the line's in quadrature.
    dispersion = DET.wvl_res.to_value(u.AA / u.pixel)
    for slit in (0.2, 1.6):
        fitted, true = measured[slit, "quadrature"]
        sigma_psf = _fwhm_to_sigma(spectral_psf_fwhm(TEL, DET, slit * u.arcsec)) * dispersion
        assert np.sqrt(fitted**2 - true**2) == pytest.approx(sigma_psf, rel=0.05)
    assert measured[1.6, "convolution"][0] > 2 * measured[0.2, "convolution"][0]
