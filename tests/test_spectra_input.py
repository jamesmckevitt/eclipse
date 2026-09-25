"""Spectra from another code go straight into the instrument simulation (issue #107).

A spectra file holds the spectral radiance leaving the Sun at each pixel and
wavelength, as some other code synthesised it. These check the file, the
units it may be in, the flux-conserving resampling from a grid that is not
evenly spaced, and that an instrument run sees a spectra file exactly as it
sees a synthesis file holding the same spectra.
"""
import sys

import astropy.constants as const
import astropy.units as u
import dill
import h5py
import numpy as np
import pytest
import yaml
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.analysis import load_instrument_response_results
from euvst_response.config import Detector_SWC, Simulation
from euvst_response.data_processing import rebin_spectra, resample_spectra
from euvst_response.spectra import RADIANCE_UNIT, Spectra, read_spectra, write_spectra
from euvst_response.utils import VELOCITY_CONVENTION, angle_to_distance

LINE = "Fe12_195.1190"
REST = 195.119 * u.Angstrom
NY, NX, N_WAVE = 6, 4, 121
# A power of two in Mm, so that edges built from it add up exactly.
PIXEL = 0.1875 * u.Mm
STEP = (5 * u.km / u.s / const.c * REST).to(u.cm)  # the synthesis's own grid


def _edges(n, size=PIXEL):
    return (np.arange(n + 1) - n / 2) * size


def _lines(ny=NY, nx=NX):
    """A Gaussian line whose Doppler shift changes from pixel to pixel, (ny, nx, N_WAVE) in erg / (s cm2 sr cm)."""
    offsets = ((np.arange(N_WAVE) - N_WAVE // 2) * STEP).to_value(u.cm)
    sigma = (20 * u.km / u.s / const.c * REST).to_value(u.cm)
    shifts = np.linspace(-30, 30, ny * nx).reshape(ny, nx, 1)
    centres = (shifts * u.km / u.s / const.c * REST).to_value(u.cm)
    brightness = np.linspace(0.5, 1.5, ny * nx).reshape(ny, nx, 1)
    return 1e13 * brightness * np.exp(-0.5 * ((offsets - centres) / sigma) ** 2)


def _spectra(intensity=None, wavelength=None, **fields):
    if intensity is None:
        intensity = _lines() * RADIANCE_UNIT
    if wavelength is None:
        wavelength = REST + (np.arange(N_WAVE) - N_WAVE // 2) * STEP
    ny, nx = intensity.shape[:2]
    return Spectra(**{"intensity": intensity, "wavelength": wavelength,
                      "x_edges": _edges(nx), "y_edges": _edges(ny), **fields})


# ----------------------------------------------------------------------
# The file
# ----------------------------------------------------------------------
def test_a_spectra_file_round_trips(tmp_path):
    spectra = _spectra(source="another code, snapshot 12")
    read = read_spectra(write_spectra(spectra, tmp_path / "spectra.h5"))
    for name in ("intensity", "wavelength", "x_edges", "y_edges"):
        assert u.allclose(getattr(read, name), getattr(spectra, name), rtol=0)
    assert read.source == "another code, snapshot 12"


def test_the_intensity_may_be_per_frequency_or_in_photons():
    """Converted at each wavelength, as I_lambda = I_nu c / lambda^2 and E = h c / lambda."""
    wavelength = REST + (np.arange(N_WAVE) - N_WAVE // 2) * STEP
    per_wavelength = _lines() * RADIANCE_UNIT
    lam = wavelength.to_value(u.cm)
    c = const.c.to_value(u.cm / u.s)
    h = const.h.to_value(u.erg * u.s)

    per_frequency = (per_wavelength.value * lam**2 / c) * u.erg / (u.s * u.cm**2 * u.sr * u.Hz)
    per_frequency_si = per_frequency.to(u.W / (u.m**2 * u.sr * u.Hz))
    photons = (per_wavelength.value * lam / (h * c)) * u.ph / (u.s * u.cm**2 * u.sr * u.cm)
    photons_per_angstrom = photons.to(u.ph / (u.s * u.cm**2 * u.sr * u.AA))

    for given in (per_wavelength.to(u.erg / (u.s * u.cm**2 * u.sr * u.AA)),
                  per_frequency_si, photons_per_angstrom):
        radiance = _spectra(intensity=given, wavelength=wavelength).radiance()
        assert radiance.unit == RADIANCE_UNIT
        assert np.allclose(radiance.value, per_wavelength.value, rtol=1e-12, atol=0)


@pytest.mark.parametrize("change, error, match", [
    (dict(intensity=np.ones((NY, NX + 1, N_WAVE)) * RADIANCE_UNIT), ValueError,
     r"\(y, x, wavelength\)"),
    (dict(intensity=np.full((NY, NX, N_WAVE), np.nan) * RADIANCE_UNIT), ValueError,
     "NaN or infinite"),
    (dict(intensity=-np.ones((NY, NX, N_WAVE)) * RADIANCE_UNIT), ValueError, "negative"),
    (dict(intensity=np.ones((NY, NX, N_WAVE)) * u.erg / (u.s * u.cm**2 * u.sr)),
     u.UnitConversionError, "spectral radiance"),
    (dict(intensity=np.ones((NY, NX, N_WAVE)) * u.erg / (u.s * u.cm**2 * u.AA)),
     u.UnitConversionError, "spectral radiance"),
    (dict(wavelength=(REST + (np.arange(N_WAVE) - N_WAVE // 2) * STEP)[::-1]), ValueError,
     "must increase"),
    (dict(wavelength=np.arange(N_WAVE) * u.km / u.s), u.UnitConversionError, "unit of length"),
    (dict(x_edges=np.array([0.0, 1.0, 2.0, 4.0, 5.0]) * u.Mm), ValueError, "evenly spaced"),
    (dict(x_edges=np.arange(NX + 1) * u.s), u.UnitConversionError, "length or angle"),
])
def test_a_spectra_file_is_checked(change, error, match):
    fields = {"intensity": _lines() * RADIANCE_UNIT,
              "wavelength": REST + (np.arange(N_WAVE) - N_WAVE // 2) * STEP,
              "x_edges": _edges(NX), "y_edges": _edges(NY), **change}
    with pytest.raises(error, match=match):
        Spectra(**fields)


def test_a_file_of_another_kind_is_refused(tmp_path):
    path = write_spectra(_spectra(), tmp_path / "spectra.h5")
    with h5py.File(path, "r+") as f:
        f.attrs["format"] = "eclipse-atmosphere"
    with pytest.raises(ValueError, match="not an ECLIPSE spectra file"):
        read_spectra(path)


# ----------------------------------------------------------------------
# Onto the detector
# ----------------------------------------------------------------------
def test_an_uneven_wavelength_grid_is_resampled_conserving_the_intensity():
    """Each input wavelength stands for the interval halfway to its neighbours, as in a synthesis."""
    rng = np.random.default_rng(107)
    wavelength = np.sort(194.9 + 0.4 * rng.random(300)) * u.AA
    # Zero near the ends, so that every interval with intensity lies inside
    # the detector grid, which starts at the first wavelength.
    intensity = np.where((wavelength > 195.0 * u.AA) & (wavelength < 195.2 * u.AA),
                         1e13 * rng.random(300), 0.0)
    edges = np.concatenate([[wavelength[0].value - 0.5 * (wavelength[1] - wavelength[0]).value],
                            0.5 * (wavelength[1:] + wavelength[:-1]).value,
                            [wavelength[-1].value + 0.5 * (wavelength[-1] - wavelength[-2]).value]])
    given = np.sum(intensity * np.diff(edges))

    pitch = 22.3e-3 * u.AA
    resampled, grid = resample_spectra(intensity[np.newaxis], RADIANCE_UNIT, wavelength,
                                       pitch, ncpu=1)
    assert np.allclose(np.diff(grid.value), pitch.value)
    assert np.sum(resampled) * pitch.value == pytest.approx(given, rel=1e-12)


def test_angles_give_the_same_cube_as_the_lengths_they_stand_for():
    det, sim = Detector_SWC(), Simulation(expos=1 * u.s, n_iter=1, slit_width=0.2 * u.arcsec,
                                          ncpu=1, instrument="SWC", psf=False)
    angle = 0.25 * u.arcsec
    in_angles = _spectra(x_edges=_edges(NX, angle), y_edges=_edges(NY, angle))
    in_lengths = _spectra(x_edges=angle_to_distance(_edges(NX, angle)),
                          y_edges=angle_to_distance(_edges(NY, angle)))
    a = rebin_spectra(in_angles, REST, det, sim)
    b = rebin_spectra(in_lengths, REST, det, sim)
    # A field exactly 5 slit widths across keeps all 5, whichever side of 5
    # the conversion to a length and back rounds it; 1.5 arcsec along the
    # slit holds 9 whole pixels of 0.159 arcsec.
    assert a.data.shape[:2] == b.data.shape[:2] == (9, 5)
    # The same to within the curvature of the conversion at 1 AU, which the
    # lengths' uneven edges carry and the angles' pixel size does not.
    assert np.allclose(a.data, b.data, rtol=1e-9, atol=0)
    assert np.allclose(a.wcs.wcs.cdelt, b.wcs.wcs.cdelt, rtol=1e-9)
    # Across the slit by the slit width, along it by the plate scale.
    pitch = [(a.wcs.wcs.cdelt[i] * u.Unit(a.wcs.wcs.cunit[i])).to_value(u.arcsec) for i in (1, 2)]
    assert np.allclose(pitch, [0.2, 0.159])
    assert a.meta["rest_wav"] == REST
    assert a.meta["velocity_convention"] == VELOCITY_CONVENTION


# ----------------------------------------------------------------------
# An instrument run
# ----------------------------------------------------------------------
def _synthesis_cube(data):
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
    wcs.wcs.cunit = ["cm", "Mm", "Mm"]
    wcs.wcs.cdelt = [STEP.to_value(u.cm), PIXEL.to_value(u.Mm), PIXEL.to_value(u.Mm)]
    wcs.wcs.crpix = [(N_WAVE + 1) / 2, (NX + 1) / 2, (NY + 1) / 2]
    wcs.wcs.crval = [REST.to_value(u.cm), 0.0, 0.0]
    return NDCube(data, wcs=wcs, unit=RADIANCE_UNIT,
                  meta={"rest_wav": REST, "integration_axis": "z",
                        "velocity_convention": VELOCITY_CONVENTION})


def _run(tmp_path, monkeypatch, name, **inputs):
    from euvst_response.main import main

    config = tmp_path / f"{name}.yaml"
    config.write_text(yaml.safe_dump({
        "instrument": "SWC", "n_iter": 1, "ncpu": 1, **inputs,
        "simulation": {"slit_width": ["0.2 arcsec", "0.4 arcsec"], "expos": "10 s",
                       "psf": True, "noise": False},
    }))
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    monkeypatch.chdir(tmp_path)
    main()
    return load_instrument_response_results(tmp_path / "run" / "result" / f"{name}.pkl")


def test_a_spectra_file_is_observed_as_a_synthesis_file_of_the_same_spectra(tmp_path, monkeypatch):
    cube = _synthesis_cube(_lines())
    with open(tmp_path / "synthesis.pkl", "wb") as f:
        dill.dump({"line_cubes": {LINE: cube}}, f)
    # The wavelengths as the synthesis cube's WCS gives them, so that both
    # runs resample from the same numbers.
    write_spectra(Spectra(intensity=cube.data * cube.unit,
                          wavelength=cube.axis_world_coords(-1)[0],
                          x_edges=_edges(NX), y_edges=_edges(NY)),
                  tmp_path / "spectra.h5")

    from_synthesis = _run(tmp_path, monkeypatch, "synthesis",
                          synthesis_file=str(tmp_path / "synthesis.pkl"), reference_line=LINE)
    from_spectra = _run(tmp_path, monkeypatch, "spectra",
                        spectra_file=str(tmp_path / "spectra.h5"),
                        rest_wavelength="195.119 Angstrom")

    assert from_spectra["cube_reb_dict"].keys() == from_synthesis["cube_reb_dict"].keys()
    for key, expected in from_synthesis["cube_reb_dict"].items():
        got = from_spectra["cube_reb_dict"][key]
        assert np.array_equal(got.data, expected.data)
        assert got.wcs.to_header_string() == expected.wcs.to_header_string()
    synthesis_runs = from_synthesis["results"]["all_combinations"]
    spectra_runs = from_spectra["results"]["all_combinations"]
    assert spectra_runs.keys() == synthesis_runs.keys()
    for key, expected in synthesis_runs.items():
        got = spectra_runs[key]
        assert np.array_equal(got["first_dn_signal_data"], expected["first_dn_signal_data"])
        assert np.array_equal(got["ground_truth"]["fit_truth_data"],
                              expected["ground_truth"]["fit_truth_data"], equal_nan=True)


@pytest.mark.parametrize("config, match", [
    ({"synthesis_file": "synthesis.pkl"}, "not both"),
    ({"reference_line": LINE}, "is not read with a 'spectra_file'"),
    ({"rest_wavelength": None}, "needs a 'rest_wavelength'"),
    ({"rest_wavelength": "300 Angstrom"}, "outside the spectra"),
    ({"rest_wavelength": "20 km / s"}, "one wavelength with its unit"),
])
def test_an_instrument_run_checks_what_comes_with_a_spectra_file(tmp_path, monkeypatch,
                                                                 config, match):
    from euvst_response.main import main

    write_spectra(_spectra(), tmp_path / "spectra.h5")
    settings = {"instrument": "SWC", "n_iter": 1, "ncpu": 1,
                "spectra_file": str(tmp_path / "spectra.h5"),
                "rest_wavelength": "195.119 Angstrom", **config,
                "simulation": {"slit_width": "0.2 arcsec", "expos": "10 s"}}
    settings = {key: value for key, value in settings.items() if value is not None}
    path = tmp_path / "checked.yaml"
    path.write_text(yaml.safe_dump(settings))
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(path)])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=match):
        main()
