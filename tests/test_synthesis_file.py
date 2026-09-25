"""The synthesis file: spectra from ECLIPSE's synthesis or any other code, as HDF5 (issue #107).

A synthesis file holds the spectral radiance leaving the Sun at each pixel
of an image, line by line, whoever synthesised it. These check the file and
what goes with it, the units the spectra may be in, the flux-conserving
resampling from a grid that is not evenly spaced, and that an instrument
run observes a synthesis file exactly as it observed the pickles of older
versions holding the same line cubes.
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
from euvst_response.atmosphere import Atmosphere, edges_from_centres, write_atmosphere
from euvst_response.config import Detector_SWC, Simulation
from euvst_response.data_processing import rebin_spectra, resample_spectra, sum_line_cubes
from euvst_response.synthesis_file import (
    RADIANCE_UNIT,
    SpectralLine,
    Synthesis,
    convert_synthesis_pickle,
    load_synthesis,
    read_synthesis,
    read_synthesis_products,
    write_line_cubes,
    write_synthesis,
)
from euvst_response.utils import VELOCITY_CONVENTION, angle_to_distance

LINE = "Fe12_195.1190"
REST = 195.119 * u.Angstrom
BLEND = "Fe12_195.1790"
BLEND_REST = 195.179 * u.Angstrom
NY, NX, N_WAVE = 6, 4, 121
# A power of two in Mm, so that edges built from it add up exactly.
PIXEL = 0.1875 * u.Mm
STEP = (5 * u.km / u.s / const.c * REST).to(u.cm)  # the synthesis's own grid


def _edges(n, size=PIXEL):
    return (np.arange(n + 1) - n / 2) * size


def _grid(rest=REST):
    return rest + (np.arange(N_WAVE) - N_WAVE // 2) * STEP


def _lines(ny=NY, nx=NX, rest=REST, scale=1.0):
    """A Gaussian line whose Doppler shift changes from pixel to pixel, (ny, nx, N_WAVE) in erg / (s cm2 sr cm)."""
    offsets = ((np.arange(N_WAVE) - N_WAVE // 2) * STEP).to_value(u.cm)
    sigma = (20 * u.km / u.s / const.c * rest).to_value(u.cm)
    shifts = np.linspace(-30, 30, ny * nx).reshape(ny, nx, 1)
    centres = (shifts * u.km / u.s / const.c * rest).to_value(u.cm)
    brightness = np.linspace(0.5, 1.5, ny * nx).reshape(ny, nx, 1)
    return scale * 1e13 * brightness * np.exp(-0.5 * ((offsets - centres) / sigma) ** 2)


def _line(intensity=None, wavelength=None, rest_wavelength=REST):
    return SpectralLine(intensity=_lines() * RADIANCE_UNIT if intensity is None else intensity,
                        wavelength=_grid() if wavelength is None else wavelength,
                        rest_wavelength=rest_wavelength)


def _synthesis(lines=None, **fields):
    lines = {LINE: _line()} if lines is None else lines
    ny, nx = next(iter(lines.values())).intensity.shape[:2]
    return Synthesis(**{"lines": lines, "x_edges": _edges(nx), "y_edges": _edges(ny), **fields})


# ----------------------------------------------------------------------
# The file
# ----------------------------------------------------------------------
def test_a_synthesis_file_round_trips(tmp_path):
    lines = {LINE: _line(),
             BLEND: _line(_lines(rest=BLEND_REST, scale=0.1) * RADIANCE_UNIT,
                          _grid(BLEND_REST), BLEND_REST),
             "Fe10_184.5370": _line(_lines(rest=184.537 * u.AA) * RADIANCE_UNIT,
                                    _grid(184.537 * u.AA), 184.537 * u.AA)}
    synthesis = _synthesis(lines, source="another code, snapshot 12", time=1250 * u.s)
    read = read_synthesis(write_synthesis(synthesis, tmp_path / "synthesis.h5"))
    # The lines keep the order they were written in, not HDF5's by name.
    assert list(read.lines) == [LINE, BLEND, "Fe10_184.5370"]
    for name, line in synthesis.lines.items():
        for field in ("intensity", "wavelength", "rest_wavelength"):
            assert u.allclose(getattr(read.lines[name], field), getattr(line, field), rtol=0)
    assert u.allclose(read.x_edges, synthesis.x_edges, rtol=0)
    assert read.source == "another code, snapshot 12"
    assert read.time == 1250 * u.s
    assert read_synthesis_products(tmp_path / "synthesis.h5") == {}
    # The time is optional, as a single snapshot does not need one.
    assert read_synthesis(write_synthesis(_synthesis(), tmp_path / "untimed.h5")).time is None


def test_a_strip_of_columns_reads_as_that_part_of_the_image(tmp_path):
    """A time series reads only the columns under the slit."""
    path = write_synthesis(_synthesis(time=10 * u.s), tmp_path / "synthesis.h5")
    whole = read_synthesis(path)
    strip = read_synthesis(path, LINE, columns=slice(1, 3))
    assert strip.shape == (NY, 2)
    assert np.array_equal(strip.lines[LINE].intensity.value, whole.lines[LINE].intensity.value[:, 1:3])
    assert u.allclose(strip.x_edges, whole.x_edges[1:4], rtol=0)
    assert u.allclose(strip.y_edges, whole.y_edges, rtol=0)
    assert strip.time == 10 * u.s
    with pytest.raises(ValueError, match="neighbouring pixels"):
        read_synthesis(path, columns=slice(0, 4, 2))


def test_only_the_lines_that_reach_the_window_are_read(tmp_path):
    lines = {LINE: _line(),
             BLEND: _line(_lines(rest=BLEND_REST, scale=0.1) * RADIANCE_UNIT,
                          _grid(BLEND_REST), BLEND_REST),
             "Fe10_184.5370": _line(_lines(rest=184.537 * u.AA) * RADIANCE_UNIT,
                                    _grid(184.537 * u.AA), 184.537 * u.AA)}
    path = write_synthesis(_synthesis(lines), tmp_path / "synthesis.h5")
    assert list(read_synthesis(path, reference_line=LINE).lines) == [LINE, BLEND]
    with pytest.raises(ValueError, match="is not in"):
        read_synthesis(path, reference_line="Fe99_100.0000")


def test_the_products_come_back_as_they_went_in(tmp_path):
    products = {
        "dem_map": np.arange(24.0).reshape(2, 3, 4),
        "vel_grid": np.linspace(-3e7, 3e7, 5) * u.cm / u.s,
        "goft": {LINE: {"g_tn": np.ones((2, 4)), "wl0": REST.to(u.cm), "atom": 26,
                        "ion": 12, "hdf5_dbase_root": None}},
        "voxel_sizes": {"dx": 0.192 * u.Mm, "dy": 0.192 * u.Mm, "dz": 0.064 * u.Mm},
        "dynamic_mode": {"enabled": True, "slit_width": 0.4 * u.arcsec,
                         "slit_rest_time": 10 * u.s, "available_timesteps": {"0001": 0.0}},
        "config": {"lines": [LINE], "vel_res": 5 * u.km / u.s, "crop_params": {"crop_x": None},
                   "integration_axis": "z"},
    }
    path = write_synthesis(_synthesis(), tmp_path / "synthesis.h5", products=products)
    read = read_synthesis_products(path)
    assert np.array_equal(read["dem_map"], products["dem_map"])
    assert u.allclose(read["vel_grid"], products["vel_grid"], rtol=0)
    assert np.array_equal(read["goft"][LINE]["g_tn"], np.ones((2, 4)))
    assert read["goft"][LINE]["wl0"] == REST.to(u.cm)
    assert read["goft"][LINE]["atom"] == 26 and read["goft"][LINE]["hdf5_dbase_root"] is None
    assert read["voxel_sizes"]["dz"] == 0.064 * u.Mm
    assert read["dynamic_mode"]["slit_width"] == 0.4 * u.arcsec
    assert read["config"]["vel_res"] == 5 * u.km / u.s
    assert read["config"]["crop_params"] == {"crop_x": None}
    # The instrument run reads only what it needs.
    assert set(read_synthesis_products(path, keys=("dynamic_mode",))) == {"dynamic_mode"}


def test_the_intensity_may_be_per_frequency_or_in_photons():
    """Converted at each wavelength, as I_lambda = I_nu c / lambda^2 and E = h c / lambda."""
    wavelength = _grid()
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
        radiance = _line(given, wavelength).radiance()
        assert radiance.unit == RADIANCE_UNIT
        assert np.allclose(radiance.value, per_wavelength.value, rtol=1e-12, atol=0)


@pytest.mark.parametrize("change, error, match", [
    (dict(intensity=np.ones((NY, NX, N_WAVE + 1)) * RADIANCE_UNIT), ValueError,
     r"\(y, x, wavelength\)"),
    (dict(intensity=np.full((NY, NX, N_WAVE), np.nan) * RADIANCE_UNIT), ValueError,
     "NaN or infinite"),
    (dict(intensity=-np.ones((NY, NX, N_WAVE)) * RADIANCE_UNIT), ValueError, "negative"),
    (dict(intensity=np.ones((NY, NX, N_WAVE)) * u.erg / (u.s * u.cm**2 * u.sr)),
     u.UnitConversionError, "spectral radiance"),
    (dict(intensity=np.ones((NY, NX, N_WAVE)) * u.erg / (u.s * u.cm**2 * u.AA)),
     u.UnitConversionError, "spectral radiance"),
    (dict(wavelength=_grid()[::-1]), ValueError, "must increase"),
    (dict(wavelength=np.arange(N_WAVE) * u.km / u.s), u.UnitConversionError, "unit of length"),
    (dict(rest_wavelength=300 * u.AA), ValueError, "outside the wavelengths"),
    (dict(rest_wavelength=20 * u.km / u.s), u.UnitConversionError, "unit of length"),
])
def test_a_line_is_checked(change, error, match):
    fields = {"intensity": _lines() * RADIANCE_UNIT, "wavelength": _grid(),
              "rest_wavelength": REST, **change}
    with pytest.raises(error, match=match):
        SpectralLine(**fields)


@pytest.mark.parametrize("change, error, match", [
    (dict(x_edges=np.array([0.0, 1.0, 2.0, 4.0, 5.0]) * u.Mm), ValueError, "evenly spaced"),
    (dict(x_edges=np.arange(NX + 1) * u.s), u.UnitConversionError, "length or angle"),
    (dict(y_edges=_edges(NY + 1)), ValueError, r"\(y, x, wavelength\)"),
    (dict(lines={}), ValueError, "at least one line"),
    (dict(lines={"Fe/12": None}), ValueError, "without '/'"),
    (dict(time=5 * u.m), u.UnitConversionError, "unit of time"),
    (dict(time=[0.0, 5.0] * u.s), ValueError, "0 dimensions"),
    (dict(time=np.nan * u.s), ValueError, "NaN or infinite"),
])
def test_a_synthesis_is_checked(change, error, match):
    fields = {"lines": {LINE: _line()}, "x_edges": _edges(NX), "y_edges": _edges(NY), **change}
    with pytest.raises(error, match=match):
        Synthesis(**fields)


def test_a_file_of_another_kind_is_refused(tmp_path):
    path = write_synthesis(_synthesis(), tmp_path / "synthesis.h5")
    with h5py.File(path, "r+") as f:
        f.attrs["format"] = "eclipse-atmosphere"
    with pytest.raises(ValueError, match="not an ECLIPSE synthesis file"):
        read_synthesis(path)


# ----------------------------------------------------------------------
# From ECLIPSE's line cubes, and from older pickles
# ----------------------------------------------------------------------
def _line_cube(data, rest=REST, view="z"):
    ctypes = {"z": ["WAVE", "SOLX", "SOLY"], "x": ["WAVE", "SOLY", "SOLZ"]}[view]
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ctypes
    wcs.wcs.cunit = ["cm", "Mm", "Mm"]
    wcs.wcs.cdelt = [STEP.to_value(u.cm), PIXEL.to_value(u.Mm), PIXEL.to_value(u.Mm)]
    wcs.wcs.crpix = [(N_WAVE + 1) / 2, (NX + 1) / 2, (NY + 1) / 2]
    wcs.wcs.crval = [rest.to_value(u.cm), 0.0, 0.0]
    return NDCube(data, wcs=wcs, unit=RADIANCE_UNIT,
                  meta={"rest_wav": rest.to(u.cm), "integration_axis": view,
                        "velocity_convention": VELOCITY_CONVENTION})


def _line_cubes():
    return {LINE: _line_cube(_lines()),
            BLEND: _line_cube(_lines(rest=BLEND_REST, scale=0.1), BLEND_REST)}


def test_line_cubes_become_a_synthesis_file_that_sums_as_they_did(tmp_path):
    cubes = _line_cubes()
    path = write_line_cubes(cubes, tmp_path / "synthesis.h5", source="a test")
    synthesis = read_synthesis(path)
    assert synthesis.pixel_size("x") == PIXEL and synthesis.centre("x") == 0 * u.Mm
    assert np.array_equal(synthesis.summed(LINE).value, sum_line_cubes(cubes, LINE).data)

    loaded = load_synthesis(path)
    cube = loaded["line_cubes"][BLEND]
    assert np.array_equal(cube.data, cubes[BLEND].data)
    assert u.allclose(cube.axis_world_coords(-1)[0], cubes[BLEND].axis_world_coords(-1)[0],
                      rtol=1e-12)
    assert cube.meta["rest_wav"] == BLEND_REST.to(u.cm)


def test_line_cubes_must_share_one_image(tmp_path):
    """Every line is written onto the first one's pixels, so a cube laid elsewhere is refused."""
    cubes = _line_cubes()
    cubes[BLEND].wcs.wcs.crval[1] += 0.5 * PIXEL.to_value(u.Mm)
    with pytest.raises(ValueError, match="other pixel positions along x"):
        write_line_cubes(cubes, tmp_path / "shifted.h5")
    cubes = _line_cubes()
    cubes[BLEND].wcs.wcs.cdelt[2] *= 2
    with pytest.raises(ValueError, match="other pixel positions along y"):
        write_line_cubes(cubes, tmp_path / "stretched.h5")


def test_uneven_wavelengths_are_read_with_read_synthesis_not_as_line_cubes(tmp_path):
    uneven = REST + np.concatenate([np.linspace(-0.3, -0.05, 6), np.linspace(-0.04, 0.04, 17),
                                    np.linspace(0.05, 0.3, 6)]) * u.AA
    line = SpectralLine(intensity=np.ones((NY, NX, uneven.size)) * RADIANCE_UNIT,
                        wavelength=uneven, rest_wavelength=REST)
    path = write_synthesis(_synthesis({LINE: line}), tmp_path / "uneven.h5")
    with pytest.raises(ValueError, match="Read the file with read_synthesis"):
        load_synthesis(path)
    assert u.allclose(read_synthesis(path).lines[LINE].wavelength, uneven, rtol=0)


def test_blends_keep_their_flux_on_the_reference_wavelengths():
    """A narrow blend between two of the reference line's wavelengths is added, not lost."""
    from euvst_response.utils import onto_wavelength_bins

    # By hand: samples standing for 0.5-1.5, 1.5-2.5 and 2.5-3.5, averaged
    # over bins 0.5-2.5 and 2.5-4.5, with nothing beyond 3.5.
    assert onto_wavelength_bins(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]),
                                np.array([1.5, 3.5])) == pytest.approx([1.5, 1.5])
    same = np.arange(6.0).reshape(2, 3)
    assert np.array_equal(onto_wavelength_bins(same, np.array([1.0, 2.0, 3.0]),
                                               np.array([1.0, 2.0, 3.0])), same)

    # A reference line sampled every 0.05 Angstrom, and a blend 0.005
    # Angstrom wide, sampled finely, centred between two of its wavelengths.
    coarse = REST + np.arange(-0.3, 0.3001, 0.05) * u.AA
    fine = REST + 0.025 * u.AA + np.arange(-0.05, 0.0501, 0.001) * u.AA
    ref_profile = np.exp(-0.5 * ((coarse - REST) / (0.04 * u.AA)).decompose().value ** 2)
    blend_profile = np.exp(-0.5 * ((fine - REST - 0.025 * u.AA) / (0.005 * u.AA)).decompose().value ** 2)
    reference = SpectralLine(intensity=np.ones((NY, NX, 1)) * ref_profile * RADIANCE_UNIT,
                             wavelength=coarse, rest_wavelength=REST)
    blend = SpectralLine(intensity=np.ones((NY, NX, 1)) * blend_profile * RADIANCE_UNIT,
                         wavelength=fine, rest_wavelength=REST + 0.025 * u.AA)

    alone = _synthesis({LINE: reference}).summed(LINE).value
    assert np.array_equal(alone, reference.intensity.value)
    together = _synthesis({LINE: reference, BLEND: blend}).summed(LINE).value
    widths = np.diff(edges_from_centres(coarse)).value
    added = ((together - alone) * widths).sum(-1)
    blend_flux = (blend_profile * np.diff(edges_from_centres(fine)).value).sum()
    assert added == pytest.approx(np.full((NY, NX), blend_flux), rel=1e-12)
    # Interpolating the blend onto the reference wavelengths, which fall
    # 5 of its widths from its centre, would have kept almost none of it.
    interpolated = np.interp(coarse.value, fine.value, blend_profile, left=0.0, right=0.0)
    assert (interpolated * widths).sum() < 1e-4 * blend_flux


def test_an_old_pickle_converts_with_everything_it_held(tmp_path):
    cubes = _line_cubes()
    with open(tmp_path / "old.pkl", "wb") as f:
        dill.dump({"line_cubes": cubes, "dem_map": np.ones((NY, NX, 3)),
                   "goft": {LINE: {"g_tn": np.ones((2, 3)), "si": cubes[LINE].data,
                                   "wl_grid": cubes[LINE].axis_world_coords(-1)[0]}},
                   "config": {"integration_axis": "z", "vel_res": 5 * u.km / u.s},
                   "atmosphere": {"source": "a simulation", "time": 42.0 * u.s},
                   "dynamic_mode": {"enabled": False}}, f)
    loaded = load_synthesis(convert_synthesis_pickle(tmp_path / "old.pkl", tmp_path / "new.h5"))
    # The snapshot's time goes with the spectra, as the synthesis now writes it.
    assert read_synthesis(tmp_path / "new.h5").time == 42.0 * u.s
    assert list(loaded["line_cubes"]) == [LINE, BLEND]
    assert np.array_equal(loaded["line_cubes"][LINE].data, cubes[LINE].data)
    assert np.array_equal(loaded["dem_map"], np.ones((NY, NX, 3)))
    # The spectra are the lines themselves, so they are not kept twice.
    assert set(loaded["goft"][LINE]) == {"g_tn"}
    assert loaded["config"]["vel_res"] == 5 * u.km / u.s

    wrong_sign = _line_cube(_lines())
    wrong_sign.meta.pop("velocity_convention")
    with open(tmp_path / "wrong.pkl", "wb") as f:
        dill.dump({"line_cubes": {LINE: wrong_sign}}, f)
    with pytest.raises(ValueError, match="wrong sign"):
        convert_synthesis_pickle(tmp_path / "wrong.pkl", tmp_path / "wrong.h5")


def test_the_synthesis_will_not_write_a_pickle(tmp_path, monkeypatch):
    from euvst_response.synthesis import main

    temperature = np.full((2, 2, 2), 1e6) * u.K
    atmosphere = write_atmosphere(Atmosphere(
        temperature=temperature, electron_density=np.full((2, 2, 2), 1e9) / u.cm**3,
        velocity_z=np.zeros((2, 2, 2)) * u.km / u.s, x_edges=np.arange(3) * u.Mm,
        y_edges=np.arange(3) * u.Mm, z_edges=np.arange(3) * u.Mm), tmp_path / "box.h5")
    monkeypatch.setattr(sys, "argv", ["synthesise-spectra", "--atmosphere", str(atmosphere),
                                      "--lines", LINE, "--output-dir", str(tmp_path),
                                      "--output-name", "old.pkl"])
    with pytest.raises(ValueError, match="ending in .h5"):
        main()


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
    in_angles = _synthesis(x_edges=_edges(NX, angle), y_edges=_edges(NY, angle))
    in_lengths = _synthesis(x_edges=angle_to_distance(_edges(NX, angle)),
                            y_edges=angle_to_distance(_edges(NY, angle)))
    a = rebin_spectra(in_angles, LINE, det, sim)
    b = rebin_spectra(in_lengths, LINE, det, sim)
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


def test_a_converted_pickle_is_observed_as_the_pickle_was(tmp_path, monkeypatch):
    """Both runs start from the same unpickled cubes, so they see the same numbers."""
    with open(tmp_path / "synthesis.pkl", "wb") as f:
        dill.dump({"line_cubes": _line_cubes()}, f)
    convert_synthesis_pickle(tmp_path / "synthesis.pkl", tmp_path / "synthesis.h5")

    with pytest.warns(FutureWarning, match="synthesis pickle"):
        from_pickle = _run(tmp_path, monkeypatch, "pickle",
                           synthesis_file=str(tmp_path / "synthesis.pkl"), reference_line=LINE)
    from_file = _run(tmp_path, monkeypatch, "file",
                     synthesis_file=str(tmp_path / "synthesis.h5"), reference_line=LINE)

    assert from_file["cube_reb_dict"].keys() == from_pickle["cube_reb_dict"].keys()
    for key, expected in from_pickle["cube_reb_dict"].items():
        got = from_file["cube_reb_dict"][key]
        assert np.array_equal(got.data, expected.data)
        assert got.wcs.to_header_string() == expected.wcs.to_header_string()
    pickle_runs = from_pickle["results"]["all_combinations"]
    file_runs = from_file["results"]["all_combinations"]
    assert file_runs.keys() == pickle_runs.keys()
    for key, expected in pickle_runs.items():
        got = file_runs[key]
        assert np.array_equal(got["first_dn_signal_data"], expected["first_dn_signal_data"])
        assert np.array_equal(got["ground_truth"]["fit_truth_data"],
                              expected["ground_truth"]["fit_truth_data"], equal_nan=True)
    # The spectra the instrument observed are kept, as they were from a pickle.
    assert np.array_equal(from_file["cube_sim"].data, from_pickle["cube_sim"].data)


def test_a_file_of_one_line_needs_no_reference_line(tmp_path, monkeypatch):
    write_synthesis(_synthesis({"Fe12_195.1193": _line(rest_wavelength=195.1193 * u.AA)}),
                    tmp_path / "one.h5")
    results = _run(tmp_path, monkeypatch, "one", synthesis_file=str(tmp_path / "one.h5"))
    assert results["cube_sim"].meta["rest_wav"] == 195.1193 * u.AA


def test_an_instrument_run_names_a_line_the_file_lacks(tmp_path, monkeypatch):
    write_synthesis(_synthesis(), tmp_path / "file.h5")
    with pytest.raises(ValueError, match="is not in"):
        _run(tmp_path, monkeypatch, "missing", synthesis_file=str(tmp_path / "file.h5"),
             reference_line="Fe10_184.5370")
