"""The synthesis file: spectra from ECLIPSE's synthesis or any other code, as HDF5 (issue #107).

A synthesis file holds the spectral radiance leaving the Sun at each pixel
of an image, line by line, whoever synthesised it. These check the file and
what goes with it, the units the spectra may be in, the flux-conserving
resampling from a grid that is not evenly spaced, and that an instrument
run observes a synthesis file exactly as it observed the pickles of older
versions holding the same line cubes.
"""
import sys
import warnings

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


def _line(intensity=None, wavelength=None, rest_wavelength=REST, **identity):
    return SpectralLine(intensity=_lines() * RADIANCE_UNIT if intensity is None else intensity,
                        wavelength=_grid() if wavelength is None else wavelength,
                        rest_wavelength=rest_wavelength, **identity)


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
    # Columns beyond the image are refused, not dropped.
    for outside in (slice(2, NX + 1), slice(-1, None)):
        with pytest.raises(ValueError, match=f"among the {NX} along x"):
            read_synthesis(path, columns=outside)


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

    # A line on a coarse grid whose wavelengths all lie beyond the window,
    # but whose first bin reaches back into it, adds what lies inside.
    coarse = REST + np.array([61.0, 64.0]) * STEP
    edge = SpectralLine(intensity=np.ones((NY, NX, 2)) * 1e13 * RADIANCE_UNIT,
                        wavelength=coarse, rest_wavelength=coarse[0])
    path = write_synthesis(_synthesis({LINE: _line(), "edge": edge}), tmp_path / "edge.h5")
    synthesis = read_synthesis(path, reference_line=LINE)
    assert list(synthesis.lines) == [LINE, "edge"]
    # Its first bin runs from 59.5 steps out; the window's last from 59.5 to
    # 60.5. The edges agree to the rounding of wavelengths in Angstrom, about
    # a part in 1e11 of a step.
    added = synthesis.summed(LINE).value - _line().radiance().value
    assert added[..., -1] == pytest.approx(np.full((NY, NX), 1e13), rel=1e-9)
    assert np.abs(added[..., :-1]).max() <= 1e-9 * 1e13

    # A blend whose wavelengths could not be read is refused rather than
    # passed over as outside the window.
    path = write_synthesis(_synthesis({LINE: _line(), BLEND: _line(rest_wavelength=BLEND_REST)}),
                           tmp_path / "broken.h5")
    for broken, match in ((_grid().to_value(u.AA)[::-1], "must increase"),
                          (np.full(N_WAVE, np.nan), "NaN or infinite")):
        with h5py.File(path, "r+") as f:
            f[f"lines/{BLEND}/wavelength"][...] = broken
        with pytest.raises(ValueError, match=f"line '{BLEND}': .*{match}"):
            read_synthesis(path, reference_line=LINE)


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
    # The instrument run reads only what it needs, named alone or in a list.
    assert set(read_synthesis_products(path, keys=("dynamic_mode",))) == {"dynamic_mode"}
    assert set(read_synthesis_products(path, keys="dynamic_mode")) == {"dynamic_mode"}
    # The lines keep their order, as the pickles kept it.
    goft = {name: {"g_tn": np.ones((2, 4)), "atom": 26, "ion": 12}
            for name in ("Fe12_195.1190", "Fe09_171.0730", "Fe24_192.0280")}
    path = write_synthesis(_synthesis(), tmp_path / "ordered.h5", products={"goft": goft})
    assert list(read_synthesis_products(path)["goft"]) == list(goft)


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
        # And a line cube, on a wavelength axis, is in it too.
        cube = _synthesis({LINE: _line(given, wavelength)}).line_cube(LINE)
        assert cube.unit == RADIANCE_UNIT
        assert np.allclose(cube.data, per_wavelength.value, rtol=1e-12, atol=0)


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


def _keeps_meta(got, expected):
    """Whether *got* has every entry of *expected*, the same."""
    for key, value in expected.items():
        assert key in got, key
        if isinstance(value, u.Quantity):
            assert u.allclose(got[key], value, rtol=1e-12), key
        else:
            assert got[key] == value, key


def test_line_cubes_become_a_synthesis_file_that_sums_as_they_did(tmp_path):
    cubes = _line_cubes()
    path = write_line_cubes(cubes, tmp_path / "synthesis.h5", source="a test")
    synthesis = read_synthesis(path)
    assert synthesis.pixel_size("x") == PIXEL and synthesis.centre("x") == 0 * u.Mm
    summed = sum_line_cubes(cubes, LINE)
    assert np.array_equal(synthesis.summed(LINE).value, summed.data)
    _keeps_meta(synthesis.summed_cube(LINE).meta, summed.meta)

    loaded = load_synthesis(path)
    cube = loaded["line_cubes"][BLEND]
    assert np.array_equal(cube.data, cubes[BLEND].data)
    assert u.allclose(cube.axis_world_coords(-1)[0], cubes[BLEND].axis_world_coords(-1)[0],
                      rtol=1e-12)
    assert cube.meta["rest_wav"] == BLEND_REST.to(u.cm)


def test_line_cubes_come_back_seen_along_the_axis_they_were(tmp_path):
    """A view along x reloads with the image axes of a view along x, however the file was written."""
    cubes = {LINE: _line_cube(_lines(), view="x"),
             BLEND: _line_cube(_lines(rest=BLEND_REST, scale=0.1), BLEND_REST, view="x")}
    path = write_line_cubes(cubes, tmp_path / "along_x.h5")
    assert read_synthesis(path).integration_axis == "x"
    for name, cube in load_synthesis(path)["line_cubes"].items():
        assert list(cube.wcs.wcs.ctype) == ["WAVE", "SOLY", "SOLZ"]
        assert cube.meta["integration_axis"] == "x"
        assert np.array_equal(cube.data, cubes[name].data)

    # A pickle holding nothing but its line cubes converts the same way.
    with open(tmp_path / "along_x.pkl", "wb") as f:
        dill.dump({"line_cubes": cubes}, f)
    converted = convert_synthesis_pickle(tmp_path / "along_x.pkl", tmp_path / "converted.h5")
    assert read_synthesis(converted).integration_axis == "x"

    # Another code's file names no view, and is taken as seen along z.
    path = write_synthesis(_synthesis(), tmp_path / "other_code.h5")
    assert read_synthesis(path).integration_axis is None
    assert list(load_synthesis(path)["line_cubes"][LINE].wcs.wcs.ctype) == ["WAVE", "SOLX", "SOLY"]

    cubes[BLEND] = _line_cube(_lines(rest=BLEND_REST, scale=0.1), BLEND_REST, view="z")
    with pytest.raises(ValueError, match="seen along different axes"):
        write_line_cubes(cubes, tmp_path / "mixed.h5")
    # A cube that names no view is seen along z, so it cannot join one along x.
    cubes[BLEND].meta.pop("integration_axis")
    with pytest.raises(ValueError, match="seen along different axes"):
        write_line_cubes(cubes, tmp_path / "unnamed.h5")
    with pytest.raises(ValueError, match="integration_axis must be one of"):
        _synthesis(integration_axis="w")


def test_each_line_keeps_its_atom_and_ion(tmp_path):
    cubes = _line_cubes()
    cubes[LINE].meta.update(atom=np.int64(26), ion=12)
    path = write_line_cubes(cubes, tmp_path / "identified.h5")
    lines = read_synthesis(path).lines
    assert (lines[LINE].atom, lines[LINE].ion) == (26, 12)
    assert (lines[BLEND].atom, lines[BLEND].ion) == (None, None)
    loaded = load_synthesis(path)["line_cubes"]
    assert (loaded[LINE].meta["atom"], loaded[LINE].meta["ion"]) == (26, 12)
    assert "atom" not in loaded[BLEND].meta

    for atom, ion, message in ((0, 1, "atom must be"), (26, 0, "ion must be"),
                               (26.0, 12, "atom must be"), (True, 1, "atom must be"),
                               (26, 28, "beyond the last stage")):
        with pytest.raises(ValueError, match=message):
            _line(atom=atom, ion=ion)
    assert _line(atom=26, ion=27).ion == 27


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
    # A sample that is not a number spoils only the bins it overlaps.
    spoilt = onto_wavelength_bins(np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0]),
                                  np.arange(1.0, 7.0), np.array([1.5, 3.5, 5.5]))
    assert np.isnan(spoilt[0]) and np.all(np.isfinite(spoilt[1:]))
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


def _flat_goft(lines, **kwargs):
    """A contribution function the same at every temperature and density, in place of fiasco."""
    logT_grid, logN_grid = np.linspace(5.0, 7.0, 21), np.linspace(8.0, 10.0, 21)
    return ({name: {"wl0": REST.to(u.cm), "g_tn": np.full((21, 21), 1e-24), "atom": 26,
                    "ion": 12, "hdf5_dbase_root": None} for name in lines},
            logT_grid, logN_grid)


def test_a_pickle_name_still_gets_the_pickle_older_versions_wrote(tmp_path, monkeypatch):
    from euvst_response import synthesis

    monkeypatch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    atmosphere = write_atmosphere(Atmosphere(
        temperature=np.full((2, 2, 2), 1e6) * u.K,
        electron_density=np.full((2, 2, 2), 1e9) / u.cm**3,
        velocity_z=np.zeros((2, 2, 2)) * u.km / u.s, x_edges=np.arange(3) * u.Mm,
        y_edges=np.arange(3) * u.Mm, z_edges=np.arange(3) * u.Mm), tmp_path / "box.h5")
    for name in ("old.pkl", "new.h5", "plain"):
        monkeypatch.setattr(sys, "argv", ["synthesise-spectra", "--atmosphere", str(atmosphere),
                                          "--lines", LINE, "--output-dir", str(tmp_path),
                                          "--output-name", name])
        if name.endswith(".pkl"):
            with pytest.warns(FutureWarning, match="Writing one is deprecated"):
                synthesis.main()
        elif name == "plain":
            # Any other name got a pickle from older versions, so it is told.
            with pytest.warns(UserWarning, match="gets a synthesis file, which is HDF5"):
                synthesis.main()
            assert h5py.is_hdf5(tmp_path / "plain")
        else:
            synthesis.main()

    with open(tmp_path / "old.pkl", "rb") as f:
        saved = dill.load(f)
    assert list(saved) == ["line_cubes", "dem_map", "em_tv", "logT_grid", "vel_grid",
                           "logN_grid", "goft", "voxel_sizes", "dynamic_mode", "atmosphere",
                           "config"]
    # Contribution functions and all, as older versions kept them.
    assert {"si", "wl_grid"} <= set(saved["goft"][LINE])
    assert np.array_equal(saved["line_cubes"][LINE].data,
                          load_synthesis(tmp_path / "new.h5")["line_cubes"][LINE].data)


def test_a_run_left_on_the_old_default_file_still_finds_it(tmp_path, monkeypatch):
    """With no synthesis_file, a pickle where older versions wrote it is observed, with a warning."""
    (tmp_path / "run" / "input").mkdir(parents=True)
    with open(tmp_path / "run" / "input" / "synthesised_spectra.pkl", "wb") as f:
        dill.dump({"line_cubes": _line_cubes()}, f)
    with pytest.warns(FutureWarning, match="synthesis pickle"):
        by_default = _run(tmp_path, monkeypatch, "default", reference_line=LINE)
    with pytest.warns(FutureWarning, match="synthesis pickle"):
        named = _run(tmp_path, monkeypatch, "named", reference_line=LINE,
                     synthesis_file="./run/input/synthesised_spectra.pkl")
    for key, expected in named["cube_reb_dict"].items():
        assert np.array_equal(by_default["cube_reb_dict"][key].data, expected.data)

    # Once there is a synthesis file where the synthesis now writes it, that
    # is the one, and the run says which it chose.
    convert_synthesis_pickle(tmp_path / "run" / "input" / "synthesised_spectra.pkl",
                             tmp_path / "run" / "input" / "synthesised_spectra.h5")
    with pytest.warns(UserWarning, match="observing ./run/input/synthesised_spectra.h5") as seen:
        _run(tmp_path, monkeypatch, "new_default", reference_line=LINE)
    assert not any(issubclass(warning.category, FutureWarning) for warning in seen)


# ----------------------------------------------------------------------
# Onto the detector
# ----------------------------------------------------------------------
def test_an_uneven_wavelength_grid_is_resampled_conserving_the_intensity():
    """Each input wavelength stands for the interval halfway to its neighbours, as in a synthesis, out to the outermost."""
    rng = np.random.default_rng(107)
    # Dense in the middle and, at the ends, coarser than the detector's
    # pixels, with intensity all the way out.
    wavelength = np.concatenate([[194.80, 194.86], np.sort(194.9 + 0.4 * rng.random(300)),
                                 [195.34, 195.40]]) * u.AA
    intensity = 1e13 * rng.random(wavelength.size)
    edges = edges_from_centres(wavelength).value
    given = np.sum(intensity * np.diff(edges))

    pitch = 22.3e-3 * u.AA
    resampled, grid = resample_spectra(intensity[np.newaxis], wavelength, pitch)
    assert np.allclose(np.diff(grid.value), pitch.value)
    assert np.sum(resampled) * pitch.value == pytest.approx(given, rel=1e-12)
    # The grid still steps from the first wavelength, and reaches past it by
    # whole pixels only as far as the outermost intervals need.
    steps = (grid.value - wavelength[0].value) / pitch.value
    assert steps == pytest.approx(np.round(steps), abs=1e-6)
    assert grid.value[0] - pitch.value / 2 <= edges[0] < grid.value[0] + pitch.value / 2
    assert grid.value[-1] - pitch.value / 2 < edges[-1] <= grid.value[-1] + pitch.value / 2

    # A grid finer than the pixels, as ECLIPSE's own is, needs none added.
    _, fine = resample_spectra(_lines(), _grid(), pitch)
    assert fine[0] == _grid()[0].to(u.AA)


@pytest.mark.parametrize("spacing", ["even", "dense in the core"])
def test_each_detector_pixel_gets_the_mean_of_the_line_over_it(spacing):
    """A finely sampled Gaussian, resampled, gives each pixel the Gaussian's exact mean over it."""
    from scipy.special import erf

    centre, sigma, pitch = REST.value + 0.0117, 0.03, 0.0223  # Angstrom
    t = np.linspace(-1.0, 1.0, 3001)
    # Out to 10 sigma either side, evenly or, as another code might, from
    # 0.00006 Angstrom apart in the core to 0.0006 in the wings.
    wavelength = REST.value + (0.3 * t if spacing == "even" else 0.3 * np.sinh(3 * t) / np.sinh(3))
    intensity = np.exp(-0.5 * ((wavelength - centre) / sigma) ** 2)

    resampled, grid = resample_spectra(intensity[np.newaxis], wavelength * u.AA, pitch * u.AA)

    def integral(lower, upper):
        scale = np.sqrt(2) * sigma
        return 0.5 * np.sqrt(np.pi) * scale * (erf((upper - centre) / scale)
                                               - erf((lower - centre) / scale))

    exact = integral(grid.value - pitch / 2, grid.value + pitch / 2) / pitch
    # What is left is the sampling of the Gaussian, 2e-6 of its peak;
    # pixels labelled one sample off would be out by 1e-3 or more.
    assert resampled[0] == pytest.approx(exact, rel=0, abs=1e-4)
    assert grid.value[np.argmax(resampled[0])] == pytest.approx(centre, abs=pitch / 2)


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
    """Both runs start from the same unpickled cubes, so they see the same numbers, and describe them alike."""
    cubes = _line_cubes()
    for cube in cubes.values():
        cube.meta.update(atom=26, ion=12)
    with open(tmp_path / "synthesis.pkl", "wb") as f:
        dill.dump({"line_cubes": cubes}, f)
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
        _keeps_meta(got.meta, expected.meta)
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
    _keeps_meta(from_file["cube_sim"].meta, from_pickle["cube_sim"].meta)


def test_a_uniform_intensity_run_says_it_ignores_a_synthesis_file(tmp_path, monkeypatch):
    """The uniform intensity is what is observed, as it always was, but not silently."""
    with pytest.warns(UserWarning, match="'synthesis_file' is ignored"):
        results = _run(tmp_path, monkeypatch, "uniform", uniform_intensity="5000 erg / (s cm2 sr)",
                       synthesis_file=str(tmp_path / "nowhere.h5"))
    assert len(results["results"]["all_combinations"]) == 2


def test_a_file_of_one_line_needs_no_reference_line(tmp_path, monkeypatch):
    write_synthesis(_synthesis({"Fe12_195.1193": _line(rest_wavelength=195.1193 * u.AA)}),
                    tmp_path / "one.h5")
    results = _run(tmp_path, monkeypatch, "one", synthesis_file=str(tmp_path / "one.h5"))
    assert results["cube_sim"].meta["rest_wav"] == 195.1193 * u.AA


def test_a_file_of_several_lines_needs_the_reference_line_it_lacks(tmp_path, monkeypatch):
    lines = {"Fe10_184.5370": _line(_lines(rest=184.537 * u.AA) * RADIANCE_UNIT,
                                    _grid(184.537 * u.AA), 184.537 * u.AA),
             "Fe09_171.0730": _line(_lines(rest=171.073 * u.AA) * RADIANCE_UNIT,
                                    _grid(171.073 * u.AA), 171.073 * u.AA)}
    write_synthesis(_synthesis(lines), tmp_path / "two.h5")
    with pytest.raises(ValueError, match="no 'reference_line' says which to observe"):
        _run(tmp_path, monkeypatch, "unsaid", synthesis_file=str(tmp_path / "two.h5"))
    # Left empty, it is the first line, as older versions took it.
    results = _run(tmp_path, monkeypatch, "empty", synthesis_file=str(tmp_path / "two.h5"),
                   reference_line=None)
    assert results["cube_sim"].meta["line_name"] == "Fe10_184.5370"


def test_load_atmosphere_reads_a_synthesis_file_as_it_read_the_pickle(tmp_path):
    from euvst_response.data_processing import load_atmosphere

    with open(tmp_path / "synthesis.pkl", "wb") as f:
        dill.dump({"line_cubes": _line_cubes(), "dynamic_mode": {"enabled": False}}, f)
    convert_synthesis_pickle(tmp_path / "synthesis.pkl", tmp_path / "synthesis.h5")
    for line in (LINE, None):
        expected, expected_mode = load_atmosphere(tmp_path / "synthesis.pkl", line)
        got, mode = load_atmosphere(tmp_path / "synthesis.h5", line)
        assert mode == expected_mode == {"enabled": False}
        assert np.array_equal(got.data, expected.data)
        # The WCS is built again from the file's wavelengths, so the same
        # axes but for rounding.
        assert list(got.wcs.wcs.ctype) == list(expected.wcs.wcs.ctype)
        for axis, n in enumerate(got.data.shape[::-1]):
            pixels = np.zeros((n, 3))
            pixels[:, axis] = np.arange(n)
            assert np.allclose(got.wcs.pixel_to_world_values(*pixels.T)[axis],
                               expected.wcs.pixel_to_world_values(*pixels.T)[axis],
                               rtol=1e-12, atol=0)
        _keeps_meta(got.meta, expected.meta)


def test_an_instrument_run_names_a_line_the_file_lacks(tmp_path, monkeypatch):
    write_synthesis(_synthesis(), tmp_path / "file.h5")
    with pytest.raises(ValueError, match="is not in"):
        _run(tmp_path, monkeypatch, "missing", synthesis_file=str(tmp_path / "file.h5"),
             reference_line="Fe10_184.5370")
