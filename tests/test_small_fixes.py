"""Small things that went wrong around the edges.

The command line hid what kind of error stopped a run, printed a syntax
warning from its logo and pointed to a folder that is not there; --debug
opened on the wrong frame; the git commit recorded could be the user's own
project's; atmosphere info described files it could not read, and files
written the IDL way were refused; a script's old option name was ignored;
the science-case tool wrote configurations the run refuses, and shared
their lists; the deprecated dynamic mode mixed its snapshots along the line
of sight, and crashed on a crop one cell wide.
"""
import argparse
import subprocess
import sys
import warnings
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pytest

from euvst_response import cli, utils
from euvst_response.atmosphere import Atmosphere, describe_atmosphere_file, read_atmosphere, write_atmosphere
from euvst_response.raster import AtmosphereSeries


def test_the_logo_is_no_syntax_warning():
    source = Path(cli.__file__).read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(source, cli.__file__, "exec")
    assert "\\ /" in cli.ASCII_LOGO


def _cli(monkeypatch, tmp_path, error):
    config = tmp_path / "run.yaml"
    config.write_text("instrument: SWC\n")

    def run():
        raise error

    monkeypatch.setattr(cli, "run_simulation", run)
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    with pytest.raises(SystemExit):
        cli.main()


def test_an_unexpected_error_says_what_it_was_and_where(monkeypatch, tmp_path, capsys):
    _cli(monkeypatch, tmp_path, TypeError("boom"))
    captured = capsys.readouterr()
    assert "Error during simulation: TypeError: boom" in captured.out
    assert "Traceback" in captured.err
    # A refusal says why, which is enough.
    _cli(monkeypatch, tmp_path, ValueError("slit_width must be 0.2"))
    captured = capsys.readouterr()
    assert "Error during simulation: ValueError: slit_width must be 0.2" in captured.out
    assert "Traceback" not in captured.err


def test_with_no_configuration_the_hint_is_the_documentation(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["eclipse"])
    cli.main()
    out = capsys.readouterr().out
    assert "solarc-eclipse.readthedocs.io" in out and "run/input" not in out


def test_debug_opens_in_the_frame_of_the_function_that_failed(monkeypatch):
    seen = {}

    def record(message, locals_dict=None, globals_dict=None, traceback=None):
        seen.update(locals_dict)

    monkeypatch.setattr(utils, "DEBUG_MODE", True)
    monkeypatch.setattr(utils, "debug_break", record)

    @utils.debug_on_error
    def run():
        config = {"instrument": "SWC"}
        raise ValueError(f"cannot run {config}")

    assert run.__name__ == "run"
    with pytest.raises(ValueError):
        run()
    assert seen["config"] == {"instrument": "SWC"}
    assert isinstance(seen["exception"], ValueError)


def test_debug_opens_in_the_frame_of_eclipses_own_function_that_raised(monkeypatch):
    """Raised in a function the decorated one called, the session opened with the caller's locals."""
    from euvst_response.config import check_pinhole_lists

    seen = {}

    def record(message, locals_dict=None, globals_dict=None, traceback=None):
        seen.update(locals_dict)

    monkeypatch.setattr(utils, "DEBUG_MODE", True)
    monkeypatch.setattr(utils, "debug_break", record)

    @utils.debug_on_error
    def run():
        config = {"instrument": "SWC"}
        check_pinhole_lists([5 * u.um], [], [])

    with pytest.raises(ValueError):
        run()
    assert "config" not in seen and seen["sizes"] == [5 * u.um]


def test_the_git_commit_is_only_eclipses_own(monkeypatch, tmp_path):
    """An environment inside some other repository is not a checkout of ECLIPSE."""
    project = tmp_path / "project"
    package = project / ".venv" / "lib" / "site-packages" / "euvst_response"
    package.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    import importlib.resources
    monkeypatch.setattr(importlib.resources, "files", lambda name: package)
    assert utils.get_git_commit_id() == "unknown (not installed from a git checkout of ECLIPSE)"


def _atmosphere(tmp_path, name="box.h5", **overrides):
    shape = (2, 3, 4)
    fields = dict(temperature=np.full(shape, 1e6) * u.K,
                  electron_density=np.full(shape, 1e9) / u.cm**3,
                  x_edges=np.arange(5) * u.Mm, y_edges=np.arange(4) * u.Mm,
                  z_edges=np.arange(3) * u.Mm, time=10 * u.s)
    fields.update(overrides)
    return write_atmosphere(Atmosphere(**fields), tmp_path / name)


@pytest.mark.parametrize("change, message", [
    (lambda f: f["x_edges"].__setitem__(slice(None), np.arange(5)[::-1]), "x_edges must increase"),
    (lambda f: f.__delitem__("electron_density"), "needs a mass_density or an electron_density"),
    (lambda f: (f.__delitem__("time"), f.create_dataset("time", data=[10.0]),
                f["time"].attrs.__setitem__("unit", "s")), "time must have 0 dimensions"),
    (lambda f: f["temperature"].attrs.__delitem__("unit"), "'temperature' in .* has no 'unit'"),
    (lambda f: f["temperature"].attrs.__setitem__("unit", "m"),
     "must be in a unit convertible to K, got m"),
])
def test_info_refuses_a_file_that_cannot_be_read_rather_than_describing_it(tmp_path, change, message):
    path = _atmosphere(tmp_path)
    with h5py.File(path, "r+") as f:
        change(f)
    with pytest.raises(ValueError, match=message):
        read_atmosphere(path)
    with pytest.raises(ValueError, match=message):
        describe_atmosphere_file(path)


def test_text_attributes_written_as_one_element_arrays_are_read(tmp_path):
    """IDL writes a string attribute as an array of one string."""
    path = _atmosphere(tmp_path)
    with h5py.File(path, "r+") as f:
        f.attrs["format"] = np.array([b"eclipse-atmosphere"])
        f.attrs["source"] = np.array([b"IDL box"])
        f["temperature"].attrs["unit"] = np.array([b"K"])
    atmosphere = read_atmosphere(path)
    assert atmosphere.source == "IDL box" and atmosphere.temperature.unit == u.K
    assert "Source: IDL box" in describe_atmosphere_file(path)


def test_the_old_name_of_the_mass_per_electron_set_by_a_script_is_used():
    from euvst_response.synthesis import resolve_mass_per_electron
    args = argparse.Namespace(mass_per_electron=None, mean_mol_wt=1.29,
                              abundance="sun_coronal_2021_chianti")
    with pytest.warns(FutureWarning, match="mean_mol_wt is the old name"):
        assert resolve_mass_per_electron(args)[0] == 1.29


def test_the_science_cases_refuse_a_time_series_in_the_base_settings():
    from euvst_response.science_cases import science_case_configs
    for key in ("atmosphere_series", "synthesis_series", "raster", "synthesis"):
        with pytest.raises(ValueError, match=f"cannot set '{key}'.*not a time series"):
            science_case_configs(base={key: "x"})


def test_each_science_case_configuration_has_its_own_lists():
    from euvst_response.science_cases import DEFAULT_SETTINGS, science_case_configs
    configs, _ = science_case_configs()
    first, second = configs[0].config, configs[1].config
    first["offchip_bin_slit"].append(99)
    first["simulation"]["psf"] = False
    assert second["offchip_bin_slit"] == [1, 2] and second["simulation"]["psf"] is True
    assert DEFAULT_SETTINGS["offchip_bin_slit"] == [1, 2]
    assert DEFAULT_SETTINGS["simulation"] == {"psf": True}


def test_dynamic_mode_refuses_a_view_along_x(monkeypatch):
    from euvst_response import synthesis
    monkeypatch.setattr(sys, "argv", ["synthesise-spectra", "--slit-rest-time", "10 s",
                                      "--slit-width", "0.4 arcsec", "--integration-axis", "x",
                                      "--lines", "Fe12_195.1190"])
    with pytest.raises(ValueError, match="lays its snapshots across x"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            synthesis.main()


def test_a_crop_one_cell_wide_keeps_the_axis():
    from euvst_response.synthesis import apply_cube_cropping, create_atmosphere_ndcube
    cubes = [create_atmosphere_ndcube(np.ones((4, 5, 6)) * unit, 1 * u.Mm, 1 * u.Mm, 1 * u.Mm)
             for unit in (u.K, u.g / u.cm**3, u.cm / u.s)]
    cropped = apply_cube_cropping(*cubes, ["0.1 Mm", "0.2 Mm"], None, None)
    assert all(cube.data.ndim == 3 and cube.data.shape[2] == 1 for cube in cropped)


def test_an_exposure_a_rounding_error_past_the_series_is_inside_it_and_one_further_is_not():
    series = object.__new__(AtmosphereSeries)
    series.times = np.array([0.0, 0.1234, 0.2468, 0.3702]) * u.s
    end = series.valid_until()[-1].to_value(u.s)
    assert series.coverage(0.3702 * u.s, end * (1 + 4e-16) * u.s) == [(3, 1.0)]
    with pytest.raises(ValueError, match="outside the series, starting 0.0001 s before it"):
        series.coverage(-1e-4 * u.s, 0.1 * u.s)


def test_a_series_of_times_far_from_zero_allows_rounding_and_no_more():
    """Scaled by the times themselves, the allowance was a second for a series at 1e9 s."""
    series = object.__new__(AtmosphereSeries)
    series.times = (1e9 + np.array([0.0, 0.5, 1.0])) * u.s
    end = series.valid_until()[-1].to_value(u.s)
    assert series.coverage((1e9 + 1.0) * u.s, np.nextafter(end, np.inf) * u.s) == [(2, 1.0)]
    with pytest.raises(ValueError, match="outside the series, ending 0.25 s after it"):
        series.coverage((1e9 + 1.0) * u.s, (end + 0.25) * u.s)


def test_an_exposure_that_starts_where_the_series_ends_is_refused():
    """Allowed as rounding and cut to the series, it had no length, and its fractions were 0 / 0."""
    series = object.__new__(AtmosphereSeries)
    series.times = np.array([0.0, 0.5, 1.0]) * u.s
    end = series.valid_until()[-1].to_value(u.s)
    with pytest.raises(ValueError, match="lies outside the series, which runs from 0 to 1.5 s"):
        series.coverage(end * u.s, np.nextafter(end, np.inf) * u.s)


def test_a_slit_a_rounding_error_past_the_edge_of_the_box_is_inside_it():
    from euvst_response.raster import RasterSynthesiser
    from euvst_response.utils import angle_to_distance

    class Series:
        x_edges = np.arange(13) * 0.1 * u.Mm

    raster = object.__new__(RasterSynthesiser)
    raster.series = Series()
    half = angle_to_distance(0.2 * u.arcsec).to(u.Mm) / 2
    first, last, fractions = raster.columns_under(1.2 * u.Mm - half + 2e-13 * u.Mm, 0.2 * u.arcsec)
    assert last == 12 and fractions.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="reaches outside the atmosphere, 0.001 Mm beyond it"):
        raster.columns_under(1.2 * u.Mm - half + 1e-3 * u.Mm, 0.2 * u.arcsec)


def test_a_synthesis_files_text_attributes_written_as_one_element_arrays_are_read(tmp_path):
    """As IDL writes them, for a synthesis another code wrote."""
    from euvst_response.synthesis_file import (FORMAT_NAME, SpectralLine, Synthesis, read_synthesis,
                                               read_synthesis_layout, write_synthesis)

    edges = np.arange(3) * 0.1 * u.Mm
    wavelength = 195.119 * u.AA + np.arange(-30, 31) * 0.003 * u.AA
    line = SpectralLine(intensity=np.ones((2, 2, 61)) * 1e13 * u.erg / (u.s * u.cm**2 * u.sr * u.cm),
                        wavelength=wavelength, rest_wavelength=195.119 * u.AA)
    path = write_synthesis(Synthesis(lines={"Fe12_195.1190": line}, x_edges=edges, y_edges=edges,
                                     integration_axis="z"), tmp_path / "file.h5")
    with h5py.File(path, "r+") as f:
        f.attrs["format"] = np.array([FORMAT_NAME.encode()])
        f.attrs["source"] = np.array([b"IDL run"])
        f.attrs["integration_axis"] = np.array([b"z"])
    synthesis = read_synthesis(path)
    assert synthesis.source == "IDL run" and synthesis.integration_axis == "z"
    assert read_synthesis_layout(path)["integration_axis"] == "z"
