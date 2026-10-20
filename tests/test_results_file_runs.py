"""Results files written by full instrument runs read back unchanged."""
import dataclasses
import gc
import importlib
import os
import sys
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
import yaml
from astropy.wcs import WCS
from ndcube import NDCube

from eclipse.analysis import load_instrument_response_results, summary_table
from eclipse.results_file import (_holds_array, convert_results_pickle, is_results_file,
                                  load_results)
from eclipse.synthesis_file import RADIANCE_UNIT, SpectralLine, Synthesis, write_synthesis

LINE = "Fe12_195.1190"
REST = 195.119 * u.Angstrom
NY, NX, N_WAVE = 6, 8, 121
PIXEL = 0.1875 * u.Mm
STEP = (5 * u.km / u.s / const.c * REST).to(u.AA)


def _synthesis_file(path, time=None, scale=1.0):
    """Write a synthesis file of a Gaussian line, Doppler shifted differently in each pixel."""
    offsets = (np.arange(N_WAVE) - N_WAVE // 2) * STEP
    shifts = np.linspace(-30, 30, NY * NX).reshape(NY, NX, 1) * u.km / u.s
    centres = (shifts / const.c * REST).to(u.AA)
    sigma = (20 * u.km / u.s / const.c * REST).to(u.AA)
    profile = np.exp(-0.5 * ((offsets - centres) / sigma).decompose().value ** 2)
    line = SpectralLine(intensity=scale * 1e13 * profile * RADIANCE_UNIT,
                        wavelength=REST + offsets, rest_wavelength=REST, atom=26, ion=12)
    edges = lambda n: (np.arange(n + 1) - n / 2) * PIXEL  # noqa: E731
    return write_synthesis(Synthesis(lines={LINE: line}, x_edges=edges(NX), y_edges=edges(NY),
                                     source="a test", time=time), path)


def _run(tmp_path, monkeypatch, name, **config):
    """Run the instrument simulation and return the saved payload and its path."""
    main_module = importlib.import_module("eclipse.main")

    saved = {}

    def keep(path, payload, **kwargs):
        saved["payload"] = payload
        saved["path"] = save(path, payload, **kwargs)
        return saved["path"]

    save = main_module.save_results
    monkeypatch.setattr(main_module, "save_results", keep)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"instrument": "SWC", "n_iter": 2, "ncpu": 1, **config}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(path)])
    main_module.main()
    return saved["payload"], saved["path"]


def _same_wcs(got, expected, where):
    # Compare copies: astropy converts a WCS to SI units in place the first
    # time it is used, and the payload shares WCSs between cubes.
    got, expected = got.deepcopy().wcs, expected.deepcopy().wcs
    assert list(got.ctype) == list(expected.ctype), where
    assert [str(c) for c in got.cunit] == [str(c) for c in expected.cunit], where
    for name in ("crval", "cdelt", "crpix"):
        assert np.array_equal(getattr(got, name), getattr(expected, name)), f"{where}: {name}"
    assert np.array_equal(got.get_pc(), expected.get_pc()), where


def _same(got, expected, where="payload", _memo=None):
    """
    Assert that `got` equals `expected`, entry by entry.

    Sharing is checked too, for whatever the file writes as a dataset or
    group: one object in the results, such as the ground truth the
    combinations share, must come back as one object, and distinct objects
    must stay distinct.
    """
    memo = {} if _memo is None else _memo
    if _holds_array(expected):
        pair = memo.setdefault(("expected", id(expected)), got)
        back = memo.setdefault(("got", id(got)), expected)
        assert pair is got and back is expected, f"{where}: shared as it was not"
    if isinstance(expected, NDCube):
        assert isinstance(got, NDCube), where
        assert got.unit == expected.unit, where
        assert got.data.dtype == np.asarray(expected.data).dtype, where
        assert np.array_equal(got.data, expected.data, equal_nan=True), where
        _same_wcs(got.wcs, expected.wcs, f"{where}.wcs")
        _same(got.meta, expected.meta if expected.meta is not None else {},
              f"{where}.meta", memo)
    elif isinstance(expected, WCS):
        _same_wcs(got, expected, where)
    elif dataclasses.is_dataclass(expected) and not isinstance(expected, type):
        assert type(got) is type(expected), where
        for field in dataclasses.fields(expected):
            _same(getattr(got, field.name), getattr(expected, field.name),
                  f"{where}.{field.name}", {})
    elif isinstance(expected, u.Quantity):
        assert isinstance(got, u.Quantity) and got.unit == expected.unit, where
        assert got.dtype == expected.dtype and got.shape == expected.shape, where
        assert np.array_equal(got.value, expected.value, equal_nan=True), where
    elif isinstance(expected, np.ndarray):
        assert isinstance(got, np.ndarray) and got.dtype == expected.dtype, where
        assert np.array_equal(got, expected, equal_nan=expected.dtype.kind == "f"), where
    elif isinstance(expected, dict):
        assert isinstance(got, dict) and list(got) == list(expected), where
        # Check the keys too: a NumPy number is not a Python number.
        for got_key, key in zip(got, expected):
            _same(got_key, key, f"{where} key {key!r}", {})
        for key in expected:
            _same(got[key], expected[key], f"{where}[{key!r}]", memo)
    elif isinstance(expected, (list, tuple)):
        assert type(got) is type(expected) and len(got) == len(expected), where
        for index, (a, b) in enumerate(zip(got, expected)):
            _same(a, b, f"{where}[{index}]", memo)
    elif isinstance(expected, float) and np.isnan(expected):
        assert type(got) is type(expected) and np.isnan(got), where
    elif isinstance(expected, type):
        assert got is expected, where
    elif isinstance(expected, u.UnitBase):
        assert got == expected, f"{where}: {got!r} != {expected!r}"
    else:
        # Check the type too, to catch a flag coming back as 1, or a NumPy
        # number as a Python one.
        assert type(got) is type(expected) and got == expected, f"{where}: {got!r} != {expected!r}"


def test_a_uniform_intensity_run_reads_back_what_it_saved(tmp_path, monkeypatch):
    payload, path = _run(tmp_path, monkeypatch, "uniform",
                         uniform_intensity="5000 erg / (s cm2 sr)",
                         simulation={"slit_width": "0.2 arcsec", "expos": ["5 s", "10 s"]})
    assert path == Path("run/result/uniform.h5") and is_results_file(path)
    assert not Path("run/result/uniform.pkl").exists()
    _same(load_results(path), payload)

    # A script asking for the old results name reads the new file.
    with pytest.warns(FutureWarning, match="uniform.h5, which the instrument simulation now"):
        results = load_instrument_response_results("run/result/uniform.pkl")
    assert results["instrument"] == "SWC"

    # An existing pickle is the one read, and the warning says when the .h5
    # beside it is newer (the results of a later run).
    import dill

    old = Path("run/result/uniform.pkl")
    with open(old, "wb") as f:
        dill.dump({"instrument": "EIS"}, f)
    written = Path("run/result/uniform.h5").stat().st_mtime
    for offset, newer in ((-100, True), (100, False)):
        os.utime(old, (written + offset, written + offset))
        with pytest.warns(FutureWarning, match="results pickle") as seen:
            assert load_results(old)["instrument"] == "EIS"
        assert any("beside it, is newer" in str(w.message) for w in seen) is newer
    with pytest.raises(FileNotFoundError):
        load_results("run/result/elsewhere.pkl")


def test_a_synthesis_file_run_reads_back_what_it_saved(tmp_path, monkeypatch):
    _synthesis_file(tmp_path / "synthesis.h5")
    payload, path = _run(tmp_path, monkeypatch, "file", synthesis_file=str(tmp_path / "synthesis.h5"),
                         simulation={"slit_width": ["0.2 arcsec", "0.4 arcsec"], "expos": "5 s"})
    loaded = load_results(path)
    _same(loaded, payload)

    # The analysis functions read it as they read a pickle.
    results = load_instrument_response_results(path)
    combination = next(iter(results["results"]["all_combinations"].values()))
    signal = combination["first_dn_signal"]
    assert signal.unit.is_equivalent(u.DN / u.pix) and np.all(np.isfinite(signal.data))
    wavelength = signal.axis_world_coords(-1)[0].to_value(u.AA)
    assert wavelength.min() < REST.value < wavelength.max()
    assert results["cube_sim"].meta["atom"] == 26
    summary_table(results)


def test_a_synthesis_series_run_reads_back_what_it_saved(tmp_path, monkeypatch):
    for index, time in enumerate((0.0, 10.0)):
        _synthesis_file(tmp_path / "series" / f"snap_{index}.h5", time * u.s, 1.0 + index)
    payload, path = _run(tmp_path, monkeypatch, "series",
                         synthesis_series=str(tmp_path / "series" / "*.h5"),
                         raster={"start": "0 s", "steps": 2},
                         simulation={"slit_width": "0.4 arcsec", "expos": "5 s", "psf": False})
    assert payload["raster"]["plan"].steps == 2
    _same(load_results(path), payload)


def test_an_atmosphere_series_run_reads_back_what_it_saved(tmp_path, monkeypatch):
    """Including the synthesis settings, whose precision is a NumPy type."""
    from eclipse import raster as raster_module
    from eclipse.atmosphere import Atmosphere, write_atmosphere
    from eclipse.utils import angle_to_distance

    def flat_goft(lines, **kwargs):
        logT, logN = np.linspace(5.0, 7.0, 21), np.linspace(8.0, 10.0, 21)
        return ({name: {"wl0": REST.to(u.cm), "g_tn": np.full((21, 21), 1e-24), "atom": 26,
                        "ion": 12, "hdf5_dbase_root": None} for name in lines}, logT, logN)

    monkeypatch.setattr(raster_module, "compute_goft_fiasco", flat_goft)
    cell = angle_to_distance(0.2 * u.arcsec).to(u.Mm)
    nz, ny, nx = 4, 6, 12
    for index, time in enumerate((0.0, 10.0)):
        write_atmosphere(Atmosphere(
            temperature=np.full((nz, ny, nx), 1e6) * u.K,
            electron_density=np.full((nz, ny, nx), 1e9) / u.cm**3,
            velocity_z=np.zeros((nz, ny, nx)) * u.km / u.s, time=time * u.s,
            x_edges=(np.arange(nx + 1) - nx / 2) * cell, y_edges=(np.arange(ny + 1) - ny / 2) * cell,
            z_edges=np.arange(nz + 1) * 0.1 * u.Mm), tmp_path / "series" / f"snap_{index}.h5")
    payload, path = _run(tmp_path, monkeypatch, "atmospheres",
                         atmosphere_series=str(tmp_path / "series" / "*.h5"), reference_line=LINE,
                         fit_signals="dn", synthesis={"lines": [LINE]},
                         raster={"start": "0 s", "steps": 2},
                         simulation={"slit_width": "0.4 arcsec", "expos": "5 s", "psf": False})
    assert payload["raster"]["settings"].precision is np.float64
    _same(load_results(path), payload)


def test_an_eis_run_reads_back_what_it_saved(tmp_path, monkeypatch):
    """Including the calibration date, which YAML reads as a datetime.date when unquoted."""
    import datetime

    payload, path = _run(tmp_path, monkeypatch, "eis", instrument="EIS",
                         uniform_intensity="5000 erg / (s cm2 sr)",
                         telescope={"calibration": "dz2025", "date": datetime.date(2012, 6, 3)},
                         simulation={"slit_width": "2 arcsec", "expos": "5 s"})
    assert payload["config"]["telescope"]["date"] == datetime.date(2012, 6, 3)
    _same(load_results(path), payload)


def test_a_results_pickle_still_loads_and_converts(tmp_path, monkeypatch):
    """Pickled results from older versions load with a warning and convert to a results file."""
    import dill

    payload, path = _run(tmp_path, monkeypatch, "uniform",
                         uniform_intensity="5000 erg / (s cm2 sr)",
                         simulation={"slit_width": "0.2 arcsec", "expos": "5 s"})
    old = tmp_path / "old.pkl"
    with open(old, "wb") as f:
        dill.dump(payload, f)
    with pytest.warns(FutureWarning, match="results pickle"):
        results = load_instrument_response_results(old)
    assert len(results["results"]["all_combinations"]) == 1

    converted = convert_results_pickle(old)
    assert converted == tmp_path / "old.h5" and is_results_file(converted)
    # Compare with what the pickle holds: astropy pickles a WCS as a header, to 14 digits.
    with open(old, "rb") as f:
        _same(load_results(converted), dill.load(f))
    with pytest.raises(ValueError, match="already an HDF5 file"):
        convert_results_pickle(converted)
    # An existing file and the pickle itself are left alone.
    with pytest.raises(FileExistsError, match="overwrite=True"):
        convert_results_pickle(old)
    assert convert_results_pickle(old, overwrite=True) == converted
    # A conversion that cannot be read back leaves the existing file unchanged.
    from eclipse import results_file

    before, real = converted.read_bytes(), results_file.load_results

    def unreadable(path, _stacklevel=2):
        raise ValueError("unreadable")

    monkeypatch.setattr(results_file, "load_results", unreadable)
    with pytest.raises(ValueError, match="cannot be read back"):
        convert_results_pickle(old, overwrite=True)
    monkeypatch.setattr(results_file, "load_results", real)
    assert converted.read_bytes() == before
    assert [p.name for p in tmp_path.glob("old.h5*")] == ["old.h5"]
    misnamed = tmp_path / "misnamed.h5"
    misnamed.write_bytes(old.read_bytes())
    with pytest.raises(ValueError, match="the pickle itself"):
        convert_results_pickle(misnamed)
    # A pickle of something else, such as a synthesis, is refused.
    with open(tmp_path / "synthesis.pkl", "wb") as f:
        dill.dump({"line_cubes": {}}, f)
    with pytest.raises(ValueError, match="convert_synthesis_pickle"):
        convert_results_pickle(tmp_path / "synthesis.pkl")
    assert not (tmp_path / "synthesis.h5").exists()
    # So is a file that is not a pickle at all.
    (tmp_path / "empty.pkl").write_bytes(b"")
    with pytest.raises(ValueError, match="is not a pickle"):
        convert_results_pickle(tmp_path / "empty.pkl")
    # A directory cannot be the output, and a symlink is written through.
    with pytest.raises(IsADirectoryError):
        convert_results_pickle(old, tmp_path)
    real, link = tmp_path / "elsewhere" / "real.h5", tmp_path / "link.h5"
    real.parent.mkdir()
    real.write_bytes(b"")
    link.symlink_to(real)
    assert convert_results_pickle(old, link, overwrite=True) == link
    assert link.is_symlink() and load_results(real)["instrument"] == "SWC"
    # `~` expands to the user's home directory.
    monkeypatch.setenv("HOME", str(tmp_path))
    assert convert_results_pickle("~/old.pkl", "~/home.h5") == tmp_path / "home.h5"

    # A configuration object missing a later setting, as in a 0.8.0 pickle,
    # gets the current default for it.
    from eclipse.config import Simulation
    simulation = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec)
    del simulation.__dict__["pinhole_positions_spectral"]
    with open(tmp_path / "v080.pkl", "wb") as f:
        dill.dump({"results": {"all_combinations": {}}, "simulation": simulation}, f)
    with pytest.warns(UserWarning, match="has no pinhole_positions_spectral"):
        converted = convert_results_pickle(tmp_path / "v080.pkl")
    restored = load_results(converted)["simulation"]
    assert restored.pinhole_positions_spectral == [] and restored.slit_width == 0.4 * u.arcsec

    # A setting from another version that this version lacks is dropped with
    # a warning, not silently.
    simulation = Simulation(instrument="SWC", slit_width=0.4 * u.arcsec)
    simulation.__dict__["allow_nonflight_slit"] = True
    with open(tmp_path / "branch.pkl", "wb") as f:
        dill.dump({"results": {"all_combinations": {}}, "simulation": simulation}, f)
    with pytest.warns(UserWarning, match="has allow_nonflight_slit, which this version of "
                                         "ECLIPSE does not have, so it is left out"):
        convert_results_pickle(tmp_path / "branch.pkl")


def test_a_rerun_moves_the_pickle_of_its_name_aside(tmp_path, monkeypatch, capsys):
    """So a script naming the pickle reads the new results, as when reruns overwrote it."""
    import dill

    stale = tmp_path / "run" / "result" / "uniform.pkl"
    stale.parent.mkdir(parents=True)
    with open(stale, "wb") as f:
        dill.dump({"instrument": "EIS"}, f)
    config = {"uniform_intensity": "5000 erg / (s cm2 sr)",
              "simulation": {"slit_width": "0.2 arcsec", "expos": "5 s"}}

    # The pickle is moved only after the results are saved, so a failed save leaves it.
    main_module = importlib.import_module("eclipse.main")
    real = main_module.save_results

    def fails(path, payload, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(main_module, "save_results", fails)
    with pytest.raises(OSError, match="no space left"):
        _run(tmp_path, monkeypatch, "uniform", **config)
    monkeypatch.setattr(main_module, "save_results", real)
    assert stale.is_file()

    _run(tmp_path, monkeypatch, "uniform", **config)
    moved = stale.parent / "uniform.pkl.old"
    assert not stale.exists() and moved.is_file()
    with pytest.warns(FutureWarning, match="does not exist"):
        assert load_instrument_response_results(stale)["instrument"] == "SWC"
    # The moved pickle still loads as a pickle.
    with pytest.warns(FutureWarning, match="results pickle"):
        assert load_results(moved)["instrument"] == "EIS"

    # A second old pickle is moved beside the first, not over it.
    with open(stale, "wb") as f:
        dill.dump({"instrument": "EIS", "second": True}, f)
    _run(tmp_path, monkeypatch, "uniform", **config)
    with pytest.warns(FutureWarning, match="results pickle"):
        assert "second" not in load_results(moved)
    with pytest.warns(FutureWarning, match="results pickle"):
        assert load_results(stale.parent / "uniform.pkl.old.1")["second"]

    # A pickle that cannot be moved is left with a message, and the run still succeeds.
    with open(stale, "wb") as f:
        dill.dump({"instrument": "EIS"}, f)
    real_replace = os.replace

    def refuse_pickles(source, target):
        if str(source).endswith(".pkl"):
            raise PermissionError("Operation not permitted")
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", refuse_pickles)
    capsys.readouterr()
    _run(tmp_path, monkeypatch, "uniform", **config)
    assert stale.is_file() and "Could not move" in capsys.readouterr().out
    assert load_results(stale.with_suffix(".h5"))["instrument"] == "SWC"


def test_a_config_the_results_file_cannot_hold_is_refused_before_the_run(tmp_path, monkeypatch):
    """For example a YAML alias inside itself, which would otherwise fail only at the save."""
    loop = ["x"]
    loop.append(loop)
    main_module = importlib.import_module("eclipse.main")
    monkeypatch.setattr(main_module, "monte_carlo", lambda *args, **kwargs: pytest.fail("ran"))
    # The loop is found by recursing until Python runs out of stack. Garbage
    # from earlier tests collected there cannot run its finalisers, which
    # pytest reports as unraisable, so collect it first.
    gc.collect()
    gc.disable()
    try:
        with pytest.raises(ValueError,
                           match="cannot be saved with the results: a value holds itself"):
            _run(tmp_path, monkeypatch, "loop", uniform_intensity="5000 erg / (s cm2 sr)",
                 reference_line=loop)
    finally:
        gc.enable()
