"""The instrument simulation writes its results as ASDF, and reads back what it wrote.

The unit tests in test_io_asdf.py check the encoder against objects built for
the purpose. These run the real thing end to end, which is the only way to
find out whether the tree ECLIPSE actually produces survives the trip: each
run keeps what it saved, and what comes back from the file is compared with
it entry by entry.
"""
import dataclasses
import importlib
import sys
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
import yaml
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.analysis import load_instrument_response_results, summary_table
from euvst_response.io import convert_results_pickle, is_asdf, load_results
from euvst_response.synthesis_file import RADIANCE_UNIT, SpectralLine, Synthesis, write_synthesis

LINE = "Fe12_195.1190"
REST = 195.119 * u.Angstrom
NY, NX, N_WAVE = 6, 8, 121
PIXEL = 0.1875 * u.Mm
STEP = (5 * u.km / u.s / const.c * REST).to(u.AA)


def _synthesis_file(path, time=None, scale=1.0):
    """A Gaussian line whose Doppler shift changes from pixel to pixel."""
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
    """Run the instrument simulation, and return what it saved and where."""
    main_module = importlib.import_module("euvst_response.main")

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
    # On copies: astropy puts a WCS into SI units in place the first time it
    # works anything out with it, and the payload shares WCSs between cubes.
    got, expected = got.deepcopy().wcs, expected.deepcopy().wcs
    assert list(got.ctype) == list(expected.ctype), where
    assert [str(c) for c in got.cunit] == [str(c) for c in expected.cunit], where
    for name in ("crval", "cdelt", "crpix"):
        assert np.allclose(getattr(got, name), getattr(expected, name), rtol=1e-12, atol=0), \
            f"{where}: {name}"
    assert np.allclose(got.get_pc(), expected.get_pc(), rtol=1e-12, atol=1e-15), where


def _same(got, expected, where="payload"):
    """That *got* is what *expected* was, entry by entry."""
    if isinstance(expected, NDCube):
        assert isinstance(got, NDCube), where
        assert got.unit == expected.unit, where
        assert got.data.dtype == np.asarray(expected.data).dtype, where
        assert np.array_equal(got.data, expected.data, equal_nan=True), where
        _same_wcs(got.wcs, expected.wcs, f"{where}.wcs")
        _same(dict(got.meta), dict(expected.meta or {}), f"{where}.meta")
    elif isinstance(expected, WCS):
        _same_wcs(got, expected, where)
    elif dataclasses.is_dataclass(expected) and not isinstance(expected, type):
        assert type(got) is type(expected), where
        for field in dataclasses.fields(expected):
            if field.init:
                _same(getattr(got, field.name), getattr(expected, field.name),
                      f"{where}.{field.name}")
    elif isinstance(expected, u.Quantity):
        assert isinstance(got, u.Quantity) and got.unit == expected.unit, where
        assert np.array_equal(got.value, expected.value, equal_nan=True), where
    elif isinstance(expected, np.ndarray):
        assert isinstance(got, np.ndarray) and got.dtype == expected.dtype, where
        assert np.array_equal(got, expected, equal_nan=expected.dtype.kind == "f"), where
    elif isinstance(expected, dict):
        assert isinstance(got, dict) and list(got) == list(expected), where
        for key in expected:
            _same(got[key], expected[key], f"{where}[{key!r}]")
    elif isinstance(expected, (list, tuple)):
        assert type(got) is type(expected) and len(got) == len(expected), where
        for index, (a, b) in enumerate(zip(got, expected)):
            _same(a, b, f"{where}[{index}]")
    elif isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(got), where
    elif isinstance(expected, type):
        assert got is expected, where
    else:
        assert got == expected, f"{where}: {got!r} != {expected!r}"


def test_a_uniform_intensity_run_reads_back_what_it_saved(tmp_path, monkeypatch):
    payload, path = _run(tmp_path, monkeypatch, "uniform",
                         uniform_intensity="5000 erg / (s cm2 sr)",
                         simulation={"slit_width": "0.2 arcsec", "expos": ["5 s", "10 s"]})
    assert path == Path("run/result/uniform.asdf") and is_asdf(path)
    assert not Path("run/result/uniform.pkl").exists()
    _same(load_results(path), payload)


def test_a_synthesis_file_run_reads_back_what_it_saved(tmp_path, monkeypatch):
    _synthesis_file(tmp_path / "synthesis.h5")
    payload, path = _run(tmp_path, monkeypatch, "file", synthesis_file=str(tmp_path / "synthesis.h5"),
                         simulation={"slit_width": ["0.2 arcsec", "0.4 arcsec"], "expos": "5 s"})
    loaded = load_results(path)
    _same(loaded, payload)

    # And the analysis reads it as it read a pickle.
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


def test_a_results_pickle_still_loads_and_converts(tmp_path, monkeypatch):
    """Results that older versions pickled read with a warning, and convert to ASDF."""
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
    assert converted == tmp_path / "old.asdf" and is_asdf(converted)
    _same(load_results(converted), payload)
    with pytest.raises(ValueError, match="already an ASDF file"):
        convert_results_pickle(converted)


def test_a_script_asking_for_the_old_results_name_reads_the_new_file(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, "uniform", uniform_intensity="5000 erg / (s cm2 sr)",
         simulation={"slit_width": "0.2 arcsec", "expos": "5 s"})
    with pytest.warns(FutureWarning, match="uniform.asdf, which the instrument simulation now"):
        results = load_instrument_response_results("run/result/uniform.pkl")
    assert results["instrument"] == "SWC"
    with pytest.raises(FileNotFoundError):
        load_results("run/result/elsewhere.pkl")
