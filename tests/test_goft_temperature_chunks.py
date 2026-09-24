"""--goft-temperature-chunk has to change the memory fiasco needs and nothing else.

fiasco solves the level populations for all the temperatures it is given at
once, so its memory grows with their number. Passing it the temperature grid a
chunk at a time lowers that peak, and the contribution functions it returns
have to be the same, value for value and in the same order, as from the whole
grid in one call.

A stand-in for fiasco, whose contribution function depends on temperature,
density and transition, lets these run without the CHIANTI database.
"""
import sys
import types

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from euvst_response import synthesis
from euvst_response.atmosphere import Atmosphere, write_atmosphere
from euvst_response.synthesis import compute_goft_fiasco

LINES = ["Fe12_195.1190", "Fe12_195.1790"]


class _Transitions:
    wavelength = np.array([171.073, 195.119, 195.179, 180.0]) * u.AA
    is_bound_bound = np.array([True, True, True, False])


class _Ion:
    """Enough of fiasco.Ion for _compute_single_ion, recording each call."""

    temperatures_per_call = []

    def __init__(self, name, temperature, abundance=None, hdf5_dbase_root=None):
        _Ion.temperatures_per_call.append(temperature.size)
        self.temperature = temperature
        self.atomic_number = 26
        self.hdf5_dbase_root = hdf5_dbase_root or "default.h5"
        self.transitions = _Transitions()

    def contribution_function(self, density):
        logt = np.log10(self.temperature.to_value(u.K))[:, np.newaxis, np.newaxis]
        logn = np.log10(density.to_value(u.cm**-3))[np.newaxis, :, np.newaxis]
        transition = np.arange(1, 4)[np.newaxis, np.newaxis, :]
        return 1e-24 * np.exp(-(logt - 6.2) ** 2) * transition / logn * u.erg * u.cm**3 / u.s

    @property
    def proton_electron_ratio(self):
        return 0.8 + 0.01 * np.log10(self.temperature.to_value(u.K))


@pytest.fixture
def fake_fiasco(monkeypatch):
    module = types.ModuleType("fiasco")
    module.Ion = _Ion
    monkeypatch.setitem(sys.modules, "fiasco", module)
    _Ion.temperatures_per_call = []
    return _Ion


@pytest.mark.parametrize("chunk", [1, 7, 100, 101, 500])
def test_chunks_give_what_the_whole_grid_gives(fake_fiasco, chunk):
    """Every value, in order, for chunks that do and do not divide the grid."""
    whole, logt, logn = compute_goft_fiasco(LINES, n_workers=1)
    chunked, logt_c, logn_c = compute_goft_fiasco(LINES, n_workers=1,
                                                  temperature_chunk=chunk)

    assert np.array_equal(logt, logt_c) and np.array_equal(logn, logn_c)
    for line in LINES:
        assert chunked[line]["g_tn"].shape == (logn.size, logt.size)
        assert np.array_equal(chunked[line]["g_tn"], whole[line]["g_tn"])
    # The two lines are different transitions, so a mix-up would show.
    assert not np.array_equal(whole[LINES[0]]["g_tn"], whole[LINES[1]]["g_tn"])


def test_fiasco_is_given_no_more_than_a_chunk(fake_fiasco):
    compute_goft_fiasco(LINES, n_workers=1, temperature_chunk=7)
    assert max(fake_fiasco.temperatures_per_call) == 7
    assert sum(fake_fiasco.temperatures_per_call) == 101

    fake_fiasco.temperatures_per_call = []
    compute_goft_fiasco(LINES, n_workers=1)
    assert fake_fiasco.temperatures_per_call == [101]


@pytest.mark.parametrize("chunk", [0, -3])
def test_a_chunk_of_no_temperatures_is_refused(fake_fiasco, chunk):
    with pytest.raises(ValueError, match="temperature_chunk"):
        compute_goft_fiasco(LINES, n_workers=1, temperature_chunk=chunk)


def _write_atmosphere(path, shape):
    """A uniform coronal box as an atmosphere file."""
    edges = {f"{axis}_edges": np.arange(n + 1) * 0.1 * u.Mm
             for axis, n in zip(("z", "y", "x"), shape)}
    density = (1e9 / u.cm**3 * 1.29 * const.u).to(u.g / u.cm**3)
    return write_atmosphere(Atmosphere(
        temperature=np.full(shape, 1e6) * u.K,
        mass_density=np.full(shape, density.value) * density.unit,
        velocity_z=np.zeros(shape) * u.cm / u.s, **edges), path)


@pytest.mark.parametrize("extra, expected", [((), None),
                                             (("--goft-temperature-chunk", "10"), 10)])
def test_the_command_line_option_reaches_the_calculation(tmp_path, monkeypatch,
                                                         extra, expected):
    atmosphere = _write_atmosphere(tmp_path / "box.h5", (4, 4, 4))

    received = {}

    def _recording_goft(lines, **kwargs):
        received.update(kwargs)
        logt = np.linspace(5.0, 7.0, 21)
        logn = np.linspace(8.0, 10.0, 21)
        goft = {LINES[0]: {"wl0": (195.119 * u.AA).to(u.cm),
                           "g_tn": np.ones((logn.size, logt.size)),
                           "atom": 26, "ion": 12, "hdf5_dbase_root": None}}
        return goft, logt, logn

    monkeypatch.setattr(sys, "argv", [
        "synthesise-spectra", "--atmosphere", str(atmosphere),
        "--output-dir", str(tmp_path / "out"), "--lines", LINES[0],
        "--mass-per-electron", "1.29", *extra,
    ])
    monkeypatch.setattr(synthesis, "compute_goft_fiasco", _recording_goft)
    synthesis.main()
    assert received["temperature_chunk"] == expected
