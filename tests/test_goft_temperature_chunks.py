"""`--goft-temperature-chunk` changes only the memory fiasco needs.

Contribution functions from chunks of the temperature grid match those from
the whole grid. A stand-in for fiasco means CHIANTI is not needed.
"""
import sys
import types

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from eclipse import synthesis, utils
from eclipse.atmosphere import Atmosphere, write_atmosphere
from eclipse.synthesis import compute_goft_fiasco

LINES = ["Fe12_195.1190", "Fe12_195.1790"]


class _Transitions:
    wavelength = np.array([171.073, 195.119, 195.179, 180.0]) * u.AA
    is_bound_bound = np.array([True, True, True, False])
    is_observed = np.array([True, True, True, False])


class _Ion:
    """Enough of `fiasco.Ion` for `_compute_single_ion`, recording where it works out G."""

    temperatures_per_call = []
    # Sizes of the temperature grids the ratio of protons to electrons is worked out on.
    ratio_temperatures = []

    def __init__(self, name, temperature, abundance=None, hdf5_dbase_root=None):
        self.temperature = temperature
        self.atomic_number = 26
        self.hdf5_dbase_root = hdf5_dbase_root or "default.h5"
        self.transitions = _Transitions()

    ion_name = "Fe 12"

    @property
    def ionization_fraction(self):
        return np.full(self.temperature.size, 0.5)

    def contribution_function(self, density):
        _Ion.temperatures_per_call.append(self.temperature.size)
        logt = np.log10(self.temperature.to_value(u.K))[:, np.newaxis, np.newaxis]
        logn = np.log10(density.to_value(u.cm**-3))[np.newaxis, :, np.newaxis]
        transition = np.arange(1, 4)[np.newaxis, np.newaxis, :]
        return 1e-24 * np.exp(-(logt - 6.2) ** 2) * transition / logn * u.erg * u.cm**3 / u.s

    @property
    def proton_electron_ratio(self):
        # Through fiasco.ions, as fiasco does, so that the ratio can be shared.
        ions = sys.modules["fiasco.ions"]
        if hasattr(ions, "proton_electron_ratio"):
            return ions.proton_electron_ratio(self.temperature)
        return 0.8 + 0.01 * np.log10(self.temperature.to_value(u.K))


@pytest.fixture
def fake_fiasco(tmp_path, monkeypatch):
    # Import the real fiasco first, so it makes its own "fiasco" logger. Made
    # under the stand-in, it would be a plain logger that breaks a later real
    # import.
    import fiasco  # noqa: F401
    module = types.ModuleType("fiasco")
    module.Ion = _Ion
    (tmp_path / "chianti_dbase.h5").touch()
    module.defaults = {"hdf5_dbase_root": str(tmp_path / "chianti_dbase.h5")}
    # The database exists, so there is no prompt to build it.
    util = types.ModuleType("fiasco.util")
    util.check_database = lambda hdf5_dbase_root, **kwargs: None
    module.util = util
    # The ratio of protons to electrons, which proton_electron_ratio_once shares.
    ions = types.ModuleType("fiasco.ions")
    def ratio(temperature, **kwargs):
        _Ion.ratio_temperatures.append(temperature.size)
        return (0.8 + 0.01 * np.log10(temperature.to_value(u.K))) * u.dimensionless_unscaled

    ions.proton_electron_ratio = ratio
    module.ions = ions
    monkeypatch.setitem(sys.modules, "fiasco", module)
    monkeypatch.setitem(sys.modules, "fiasco.util", util)
    monkeypatch.setitem(sys.modules, "fiasco.ions", ions)
    # Ratios shared by earlier tests are not reused.
    monkeypatch.setattr(utils, "_PROTON_ELECTRON_RATIOS", {})
    _Ion.temperatures_per_call = []
    _Ion.ratio_temperatures = []
    return _Ion


@pytest.mark.parametrize("chunk", [1, 7, 101])
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


@pytest.mark.parametrize("n_workers, chunk", [(1, None), (1, 7), (6, None)])
def test_lines_are_worked_out_only_where_the_ion_exists(fake_fiasco, monkeypatch, n_workers,
                                                        chunk):
    """Where the ionisation fraction is zero the ion is absent, so G is zero and fiasco is
    not asked for it."""
    everywhere, logt, _ = compute_goft_fiasco(LINES, n_workers=1, temperature_chunk=chunk)

    def fraction(self):
        log_t = np.log10(self.temperature.to_value(u.K))
        return np.where((log_t > 5.5) & (log_t < 7.0), 0.5, 0.0)

    monkeypatch.setattr(fake_fiasco, "ionization_fraction", property(fraction))
    fake_fiasco.temperatures_per_call = []
    goft, _, _ = compute_goft_fiasco(LINES, n_workers=n_workers, temperature_chunk=chunk)
    present = (logt > 5.5) & (logt < 7.0)
    assert sum(fake_fiasco.temperatures_per_call) == present.sum()
    if n_workers > 1:
        # Threads share the temperatures where the ion exists, at least 8 each.
        assert len(fake_fiasco.temperatures_per_call) > 1
        assert min(fake_fiasco.temperatures_per_call) >= 8
    for line in LINES:
        assert np.array_equal(goft[line]["g_tn"][:, present], everywhere[line]["g_tn"][:, present])
        assert not goft[line]["g_tn"][:, ~present].any()


@pytest.mark.parametrize("n_workers, chunk", [(1, 7), (6, None)])
def test_where_the_rates_stop_and_dropped_levels_are_reported(fake_fiasco, monkeypatch, capsys,
                                                                 n_workers, chunk):
    """The single-ion model is used from the first temperature beyond the level-resolved
    rates, found across chunks or threads, and levels left out are counted."""
    from eclipse import band as band_module

    end = 10 ** 7.0 * (1 - 1e-9)

    def contribution(ion, density, dropped, **kwargs):
        temperature = ion.temperature.to_value(u.K)
        if kwargs.get("use_two_ion_model", True) and temperature.max() >= end:
            raise ValueError("A value in x_new is above the interpolation range.")
        dropped.update({k: 1 for k in range(temperature.size)})
        return ion.contribution_function(density)

    monkeypatch.setattr(band_module, "_contribution_function", contribution)
    goft, _, _ = compute_goft_fiasco(LINES, n_workers=n_workers, temperature_chunk=chunk)
    out = capsys.readouterr().out
    assert "from log T 7.00 its level populations are worked out with fiasco's single-ion" in out
    assert "Fe 12: levels nothing populates were left out at 101 temperatures" in out
    assert all(np.isfinite(goft[line]["g_tn"]).all() for line in LINES)


@pytest.mark.parametrize("n_workers, sizes", [(6, [17] * 5 + [16]), (100, [13] * 5 + [12] * 3)])
def test_an_ions_temperatures_shared_between_threads_give_what_one_thread_gives(
        fake_fiasco, n_workers, sizes):
    """Workers beyond one per ion are threads, at most 8, each with part of the temperatures."""
    whole, _, _ = compute_goft_fiasco(LINES, n_workers=1)
    fake_fiasco.temperatures_per_call = []
    utils._PROTON_ELECTRON_RATIOS.clear()
    fake_fiasco.ratio_temperatures = []
    shared, _, _ = compute_goft_fiasco(LINES, n_workers=n_workers)
    assert sorted(fake_fiasco.temperatures_per_call) == sorted(sizes)
    # The ratio of protons to electrons is worked out once, on the whole grid.
    assert fake_fiasco.ratio_temperatures == [101]
    for line in LINES:
        assert np.array_equal(shared[line]["g_tn"], whole[line]["g_tn"])

    # A chunk size limits the memory, so the ion keeps to one thread, as it does
    # without a shared ratio, which each thread would work out again.
    fake_fiasco.temperatures_per_call = []
    compute_goft_fiasco(LINES, n_workers=n_workers, temperature_chunk=7)
    assert max(fake_fiasco.temperatures_per_call) == 7
    assert sum(fake_fiasco.temperatures_per_call) == 101
    fake_fiasco.temperatures_per_call = []
    compute_goft_fiasco(LINES, n_workers=n_workers, temperature_chunk=500)
    assert sorted(fake_fiasco.temperatures_per_call) == sorted(sizes)


def test_an_ion_keeps_one_thread_without_a_shared_ratio(fake_fiasco, monkeypatch):
    monkeypatch.delattr(sys.modules["fiasco.ions"], "proton_electron_ratio")
    compute_goft_fiasco(LINES, n_workers=6)
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


def test_a_density_grid_carried_on_from_the_default_has_its_values_at_its_points(fake_fiasco):
    """A grid sized to an atmosphere matches the default at the points they share.

    Each density is computed on its own, so the grid range changes nothing.
    """
    from eclipse.synthesis import _density_point

    whole, _, logn = compute_goft_fiasco(LINES, n_workers=1)
    part, _, logn_part = compute_goft_fiasco(LINES, n_workers=1, logN_min=_density_point(4),
                                             logN_max=_density_point(9), nN=6)
    assert np.allclose(logn_part, logn[4:10], rtol=0, atol=1e-12)
    for line in LINES:
        assert np.allclose(part[line]["g_tn"], whole[line]["g_tn"][4:10], rtol=1e-12, atol=0)
