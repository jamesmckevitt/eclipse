"""fiasco's proton to electron ratio, worked out once per process within proton_electron_ratio_once."""
import sys
import types

import astropy.units as u
import numpy as np
import pytest

from eclipse.utils import proton_electron_ratio_once

GRID = 10.0 ** np.linspace(4.0, 9.0, 101)


def _fake_ions(monkeypatch):
    """A stand-in fiasco.ions whose ratio, interpolated point by point like fiasco's, counts its calls."""
    calls = []

    def proton_electron_ratio(temperature, **kwargs):
        calls.append(np.size(temperature))
        logt = np.log10(u.Quantity(temperature).to_value(u.K))
        return u.Quantity(np.interp(logt, [4.0, 6.0, 9.0], [0.7, 0.83, 0.85]))

    ions = types.ModuleType("fiasco.ions")
    ions.proton_electron_ratio = proton_electron_ratio
    monkeypatch.setitem(sys.modules, "fiasco.ions", ions)
    monkeypatch.setattr(sys.modules["eclipse.utils"], "_PROTON_ELECTRON_RATIOS", {})
    return ions, proton_electron_ratio, calls


def test_ions_on_the_grid_get_fiascos_values_from_one_evaluation(monkeypatch):
    ions, original, calls = _fake_ions(monkeypatch)
    picks = [slice(20, 40), np.array([3, 50, 99]), slice(60, 61), slice(None)]
    expected = [original(GRID[pick] * u.K, abundance="a") for pick in picks]
    calls.clear()
    with proton_electron_ratio_once(GRID):
        got = [ions.proton_electron_ratio(GRID[pick] * u.K, abundance="a") for pick in picks]
    for want, have in zip(expected, got):
        assert np.array_equal(have.value, want.value) and have.unit == want.unit
    assert calls == [GRID.size]
    assert ions.proton_electron_ratio is original


def test_other_temperatures_and_settings_are_worked_out_by_fiasco(monkeypatch):
    ions, original, calls = _fake_ions(monkeypatch)
    off_grid = np.array([1.5e5, GRID[10]]) * u.K
    with proton_electron_ratio_once(GRID):
        ions.proton_electron_ratio(off_grid, abundance="a")
        ions.proton_electron_ratio(GRID[:5] * u.K, abundance="a")
        ions.proton_electron_ratio(GRID[:5] * u.K, abundance="b")
    assert calls == [2, GRID.size, GRID.size]


def test_without_fiasco_the_block_runs_unchanged(monkeypatch):
    monkeypatch.setitem(sys.modules, "fiasco.ions", None)
    ran = []
    with proton_electron_ratio_once(GRID):
        ran.append(True)
    assert ran == [True]


def test_a_decreasing_grid_is_used_as_well(monkeypatch):
    ions, original, calls = _fake_ions(monkeypatch)
    expected = original(GRID[10:20] * u.K)
    calls.clear()
    with proton_electron_ratio_once(GRID[::-1]):
        got = ions.proton_electron_ratio(GRID[10:20] * u.K)
    assert np.array_equal(got.value, expected.value) and calls == [GRID.size]


@pytest.mark.chianti
def test_fiascos_own_ions_get_the_same_ratio_and_contribution_function_in_the_block():
    import fiasco
    import fiasco.ions

    temperature = 10.0 ** np.linspace(5.8, 6.6, 9)
    density = [1e9, 1e10] * u.cm**-3
    plain = fiasco.Ion("Fe 12", temperature[2:5] * u.K, abundance="sun_coronal_2021_chianti")
    expected_ratio = plain.proton_electron_ratio
    expected_g = plain.contribution_function(density)
    calls = []
    original = fiasco.ions.proton_electron_ratio

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    fiasco.ions.proton_electron_ratio = counted
    try:
        with proton_electron_ratio_once(temperature):
            ion = fiasco.Ion("Fe 12", temperature[2:5] * u.K,
                             abundance="sun_coronal_2021_chianti")
            ratio = ion.proton_electron_ratio
            g = ion.contribution_function(density)
    finally:
        fiasco.ions.proton_electron_ratio = original
    assert np.array_equal(ratio.value, expected_ratio.value)
    assert np.array_equal(g.value, expected_g.value)
    assert len(calls) == 1
