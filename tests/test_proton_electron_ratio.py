"""fiasco's proton to electron ratio, worked out once per process within proton_electron_ratio_once."""
import sys
import types

import astropy.units as u
import numpy as np

from eclipse.utils import _PROTON_ELECTRON_RATIOS, proton_electron_ratio_once

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
    with proton_electron_ratio_once(GRID):
        pass
    assert isinstance(_PROTON_ELECTRON_RATIOS, dict)
