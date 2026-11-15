"""Spectra built a block of image rows at a time are those from the whole emission measure."""

import warnings

import astropy.units as u
import numpy as np
import pytest

from eclipse import synthesis
from eclipse.synthesis import synthesise_cubes

LOGT = np.linspace(5.6, 6.6, 11)
LOGN = np.linspace(8.0, 10.0, 9)
VEL = np.arange(-150.0, 151.0, 10.0) * u.km / u.s


def _goft():
    """One line whose contribution function changes with temperature and density."""
    g_tn = (1e-25 * np.exp(-((LOGT[None, :] - 6.1) / 0.2) ** 2)
            * (1.0 + 0.1 * (LOGN[:, None] - 9.0)))
    return {"Fe12_195.1190": {"wl0": (195.119 * u.AA).to(u.cm), "g_tn": g_tn,
                              "atom": 26, "ion": 12}}


def _box(fast=False):
    """Temperature, electron density and velocity in cm/s of a (6, 7, 5) box."""
    generator = np.random.default_rng(3)
    shape = (6, 7, 5)
    temperature = 10.0 ** generator.uniform(5.7, 6.5, shape)
    density = 10.0 ** generator.uniform(8.5, 9.5, shape)
    velocity = generator.uniform(-120e5, 120e5, shape)
    if fast:
        velocity[0, 0, 0] = 400e5
    return temperature, density, velocity


def _synthesise(view, keep_em_tv, precision=np.float64, fast=False):
    temperature, density, velocity = _box(fast)
    axis = view[-1]
    dh = np.linspace(1e7, 3e7, temperature.shape["zyx".index(axis)])
    return synthesise_cubes(temperature, density, velocity, dh, _goft(), LOGT, LOGN, VEL,
                            view, precision, return_density=True, keep_em_tv=keep_em_tv)


@pytest.mark.parametrize("view", ["z", "-x", "y", "-z"])
def test_the_spectra_by_rows_are_those_from_the_whole_emission_measure(view, monkeypatch):
    whole, dem, em_tv, density = _synthesise(view, keep_em_tv=True)
    # One image row per block, so every block boundary is crossed.
    monkeypatch.setattr(synthesis, "EM_BLOCK_BYTES", 1)
    by_rows, dem_rows, none, density_rows = _synthesise(view, keep_em_tv=False)
    assert none is None
    for name in whole:
        assert np.array_equal(by_rows[name]["si"], whole[name]["si"])
        assert np.array_equal(by_rows[name]["wl_grid"], whole[name]["wl_grid"])
    assert np.array_equal(dem_rows, dem) and np.array_equal(density_rows, density)
    assert em_tv.shape[:2] == whole["Fe12_195.1190"]["si"].shape[:2]


def test_plasma_beyond_the_velocity_grid_is_warned_of_once_with_the_whole_fraction(monkeypatch):
    with pytest.warns(UserWarning, match="faster") as whole:
        _synthesise("z", keep_em_tv=True, fast=True)
    monkeypatch.setattr(synthesis, "EM_BLOCK_BYTES", 1)
    with warnings.catch_warnings(record=True) as by_rows:
        warnings.simplefilter("always")
        _synthesise("z", keep_em_tv=False, fast=True)
    faster = [str(w.message) for w in by_rows if "faster" in str(w.message)]
    assert faster == [str(w.message) for w in whole if "faster" in str(w.message)]



@pytest.mark.parametrize("view", ["z", "-x"])
def test_g_by_rows_is_g_from_the_whole_box(view, monkeypatch):
    whole, _, _, _ = _synthesise(view, keep_em_tv=True)
    monkeypatch.setattr(synthesis, "EM_BLOCK_BYTES", 1)
    by_rows, _, _, _ = _synthesise(view, keep_em_tv=False)
    for name in whole:
        assert by_rows[name]["g"].dtype == whole[name]["g"].dtype
        assert np.array_equal(by_rows[name]["g"], whole[name]["g"])


def test_the_logs_a_slab_at_a_time_are_the_logs_of_the_whole_box():
    temperature, density, _ = _box()
    mass = (density / u.cm**3 * 1.2 * u.u).to(u.g / u.cm**3).astype(np.float32)
    per_electron = 1.2 * u.u
    whole = synthesis._log_cubes(temperature.astype(np.float32),
                                 (mass / per_electron).to_value(u.cm**-3), np.float32)
    by_slab = synthesis._log_cubes_by_slab(
        temperature.astype(np.float32), mass,
        lambda part: (part / per_electron).to_value(u.cm**-3), np.float32, slab=4)
    for expected, got in zip(whole, by_slab):
        assert got.dtype == np.float32 and np.array_equal(got, expected)
