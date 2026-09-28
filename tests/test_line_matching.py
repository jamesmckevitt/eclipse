"""A line name is matched to the line CHIANTI has observed nearest to it.

CHIANTI also lists theoretical wavelengths, many of them for weak
transitions within a few mA of strong lines, and matching against those
turned a name a little off the observed wavelength into a line up to 1e12
times fainter. The first tests use a stand-in for fiasco with one such pair;
those against CHIANTI itself need fiasco's database, and are skipped
without it.
"""
import sys
import types
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from euvst_response.synthesis import compute_goft_fiasco

# A few temperatures and densities are enough to tell the lines apart.
GRID = {"nT": 3, "logT_min": 5.6, "logT_max": 6.4, "nN": 2, "logN_min": 9.0,
        "logN_max": 9.5, "n_workers": 1}


# ---------------------------------------------------------------------------
# A stand-in for fiasco
# ---------------------------------------------------------------------------
class _Transitions:
    """A strong observed line, a weak theoretical one 1.5 mA from it, another observed line."""

    def __init__(self, any_observed):
        self.wavelength = np.array([195.119, 195.1205, 186.880, 180.0]) * u.AA
        self.is_bound_bound = np.array([True, True, True, False])
        self.is_observed = np.array([True, False, True, False]) & any_observed


class _Ion:
    """Enough of fiasco.Ion for the matching; its Ca XIV has no observed lines."""

    def __init__(self, name, temperature, abundance=None, hdf5_dbase_root=None):
        self.temperature = temperature
        self.atomic_number = 20 if name.startswith("Ca") else 26
        self.hdf5_dbase_root = hdf5_dbase_root or "default.h5"
        self.transitions = _Transitions(any_observed=not name.startswith("Ca"))

    def contribution_function(self, density):
        strength = np.array([1.0, 1e-5, 0.3])
        ones = np.ones((self.temperature.size, density.size, 1))
        return 1e-24 * ones * strength * u.erg * u.cm**3 / u.s

    @property
    def proton_electron_ratio(self):
        return np.full(self.temperature.size, 0.83)


@pytest.fixture
def fake_fiasco(tmp_path, monkeypatch):
    module = types.ModuleType("fiasco")
    module.Ion = _Ion
    (tmp_path / "chianti_dbase.h5").touch()
    module.defaults = {"hdf5_dbase_root": str(tmp_path / "chianti_dbase.h5")}
    # The database is there, so the offer to build it asks nothing.
    util = types.ModuleType("fiasco.util")
    util.check_database = lambda hdf5_dbase_root, **kwargs: None
    module.util = util
    monkeypatch.setitem(sys.modules, "fiasco", module)
    monkeypatch.setitem(sys.modules, "fiasco.util", util)


def test_a_name_nearer_a_theoretical_transition_gets_the_observed_line(fake_fiasco):
    """195.12 is nearer the weak transition at 195.1205 than the observed line at 195.119."""
    goft, _, _ = compute_goft_fiasco(["Fe12_195.12"], **GRID)
    assert goft["Fe12_195.12"]["transition"] == 0
    assert goft["Fe12_195.12"]["wl0"].to_value(u.AA) == pytest.approx(195.119)


def test_a_name_off_every_observed_line_gets_the_nearest_with_a_warning(fake_fiasco):
    with pytest.warns(UserWarning, match="nearest line CHIANTI has observed .* 195.1190"):
        goft, _, _ = compute_goft_fiasco(["Fe12_195.125"], **GRID)
    assert goft["Fe12_195.125"]["transition"] == 0


def test_one_line_named_twice_is_refused_whatever_the_names(fake_fiasco):
    with pytest.raises(ValueError, match="are the same line"):
        compute_goft_fiasco(["Fe12_195.1190", "Fe12_195.12"], **GRID)


def test_an_ion_with_no_observed_lines_is_matched_to_a_theoretical_one_saying_so(fake_fiasco):
    with pytest.warns(UserWarning) as caught:
        goft, _, _ = compute_goft_fiasco(["Ca14_195.1205"], **GRID)
    messages = [str(w.message) for w in caught]
    assert any("no observed wavelengths" in m and "195.1205" in m for m in messages)
    assert not any("has observed" in m for m in messages)
    assert goft["Ca14_195.1205"]["transition"] == 1


# ---------------------------------------------------------------------------
# Against CHIANTI itself
# ---------------------------------------------------------------------------
def _chianti_available():
    try:
        import fiasco
    except ImportError:
        return False
    return Path(fiasco.defaults["hdf5_dbase_root"]).is_file()


chianti = pytest.mark.skipif(not _chianti_available(), reason="needs fiasco's CHIANTI database")


@chianti
def test_a_name_to_fewer_digits_gets_the_observed_line_it_names():
    """Fe IX has a weak theoretical transition at 177.590, and the observed line at 177.592."""
    short, _, _ = compute_goft_fiasco(["Fe09_177.59"], **GRID)
    exact, _, _ = compute_goft_fiasco(["Fe09_177.5920"], **GRID)
    assert np.array_equal(short["Fe09_177.59"]["g_tn"], exact["Fe09_177.5920"]["g_tn"])
    # The line is synthesised where CHIANTI has it, whatever the name's digits.
    assert short["Fe09_177.59"]["wl0"].to_value(u.AA) == pytest.approx(177.592)


@chianti
def test_a_name_off_the_observed_wavelength_gets_it_with_a_warning():
    """193.874 is the EIS literature's Ca XIV wavelength; CHIANTI observed it at 193.866."""
    with pytest.warns(UserWarning, match="nearest line CHIANTI has observed .* 193.8660"):
        goft, _, _ = compute_goft_fiasco(["Ca14_193.874"], **GRID)
    assert goft["Ca14_193.874"]["wl0"].to_value(u.AA) == pytest.approx(193.866)
    assert goft["Ca14_193.874"]["g_tn"].max() > 1e-27


@chianti
def test_one_line_named_twice_is_refused():
    """Both would be synthesised and summed, which doubles the line."""
    with pytest.raises(ValueError, match="are the same line"):
        compute_goft_fiasco(["Fe09_177.5920", "Fe09_177.59"], **GRID)
