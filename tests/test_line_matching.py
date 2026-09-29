"""A line name is matched to the line at its wavelength, or failing one to the nearest observed line.

CHIANTI lists theoretical wavelengths as well as observed ones, many of them
for weak transitions within a few mA of strong lines, and matching a name to
the nearest of all of them turned a name a little off the observed
wavelength into a line up to 1e12 times fainter. A theoretical line named at
its own wavelength is still synthesised. The first tests use a stand-in for
fiasco with such lines; those against CHIANTI itself need fiasco's
database, and are skipped without it.
"""
import sys
import types

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
    """
    A strong observed line and a weak theoretical one 1.5 mA from it, and two theoretical
    transitions at one wavelength, the second the brighter.
    """

    def __init__(self, any_observed):
        self.wavelength = np.array([195.119, 195.1205, 186.883, 186.883, 180.0]) * u.AA
        self.is_bound_bound = np.array([True, True, True, True, False])
        self.is_observed = np.array([True, False, False, False, False]) & any_observed


class _Ion:
    """Enough of fiasco.Ion for the matching; its Ca XIV has no observed lines."""

    def __init__(self, name, temperature, abundance=None, hdf5_dbase_root=None):
        self.temperature = temperature
        self.atomic_number = 20 if name.startswith("Ca") else 26
        self.hdf5_dbase_root = hdf5_dbase_root or "default.h5"
        self.transitions = _Transitions(any_observed=not name.startswith("Ca"))

    def contribution_function(self, density):
        strength = np.array([1.0, 1e-5, 1e-10, 1e-5])
        ones = np.ones((self.temperature.size, density.size, 1))
        return 1e-24 * ones * strength * u.erg * u.cm**3 / u.s

    @property
    def proton_electron_ratio(self):
        return np.full(self.temperature.size, 0.83)


@pytest.fixture
def fake_fiasco(tmp_path, monkeypatch):
    # The real fiasco is imported first, so that the logger it names after
    # itself is its own. Made first by the synthesis, under the stand-in, it
    # is a plain one, which the real fiasco fails to set up when a later test
    # imports it.
    import fiasco  # noqa: F401
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


def test_a_name_at_both_to_its_digits_gets_the_observed_line(fake_fiasco):
    """195.12 is nearer the weak transition at 195.1205 than the observed line at 195.119."""
    goft, _, _ = compute_goft_fiasco(["Fe12_195.12"], **GRID)
    assert goft["Fe12_195.12"]["transition"] == 0
    assert goft["Fe12_195.12"]["wl0"].to_value(u.AA) == pytest.approx(195.119)


def test_a_theoretical_line_named_at_its_wavelength_is_synthesised(fake_fiasco, recwarn):
    goft, _, _ = compute_goft_fiasco(["Fe12_195.1205"], **GRID)
    assert goft["Fe12_195.1205"]["transition"] == 1
    assert goft["Fe12_195.1205"]["wl0"].to_value(u.AA) == pytest.approx(195.1205)
    assert not [w for w in recwarn if "Fe12_195.1205" in str(w.message)]


def test_of_two_transitions_at_one_wavelength_the_brighter_is_taken(fake_fiasco):
    goft, _, _ = compute_goft_fiasco(["Fe12_186.883"], **GRID)
    assert goft["Fe12_186.883"]["transition"] == 3


def test_a_name_off_every_line_gets_the_nearest_observed_with_a_warning(fake_fiasco):
    """195.125 is nearer the theoretical 195.1205, as a literature wavelength off CHIANTI's can be."""
    with pytest.warns(UserWarning, match="nearest line CHIANTI has observed .* 195.1190"):
        goft, _, _ = compute_goft_fiasco(["Fe12_195.125"], **GRID)
    assert goft["Fe12_195.125"]["transition"] == 0


def test_one_line_named_twice_is_refused_whatever_the_names(fake_fiasco):
    with pytest.raises(ValueError, match="are the same line"):
        compute_goft_fiasco(["Fe12_195.1190", "Fe12_195.12"], **GRID)


def test_an_ion_with_no_observed_lines_falls_back_to_a_theoretical_one_saying_so(fake_fiasco):
    with pytest.warns(UserWarning) as caught:
        goft, _, _ = compute_goft_fiasco(["Ca14_195.125"], **GRID)
    messages = [str(w.message) for w in caught]
    assert any("observed none of its lines" in m and "195.1205" in m for m in messages)
    assert goft["Ca14_195.125"]["transition"] == 1


# ---------------------------------------------------------------------------
# Against CHIANTI itself
# ---------------------------------------------------------------------------
# Skipped where fiasco's database is not built; see conftest.
chianti = pytest.mark.chianti


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
def test_a_theoretical_line_of_chianti_is_synthesised_where_it_is_named():
    """Fe XII has a theoretical transition at 193.526, 2e-26 at its brightest, beside the observed 193.509."""
    goft, _, _ = compute_goft_fiasco(["Fe12_193.526", "Fe12_195.12"], **GRID)
    assert goft["Fe12_193.526"]["wl0"].to_value(u.AA) == pytest.approx(193.526)
    assert goft["Fe12_195.12"]["wl0"].to_value(u.AA) == pytest.approx(195.119)


@chianti
def test_one_line_named_twice_is_refused():
    """Both would be synthesised and summed, which doubles the line."""
    with pytest.raises(ValueError, match="are the same line"):
        compute_goft_fiasco(["Fe09_177.5920", "Fe09_177.59"], **GRID)
