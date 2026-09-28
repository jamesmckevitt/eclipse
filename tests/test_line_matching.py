"""A line name is matched to the line CHIANTI has observed nearest to it.

CHIANTI also lists theoretical wavelengths, many of them for weak
transitions within a few mA of strong lines, and matching against those
turned a name a little off the observed wavelength into a line up to 1e12
times fainter. These need fiasco's CHIANTI database, and are skipped
without it.
"""
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from euvst_response.synthesis import compute_goft_fiasco


def _chianti_available():
    try:
        import fiasco
    except ImportError:
        return False
    return Path(fiasco.defaults["hdf5_dbase_root"]).is_file()


pytestmark = pytest.mark.skipif(not _chianti_available(),
                                reason="needs fiasco's CHIANTI database")

# A few temperatures and densities are enough to tell the lines apart.
GRID = {"nT": 3, "logT_min": 5.6, "logT_max": 6.4, "nN": 2, "logN_min": 9.0,
        "logN_max": 9.5, "n_workers": 1}


def test_a_name_to_fewer_digits_gets_the_observed_line_it_names():
    """Fe IX has a weak theoretical transition at 177.590, and the observed line at 177.592."""
    short, _, _ = compute_goft_fiasco(["Fe09_177.59"], **GRID)
    exact, _, _ = compute_goft_fiasco(["Fe09_177.5920"], **GRID)
    assert np.array_equal(short["Fe09_177.59"]["g_tn"], exact["Fe09_177.5920"]["g_tn"])
    # The line is synthesised where CHIANTI has it, whatever the name's digits.
    assert short["Fe09_177.59"]["wl0"].to_value(u.AA) == pytest.approx(177.592)


def test_a_name_off_the_observed_wavelength_gets_it_with_a_warning():
    """193.874 is the EIS literature's Ca XIV wavelength; CHIANTI observed it at 193.866."""
    with pytest.warns(UserWarning, match="nearest line CHIANTI has observed .* 193.8660"):
        goft, _, _ = compute_goft_fiasco(["Ca14_193.874"], **GRID)
    assert goft["Ca14_193.874"]["wl0"].to_value(u.AA) == pytest.approx(193.866)
    assert goft["Ca14_193.874"]["g_tn"].max() > 1e-27


def test_one_line_named_twice_is_refused():
    """Both would be synthesised and summed, which doubles the line."""
    with pytest.raises(ValueError, match="are the same line"):
        compute_goft_fiasco(["Fe09_177.5920", "Fe09_177.59"], **GRID)
