"""Finding a combination, dating a map and reading fit_cube_gauss output.

A combination is matched with its units, a date keeps its offset, the
exposure-time map holds seconds, and the velocity and width helpers take
fit_cube_gauss's plain array.
"""
import astropy.units as u
import numpy as np
import pytest

from euvst_response.analysis import _utc_date, get_results_for_combination
from euvst_response.fitting import velocity_from_fit, width_from_fit


def _results(values, name="telescope.microroughness_sigma"):
    return {"results": {"all_combinations": {
        f"combination {i}": {"parameters": {name: value, "simulation.expos": 5 * u.s},
                             "index": i}
        for i, value in enumerate(values)}}}


def test_a_small_length_swept_picks_out_its_combination():
    results = _results([0.3 * u.nm, 0.5 * u.nm, 1.0 * u.nm])
    found = get_results_for_combination(results, **{"telescope.microroughness_sigma": 0.5 * u.nm})
    assert found["index"] == 1
    # The same length in another unit matches.
    found = get_results_for_combination(results, **{"telescope.microroughness_sigma": 5 * u.AA})
    assert found["index"] == 1
    # A value that was never run matches none.
    with pytest.raises(ValueError, match="No combination matches"):
        get_results_for_combination(results, **{"telescope.microroughness_sigma": 0.4 * u.nm})


def test_a_value_of_the_wrong_kind_is_refused_rather_than_matched():
    results = _results([0.3 * u.nm, 0.5 * u.nm])
    with pytest.raises(ValueError, match="simulation.expos was run in s, a unit of time"):
        get_results_for_combination(results, **{"telescope.microroughness_sigma": 0.3 * u.nm,
                                                "simulation.expos": 5 * u.m})


def test_a_setting_run_without_a_unit_takes_a_number_or_a_fraction_but_not_a_length():
    results = _results([0.64, 0.76], name="detector.qe_euv")
    assert get_results_for_combination(results, **{"detector.qe_euv": 0.76})["index"] == 1
    assert get_results_for_combination(results, **{"detector.qe_euv": 76 * u.percent})["index"] == 1
    with pytest.raises(ValueError, match="detector.qe_euv was run as a number with no unit"):
        get_results_for_combination(results, **{"detector.qe_euv": 0.76 * u.m})


def test_a_switch_is_not_matched_by_a_quantity():
    results = _results([False, True], name="simulation.psf")
    assert get_results_for_combination(results, **{"simulation.psf": True})["index"] == 1
    with pytest.raises(ValueError, match="simulation.psf is a switch, true or false"):
        get_results_for_combination(results, **{"simulation.psf": 1 * u.m})


@pytest.mark.parametrize("given, expected", [
    ("2012-06-03T12:00:00Z", "2012-06-03T12:00:00"),
    ("2012-06-03T14:00:00+02:00", "2012-06-03T12:00:00"),
    ("2012-06-03", "2012-06-03T00:00:00"),
])
def test_a_date_with_an_offset_is_written_in_utc(given, expected):
    assert _utc_date(given, "date_obs") == expected


def test_the_fits_as_fit_cube_gauss_returns_them_give_velocities_and_widths():
    """Plain array in cm, as returned by fit_cube_gauss."""
    rest = 195.119 * u.AA
    fits = np.zeros((2, 3, 4))
    fits[..., 1] = (rest * (1 + 10 * u.km / u.s / (299792.458 * u.km / u.s))).to_value(u.cm)
    fits[..., 2] = (0.03 * u.AA).to_value(u.cm)
    assert u.allclose(velocity_from_fit(fits[np.newaxis], rest, n_jobs=1),
                      10 * u.km / u.s, rtol=1e-9)
    assert u.allclose(width_from_fit(fits[np.newaxis]), 0.03 * u.AA, rtol=1e-12)


def test_the_dem_helper_says_it_is_deprecated_and_where_the_dem_is():
    from euvst_response.analysis import get_dem_data_from_results

    with pytest.warns(FutureWarning, match="get_dem_data_from_results is deprecated"):
        with pytest.raises(KeyError, match="read_synthesis_products"):
            get_dem_data_from_results({})


def test_the_deprecated_functions_warn_at_the_line_that_calls_them():
    import importlib
    import inspect
    import sys
    import warnings

    import astropy.units as u
    import numpy as np

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sys.modules.pop("euvst_response.psf", None)
        psf = importlib.import_module("euvst_response.psf")
    from euvst_response import get_dem_data_from_results

    calls = [lambda: psf._resample_psf(np.ones((4, 4)), 1 * u.arcsec, 2 * u.arcsec),
             lambda: psf._combine_psfs(np.ones((3, 3)), np.ones((3, 3)), size=3),
             lambda: get_dem_data_from_results({"dem_data": {}})]
    for call in calls:
        line = inspect.getsourcelines(call)[1]
        with pytest.warns(FutureWarning, match="is deprecated") as caught:
            call()
        assert len(caught) == 1
        assert (caught[0].filename, caught[0].lineno) == (__file__, line)
