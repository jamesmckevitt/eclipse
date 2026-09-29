"""Finding a combination, dating a map, and reading fits as fit_cube_gauss returns them.

A combination was found by its SI values to 1e-8 of at least one, so every
length under 10 nm matched every other, and units were dropped, so 5 m
matched 5 s. A date with an offset stamped the maps with the time they were
made. The exposure-time map held the exposure's index under a unit of
seconds, and the velocity and width helpers failed on the plain array
fit_cube_gauss returns.
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
    # The same length in another unit is the same value.
    found = get_results_for_combination(results, **{"telescope.microroughness_sigma": 5 * u.AA})
    assert found["index"] == 1
    # A value that was never run is not any of them.
    with pytest.raises(ValueError, match="No combination matches"):
        get_results_for_combination(results, **{"telescope.microroughness_sigma": 0.4 * u.nm})


def test_a_value_of_the_wrong_kind_is_refused_rather_than_matched():
    results = _results([0.3 * u.nm, 0.5 * u.nm])
    with pytest.raises(ValueError, match="simulation.expos was run in s, a unit of time"):
        get_results_for_combination(results, **{"telescope.microroughness_sigma": 0.3 * u.nm,
                                                "simulation.expos": 5 * u.m})


@pytest.mark.parametrize("given, expected", [
    ("2012-06-03T12:00:00Z", "2012-06-03T12:00:00"),
    ("2012-06-03T14:00:00+02:00", "2012-06-03T12:00:00"),
    ("2012-06-03", "2012-06-03T00:00:00"),
])
def test_a_date_with_an_offset_is_written_in_utc(given, expected):
    assert _utc_date(given, "date_obs") == expected


def test_the_fits_as_fit_cube_gauss_returns_them_give_velocities_and_widths():
    """A plain array, in the cm fit_cube_gauss fits in."""
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
