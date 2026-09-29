"""Each cell's emission measure is shared between the two bins about it in temperature and velocity.

Put wholly in the nearest bin, a uniform 2.4 km/s flow on the default 5 km/s
grid was synthesised at rest and a 2.6 km/s one at 5 km/s, and a cell's
contribution function was that of the nearest temperature bin, 0.05 dex away
at most. Shared in proportion to how near each bin is, a line's mean velocity
is the cell's, and its contribution function is interpolated between the two
temperatures.
"""
import astropy.units as u
import numpy as np
import pytest

from euvst_response.synthesis import build_em_tv, compute_dem, synthesise_spectra
from euvst_response.utils import velocity_centers_to_edges, velocity_grid

LOGT = np.array([6.10, 6.15, 6.20, 6.25])
GRID = velocity_grid(5 * u.km / u.s, 300 * u.km / u.s)
KM = 1.0e5


def _one_cell(log_t, velocity_km_s, em=1.0):
    shape = (1, 1, 1)
    return (np.full(shape, log_t), np.full(shape, velocity_km_s * KM), np.full(shape, em))


def test_a_flow_between_two_velocity_bins_is_shared_between_them():
    log_t, velocity, em = _one_cell(6.15, 2.4)
    em_tv = build_em_tv(log_t, velocity, LOGT, GRID, em, "z")[0, 0, 1]
    at = {v: em_tv[np.argmin(np.abs(GRID.to_value(u.km / u.s) - v))] for v in (0.0, 5.0)}
    assert at[0.0] == pytest.approx(0.52) and at[5.0] == pytest.approx(0.48)
    assert em_tv.sum() == pytest.approx(1.0)
    assert (em_tv * GRID.to_value(u.km / u.s)).sum() == pytest.approx(2.4, rel=1e-12)


def test_a_temperature_between_two_bins_is_shared_between_them():
    log_t, velocity, em = _one_cell(6.174, 0.0)
    em_t = build_em_tv(log_t, velocity, LOGT, GRID, em, "z")[0, 0].sum(axis=-1)
    assert em_t == pytest.approx([0.0, 0.52, 0.48, 0.0])


@pytest.mark.parametrize("velocity", [2.4, -17.3, 2.6])
def test_the_synthesised_line_is_centred_on_the_flow(velocity):
    """At rest, or at 5 km/s, as the nearest bin was, a flow of 2.4 or 2.6 km/s was off by half a bin."""
    log_t, v, em = _one_cell(6.15, velocity, 1.0e27)
    em_tv = build_em_tv(log_t, v, LOGT, GRID, em, "z")
    line = {"Fe12_195.1190": {"wl0": (195.119 * u.AA).to(u.cm), "g": np.ones((1, 1, LOGT.size)),
                              "atom": 26, "ion": 12}}
    synthesise_spectra(line, em_tv, GRID, LOGT)
    spectrum = line["Fe12_195.1190"]["si"][0, 0]
    assert (spectrum * GRID.to_value(u.km / u.s)).sum() / spectrum.sum() == pytest.approx(
        velocity, abs=1e-6)


@pytest.mark.parametrize("log_t, velocity, bin_t, bin_v", [
    (6.08, 0.0, 0, 0.0),           # below the first centre, within its bin: that bin alone
    (6.27, 0.0, 3, 0.0),           # above the last centre, within its bin
    (6.15, -302.0, 1, -300.0),     # between the lowest velocity centre and its edge
    (6.15, 302.0, 1, 300.0),
])
def test_a_value_past_the_outermost_centre_stays_in_the_outermost_bin(log_t, velocity, bin_t,
                                                                      bin_v):
    log_t, velocity, em = _one_cell(log_t, velocity)
    em_tv = build_em_tv(log_t, velocity, LOGT, GRID, em, "z")[0, 0]
    assert em_tv[bin_t, np.argmin(np.abs(GRID.to_value(u.km / u.s) - bin_v))] == pytest.approx(1.0)
    assert em_tv.sum() == pytest.approx(1.0)


def test_a_value_beyond_the_edges_is_in_no_bin():
    log_t, velocity, em = _one_cell(6.15, 305.0)
    with pytest.warns(UserWarning, match="faster than the velocity grid reaches"):
        beyond_v = build_em_tv(log_t, velocity, LOGT, GRID, em, "z")
    log_t, velocity, em = _one_cell(6.40, 0.0)
    beyond_t = build_em_tv(log_t, velocity, LOGT, GRID, em, "z")
    assert beyond_v.sum() == 0.0 and beyond_t.sum() == 0.0


def _shared(values, centres, edges):
    """The bins a value is shared between, worked out one value at a time."""
    if not edges[0] <= values < edges[-1]:
        return {}
    if values <= centres[0]:
        return {0: 1.0}
    if values >= centres[-1]:
        return {centres.size - 1: 1.0}
    k = int(np.flatnonzero(centres <= values)[-1])
    share = (values - centres[k]) / (centres[k + 1] - centres[k])
    return {k: 1.0 - share, k + 1: share}


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_every_cell_is_shared_along_every_line_of_sight_as_worked_out_one_by_one(axis):
    rng = np.random.default_rng(3)
    shape = (4, 5, 6)
    log_t = rng.uniform(6.05, 6.30, shape)
    velocity = rng.uniform(-320.0, 320.0, shape) * KM
    log_n = rng.uniform(8.5, 9.5, shape)
    em = (10.0 ** log_n) ** 2 * 1.0e8
    with pytest.warns(UserWarning, match="faster than the velocity grid"):
        em_tv = build_em_tv(log_t, velocity, LOGT, GRID, em, axis)
    dem, avg_ne = compute_dem(log_t, log_n, 1.0e8, LOGT, axis)

    v_centres = GRID.to_value(u.cm / u.s)
    v_edges = velocity_centers_to_edges(v_centres)
    t_edges = np.concatenate([[LOGT[0] - 0.025], LOGT[:-1] + 0.025, [LOGT[-1] + 0.025]])
    expected = np.zeros(em_tv.shape)
    expected_t = np.zeros(dem.shape)
    expected_n = np.zeros(dem.shape)
    for z, y, x in np.ndindex(shape):
        pixel = {"x": (z, y), "y": (z, x), "z": (y, x)}[axis]
        for k, w_t in _shared(log_t[z, y, x], LOGT, t_edges).items():
            expected_t[pixel + (k,)] += em[z, y, x] * w_t
            expected_n[pixel + (k,)] += em[z, y, x] * w_t * 10.0 ** log_n[z, y, x]
            for m, w_v in _shared(velocity[z, y, x], v_centres, v_edges).items():
                expected[pixel + (k, m)] += em[z, y, x] * w_t * w_v
    np.testing.assert_allclose(em_tv, expected, rtol=1e-12, atol=0)
    np.testing.assert_allclose(dem * 0.05, expected_t, rtol=1e-10, atol=0)
    filled = expected_t > 0
    np.testing.assert_allclose(avg_ne[filled], expected_n[filled] / expected_t[filled],
                               rtol=1e-12)
    assert np.all(avg_ne[~filled] == 0.0)
