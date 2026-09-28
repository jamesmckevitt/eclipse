"""The velocity and wavelength grids have to be uniform, and now say so.

ECLIPSE takes one spacing for the whole velocity grid and uses it for every
bin edge, and writes the output cube's wavelength CDELT from it. Both are
correct for a uniform grid and quietly wrong otherwise, which is what these
tests pin down: the shape of the right answer for a uniform grid, and a
refusal rather than a plausible number for every way a grid can fail to be
one.
"""
import warnings

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from euvst_response.synthesis import (
    create_atmosphere_ndcube,
    create_line_cube,
    require_uniform_grid,
    synthesise_spectra,
    velocity_centers_to_edges,
)

REST = 195.119 * u.Angstrom
INTENSITY_UNIT = u.erg / u.s / u.cm**2 / u.sr / u.cm


def test_edges_are_the_midpoints_of_a_uniform_grid():
    """Interior edges sit halfway between centres, ends half a bin beyond."""
    centres = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    edges = velocity_centers_to_edges(centres)

    assert edges.shape == (centres.size + 1,)
    assert np.allclose(edges, [-12.5, -7.5, -2.5, 2.5, 7.5, 12.5])
    # Every centre lies in its own bin, which is the property the binning in
    # build_em_tv relies on.
    assert np.all(edges[:-1] <= centres)
    assert np.all(centres < edges[1:])


def test_arange_and_linspace_grids_are_accepted():
    """Float rounding in the usual constructors must not trip the check."""
    step = require_uniform_grid(
        np.arange(-300.0, 300.0 + 5.0, 5.0), "vel_grid")
    assert step == pytest.approx(5.0)

    step = require_uniform_grid(np.linspace(-300.0, 300.0, 121), "vel_grid")
    assert step == pytest.approx(5.0)


def test_quantity_grids_are_accepted_and_keep_their_units():
    """synthesise_spectra passes a Quantity, build_em_tv a bare array."""
    grid = np.arange(-100.0, 100.0 + 10.0, 10.0) * u.km / u.s
    assert require_uniform_grid(grid, "vel_grid") == pytest.approx(10.0)
    assert require_uniform_grid(grid.value, "vel_grid") == pytest.approx(10.0)


def test_uneven_grid_is_refused():
    """A grid that is uniform at the start is the dangerous case: it looks fine and is not."""
    centres = np.array([0.0, 5.0, 10.0, 15.0, 30.0])
    with pytest.raises(ValueError, match="evenly spaced"):
        velocity_centers_to_edges(centres)


def test_error_points_at_the_element_furthest_from_even():
    """The grid through the first and last elements, spaced by 8, has element 4 at 32, not 21."""
    centres = np.array([0.0, 5.0, 10.0, 16.0, 21.0, 40.0])
    with pytest.raises(ValueError, match=r"Element 4 is 21, -1.38 of a spacing \(8\)"):
        require_uniform_grid(centres, "vel_grid")


def test_edges_use_the_mean_spacing():
    """
    Within the tolerance, the one spacing applied everywhere is the mean.

    The second spacing here is off by 5e-7 of a bin, which the check lets
    through. The first spacing would carry that slip to every later edge.
    """
    centres = np.array([0.0, 1.0, 2.0 + 5e-7, 3.0 + 5e-7])
    step = (3.0 + 5e-7) / 3
    assert require_uniform_grid(centres, "vel_grid") == pytest.approx(step, rel=1e-15)
    assert velocity_centers_to_edges(centres)[0] == pytest.approx(-step / 2, rel=1e-15)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_grid_rounded_to_single_precision_counts_as_even(dtype):
    """
    As MURaM's heights are, 0.064 Mm apart up to 42 Mm and stored as float32.

    The rounding stays in the values when they are cast to double. A slip of
    two thousandths of a spacing is still refused.
    """
    heights = (np.arange(656, dtype=np.float32) * np.float32(0.064)).astype(dtype)
    assert heights[-1] > 41.9 and np.ptp(np.diff(heights.astype(float))) > 1e-6 * 0.064
    assert require_uniform_grid(heights, "z") == pytest.approx(0.064, rel=1e-6)
    slipped = heights.astype(float)
    slipped[300:] += 2e-3 * 0.064
    with pytest.raises(ValueError, match="z must be evenly spaced"):
        require_uniform_grid(slipped, "z")


def test_single_precision_rounding_is_not_admitted_where_it_reaches_half_a_spacing():
    """
    Near 1e8 single precision rounds by more than a spacing of 1, so it cannot hold such a grid.

    Spacings of 1, 1, 1 and 7 are uneven there as anywhere, and an even grid
    there is held to the usual tolerance and passes it.
    """
    with pytest.raises(ValueError, match=r"Element 3 is 100000003, -1.8 of a spacing"):
        require_uniform_grid(1e8 + np.array([0.0, 1.0, 2.0, 3.0, 10.0]), "z")
    assert require_uniform_grid(1e8 + np.arange(5.0), "z") == 1.0


@pytest.mark.parametrize("centres", [
    np.array([np.nan, 5.0, 10.0, 15.0]),
    np.array([0.0, 5.0, np.nan, 15.0]),
    np.array([0.0, 5.0, 10.0, np.inf]),
    np.array([-np.inf, 5.0, 10.0, 15.0]),
])
def test_non_finite_grid_is_refused(centres):
    """NaN compares false with everything, so it would otherwise pass."""
    with pytest.raises(ValueError, match="must be finite"):
        velocity_centers_to_edges(centres)


def test_decreasing_grid_is_refused():
    """Descending centres give descending edges, so every bin is empty."""
    with pytest.raises(ValueError, match="must increase"):
        velocity_centers_to_edges(np.array([10.0, 5.0, 0.0, -5.0]))


def test_short_grid_is_refused():
    for bad in (np.array([1.0]), np.array([])):
        with pytest.raises(ValueError, match="at least 2 elements"):
            velocity_centers_to_edges(bad)


def _one_line_goft(n_rows, n_cols, n_temp):
    return {"Fe12_195.1190": {
        "wl0": REST.to(u.cm),
        "g": np.ones((n_rows, n_cols, n_temp)),
        "atom": 26,
        "ion": 12,
    }}


def test_synthesise_spectra_refuses_an_uneven_velocity_grid():
    """The wavelength axis is the velocity axis, so it inherits the problem."""
    ny, nx, n_temp = 2, 3, 2
    logT_grid = np.array([6.0, 6.2])
    uneven = np.array([-50.0e5, 0.0, 10.0e5]) * (u.cm / u.s)
    em_tv = np.zeros((ny, nx, n_temp, uneven.size))

    with pytest.raises(ValueError, match="evenly spaced"):
        synthesise_spectra(_one_line_goft(ny, nx, n_temp), em_tv, uneven,
                           logT_grid)


def test_create_line_cube_refuses_an_uneven_wavelength_grid():
    """Checked here too, since the DEM and VDEM routes call it directly."""
    nz, ny, nx, n_lambda = 4, 2, 3, 3
    line_data = {
        "si": np.ones((ny, nx, n_lambda)),
        "wl_grid": np.array([195.0, 195.1, 195.4]) * u.Angstrom,
        "wl0": REST.to(u.cm),
        "atom": 26,
        "ion": 12,
    }
    reference = create_atmosphere_ndcube(
        np.zeros((nz, ny, nx)) * u.K, 1 * u.Mm, 1 * u.Mm, 1 * u.Mm)

    with pytest.raises(ValueError, match="evenly spaced"):
        create_line_cube("Fe12_195.1190", line_data, reference,
                         INTENSITY_UNIT, integration_axis="z")


def test_a_uniform_grid_still_synthesises_end_to_end():
    """The guard must not stand in the way of the grids people actually use."""
    nz, ny, nx, n_temp = 4, 2, 3, 2
    logT_grid = np.array([6.0, 6.2])
    vel_grid = np.arange(-50.0, 50.0 + 25.0, 25.0) * u.km / u.s
    em_tv = np.zeros((ny, nx, n_temp, vel_grid.size))
    em_tv[:, :, 0, vel_grid.size // 2] = 1.0e27

    goft = _one_line_goft(ny, nx, n_temp)
    synthesise_spectra(goft, em_tv, vel_grid.to(u.cm / u.s), logT_grid)

    reference = create_atmosphere_ndcube(
        np.zeros((nz, ny, nx)) * u.K, 1 * u.Mm, 1 * u.Mm, 1 * u.Mm)
    cube = create_line_cube("Fe12_195.1190", goft["Fe12_195.1190"], reference,
                            INTENSITY_UNIT, integration_axis="z")

    assert cube.data.shape == (ny, nx, vel_grid.size)
    assert np.all(np.isfinite(cube.data))
    # CDELT is the wavelength step the uniform velocity grid implies.
    expected = (REST * (25.0 * u.km / u.s) / const.c).to_value(u.cm)
    assert cube.wcs.wcs.cdelt[0] == pytest.approx(expected, rel=1e-6)


# ----------------------------------------------------------------------
# The velocity grid, and what falls off it
# ----------------------------------------------------------------------
from euvst_response.synthesis import build_em_tv, compute_dem  # noqa: E402
from euvst_response.utils import velocity_grid  # noqa: E402


@pytest.mark.parametrize("res, lim", [(3, 10), (0.7, 300), (20, 250), (5, 300)])
def test_the_velocity_grid_has_a_bin_centred_on_zero_and_reaches_its_limit(res, lim):
    """Built from -lim, a limit that was not a whole number of steps left zero between bins."""
    grid = velocity_grid(res * u.km / u.s, lim * u.km / u.s).to_value(u.km / u.s)
    assert np.min(np.abs(grid)) == 0.0
    assert grid == pytest.approx(-grid[::-1], abs=1e-9)
    assert grid[-1] >= lim - 1e-9 and grid[-1] < lim + res
    assert np.diff(grid) == pytest.approx(res)
    if (res, lim) == (5, 300):
        # The default grid is the one there always was.
        assert grid.size == 121


@pytest.mark.parametrize("res, lim", [("0 km/s", "300 km/s"), ("5 km/s", "-5 km/s"),
                                      ("5 km", "300 km/s")])
def test_a_velocity_grid_that_cannot_be_built_is_refused_naming_the_option(res, lim):
    with pytest.raises(ValueError, match="--vel-.* must be a positive velocity"):
        velocity_grid(u.Quantity(res), u.Quantity(lim), ("--vel-res", "--vel-lim"))


def test_a_velocity_grid_in_any_unit_shifts_the_lines_alike():
    """A grid in km/s was taken to be in cm/s, moving every line 1e5 times too little."""
    ny, nx, n_temp = 1, 1, 2
    grid = velocity_grid(10 * u.km / u.s, 100 * u.km / u.s)
    em_tv = np.zeros((ny, nx, n_temp, grid.size))
    em_tv[0, 0, 0, -3] = 1.0e27  # +80 km/s
    spectra = {}
    for unit in (u.cm / u.s, u.km / u.s, u.m / u.s):
        goft = _one_line_goft(ny, nx, n_temp)
        synthesise_spectra(goft, em_tv, grid.to(unit), np.array([6.0, 6.2]))
        spectra[unit] = goft["Fe12_195.1190"]
    reference = spectra[u.cm / u.s]
    for spectrum in spectra.values():
        assert np.allclose(spectrum["si"], reference["si"], rtol=1e-12)
        assert u.allclose(spectrum["wl_grid"], reference["wl_grid"], rtol=1e-15)


def test_emission_beyond_the_velocity_grid_is_left_out_and_said_to_be():
    """A 500 km/s flow emits beyond a +-300 km/s window, while the DEM keeps it."""
    logT = np.full((2, 1, 1), 6.1)
    velocity = np.array([0.0, 500.0e5]).reshape(2, 1, 1)
    grid = velocity_grid(5 * u.km / u.s, 300 * u.km / u.s).to_value(u.cm / u.s)
    with pytest.warns(UserWarning, match="^50 per cent of the emission measure"):
        em_tv = build_em_tv(logT, velocity, np.array([6.0, 6.2]), grid * (u.cm / u.s),
                            np.ones((2, 1, 1)), "z")
    assert em_tv.sum() == pytest.approx(1.0)


def test_the_density_of_a_temperature_with_no_plasma_is_zero():
    """It was whatever memory the division left, which reached the saved G diagnostic."""
    logT = np.full((3, 4, 5), 6.0)
    logN = np.full((3, 4, 5), 9.0)
    _, avg_ne = compute_dem(logT, logN, 1.0e8, np.array([5.5, 6.0, 6.5]), "z")
    assert np.all(avg_ne[..., [0, 2]] == 0.0)
    assert avg_ne[..., 1] == pytest.approx(1.0e9)


def test_a_plain_array_makes_a_cube_with_no_unit():
    """As the docstring allows."""
    cube = create_atmosphere_ndcube(np.ones((2, 3, 4)), 1 * u.Mm, 1 * u.Mm, 1 * u.Mm)
    assert cube.unit is None and cube.data.shape == (2, 3, 4)


# ----------------------------------------------------------------------
# The density grid, sized to the atmosphere
# ----------------------------------------------------------------------
from euvst_response.synthesis import density_grid, synthesise_cubes  # noqa: E402

LOG_T = np.linspace(4.0, 9.0, 101)


def _grid_points(log_ne, temperature=1.0e6):
    density = 10.0 ** np.asarray(log_ne, dtype=float).reshape(-1, 1, 1)
    logN_min, logN_max, nN = density_grid(np.full(density.shape, temperature), density, LOG_T)
    return np.linspace(logN_min, logN_max, nN)


@pytest.mark.parametrize("log_ne, expected", [
    # Inside the default grid: its own points, past the densities, and no more.
    ([8.5, 11.0], [8.2, 8.5, 8.8, 9.1, 9.4, 9.7, 10.0, 10.3, 10.6, 10.9, 11.2]),
    # On one of its points: the points either side, so that a mean density a
    # rounding error off it is still on the grid.
    ([10.0], [9.7, 10.0, 10.3]),
    # Beyond it either way: carried on, on the same points.
    ([5.0, 7.2], np.round(np.arange(4.9, 7.5, 0.3), 1)),
    ([12.5, 14.0], np.round(np.arange(12.4, 14.4, 0.3), 1)),
])
def test_the_density_grid_takes_in_every_density_on_the_default_grids_points(log_ne, expected):
    assert _grid_points(log_ne) == pytest.approx(expected, abs=1e-9)


def test_only_cells_with_electrons_at_a_temperature_on_the_grid_count():
    """A photosphere below 10^4 K, or a cell with no electrons, needs no contribution function."""
    temperature = np.array([5.0e3, 1.0e6, 1.0e6]).reshape(-1, 1, 1)
    density = np.array([1.0e17, 1.0e9, 0.0]).reshape(-1, 1, 1)
    logN_min, logN_max, _ = density_grid(temperature, density, LOG_T)
    assert (logN_min, logN_max) == pytest.approx((8.8, 9.1))
    # With none, the default grid.
    assert density_grid(temperature[:1], density[:1], LOG_T) == (7.0, 13.0, 21)


def _flat_goft_on(logN_grid, n_temp):
    return {"Fe12_195.1190": {"wl0": REST.to(u.cm), "atom": 26, "ion": 12,
                              "g_tn": np.full((logN_grid.size, n_temp), 1.0e-24)}}


@pytest.mark.parametrize("log_ne", [5.0, 7.0, 13.0, 14.5])
def test_no_emission_is_lost_at_densities_the_default_grid_did_not_reach(log_ne):
    """G(T, n) was taken as zero below 10^7 and above 10^13 cm^-3."""
    shape = (3, 1, 2)
    temperature = np.full(shape, 1.0e6)
    density = np.full(shape, 10.0 ** log_ne)
    density[:, :, 1] = 0.0  # a column with no electrons emits nothing and is no warning
    vel_grid = velocity_grid(5 * u.km / u.s, 150 * u.km / u.s)
    logN_min, logN_max, nN = density_grid(temperature, density, LOG_T)
    logN_grid = np.linspace(logN_min, logN_max, nN)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lines, dem_map, _ = synthesise_cubes(
            temperature, density, np.zeros(shape), 1.0e8, _flat_goft_on(logN_grid, LOG_T.size),
            LOG_T, logN_grid, vel_grid, "z", np.float64)
    intensity = lines["Fe12_195.1190"]["si"].sum(axis=-1)
    assert intensity[0, 0] > 0 and intensity[0, 1] == 0
    # G n_e^2 dh over 4 pi, in erg / (s cm^2 sr), whatever the density.
    wavelength_step = np.diff(lines["Fe12_195.1190"]["wl_grid"].to_value(u.cm))[0]
    expected = 1.0e-24 * 3 * 10.0 ** (2 * log_ne) * 1.0e8 / (4 * np.pi)
    assert intensity[0, 0] * wavelength_step == pytest.approx(expected, rel=1e-6)
