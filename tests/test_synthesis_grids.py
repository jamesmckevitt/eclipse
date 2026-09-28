"""The velocity and wavelength grids have to be uniform, and now say so.

ECLIPSE takes one spacing for the whole velocity grid and uses it for every
bin edge, and writes the output cube's wavelength CDELT from it. Both are
correct for a uniform grid and quietly wrong otherwise, which is what these
tests pin down: the shape of the right answer for a uniform grid, and a
refusal rather than a plausible number for every way a grid can fail to be
one.
"""
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
