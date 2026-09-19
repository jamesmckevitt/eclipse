"""A line cube's coordinates are those of the atmosphere it came from, whatever its size.

create_line_cube puts the reference pixel of each centred axis at the middle
of that axis. With an even number of pixels the middle lies between two
pixels, and the reference value used to be the coordinate of the pixel
below it, so every position along that axis was half a cell off; the
wavelength axis took the rest wavelength as its reference value, which is
only right when the velocity grid is symmetric about zero. Both now take the
value the grid has at the reference pixel.
"""
import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from euvst_response.synthesis import create_atmosphere_ndcube, create_line_cube

REST = 195.119 * u.Angstrom
INTENSITY_UNIT = u.erg / u.s / u.cm**2 / u.sr / u.cm


def _line(wl_grid, shape):
    return {"si": np.zeros((*shape, wl_grid.size)), "wl_grid": wl_grid.to(u.cm),
            "wl0": REST.to(u.cm), "atom": 26, "ion": 12}


def _wavelengths(velocities):
    return (REST * (1 + velocities / const.c)).to(u.cm)


@pytest.mark.parametrize("nx, ny", [(4, 6), (5, 7), (4, 7)])
@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_the_image_axes_carry_the_atmospheres_coordinates(nx, ny, axis):
    """Even and odd sizes alike, each image axis reads back the reference cube's coordinates."""
    nz = 3
    reference = create_atmosphere_ndcube(np.zeros((nz, ny, nx)) * u.K,
                                         voxel_dx=1.5 * u.Mm, voxel_dy=2.5 * u.Mm,
                                         voxel_dz=0.5 * u.Mm)
    # The volume is (z, y, x); the view along one axis keeps the other two as
    # (row, column) in the order create_line_cube lays them out.
    shape = {"x": (nz, ny), "y": (nz, nx), "z": (ny, nx)}[axis]
    volume_axes = {"x": (0, 1), "y": (0, 2), "z": (1, 2)}[axis]
    wl_grid = _wavelengths(np.linspace(-300, 300, 121) * u.km / u.s)

    cube = create_line_cube("Fe12_195.1190", _line(wl_grid, shape), reference,
                            INTENSITY_UNIT, axis)

    for cube_axis, volume_axis in zip((0, 1), volume_axes):
        expected = reference.axis_world_coords(volume_axis)[0].to_value(u.Mm)
        assert np.allclose(cube.axis_world_coords(cube_axis)[0].to_value(u.Mm), expected)


@pytest.mark.parametrize("velocities", [
    np.linspace(-300, 300, 121),           # symmetric about rest, odd
    np.arange(-300, 300, 5.0),             # even count, no point at rest
    np.arange(-300, 307, 7.0),             # ends past rest, not symmetric
])
def test_the_wavelength_axis_carries_the_grid_it_was_built_on(velocities):
    wl_grid = _wavelengths(velocities * u.km / u.s)
    reference = create_atmosphere_ndcube(np.zeros((3, 4, 4)) * u.K, voxel_dx=1 * u.Mm,
                                         voxel_dy=1 * u.Mm, voxel_dz=1 * u.Mm)

    cube = create_line_cube("Fe12_195.1190", _line(wl_grid, (4, 4)), reference,
                            INTENSITY_UNIT, "z")

    assert np.allclose(cube.axis_world_coords(2)[0].to_value(u.cm), wl_grid.value,
                       rtol=0, atol=1e-13)
