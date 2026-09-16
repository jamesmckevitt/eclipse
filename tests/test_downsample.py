"""--downsample has to describe the same Sun as a full-resolution run.

Striding by a factor keeps one cell in every *factor* along each axis, so each
kept cell stands for that many cells of the simulation. The voxel sizes have to
grow by the factor exactly once: in the WCS, in the path length used for the
emission measure, and in what the synthesis file records. The factor used to
be applied in main() and again in every load_cube call, in place, so it
compounded across the cubes (issue #72).

These run the real synthesis on a small uniform atmosphere with a flat
contribution function standing in for fiasco, so a uniform column has to come
out with the same intensity at any downsampling.
"""
import sys

import astropy.constants as const
import astropy.units as u
import dill
import numpy as np
import pytest

from euvst_response import synthesis
from euvst_response.synthesis import load_cube

LINE = "Fe12_195.1190"
REST = 195.119 * u.Angstrom
TEMPERATURE = 1.0e6 * u.K
ELECTRON_DENSITY = 1.0e9 / u.cm**3
MEAN_MOL_WT = 1.29
VOXEL = {"dx": 0.1 * u.Mm, "dy": 0.15 * u.Mm, "dz": 0.05 * u.Mm}


def _write_cube(path, shape, value):
    """A uniform cube in the file's own (nx, nz, ny) Fortran layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.full(shape, value, dtype=np.float32).ravel(order="F").tofile(path)


def _write_atmosphere(root, shape, suffix="0270000"):
    """Temperature, density and vertical velocity for one snapshot."""
    density = (ELECTRON_DENSITY * MEAN_MOL_WT * const.u).to_value(u.g / u.cm**3)
    _write_cube(root / "temp" / f"eosT.{suffix}", shape,
                TEMPERATURE.to_value(u.K))
    _write_cube(root / "rho" / f"result_prim_0.{suffix}", shape, density)
    _write_cube(root / "vz" / f"result_prim_2.{suffix}", shape, 0.0)


def _flat_goft(lines, **kwargs):
    """A contribution function of 1 everywhere, in place of fiasco."""
    logT_grid = np.linspace(5.0, 7.0, 21)
    logN_grid = np.linspace(8.0, 10.0, 21)
    goft = {LINE: {
        "wl0": REST.to(u.cm),
        "g_tn": np.ones((logN_grid.size, logT_grid.size)),
        "atom": 26,
        "ion": 12,
        "hdf5_dbase_root": None,
    }}
    return goft, logT_grid, logN_grid


def _synthesise(tmp_path, monkeypatch, shape, downsample, extra=()):
    """Run synthesis main() and return what it saved."""
    output_name = f"out_{downsample}_{len(extra)}.pkl"
    argv = [
        "synthesise-spectra",
        "--data-dir", str(tmp_path / "atmosphere"),
        "--output-dir", str(tmp_path / "out"),
        "--output-name", output_name,
        "--lines", LINE,
        "--cube-shape", *[str(n) for n in shape],
        "--voxel-dx", str(VOXEL["dx"]),
        "--voxel-dy", str(VOXEL["dy"]),
        "--voxel-dz", str(VOXEL["dz"]),
        "--mean-mol-wt", str(MEAN_MOL_WT),
        "--downsample", str(downsample),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    synthesis.main()
    with open(tmp_path / "out" / output_name, "rb") as f:
        return dill.load(f)


def _cdelt(saved):
    """(x, y) pixel size of the line cube in Mm; its axes are (y, x, lambda)."""
    wcs = saved["line_cubes"][LINE].wcs.wcs
    assert list(wcs.ctype)[1:] == ["SOLX", "SOLY"]
    return wcs.cdelt[1], wcs.cdelt[2]


def _intensity(saved):
    """Wavelength-summed intensity of every pixel."""
    return saved["line_cubes"][LINE].data.sum(axis=-1)


def test_load_cube_leaves_the_callers_voxel_sizes_alone(tmp_path):
    """Each call scales its own copy, so repeated calls agree."""
    nx, nz, ny = 4, 4, 4
    path = tmp_path / "eosT.0270000"
    _write_cube(path, (nx, nz, ny), 1.0)
    dx, dy, dz = VOXEL["dx"].copy(), VOXEL["dy"].copy(), VOXEL["dz"].copy()

    cdelts = []
    for _ in range(3):
        cube = load_cube(path, shape=(nx, nz, ny), unit=u.K, downsample=2,
                         voxel_dx=dx, voxel_dy=dy, voxel_dz=dz,
                         create_ndcube=True)
        cdelts.append(list(cube.wcs.wcs.cdelt))

    assert [dx, dy, dz] == [0.1 * u.Mm, 0.15 * u.Mm, 0.05 * u.Mm]
    for cdelt in cdelts:
        assert cdelt == pytest.approx([0.2, 0.3, 0.1])


@pytest.mark.parametrize("downsample", [2, 4])
def test_downsampling_keeps_the_field_of_view_and_the_intensity(
        tmp_path, monkeypatch, downsample):
    shape = (8, 8, 8)
    _write_atmosphere(tmp_path / "atmosphere", shape)

    full = _synthesise(tmp_path, monkeypatch, shape, 1)
    reduced = _synthesise(tmp_path, monkeypatch, shape, downsample)

    full_dx, full_dy = _cdelt(full)
    dx, dy = _cdelt(reduced)
    assert dx == pytest.approx(downsample * full_dx)
    assert dy == pytest.approx(downsample * full_dy)

    # The same field of view, in fewer, larger pixels.
    full_ny, full_nx = _intensity(full).shape
    ny, nx = _intensity(reduced).shape
    assert nx * dx == pytest.approx(full_nx * full_dx)
    assert ny * dy == pytest.approx(full_ny * full_dy)

    # A uniform column holds the same emission measure however finely it is
    # sampled, so its intensity cannot depend on the downsampling.
    assert np.all(_intensity(full) > 0)
    assert _intensity(reduced) == pytest.approx(
        np.full((ny, nx), _intensity(full).mean()), rel=1e-6)

    assert reduced["voxel_sizes"]["dx"] == downsample * VOXEL["dx"]
    assert reduced["voxel_sizes"]["dy"] == downsample * VOXEL["dy"]
    assert reduced["voxel_sizes"]["dz"] == downsample * VOXEL["dz"]


def test_a_crop_selects_the_same_world_box_when_downsampled(tmp_path,
                                                            monkeypatch):
    """Crops are in world coordinates, so they rely on the WCS being right."""
    shape = (16, 8, 8)
    _write_atmosphere(tmp_path / "atmosphere", shape)
    crop = ["--crop-x", "-0.4 Mm", "0.4 Mm"]

    full = _synthesise(tmp_path, monkeypatch, shape, 1, crop)
    reduced = _synthesise(tmp_path, monkeypatch, shape, 2, crop)

    # A world box a fixed width wide spans about that width at any pixel size,
    # so what shows a wrong WCS is how many simulation cells it keeps. Each
    # downsampled cell stands for two, and striding moves the cell edges, so
    # allow one downsampled cell either way.
    full_cells = _intensity(full).shape[1]
    cells = _intensity(reduced).shape[1]
    assert full_cells < 16
    assert abs(2 * cells - full_cells) <= 2


def _write_dynamic_atmosphere(root, shape):
    """Two snapshots, with the header files dynamic mode reads times from."""
    for suffix, time in [("0270000", 0.0), ("0280000", 1000.0)]:
        _write_atmosphere(root, shape, suffix)
        header = root / "header" / f"Header.{suffix}"
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text(f"8 8 8 1e7 1.5e7 5e6 {time} 0.1 1e6\n")


DYNAMIC = ["--slit-rest-time", "40 s", "--slit-width", "0.2 arcsec"]


def test_dynamic_mode_downsamples_the_same_way(tmp_path, monkeypatch):
    shape = (8, 8, 8)
    _write_dynamic_atmosphere(tmp_path / "atmosphere", shape)

    full = _synthesise(tmp_path, monkeypatch, shape, 1, DYNAMIC)
    reduced = _synthesise(tmp_path, monkeypatch, shape, 2, DYNAMIC)

    assert _cdelt(reduced)[0] == pytest.approx(2 * _cdelt(full)[0])
    assert _intensity(reduced).shape[1] * _cdelt(reduced)[0] == pytest.approx(
        _intensity(full).shape[1] * _cdelt(full)[0])
    assert _intensity(reduced) == pytest.approx(
        np.full(_intensity(reduced).shape, _intensity(full).mean()), rel=1e-6)


def test_load_cube_refuses_a_factor_that_does_not_divide_the_cube(tmp_path):
    """9 cells by 2 would keep 5 cells of 2 voxels each, a domain of 10."""
    path = tmp_path / "eosT.0270000"
    _write_cube(path, (9, 8, 8), 1.0)
    with pytest.raises(ValueError, match="does not divide the cube shape"):
        load_cube(path, shape=(9, 8, 8), unit=u.K, downsample=2,
                  voxel_dx=VOXEL["dx"], voxel_dy=VOXEL["dy"],
                  voxel_dz=VOXEL["dz"], create_ndcube=True)


@pytest.mark.parametrize("mode", [[], DYNAMIC], ids=["static", "dynamic"])
def test_synthesis_refuses_a_factor_that_does_not_divide_the_cube(
        tmp_path, monkeypatch, mode):
    """In either mode, rather than a domain and path length that are too large."""
    shape = (8, 8, 9)
    _write_dynamic_atmosphere(tmp_path / "atmosphere", shape)
    with pytest.raises(ValueError, match="does not divide the cube shape"):
        _synthesise(tmp_path, monkeypatch, shape, 2, mode)
