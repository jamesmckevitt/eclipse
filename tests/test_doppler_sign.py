"""A simulated flow has the Doppler shift the observer of the synthesised map sees.

Simulation velocities are positive towards increasing coordinate, so an upflow
has a positive vz. Seen from above, it moves towards the observer and is
blueshifted. ECLIPSE 0.8.0 and earlier used the velocity along the integration
axis directly as the line-of-sight velocity. For views along z and x that gave
every velocity the wrong sign, so upflows seen from above came out redshifted;
for views along y, whose observer is at -y, it happened to be right.

These run the real synthesis on small uniform atmospheres, with a flat
contribution function standing in for fiasco, and measure where each line
sits in wavelength.
"""
import sys
import warnings

import astropy.constants as const
import astropy.units as u
import dill
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response import synthesis
from euvst_response.analysis import load_instrument_response_results
from euvst_response.data_processing import load_atmosphere
from euvst_response.utils import VELOCITY_CONVENTION

LINE = "Fe12_195.1190"
REST = 195.119 * u.Angstrom
SHAPE = (8, 6, 4)  # the files' own (nx, nz, ny) layout
FLOW = 20.0 * u.km / u.s
MEAN_MOL_WT = 1.29
VELOCITY_FILES = {"x": "vx/result_prim_1", "y": "vy/result_prim_3",
                  "z": "vz/result_prim_2"}
UNIT_VECTORS = {"SOLX": (1, 0, 0), "SOLY": (0, 1, 0), "SOLZ": (0, 0, 1)}
DYNAMIC = ["--slit-rest-time", "40 s", "--slit-width", "0.2 arcsec"]


def _write_cube(path, data):
    """A cube in the files' own (nx, nz, ny) Fortran layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(data, dtype=np.float32).ravel(order="F").tofile(path)


def _write_atmosphere(root, velocity, suffix="0270000", time=None):
    """Uniform coronal plasma, with *velocity* mapping an axis to its cube."""
    density = (1.0e9 / u.cm**3 * MEAN_MOL_WT * const.u).to_value(u.g / u.cm**3)
    _write_cube(root / "temp" / f"eosT.{suffix}", np.full(SHAPE, 1.0e6))
    _write_cube(root / "rho" / f"result_prim_0.{suffix}", np.full(SHAPE, density))
    for axis, name in VELOCITY_FILES.items():
        _write_cube(root / f"{name}.{suffix}",
                    velocity.get(axis, np.zeros(SHAPE)))
    if time is not None:
        header = root / "header" / f"Header.{suffix}"
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text(f"8 6 4 1e7 1.5e7 5e6 {time} 0.1 1e6\n")


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


def _synthesise(tmp_path, monkeypatch, axis, extra=()):
    """Run synthesis main() and return the path it saved to."""
    argv = [
        "synthesise-spectra",
        "--data-dir", str(tmp_path / "atmosphere"),
        "--output-dir", str(tmp_path / "out"),
        "--output-name", f"{axis}.pkl",
        "--lines", LINE,
        "--cube-shape", *[str(n) for n in SHAPE],
        "--voxel-dx", "0.1 Mm", "--voxel-dy", "0.15 Mm", "--voxel-dz", "0.05 Mm",
        "--mean-mol-wt", str(MEAN_MOL_WT),
        "--integration-axis", axis,
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    synthesis.main()
    return tmp_path / "out" / f"{axis}.pkl"


def _load(path):
    with open(path, "rb") as f:
        return dill.load(f)


def _doppler_velocity(cube):
    """Line-of-sight velocity of every pixel, from its mean wavelength."""
    wcs = cube.wcs.wcs
    pixels = np.arange(cube.data.shape[-1]) + 1
    wavelength = ((wcs.crval[0] + (pixels - wcs.crpix[0]) * wcs.cdelt[0])
                  * u.Unit(wcs.cunit[0]))
    mean = (cube.data * wavelength.value).sum(-1) / cube.data.sum(-1)
    return ((mean * wavelength.unit - REST) / REST * const.c).to_value(u.km / u.s)


def test_an_upflow_seen_from_above_is_blueshifted(tmp_path, monkeypatch):
    """Half the box rises and half sinks; the top-down map has rows y, columns x."""
    vz = np.zeros(SHAPE)
    half = SHAPE[0] // 2
    vz[:half] = FLOW.to_value(u.cm / u.s)
    vz[half:] = -FLOW.to_value(u.cm / u.s)
    _write_atmosphere(tmp_path / "atmosphere", {"z": vz})

    saved = _load(_synthesise(tmp_path, monkeypatch, "z"))

    velocity = _doppler_velocity(saved["line_cubes"][LINE])
    assert velocity[:, :half] == pytest.approx(-FLOW.value, abs=0.1)
    assert velocity[:, half:] == pytest.approx(FLOW.value, abs=0.1)

    # The EM(T,v) cube carries the same line-of-sight velocity.
    vel_grid = saved["vel_grid"].to_value(u.km / u.s)
    rising = saved["em_tv"][0, 0].sum(axis=0)
    sinking = saved["em_tv"][0, -1].sum(axis=0)
    assert vel_grid[np.argmax(rising)] == pytest.approx(-FLOW.value)
    assert vel_grid[np.argmax(sinking)] == pytest.approx(FLOW.value)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_the_observer_is_where_the_map_is_the_right_way_round(
        tmp_path, monkeypatch, axis):
    """A flow along the line of sight is blueshifted when it moves towards that observer.

    The line cube's columns run left to right and its rows bottom to top, so
    the column axis crossed with the row axis points at the observer who sees
    the map that way round. The expected shift is worked out from the cube's
    own WCS, not from ECLIPSE's table of observers.
    """
    flow_vector = FLOW.value * np.array(UNIT_VECTORS[f"SOL{axis.upper()}"])
    _write_atmosphere(tmp_path / "atmosphere",
                      {axis: np.full(SHAPE, FLOW.to_value(u.cm / u.s))})

    cube = _load(_synthesise(tmp_path, monkeypatch, axis))["line_cubes"][LINE]

    ctype = list(cube.wcs.wcs.ctype)
    columns, rows = (np.array(UNIT_VECTORS[c]) for c in ctype[1:])
    towards_observer = np.cross(columns, rows)
    away_from_observer = -np.dot(flow_vector, towards_observer)
    assert away_from_observer != 0
    assert _doppler_velocity(cube) == pytest.approx(away_from_observer, abs=0.1)


def test_dynamic_mode_has_the_same_sign(tmp_path, monkeypatch):
    for suffix, time in [("0270000", 0.0), ("0280000", 1000.0)]:
        _write_atmosphere(tmp_path / "atmosphere",
                          {"z": np.full(SHAPE, FLOW.to_value(u.cm / u.s))},
                          suffix=suffix, time=time)

    saved = _load(_synthesise(tmp_path, monkeypatch, "z", DYNAMIC))

    assert saved["dynamic_mode"]["enabled"]
    assert _doppler_velocity(saved["line_cubes"][LINE]) == pytest.approx(
        -FLOW.value, abs=0.1)


@pytest.mark.parametrize("axis, refused", [("x", True), ("y", False), ("z", True)])
def test_synthesis_files_from_before_the_fix_are_refused_where_their_sign_is_wrong(
        tmp_path, monkeypatch, axis, refused):
    """Views along y kept their sign, so older ones are still right and still load."""
    _write_atmosphere(tmp_path / "atmosphere", {})
    path = _synthesise(tmp_path, monkeypatch, axis)
    cube, _ = load_atmosphere(str(path))
    assert cube.meta["velocity_convention"] == VELOCITY_CONVENTION

    saved = _load(path)
    del saved["line_cubes"][LINE].meta["velocity_convention"]
    old = tmp_path / "old.pkl"
    with open(old, "wb") as f:
        dill.dump(saved, f)
    if refused:
        with pytest.raises(ValueError, match=f"view along {axis}.*wrong sign"):
            load_atmosphere(str(old))
    else:
        load_atmosphere(str(old))


def _write_results_file(path, meta):
    """A results file whose atmosphere cube carries *meta*; None means uniform intensity mode."""
    cube_sim = None if meta is None else NDCube(np.ones((2, 2, 3)),
                                                wcs=WCS(naxis=3), meta=meta)
    with open(path, "wb") as f:
        dill.dump({"results": {"all_combinations": {}}, "cube_sim": cube_sim}, f)
    return path


@pytest.mark.parametrize("meta", [
    {"integration_axis": "z"},
    {"integration_axis": "x"},
    {},  # written before the side views existed, so a view along z
], ids=["z", "x", "no axis"])
def test_results_from_before_the_fix_are_refused_unless_asked_for(tmp_path, meta):
    old = _write_results_file(tmp_path / "old.pkl", meta)
    with pytest.raises(ValueError, match="wrong sign"):
        load_instrument_response_results(old)
    with pytest.warns(UserWarning, match="wrong sign"):
        load_instrument_response_results(old, allow_wrong_velocity_sign=True)


@pytest.mark.parametrize("meta", [
    {"velocity_convention": VELOCITY_CONVENTION, "integration_axis": "z"},
    {"integration_axis": "y"},
    None,  # uniform intensity mode, with no synthesis file
], ids=["current", "older view along y", "uniform intensity"])
def test_results_whose_velocities_are_right_load_without_a_warning(tmp_path, meta):
    path = _write_results_file(tmp_path / "results.pkl", meta)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_instrument_response_results(path)
