"""A slit observing a time series sees each strip of the box at the moment it is exposed.

The observing plan lives in the instrument configuration, the atmosphere is
a series of files each carrying its time, and each exposure is synthesised
from the columns under the slit and the snapshots that overlap it. These
build small series in which the temperature, density and velocity of each
snapshot are known, so what an exposure should collect can be worked out by
hand, and run the synthesis with a flat contribution function in place of
fiasco, so the intensity of a column is its emission measure.
"""
import subprocess
import sys

import astropy.constants as const
import astropy.units as u
import dill
import numpy as np
import pytest
import yaml

from euvst_response import raster as raster_module
from euvst_response.atmosphere import Atmosphere, write_atmosphere
from euvst_response.raster import (
    AtmosphereSeries,
    RasterPlan,
    RasterSynthesiser,
    SynthesisSettings,
)
from euvst_response.utils import angle_to_distance

LINE = "Fe12_195.1190"
REST = 195.119 * u.Angstrom
TEMPERATURE = 1.0e6 * u.K
ELECTRON_DENSITY = 1.0e9 / u.cm**3
SHAPE = (4, 6, 12)  # (nz, ny, nx)
# Cells 0.2 arcsec wide, so a 0.4 arcsec slit covers two of them exactly.
CELL = angle_to_distance(0.2 * u.arcsec).to(u.Mm)


def _edges(shape=SHAPE):
    nz, ny, nx = shape
    return {"x_edges": (np.arange(nx + 1) - nx / 2) * CELL,
            "y_edges": (np.arange(ny + 1) - ny / 2) * CELL,
            "z_edges": np.arange(nz + 1) * 0.1 * u.Mm}


def _snapshot(time, density_scale=1.0, velocity=0.0, columns=None):
    """A uniform box at *time*; *columns* maps x indices to their own (density scale, velocity in km/s)."""
    nz, ny, nx = SHAPE
    density = np.full(SHAPE, ELECTRON_DENSITY.value * density_scale)
    vz = np.full(SHAPE, velocity)
    for column, (scale, speed) in (columns or {}).items():
        density[:, :, column] = ELECTRON_DENSITY.value * scale
        vz[:, :, column] = speed
    return Atmosphere(temperature=np.full(SHAPE, TEMPERATURE.value) * u.K,
                      electron_density=density / u.cm**3,
                      velocity_z=vz * u.km / u.s, time=time * u.s, **_edges())


def _series(tmp_path, snapshots):
    paths = []
    for index, atmosphere in enumerate(snapshots):
        paths.append(write_atmosphere(atmosphere, tmp_path / f"snap_{index}.h5"))
    return paths


GOFT = 1.0e-24  # erg cm^3 / s, the order of a strong coronal line's peak


def _flat_goft(lines, **kwargs):
    """A contribution function that is the same everywhere, in place of fiasco.

    Flat in temperature and density, so a column's intensity is its emission
    measure; of a realistic size, so the instrument run's photon counts are
    ones a detector can hold."""
    logT_grid = np.linspace(5.0, 7.0, 21)
    logN_grid = np.linspace(8.0, 10.0, 21)
    goft = {name: {"wl0": REST.to(u.cm), "g_tn": np.full((logN_grid.size, logT_grid.size), GOFT),
                   "atom": 26, "ion": 12, "hdf5_dbase_root": None} for name in lines}
    return goft, logT_grid, logN_grid


@pytest.fixture
def flat_goft(monkeypatch):
    monkeypatch.setattr(raster_module, "compute_goft_fiasco", _flat_goft)


def _settings(**overrides):
    return SynthesisSettings(lines=(LINE,), **overrides)


def _centroid_velocity(spectrum, wl_grid):
    """Velocity of a spectrum's intensity-weighted centre, in km/s, positive away."""
    wl = wl_grid.to_value(u.Angstrom)
    centre = (spectrum * wl).sum(-1) / spectrum.sum(-1)
    return (centre - REST.value) / REST.value * const.c.to_value(u.km / u.s)


# ----------------------------------------------------------------------
# The series
# ----------------------------------------------------------------------
def test_a_series_orders_its_files_by_time_and_needs_one_in_each(tmp_path):
    paths = _series(tmp_path, [_snapshot(20.0), _snapshot(0.0), _snapshot(10.0)])
    series = AtmosphereSeries(paths)
    assert list(series.times.to_value(u.s)) == [0.0, 10.0, 20.0]
    assert [p.name for p in series.paths] == ["snap_1.h5", "snap_2.h5", "snap_0.h5"]
    assert np.allclose(series.valid_until().to_value(u.s), [10.0, 20.0, 30.0])

    untimed = write_atmosphere(Atmosphere(
        temperature=np.full(SHAPE, TEMPERATURE.value) * u.K,
        electron_density=np.full(SHAPE, ELECTRON_DENSITY.value) / u.cm**3,
        velocity_z=np.zeros(SHAPE) * u.km / u.s, **_edges()), tmp_path / "untimed.h5")
    with pytest.raises(ValueError, match="records no time"):
        AtmosphereSeries(paths + [untimed])
    with pytest.raises(ValueError, match="same time"):
        AtmosphereSeries(paths + [write_atmosphere(_snapshot(10.0), tmp_path / "twin.h5")])
    with pytest.raises(ValueError, match="at least two"):
        AtmosphereSeries(paths[:1]).valid_until()


def test_an_exposure_is_shared_between_the_snapshots_it_spans(tmp_path):
    series = AtmosphereSeries(_series(tmp_path, [_snapshot(0.0), _snapshot(10.0), _snapshot(30.0)]))
    # Within one snapshot's span.
    assert series.coverage(2 * u.s, 8 * u.s) == [(0, 1.0)]
    # A quarter in the first, three quarters in the second.
    assert series.coverage(5 * u.s, 25 * u.s) == pytest.approx([(0, 0.25), (1, 0.75)])
    # The last snapshot stands for as long as the gap before it.
    assert series.coverage(40 * u.s, 50 * u.s) == [(2, 1.0)]
    for start, end in ((-1, 5), (45, 51), (5, 5)):
        with pytest.raises(ValueError):
            series.coverage(start * u.s, end * u.s)


# ----------------------------------------------------------------------
# The plan
# ----------------------------------------------------------------------
def test_a_plan_places_its_exposures_in_space_and_time():
    plan = RasterPlan(start=100 * u.s, steps=3, repeats=2, cadence=5 * u.s)
    exposures = plan.exposures(0.4 * u.arcsec, 2 * u.s, atmosphere_centre=0 * u.Mm)
    assert len(exposures) == 6
    step = angle_to_distance(0.4 * u.arcsec).to_value(u.Mm)
    assert [e.position.to_value(u.Mm) for e in exposures] == pytest.approx(
        [-step, 0.0, step, -step, 0.0, step])
    assert [e.start.to_value(u.s) for e in exposures] == pytest.approx([100, 105, 110, 115, 120, 125])
    assert all((e.end - e.start).to_value(u.s) == pytest.approx(2.0) for e in exposures)

    # The step and the cadence default to the slit width and the exposure.
    stare = RasterPlan(start=0 * u.s, repeats=3)
    exposures = stare.exposures(0.4 * u.arcsec, 10 * u.s, atmosphere_centre=1 * u.Mm)
    assert [e.position.to_value(u.Mm) for e in exposures] == pytest.approx([1.0, 1.0, 1.0])
    assert [e.start.to_value(u.s) for e in exposures] == pytest.approx([0, 10, 20])

    with pytest.raises(ValueError, match="overlap"):
        RasterPlan(start=0 * u.s, cadence=1 * u.s).exposures(0.4 * u.arcsec, 2 * u.s, 0 * u.Mm)
    for bad in (dict(steps=0), dict(repeats=-1), dict(step=-1 * u.arcsec),
                dict(cadence=0 * u.s), dict(centre=3 * u.s), dict(start=5 * u.Mm)):
        with pytest.raises(ValueError):
            RasterPlan(**{"start": 0 * u.s, **bad})


# ----------------------------------------------------------------------
# Synthesising what the slit sees
# ----------------------------------------------------------------------
def test_the_slit_covers_the_columns_under_it(tmp_path, flat_goft):
    series = AtmosphereSeries(_series(tmp_path, [_snapshot(0.0), _snapshot(10.0)]))
    synthesiser = RasterSynthesiser(series, _settings())
    # A slit two cells wide centred on the boundary between cells 6 and 7.
    first, last, fractions = synthesiser.columns_under(CELL, 0.4 * u.arcsec)
    assert (first, last) == (6, 8)
    assert fractions == pytest.approx([0.5, 0.5])
    # Centred on cell 6, it covers half of 5, all of 6 and half of 7.
    first, last, fractions = synthesiser.columns_under(0.5 * CELL, 0.4 * u.arcsec)
    assert (first, last) == (5, 8)
    assert fractions == pytest.approx([0.25, 0.5, 0.25])
    with pytest.raises(ValueError, match="outside the atmosphere"):
        synthesiser.columns_under(10 * CELL, 0.4 * u.arcsec)


def test_an_exposure_averages_over_the_slit_and_over_time(tmp_path, flat_goft):
    """Two columns of different brightness, seen across two snapshots of different brightness."""
    first = _snapshot(0.0, columns={6: (1.0, 0.0), 7: (2.0, 0.0)})
    second = _snapshot(10.0, density_scale=3.0)
    series = AtmosphereSeries(_series(tmp_path, [first, second]))
    synthesiser = RasterSynthesiser(series, _settings())
    wl_grid = None

    # Whole exposure in the first snapshot: the mean of the two columns'
    # emission measures, which go as the density squared.
    exposure = raster_module.Exposure(0, CELL, 2 * u.s, 8 * u.s)
    spectra = synthesiser.exposure_spectra(exposure, 0.4 * u.arcsec)[LINE]
    col6 = synthesiser.column_spectra(0, 6)[LINE]
    col7 = synthesiser.column_spectra(0, 7)[LINE]
    assert col7.sum() == pytest.approx(4 * col6.sum(), rel=1e-6)
    assert spectra == pytest.approx(0.5 * col6 + 0.5 * col7, rel=1e-6)

    # Half the exposure in each snapshot: the second is nine times brighter.
    exposure = raster_module.Exposure(0, CELL, 5 * u.s, 15 * u.s)
    spectra = synthesiser.exposure_spectra(exposure, 0.4 * u.arcsec)[LINE]
    later = synthesiser.column_spectra(1, 6)[LINE]
    assert later.sum() == pytest.approx(9 * col6.sum(), rel=1e-6)
    assert spectra == pytest.approx(0.5 * (0.5 * col6 + 0.5 * col7) + 0.5 * later, rel=1e-6)
    # Each column of each snapshot was synthesised once.
    assert synthesiser.strips_synthesised == 2


def test_a_flow_seen_from_above_is_blueshifted_in_the_exposure(tmp_path, flat_goft):
    rising = _snapshot(0.0, velocity=20.0)
    series = AtmosphereSeries(_series(tmp_path, [rising, _snapshot(10.0, velocity=20.0)]))
    synthesiser = RasterSynthesiser(series, _settings())
    exposure = raster_module.Exposure(0, 0 * u.Mm, 1 * u.s, 3 * u.s)
    spectrum = synthesiser.exposure_spectra(exposure, 0.4 * u.arcsec)[LINE]
    velocity = _centroid_velocity(spectrum, synthesiser._wl_grids[LINE])
    assert velocity == pytest.approx(-20.0, abs=0.1)


def test_the_raster_cube_has_one_column_per_exposure_at_its_position(tmp_path, flat_goft):
    snapshots = [_snapshot(t, density_scale=1.0 + t / 10) for t in (0.0, 10.0, 20.0, 30.0)]
    series = AtmosphereSeries(_series(tmp_path, snapshots))
    synthesiser = RasterSynthesiser(series, _settings())
    plan = RasterPlan(start=0 * u.s, steps=3, repeats=1)

    cubes = synthesiser.line_cubes(plan, 0.4 * u.arcsec, 10 * u.s)
    cube = cubes[LINE]
    assert cube.data.shape == (SHAPE[1], 3, synthesiser.vel_grid.size)
    step = angle_to_distance(0.4 * u.arcsec).to_value(u.Mm)
    x = cube.axis_world_coords(1)[0].to_value(u.Mm)
    assert x == pytest.approx([-step, 0.0, step])
    assert cube.meta["raster"] is True
    assert cube.meta["starts"].to_value(u.s) == pytest.approx([0.0, 10.0, 20.0])
    assert cube.meta["velocity_convention"]
    # Each exposure fell in one snapshot, and the density grew with time, so
    # the columns brighten along the raster as the square of the density.
    columns = cube.data.sum(axis=(0, 2))
    assert columns / columns[0] == pytest.approx([1.0, 4.0, 9.0], rel=1e-6)

    summed = synthesiser.summed_cube(plan, 0.4 * u.arcsec, 10 * u.s, LINE)
    assert summed.data == pytest.approx(cube.data)
    assert summed.meta["combined_lines"] == [LINE]

    stare = synthesiser.line_cubes(RasterPlan(start=0 * u.s, repeats=2), 0.4 * u.arcsec, 10 * u.s)[LINE]
    assert stare.data.shape[1] == 2
    assert stare.wcs.wcs.cdelt[1] == pytest.approx(step)
    assert stare.meta["positions"].to_value(u.Mm) == pytest.approx([0.0, 0.0])

    with pytest.raises(ValueError, match="outside the series"):
        synthesiser.line_cubes(RasterPlan(start=35 * u.s), 0.4 * u.arcsec, 10 * u.s)
    with pytest.raises(ValueError, match="not among the synthesised lines"):
        synthesiser.summed_cube(plan, 0.4 * u.arcsec, 10 * u.s, "Fe09_171.0730")


# ----------------------------------------------------------------------
# Through the instrument run
# ----------------------------------------------------------------------
def _config(tmp_path, series_glob, **extra):
    config = {
        "instrument": "SWC",
        "atmosphere_series": series_glob,
        "reference_line": LINE,
        "n_iter": 2,
        "ncpu": 1,
        "fit_signals": "dn",
        "synthesis": {"lines": [LINE]},
        "raster": {"start": "0 s", "steps": 2},
        "simulation": {"slit_width": "0.4 arcsec", "expos": ["5 s", "20 s"], "psf": False},
        **extra,
    }
    path = tmp_path / "series.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_an_instrument_run_sweeps_the_exposure_over_a_series(tmp_path, monkeypatch, flat_goft):
    snapshots = [_snapshot(t, density_scale=1.0 + t / 10) for t in (0.0, 10.0, 20.0, 30.0)]
    _series(tmp_path / "series", snapshots)
    config = _config(tmp_path, str(tmp_path / "series" / "*.h5"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    from euvst_response.main import main as run_main
    run_main()

    with open(tmp_path / "run" / "result" / "series.pkl", "rb") as f:
        saved = dill.load(f)
    assert len(saved["results"]["all_combinations"]) == 2
    raster = saved["raster"]
    assert raster["plan"].steps == 2
    assert [str(p) for p in raster["series"]] == sorted(str(p) for p in raster["series"])
    assert raster["times"].to_value(u.s) == pytest.approx([0.0, 10.0, 20.0, 30.0])
    keys = sorted(raster["cubes"])
    assert [k[1] for k in keys] == pytest.approx([5.0, 20.0])
    # A 20 s exposure from t = 0 spans the first two snapshots; a 5 s one only
    # the first, so the first column differs between the two.
    short, long = (raster["cubes"][k].data.sum(axis=(0, 2)) for k in keys)
    assert long[0] > short[0]
    assert saved["cube_sim"].meta["raster"] is True


def test_a_series_run_refuses_what_it_cannot_do(tmp_path, monkeypatch, flat_goft):
    _series(tmp_path / "series", [_snapshot(0.0), _snapshot(10.0)])
    glob_pattern = str(tmp_path / "series" / "*.h5")
    monkeypatch.chdir(tmp_path)
    from euvst_response.main import main as run_main

    def run(**extra):
        config = _config(tmp_path, glob_pattern, **extra)
        monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
        run_main()

    with pytest.raises(ValueError, match="not both"):
        run(synthesis_file="x.pkl")
    with pytest.raises(ValueError, match="'raster.start' is needed"):
        run(raster={"steps": 2})
    with pytest.raises(ValueError, match="Unrecognised 'raster' section"):
        run(raster={"start": "0 s", "cadance": "5 s"})
    with pytest.raises(ValueError, match="not one of 'synthesis.lines'"):
        run(reference_line="Fe09_171.0730")
    with pytest.raises(FileNotFoundError, match="matches no file"):
        run(atmosphere_series=str(tmp_path / "nowhere" / "*.h5"))
    with pytest.raises(ValueError, match="belongs to an 'atmosphere_series' run"):
        config = tmp_path / "plain.yaml"
        config.write_text(yaml.safe_dump({"instrument": "SWC", "uniform_intensity": "100 erg / (s cm2 sr)",
                                          "raster": {"start": "0 s"}}))
        monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
        run_main()


def test_an_old_dynamic_mode_synthesis_file_is_refused(tmp_path, monkeypatch):
    from euvst_response.main import main as run_main
    from euvst_response.synthesis import create_atmosphere_ndcube
    from euvst_response.atmosphere import Atmosphere as _A  # noqa: F401
    line = {"si": np.zeros((4, 4, 3)), "wl_grid": (REST + np.array([-0.1, 0.0, 0.1]) * u.Angstrom).to(u.cm),
            "wl0": REST.to(u.cm), "atom": 26, "ion": 12}
    from euvst_response.synthesis import create_line_cube
    reference = create_atmosphere_ndcube(np.zeros((2, 4, 4)) * u.K, voxel_dx=1 * u.Mm,
                                         voxel_dy=1 * u.Mm, voxel_dz=1 * u.Mm)
    cube = create_line_cube(LINE, line, reference, u.erg / u.s / u.cm**2 / u.sr / u.cm, "z")
    path = tmp_path / "old_dynamic.pkl"
    with open(path, "wb") as f:
        dill.dump({"line_cubes": {LINE: cube}, "dynamic_mode": {"enabled": True}}, f)
    config = tmp_path / "old.yaml"
    config.write_text(yaml.safe_dump({"instrument": "SWC", "synthesis_file": str(path),
                                      "reference_line": LINE}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    with pytest.raises(ValueError, match="old dynamic mode"):
        run_main()
