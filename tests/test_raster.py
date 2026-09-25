"""A slit observing a time series sees each strip of the box at the moment it is exposed.

The observing plan lives in the instrument configuration, the atmosphere is
a series of files each carrying its time, and each exposure is synthesised
from the columns under the slit and the snapshots that overlap it. These
build small series in which the temperature, density and velocity of each
snapshot are known, so what an exposure should collect can be worked out by
hand, and run the synthesis with a flat contribution function in place of
fiasco, so the intensity of a column is its emission measure.

A series of synthesis files is observed the same way, with each column's
spectra read rather than synthesised, whether ECLIPSE or another code wrote
them.
"""
import subprocess
import sys

import astropy.constants as const
import astropy.units as u
import dill
import h5py
import numpy as np
import pytest
import yaml

from euvst_response import raster as raster_module
from euvst_response.atmosphere import Atmosphere, write_atmosphere
from euvst_response.raster import (
    AtmosphereSeries,
    RasterPlan,
    RasterSynthesiser,
    SynthesisRaster,
    SynthesisSeries,
    SynthesisSettings,
)
from euvst_response.synthesis_file import (RADIANCE_UNIT, SpectralLine, Synthesis, load_synthesis,
                                           read_synthesis, write_synthesis)
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
    # A time that is not a finite number would slip through the ordering.
    for bad in (np.nan, np.inf):
        broken = write_atmosphere(_snapshot(30.0), tmp_path / "broken.h5")
        with h5py.File(broken, "r+") as f:
            f["time"][()] = bad
        with pytest.raises(ValueError, match="broken.h5: time contains NaN or infinite"):
            AtmosphereSeries(paths + [broken])
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
    step = angle_to_distance(0.4 * u.arcsec).to_value(u.Mm)
    # Each raster is its own set of exposures; the second follows the first.
    for repeat, starts in ((0, [100, 105, 110]), (1, [115, 120, 125])):
        exposures = plan.exposures(0.4 * u.arcsec, 2 * u.s, atmosphere_centre=0 * u.Mm,
                                   repeat=repeat)
        assert [e.position.to_value(u.Mm) for e in exposures] == pytest.approx([-step, 0.0, step])
        assert [e.start.to_value(u.s) for e in exposures] == pytest.approx(starts)
        assert all((e.end - e.start).to_value(u.s) == pytest.approx(2.0) for e in exposures)
        assert [e.index for e in exposures] == [3 * repeat + i for i in range(3)]
    with pytest.raises(ValueError, match="say which one"):
        plan.exposures(0.4 * u.arcsec, 2 * u.s, 0 * u.Mm)
    with pytest.raises(ValueError, match="between 0 and 1"):
        plan.exposures(0.4 * u.arcsec, 2 * u.s, 0 * u.Mm, repeat=2)
    for bad in (0.5, True):
        with pytest.raises(ValueError, match="whole number"):
            plan.exposures(0.4 * u.arcsec, 2 * u.s, 0 * u.Mm, repeat=bad)

    # The step and the cadence default to the slit width and the exposure,
    # and a sit-and-stare's exposures are rasters of one position.
    stare = RasterPlan(start=0 * u.s, repeats=3)
    for repeat in range(3):
        exposures = stare.exposures(0.4 * u.arcsec, 10 * u.s, atmosphere_centre=1 * u.Mm,
                                    repeat=repeat)
        assert [e.position.to_value(u.Mm) for e in exposures] == pytest.approx([1.0])
        assert [e.start.to_value(u.s) for e in exposures] == pytest.approx([10 * repeat])

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


def test_a_wider_slit_synthesises_only_the_columns_a_narrower_one_has_not(tmp_path, flat_goft, monkeypatch):
    series = AtmosphereSeries(_series(tmp_path, [_snapshot(0.0), _snapshot(10.0)]))
    synthesiser = RasterSynthesiser(series, _settings())
    strips = []
    synthesise_strip = synthesiser._synthesise_strip

    def recording(snapshot, first, last):
        strips.append((snapshot, first, last))
        synthesise_strip(snapshot, first, last)

    monkeypatch.setattr(synthesiser, "_synthesise_strip", recording)
    # Centred on cell 6, a 0.2 arcsec slit covers it alone, and a 0.4 arcsec
    # one half of each neighbour as well.
    exposure = raster_module.Exposure(0, 0.5 * CELL, 2 * u.s, 8 * u.s)
    synthesiser.exposure_spectra(exposure, 0.2 * u.arcsec)
    synthesiser.exposure_spectra(exposure, 0.4 * u.arcsec)
    assert strips == [(0, 6, 7), (0, 5, 6), (0, 7, 8)]


def test_a_flow_seen_from_above_is_blueshifted_in_the_exposure(tmp_path, flat_goft):
    rising = _snapshot(0.0, velocity=20.0)
    series = AtmosphereSeries(_series(tmp_path, [rising, _snapshot(10.0, velocity=20.0)]))
    synthesiser = RasterSynthesiser(series, _settings())
    exposure = raster_module.Exposure(0, 0 * u.Mm, 1 * u.s, 3 * u.s)
    spectrum = synthesiser.exposure_spectra(exposure, 0.4 * u.arcsec)[LINE]
    velocity = _centroid_velocity(spectrum, synthesiser._wl_grids[LINE])
    assert velocity == pytest.approx(-20.0, abs=0.1)


def test_every_snapshot_must_share_the_first_ones_grid(tmp_path, flat_goft):
    """The slit's columns are chosen on the first file's x grid, so the others must match it."""
    shifted = _snapshot(10.0)
    shifted = Atmosphere(**{**{k: getattr(shifted, k) for k in ("temperature", "electron_density",
                                                                "velocity_z", "time")},
                            "x_edges": shifted.x_edges + 0.5 * CELL,
                            "y_edges": shifted.y_edges, "z_edges": shifted.z_edges})
    series = AtmosphereSeries(_series(tmp_path, [_snapshot(0.0), shifted]))
    synthesiser = RasterSynthesiser(series, _settings())
    exposure = raster_module.Exposure(0, 0 * u.Mm, 5 * u.s, 15 * u.s)
    with pytest.raises(ValueError, match="different x grid"):
        synthesiser.exposure_spectra(exposure, 0.4 * u.arcsec)


def test_the_rebinning_keeps_the_raster_where_it_is(tmp_path, flat_goft):
    """Through the instrument's spatial rebinning, the columns stay at their slit positions."""
    from euvst_response.config import Detector_SWC, Simulation
    from euvst_response.data_processing import rebin_atmosphere
    from euvst_response.utils import distance_to_angle

    series = AtmosphereSeries(_series(tmp_path, [_snapshot(0.0), _snapshot(10.0)]))
    synthesiser = RasterSynthesiser(series, _settings())
    plan = RasterPlan(start=0 * u.s, steps=4, centre=0.5 * CELL)
    cube = synthesiser.summed_cube(plan, 0.4 * u.arcsec, 2 * u.s, LINE)
    rebinned = rebin_atmosphere(cube, Detector_SWC(),
                                Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1))

    assert rebinned.data.shape[1] == 4
    expected = distance_to_angle(cube.meta["positions"]).to_value(u.arcsec)
    # The two sky axes are coupled, so the coordinates come as a grid; every
    # row has the same scan positions.
    scan = rebinned.axis_world_coords(1)[0]
    assert scan.Tx.to_value(u.arcsec)[0] == pytest.approx(expected, abs=1e-6)

    # A strip cropped to one row along the slit, 0.2 arcsec, is smaller than
    # an EIS pixel of 1 arcsec, which the rebinning says rather than producing
    # nothing. Two 1 arcsec slit positions keep the raster inside the box.
    from euvst_response.config import Detector_EIS
    one_row = RasterSynthesiser(series, _settings(crop_y=(0 * u.Mm, 0.5 * CELL)))
    narrow = one_row.summed_cube(RasterPlan(start=0 * u.s, steps=2, centre=0.5 * CELL),
                                 1 * u.arcsec, 2 * u.s, LINE)
    assert narrow.data.shape[0] == 1
    with pytest.raises(ValueError, match="smaller than one detector pixel"):
        rebin_atmosphere(narrow, Detector_EIS(),
                         Simulation(instrument="EIS", slit_width=1 * u.arcsec, ncpu=1))


def test_a_static_atmosphere_one_cell_wide_synthesises(tmp_path, monkeypatch):
    """A single column has a size from its edges, so the line cube can still carry a WCS."""
    from euvst_response import synthesis
    column = Atmosphere(temperature=np.full((4, 6, 1), TEMPERATURE.value) * u.K,
                        electron_density=np.full((4, 6, 1), ELECTRON_DENSITY.value) / u.cm**3,
                        velocity_z=np.zeros((4, 6, 1)) * u.km / u.s,
                        x_edges=np.array([-0.5, 0.5]) * CELL, y_edges=(np.arange(7) - 3) * CELL,
                        z_edges=np.arange(5) * 0.1 * u.Mm)
    path = write_atmosphere(column, tmp_path / "column.h5")
    monkeypatch.setattr(sys, "argv", ["synthesise-spectra", "--atmosphere", str(path),
                                      "--lines", LINE, "--output-dir", str(tmp_path / "out"),
                                      "--output-name", "column.h5"])
    monkeypatch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    synthesis.main()
    cube = load_synthesis(tmp_path / "out" / "column.h5")["line_cubes"][LINE]
    assert cube.data.shape[:2] == (6, 1)
    assert cube.wcs.wcs.cdelt[1] == pytest.approx(CELL.value)
    assert cube.axis_world_coords(1)[0].to_value(u.Mm) == pytest.approx([0.0])


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

    # The reference pixel sits at the middle of the raster, with the
    # coordinate the grid has there, like a static line cube's.
    assert cube.wcs.wcs.crpix[1] == pytest.approx(2.0)
    assert cube.wcs.wcs.crval[1] == pytest.approx(0.0)
    assert cube.wcs.wcs.crpix[2] == pytest.approx((SHAPE[1] + 1) / 2)
    y = cube.axis_world_coords(0)[0].to_value(u.Mm)
    assert y == pytest.approx(synthesiser.series.y_edges.to_value(u.Mm)[:-1] + CELL.value / 2)

    # A sit-and-stare of two exposures is two cubes of one column each, at
    # the same position, one exposure apart.
    stare = RasterPlan(start=0 * u.s, repeats=2)
    for repeat in range(2):
        one = synthesiser.line_cubes(stare, 0.4 * u.arcsec, 10 * u.s, repeat=repeat)[LINE]
        assert one.data.shape[1] == 1
        assert one.wcs.wcs.cdelt[1] == pytest.approx(step)
        assert one.axis_world_coords(1)[0].to_value(u.Mm) == pytest.approx([0.0])
        assert one.meta["starts"].to_value(u.s) == pytest.approx([10.0 * repeat])
        assert one.meta["repeat"] == repeat
    with pytest.raises(ValueError, match="say which one"):
        synthesiser.line_cubes(stare, 0.4 * u.arcsec, 10 * u.s)

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
        "raster": {"start": "0 s", "steps": 2, "repeats": 2},
        "simulation": {"slit_width": "0.4 arcsec", "expos": ["5 s", "10 s"], "psf": False},
        **extra,
    }
    path = tmp_path / "series.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_the_contribution_functions_can_be_computed_in_temperature_chunks(tmp_path, monkeypatch):
    """The synthesis: section takes goft_temperature_chunk as synthesise-spectra takes the option."""
    from euvst_response.main import _parse_synthesis_settings
    received = {}

    def recording_goft(lines, **kwargs):
        received.update(kwargs)
        return _flat_goft(lines, **kwargs)

    monkeypatch.setattr(raster_module, "compute_goft_fiasco", recording_goft)
    settings = _parse_synthesis_settings({"synthesis": {"lines": [LINE],
                                                        "goft_temperature_chunk": 10}})
    RasterSynthesiser(AtmosphereSeries(_series(tmp_path, [_snapshot(0.0)])), settings)
    assert received["temperature_chunk"] == 10


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
    # Two exposure times and two rasters: four observations.
    assert len(saved["results"]["all_combinations"]) == 4
    raster = saved["raster"]
    assert raster["plan"].steps == 2
    assert [str(p) for p in raster["series"]] == sorted(str(p) for p in raster["series"])
    assert raster["times"].to_value(u.s) == pytest.approx([0.0, 10.0, 20.0, 30.0])
    keys = sorted(raster["cubes"])
    assert [(k[1], k[2]) for k in keys] == [(5.0, 0), (5.0, 1), (10.0, 0), (10.0, 1)]
    # The second exposure of the 10 s raster, from 10 to 20 s, sees the
    # second snapshot; that of the 5 s raster, from 5 to 10 s, still the
    # first, so the second column is brighter at 10 s.
    short, long = (raster["cubes"][k].data.sum(axis=(0, 2)) for k in (keys[0], keys[2]))
    assert long[1] > short[1]
    assert long[0] == pytest.approx(short[0], rel=1e-6)
    # The second raster of the 5 s sweep starts 10 s in and sees the second snapshot.
    second = raster["cubes"][keys[1]]
    assert second.meta["repeat"] == 1
    assert second.meta["starts"].to_value(u.s) == pytest.approx([10.0, 15.0])
    assert second.data.sum() > raster["cubes"][keys[0]].data.sum()
    assert saved["cube_sim"].meta["raster"] is True

    from euvst_response import get_results_for_combination, load_instrument_response_results
    results = load_instrument_response_results(tmp_path / "run" / "result" / "series.pkl")
    chosen = get_results_for_combination(results, **{
        "simulation.slit_width": 0.4 * u.arcsec, "simulation.expos": 5 * u.s,
        "offchip_bin_slit": 1, "raster.repeat": 1})
    assert chosen["parameters"]["raster.repeat"] == 1


def test_a_series_run_refuses_what_it_cannot_do(tmp_path, monkeypatch, flat_goft):
    _series(tmp_path / "series", [_snapshot(0.0), _snapshot(10.0)])
    glob_pattern = str(tmp_path / "series" / "*.h5")
    monkeypatch.chdir(tmp_path)
    from euvst_response.main import main as run_main

    def run(**extra):
        config = _config(tmp_path, glob_pattern, **extra)
        monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
        run_main()

    with pytest.raises(ValueError, match="only one of"):
        run(synthesis_file="x.pkl")
    with pytest.raises(ValueError, match="'raster.start' is needed"):
        run(raster={"steps": 2})
    with pytest.raises(ValueError, match="Unrecognised 'raster' section"):
        run(raster={"start": "0 s", "cadance": "5 s"})
    with pytest.raises(ValueError, match="not one of 'synthesis.lines'"):
        run(reference_line="Fe09_171.0730")
    with pytest.raises(FileNotFoundError, match="matches no file"):
        run(atmosphere_series=str(tmp_path / "nowhere" / "*.h5"))
    with pytest.raises(ValueError, match="belongs to an 'atmosphere_series' or 'synthesis_series' run"):
        config = tmp_path / "plain.yaml"
        config.write_text(yaml.safe_dump({"instrument": "SWC", "uniform_intensity": "100 erg / (s cm2 sr)",
                                          "raster": {"start": "0 s"}}))
        monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
        run_main()


# ----------------------------------------------------------------------
# A series of synthesis files
# ----------------------------------------------------------------------
# On the detector, the two routes resample from the same wavelengths reached
# through WCSs referenced at different pixels, which differ in the last bit,
# about 4e-22 cm at 195 Angstrom. Against the 5 km/s bins of the synthesis,
# 3e-11 cm, that is about a part in 1e11 of a bin, which is how far apart the
# resampled spectra can then be.
DETECTOR_ROUNDING = 1e-10


def _assert_close(found, expected, rel=1e-12):
    """Equal but for rounding: no element further from *expected* than *rel* of its largest."""
    found, expected = np.asarray(found), np.asarray(expected)
    assert found.shape == expected.shape
    assert np.max(np.abs(found - expected)) <= rel * np.max(np.abs(expected))


def _synthesise(out_dir, atmospheres, monkeypatch):
    """ECLIPSE's synthesis of each atmosphere file, with the flat contribution function."""
    from euvst_response import synthesis
    monkeypatch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    paths = []
    for index, atmosphere in enumerate(atmospheres):
        name = f"synth_{index}.h5"
        monkeypatch.setattr(sys, "argv", ["synthesise-spectra", "--atmosphere", str(atmosphere),
                                          "--lines", LINE, "--output-dir", str(out_dir),
                                          "--output-name", name])
        synthesis.main()
        paths.append(out_dir / name)
    return paths


def _ramp(times=(0.0, 10.0, 20.0, 30.0)):
    """Snapshots that brighten and speed up with time, with two columns of their own."""
    return [_snapshot(t, density_scale=1.0 + t / 10, velocity=t / 2,
                      columns={5: (2.0 + t / 10, -10.0), 8: (0.5, 30.0)}) for t in times]


def test_a_series_of_eclipse_syntheses_is_seen_as_the_atmospheres_they_came_from(tmp_path, monkeypatch, flat_goft):
    """Synthesising each snapshot first and observing the files gives what synthesising under the slit gives."""
    from euvst_response.config import Detector_SWC, Simulation
    from euvst_response.data_processing import rebin_atmosphere, rebin_spectra

    atmospheres = _series(tmp_path / "atmospheres", _ramp())
    syntheses = _synthesise(tmp_path / "syntheses", atmospheres, monkeypatch)
    # The synthesis keeps each snapshot's time, which places it in the series.
    assert read_synthesis(syntheses[2]).time == 20.0 * u.s

    from_atmospheres = RasterSynthesiser(AtmosphereSeries(atmospheres), _settings())
    from_syntheses = SynthesisRaster(SynthesisSeries(syntheses, LINE))
    # Exposures of 7 s from 2 s, some within one snapshot and some across two.
    plan = RasterPlan(start=2 * u.s, steps=4, centre=0.5 * CELL)
    for slit_width in (0.2 * u.arcsec, 0.4 * u.arcsec):
        cube = from_atmospheres.summed_cube(plan, slit_width, 7 * u.s, LINE)
        synthesis, meta = from_syntheses.synthesis(plan, slit_width, 7 * u.s)
        _assert_close(synthesis.summed(LINE).to_value(RADIANCE_UNIT), cube.data)
        assert meta["positions"].to_value(u.Mm) == pytest.approx(cube.meta["positions"].to_value(u.Mm))
        assert meta["starts"].to_value(u.s) == pytest.approx(cube.meta["starts"].to_value(u.s))

        # And on the detector, where the columns stay one per exposure.
        sim = Simulation(instrument="SWC", slit_width=slit_width, ncpu=1)
        expected = rebin_atmosphere(cube, Detector_SWC(), sim)
        found = rebin_spectra(synthesis, LINE, Detector_SWC(), sim, meta=meta)
        assert found.data.shape[1] == 4
        _assert_close(found.data, expected.data, rel=DETECTOR_ROUNDING)
        assert list(found.wcs.wcs.ctype) == list(expected.wcs.wcs.ctype)
        for field in ("crpix", "crval", "cdelt"):
            assert getattr(found.wcs.wcs, field) == pytest.approx(getattr(expected.wcs.wcs, field),
                                                                  rel=1e-12)


def test_an_instrument_run_observes_a_synthesis_series_as_its_atmosphere_series(tmp_path, monkeypatch, flat_goft):
    atmospheres = _series(tmp_path / "atmospheres", _ramp())
    _synthesise(tmp_path / "syntheses", atmospheres, monkeypatch)
    noise_free = {"slit_width": "0.4 arcsec", "expos": ["5 s", "10 s"], "psf": False, "noise": False}
    by_atmosphere = _config(tmp_path, str(tmp_path / "atmospheres" / "*.h5"), simulation=noise_free,
                            n_iter=1)
    config = yaml.safe_load(by_atmosphere.read_text())
    del config["atmosphere_series"], config["synthesis"]
    config["synthesis_series"] = str(tmp_path / "syntheses" / "*.h5")
    by_synthesis = tmp_path / "syntheses.yaml"
    by_synthesis.write_text(yaml.safe_dump(config))

    monkeypatch.chdir(tmp_path)
    from euvst_response.main import main as run_main
    saved = {}
    for path in (by_atmosphere, by_synthesis):
        monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(path)])
        run_main()
        with open(tmp_path / "run" / "result" / f"{path.stem}.pkl", "rb") as f:
            saved[path.stem] = dill.load(f)
    atmosphere_run, synthesis_run = saved["series"], saved["syntheses"]

    assert sorted(synthesis_run["cube_reb_dict"]) == sorted(atmosphere_run["cube_reb_dict"])
    for key, cube in atmosphere_run["cube_reb_dict"].items():
        _assert_close(synthesis_run["cube_reb_dict"][key].data, cube.data, rel=DETECTOR_ROUNDING)
        seen = synthesis_run["raster"]["cubes"][key]
        _assert_close(seen.data, atmosphere_run["raster"]["cubes"][key].data)
        assert seen.meta["starts"].to_value(u.s) == pytest.approx(
            atmosphere_run["raster"]["cubes"][key].meta["starts"].to_value(u.s))
    for key, combo in atmosphere_run["results"]["all_combinations"].items():
        assert np.array_equal(synthesis_run["results"]["all_combinations"][key]["first_dn_signal_data"],
                              combo["first_dn_signal_data"])
    raster = synthesis_run["raster"]
    assert raster["times"].to_value(u.s) == pytest.approx([0.0, 10.0, 20.0, 30.0])
    assert [p.split("/")[-1] for p in raster["series"]] == [f"synth_{i}.h5" for i in range(4)]
    assert raster["settings"] is None and raster["hdf5_dbase_root"] is None


# A line profile on wavelengths that are denser in the core, as optically
# thick codes give them, bright enough for the detector to count.
UNEVEN = REST + np.concatenate([np.linspace(-0.3, -0.05, 6), np.linspace(-0.04, 0.04, 17),
                                np.linspace(0.05, 0.3, 6)]) * u.Angstrom
PEAK = 2.0e4 * u.erg / (u.s * u.cm**2 * u.sr * u.Angstrom)


def _other_code_file(path, time, scale, wavelength=UNEVEN, x_edges=None, extra=None):
    """A synthesis file as another code might write it: each column the line profile times its *scale*."""
    ny = SHAPE[1]
    profile = np.exp(-0.5 * ((wavelength - REST) / (0.02 * u.Angstrom)).decompose().value ** 2)
    intensity = np.ones((ny, 1, 1)) * np.asarray(scale, float)[None, :, None] * profile
    lines = {LINE: SpectralLine(intensity=intensity * PEAK, wavelength=wavelength,
                                rest_wavelength=REST), **(extra or {})}
    edges = _edges()
    synthesis = Synthesis(lines=lines, x_edges=edges["x_edges"] if x_edges is None else x_edges,
                          y_edges=edges["y_edges"], source="another code",
                          time=None if time is None else time * u.s)
    return write_synthesis(synthesis, path)


def test_an_exposure_of_a_synthesis_series_averages_over_the_slit_and_over_time(tmp_path):
    nx = SHAPE[2]
    first = np.ones(nx)
    first[7] = 2.0
    paths = [_other_code_file(tmp_path / "a.h5", 0.0, first),
             _other_code_file(tmp_path / "b.h5", 10.0, 3.0 * np.ones(nx))]
    raster = SynthesisRaster(SynthesisSeries(paths, LINE))
    profile = raster.column_spectra(0, 6)[LINE]
    # In the radiance ECLIPSE works in, per cm rather than per Angstrom.
    assert profile.max() == pytest.approx(PEAK.to_value(RADIANCE_UNIT))

    # A slit two cells wide on the boundary of columns 6 and 7, open for
    # half its exposure in each snapshot.
    exposure = raster_module.Exposure(0, CELL, 5 * u.s, 15 * u.s)
    spectra = raster.exposure_spectra(exposure, 0.4 * u.arcsec)[LINE]
    expected = 0.5 * (0.5 * 1.0 + 0.5 * 2.0) + 0.5 * 3.0
    assert spectra == pytest.approx(expected * profile, rel=1e-12)
    # Column 6 of the first snapshot was read already; then 7 of the first,
    # and 6 and 7 of the second as one strip.
    assert raster.strips_read == 3
    with pytest.raises(ValueError, match="outside the image"):
        raster.columns_under(10 * CELL, 0.4 * u.arcsec)


def test_a_synthesis_series_goes_onto_the_detector_one_column_per_exposure(tmp_path):
    from euvst_response.config import Detector_SWC, Simulation
    from euvst_response.data_processing import rebin_spectra
    from euvst_response.utils import distance_to_angle

    nx = SHAPE[2]
    paths = [_other_code_file(tmp_path / f"{i}.h5", t, (1.0 + t / 10) * np.ones(nx))
             for i, t in enumerate((0.0, 10.0, 20.0))]
    raster = SynthesisRaster(SynthesisSeries(paths, LINE))
    synthesis, meta = raster.synthesis(RasterPlan(start=0 * u.s, steps=2, centre=0.5 * CELL),
                                       0.4 * u.arcsec, 10 * u.s)
    assert synthesis.shape == (SHAPE[1], 2)
    assert not synthesis.evenly_spaced(LINE)
    rebinned = rebin_spectra(synthesis, LINE, Detector_SWC(),
                             Simulation(instrument="SWC", slit_width=0.4 * u.arcsec, ncpu=1),
                             meta=meta)
    assert rebinned.data.shape[1] == 2
    assert rebinned.meta["raster"] is True
    scan = rebinned.axis_world_coords(1)[0]
    expected = distance_to_angle(meta["positions"]).to_value(u.arcsec)
    assert scan.Tx.to_value(u.arcsec)[0] == pytest.approx(expected, abs=1e-6)
    # The second exposure, from 10 to 20 s, saw the second snapshot, twice as bright.
    columns = rebinned.data.sum(axis=(0, 2))
    assert columns[1] / columns[0] == pytest.approx(2.0, rel=1e-9)


def test_a_synthesis_series_refuses_files_that_do_not_fit_together(tmp_path):
    import dataclasses

    from euvst_response.utils import distance_to_angle

    nx = SHAPE[2]
    ones = np.ones(nx)
    good = [_other_code_file(tmp_path / "a.h5", 0.0, ones),
            _other_code_file(tmp_path / "b.h5", 10.0, ones)]
    with pytest.raises(ValueError, match="records no time"):
        SynthesisSeries(good + [_other_code_file(tmp_path / "untimed.h5", None, ones)], LINE)
    with pytest.raises(ValueError, match="same time"):
        SynthesisSeries(good + [_other_code_file(tmp_path / "twin.h5", 10.0, ones)], LINE)
    shifted = _edges()["x_edges"] + 0.5 * CELL
    with pytest.raises(ValueError, match="different x grid"):
        SynthesisSeries(good + [_other_code_file(tmp_path / "shifted.h5", 20.0, ones,
                                                 x_edges=shifted)], LINE)
    with pytest.raises(ValueError, match="different wavelengths"):
        SynthesisSeries(good + [_other_code_file(tmp_path / "regridded.h5", 20.0, ones,
                                                 wavelength=UNEVEN * (1 + 1e-6))], LINE)
    blend = SpectralLine(intensity=np.ones((SHAPE[1], nx, UNEVEN.size)) * PEAK,
                         wavelength=UNEVEN, rest_wavelength=REST + 0.06 * u.Angstrom)
    with pytest.raises(ValueError, match="must have the same"):
        SynthesisSeries(good + [_other_code_file(tmp_path / "blended.h5", 20.0, ones,
                                                 extra={"Fe12_195.1790": blend})], LINE)
    dynamic = dataclasses.replace(read_synthesis(good[1]), time=20.0 * u.s)
    with pytest.raises(ValueError, match="dynamic mode"):
        SynthesisSeries(good + [write_synthesis(dynamic, tmp_path / "dynamic.h5",
                                                products={"dynamic_mode": {"enabled": True}})],
                        LINE)
    # The same image given in arcsec is the same image.
    in_arcsec = distance_to_angle(_edges()["x_edges"]).to(u.arcsec)
    series = SynthesisSeries(good + [_other_code_file(tmp_path / "arcsec.h5", 20.0, ones,
                                                      x_edges=in_arcsec)], LINE)
    assert len(series) == 3


def test_an_instrument_run_observes_another_codes_series(tmp_path, monkeypatch):
    nx = SHAPE[2]
    for index, time in enumerate((0.0, 10.0, 20.0, 30.0)):
        _other_code_file(tmp_path / "other" / f"snap_{index}.h5", time,
                         (1.0 + time / 10) * np.ones(nx))
    config = {"instrument": "SWC", "synthesis_series": str(tmp_path / "other" / "*.h5"),
              "n_iter": 2, "ncpu": 1, "fit_signals": "dn",
              "raster": {"start": "0 s", "steps": 2, "repeats": 2},
              "simulation": {"slit_width": "0.4 arcsec", "expos": "10 s", "psf": False}}
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(path)])
    from euvst_response.main import main as run_main
    run_main()

    with open(tmp_path / "run" / "result" / "other.pkl", "rb") as f:
        saved = dill.load(f)
    # The files hold one line, which is observed without a reference_line.
    assert len(saved["results"]["all_combinations"]) == 2
    for cube in saved["cube_reb_dict"].values():
        assert cube.data.shape[1] == 2
        assert cube.meta["line_name"] == LINE
    # The second raster starts at 20 s and sees the third and fourth snapshots.
    keys = sorted(saved["cube_reb_dict"])
    first, second = (saved["cube_reb_dict"][k].data.sum(axis=(0, 2)) for k in keys)
    assert second / first == pytest.approx([3.0 / 1.0, 4.0 / 2.0], rel=1e-9)
    # Uneven wavelengths have no WCS, so the spectra before the instrument are
    # not kept as a cube, as for a single synthesis file.
    assert all(cube is None for cube in saved["raster"]["cubes"].values())


def test_a_synthesis_series_run_refuses_what_it_cannot_do(tmp_path, monkeypatch):
    nx = SHAPE[2]
    for index, time in enumerate((0.0, 10.0)):
        _other_code_file(tmp_path / "other" / f"snap_{index}.h5", time, np.ones(nx))
    monkeypatch.chdir(tmp_path)
    from euvst_response.main import main as run_main

    def run(**changes):
        config = {"instrument": "SWC", "synthesis_series": str(tmp_path / "other" / "*.h5"),
                  "n_iter": 2, "ncpu": 1, "raster": {"start": "0 s"},
                  "simulation": {"slit_width": "0.4 arcsec", "expos": "5 s", "psf": False}}
        config.update(changes)
        config = {key: value for key, value in config.items() if value is not None}
        path = tmp_path / "refused.yaml"
        path.write_text(yaml.safe_dump(config))
        monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(path)])
        run_main()

    with pytest.raises(ValueError, match="only one of"):
        run(atmosphere_series=str(tmp_path / "other" / "*.h5"))
    with pytest.raises(ValueError, match="only one of"):
        run(synthesis_file=str(tmp_path / "other" / "snap_0.h5"))
    with pytest.raises(ValueError, match="belongs to an 'atmosphere_series' run"):
        run(synthesis={"lines": [LINE]})
    with pytest.raises(ValueError, match="needs a 'raster:' section"):
        run(raster=None)
    with pytest.raises(ValueError, match="is not in"):
        run(reference_line="Fe09_171.0730")
    with pytest.raises(ValueError, match="outside the image"):
        run(raster={"start": "0 s", "centre": "5 Mm"})
