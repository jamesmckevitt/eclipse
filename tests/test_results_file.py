"""Results survive a round trip through the results file, and old pickles still load.

The tests that matter here are the ones that compare against the object that
went in, field by field, because a format change that loses something does
not announce itself: the file still opens and the numbers still look like
numbers.
"""
import dataclasses
import datetime
import json
import os
import re
import warnings
import zlib

import astropy.units as u
import h5py
import numpy as np
import pytest
import yaml
from astropy.io import fits
from astropy.wcs import WCS, Sip
from ndcube import NDCube

from euvst_response import results_file
from euvst_response.config import (AluminiumFilter, Detector_EIS,
                                   Detector_SWC, Simulation, Telescope_EIS,
                                   Telescope_EUVST)
from euvst_response.fitting import FitComponent, FitConfig
from euvst_response.raster import RasterPlan, SynthesisSettings
from euvst_response.results_file import is_results_file, load_results, save_results

REST = 195.119 * u.Angstrom


def _wcs():
    """The WCS of a (n_slit, n_scan, n_wavelength) = (2, 3, 4) cube."""
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [1.69e-11, 0.2, 0.159]
    wcs.wcs.crpix = [2.5, 2.0, 1.5]
    wcs.wcs.crval = [1.95119e-06, 0.0, 0.0]
    return wcs


def _cube(offset=0.0):
    return NDCube(np.arange(24, dtype=float).reshape((2, 3, 4)) + offset,
                  wcs=_wcs(), unit=u.DN / u.pix,
                  meta={"rest_wav": REST, "line_name": "Fe12_195.1190",
                        "uniform_mode": False})


def _payload():
    """A tree with one of everything a real result file holds."""
    return {
        "instrument": "SWC",
        "software_version": "0.7.0",
        "git_commit_id": "abc123",
        "config": {"instrument": "SWC", "n_iter": 5,
                   "simulation": {"expos": ["5 s", "20 s"]}},
        "cube_sim": _cube(),
        # Cubes alike but for their data, which must not come back as one.
        "cube_reb_dict": {(0.2, 1): _cube(), (0.4, 2): _cube(offset=100.0)},
        "results": {
            "all_combinations": {
                (("simulation.expos", 5.0), ("detector.qe_euv", 0.76)): {
                    "parameters": {"simulation.expos": 5.0 * u.s,
                                   "simulation.psf": True,
                                   "detector.material": "silicon",
                                   "offchip_bin_slit": 2},
                    "config_objects": {
                        "detector": Detector_SWC(),
                        "telescope": Telescope_EUVST(
                            filter=AluminiumFilter(al_thickness=1500 * u.AA)),
                        "simulation": Simulation(instrument="SWC",
                                                 slit_width=0.2 * u.arcsec),
                    },
                    "first_dn_signal_data": np.ones((2, 3, 4)),
                    "first_dn_signal_unit": u.DN / u.pix,
                    "first_signal_wcs": _wcs(),
                    "dn_fit_stats": {
                        "first_fit_data": np.zeros((2, 3, 4)),
                        "units": [u.DN / u.pix, u.Angstrom, u.Angstrom,
                                  u.DN / u.pix],
                    },
                    "photon_fit_stats": None,
                },
            },
            "sweep_dimensions": {"simulation.expos": [5.0 * u.s, 20.0 * u.s]},
            "fit_config": FitConfig(
                components=[FitComponent(wavelength=195.119 * u.AA),
                            FitComponent(wavelength=195.179 * u.AA,
                                         tie_center=0, tie_width=0)],
                primary_component=0, max_iter=500),
            "fit_signals": "both",
            "fit_weighted": True,
        },
    }


def _round_trip(tmp_path, payload=None, **kwargs):
    path = save_results(tmp_path / "out.h5", payload or _payload(), **kwargs)
    return load_results(path)


def test_the_file_is_hdf5_not_a_pickle(tmp_path):
    path = save_results(tmp_path / "out.h5", _payload())
    assert is_results_file(path)
    with h5py.File(path, "r") as f:
        assert f.attrs["format"] == "eclipse-results" and f.attrs["version"] == 1
        # Any HDF5 reader finds the arrays, with their units.
        assert f["cube_sim/data"].shape == (2, 3, 4)
        assert f["cube_sim/data"].dtype == np.float64
        assert f["cube_sim/data"].attrs["unit"] == "DN / pix"


def test_a_failed_write_leaves_the_file_there_as_it_was(tmp_path):
    path = save_results(tmp_path / "out.h5", {"instrument": "SWC"})
    # And says where what it cannot write is.
    with pytest.raises(TypeError, match="/deep: Values of type object cannot be written"):
        save_results(path, {"instrument": "EIS", "deep": {"odd": object()}})
    assert load_results(path)["instrument"] == "SWC"
    assert list(tmp_path.iterdir()) == [path]


def test_what_yaml_can_put_in_a_configuration_survives(tmp_path):
    config = yaml.safe_load("lines: !!set {Fe12_195.1190, Fe09_171.0730}\n"
                            "blob: !!binary aGVsbG8=\n")
    out = _round_trip(tmp_path, {"config": {**config, "frozen": frozenset({1, 2})}})["config"]
    assert out["lines"] == {"Fe12_195.1190", "Fe09_171.0730"} and type(out["lines"]) is set
    assert out["blob"] == b"hello" and out["frozen"] == frozenset({1, 2})
    assert type(out["frozen"]) is frozenset


def test_the_file_is_made_as_any_file_is(tmp_path):
    """With the permissions any other file there gets, so that colleagues can read it, and through a link."""
    # A umask that lets the group write, which a file made only for its owner,
    # or with permissions of its own, would not follow.
    previous = os.umask(0o002)
    try:
        path = save_results(tmp_path / "out.h5", {"instrument": "SWC"})
        (tmp_path / "plain").write_text("")
    finally:
        os.umask(previous)
    assert path.stat().st_mode & 0o777 == (tmp_path / "plain").stat().st_mode & 0o777
    assert path.stat().st_mode & 0o060 == 0o060
    target = save_results(tmp_path / "real.h5", {"instrument": "SWC"})
    link = tmp_path / "link.h5"
    link.symlink_to(target)
    save_results(link, {"instrument": "EIS"})
    assert link.is_symlink() and load_results(target)["instrument"] == "EIS"


def test_a_later_layout_or_a_link_elsewhere_is_refused(tmp_path):
    path = save_results(tmp_path / "out.h5", {"instrument": "SWC", "data": np.ones(3)})
    with h5py.File(path, "r+") as f:
        f.attrs["version"] = 2
    with pytest.raises(ValueError, match="reads version 1"):
        load_results(path)

    path = save_results(tmp_path / "out.h5", {"instrument": "SWC", "data": np.ones(3)})
    save_results(tmp_path / "other.h5", {"secret": np.zeros(3)})
    with h5py.File(path, "r+") as f:
        del f["data"]
        f["data"] = h5py.ExternalLink("other.h5", "/secret")
    with pytest.raises(ValueError, match="is a link"):
        load_results(path)


def test_scalars_and_strings_survive(tmp_path):
    out = _round_trip(tmp_path)
    assert out["instrument"] == "SWC"
    assert out["software_version"] == "0.7.0"
    assert out["config"]["n_iter"] == 5
    assert out["config"]["simulation"]["expos"] == ["5 s", "20 s"]


def test_an_unquoted_yaml_date_survives(tmp_path):
    """The raw config is stored as read, dates included.

    YAML reads an unquoted 2012-06-03 as a datetime.date, which is how an EIS
    observation date usually arrives.
    """
    config = yaml.safe_load("telescope:\n  calibration: dz2025\n"
                            "  date: 2012-06-03\n")
    out = _round_trip(tmp_path, {"config": config})
    date = out["config"]["telescope"]["date"]
    assert type(date) is datetime.date
    assert date == datetime.date(2012, 6, 3)


def test_the_cube_comes_back_whole(tmp_path):
    out = _round_trip(tmp_path)
    original, restored = _cube(), out["cube_sim"]

    assert np.array_equal(restored.data, original.data)
    assert restored.unit == original.unit
    assert restored.meta["line_name"] == "Fe12_195.1190"
    assert restored.meta["rest_wav"] == REST
    assert restored.meta["uniform_mode"] is False


def test_the_wcs_keeps_the_units_it_was_written_in(tmp_path):
    """A WCS written as a FITS header comes back in SI units unless its own are put back.

    Without this, a wavelength axis written in cm comes back in m. The
    coordinates are the same, but anything reading wcs.wcs.cdelt directly
    would be out by a factor of a hundred.
    """
    out = _round_trip(tmp_path)
    restored = out["cube_sim"].wcs

    assert list(restored.wcs.ctype) == ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    assert [str(c) for c in restored.wcs.cunit] == ["cm", "arcsec", "arcsec"]
    assert list(restored.wcs.cdelt) == list(_wcs().wcs.cdelt)
    assert list(restored.wcs.crpix) == list(_wcs().wcs.crpix)


def test_the_wcs_describes_the_same_world_coordinates(tmp_path):
    """The property that actually matters, whatever the keywords say."""
    out = _round_trip(tmp_path)
    before = _cube().axis_world_coords(2)[0].to_value(u.cm)
    after = out["cube_sim"].axis_world_coords(2)[0].to_value(u.cm)
    assert np.allclose(after, before, rtol=1e-12, atol=0.0)


def test_saving_leaves_the_callers_wcs_alone(tmp_path):
    """Writing a WCS as a header normalises it to SI in place.

    Saving must not do that to the cube or WCS the caller still holds, or a
    script that saves and then carries on would find its cdelt rescaled.
    """
    cube, wcs = _cube(), _wcs()
    save_results(tmp_path / "out.h5", {"cube_sim": cube, "wcs": wcs})

    for held in (cube.wcs, wcs):
        assert [str(c) for c in held.wcs.cunit] == ["cm", "arcsec", "arcsec"]
        assert held.wcs.cdelt == pytest.approx(_wcs().wcs.cdelt, rel=1e-12)


def test_tuple_keyed_dicts_survive(tmp_path):
    """HDF5 names members by strings; results are keyed by tuple."""
    out = _round_trip(tmp_path)

    assert set(out["cube_reb_dict"]) == {(0.2, 1), (0.4, 2)}
    first, second = out["cube_reb_dict"][(0.2, 1)], out["cube_reb_dict"][(0.4, 2)]
    assert first is not second
    assert np.array_equal(first.data, _cube().data)
    assert np.array_equal(second.data, _cube(offset=100.0).data)
    combos = out["results"]["all_combinations"]
    key, = combos
    assert key == (("simulation.expos", 5.0), ("detector.qe_euv", 0.76))


def test_quantities_and_units_survive(tmp_path):
    out = _round_trip(tmp_path)
    combo = next(iter(out["results"]["all_combinations"].values()))

    assert combo["parameters"]["simulation.expos"] == 5.0 * u.s
    assert combo["first_dn_signal_unit"] == u.DN / u.pix
    assert combo["dn_fit_stats"]["units"][1] == u.Angstrom
    assert out["results"]["sweep_dimensions"]["simulation.expos"][1] == 20 * u.s


def test_booleans_stay_boolean(tmp_path):
    """A bool that comes back as 1 would quietly change a config comparison."""
    out = _round_trip(tmp_path)
    combo = next(iter(out["results"]["all_combinations"].values()))
    assert combo["parameters"]["simulation.psf"] is True


def test_none_stays_none(tmp_path):
    out = _round_trip(tmp_path)
    combo = next(iter(out["results"]["all_combinations"].values()))
    assert combo["photon_fit_stats"] is None


def test_config_objects_come_back_as_the_right_classes(tmp_path):
    out = _round_trip(tmp_path)
    objects = next(iter(
        out["results"]["all_combinations"].values()))["config_objects"]

    assert isinstance(objects["detector"], Detector_SWC)
    assert isinstance(objects["simulation"], Simulation)
    assert isinstance(objects["telescope"], Telescope_EUVST)
    # nested dataclass, and a value that was not the default
    assert isinstance(objects["telescope"].filter, AluminiumFilter)
    assert objects["telescope"].filter.al_thickness == 1500 * u.AA


def _same(before, after):
    """Equal in value and in type, looking inside dataclasses and lists."""
    if type(after) is not type(before):
        return False
    if dataclasses.is_dataclass(before):
        return all(_same(getattr(before, f.name), getattr(after, f.name))
                   for f in dataclasses.fields(before))
    if isinstance(before, (list, tuple)):
        return (len(after) == len(before)
                and all(_same(b, a) for b, a in zip(before, after)))
    if isinstance(before, u.Quantity):
        return after.unit == before.unit and np.array_equal(after.value,
                                                            before.value)
    return after == before


@pytest.mark.parametrize("config_object", [
    # Fields away from their defaults, and each stored by name (checked below),
    # so that a field dropped on the way out cannot pass as its default.
    Simulation(expos=20 * u.s, n_iter=3, slit_width=0.4 * u.arcsec, ncpu=2,
               instrument="SWC", vis_sl=10 * u.photon / (u.s * u.cm**2),
               psf=True, noise=False, enable_pinholes=True,
               pinhole_sizes=[5 * u.um, 10 * u.um],
               pinhole_positions=[0.25, 0.75],
               pinhole_positions_spectral=[0.1, 0.9]),
    Simulation(instrument="EIS", slit_width=2 * u.arcsec, noise=False),
    Detector_SWC(ccd_temperature=-40 * u.deg_C, qe_vis=0.9, qe_euv=0.7,
                 read_noise_rms=8 * u.electron / u.pixel,
                 gain_e_per_dn=3.0 * u.electron / u.DN,
                 filter_distance=200 * u.mm),
    Detector_EIS(ccd_temperature=-50 * u.deg_C, qe_euv=0.6,
                 read_noise_rms=6 * u.electron / u.pixel),
    Telescope_EUVST(D_ap=0.3 * u.m, microroughness_sigma=0.5 * u.nm,
                    filter=AluminiumFilter(oxide_thickness=80 * u.AA,
                                           c_thickness=20 * u.AA,
                                           mesh_throughput=0.75),
                    psf_params=[2.0 * u.pixel, 2.5 * u.pixel]),
    Telescope_EIS(psf_params=[2.0 * u.pixel, 2.5 * u.pixel],
                  calibration="dz2025", date="2012-06-03"),
    RasterPlan(start=100 * u.s, steps=4, step=0.3 * u.arcsec, repeats=2,
               cadence=12 * u.s, centre=1.5 * u.Mm),
    # A time series' synthesis settings, the precision being a NumPy type.
    SynthesisSettings(lines=("Fe12_195.1190", "Fe09_171.0730"),
                      abundance="sun_photospheric_2015_scott",
                      vel_res=10 * u.km / u.s, vel_lim=500 * u.km / u.s,
                      crop_y=(-1 * u.Mm, 1 * u.Mm), crop_z=(0 * u.Mm, 10 * u.Mm),
                      precision=np.float32, mass_per_electron=1.2,
                      hdf5_dbase_root="/somewhere/chianti", n_workers=4,
                      goft_temperature_chunk=16),
    FitComponent(wavelength=195.179 * u.AA, tie_center=0, tie_width=0,
                 amplitude_greater_than=0, name="blend"),
    FitConfig(components=[FitComponent(wavelength=195.119 * u.AA, name="main"),
                          FitComponent(wavelength=195.179 * u.AA, tie_width=0,
                                       amplitude_greater_than=0, name="blend")],
              primary_component=1, constrain_positive_intensity=True, backend="scipy",
              max_iter=200, bessel_correction=True, save_iterations=True),
], ids=lambda obj: type(obj).__name__)
def test_every_config_field_survives(tmp_path, config_object):
    """Every field, Simulation.noise included, comes back unchanged.

    A run with the noise off that read back with it on would be mistaken for
    a noisy one.
    """
    stored = results_file._jsonable(config_object)["fields"]
    assert set(stored) == {f.name for f in dataclasses.fields(config_object) if f.init}
    restored = _round_trip(tmp_path, {"object": config_object})["object"]

    for f in dataclasses.fields(config_object):
        before = getattr(config_object, f.name)
        after = getattr(restored, f.name)
        assert _same(before, after), f"{f.name}: {before!r} -> {after!r}"


def test_a_restored_detector_still_computes(tmp_path):
    """Rebuilt objects have to work, not merely have the right fields."""
    out = _round_trip(tmp_path)
    objects = next(iter(
        out["results"]["all_combinations"].values()))["config_objects"]

    assert objects["detector"].dark_current == Detector_SWC().dark_current
    area = objects["telescope"].ea_and_throughput(REST)
    assert area.to_value(u.cm**2) == pytest.approx(
        Telescope_EUVST(filter=AluminiumFilter(al_thickness=1500 * u.AA))
        .ea_and_throughput(REST).to_value(u.cm**2), rel=1e-12)


def test_package_tables_are_not_stored_as_the_writers_paths(tmp_path,
                                                            monkeypatch):
    """Package data is found in the reader's installation, not the writer's.

    An absolute path to a throughput table names the installation that wrote
    the file, which a colleague reading it elsewhere does not have.
    """
    telescope = Telescope_EUVST()
    path = save_results(tmp_path / "out.h5", {"telescope": telescope})

    assert str(results_file._package_root()).encode() not in path.read_bytes()

    elsewhere = tmp_path / "another_install" / "euvst_response"
    monkeypatch.setattr(results_file, "_package_root", lambda: elsewhere)
    restored = load_results(path)["telescope"]

    tables = elsewhere / "data" / "throughput"
    assert restored.pm_table == tables / telescope.pm_table.name
    assert restored.grating_table == tables / telescope.grating_table.name
    assert restored.filter.al_table == tables / telescope.filter.al_table.name


def test_a_path_outside_the_package_is_kept_as_given(tmp_path):
    table = tmp_path / "my_tables" / "aluminium.dat"
    out = _round_trip(tmp_path, {"filter": AluminiumFilter(al_table=table)})
    assert out["filter"].al_table == table


def test_the_fit_config_survives(tmp_path):
    out = _round_trip(tmp_path)
    fit_config = out["results"]["fit_config"]

    assert isinstance(fit_config, FitConfig)
    assert fit_config.max_iter == 500
    assert len(fit_config.components) == 2
    assert isinstance(fit_config.components[1], FitComponent)
    assert fit_config.components[1].tie_center == 0
    assert fit_config.components[0].wavelength == 195.119 * u.AA
    # a derived property, so the object is genuinely functional
    assert fit_config.n_components == 2


def test_arrays_keep_their_values_and_dtype(tmp_path):
    out = _round_trip(tmp_path)
    combo = next(iter(out["results"]["all_combinations"].values()))
    assert np.array_equal(combo["first_dn_signal_data"], np.ones((2, 3, 4)))
    assert combo["first_dn_signal_data"].dtype == np.float64


def test_arrays_outlive_the_closed_file(tmp_path):
    """The arrays are read out of the file, not left pointing into it."""
    out = _round_trip(tmp_path)
    array = out["cube_sim"].data
    assert isinstance(array, np.ndarray)
    assert float(array.sum()) == pytest.approx(276.0)


def test_the_larger_arrays_are_compressed_unless_asked_not_to_be(tmp_path):
    payload = {"large": np.arange(5000.0), "small": np.arange(10.0)}
    for compression, expected in (("gzip", "gzip"), (None, None)):
        path = save_results(tmp_path / f"{compression}.h5", payload, compression=compression)
        with h5py.File(path, "r") as f:
            assert f["large"].compression == expected and f["small"].compression is None
        assert np.array_equal(load_results(path)["large"], payload["large"])


def test_compression_actually_shrinks_a_large_array(tmp_path):
    payload = {"big": np.zeros((60, 60, 60))}
    small = save_results(tmp_path / "z.h5", payload, compression="gzip")
    big = save_results(tmp_path / "n.h5", payload, compression=None)
    assert small.stat().st_size < big.stat().st_size / 10


@pytest.mark.parametrize("name", ["result.pkl", "result.pickle", "result.PKL"])
def test_a_pkl_name_is_corrected_rather_than_written(tmp_path, name):
    """A file named .pkl that is not a pickle is worse than a renamed one."""
    with pytest.warns(UserWarning, match="an HDF5 file"):
        path = save_results(tmp_path / name, {"instrument": "SWC"})
    assert path.name == "result.h5"
    assert path.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["result.h5"]


def test_an_old_pickle_still_loads(tmp_path):
    """Existing result files have to keep working, whatever they are named."""
    import dill

    path = tmp_path / "old.pkl"
    with open(path, "wb") as handle:
        dill.dump({"instrument": "EIS", "cube": _cube()}, handle)

    with pytest.warns(FutureWarning, match="results pickle"):
        out = load_results(path)

    assert out["instrument"] == "EIS"
    assert np.array_equal(out["cube"].data, _cube().data)


def test_a_pickle_named_as_an_hdf5_file_is_not_unpickled(tmp_path):
    """A reader takes a .h5 file to be safe to open; a pickle under any other name loads, with a warning."""
    import dill

    misnamed = tmp_path / "actually_a_pickle.h5"
    with open(misnamed, "wb") as handle:
        dill.dump({"instrument": "EIS"}, handle)

    assert not is_results_file(misnamed)
    with pytest.raises(ValueError, match="is named as an HDF5 file but is a pickle"):
        load_results(misnamed)
    for name in ("other.hdf5", "other.hdf", "other.he5"):
        named = tmp_path / name
        named.write_bytes(misnamed.read_bytes())
        with pytest.raises(ValueError, match="is named as an HDF5 file but is a pickle"):
            load_results(named)
    for name in ("upper.PKL", "other.pickle", "run.pkl.old", "x.results"):
        named = tmp_path / name
        named.write_bytes(misnamed.read_bytes())
        with pytest.warns(FutureWarning, match="results pickle"):
            assert load_results(named)["instrument"] == "EIS"

    # And a results file named .pkl, with no .h5 beside it, reads as one.
    results = save_results(tmp_path / "results.h5", {"instrument": "SWC"})
    renamed = results.rename(tmp_path / "renamed.pkl")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_results(renamed)["instrument"] == "SWC"


def test_a_file_cannot_name_an_arbitrary_class(tmp_path):
    """The point of leaving pickle: reading must not construct what it likes."""
    path = tmp_path / "hostile.h5"
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "eclipse-results"
        f.attrs["version"] = 1
        f.attrs["eclipse_order"] = '["thing"]'
        f.attrs["thing"] = '{"__eclipse__": "dataclass", "class": "os.system", "fields": {}}'

    with pytest.raises(ValueError, match="will not construct"):
        load_results(path)


def test_an_eis_telescope_survives(tmp_path):
    """It validates its calibration and date in __post_init__."""
    telescope = Telescope_EIS(calibration="dz2025", date="2012-06-03")
    out = _round_trip(tmp_path, {"telescope": telescope})
    assert isinstance(out["telescope"], Telescope_EIS)
    assert out["telescope"].date == "2012-06-03"
    assert np.isfinite(
        out["telescope"].effective_area(REST).to_value(u.cm**2))


def test_numpy_scalars_do_not_stop_a_write(tmp_path):
    """np.bool_ and the like turn up in parsed configs."""
    out = _round_trip(tmp_path, {"flag": np.bool_(True),
                                 "count": np.int64(7),
                                 "value": np.float32(1.5)})
    assert out["flag"] is np.True_
    assert type(out["count"]) is np.int64 and out["count"] == 7
    assert type(out["value"]) is np.float32 and out["value"] == 1.5


def test_sequences_and_meta_holding_arrays_are_groups(tmp_path):
    cube = NDCube(np.ones((2, 3)), wcs=WCS(naxis=2),
                  meta={"positions": np.arange(3.0) * u.Mm, "raster": True})
    out = _round_trip(tmp_path, {"pair": (np.ones(2), "a"), "items": [np.zeros(3), 1.5],
                                 "cube": cube})
    assert type(out["pair"]) is tuple and out["pair"][1] == "a"
    assert np.array_equal(out["pair"][0], np.ones(2))
    assert type(out["items"]) is list and out["items"][1] == 1.5
    assert u.allclose(out["cube"].meta["positions"], np.arange(3.0) * u.Mm)
    assert out["cube"].meta["raster"] is True and out["cube"].unit is None
    with h5py.File(tmp_path / "out.h5", "r") as f:
        assert f["pair"].attrs["eclipse_type"] == "tuple" and "0" in f["pair"]
        assert f["items"].attrs["eclipse_type"] == "list" and "0" in f["items"]
        assert isinstance(f["cube/meta"], h5py.Group) and "positions" in f["cube/meta"]


def test_keys_that_cannot_name_members_are_kept(tmp_path):
    """Keys HDF5 cannot use as names, or that the file uses itself, are kept in order."""
    nested = {"a/b": np.ones(2), "version": 1, "": 2, ".": 3, 4: "four", "eclipse_order": "x"}
    plain = {"a/b": 1, 2: "two", "__eclipse__": 3}
    out = _round_trip(tmp_path, {"nested": nested, "plain": plain})
    assert list(out["nested"]) == list(nested)
    assert np.array_equal(out["nested"]["a/b"], np.ones(2)) and out["nested"][4] == "four"
    assert out["plain"] == plain
    with pytest.raises(ValueError, match="cannot be keyed by"):
        save_results(tmp_path / "bad.h5", {"version": 1})


def test_dates_times_text_and_edge_values_survive(tmp_path):
    when = datetime.datetime(2026, 9, 25, 12, 30, 5)
    names = np.array(["Fe12_195.1190", "Fe09_171.0730"])
    out = _round_trip(tmp_path, {"when": when, "names": names, "empty": np.zeros((0, 3)),
                                 "nan": float("nan"), "big": 2**70, "flags": np.array([True, False])})
    assert type(out["when"]) is datetime.datetime and out["when"] == when
    assert out["names"].dtype == names.dtype and list(out["names"]) == list(names)
    assert out["empty"].shape == (0, 3)
    assert np.isnan(out["nan"]) and out["big"] == 2**70
    assert out["flags"].dtype == bool and list(out["flags"]) == [True, False]


def test_text_arrays_keep_their_shape_and_width(tmp_path):
    """Written as their items, which do not give an empty array's shape or a width wider than the text."""
    text = {"wide": np.array(["a"], dtype="U5"), "empty": np.empty((0, 3), dtype="U5"),
            "grid": np.array([["ab", "c"], ["d", "e"]]), "none": np.empty((2, 0), dtype="U1"),
            "padded": np.array(["a"] * 100, dtype="U100"), "uneven": np.array(["x" * 200] + [""] * 2000),
            "one": np.array("abc")}
    out = _round_trip(tmp_path, text)
    for name, array in text.items():
        assert out[name].dtype == array.dtype and out[name].shape == array.shape, name
        assert out[name].tolist() == array.tolist(), name


def test_a_file_that_does_not_hold_together_is_refused(tmp_path):
    path = save_results(tmp_path / "out.h5", {"group": {"data": np.ones(3), "note": "x"}})
    with h5py.File(path, "r+") as f:
        del f["group"].attrs["note"]
    with pytest.raises(ValueError, match="not those it lists"):
        load_results(path)

    path = save_results(tmp_path / "out.h5", {"group": {"data": np.ones(3)}})
    with h5py.File(path, "r+") as f:
        del f["group"].attrs["eclipse_order"]
    with pytest.raises(ValueError, match="no 'eclipse_order' attribute"):
        load_results(path)

    # A soft link, or a dataset keeping its data in another file, is refused too.
    path = save_results(tmp_path / "out.h5", {"group": {"data": np.ones(3)}, "other": np.ones(3)})
    with h5py.File(path, "r+") as f:
        del f["other"]
        f["other"] = h5py.SoftLink("/group/data")
    with pytest.raises(ValueError, match="is a link"):
        load_results(path)

    raw = tmp_path / "raw.bin"
    np.ones(3).tofile(raw)
    path = save_results(tmp_path / "out.h5", {"other": np.ones(3)})
    with h5py.File(path, "r+") as f:
        del f["other"]
        f.create_dataset("other", shape=(3,), dtype="f8", external=[(str(raw), 0, 24)])
    with pytest.raises(ValueError, match="keeps its data in another file"):
        load_results(path)


def test_only_the_configuration_objects_it_rebuilds_are_written(tmp_path):
    """So that a file is not written that cannot then be read."""
    @dataclasses.dataclass
    class Mine:
        value: int = 1

    with pytest.raises(TypeError, match="only .* are rebuilt"):
        save_results(tmp_path / "mine.h5", {"mine": Mine()})
    assert not (tmp_path / "mine.h5").exists()

    # Nor one only named like one of them.
    @dataclasses.dataclass
    class Simulation:
        value: int = 1

    with pytest.raises(TypeError, match="only .* are rebuilt"):
        save_results(tmp_path / "mine.h5", {"mine": Simulation()})


def _with_attribute(tmp_path, name, value):
    """A results file with *value* written by hand as the JSON of member *name*."""
    path = save_results(tmp_path / "hand.h5", {"instrument": "SWC"})
    with h5py.File(path, "r+") as f:
        f.attrs[name] = json.dumps(value)
        f.attrs["eclipse_order"] = json.dumps(["instrument", name])
    return path


def test_a_configuration_object_this_version_would_not_make_still_reads(tmp_path):
    """As one from a later check, or with a setting since removed, would be."""
    fit = FitConfig(components=[FitComponent(wavelength=195.119 * u.AA),
                                FitComponent(wavelength=195.179 * u.AA)])
    encoded = results_file._jsonable(fit)
    # One component, which this version refuses, and a setting it lacks.
    encoded["fields"]["components"] = encoded["fields"]["components"][:1]
    encoded["fields"]["retired_setting"] = 3
    with pytest.warns(UserWarning) as seen:
        restored = load_results(_with_attribute(tmp_path, "fit", encoded))["fit"]
    messages = " ".join(str(w.message) for w in seen)
    assert "retired_setting" in messages and "rebuilt as it was stored" in messages
    assert isinstance(restored, FitConfig) and len(restored.components) == 1
    assert restored.components[0].wavelength == 195.119 * u.AA

    # Rebuilt unchecked, it still gets today's default for a setting it lacks.
    encoded = results_file._jsonable(Simulation(instrument="SWC", slit_width=0.4 * u.arcsec))
    encoded["fields"]["slit_width"] = results_file._jsonable(0.3 * u.arcsec)
    del encoded["fields"]["pinhole_positions_spectral"]
    with pytest.warns(UserWarning, match="rebuilt as it was stored"):
        restored = load_results(_with_attribute(tmp_path, "simulation", encoded))["simulation"]
    assert restored.slit_width == 0.3 * u.arcsec and restored.pinhole_positions_spectral == []


def test_a_setting_added_since_the_file_was_made_is_said_to_take_todays_default(tmp_path):
    """A file from an earlier version lacks the settings added since; they are defaulted, and said."""
    encoded = results_file._jsonable(Detector_SWC(row_transfer_time=20 * u.us))
    del encoded["fields"]["shutter"], encoded["fields"]["row_transfer_time"]
    with pytest.warns(UserWarning, match="Detector_SWC in the results file has no "
                                         "row_transfer_time, shutter, which ECLIPSE added after "
                                         "it was made; they get today's defaults"):
        restored = load_results(_with_attribute(tmp_path, "detector", encoded))["detector"]
    assert restored.shutter is True and restored.row_transfer_time == 15 * u.us


def test_a_fit_from_before_weighting_existed_reads_back_as_unweighted(tmp_path):
    """Its fits were unweighted, so it is not given today's default, which weights them."""
    encoded = results_file._jsonable(FitConfig())
    del encoded["fields"]["weighted"]
    with pytest.warns(UserWarning, match="FitConfig in the results file has no weighted, which "
                                         "ECLIPSE added after it was made; it gets False, as "
                                         "runs then had"):
        restored = load_results(_with_attribute(tmp_path, "fit", encoded))["fit"]
    assert restored.weighted is False


def test_results_from_before_weighting_existed_are_said_to_be_unweighted(tmp_path):
    """With no fitting block there is no FitConfig to say so, so the results do."""
    payload = _payload()
    del payload["results"]["fit_weighted"]
    payload["results"]["fit_config"] = None
    path = save_results(tmp_path / "out.h5", payload)
    with pytest.warns(UserWarning, match="made before ECLIPSE weighted its fits"):
        assert load_results(path)["results"]["fit_weighted"] is False


def test_a_pickle_from_before_weighting_existed_is_said_to_be_unweighted(tmp_path):
    """Its FitConfig lacks the setting, and would otherwise read the class's default."""
    import dill

    fit_config = FitConfig()
    del fit_config.__dict__["weighted"]
    path = tmp_path / "old.pkl"
    with open(path, "wb") as handle:
        dill.dump({"results": {"all_combinations": {}, "fit_config": fit_config}}, handle)
    with pytest.warns(UserWarning, match="made before ECLIPSE weighted its fits"):
        results = load_results(path)["results"]
    assert results["fit_weighted"] is False and results["fit_config"].weighted is False


def test_what_a_configuration_object_worked_out_is_kept_as_the_run_had_it(tmp_path):
    """A detector's dark current, as the run used it, whatever this version works out."""
    detector = Detector_SWC()
    as_run = 42 * detector.dark_current.unit
    object.__setattr__(detector, "dark_current", as_run)
    assert _round_trip(tmp_path, {"detector": detector})["detector"].dark_current == as_run

    # Only what it works out, so that a file cannot set a setting past its checks.
    encoded = results_file._jsonable(Detector_SWC())
    encoded["derived"].update(qe_euv=7.0, __dict__={"qe_euv": 7.0}, not_a_field=1)
    restored = load_results(_with_attribute(tmp_path, "detector", encoded))["detector"]
    assert restored.qe_euv == Detector_SWC().qe_euv and not hasattr(restored, "not_a_field")


def test_a_crafted_file_cannot_have_the_reader_repeat_itself_or_swell(tmp_path):
    path = save_results(tmp_path / "out.h5", {"data": np.ones(3), "more": {"x": np.ones(2)}})
    with h5py.File(path, "r+") as f:
        f["more"]["again"] = f["data"]
        f["more"].attrs["eclipse_order"] = json.dumps(["x", "again"])
    with pytest.raises(ValueError, match="reached from more than one place"):
        load_results(path)

    swelling = {"__eclipse__": "array", "dtype": "(1000,1000)f8", "value": [0.0]}
    with pytest.raises(ValueError, match="which a results file does not"):
        load_results(_with_attribute(tmp_path, "swell", swelling))
    structured = {"__eclipse__": "numpy_type", "value": "f8,i4"}
    with pytest.raises(ValueError, match="which a results file does not"):
        load_results(_with_attribute(tmp_path, "kind", structured))

    # A member listed more than once would be read again for each.
    path = save_results(tmp_path / "out.h5", {"more": {"x": np.ones(2)}})
    with h5py.File(path, "r+") as f:
        f["more"].attrs["eclipse_order"] = json.dumps(["x", "x"])
    with pytest.raises(ValueError, match="not those it lists"):
        load_results(path)


def _crafted_dataset(tmp_path, write=None, **options):
    """A results file whose one dataset is made with *options* by hand, and then *write* done to it."""
    path = save_results(tmp_path / "crafted.h5", {"instrument": "SWC"})
    with h5py.File(path, "r+") as f:
        data = f.create_dataset("data", **options)
        if write is not None:
            write(data)
        f.attrs["eclipse_order"] = json.dumps(["instrument", "data"])
    return path


def _first_quarter(data):
    data[:1000] = 1.0


def _one_small_chunk(data):
    """A chunk stored as a few bytes of deflate, which could not give back all it declares."""
    data.id.write_direct_chunk((0,), zlib.compress(bytes(16)))


@pytest.mark.parametrize("options, write, match", [
    (dict(data=np.array([b"text"])), None, "values of type"),
    (dict(data=np.arange(4.0), compression="lzf"), None, "compressed in a way"),
    (dict(shape=(4000,), chunks=(1000,), dtype="f8"), _first_quarter,
     "does not hold all of its data"),
    (dict(shape=(4000,), dtype="f8"), None, "does not hold all of its data"),
    (dict(shape=(10**6,), chunks=(10**6,), dtype="f8", compression="gzip"), _one_small_chunk,
     "does not hold all of its data"),
], ids=["text", "lzf", "partly written", "unwritten", "swelling chunk"])
def test_a_dataset_not_as_eclipse_writes_one_is_refused(tmp_path, options, write, match):
    with pytest.raises(ValueError, match=match):
        load_results(_crafted_dataset(tmp_path, write, **options))


def test_datasets_that_store_more_than_the_file_are_refused(tmp_path, monkeypatch):
    """As they would if several read the same stored data, which a crafted file could have."""
    path = save_results(tmp_path / "out.h5", {"data": np.arange(4000.0)}, compression=None)
    reading = results_file._Reading
    monkeypatch.setattr(results_file, "_Reading", lambda size: reading(size=16000))
    with pytest.raises(ValueError, match="shares what it stores with another dataset"):
        load_results(path)


def test_a_damaged_file_is_named(tmp_path):
    """HDF5's own error, as for a chunk that does not decompress, is given with the file's name, as the OSError it is."""
    path = _crafted_dataset(tmp_path, lambda data: data.id.write_direct_chunk((0,), bytes(100)),
                            shape=(100,), chunks=(100,), dtype="f8", compression="gzip")
    with pytest.raises(OSError, match="crafted.h5 could not be read"):
        load_results(path)


def test_a_virtual_dataset_is_refused(tmp_path):
    source = save_results(tmp_path / "source.h5", {"data": np.arange(4.0)})
    layout = h5py.VirtualLayout(shape=(4,), dtype="f8")
    layout[:] = h5py.VirtualSource(str(source), "data", shape=(4,))
    path = save_results(tmp_path / "virtual.h5", {"instrument": "SWC"})
    with h5py.File(path, "r+") as f:
        f.create_virtual_dataset("data", layout)
        f.attrs["eclipse_order"] = json.dumps(["instrument", "data"])
    with pytest.raises(ValueError, match="keeps its data in another file"):
        load_results(path)


@pytest.mark.parametrize("attribute, value, match", [
    ("thing", {"__eclipse__": "not_a_tag"}, "tagged 'not_a_tag'"),
    ("thing", {"__eclipse__": "resource", "value": "../../outside"}, "outside the package"),
    ("thing", {"__eclipse__": "resource", "value": "/etc/hosts"}, "outside the package"),
    ("thing", {"__eclipse__": "resource", "value": "C:outside"}, "outside the package"),
    ("thing", {"__eclipse__": "resource", "value": "data\\..\\..\\x"}, "outside the package"),
    ("thing", {"__eclipse__": "quantity"}, "not a results file this ECLIPSE can read"),
    ("thing", {"__eclipse__": "map", "items": [[["a"], 1]]}, "TypeError: .*unhashable"),
    ("thing", {"__eclipse__": "array", "dtype": "<U1", "value": [0, "x"]},
     "holds more than text"),
    ("thing", {"__eclipse__": "array", "dtype": "<U1", "value": ["", "xy"]},
     "holds more than text of its width"),
    ("thing", {"__eclipse__": "array", "dtype": "<f8", "shape": [2, 2], "value": [1.0, 2.0]},
     "does not have the shape it gives"),
], ids=["unknown tag", "resource outside", "resource absolute", "resource on a drive",
        "resource with backslashes", "missing entry", "unhashable key", "text and numbers",
        "text wider than it says", "wrong shape"])
def test_a_value_not_as_eclipse_writes_one_is_refused(tmp_path, attribute, value, match):
    with pytest.raises(ValueError, match=match):
        load_results(_with_attribute(tmp_path, attribute, value))


@pytest.mark.parametrize("name, value", [
    ("thing", np.array(['"a"', '"b"'], dtype=h5py.string_dtype())),
    ("thing", np.array([(1, 2)], dtype=[("a", "i4"), ("b", "i4")])[0]),
    ("format", np.array(["eclipse-results"] * 2, dtype=h5py.string_dtype())),
], ids=["many", "fields", "format"])
def test_an_attribute_that_is_not_one_value_is_refused_before_it_is_read(tmp_path, name, value):
    """Each element could refer to the same text elsewhere in the file, copied for each."""
    path = _with_attribute(tmp_path, "thing", "x")
    with h5py.File(path, "r+") as f:
        f.attrs[name] = value
    with pytest.raises(ValueError, match=f"attribute '{name}' of / is not one string or integer"):
        load_results(path)


def _with_cards(wcs, **cards):
    """The tree :func:`_jsonable` writes for *wcs*, with *cards* set in its header."""
    tree = results_file._jsonable(wcs)
    header = fits.Header.fromstring(tree["header"], sep="\n")
    header.update(cards)
    tree["header"] = header.tostring(sep="\n")
    return tree


@pytest.mark.filterwarnings("ignore:Some non-standard WCS keywords were excluded")
def test_a_wcs_is_read_back_without_what_a_results_file_does_not_hold(tmp_path):
    """A distortion, which is not written, and which astropy would tabulate at any size a header asks."""
    sky = WCS(naxis=2)
    sky.wcs.ctype, sky.wcs.cunit, sky.wcs.cdelt = ["HPLN-TAN", "HPLT-TAN"], ["arcsec"] * 2, [0.2, 0.2]
    tree = _with_cards(sky, A_ORDER=2, B_ORDER=2, A_2_0=0.001, B_0_2=0.001)
    assert WCS(fits.Header.fromstring(tree["header"], sep="\n")).sip is not None
    restored = load_results(_with_attribute(tmp_path, "wcs", tree))["wcs"]
    assert restored.sip is None and restored.wcs.cdelt.tolist() == [0.2, 0.2]

    # Nor its number of axes, which wcslib makes room for as the square of.
    tree = _with_cards(_wcs(), WCSAXES=5)
    assert WCS(fits.Header.fromstring(tree["header"], sep="\n")).naxis == 5
    assert load_results(_with_attribute(tmp_path, "wcs", tree))["wcs"].naxis == 3
    tree = _with_cards(_wcs(), WCSAXES=10**6)
    assert load_results(_with_attribute(tmp_path, "wcs", tree))["wcs"].naxis == 3
    with pytest.raises(ValueError, match="does not have its axes"):
        load_results(_with_attribute(tmp_path, "wcs", _with_cards(_wcs(), CTYPE9="EXTRA")))

    sky = WCS(naxis=2)
    sky.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    sky.sip = Sip(np.full((3, 3), 1e-3), np.full((3, 3), 1e-3), None, None, sky.wcs.crpix)
    with pytest.warns(UserWarning, match="does not hold a WCS's distortion"):
        path = save_results(tmp_path / "sky.h5", {"sky": sky})
    assert load_results(path)["sky"].sip is None


@pytest.mark.parametrize("payload, change, match", [
    ({"group": {"x": np.ones(2)}}, lambda f: f["group"].attrs.__setitem__("eclipse_type", "odd"),
     "a kind, 'odd'"),
    ({"group": {(1, 2): np.ones(2)}}, lambda f: f["group"].attrs.__delitem__("eclipse_keys"),
     "no 'eclipse_keys' attribute"),
    ({"group": [np.ones(2)]}, lambda f: f["group"].attrs.__delitem__("eclipse_length"),
     "no 'eclipse_length' attribute"),
    ({"group": [np.ones(2)]}, lambda f: f["group"].attrs.__setitem__("eclipse_length", 3),
     "has no member '1'"),
], ids=["unknown kind", "no keys", "no length", "missing member"])
def test_a_group_not_as_eclipse_writes_one_is_refused(tmp_path, payload, change, match):
    path = save_results(tmp_path / "out.h5", payload)
    with h5py.File(path, "r+") as f:
        change(f)
    with pytest.raises(ValueError, match=match):
        load_results(path)


def test_another_kind_of_eclipse_file_is_refused(tmp_path):
    with h5py.File(tmp_path / "synthesis.h5", "w") as f:
        f.attrs["format"] = "eclipse-synthesis"
        f.attrs["version"] = 1
    with pytest.raises(ValueError, match="is not an ECLIPSE results file") as error:
        load_results(tmp_path / "synthesis.h5")
    # Nor sent to docs that do not give the layout.
    assert "documentation" not in str(error.value)


def test_every_value_is_strict_json(tmp_path):
    """So that a reader in another language parses it, NaN and infinity included."""
    payload = {"nan": float("nan"), "inf": -np.inf * u.s, "list": [1.0, float("inf")],
               "array": np.array(["a"]), "quantities": [np.nan * u.m],
               "group": {"data": np.ones(3), "nan": float("nan"), (1, 2): {"x": np.ones(2)}},
               "cube": _cube()}
    path = save_results(tmp_path / "out.h5", payload)

    def refuse(constant):
        raise ValueError(f"{constant} is not JSON")

    # Every attribute but those of plain text or a number.
    plain = {"format", "version", "unit", "eclipse_type", "eclipse_length", "target"}
    checked = []
    with h5py.File(path, "r") as f:
        nodes = [f]
        f.visititems(lambda name, node: nodes.append(node))
        for node in nodes:
            for name, text in node.attrs.items():
                if name not in plain:
                    json.loads(text, parse_constant=refuse)
                    checked.append(f"{node.name.rstrip('/')}/{name}")
    assert {"/group/1", "/group/eclipse_keys", "/cube/wcs", "/cube/meta"} <= set(checked)
    out = load_results(path)
    assert np.isnan(out["nan"]) and out["inf"] == -np.inf * u.s
    assert out["list"][1] == float("inf") and np.isnan(out["quantities"][0].value)
    assert np.isnan(out["group"]["nan"])


def test_whole_numbers_with_a_unit_stay_whole(tmp_path):
    counts = NDCube(np.arange(6, dtype=np.uint16).reshape(2, 3), wcs=WCS(naxis=2),
                    unit=u.photon)
    steps = u.Quantity(np.arange(3), u.pix, dtype=None)
    out = _round_trip(tmp_path, {"counts": counts, "steps": steps})
    assert out["counts"].data.dtype == np.uint16 and out["counts"].unit == u.photon
    assert steps.dtype.kind == "i" and out["steps"].dtype == steps.dtype


def test_a_sweep_of_many_combinations_is_written(tmp_path):
    """Its keys make one large attribute, which a plain HDF5 object header could not hold."""
    keys = [tuple((f"detector.parameter_{i}", float(j)) for i in range(40)) for j in range(100)]
    combinations = {key: {"data": np.full(3, j)} for j, key in enumerate(keys)}
    out = _round_trip(tmp_path, {"results": {"all_combinations": combinations}})
    assert list(out["results"]["all_combinations"]) == keys


def test_what_save_results_is_given_is_checked_first(tmp_path, monkeypatch):
    with pytest.raises(IsADirectoryError):
        save_results(tmp_path, {"instrument": "SWC"})
    with pytest.raises(ValueError, match="'gzip' or None"):
        save_results(tmp_path / "out.h5", {"instrument": "SWC"}, compression="lzf")
    with pytest.raises(TypeError, match="must be a mapping"):
        save_results(tmp_path / "out.h5", ["ab", "cd"])
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setenv("HOME", str(tmp_path))
    save_results("~/home.h5", {"instrument": "SWC"})
    assert load_results("~/home.h5")["instrument"] == "SWC"


def test_another_save_of_the_same_name_is_left_alone(tmp_path, monkeypatch):
    """Two runs saving one name at once write their own partial files, even if they draw the same name."""
    draws = iter(["aaaa", "bbbb"])
    monkeypatch.setattr("euvst_response.atmosphere.secrets.token_hex", lambda size: next(draws))
    theirs = tmp_path / "out.h5.aaaa.part"
    theirs.write_bytes(b"another run's")
    path = save_results(tmp_path / "out.h5", {"instrument": "SWC"})
    assert theirs.read_bytes() == b"another run's"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.h5", "out.h5.aaaa.part"]
    assert load_results(path)["instrument"] == "SWC"


def test_two_saves_of_one_name_at_once_do_not_meet(tmp_path, monkeypatch):
    """Another save of the same name starts and finishes while this one writes."""
    write, started = results_file._put_dict, []

    def meanwhile(*args):
        if not started:
            started.append(True)
            save_results(tmp_path / "out.h5", {"instrument": "EIS"})
        return write(*args)

    monkeypatch.setattr(results_file, "_put_dict", meanwhile)
    path = save_results(tmp_path / "out.h5", {"instrument": "SWC"})
    assert started and load_results(path)["instrument"] == "SWC"
    assert [p.name for p in tmp_path.iterdir()] == ["out.h5"]


def test_what_a_results_file_cannot_hold_is_said(tmp_path):
    masked = NDCube(np.ones((2, 3)), wcs=WCS(naxis=2), mask=np.zeros((2, 3), bool))
    with pytest.warns(UserWarning, match="/cube: a results file does not hold a cube.s mask"):
        out = _round_trip(tmp_path, {"cube": masked})
    assert out["cube"].mask is None
    with pytest.raises(TypeError, match="cannot be written"):
        save_results(tmp_path / "long.h5", {"value": np.longdouble(1.5)})


def test_a_missing_or_foreign_file_is_named_for_what_it_is(tmp_path):
    import dill

    with open(tmp_path / "run.pkl", "wb") as f:
        dill.dump({"instrument": "SWC"}, f)
    with pytest.raises(FileNotFoundError, match="convert_results_pickle"):
        load_results(tmp_path / "run.h5")
    (tmp_path / "notes.txt").write_text("instrument: SWC\n")
    with pytest.raises(ValueError, match="neither a results file nor a results pickle"):
        load_results(tmp_path / "notes.txt")


def test_what_the_results_hold_in_more_than_one_place_is_written_once(tmp_path):
    """As a ground truth the combinations share, or the last raster cube kept as cube_sim."""
    shared, cube = {"fit": np.arange(2000.0)}, _cube()
    payload = {"a": {"truth": shared, "cube": cube}, "b": {"truth": shared}, "cube_sim": cube}
    path = save_results(tmp_path / "out.h5", payload)
    datasets = []
    with h5py.File(path, "r") as f:
        f.visititems(lambda name, node: datasets.append(name)
                     if isinstance(node, h5py.Dataset) else None)
    assert sorted(datasets) == ["a/cube/data", "a/truth/fit"]
    out = load_results(path)
    assert out["a"]["truth"] is out["b"]["truth"] and out["a"]["cube"] is out["cube_sim"]
    assert np.array_equal(out["b"]["truth"]["fit"], shared["fit"])


@pytest.mark.parametrize("target, match", [
    ("/a/missing", "which the file does not hold"),
    ("/b", "which it is inside"),
    ("/ext/x", "which the file does not hold"),
    ("nowhere", "refers to nothing"),
    ("/a//x", "which the file does not hold"),
    ("/a/./x", "which the file does not hold"),
], ids=["missing", "inside itself", "through a link", "not a path", "empty part", "dot"])
def test_a_reference_is_only_to_what_the_file_holds(tmp_path, target, match):
    save_results(tmp_path / "other.h5", {"x": np.ones(2)})
    path = save_results(tmp_path / "out.h5", {"a": {"x": np.ones(2)}, "b": {"y": np.ones(2)}})
    with h5py.File(path, "r+") as f:
        del f["b"]["y"]
        reference = f["b"].create_group("y")
        reference.attrs["eclipse_type"] = "ref"
        reference.attrs["target"] = target
        f["ext"] = h5py.ExternalLink("other.h5", "/")
        f.attrs["eclipse_order"] = json.dumps(["a", "b", "ext"])
    with pytest.raises(ValueError, match=match):
        load_results(path)


def test_the_numbers_of_a_wcs_come_back_exactly(tmp_path):
    """A header holds 14 digits, which 0.2 arcsec in degrees is not."""
    wcs = _wcs()
    wcs.wcs.cunit = ["cm", "deg", "deg"]
    wcs.wcs.cdelt = [1.69e-11, (0.2 * u.arcsec).to_value(u.deg), (0.159 * u.arcsec).to_value(u.deg)]
    wcs.wcs.crpix = [2.5, 85.83333333333333, 1.5]
    restored = _round_trip(tmp_path, {"wcs": wcs})["wcs"]
    for name in ("cdelt", "crpix", "crval"):
        assert list(getattr(restored.wcs, name)) == list(getattr(wcs.wcs, name))


@pytest.mark.filterwarnings("ignore:cdelt will be ignored since cd is present")
def test_a_rotated_wcs_or_one_given_by_a_cd_matrix_comes_back_exactly(tmp_path):
    rotated = _wcs()
    angle = np.deg2rad(7.0)
    rotated.wcs.pc = [[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)],
                      [0.0, np.sin(angle), np.cos(angle)]]
    matrix = WCS(naxis=2)
    matrix.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
    matrix.wcs.cunit = ["arcsec", "arcsec"]
    matrix.wcs.cd = [[0.2, 0.013], [-0.011, 0.159]]
    matrix.wcs.crpix = [85.83333333333333, 1.5]
    # With both, wcslib takes the PC matrix.
    both = WCS(naxis=2)
    both.wcs.ctype, both.wcs.cunit = matrix.wcs.ctype, matrix.wcs.cunit
    both.wcs.cdelt, both.wcs.pc, both.wcs.cd = [0.2, 0.159], [[1.0, 0.1], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]
    # As an older solar FITS header can give it.
    crota = WCS(naxis=2)
    crota.wcs.ctype, crota.wcs.cunit = matrix.wcs.ctype, matrix.wcs.cunit
    crota.wcs.cdelt, crota.wcs.crota = [0.6, 0.6], [0.0, 12.0]
    out = _round_trip(tmp_path, {"rotated": rotated, "matrix": matrix, "both": both,
                                 "crota": crota})
    assert out["crota"].wcs.has_crota() and out["crota"].wcs.crota.tolist() == [0.0, 12.0]
    for name, before in (("both", both), ("crota", crota)):
        assert np.allclose(out[name].pixel_to_world_values(10, 20),
                           before.pixel_to_world_values(10, 20), rtol=0, atol=1e-12), name
    assert out["rotated"].wcs.pc.tolist() == rotated.wcs.pc.tolist()
    assert out["rotated"].wcs.cdelt.tolist() == rotated.wcs.cdelt.tolist()
    assert out["matrix"].wcs.has_cd() and out["matrix"].wcs.cd.tolist() == matrix.wcs.cd.tolist()
    assert list(out["matrix"].wcs.cunit) == list(matrix.wcs.cunit)
    assert np.allclose(out["matrix"].pixel_to_world_values(10, 20),
                       matrix.pixel_to_world_values(10, 20), rtol=0, atol=1e-12)


def test_a_file_that_holds_no_mapping_is_refused(tmp_path):
    path = save_results(tmp_path / "out.h5", {"instrument": "SWC"})
    with h5py.File(path, "r+") as f:
        f.attrs["eclipse_type"] = "list"
        f.attrs["eclipse_length"] = 0
    with pytest.raises(ValueError, match="holds a list, not a mapping"):
        load_results(path)


def _combinations(first, second):
    """Results of two combinations, with the DN signal data and unit of each given."""
    return {"results": {"all_combinations": {
        ("a",): {"first_signal_wcs": _wcs(), "first_dn_signal_data": first[0],
                 "first_dn_signal_unit": first[1], "first_photon_signal_data": np.ones((2, 3, 4)),
                 "first_photon_signal_unit": u.photon},
        ("b",): {"first_signal_wcs": _wcs(), "first_dn_signal_data": second[0],
                 "first_dn_signal_unit": second[1], "first_photon_signal_data": np.ones((2, 3, 4)),
                 "first_photon_signal_unit": u.photon},
    }}}


def test_the_simulation_results_are_checked_as_they_are_put_together(tmp_path):
    """Signals are multiplied by units only, each signal once, and what is missing is named."""
    from euvst_response.analysis import load_instrument_response_results

    signal = np.arange(24.0).reshape(2, 3, 4)
    path = save_results(tmp_path / "out.h5", _combinations((signal, u.DN), (signal * 2, u.DN)))
    results = load_instrument_response_results(path)["results"]["all_combinations"]
    assert results[("b",)]["first_dn_signal"].data.tolist() == (signal * 2).tolist()
    assert results[("b",)]["first_dn_signal"].unit == u.DN

    # Held once by the file, as a crafted file could for any number of combinations.
    path = save_results(tmp_path / "out.h5", _combinations((signal, u.DN), (signal, u.DN)))
    results = load_instrument_response_results(path)["results"]["all_combinations"]
    assert np.shares_memory(results[("a",)]["first_dn_signal"].data,
                            results[("b",)]["first_dn_signal"].data)
    assert results[("b",)]["first_dn_signal"].data.tolist() == signal.tolist()
    path = save_results(tmp_path / "out.h5",
                        _combinations((signal, u.DN), (signal * 2, np.ones((2, 3, 4)))))
    with pytest.raises(ValueError, match="out.h5: the dn signal.s unit is of type ndarray, not a unit"):
        load_instrument_response_results(path)
    path = save_results(tmp_path / "out.h5", {"results": {"all_combinations": {("a",): {}}}})
    with pytest.raises(ValueError, match="does not hold the results of an instrument simulation: "
                                         "KeyError"):
        load_instrument_response_results(path)


def test_numbers_keep_their_numpy_types(tmp_path):
    """As the keys of a cube's sampling have them, and a setting given in single precision."""
    values = {"f32": np.float32(0.76), "f64": np.float64(0.2), "i32": np.int32(26),
              "flag": np.bool_(True), "zero_d": np.array(5.0), "count": u.Quantity(3, u.pix, dtype=int),
              "single": np.float32(1.5) * u.s, "keys": {(np.float64(0.2), 1): np.zeros(3)}}
    out = _round_trip(tmp_path, values)
    for name in ("f32", "f64", "i32", "flag"):
        assert type(out[name]) is type(values[name]) and out[name] == values[name], name
    assert type(out["zero_d"]) is np.ndarray and out["zero_d"].shape == () and out["zero_d"] == 5.0
    for name in ("count", "single"):
        assert out[name].dtype == values[name].dtype and out[name] == values[name], name
    (key,) = out["keys"]
    assert type(key[0]) is np.float64 and key == (0.2, 1)


def test_a_unit_keeps_its_whole_scale(tmp_path):
    units = {"third": u.CompositeUnit(1 / 3, [u.m], [1]), "arcsec": u.arcsec.decompose()}
    out = _round_trip(tmp_path, {**units, "quantity": 2.0 * units["third"]})
    for name, unit in units.items():
        assert out[name] == unit and out[name].scale == unit.scale, name
    assert out["quantity"].unit.scale == 1 / 3


@pytest.mark.parametrize("value, match", [
    (1.0 * u.def_unit("widget_for_a_test"), "would not read back as itself"),
    (u.Magnitude(3.0), "is not a plain unit"),
    (np.ma.masked_array([1.0, 2.0], mask=[False, True]), "masked array"),
    (np.datetime64("2020-01-01T00:00:00.000000001"), "datetime64"),
    (np.datetime64, "type datetime64"),
    (np.complex128(1 + 2j), "complex128"),
], ids=["unit of a script's own", "function unit", "masked array", "numpy date", "numpy date type",
        "complex number"])
def test_what_would_not_read_back_is_refused_when_written(tmp_path, value, match):
    """With where it is, rather than written as something that cannot be read or reads as another value."""
    with pytest.raises(TypeError, match=f"/group/value: .*{match}"):
        save_results(tmp_path / "out.h5", {"group": {"value": value, "data": np.ones(3)}})
    assert list(tmp_path.iterdir()) == []


def test_a_cube_or_value_that_cannot_be_written_is_said_where_it_is(tmp_path):
    loop = [1]
    loop.append(loop)
    with pytest.raises(TypeError, match="/loop: A value that holds itself"):
        save_results(tmp_path / "out.h5", {"loop": loop})
    # A sliced cube, whose WCS is a view of another.
    with warnings.catch_warnings(), pytest.raises(TypeError, match="/cube/wcs: Values of type"):
        warnings.simplefilter("ignore")
        save_results(tmp_path / "out.h5", {"cube": _cube()[0]})


def test_names_hdf5_cannot_hold_are_kept_as_keys(tmp_path):
    """A NUL, which HDF5 would cut the name at, or text UTF-8 has no form for."""
    payload = {"group": {"a\x00b": np.ones(2), "a": np.zeros(2), "\ud800": 1}}
    out = _round_trip(tmp_path, payload)["group"]
    assert list(out) == ["a\x00b", "a", "\ud800"]
    assert np.array_equal(out["a\x00b"], np.ones(2)) and out["\ud800"] == 1


def test_a_path_the_reader_would_not_follow_is_kept_as_given(tmp_path):
    """The package itself, say, which is no file in it."""
    root = results_file._package_root()
    out = _round_trip(tmp_path, {"root": root, "table": root / "data" / "throughput" / "source.txt"})
    assert out["root"] == root and out["table"] == root / "data" / "throughput" / "source.txt"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes over read-only files")
def test_a_read_only_results_file_is_not_written_over(tmp_path):
    path = save_results(tmp_path / "out.h5", {"instrument": "SWC"})
    path.chmod(0o444)
    with pytest.raises(PermissionError, match="read-only"):
        save_results(path, {"instrument": "EIS"})
    assert load_results(path)["instrument"] == "SWC"


def test_a_long_name_is_written(tmp_path):
    """Its partial file takes no more of the name than a filesystem allows."""
    path = save_results(tmp_path / ("x" * 240 + ".h5"), {"instrument": "SWC"})
    assert load_results(path)["instrument"] == "SWC"
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_odd_paths_are_named_for_what_they_are(tmp_path, monkeypatch):
    import dill

    (tmp_path / "x.h5").mkdir()
    for path in (tmp_path / "x.h5", tmp_path, tmp_path / "."):
        with pytest.raises(IsADirectoryError):
            load_results(path)
    with open(tmp_path / "run.pickle", "wb") as f:
        dill.dump({"instrument": "SWC"}, f)
    with pytest.raises(FileNotFoundError, match="run.pickle, beside it, is a results pickle"):
        load_results(tmp_path / "run.h5")
    monkeypatch.setenv("HOME", str(tmp_path))
    save_results(tmp_path / "home.h5", {"instrument": "SWC"})
    assert is_results_file("~/home.h5")


class _Gone:
    """A class a pickle names, which a later version no longer has."""


def test_a_pickle_of_a_class_this_version_lacks_is_named(tmp_path, monkeypatch):
    import sys

    import dill

    with open(tmp_path / "old.pkl", "wb") as f:
        dill.dump({"results": {"all_combinations": {}}, "thing": _Gone()}, f)
    monkeypatch.delattr(sys.modules[__name__], "_Gone")
    with pytest.warns(FutureWarning), pytest.raises(ValueError, match="cannot be unpickled"):
        load_results(tmp_path / "old.pkl")
    with pytest.raises(ValueError, match="cannot be unpickled"):
        results_file.convert_results_pickle(tmp_path / "old.pkl")


def test_a_wcs_keeps_its_time_and_its_observer(tmp_path):
    """The file promises them back; a sunpy map's WCS carries both."""
    wcs = _wcs()
    wcs.wcs.dateobs = "2024-03-20T00:00:00"
    wcs.wcs.aux.hgln_obs = 0.0
    wcs.wcs.aux.hglt_obs = 7.1
    wcs.wcs.aux.dsun_obs = 1.4e11
    wcs.wcs.aux.rsun_ref = 695700000.0
    save_results(tmp_path / "out.h5", {"wcs": wcs})
    back = load_results(tmp_path / "out.h5")["wcs"]
    assert back.wcs.dateobs.startswith("2024-03-20T00:00:00")
    assert (back.wcs.aux.hgln_obs, back.wcs.aux.hglt_obs, back.wcs.aux.dsun_obs,
            back.wcs.aux.rsun_ref) == pytest.approx((0.0, 7.1, 1.4e11, 695700000.0))
