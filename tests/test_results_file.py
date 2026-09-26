"""Results survive a round trip through the results file, and old pickles still load.

The tests that matter here are the ones that compare against the object that
went in, field by field, because a format change that loses something does
not announce itself: the file still opens and the numbers still look like
numbers.
"""
import dataclasses
import datetime
import json
import warnings

import astropy.units as u
import h5py
import numpy as np
import pytest
import yaml
from astropy.wcs import WCS
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


def _cube():
    return NDCube(np.arange(24, dtype=float).reshape((2, 3, 4)),
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
        "cube_reb_dict": {(0.2, 1): _cube(), (0.4, 2): _cube()},
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


def test_a_view_is_written_without_the_array_it_views(tmp_path):
    """A slice of the Monte Carlo stack, as the first fit is, is written without the rest of it."""
    stack = np.random.default_rng(1).random((200, 50, 50))
    path = save_results(tmp_path / "view.h5", {"first": stack[0]}, compression=None)
    assert path.stat().st_size < 100_000
    assert np.array_equal(load_results(path)["first"], stack[0])


def test_a_failed_write_leaves_the_file_there_as_it_was(tmp_path):
    path = save_results(tmp_path / "out.h5", {"instrument": "SWC"})
    with pytest.raises(TypeError, match="cannot be written"):
        save_results(path, {"instrument": "EIS", "odd": {1, 2}})
    assert load_results(path)["instrument"] == "SWC"
    assert list(tmp_path.iterdir()) == [path]


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
    assert restored.wcs.cdelt == pytest.approx(_wcs().wcs.cdelt, rel=1e-12)
    assert restored.wcs.crpix == pytest.approx(_wcs().wcs.crpix, rel=1e-12)


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
    if isinstance(before, list):
        return (len(after) == len(before)
                and all(_same(b, a) for b, a in zip(before, after)))
    if isinstance(before, u.Quantity):
        return after.unit == before.unit and np.array_equal(after.value,
                                                            before.value)
    return after == before


@pytest.mark.parametrize("config_object", [
    # Every field away from its default, so a field that is dropped on the
    # way out cannot pass by being rebuilt from the default on the way in.
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
], ids=lambda obj: type(obj).__name__)
def test_every_config_field_survives(tmp_path, config_object):
    """Every field, Simulation.noise included, comes back unchanged.

    A run with the noise off that read back with it on would be mistaken for
    a noisy one.
    """
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


def test_a_pkl_name_is_corrected_rather_than_written(tmp_path):
    """A file named .pkl that is not a pickle is worse than a renamed one."""
    with pytest.warns(UserWarning, match="an HDF5 file"):
        path = save_results(tmp_path / "result.pkl", {"instrument": "SWC"})
    assert path.name == "result.h5"
    assert path.exists()
    assert not (tmp_path / "result.pkl").exists()


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


def test_the_format_is_detected_from_content_not_the_name(tmp_path):
    """A pickle named .h5 must not be read as HDF5, and the reverse."""
    import dill

    misnamed = tmp_path / "actually_a_pickle.h5"
    with open(misnamed, "wb") as handle:
        dill.dump({"instrument": "EIS"}, handle)

    assert not is_results_file(misnamed)
    with pytest.warns(FutureWarning, match="results pickle"):
        assert load_results(misnamed)["instrument"] == "EIS"

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
    assert out["flag"] is True
    assert out["count"] == 7
    assert out["value"] == pytest.approx(1.5)


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


def test_what_a_configuration_object_worked_out_is_kept_as_the_run_had_it(tmp_path):
    """A detector's dark current, as the run used it, whatever this version works out."""
    detector = Detector_SWC()
    as_run = 42 * detector.dark_current.unit
    object.__setattr__(detector, "dark_current", as_run)
    assert _round_trip(tmp_path, {"detector": detector})["detector"].dark_current == as_run


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


def test_another_save_of_the_same_name_is_left_alone(tmp_path):
    """Two runs saving one name at once write their own partial files."""
    theirs = tmp_path / "out.h5.part"
    theirs.write_bytes(b"another run's")
    path = save_results(tmp_path / "out.h5", {"instrument": "SWC"})
    assert theirs.read_bytes() == b"another run's"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.h5", "out.h5.part"]
    assert load_results(path)["instrument"] == "SWC"


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
