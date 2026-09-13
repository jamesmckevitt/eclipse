"""Results survive a round trip through ASDF, and old pickles still load.

The tests that matter here are the ones that compare against the object that
went in, field by field, because a format change that loses something does
not announce itself: the file still opens and the numbers still look like
numbers.
"""
import warnings

import asdf
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube

from euvst_response.config import (AluminiumFilter, Detector_SWC, Simulation,
                                   Telescope_EIS, Telescope_EUVST)
from euvst_response.fitting import FitComponent, FitConfig
from euvst_response.io import is_asdf, load_results, save_results

REST = 195.119 * u.Angstrom


def _wcs():
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLT-TAN", "HPLN-TAN"]
    wcs.wcs.cunit = ["cm", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [1.69e-11, 0.159, 0.2]
    wcs.wcs.crpix = [4.0, 2.5, 1.0]
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
    path = save_results(tmp_path / "out.asdf", payload or _payload(), **kwargs)
    return load_results(path)


def test_the_file_is_asdf_not_a_pickle(tmp_path):
    path = save_results(tmp_path / "out.asdf", _payload())
    assert is_asdf(path)
    with asdf.open(path) as af:
        assert af["instrument"] == "SWC"
        assert af["eclipse_format_version"] == 1


def test_scalars_and_strings_survive(tmp_path):
    out = _round_trip(tmp_path)
    assert out["instrument"] == "SWC"
    assert out["software_version"] == "0.7.0"
    assert out["config"]["n_iter"] == 5
    assert out["config"]["simulation"]["expos"] == ["5 s", "20 s"]


def test_the_cube_comes_back_whole(tmp_path):
    out = _round_trip(tmp_path)
    original, restored = _cube(), out["cube_sim"]

    assert np.array_equal(restored.data, original.data)
    assert restored.unit == original.unit
    assert restored.meta["line_name"] == "Fe12_195.1190"
    assert restored.meta["rest_wav"] == REST
    assert restored.meta["uniform_mode"] is False


def test_the_wcs_keeps_the_units_it_was_written_in(tmp_path):
    """asdf-astropy normalises a tagged WCS through SI; a header does not.

    Without this, a wavelength axis written in cm comes back in m. The
    coordinates are the same, but anything reading wcs.wcs.cdelt directly
    would be out by a factor of a hundred.
    """
    out = _round_trip(tmp_path)
    restored = out["cube_sim"].wcs

    assert list(restored.wcs.ctype) == ["WAVE", "HPLT-TAN", "HPLN-TAN"]
    assert [str(c) for c in restored.wcs.cunit] == ["cm", "arcsec", "arcsec"]
    assert restored.wcs.cdelt == pytest.approx(_wcs().wcs.cdelt, rel=1e-12)
    assert restored.wcs.crpix == pytest.approx(_wcs().wcs.crpix, rel=1e-12)


def test_the_wcs_describes_the_same_world_coordinates(tmp_path):
    """The property that actually matters, whatever the keywords say."""
    out = _round_trip(tmp_path)
    before = _cube().axis_world_coords(2)[0].to_value(u.cm)
    after = out["cube_sim"].axis_world_coords(2)[0].to_value(u.cm)
    assert np.allclose(after, before, rtol=1e-12, atol=0.0)


def test_tuple_keyed_dicts_survive(tmp_path):
    """ASDF only allows str, int and bool keys; results are keyed by tuple."""
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
    """ASDF memory-maps by default, which would leave dangling arrays."""
    out = _round_trip(tmp_path)
    array = out["cube_sim"].data
    assert isinstance(array, np.ndarray)
    assert float(array.sum()) == pytest.approx(276.0)


def test_uncompressed_writing_works(tmp_path):
    out = _round_trip(tmp_path, compression=None)
    assert np.array_equal(out["cube_sim"].data, _cube().data)


def test_compression_actually_shrinks_a_large_array(tmp_path):
    payload = {"big": np.zeros((60, 60, 60))}
    small = save_results(tmp_path / "z.asdf", payload, compression="zlib")
    big = save_results(tmp_path / "n.asdf", payload, compression=None)
    assert small.stat().st_size < big.stat().st_size / 10


def test_a_pkl_name_is_corrected_rather_than_written(tmp_path):
    """A file named .pkl that is not a pickle is worse than a renamed one."""
    with pytest.warns(UserWarning, match="ASDF"):
        path = save_results(tmp_path / "result.pkl", {"instrument": "SWC"})
    assert path.name == "result.asdf"
    assert path.exists()
    assert not (tmp_path / "result.pkl").exists()


def test_an_old_pickle_still_loads(tmp_path):
    """Existing result files have to keep working, whatever they are named."""
    import dill

    path = tmp_path / "old.pkl"
    with open(path, "wb") as handle:
        dill.dump({"instrument": "EIS", "cube": _cube()}, handle)

    with pytest.warns(UserWarning, match="pickle"):
        out = load_results(path)

    assert out["instrument"] == "EIS"
    assert np.array_equal(out["cube"].data, _cube().data)


def test_the_format_is_detected_from_content_not_the_name(tmp_path):
    """A pickle named .asdf must not be read as ASDF, and the reverse."""
    import dill

    misnamed = tmp_path / "actually_a_pickle.asdf"
    with open(misnamed, "wb") as handle:
        dill.dump({"instrument": "EIS"}, handle)

    assert not is_asdf(misnamed)
    with pytest.warns(UserWarning, match="pickle"):
        assert load_results(misnamed)["instrument"] == "EIS"


def test_a_file_cannot_name_an_arbitrary_class(tmp_path):
    """The point of leaving pickle: reading must not construct what it likes."""
    path = tmp_path / "hostile.asdf"
    af = asdf.AsdfFile({
        "eclipse_format_version": 1,
        "thing": {"__eclipse__": "dataclass", "class": "os.system",
                  "fields": {}},
    })
    af.write_to(str(path))

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
    """np.bool_ has no ASDF representation and turns up in parsed configs."""
    out = _round_trip(tmp_path, {"flag": np.bool_(True),
                                 "count": np.int64(7),
                                 "value": np.float32(1.5)})
    assert out["flag"] is True
    assert out["count"] == 7
    assert out["value"] == pytest.approx(1.5)
