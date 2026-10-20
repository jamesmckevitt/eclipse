"""Instrument facts set in the configuration: defaults, checks, old names and results files."""
import copy
import copyreg
import dataclasses
import json
import pickle
import sys
import warnings

import astropy.constants as const
import astropy.units as u
import h5py
import numpy as np
import pytest
import yaml

from euvst_response import results_file
from euvst_response.config import (DETECTOR_MATERIALS, DISPERSION_COEFFICIENTS,
                                   MEASURED_SLIT_IMAGE_TILT, AluminiumFilter, Detector_EIS,
                                   Detector_SWC, Simulation)
from euvst_response.main import _validate_config_keys, main
from euvst_response.radiometric import electrons_per_photon
from euvst_response.readout import FocalPlane_SWC
from euvst_response.results_file import load_results, save_results
from euvst_response.utils import _parse_section, load_yaml_config

UNIFORM = {"instrument": "SWC", "n_iter": 1, "uniform_intensity": "5000 erg / (s cm2 sr)"}
DARK = u.electron / (u.pixel * u.s)


def _run(tmp_path, monkeypatch, config):
    """Run `config` and return the payload it saves."""
    main_module = sys.modules["euvst_response.main"]
    saved, save = {}, main_module.save_results

    def keep(path, payload, **kwargs):
        saved["payload"] = payload
        return save(path, payload, **kwargs)

    monkeypatch.setattr(main_module, "save_results", keep)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(path)])
    monkeypatch.chdir(tmp_path)
    main()
    return saved["payload"]


# --- defaults --------------------------------------------------------------------------

@pytest.mark.parametrize("det", [Detector_SWC(), Detector_EIS(),
                                 Detector_SWC(ccd_temperature=-100 * u.deg_C)])
def test_the_default_pair_energy_is_the_silicon_law(det):
    """w(T) = 3.71 - 0.0006 (T - 300) eV, to the last bit."""
    temp_k = det.ccd_temperature.to_value(u.K, equivalencies=u.temperature())
    wavelength = np.linspace(170, 212, 7) * u.AA
    photon_ev = (const.h * const.c / wavelength).to_value(u.eV)
    np.testing.assert_array_equal(electrons_per_photon(wavelength, det),
                                  photon_ev / (3.71 - 0.0006 * (temp_k - 300.0)))
    assert det.si_fano == 0.115 == DETECTOR_MATERIALS["silicon"]["fano_factor"]
    # Filled in from the material, so a run records the values it used.
    assert det.fano_factor == 0.115 and det.pair_energy_300k == 3.71 * u.eV


def test_electrons_per_photon_takes_any_object_with_a_ccd_temperature():
    from types import SimpleNamespace

    temperature = -60 * u.deg_C
    np.testing.assert_array_equal(
        electrons_per_photon(195 * u.AA, SimpleNamespace(ccd_temperature=temperature)),
        electrons_per_photon(195 * u.AA, Detector_SWC(ccd_temperature=temperature)))
    with pytest.raises(ValueError, match="no ccd_temperature"):
        electrons_per_photon(195 * u.AA, SimpleNamespace())


def test_the_default_filter_and_pinhole_light_are_unchanged():
    assert AluminiumFilter().visible_light_throughput() == 10.0 ** (-1485 / 170.0) * 0.8
    assert Simulation().pinhole_visible_wavelength == 600 * u.nm


def test_the_default_focal_plane_is_the_detectors():
    det = Detector_SWC()
    assert det.dispersion == DISPERSION_COEFFICIENTS and det.slit_image_tilt == (0.0, 0.0)
    assert det.wavelength_offset == 0 * u.AA
    assert det.lit_band == (170.0 * u.AA, 212.3 * u.AA)
    assert FocalPlane_SWC.from_detector(det) == FocalPlane_SWC()
    # The constants can still be imported from the read-out module.
    from euvst_response.readout import DISPERSION_COEFFICIENTS as from_readout
    assert from_readout is DISPERSION_COEFFICIENTS


# --- the settings take effect ----------------------------------------------------------

def test_the_material_settings_replace_the_materials():
    det = Detector_SWC(fano_factor=0.12, pair_energy_300k=3.65 * u.eV,
                       pair_energy_slope=0 * u.eV / u.K)
    assert det.si_fano == 0.12 and det.pair_energy == 3.65 * u.eV
    ratio = electrons_per_photon(195 * u.AA, det) / electrons_per_photon(195 * u.AA, Detector_SWC())
    assert ratio == pytest.approx(Detector_SWC().pair_energy.value / 3.65, rel=1e-14)
    assert Detector_EIS(fano_factor=0.2).si_fano == 0.2


def test_the_visible_decade_thickness_sets_the_filters_visible_transmission():
    flt = AluminiumFilter(visible_decade_thickness=297 * u.AA)
    assert flt.visible_light_throughput() == pytest.approx(10 ** -5 * 0.8, rel=1e-12)


def test_the_focal_plane_settings_reach_the_focal_plane():
    det = Detector_SWC(dispersion=[199.0, 1.25], wavelength_offset=0.1 * u.AA,
                       lit_band=[180 * u.AA, 190 * u.AA], slit_image_tilt=MEASURED_SLIT_IMAGE_TILT)
    fp = FocalPlane_SWC.from_detector(det)
    assert fp.dispersion == (199.0, 1.25) and fp.wavelength_offset == 0.1 * u.AA
    assert fp.lit_band == (180 * u.AA, 190 * u.AA)
    assert fp.slit_image_tilt == MEASURED_SLIT_IMAGE_TILT
    # A setting given to from_detector replaces the detector's.
    assert FocalPlane_SWC.from_detector(det, slit_image_tilt=(0.0, 0.0)).slit_image_tilt == (0, 0)


@pytest.mark.parametrize("lit_band", [[170.0, 212.3] * u.AA, ("170 Angstrom", "212.3 Angstrom"),
                                      [17 * u.nm, 212.3 * u.AA], (212.3 * u.AA, 170 * u.AA),
                                      np.array([170.0, 212.3]) * u.AA])
def test_the_focal_plane_takes_any_lit_band_that_quantity_reads(lit_band):
    fp = FocalPlane_SWC(lit_band=lit_band)
    assert all(isinstance(limit, u.Quantity) and limit.isscalar for limit in fp.lit_band)
    for ccd in ("left", "right"):
        assert fp.lit_rows(ccd) == FocalPlane_SWC().lit_rows(ccd)


@pytest.mark.parametrize("offset, shift", [(0, 0.0), (np.float64(0.0), 0.0),
                                           ([0.1] * u.AA, 0.1), (0.01 * u.nm, 0.1),
                                           ("0.1 Angstrom", 0.1)])
def test_the_focal_plane_takes_any_wavelength_offset_that_adds_to_a_wavelength(offset, shift):
    fp = FocalPlane_SWC(wavelength_offset=offset)
    assert fp.wavelength_offset.isscalar
    moved = fp.wavelength(100, "left") - FocalPlane_SWC().wavelength(100, "left")
    assert moved.to_value(u.AA) == pytest.approx(shift, abs=1e-12)


def test_a_scalar_dispersion_is_refused_with_a_message():
    with pytest.raises(ValueError, match="focal_plane.dispersion must be two or more"):
        FocalPlane_SWC(dispersion=199.0)


@pytest.mark.parametrize("build, message", [
    (lambda: Detector_SWC(fano_factor=-0.1), "detector.fano_factor must be a number, zero or more"),
    (lambda: Detector_EIS(fano_factor="0.1"), "detector.fano_factor must be a number"),
    (lambda: Detector_SWC(pair_energy_300k=3.71), "pair_energy_300k must be in a unit of energy"),
    (lambda: Detector_SWC(pair_energy_slope=1 * u.eV), "pair_energy_slope must be in a unit"),
    (lambda: Detector_SWC(pair_energy_300k=-1 * u.eV), "must be above zero"),
    (lambda: Detector_SWC(dispersion=[199.0]), "detector.dispersion must be two or more"),
    (lambda: Detector_SWC(dispersion=["a", 1.0]), "detector.dispersion must be two or more"),
    (lambda: Detector_SWC(slit_image_tilt=(1.0, 2.0, 3.0)), "detector.slit_image_tilt must be two"),
    (lambda: Detector_SWC(lit_band=(170.0, 212.3)), "detector.lit_band must be two wavelengths"),
    (lambda: Detector_SWC(wavelength_offset=0.1), "detector.wavelength_offset needs a unit"),
    (lambda: Detector_SWC(wavelength_offset=[0, 0.1] * u.AA), "must be one wavelength"),
    (lambda: FocalPlane_SWC(lit_band=(170.0, 212.3)), "focal_plane.lit_band must be two"),
    (lambda: FocalPlane_SWC.from_detector(Detector_SWC(), slit_image_tilt=(1, 2, 3)),
     "focal_plane.slit_image_tilt must be two"),
    (lambda: AluminiumFilter(visible_decade_thickness=0 * u.AA),
     "filter.visible_decade_thickness must be more than zero"),
    (lambda: Simulation(pinhole_visible_wavelength=600), "pinhole_visible_wavelength needs"),
])
def test_a_bad_setting_is_refused(build, message):
    with pytest.raises(ValueError, match=message):
        build()


# --- the old name of the dark current at 293 K ------------------------------------------

def test_the_old_dark_current_name_still_works_with_a_warning():
    with pytest.warns(FutureWarning, match="_dark_current_293k is renamed dark_current_293k"):
        det = Detector_SWC(_dark_current_293k=100 * DARK)
    assert det.dark_current_293k == 100 * DARK
    assert det.dark_current == Detector_SWC(dark_current_293k=100 * DARK).dark_current
    with pytest.warns(FutureWarning):
        assert det._dark_current_293k == 100 * DARK
    with pytest.warns(FutureWarning):
        det._dark_current_293k = 50 * DARK
    assert det.dark_current_293k == 50 * DARK
    with pytest.warns(FutureWarning):
        assert Detector_EIS._dark_current_293k == 250 * DARK
    with pytest.raises(TypeError, match="Give dark_current_293k only"):
        Detector_EIS(_dark_current_293k=1 * DARK, dark_current_293k=2 * DARK)


@pytest.mark.parametrize("cls", [Detector_SWC, Detector_EIS])
def test_replace_with_the_old_dark_current_name_uses_its_value(cls):
    """`dataclasses.replace` passes both names; the one asked for is the old one."""
    det = cls(dark_current_293k=7 * DARK)
    replacers = [dataclasses.replace] + ([copy.replace] if hasattr(copy, "replace") else [])
    for replace in replacers:
        with pytest.warns(FutureWarning, match="_dark_current_293k is renamed") as seen:
            changed = replace(det, _dark_current_293k=9 * DARK)
        assert seen[0].filename == __file__
        assert changed.dark_current_293k == 9 * DARK
        assert changed.dark_current == cls(dark_current_293k=9 * DARK).dark_current
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert replace(det, ccd_temperature=-50 * u.deg_C).dark_current_293k == 7 * DARK


def test_a_pickled_detector_with_the_old_name_reads_back_with_it():
    """As pickled by an older version, before the new settings existed."""
    state = dict(Detector_SWC(dark_current_293k=7 * DARK).__dict__)
    state["_dark_current_293k"] = state.pop("dark_current_293k")
    for name in ("fano_factor", "pair_energy_300k", "pair_energy_slope", "dispersion",
                 "wavelength_offset", "lit_band", "slit_image_tilt"):
        del state[name]

    class Old:
        def __reduce__(self):
            return copyreg._reconstructor, (Detector_SWC, object, None), state

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        det = pickle.loads(pickle.dumps(Old()))
    assert det.dark_current_293k == 7 * DARK and "_dark_current_293k" not in det.__dict__
    assert det.si_fano == 0.115 and det.pair_energy == Detector_SWC().pair_energy
    assert FocalPlane_SWC.from_detector(det) == FocalPlane_SWC()
    np.testing.assert_array_equal(electrons_per_photon(195 * u.AA, det),
                                  electrons_per_photon(195 * u.AA, Detector_SWC()))


def test_a_results_file_with_the_old_name_and_none_of_the_new_settings_reads_back(tmp_path):
    """Without a warning, as the new settings' defaults are the values runs had before."""
    encoded = results_file._jsonable(Detector_SWC(dark_current_293k=7 * DARK))
    fields = encoded["fields"]
    fields["_dark_current_293k"] = fields.pop("dark_current_293k")
    for name in ("fano_factor", "pair_energy_300k", "pair_energy_slope", "dispersion",
                 "wavelength_offset", "lit_band", "slit_image_tilt"):
        del fields[name]
    path = save_results(tmp_path / "hand.h5", {"instrument": "SWC"})
    with h5py.File(path, "r+") as f:
        f.attrs["detector"] = json.dumps(encoded)
        f.attrs["eclipse_order"] = json.dumps(["instrument", "detector"])
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        det = load_results(path)["detector"]
    assert not seen, [str(w.message) for w in seen]
    assert det.dark_current_293k == 7 * DARK
    assert FocalPlane_SWC.from_detector(det) == FocalPlane_SWC()
    assert det.si_fano == 0.115 and det.pair_energy == Detector_SWC().pair_energy


def test_a_results_file_without_the_other_new_settings_reads_back_without_a_warning(tmp_path):
    objects = {"eis": (Detector_EIS(), ("fano_factor", "pair_energy_300k", "pair_energy_slope")),
               "filter": (AluminiumFilter(), ("visible_decade_thickness",)),
               "simulation": (Simulation(), ("pinhole_visible_wavelength",))}
    path = save_results(tmp_path / "hand.h5", {"instrument": "SWC"})
    with h5py.File(path, "r+") as f:
        for key, (obj, names) in objects.items():
            encoded = results_file._jsonable(obj)
            for name in names:
                del encoded["fields"][name]
            f.attrs[key] = json.dumps(encoded)
        f.attrs["eclipse_order"] = json.dumps(["instrument", *objects])
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        restored = load_results(path)
    assert not seen, [str(w.message) for w in seen]
    for key, (obj, _) in objects.items():
        assert repr(restored[key]) == repr(obj)


def test_the_new_settings_round_trip_through_a_results_file(tmp_path):
    det = Detector_SWC(fano_factor=0.12, pair_energy_300k=3.65 * u.eV,
                       dispersion=(199.0, 1.25, -1e-4), wavelength_offset=0.2 * u.AA,
                       lit_band=(171 * u.AA, 211 * u.AA), slit_image_tilt=MEASURED_SLIT_IMAGE_TILT)
    payload = {"instrument": "SWC", "detector": det, "eis": Detector_EIS(fano_factor=0.2),
               "filter": AluminiumFilter(visible_decade_thickness=160 * u.AA),
               "simulation": Simulation(pinhole_visible_wavelength=500 * u.nm)}
    restored = load_results(save_results(tmp_path / "out.h5", payload))
    for name in ("detector", "eis", "filter", "simulation"):
        assert repr(restored[name]) == repr(payload[name])


# --- the configuration file ---------------------------------------------------------------

def test_the_new_settings_are_read_from_yaml():
    config = load_yaml_config(yaml.safe_dump({**UNIFORM,
        "simulation": {"pinhole_visible_wavelength": "550 nm"},
        "filter": {"visible_decade_thickness": "160 angstrom"},
        "detector": {"dark_current_293k": "100 electron / (pix s)", "fano_factor": 0.12,
                     "pair_energy_300k": "3.65 eV", "pair_energy_slope": "-0.0005 eV / K",
                     "dispersion": [198.885944, 1.2516577, -1.361114e-4],
                     "wavelength_offset": "0.1 Angstrom", "lit_band": ["171 AA", "211 AA"],
                     "slit_image_tilt": [0.991, 0.605]}}))
    _validate_config_keys(config, "SWC")
    fixed, sweep = _parse_section(config["detector"], "detector")
    assert not sweep
    det = Detector_SWC(**fixed)
    assert det.dispersion == DISPERSION_COEFFICIENTS and det.slit_image_tilt == (0.991, 0.605)
    assert det.lit_band == (171 * u.AA, 211 * u.AA) and det.wavelength_offset == 0.1 * u.AA
    assert det.pair_energy_slope == -0.0005 * u.eV / u.K and det.si_fano == 0.12
    fixed, _ = _parse_section(config["filter"], "filter")
    assert AluminiumFilter(**fixed).visible_decade_thickness == 160 * u.AA
    fixed, _ = _parse_section(config["simulation"], "simulation")
    assert Simulation(**fixed).pinhole_visible_wavelength == 550 * u.nm


def test_the_old_dark_current_key_is_accepted_but_not_advertised():
    _validate_config_keys(dict(UNIFORM, detector={"_dark_current_293k": "1 electron / (pix s)"}),
                          "EIS")
    with pytest.raises(ValueError) as excinfo:
        _validate_config_keys(dict(UNIFORM, detector={"nope": 1}), "SWC")
    assert "_dark_current_293k" not in str(excinfo.value)
    assert "dark_current_293k" in str(excinfo.value)


def test_a_run_records_the_settings_it_is_given(tmp_path, monkeypatch):
    payload = _run(tmp_path, monkeypatch, {**UNIFORM,
        "simulation": {"expos": "1 s", "noise": False},
        "detector": {"fano_factor": [0.115, 0.2], "wavelength_offset": "0 Angstrom",
                     "lit_band": ["17 nm", "212.3 AA"]}})
    combos = list(payload["results"]["all_combinations"].values())
    assert sorted(c["parameters"]["detector.fano_factor"] for c in combos) == [0.115, 0.2]
    det = combos[0]["config_objects"]["detector"]
    assert det.lit_band == (17 * u.nm, 212.3 * u.AA)


@pytest.mark.parametrize("detector", [
    {"wavelength_offset": "0.1 Angstrom"}, {"wavelength_offset": ["0 AA", "0.1 AA"]},
    {"lit_band": ["171 AA", "211 AA"]}, {"slit_image_tilt": [0.991, 0.605]},
    {"dispersion": [199.0, 1.25]}])
def test_a_run_refuses_focal_plane_settings_it_does_not_use(tmp_path, monkeypatch, detector):
    name = next(iter(detector))
    with pytest.raises(ValueError, match=f"detector.{name} must be left at its default for an "
                                         f"instrument run, which does not use the SW focal"):
        _run(tmp_path, monkeypatch, {**UNIFORM, "detector": detector})


def test_a_run_records_the_old_dark_current_key_as_the_config_gives_it(tmp_path, monkeypatch):
    with pytest.warns(FutureWarning, match="detector._dark_current_293k is renamed") as seen:
        payload = _run(tmp_path, monkeypatch, {**UNIFORM, "simulation": {"noise": False},
                       "detector": {"_dark_current_293k": ["100 electron / (pix s)",
                                                           "200 electron / (pix s)"]}})
    assert sum("_dark_current_293k is renamed" in str(w.message) for w in seen) == 1
    results = payload["results"]
    assert "detector._dark_current_293k" in results["sweep_dimensions"]
    assert "detector.dark_current_293k" not in results["sweep_dimensions"]
    values = sorted(c["parameters"]["detector.dark_current_293k"].value
                    for c in results["all_combinations"].values())
    assert values == [100, 200]
    with pytest.warns(FutureWarning):
        payload = _run(tmp_path, monkeypatch, {**UNIFORM, "simulation": {"noise": False},
                       "detector": {"_dark_current_293k": "100 electron / (pix s)"}})
    assert payload["results"]["fixed_params"]["detector._dark_current_293k"] == 100 * DARK
    with pytest.raises(ValueError, match="Give detector.dark_current_293k only"):
        _run(tmp_path, monkeypatch, {**UNIFORM, "detector": {
            "_dark_current_293k": "1 electron / (pix s)",
            "dark_current_293k": ["1 electron / (pix s)", "2 electron / (pix s)"]}})


def test_a_sweep_of_the_visible_decade_thickness_lays_the_scene_once(tmp_path, monkeypatch,
                                                                     capsys):
    """It changes only the visible light, which the Monte Carlo adds, not the EUV cube."""
    from euvst_response.synthesis_file import SpectralLine, Synthesis, write_synthesis

    edges = np.arange(3) * 0.1 * u.Mm
    wavelength = 195.119 * u.AA + np.arange(-30, 31) * 0.003 * u.AA
    line = SpectralLine(intensity=np.ones((2, 2, 61)) * 1e13 * u.erg / (u.s * u.cm**2 * u.sr * u.cm),
                        wavelength=wavelength, rest_wavelength=195.119 * u.AA)
    write_synthesis(Synthesis(lines={"Fe12_195.1190": line}, x_edges=edges, y_edges=edges),
                    tmp_path / "file.h5")
    payload = _run(tmp_path, monkeypatch, {
        "instrument": "SWC", "n_iter": 1, "synthesis_file": str(tmp_path / "file.h5"),
        "simulation": {"noise": False, "vis_sl": "1e12 photon / (s cm2)"},
        "filter": {"visible_decade_thickness": ["170 angstrom", "160 angstrom"]}})
    assert capsys.readouterr().out.count("Laying the scene onto the detector") == 1
    assert len(payload["results"]["all_combinations"]) == 2


def test_a_run_refuses_a_sweep_of_the_pinhole_wavelength_with_the_pinholes_off(tmp_path,
                                                                                monkeypatch):
    sweep = {"pinhole_visible_wavelength": ["500 nm", "600 nm"]}
    with pytest.raises(ValueError, match="pinhole_visible_wavelength sets the light through"):
        _run(tmp_path, monkeypatch, {**UNIFORM, "simulation": sweep})
    with pytest.raises(ValueError, match="pinhole_visible_wavelength sets the light through"):
        _run(tmp_path, monkeypatch, {**UNIFORM, "simulation": {**sweep, "enable_pinholes": True}})


def test_an_eis_run_refuses_the_pinhole_wavelength(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Pinhole effects are not supported for EIS"):
        _run(tmp_path, monkeypatch, {**UNIFORM, "instrument": "EIS", "simulation": {
            "slit_width": "1 arcsec", "pinhole_visible_wavelength": "500 nm"}})
