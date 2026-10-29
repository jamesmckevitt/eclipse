"""The science case table and the configurations written from it."""
import sys

import astropy.units as u
import pytest
import yaml

from eclipse import science_cases as sc
from eclipse.analysis import load_instrument_response_results
from eclipse.config import Simulation, check_lw_wavelength
from eclipse.utils import parse_yaml_input

TABLE = sc.load_science_cases()
CASE_KEYS = {"task", "name", "description", "target", "observables", "source",
             "filling_factor", "slit_width", "exposure", "raster_steps", "bin_x",
             "bin_y", "slit_length", "duration", "sji_cadence", "sji_waves",
             "lines"}


def _by_name(configs):
    return {item.name: item for item in configs}


# --- the table -----------------------------------------------------------

def test_the_table_has_every_case_and_line():
    assert len(TABLE) == 13
    assert sum(len(case["lines"]) for case in TABLE) == 121
    assert len({case["task"] for case in TABLE}) == 13
    assert len({case["name"] for case in TABLE}) == 13
    for case in TABLE:
        assert set(case) == CASE_KEYS, case["task"]


@pytest.mark.parametrize("case", TABLE, ids=lambda case: case["task"])
def test_every_case_and_line_reads(case):
    assert float(case["filling_factor"]) > 0
    assert u.Quantity(case["exposure"]).to_value(u.s) > 0
    # Simulation refuses a slit width the instrument does not have.
    Simulation(instrument="SWC", slit_width=u.Quantity(case["slit_width"]))

    labels = [sc.line_label(line) for line in case["lines"]]
    assert len(labels) == len(set(labels))
    for line in case["lines"]:
        symbol, numeral = line["ion"].split()
        assert numeral in sc._ROMAN
        assert u.Quantity(line["wavelength"]).unit == u.AA
        assert 3.5 < float(line["log_t_max"]) < 8.0
        intensity = u.Quantity(line["intensity"])
        assert intensity.unit.is_equivalent(u.erg / (u.s * u.cm**2 * u.sr))
        assert intensity.value > 0
        assert sc.thermal_width(line).to_value(u.km / u.s) > 0


def test_the_short_wavelength_range_comes_from_the_efficiency_tables():
    low, high = sc.short_wavelength_range()
    assert low.to_value(u.AA) == pytest.approx(170.0)
    assert high.to_value(u.AA) == pytest.approx(214.0)


# --- the configurations ----------------------------------------------------

def test_every_line_eclipse_can_simulate_gets_a_valid_config():
    configs, skipped = sc.science_case_configs()
    assert len(configs) == 119
    assert sum(item.config["instrument"] == "LW" for item in configs) == 85
    assert len({item.name for item in configs}) == 119
    # Fe XXII 845.57 lands on LW1 beyond the sensitivity budget's tables.
    assert sorted(skipped) == ["II-1-1 Fe XXII 845.57", "II-1-2 Fe XXII 845.57"]

    low, high = sc.short_wavelength_range()
    for item in configs:
        config = item.config
        rest = parse_yaml_input(config["rest_wavelength"])
        if config["instrument"] == "SWC":
            assert low <= rest <= high
            assert "detector" not in config
        else:
            check_lw_wavelength(rest, config["detector"]["band"], config["detector"]["order"])
        assert item.name.endswith(sc.line_id(item.line))
        Simulation(instrument=config["instrument"],
                   slit_width=parse_yaml_input(config["simulation"]["slit_width"]),
                   expos=parse_yaml_input(config["simulation"]["expos"]))


@pytest.mark.parametrize("name, intensity, width, expos, slit", [
    ("1.1.1-nanoflares_events_fe12_195119", "3756 erg / (s cm2 sr)", "15.36 km/s", "5 s", "0.4 arcsec"),
    ("2.1.2-flare_ribbons_fe24_192030", "864923 erg / (s cm2 sr)", "51.45 km/s", "0.5 s", "0.4 arcsec"),
    ("1.4.1-sw_source_regions_fe10_174531", "1064 erg / (s cm2 sr)", "12.92 km/s", "6 s", "0.8 arcsec"),
])
def test_configs_match_the_nasa_pdr_analysis(name, intensity, width, expos, slit):
    """The intensity includes the filling factor: 939 x 4, 864923 x 1, 5320 x 0.2."""
    config = _by_name(sc.science_case_configs()[0])[name].config
    assert config["uniform_intensity"] == intensity
    assert config["thermal_width"] == width
    assert config["simulation"] == {"psf": True, "expos": expos, "slit_width": slit}
    assert (config["n_iter"], config["ncpu"], config["offchip_bin_slit"]) == (512, -1, [1, 2])


@pytest.mark.parametrize("task, label, intensity", [
    ("II-1-2", "C III 977.02", "1647325 erg / (s cm2 sr)"),
    ("I-4-1", "C III 1176.0", "164.6 erg / (s cm2 sr)"),
])
def test_the_intensity_keeps_every_digit_of_the_table(task, label, intensity):
    (case,) = [case for case in TABLE if case["task"] == task]
    (line,) = [line for line in case["lines"] if sc.line_label(line) == label]
    assert sc._config(case, line, "SWC", {})["uniform_intensity"] == intensity


def test_a_written_config_reads_back_with_its_header(tmp_path):
    item = _by_name(sc.science_case_configs()[0])["1.1.1-nanoflares_events_fe12_195119"]
    text = item.to_yaml()
    assert text.startswith("# Science case I-1-1 (1.1.1-nanoflares_events)")
    assert "# Line: Fe XII 195.119, log T_max = 6.20, intensity 939 erg / (s cm2 sr) times filling factor 4" in text
    assert yaml.safe_load(text) == item.config


# --- choosing cases and lines ------------------------------------------------

def test_cases_are_chosen_by_task_or_name():
    by_task, skipped = sc.science_case_configs(cases=["I-1-1"])
    by_name, _ = sc.science_case_configs(cases=["1.1.1-nanoflares_events"])
    assert [item.name for item in by_task] == [item.name for item in by_name]
    assert sorted((sc.line_label(item.line), item.config["instrument"],
                   item.config.get("detector")) for item in by_task) == [
        ("Ca XIV 193.874", "SWC", None), ("Fe IX 171.073", "SWC", None),
        ("Fe XII 195.119", "SWC", None), ("Fe XIV 211.317", "SWC", None),
        ("Fe XIX 1118.076", "LW", {"band": "LW3", "order": 1}),
        ("Fe XVIII 974.860", "LW", {"band": "LW2", "order": 1}),
        ("Ne VIII 770.428", "LW", {"band": "LW1", "order": 1}),
        ("Si XII 499.41", "LW", {"band": "LW2", "order": 2})]
    assert skipped == []


@pytest.mark.parametrize("selector", ["Fe XII", "Fe XII 195.119", "Fe XII 195.12"])
def test_a_line_is_chosen_by_ion_or_wavelength_across_cases(selector):
    """195.119 and 195.12 are the same line, written to different precision."""
    configs, _ = sc.science_case_configs(lines=[selector])
    assert sorted(item.case["task"] for item in configs) == [
        "I-1-1", "I-1-2", "I-1-3", "I-2-1", "I-4-1", "II-2-1"]


def test_a_line_eclipse_cannot_simulate_yet_is_skipped_not_refused():
    configs, skipped = sc.science_case_configs(lines=["Fe XXII 845.57"])
    assert configs == []
    assert sorted(skipped) == ["II-1-1 Fe XXII 845.57", "II-1-2 Fe XXII 845.57"]


def test_the_line_sets_the_lw_band_and_the_base_cannot():
    configs, _ = sc.science_case_configs(lines=["O VI 1031.91"],
                                         base={"detector": {"read_noise_rms": "30 electron / pix"}})
    assert sorted(item.case["task"] for item in configs) == [
        "I-1-2", "I-2-1", "I-2-2", "I-3-1", "I-4-1", "I-4-2", "II-1-2", "II-1-3", "II-2-1",
        "II-2-2"]
    for item in configs:
        assert item.config["detector"] == {"read_noise_rms": "30 electron / pix",
                                           "band": "LW2", "order": 1}
    for section in ("detector", "telescope"):
        with pytest.raises(ValueError, match=f"cannot set {section}.band"):
            sc.science_case_configs(lines=["O VI 1031.91"], base={section: {"band": "LW3"}})


@pytest.mark.parametrize("kwargs, message", [
    ({"cases": ["I-9-9"]}, "No science case is called"),
    ({"lines": ["Fe XII 195.5"]}, "No line in the chosen cases matches"),
    ({"cases": ["I-3-1"], "lines": ["Fe XII"]}, "No line in the chosen cases matches"),
    ({"lines": ["Fe"]}, "Choose a line by ion"),
])
def test_a_choice_that_matches_nothing_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        sc.science_case_configs(**kwargs)


# --- base settings ------------------------------------------------------------

def test_base_settings_go_into_every_config():
    base = {"n_iter": 100, "fit_signals": "dn", "simulation": {"expos": ["1 s", "5 s"]}}
    configs, _ = sc.science_case_configs(cases=["I-4-1"], base=base)
    assert len(configs) == 13
    for item in configs:
        config = item.config
        assert (config["n_iter"], config["fit_signals"]) == (100, "dn")
        assert config["simulation"] == {"psf": True, "expos": ["1 s", "5 s"],
                                        "slit_width": "0.8 arcsec"}


@pytest.mark.parametrize("base, message", [
    ({"rest_wavelength": "195.119 AA"}, "cannot set 'rest_wavelength'"),
    ({"instrument": "EIS"}, "cannot set 'instrument'"),
    ({"exposure": "5 s"}, "exposure"),
    # An empty heading parses to None.
    ({"simulation": None}, "'simulation:' section must be a mapping"),
    ({"simulation": ["5 s"]}, "'simulation:' section must be a mapping"),
    ({"telescope": None}, "'telescope:' section must be a mapping"),
])
def test_base_settings_cannot_replace_the_line_or_be_what_eclipse_does_not_read(base, message):
    with pytest.raises(ValueError, match=message):
        sc.science_case_configs(cases=["I-3-1"], base=base)


# --- the command ----------------------------------------------------------------

def test_the_command_writes_the_chosen_configs(tmp_path, capsys):
    base = tmp_path / "base.yaml"
    base.write_text("n_iter: 50\n")
    sc.main(["--out", str(tmp_path / "out"), "--case", "II-1-1",
             "--base", str(base)])

    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert written == ["2.1.1-flare_reconnection_fe19_1118060.yaml",
                       "2.1.1-flare_reconnection_fe20_721560.yaml",
                       "2.1.1-flare_reconnection_fe24_192030.yaml"]
    for name in written:
        assert yaml.safe_load((tmp_path / "out" / name).read_text())["n_iter"] == 50
    out = capsys.readouterr().out
    assert "Wrote 3 configurations to" in out
    assert "Skipped 1 line ECLIPSE cannot simulate yet" in out


def test_the_command_lists_cases_and_what_can_be_simulated(capsys):
    sc.main(["--list", "--case", "I-1-1"])
    out = capsys.readouterr().out
    assert out.startswith("I-1-1  1.1.1-nanoflares_events: Observe small scale heating events")
    assert "Fe XII 195.119         SWC" in out
    assert "Ne VIII 770.428        LW" in out


def test_the_list_refuses_a_line_choice_that_matches_nothing(capsys):
    with pytest.raises(ValueError, match="No line in the chosen cases matches"):
        sc.main(["--list", "--case", "I-3-1", "--line", "Fe XII"])
    assert capsys.readouterr().out == ""


def test_a_science_case_runs_from_start_to_finish(tmp_path, monkeypatch):
    """Uses the brightest line, with few iterations to keep it quick."""
    from eclipse.main import main

    paths, _ = sc.write_science_case_configs(
        tmp_path, cases=["II-1-2"], lines=["Fe XXIV"],
        base={"n_iter": 2, "ncpu": 1})
    (config,) = paths

    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    monkeypatch.chdir(tmp_path)
    main()

    results = load_instrument_response_results(
        tmp_path / "run" / "result" / f"{config.stem}.h5")
    combinations = results["results"]["all_combinations"].values()
    assert sorted(c["parameters"]["offchip_bin_slit"] for c in combinations) == [1, 2]


def test_an_lw_science_case_runs_from_start_to_finish(tmp_path, monkeypatch):
    """An LW line in second order, with the iterations cut down so that it is quick."""
    from eclipse.main import main

    paths, _ = sc.write_science_case_configs(
        tmp_path, cases=["I-1-1"], lines=["Si XII"], base={"n_iter": 2, "ncpu": 1})
    (config,) = paths

    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    monkeypatch.chdir(tmp_path)
    main()

    results = load_instrument_response_results(
        tmp_path / "run" / "result" / f"{config.stem}.h5")
    combinations = list(results["results"]["all_combinations"].values())
    assert sorted(c["parameters"]["offchip_bin_slit"] for c in combinations) == [1, 2]
    detector = combinations[0]["config_objects"]["detector"]
    assert (type(detector).__name__, detector.band, detector.order) == ("Detector_LW", "LW2", 2)
    assert combinations[0]["config_objects"]["telescope"].order == 2


@pytest.mark.parametrize("base", [{"detector": {"ccd_temperature": "-50 Celsius"}},
                                  {"detector": {"read_noise_rms": "5 electron / pix"}},
                                  {"simulation": {"vis_sl": "5 photon / (s cm2)"}},
                                  {"simulation": {"slit_mask": True}},
                                  {"filter": {"al_thickness": "1500 angstrom"}}])
def test_a_base_for_the_sw_hardware_skips_the_lw_lines_of_both_channels(base):
    """Settings both detectors have, such as the read noise, are SW's too, as the base was written for both."""
    with pytest.warns(UserWarning, match="applied to the SW lines only"):
        configs, skipped = sc.science_case_configs(cases=["I-1-1"], base=base)
    assert {item.config["instrument"] for item in configs} == {"SWC"}
    assert "I-1-1 Si XII 499.41" in skipped


def test_the_lw_lines_alone_take_their_own_base_and_its_mistakes_are_errors():
    base = {"detector": {"read_noise_rms": "30 electron / pix"}}
    configs, _ = sc.science_case_configs(lines=["Si XII 499.41"], base=base)
    assert configs and all(item.config["detector"]["read_noise_rms"] == "30 electron / pix"
                           for item in configs)
    with pytest.raises(ValueError, match="read_nosie_rms"):
        sc.science_case_configs(lines=["Si XII 499.41"],
                                base={"detector": {"read_nosie_rms": "30 electron / pix"}})
    with pytest.raises(ValueError, match="vis_sl must be 0"):
        sc.science_case_configs(lines=["Si XII 499.41"],
                                base={"simulation": {"vis_sl": "5 photon / (s cm2)"}})


def test_a_mistake_in_a_base_for_both_channels_is_still_an_error():
    with pytest.raises(ValueError, match="read_nosie_rms"):
        sc.science_case_configs(cases=["I-1-1"],
                                base={"detector": {"read_nosie_rms": "5 electron / pix"}})
