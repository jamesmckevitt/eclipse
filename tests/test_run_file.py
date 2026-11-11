"""Run files: one YAML file for the synthesis, the observation and the frame."""
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pytest
import yaml

import eclipse
from eclipse import run_file
from eclipse.run_file import (adjust, apply_setting, check, load, observation_config, runs,
                              synthesis_arguments)

LINE = "Fe12_195.1190"
REST = 195.119 * u.AA


def _flat_goft(lines, **kwargs):
    """Contribution function of 1e-24 everywhere, in place of fiasco."""
    logT, logN = np.linspace(5.0, 7.0, 21), np.linspace(8.0, 10.0, 21)
    return ({name: {"wl0": REST.to(u.cm), "g_tn": np.full((21, 21), 1e-24), "atom": 26,
                    "ion": 12, "hdf5_dbase_root": None} for name in lines}, logT, logN)


@pytest.fixture
def atmosphere(tmp_path):
    from eclipse.atmosphere import Atmosphere, write_atmosphere
    from eclipse.utils import angle_to_distance

    cell = angle_to_distance(0.2 * u.arcsec).to(u.Mm)
    nz, ny, nx = 4, 6, 8
    return write_atmosphere(Atmosphere(
        temperature=np.full((nz, ny, nx), 1e6) * u.K,
        electron_density=np.full((nz, ny, nx), 1e9) / u.cm**3,
        velocity_z=np.zeros((nz, ny, nx)) * u.km / u.s,
        x_edges=(np.arange(nx + 1) - nx / 2) * cell, y_edges=(np.arange(ny + 1) - ny / 2) * cell,
        z_edges=np.arange(nz + 1) * 0.1 * u.Mm), tmp_path / "box.h5")


def _job(atmosphere, **settings):
    """A run that synthesises `atmosphere` and observes it briefly."""
    return {"instrument": "SWC", "atmosphere": str(atmosphere), "n_iter": 2, "ncpu": 1,
            "fit_signals": "dn", "synthesis": {"lines": [LINE]},
            "simulation": {"expos": "5 s", "psf": False}, **settings}


@pytest.fixture
def fake_synthesis(monkeypatch):
    """Synthesis that writes an empty HDF5 file and counts how often it ran."""
    from eclipse import synthesis

    made = []

    def write(args):
        path = Path(args.output_dir) / args.output_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w"):
            pass
        made.append(args)

    monkeypatch.setattr(synthesis, "main", write)
    return made


def test_a_setting_is_changed_by_its_dotted_key():
    config = {"simulation": {"expos": "1 s"}}
    changed = apply_setting(config, "simulation.expos=[5 s, 20 s]")
    assert changed["simulation"]["expos"] == ["5 s", "20 s"]
    assert config["simulation"]["expos"] == "1 s"
    assert apply_setting(config, "detector.qe_euv=0.6")["detector"] == {"qe_euv": 0.6}
    with pytest.raises(ValueError, match="key=value"):
        apply_setting(config, "n_iter")
    with pytest.raises(ValueError, match="not a section"):
        apply_setting(config, "simulation.expos.value=3")


def test_sections_and_settings_are_laid_over_the_file(tmp_path):
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump({"n_iter": 5, "simulation": {"expos": "1 s", "psf": True}}))
    config, name, source = load(path)
    assert (name, source) == ("job", str(path))
    config = adjust(config, ["simulation.psf=false"],
                    {"n_iter": 3, "simulation": {"expos": "5 s"}})
    assert config == {"n_iter": 3, "simulation": {"expos": "5 s", "psf": False}}


def test_each_run_is_the_file_with_its_own_settings():
    config = {"n_iter": 5, "simulation": {"expos": "1 s", "psf": True},
              "runs": [{"name": "short"}, {"name": "long", "simulation": {"expos": "60 s"}}]}
    found = runs(config, "job")
    assert [name for name, _ in found] == ["job_short", "job_long"]
    assert found[1][1] == {"n_iter": 5, "simulation": {"expos": "60 s", "psf": True}}
    assert runs({"n_iter": 5}, "job") == [("job", {"n_iter": 5})]
    with pytest.raises(ValueError, match="Two runs are named"):
        runs({"runs": [{"name": "a"}, {"name": "a"}]}, "job")
    with pytest.raises(ValueError, match="needs a 'name'"):
        runs({"runs": [{"n_iter": 3}]}, "job")


def test_a_synthesis_section_gives_the_synthesis_options(atmosphere):
    from eclipse.synthesis import parse_arguments

    config = _job(atmosphere, synthesis={
        "lines": [LINE, "Fe09_171.0730"], "continuum": True, "crop_z": ["0 Mm", "0.2 Mm"],
        "integration_axis": "-x", "vel_res": "2 km/s", "mass_per_electron": 1.2})
    args = parse_arguments(synthesis_arguments(config, "job"))
    assert args.atmosphere == str(atmosphere)
    assert args.lines == [LINE, "Fe09_171.0730"]
    assert args.continuum is True
    assert args.crop_z == ["0 Mm", "0.2 Mm"]
    assert args.integration_axis == "-x"
    assert (args.vel_res, args.mass_per_electron) == ("2 km/s", 1.2)
    assert Path(args.output_dir) / args.output_name == Path("run/input/job.h5")


@pytest.mark.parametrize("synthesis, message", [
    ({"lines": [LINE], "vel_resolution": "2 km/s"}, "vel_res"),
    ({"abundance": "sun_coronal_2021_chianti"}, "lines"),
    ({"lines": [LINE], "crop_z": "0 Mm"}, "takes 2 values"),
    ({"lines": [LINE], "continuum": "yes"}, "true or false"),
])
def test_a_mistake_in_the_synthesis_section_is_reported(atmosphere, synthesis, message):
    with pytest.raises(ValueError, match=message):
        synthesis_arguments(_job(atmosphere, synthesis=synthesis), "job")


def test_synthesise_spectra_reads_a_run_file_and_its_options_override_it(tmp_path, atmosphere):
    from eclipse.synthesis import parse_arguments

    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(_job(atmosphere, synthesis_file="out/syn.h5")))
    args = parse_arguments(["--config", str(path), "--vel-res", "2 km/s"])
    assert args.lines == [LINE]
    assert args.vel_res == "2 km/s"
    assert (args.output_dir, args.output_name) == ("out", "syn.h5")


def test_the_synthesis_help_is_grouped_and_leaves_out_the_old_options():
    from eclipse.synthesis import build_parser

    text = build_parser().format_help()
    for heading in ("Input and output", "Lines and atomic data",
                    "The view and the part of the box", "Grids and precision", "Computing"):
        assert heading in text
    assert "--slit-rest-time" not in text and "--temp-dir" not in text


def test_the_observation_takes_the_synthesis_and_its_first_line(atmosphere):
    observed = observation_config(_job(atmosphere, synthesis_file="syn.h5"), "job")
    assert "atmosphere" not in observed and "synthesis" not in observed
    assert (observed["synthesis_file"], observed["reference_line"]) == ("syn.h5", LINE)
    with pytest.raises(ValueError, match="cannot both be given"):
        observation_config(_job(atmosphere, uniform_intensity="100 erg/(s cm2 sr)"), "job")


def test_a_mistake_in_the_observation_is_found_before_the_synthesis(
        tmp_path, monkeypatch, atmosphere, fake_synthesis):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="expos"):
        eclipse.run(_job(atmosphere, simulation={"exposure": "5 s"}))
    with pytest.raises(ValueError, match="slit_positon"):
        eclipse.run(_job(atmosphere, frame={"slit_positon": "0 arcsec"}))
    assert not fake_synthesis


def test_a_synthesis_is_reused_until_its_settings_change(tmp_path, monkeypatch, atmosphere,
                                                         fake_synthesis):
    monkeypatch.chdir(tmp_path)
    [first] = eclipse.synthesise(_job(atmosphere), name="job")
    assert first.synthesis == Path("run/input/job.h5")
    eclipse.synthesise(_job(atmosphere), name="job")
    assert len(fake_synthesis) == 1
    eclipse.synthesise(_job(atmosphere), name="job", settings=["synthesis.vel_res='2 km/s'"])
    assert len(fake_synthesis) == 2
    # A file the run did not make is left alone.
    with h5py.File(tmp_path / "theirs.h5", "w"):
        pass
    with pytest.raises(ValueError, match="not made by a run file"):
        eclipse.synthesise(_job(atmosphere, synthesis_file="theirs.h5"), name="job")


def test_observing_before_synthesising_says_to_synthesise(tmp_path, monkeypatch, atmosphere):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="eclipse synthesise"):
        eclipse.observe(_job(atmosphere), name="job")


def test_a_run_synthesises_its_atmosphere_then_observes_it(tmp_path, monkeypatch, atmosphere):
    from eclipse import synthesis
    from eclipse.analysis import load_instrument_response_results

    monkeypatch.setattr(synthesis, "compute_goft_fiasco", _flat_goft)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(_job(atmosphere)))
    [output] = eclipse.run(path)
    assert output.synthesis == Path("run/input/job.h5")
    assert output.results == Path("run/result/job.h5")
    results = load_instrument_response_results(output.results)
    assert results["config"]["synthesis_file"] == "run/input/job.h5"
    assert results["config"]["reference_line"] == LINE


def test_runs_are_picked_by_name(tmp_path, monkeypatch, atmosphere, fake_synthesis):
    monkeypatch.chdir(tmp_path)
    job = _job(atmosphere, runs=[{"name": "fe12"},
                                 {"name": "fe09", "synthesis": {"lines": ["Fe09_171.0730"]}}])
    [output] = eclipse.synthesise(job, name="job", only="fe09")
    assert output.synthesis == Path("run/input/job_fe09.h5")
    assert fake_synthesis[0].lines == ["Fe09_171.0730"]
    with pytest.raises(ValueError, match="the runs are fe12, fe09"):
        eclipse.synthesise(job, name="job", only="fe10")


def test_a_frame_takes_the_run_files_settings(tmp_path, monkeypatch):
    from eclipse import full_frame

    monkeypatch.chdir(tmp_path)
    with h5py.File(tmp_path / "syn.h5", "w"):
        pass
    taken = {}
    monkeypatch.setattr(run_file, "_band_lines", lambda *args: "lines")
    monkeypatch.setattr(full_frame, "full_ccd_frame",
                        lambda *args, **kwargs: taken.update(args=args, kwargs=kwargs))
    monkeypatch.setattr(full_frame, "write_full_frame", lambda frame, path: path)
    job = {"synthesis_file": "syn.h5",
           "simulation": {"expos": "20 s", "slit_width": "0.4 arcsec", "slit_mask": True},
           "detector": {"qe_euv": 0.6}, "frame": {"slit_position": "1 arcsec"}}
    [output] = eclipse.make_frame(job, name="job")
    assert output.frame == Path("run/result/job_frame.h5")
    synthesis, lines, position, exposure = taken["args"]
    assert (synthesis, lines) == (Path("syn.h5"), "lines")
    assert (position, exposure) == (1 * u.arcsec, 20 * u.s)
    assert taken["kwargs"]["slit_width"] == 0.4 * u.arcsec
    assert taken["kwargs"]["slit_mask"] is True
    assert taken["kwargs"]["det"].qe_euv == 0.6
    with pytest.raises(ValueError, match="one value"):
        eclipse.make_frame(job, name="job", settings=["simulation.expos=[5 s, 20 s]"])
    with pytest.raises(ValueError, match="no 'frame:' section"):
        eclipse.make_frame({"synthesis_file": "syn.h5"}, name="job")


def test_the_eclipse_command_runs_the_steps_it_names(monkeypatch):
    from eclipse import cli

    called = []
    monkeypatch.setattr(run_file, "run_all", lambda *args, **kwargs: called.append(
        (args, kwargs)))
    cli.main(["run", "job.yaml", "--set", "n_iter=3", "--only", "fe12", "--debug"])
    assert called[-1] == (("job.yaml", "run"),
                          {"settings": ["n_iter=3"], "only": "fe12", "debug": True})
    cli.main(["observe", "job.yaml"])
    assert called[-1] == (("job.yaml", "observe"),
                          {"settings": [], "only": None, "debug": False})
    with pytest.warns(FutureWarning, match="eclipse run job.yaml"):
        cli.main(["--config", "job.yaml"])
    assert called[-1][0] == ("job.yaml", "run")


def test_the_eclipse_command_passes_the_other_commands_on(monkeypatch):
    from eclipse import atmosphere, cli, science_cases

    passed = []
    monkeypatch.setattr(atmosphere, "main", lambda argv: passed.append(("atmosphere", argv)))
    monkeypatch.setattr(science_cases, "main", lambda argv: passed.append(("cases", argv)))
    cli.main(["atmosphere", "info", "box.h5"])
    cli.main(["science-cases", "--list", "--case", "I-1-1"])
    assert passed == [("atmosphere", ["info", "box.h5"]),
                      ("cases", ["--list", "--case", "I-1-1"])]


def test_the_science_cases_as_one_run_file_are_the_separate_configurations(tmp_path):
    from eclipse.science_cases import science_case_configs, write_science_case_run_file

    path, _ = write_science_case_run_file(tmp_path / "cases.yaml", base={"n_iter": 8})
    config, name, _ = load(path)
    expected = {f"cases_{item.name}": item.config
                for item in science_case_configs(base={"n_iter": 8})[0]}
    found = dict(runs(config, name))
    assert found == expected
    assert config["n_iter"] == 8 and "uniform_intensity" not in config
    for run_name, run_config in found.items():
        check(run_config, run_name)


def test_a_setting_changed_from_outside_wins_over_a_runs_own(tmp_path, monkeypatch, atmosphere,
                                                             fake_synthesis):
    monkeypatch.chdir(tmp_path)
    job = _job(atmosphere, runs=[{"name": "fe12", "synthesis": {"lines": [LINE],
                                                                "vel_res": "2 km/s"}}])
    eclipse.synthesise(job, name="job", settings=["synthesis.vel_res='1 km/s'"])
    assert fake_synthesis[0].vel_res == "1 km/s"


def test_a_synthesis_stopped_part_way_leaves_nothing_in_the_way(tmp_path, monkeypatch,
                                                                atmosphere):
    from eclipse import synthesis

    def stops(args):
        (Path(args.output_dir) / args.output_name).write_text("half a file")
        raise KeyboardInterrupt

    monkeypatch.setattr(synthesis, "main", stops)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        eclipse.synthesise(_job(atmosphere), name="job")
    assert not Path("run/input/job.h5").exists()


def test_settings_that_change_only_how_it_is_computed_keep_the_synthesis(
        tmp_path, monkeypatch, atmosphere, fake_synthesis):
    monkeypatch.chdir(tmp_path)
    eclipse.synthesise(_job(atmosphere), name="job")
    eclipse.synthesise(_job(atmosphere), name="job",
                       settings=["synthesis.n_workers=4", "synthesis.goft_temperature_chunk=5"])
    assert len(fake_synthesis) == 1


def test_a_synthesis_made_from_other_settings_is_not_observed(tmp_path, monkeypatch, atmosphere,
                                                              fake_synthesis):
    monkeypatch.chdir(tmp_path)
    eclipse.synthesise(_job(atmosphere), name="job")
    with pytest.raises(ValueError, match="not made from this atmosphere with these settings"):
        eclipse.observe(_job(atmosphere), name="job",
                        settings=["synthesis.lines=[Fe09_171.0730]"])


def test_runs_that_would_overwrite_each_others_synthesis_are_refused(
        tmp_path, monkeypatch, atmosphere, fake_synthesis):
    monkeypatch.chdir(tmp_path)
    job = _job(atmosphere, synthesis_file="run/input/shared.h5",
               runs=[{"name": "fe12"},
                     {"name": "fe09", "synthesis": {"lines": ["Fe09_171.0730"]}}])
    with pytest.raises(ValueError, match="would synthesise differently"):
        eclipse.synthesise(job, name="job")
    assert not fake_synthesis


def test_a_frame_with_its_own_exposure_leaves_the_runs_sweep_alone(tmp_path, monkeypatch,
                                                                   atmosphere, fake_synthesis):
    monkeypatch.chdir(tmp_path)
    job = _job(atmosphere, simulation={"expos": ["5 s", "20 s"]},
               frame={"slit_position": "0 arcsec", "exposure": "60 s"})
    check(job, "job")
    with pytest.raises(ValueError, match="'simulation.expos' has several"):
        check(_job(atmosphere, simulation={"expos": ["5 s", "20 s"]},
                   frame={"slit_position": "0 arcsec"}), "job")
    with pytest.raises(ValueError, match="slit_position"):
        eclipse.run(_job(atmosphere, frame={"exposure": "60 s"}), name="job")
    assert not fake_synthesis


def test_options_before_a_passed_on_command_are_refused(monkeypatch):
    from eclipse import cli, science_cases

    passed = []
    monkeypatch.setattr(science_cases, "main", lambda argv: passed.append(argv))
    with pytest.raises(SystemExit):
        cli.main(["--debug", "science-cases"])
    assert not passed


def test_synthesise_spectra_given_only_a_run_file_makes_the_runs_synthesis(
        tmp_path, monkeypatch, atmosphere, fake_synthesis):
    import sys

    from eclipse import synthesis_cli

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "job.yaml"
    path.write_text(yaml.safe_dump(_job(atmosphere)))
    monkeypatch.setattr(sys, "argv", ["synthesise-spectra", "--config", str(path)])
    synthesis_cli.main()
    synthesis_cli.main()
    assert len(fake_synthesis) == 1
    [output] = eclipse.synthesise(path)
    assert output.synthesis == Path("run/input/job.h5") and len(fake_synthesis) == 1


def test_a_synthesis_is_observed_after_its_atmosphere_is_moved(tmp_path, monkeypatch, atmosphere,
                                                               fake_synthesis):
    import importlib

    main = importlib.import_module("eclipse.main")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "run_config", lambda config, name, **kwargs: config)
    eclipse.synthesise(_job(atmosphere), name="job")
    atmosphere.rename(tmp_path / "archived.h5")
    with pytest.warns(UserWarning, match="is not there"):
        eclipse.observe(_job(atmosphere), name="job")
    with pytest.raises(ValueError, match="not made from this atmosphere"):
        eclipse.observe(_job(atmosphere), name="job", settings=["synthesis.vel_res='1 km/s'"])
    with pytest.raises(FileNotFoundError, match="Atmosphere file not found"):
        eclipse.synthesise(_job(atmosphere), name="job",
                           settings=["synthesis.vel_res='1 km/s'"])


def test_a_frame_takes_one_exposure(atmosphere):
    with pytest.raises(ValueError, match="one exposure time"):
        check(_job(atmosphere, frame={"slit_position": "0 arcsec",
                                      "exposure": ["5 s", "60 s"]}), "job")


def test_a_runs_setting_replaces_the_list(tmp_path, monkeypatch, atmosphere, fake_synthesis):
    monkeypatch.chdir(tmp_path)
    job = _job(atmosphere, runs=[{"name": "fe12"}, {"name": "fe09"}])
    outputs = eclipse.synthesise(job, name="job", settings=["runs=[{name: fe11}]"])
    assert [output.name for output in outputs] == ["job_fe11"]
