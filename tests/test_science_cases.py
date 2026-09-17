"""The stored science-case configurations have to keep working.

They are there to test changes to the instrument or to ECLIPSE against a fixed
set of observations. A configuration that stopped running, or that ran with a
key ignored, would quietly remove a case from that set.
"""
import sys
from pathlib import Path

import astropy.units as u
import pytest
import yaml

from euvst_response.analysis import load_instrument_response_results
from euvst_response.config import Simulation
from euvst_response.main import _validate_config_keys
from euvst_response.utils import parse_yaml_input

CASES_DIR = Path(__file__).resolve().parents[1] / "science_cases"
CASES = sorted(CASES_DIR.glob("*.yaml"))


def test_every_science_case_line_has_a_config():
    """13 cases, with 33 lines in the short wavelength channel between them."""
    assert len(CASES) == 33
    assert len({path.stem.split("-")[0] for path in CASES}) == 13


@pytest.mark.parametrize("path", CASES, ids=lambda path: path.stem)
def test_each_config_is_valid(path):
    config = yaml.safe_load(path.read_text())
    _validate_config_keys(config, config["instrument"])

    rest = parse_yaml_input(config["rest_wavelength"])
    assert 170 * u.AA <= rest <= 210 * u.AA
    # The file name carries the line, so it has to agree with the config.
    assert path.stem.endswith(f"{rest.to_value(u.AA):.3f}".replace(".", ""))

    intensity = parse_yaml_input(config["uniform_intensity"])
    assert intensity.unit.is_equivalent(u.erg / (u.s * u.cm**2 * u.sr))
    assert intensity.value > 0
    assert parse_yaml_input(config["thermal_width"]).to_value(u.km / u.s) > 0

    # Simulation refuses a slit width the instrument does not have.
    simulation = config["simulation"]
    Simulation(instrument=config["instrument"],
               slit_width=parse_yaml_input(simulation["slit_width"]),
               expos=parse_yaml_input(simulation["expos"]))


def test_a_science_case_runs_from_start_to_finish(tmp_path, monkeypatch):
    """The brightest case, with the iterations cut down so that it is quick."""
    source = CASES_DIR / "2.1.2-flare_ribbons_fe24_192030.yaml"
    text = source.read_text()
    assert "n_iter: 512" in text and "ncpu: -1" in text
    config = tmp_path / source.name
    config.write_text(text.replace("n_iter: 512", "n_iter: 2")
                          .replace("ncpu: -1", "ncpu: 1"))

    from euvst_response.main import main

    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    monkeypatch.chdir(tmp_path)
    main()

    results = load_instrument_response_results(
        tmp_path / "run" / "result" / f"{source.stem}.pkl")
    combinations = results["results"]["all_combinations"].values()
    assert sorted(c["parameters"]["offchip_bin_slit"] for c in combinations) == [1, 2]
