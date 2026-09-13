"""The pinhole config lists are paired, and a half-specified one now raises.

pinhole_sizes and pinhole_positions carry one entry per pinhole, so a config
holding one without the other describes no pinhole at all. That used to run to
completion with no pinholes and no warning, which looks exactly like a run
where the pinholes were simulated and turned out not to matter.
"""
import sys

import astropy.units as u
import pytest

from euvst_response.main import _parse_pinhole_config

VALID = {
    "pinhole_sizes": ["20 um", "5 um"],
    "pinhole_positions": [0.3, 0.7],
}


def test_a_matched_pair_is_accepted():
    sizes, positions, spectral = _parse_pinhole_config(dict(VALID))
    assert [s.to_value(u.um) for s in sizes] == [20.0, 5.0]
    assert positions == [0.3, 0.7]
    assert spectral == []


def test_no_pinhole_keys_at_all_is_accepted():
    assert _parse_pinhole_config({"instrument": "SWC"}) == ([], [], [])


def test_positions_without_sizes_raises():
    """The case from the issue: this used to run and produce no pinholes."""
    with pytest.raises(ValueError, match="paired list"):
        _parse_pinhole_config({"pinhole_positions": [0.1, 0.2, 0.3]})


def test_sizes_without_positions_raises():
    with pytest.raises(ValueError, match="paired list"):
        _parse_pinhole_config({"pinhole_sizes": ["20 um"]})


def test_mismatched_lengths_raise_and_say_both_lengths():
    with pytest.raises(ValueError, match="2 size\\(s\\) and 1 position\\(s\\)"):
        _parse_pinhole_config({"pinhole_sizes": ["20 um", "5 um"],
                               "pinhole_positions": [0.3]})


def test_a_single_pinhole_may_be_written_unwrapped():
    """YAML scalars, not lists, are how a one-pinhole config usually reads."""
    sizes, positions, _ = _parse_pinhole_config(
        {"pinhole_sizes": "20 um", "pinhole_positions": 0.5})
    assert [s.to_value(u.um) for s in sizes] == [20.0]
    assert positions == [0.5]


@pytest.mark.parametrize("bad", [-0.1, 1.5, 10])
def test_positions_outside_the_detector_raise(bad):
    """The issue's example uses 10, 20, 30, which are not fractions."""
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        _parse_pinhole_config({"pinhole_sizes": ["20 um"],
                               "pinhole_positions": [bad]})


def test_positions_carrying_units_raise():
    """A position is a fraction, so '0.3 arcsec' is a misunderstanding."""
    with pytest.raises(ValueError, match="carry no units"):
        _parse_pinhole_config({"pinhole_sizes": ["20 um"],
                               "pinhole_positions": ["0.3 arcsec"]})


def test_spectral_positions_must_match_the_sizes():
    config = dict(VALID, pinhole_positions_spectral=[0.5])
    with pytest.raises(ValueError, match="same length as pinhole_sizes"):
        _parse_pinhole_config(config)


def test_spectral_positions_without_sizes_raise():
    with pytest.raises(ValueError, match="same length as pinhole_sizes"):
        _parse_pinhole_config({"pinhole_positions_spectral": [0.5]})


def test_spectral_positions_are_range_checked_at_config_time():
    """Checked here as well as at use, so a long run fails before it starts."""
    config = dict(VALID, pinhole_positions_spectral=[0.5, 1.2])
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        _parse_pinhole_config(config)


def test_a_fully_specified_config_survives():
    config = dict(VALID, pinhole_positions_spectral=[0.1, 0.8])
    sizes, positions, spectral = _parse_pinhole_config(config)
    assert len(sizes) == len(positions) == len(spectral) == 2
    assert spectral == [0.1, 0.8]


# --- the EIS guard, exercised through main() ---------------------------------
#
# EIS has no aluminium filter, so it has no pinholes to model. The guard used
# to test the sizes only, so an EIS config carrying positions alone passed it.

EIS_UNIFORM = """
instrument: EIS
uniform_intensity: 5000 erg / (s cm2 sr)
n_iter: 1
"""


def _run_main(tmp_path, monkeypatch, extra):
    """Run main() on a uniform-intensity EIS config plus *extra* YAML.

    Uniform-intensity mode needs no synthesis file, and every case here is
    rejected during configuration, so nothing is simulated.
    """
    from euvst_response.main import main

    cfg = tmp_path / "config.yaml"
    cfg.write_text(EIS_UNIFORM + extra)
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(cfg)])
    monkeypatch.chdir(tmp_path)
    main()


def test_eis_rejects_pinhole_sizes_and_positions(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="not supported for EIS"):
        _run_main(tmp_path, monkeypatch,
                  'pinhole_sizes: ["20 um"]\npinhole_positions: [0.3]\n')


def test_eis_rejects_enable_pinholes(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="not supported for EIS"):
        _run_main(tmp_path, monkeypatch,
                  "simulation:\n  enable_pinholes: True\n")


def test_eis_rejects_positions_alone(tmp_path, monkeypatch):
    """The issue's config. Previously ran to completion with no pinholes."""
    with pytest.raises(ValueError):
        _run_main(tmp_path, monkeypatch, "pinhole_positions: [0.1, 0.2, 0.3]\n")
