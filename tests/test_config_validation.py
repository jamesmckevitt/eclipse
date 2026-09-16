"""Config keys ECLIPSE does not read are now rejected, not ignored.

A key at the wrong depth is the dangerous case. It is a real parameter name,
so it looks right, but nothing reads it and the run silently uses the default.
A sweep written that way is worse still: it returns a full set of results that
happen to be identical across every combination.
"""
import sys

import pytest

from euvst_response.main import (
    _FITTING_COMPONENT_KEYS, _FITTING_KEYS, _SIMULATION_KEYS, _TOP_LEVEL_KEYS,
    _validate_config_keys, main,
)
from euvst_response.utils import check_config_keys, suggest_config_key

MINIMAL = {"instrument": "SWC", "uniform_intensity": "5000 erg / (s cm2 sr)"}


def test_a_valid_config_passes():
    config = dict(
        MINIMAL,
        n_iter=10, ncpu=-1, offchip_bin_slit=[1, 2], fit_signals="dn",
        simulation={"slit_width": "0.2 arcsec", "expos": ["5 s", "20 s"],
                    "psf": True, "vis_sl": "0 ph / (s cm2)", "noise": False,
                    "enable_pinholes": False},
        detector={"ccd_temperature": "-60 Celsius", "qe_euv": 0.76},
        telescope={"microroughness_sigma": "0.3 nm"},
        filter={"al_thickness": "1500 AA"},
        fitting={"components": [{"wavelength": "195.119 AA"},
                                {"wavelength": "195.179 AA",
                                 "tie_center": 0, "tie_width": 0}],
                 "primary_component": 0, "max_iter": 500},
    )
    _validate_config_keys(config, "SWC")


def test_the_old_flat_layout_is_rejected_with_the_right_place():
    """The case from the issue: parameters one level too high."""
    config = dict(MINIMAL, expos=["5 s", "20 s"], ccd_temperature="-60 Celsius")

    with pytest.raises(ValueError) as excinfo:
        _validate_config_keys(config, "SWC")

    message = str(excinfo.value)
    assert "'expos' in the 'simulation:' section" in message
    assert "'ccd_temperature' in the 'detector:' section" in message


def test_renamed_keys_name_their_replacement():
    for old, new in (("aluminium_thickness", "filter.al_thickness"),
                     ("slit_bin_pairs", "offchip_bin_slit")):
        with pytest.raises(ValueError, match=new.replace(".", r"\.")):
            _validate_config_keys(dict(MINIMAL, **{old: 1}), "SWC")


def test_a_typo_suggests_the_near_miss():
    with pytest.raises(ValueError, match="'reference_line'"):
        _validate_config_keys(dict(MINIMAL, refernce_line="Fe12_195.1190"),
                              "SWC")


def test_an_unrecognisable_key_is_still_reported():
    """No suggestion is better than a misleading one."""
    with pytest.raises(ValueError) as excinfo:
        _validate_config_keys(dict(MINIMAL, zzzzqqq=1), "SWC")
    assert "zzzzqqq" in str(excinfo.value)


def test_section_keys_are_checked_too():
    with pytest.raises(ValueError, match="'simulation' section"):
        _validate_config_keys(
            dict(MINIMAL, simulation={"slit_widht": "0.2 arcsec"}), "SWC")


def test_simulation_rejects_keys_that_belong_at_the_top_level():
    """n_iter under simulation: was parsed and then dropped on the floor."""
    with pytest.raises(ValueError, match="at the top level"):
        _validate_config_keys(
            dict(MINIMAL, simulation={"n_iter": 100}), "SWC")


def test_simulation_rejects_the_pinhole_lists():
    """They are top-level keys; main() never looks for them in the section."""
    with pytest.raises(ValueError, match="at the top level"):
        _validate_config_keys(
            dict(MINIMAL, simulation={"pinhole_sizes": ["20 um"]}), "SWC")


def test_telescope_rejects_a_nested_filter():
    """main() builds the filter from the top-level section and drops this."""
    with pytest.raises(ValueError, match="'telescope' section"):
        _validate_config_keys(
            dict(MINIMAL, telescope={"filter": {"al_thickness": "1500 AA"}}),
            "SWC")


def test_eis_and_swc_have_different_valid_keys():
    """filter_distance is an SWC detector field; EIS has no filter."""
    config = dict(MINIMAL, instrument="EIS",
                  detector={"filter_distance": "250 mm"})
    _validate_config_keys(dict(MINIMAL, detector={"filter_distance": "250 mm"}),
                          "SWC")
    with pytest.raises(ValueError, match="'detector' section"):
        _validate_config_keys(config, "EIS")


def test_eis_still_accepts_the_keys_it_warns_about():
    """microroughness_sigma and filter: are ignored loudly, not silently."""
    _validate_config_keys(
        dict(MINIMAL, instrument="EIS",
             telescope={"microroughness_sigma": "0.3 nm"},
             filter={"al_thickness": "1500 AA"}),
        "EIS")


def test_eis_telescope_keys_are_accepted():
    _validate_config_keys(
        dict(MINIMAL, instrument="EIS",
             telescope={"calibration": "dz2025", "date": "2012-06-03"}),
        "EIS")


def test_fitting_section_and_components_are_checked():
    with pytest.raises(ValueError, match="'fitting' section"):
        _validate_config_keys(
            dict(MINIMAL, fitting={"componets": []}), "SWC")

    with pytest.raises(ValueError, match=r"fitting\.components\[1\]"):
        _validate_config_keys(
            dict(MINIMAL, fitting={"components": [
                {"wavelength": "195.119 AA"},
                {"wavelength": "195.179 AA", "tie_centre": 0}]}),
            "SWC")


@pytest.mark.parametrize("value", [["0.2 arcsec"], None])
@pytest.mark.parametrize(
    "name", ["simulation", "detector", "telescope", "filter", "fitting"])
def test_a_section_that_is_not_a_mapping_is_rejected(name, value):
    """None is what a heading left empty parses to."""
    with pytest.raises(ValueError, match=f"'{name}:' section must be a mapping"):
        _validate_config_keys(dict(MINIMAL, **{name: value}), "SWC")


def test_fitting_components_must_be_a_list_of_mappings():
    """main() would otherwise fail on these with a TypeError, or skip fitting."""
    with pytest.raises(ValueError, match="'fitting.components' must be a list"):
        _validate_config_keys(
            dict(MINIMAL, fitting={"components": {"wavelength": "195.119 AA"}}),
            "SWC")

    with pytest.raises(ValueError,
                       match=r"'fitting\.components\[0\]' must be a mapping"):
        _validate_config_keys(
            dict(MINIMAL, fitting={"components": [
                None, {"wavelength": "195.179 AA"}]}),
            "SWC")


def test_an_unknown_instrument_is_rejected():
    """main() would build it as EIS; there are no valid keys to check against."""
    with pytest.raises(ValueError, match="Unknown instrument 'FOO'"):
        _validate_config_keys(dict(MINIMAL, instrument="FOO"), "FOO")


def test_main_reports_a_simulation_section_that_is_not_a_mapping(
        tmp_path, monkeypatch):
    """The instrument check before validation must not fail on it first."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("simulation: 1\n")
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config_file)])
    with pytest.raises(ValueError, match="'simulation:' section must be a mapping"):
        main()


def test_every_unknown_key_is_listed_at_once():
    """One run per typo would be a poor way to migrate 160 configs."""
    with pytest.raises(ValueError) as excinfo:
        _validate_config_keys(dict(MINIMAL, aaa=1, bbb=2, ccc=3), "SWC")
    message = str(excinfo.value)
    assert "aaa" in message and "bbb" in message and "ccc" in message


def test_the_message_lists_what_is_valid():
    with pytest.raises(ValueError, match="Valid top-level keys"):
        _validate_config_keys(dict(MINIMAL, nope=1), "SWC")


def test_private_fields_are_accepted_but_not_advertised():
    """_dark_current_293k is a real constructor argument, just not a knob."""
    _validate_config_keys(
        dict(MINIMAL, detector={"_dark_current_293k": 1.0}), "SWC")
    with pytest.raises(ValueError) as excinfo:
        _validate_config_keys(dict(MINIMAL, detector={"nope": 1}), "SWC")
    assert "_dark_current_293k" not in str(excinfo.value)


def test_check_config_keys_passes_when_everything_is_known():
    assert check_config_keys({"a": 1}, {"a", "b"}, "test") is None


def test_suggest_returns_none_when_nothing_is_close():
    assert suggest_config_key("zzzzqqq", {"alpha", "beta"}) is None


# --- the key lists have to agree with what main() reads ----------------------

# main() and the helpers it hands the parsed config to.
_CONFIG_READERS = ("main", "_parse_pinhole_config")


def _keys_main_reads(*names):
    """String keys main() looks up in any of the dicts called *names*.

    Covers ``d["key"]``, ``d.get("key")`` and ``"key" in d``, in main() and in
    the helpers listed in ``_CONFIG_READERS``.
    """
    import ast
    import importlib

    from pathlib import Path

    source = Path(importlib.import_module("euvst_response.main").__file__)
    tree = ast.parse(source.read_text())
    readers = [node for node in tree.body
               if isinstance(node, ast.FunctionDef)
               and node.name in _CONFIG_READERS]
    assert {node.name for node in readers} == set(_CONFIG_READERS)

    keys = set()
    for node in (n for reader in readers for n in ast.walk(reader)):
        if isinstance(node, ast.Subscript):
            target, key = node.value, node.slice
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "get" and node.args):
            target, key = node.func.value, node.args[0]
        elif (isinstance(node, ast.Compare) and len(node.ops) == 1
              and isinstance(node.ops[0], (ast.In, ast.NotIn))):
            target, key = node.comparators[0], node.left
        else:
            continue
        if (isinstance(target, ast.Name) and target.id in names
                and isinstance(key, ast.Constant) and isinstance(key.value, str)):
            keys.add(key.value)
    return keys


@pytest.mark.parametrize("names, accepted", [
    (("config",), _TOP_LEVEL_KEYS),
    (("all_sim", "sim_fixed", "sim_sweep"), _SIMULATION_KEYS),
    (("fitting_cfg",), _FITTING_KEYS),
    (("comp_dict",), _FITTING_COMPONENT_KEYS),
], ids=["top-level", "simulation", "fitting", "fitting.components"])
def test_the_key_lists_match_what_main_reads(names, accepted):
    """The lists are written out by hand, so a new key can be missed.

    simulation.noise is the example: main() reads it, and a key list written
    before it existed would refuse every config that sets it.
    """
    read = _keys_main_reads(*names)
    assert read - accepted == set(), (
        f"main() reads {sorted(read - accepted)} but the validator rejects them")
    assert accepted - read == set(), (
        f"the validator accepts {sorted(accepted - read)} but main() never "
        f"reads them from {', '.join(names)}")


# --- the documentation has to agree with the validator -----------------------

def _doc_yaml_blocks():
    """Every config example in the docs, with the page it came from.

    That is each ```yaml block in a page, and each notebook cell that writes a
    YAML file with %%writefile.
    """
    import json
    import re

    from pathlib import Path

    docs = Path(__file__).resolve().parent.parent / "docs"
    blocks = []
    for page in sorted(docs.glob("*.md")):
        for match in re.finditer(r"```yaml\n(.*?)```", page.read_text(), re.S):
            blocks.append((page.name, match.group(1)))
    for notebook in sorted(docs.glob("*.ipynb")):
        for cell in json.loads(notebook.read_text())["cells"]:
            source = "".join(cell["source"])
            first_line, _, body = source.partition("\n")
            if (cell["cell_type"] == "code"
                    and re.fullmatch(r"%%writefile\s+\S+\.ya?ml\s*", first_line)):
                blocks.append((notebook.name, body))
    return blocks


def test_every_config_example_in_the_docs_is_valid():
    """Otherwise the docs teach a config the code refuses.

    This is the check that would have caught an over-tight key list: the
    examples are the closest thing to a corpus of configs people write.

    A block that names its instrument is checked against that one. Several
    examples are fragments showing one section, and take their instrument
    from the prose around them, so those have to pass for either instrument
    rather than for an assumed default. That still catches a key no
    instrument accepts, which is the mismatch worth finding.
    """
    import yaml

    blocks = _doc_yaml_blocks()
    assert len(blocks) > 5, "expected the docs to carry config examples"
    assert any(page.endswith(".ipynb") for page, _ in blocks), (
        "expected the config written by the reproduction notebook")

    for page, block in blocks:
        config = yaml.safe_load(block)
        if not isinstance(config, dict):
            continue

        named = config.get("instrument")
        candidates = [str(named).upper()] if named else ["SWC", "EIS"]

        errors = []
        for instrument in candidates:
            try:
                _validate_config_keys(config, instrument)
                break
            except ValueError as error:
                errors.append(f"as {instrument}: {error}")
        else:
            raise AssertionError(
                f"config example in docs/{page} is rejected:\n"
                + "\n".join(errors))
