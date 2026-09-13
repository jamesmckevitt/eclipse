"""Config keys ECLIPSE does not read are now rejected, not ignored.

A key at the wrong depth is the dangerous case. It is a real parameter name,
so it looks right, but nothing reads it and the run silently uses the default.
A sweep written that way is worse still: it returns a full set of results that
happen to be identical across every combination.
"""
import pytest

from euvst_response.main import _validate_config_keys
from euvst_response.utils import check_config_keys, suggest_config_key

MINIMAL = {"instrument": "SWC", "uniform_intensity": "5000 erg / (s cm2 sr)"}


def test_a_valid_config_passes():
    config = dict(
        MINIMAL,
        n_iter=10, ncpu=-1, offchip_bin_slit=[1, 2], fit_signals="dn",
        simulation={"slit_width": "0.2 arcsec", "expos": ["5 s", "20 s"],
                    "psf": True, "vis_sl": "0 ph / (s cm2)",
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


def test_a_section_that_is_not_a_mapping_is_rejected():
    with pytest.raises(ValueError, match="must be a mapping"):
        _validate_config_keys(dict(MINIMAL, simulation=["0.2 arcsec"]), "SWC")


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
