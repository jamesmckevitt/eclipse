"""Every instrument setting has a source, and docs/instrument-sources.md shows the code's."""
import dataclasses
import importlib.util
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from euvst_response import resolution_budget
from euvst_response.config import (MEASURED_SLIT_IMAGE_TILT, TILT_REFERENCE_ANGLE,
                                   AluminiumFilter, Detector_EIS, Detector_SWC, Simulation,
                                   Telescope_EIS, Telescope_EUVST)

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "update_instrument_sources", ROOT / "tools" / "update_instrument_sources.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

RUN = "Run python tools/update_instrument_sources.py and see CONTRIBUTING.md."

# Settings with no source of their own, and why.
WITHOUT_SOURCE = {
    AluminiumFilter: set(),
    # The material is a modelling choice; its properties have rows of their own.
    Detector_SWC: {"material"},
    # As for SWC, and EIS takes its material properties from the same table.
    Detector_EIS: {"material", "fano_factor", "pair_energy_300k", "pair_energy_slope"},
    # The PSF shape is a modelling choice, and the filter is a section of its own.
    Telescope_EUVST: {"psf_type", "filter"},
    # Modelling choices left off by default, and the date of an observation.
    Telescope_EIS: {"psf_type", "psf_slit_width", "psf_across_slit", "date"},
    # Choices of a run, not facts about the instrument.
    Simulation: {"expos", "n_iter", "ncpu", "instrument", "vis_sl", "psf", "psf_boundary",
                 "spectral_psf", "slit_mask", "noise", "enable_pinholes", "pinhole_sizes",
                 "pinhole_positions", "pinhole_positions_spectral"},
}


def _sourced(cls):
    return {f.name for f in dataclasses.fields(cls) if f.metadata.get("source")}


def _page():
    return tool.PAGE.read_text()


def _cell(instrument, first_cell):
    """Default cell of the row with this first cell in a table of `instrument`."""
    lines = _page().split("\n")
    found = [cells[1] for _, inst, cells in tool.table_rows(lines)
             if inst == instrument and cells[0] == first_cell]
    assert len(found) == 1, (f"Expected one {instrument} row {first_cell!r} in "
                             f"{tool.PAGE_NAME}; restore it or update this test.")
    return found[0]


@pytest.mark.parametrize("cls", WITHOUT_SOURCE, ids=lambda cls: cls.__name__)
def test_every_instrument_setting_has_a_source(cls):
    settings = {f.name for f in dataclasses.fields(cls) if f.init}
    missing = settings - _sourced(cls) - WITHOUT_SOURCE[cls]
    assert not missing, (f"Give {cls.__name__}.{', '.join(sorted(missing))} a `source` in the "
                         f"field metadata, or add it to WITHOUT_SOURCE here with a reason.")
    extra = _sourced(cls) & WITHOUT_SOURCE[cls]
    assert not extra, f"Remove {', '.join(sorted(extra))} from WITHOUT_SOURCE here."


def test_page_has_a_row_for_every_setting_with_a_source():
    shown, seen = set(), set()
    for _, instrument, cells in tool.table_rows(_page().split("\n")):
        for key in tool.SETTING.findall(cells[0]):
            assert (instrument, key) not in seen, (
                f"{key} has two rows in the {instrument} tables of {tool.PAGE_NAME}; keep one.")
            seen.add((instrument, key))
            section, _, name = key.partition(".")
            cls = tool.INSTRUMENTS[instrument].get(section)
            assert cls, f"{key} is not an {instrument} setting; move its row in {tool.PAGE_NAME}."
            shown.add((cls, name))
    for cls in WITHOUT_SOURCE:
        missing = sorted(name for name in _sourced(cls) if (cls, name) not in shown)
        assert not missing, (f"Add a row for {cls.__name__}.{', '.join(missing)} to a table of "
                             f"{tool.PAGE_NAME}. {RUN}")


def test_page_is_up_to_date():
    page = _page()
    assert tool.updated_page(page) == page, f"{tool.PAGE_NAME} differs from the code. {RUN}"
    problems = tool.hand_written_problems(page)
    assert not problems, "\n".join(problems)


def test_script_changes_nothing_the_second_time():
    once = tool.updated_page(_page())
    assert tool.updated_page(once) == once, "update_instrument_sources.py must be idempotent."


SYNTHETIC = """Text.

<!-- SWC settings: written by the script -->

| Setting | Default | Source |
| --- | --- | --- |
| `detector.max_dn` | 1 DN | old |
| `detector.gain_e_per_dn` | 1 electrons per DN, by hand | old |
| Something set in the code | 0.5 | Not recorded |

## Next

| Setting | Default | Source |
| --- | --- | --- |
| `detector.max_dn` | outside a marked table | kept |
"""


def test_writes_sources_and_plain_defaults_only():
    lines = tool.updated_page(SYNTHETIC).split("\n")
    max_dn, _ = tool.setting("SWC", "detector.max_dn")
    gain, _ = tool.setting("SWC", "detector.gain_e_per_dn")
    assert lines[6] == (f"| `detector.max_dn` | "
                        f"{tool.plain_default('', Detector_SWC.max_dn, 'DN')} | "
                        f"{max_dn.metadata['source']} |")
    assert lines[7] == (f"| `detector.gain_e_per_dn` | 1 electrons per DN, by hand | "
                        f"{gain.metadata['source']} |")
    assert lines[8:] == SYNTHETIC.split("\n")[8:]
    assert any("detector.gain_e_per_dn" in p for p in tool.hand_written_problems(SYNTHETIC))


@pytest.mark.parametrize("key, value", [
    ("detector.serial_prescan", 51),
    # The new value appears elsewhere in the cell, for another setting or in a note.
    ("detector.serial_prescan", 20),
    ("detector.serial_overscan", 200),
    ("detector.qe_vis", 0.76),
    ("detector.slit_image_tilt", (0.991, 0.605)),
    ("telescope.psf_across_slit", 0.3 * u.arcsec),
    ("detector.n_columns", 4096),
    ("detector.cte_serial", 0.99999),
    ("detector.lit_band", (171.0 * u.AA, 212.3 * u.AA)),
    ("simulation.slit_width", None),
    ("telescope.fiducial_centres", []),
    ("telescope.fiducial_lengths", []),
    ("telescope.pointing_jitter", 0.0 * u.arcsec),
])
def test_hand_written_defaults_are_checked(monkeypatch, key, value):
    """A changed default fails the check until its hand-written cell is edited."""
    f, _ = tool.setting("SWC", key)
    if f.default_factory is dataclasses.MISSING:
        monkeypatch.setattr(f, "default", value)
    else:
        monkeypatch.setattr(f, "default_factory", lambda: value)
    problems = tool.hand_written_problems(_page())
    assert len(problems) == 1 and f"`{key}`" in problems[0]


@pytest.mark.parametrize("value, label, text", [
    ((13.5 * u.um).cgs / u.pix, "micron", "13.5 micron"),
    ((16.9 * u.mAA).cgs / u.pix, "mA per pixel", "16.9 mA per pixel"),
    (7.0 * u.electron / u.pix, "electrons", "7 electrons"),
    (-40 * u.deg_C, "Celsius", "-40 Celsius"),
    (1e6 * u.electron / u.pix, "electrons", "1000000 electrons"),
    (2e-7, "", "0.0000002"),
    ([1.5 * u.arcsec, -2.0 * u.arcsec], "arcsec", "1.5 and -2 arcsec"),
    (None, "", "none"), (True, "", "true"), ("ground", "", "`ground`"),
])
def test_plain_default_in_its_display_unit(value, label, text):
    assert tool.plain_default("x", value, label) == text


def test_refuses_bad_units_marks_and_cells(monkeypatch):
    with pytest.raises(ValueError, match="does not know"):
        tool.display_numbers("x", 1 * u.arcsec, "arcseconds")
    with pytest.raises(ValueError, match="x is a quantity"):
        tool.display_numbers("x", 1 * u.arcsec, "")
    with pytest.raises(ValueError, match="x is a plain number"):
        tool.display_numbers("x", 0.5, "arcsec")
    with pytest.raises(ValueError, match="cannot be shown"):
        tool.display_numbers("x", 1 * u.s, "arcsec")
    with pytest.raises(ValueError, match="not an EIS setting"):
        tool.setting("EIS", "simulation.slit_width")
    with pytest.raises(ValueError, match="Unknown instrument"):
        tool.setting("SW", "detector.qe_euv")
    for page in (SYNTHETIC.replace("SWC settings: written", "SWC settings : written -- >"),
                 SYNTHETIC.replace("<!-- SWC settings", "<!-- SW settings"),
                 SYNTHETIC.replace("Text.\n\n", "Text.\n"),
                 SYNTHETIC.replace("script -->\n\n", "script -->\n\n```\ncode\n```\n\n"),
                 SYNTHETIC.replace("script -->\n\n", "script -->\n")):
        with pytest.raises(ValueError, match="table mark"):
            list(tool.table_rows(page.split("\n")))
    f, _ = tool.setting("SWC", "detector.max_dn")
    for bad in ("Table 6-1 | 6-2", "two\nlines", "ends in \\"):
        monkeypatch.setattr(f, "metadata", {**f.metadata, "source": bad})
        with pytest.raises(ValueError, match="detector.max_dn"):
            tool.updated_page(SYNTHETIC)


def test_rows_split_at_pipes_outside_code():
    assert tool._split_row(r"| `a|b` | c \| d | ``x`|`` |") == ["`a|b`", r"c \| d", "``x`|``"]
    assert tool._split_row("| `a | b |") == ["`a", "b"]


def test_a_mark_applies_to_its_table_only():
    page = SYNTHETIC.replace("## Next\n\n", "")
    page = page.replace("| Something set in the code | 0.5 | Not recorded |\n\n",
                        "| Something set in the code | 0.5 | Not recorded |\n\nMore text.\n\n")
    assert page.endswith("| `detector.max_dn` | outside a marked table | kept |\n")
    assert tool.updated_page(page).endswith(
        "| `detector.max_dn` | outside a marked table | kept |\n")


def test_hand_written_facts_match_the_code():
    """Hand-written values that are not settings match the code."""
    psf = _cell("SWC", "`telescope.psf_params`, through `telescope.field_of_view`")
    fields = Telescope_EUVST.FIELDS_OF_VIEW
    plate_scale = Detector_SWC.plate_scale_angle.to_value(u.arcsec / u.pix)
    spectral_step = Detector_SWC.wvl_res.to_value(u.mAA / u.pix)
    # The design FWHMs, in pixels rounded to 0.01 as the code holds them.
    for field, arcsec in (("280x280", 0.423), ("100x100", 0.195), ("70x70", 0.161)):
        pixels = round(arcsec / plate_scale, 2)
        assert f"{arcsec:.3f}" in psf and pixels == fields[field][0].value, (
            f"Edit the SWC psf_params row of {tool.PAGE_NAME} to the design FWHM of {field}.")
    assert "43.00 mA" in psf and all(round(43.00 / spectral_step, 2) == f[1].value
                                     for f in fields.values()), (
        f"Edit the spectral FWHM in the SWC psf_params row of {tool.PAGE_NAME}.")
    assert Telescope_EUVST().field_of_view == "280x280", (
        f"Edit the default field in the SWC psf_params row of {tool.PAGE_NAME}.")

    assert len(Detector_SWC().dispersion) == 3 and _cell("SWC", "`detector.dispersion`") == (
        "quadratic in position along the dispersion"), (
        f"Edit the dispersion row of {tool.PAGE_NAME} to the polynomial's degree.")

    tilt = _cell("SWC", "`detector.slit_image_tilt`")
    assert (f"{MEASURED_SLIT_IMAGE_TILT[0]} f + {MEASURED_SLIT_IMAGE_TILT[1]} f^2" in tilt
            and f"over {TILT_REFERENCE_ANGLE.to_value(u.arcsec):g} arcsec" in tilt), (
        f"Edit the slit_image_tilt row of {tool.PAGE_NAME} to MEASURED_SLIT_IMAGE_TILT.")

    # Column 0 of the budget's rows is the current best estimate, the default budget.
    jitter = resolution_budget.SPATIAL_ROWS[resolution_budget.JITTER_ROW][0]
    assert (Telescope_EUVST().resolution_budget == "current_estimate"
            and f"the budget's {jitter:.3f} arcsec" in _cell("SWC", "`telescope.pointing_jitter`")
            ), f"Edit the pointing_jitter row of {tool.PAGE_NAME} to the budget's jitter."

    for name in ("serial_prescan", "serial_overscan"):
        Detector_SWC(**{name: 0})
        Detector_SWC(**{name: 200})
        for bad in (-1, 201):
            with pytest.raises(ValueError):
                Detector_SWC(**{name: bad})
    assert "each settable from 0 to 200" in _cell(
        "SWC", "`detector.serial_prescan`, `detector.serial_overscan`")

    for width in (0.2, 0.4, 0.8, 1.6):
        Simulation(slit_width=width * u.arcsec)
    with pytest.raises(ValueError):
        Simulation(slit_width=0.3 * u.arcsec)
    assert "the slits are 0.2, 0.4, 0.8 and 1.6 arcsec wide" in _cell(
        "SWC", "`simulation.slit_width`")


@pytest.mark.parametrize("instrument, cls, law", [
    ("SWC", Detector_SWC, "the 293 K value times 122 T^3 exp(-6400/T), held at its 198 K value "
                          "below 198 K"),
    ("EIS", Detector_EIS, "the 293 K value times 1.14e6 T^3 exp(-9080/T), held at its 198 K "
                          "value below 198 K"),
])
def test_dark_current_law_rows_match_the_code(instrument, cls, law):
    assert _cell(instrument, "Dark current against temperature, set in the code") == law
    factor, activation = (122, 6400) if cls is Detector_SWC else (1.14e6, 9080)
    rate = cls.dark_current_293k
    for kelvin, used in ((250.0, 250.0), (198.0, 198.0), (180.0, 198.0)):
        got = cls.calculate_dark_current(kelvin * u.K, rate).value
        assert got == pytest.approx(rate.value * factor * used**3 * np.exp(-activation / used),
                                    rel=1e-12), (
            f"Edit the {instrument} dark current row of {tool.PAGE_NAME} to the law in "
            f"calculate_dark_current.")


def test_aperture_share_row_matches_the_code():
    assert _cell("SWC", "Half the aperture feeding the SW channel, set in the code") == "0.5"
    telescope = Telescope_EUVST()
    assert telescope.collecting_area.to_value(u.m**2) == pytest.approx(
        0.5 * np.pi * (telescope.D_ap.to_value(u.m) / 2) ** 2), (
        f"Edit the aperture share row of {tool.PAGE_NAME} to Telescope_EUVST.collecting_area.")


def test_script_leaves_hand_written_none_and_design_values():
    tilt, _ = tool.setting("SWC", "detector.slit_image_tilt")
    assert not tool._written_by_script("SWC", "none", [("detector.slit_image_tilt", tilt,
                                                        (0.0, 0.0))])
    dispersion, value = tool.setting("SWC", "detector.dispersion")
    assert not tool._written_by_script("SWC", "", [("detector.dispersion", dispersion, value)])
