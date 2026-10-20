"""Every instrument setting has a source, and docs/instrument-sources.md shows the code's."""
import dataclasses
import importlib.util
import re
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
    # The PSF shape is a modelling choice, the slit width the PSF is for and the
    # blur across the slit are off by default, and the date is an observation's.
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
    shown = set()
    for _, instrument, cells in tool.table_rows(_page().split("\n")):
        for key in tool.SETTING.findall(cells[0]):
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


def _default(key):
    return tool.setting("SWC", key)[1]


def _not_none(value, unit):
    """A nonzero value in `unit` for a default that is None."""
    return 0.3 * unit if value is None else None


# Each changes a default to a value derived from the current one, so the
# hand-written cell no longer shows it.
CHANGES = {
    "prescan up by one": ("detector.serial_prescan", lambda v: v + 1),
    # The new value appears elsewhere in the cell, for another setting or in a note.
    "prescan to the overscan": ("detector.serial_prescan",
                                lambda v: _default("detector.serial_overscan")),
    "qe_vis to qe_euv": ("detector.qe_vis", lambda v: _default("detector.qe_euv")),
    "tilt measured": ("detector.slit_image_tilt",
                      lambda v: MEASURED_SLIT_IMAGE_TILT if not any(v) else (0.0, 0.0)),
    "blur across the slit": ("telescope.psf_across_slit",
                             lambda v: _not_none(v, u.arcsec) if v is None else 2 * v),
    "columns doubled": ("detector.n_columns", lambda v: 2 * v),
    "imperfect transfer": ("detector.cte_serial", lambda v: v * 0.99999),
    "band moved": ("detector.lit_band", lambda v: (v[0] + 1 * u.AA, v[1])),
    "no slit width": ("simulation.slit_width", lambda v: None),
    "no fiducials": ("telescope.fiducial_centres", lambda v: []),
    "jitter given": ("telescope.pointing_jitter",
                     lambda v: _not_none(v, u.arcsec) if v is None else None),
}


@pytest.mark.parametrize("key, change", CHANGES.values(), ids=CHANGES)
def test_hand_written_defaults_are_checked(monkeypatch, key, change):
    """A changed default fails the check until its hand-written cell is edited."""
    f, current = tool.setting("SWC", key)
    value = change(current)
    assert repr(value) != repr(current), f"Make the change for {key} differ from its default."
    if f.default_factory is dataclasses.MISSING:
        monkeypatch.setattr(f, "default", value)
    else:
        monkeypatch.setattr(f, "default_factory", lambda: value)
    problems = tool.hand_written_problems(_page())
    assert len(problems) == 1 and f"`{key}`" in problems[0], (
        f"Changing {key} to {value!r} should give one problem naming it, got {problems}.")


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
    for page in (SYNTHETIC.replace("SWC settings: written", "SWC settings written"),
                 SYNTHETIC.replace("SWC settings: written", "swc settings: written -- >"),
                 SYNTHETIC.replace("SWC settings: written", "SWC Settings written"),
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


NUM = r"[-+]?\d+(?:\.\d+)?(?:e\d+)?"


def _read(instrument, first_cell, pattern):
    """Numbers of a hand-written cell read in a fixed form."""
    cell = _cell(instrument, first_cell)
    match = re.fullmatch(pattern.replace("NUM", f"({NUM})"), cell)
    assert match, (f"The {instrument} row {first_cell} of {tool.PAGE_NAME} is no longer in the "
                   f"form tests/test_instrument_sources.py reads; update the pattern there.")
    return [float(group) for group in match.groups()]


def test_psf_row_matches_the_fields_of_view():
    """The design FWHMs, in pixels rounded to 0.01 as `FIELDS_OF_VIEW` holds them."""
    first = "`telescope.psf_params`, through `telescope.field_of_view`"
    along, side, _, small, smaller, spectral, slit = _read(
        "SWC", first, r"NUM arcsec along the slit for the default NUM x NUM arcsec field "
                      r"\(NUM and NUM for the smaller fields\); NUM mA in wavelength with the "
                      r"NUM arcsec slit")
    telescope = Telescope_EUVST()
    plate_scale = Detector_SWC.plate_scale_angle.to_value(u.arcsec / u.pix)
    step = Detector_SWC.wvl_res.to_value(u.mAA / u.pix)
    fields = sorted(Telescope_EUVST.FIELDS_OF_VIEW.items(), key=lambda item: -item[1][0].value)
    edit = f"Edit the SWC psf_params row of {tool.PAGE_NAME} or Telescope_EUVST.FIELDS_OF_VIEW"
    assert telescope.field_of_view == f"{side:g}x{side:g}" == fields[0][0], (
        f"{edit}: the default field is {telescope.field_of_view}.")
    for (name, widths), arcsec in zip(fields, (along, small, smaller)):
        assert round(arcsec / plate_scale, 2) == widths[0].value, (
            f"{edit}: field {name} is {widths[0].value} pixels along the slit, "
            f"{widths[0].value * plate_scale:.3f} arcsec.")
        assert round(spectral / step, 2) == widths[1].value, (
            f"{edit}: field {name} is {widths[1].value} pixels in wavelength, "
            f"{widths[1].value * step:.2f} mA.")
    assert slit == telescope.psf_slit_width.to_value(u.arcsec), (
        f"{edit}: the spectral FWHM is for the {telescope.psf_slit_width} slit.")


def test_fiducial_lengths_row_matches_the_code():
    narrow, widest = _read("SWC", "`telescope.fiducial_centres`, `telescope.fiducial_lengths`",
                           r"[^;]*; NUM arcsec, NUM arcsec for the widest slit")
    lengths = [length.to_value(u.arcsec) for length in Telescope_EUVST().fiducial_lengths]
    assert lengths == [narrow] * (len(lengths) - 1) + [widest], (
        f"Edit the fiducial row of {tool.PAGE_NAME}: the lengths for the slits from narrowest "
        f"to widest are {lengths} arcsec.")


def test_dispersion_and_tilt_rows_match_the_code():
    degree = {2: "linear", 3: "quadratic", 4: "cubic"}[len(Detector_SWC().dispersion)]
    assert _cell("SWC", "`detector.dispersion`").startswith(degree), (
        f"Edit the dispersion row of {tool.PAGE_NAME}: the polynomial is {degree}.")
    linear, quadratic, angle = _read(
        "SWC", "`detector.slit_image_tilt`",
        r"[^;]*; measured as a drift along the dispersion of NUM f \+ NUM f\^2 rows, where f "
        r"is the angle along the slit over NUM arcsec")
    assert ((linear, quadratic) == tuple(MEASURED_SLIT_IMAGE_TILT)
            and angle == TILT_REFERENCE_ANGLE.to_value(u.arcsec)), (
        f"Edit the slit_image_tilt row of {tool.PAGE_NAME} to MEASURED_SLIT_IMAGE_TILT "
        f"{MEASURED_SLIT_IMAGE_TILT} and TILT_REFERENCE_ANGLE {TILT_REFERENCE_ANGLE}.")


def test_jitter_scan_and_slit_rows_match_the_code():
    budget = Telescope_EUVST().resolution_budget
    jitter = _read("SWC", "`telescope.pointing_jitter`",
                   r"none: the budget's NUM arcsec, or no jitter with `resolution_budget: none`")
    assert budget in resolution_budget.BUDGETS[:2], (
        f"Edit the pointing_jitter row of {tool.PAGE_NAME}: the default budget is {budget}.")
    row = resolution_budget.SPATIAL_ROWS[resolution_budget.JITTER_ROW]
    expected = row[resolution_budget.BUDGETS.index(budget)]
    assert jitter == [expected], (
        f"Edit the pointing_jitter row of {tool.PAGE_NAME}: the {budget} budget's jitter is "
        f"{expected:.3f} arcsec.")

    least, most = _read("SWC", "`detector.serial_prescan`, `detector.serial_overscan`",
                        r"[^;]*; each settable from NUM to NUM")
    for name in ("serial_prescan", "serial_overscan"):
        for good, bad in ((least, least - 1), (most, most + 1)):
            Detector_SWC(**{name: int(good)})
            with pytest.raises(ValueError):
                Detector_SWC(**{name: int(bad)})

    cell = _cell("SWC", "`simulation.slit_width`")
    widths = [float(w) for w in re.findall(NUM, cell.split("the slits are", 1)[-1])]
    for width in widths:
        Simulation(slit_width=width * u.arcsec)
    for width in np.arange(0.1, 2.0, 0.1):
        if not np.isclose(widths, width).any():
            with pytest.raises(ValueError, match="slit_width"):
                Simulation(slit_width=round(width, 1) * u.arcsec)


@pytest.mark.parametrize("instrument, cls", [("SWC", Detector_SWC), ("EIS", Detector_EIS)])
def test_dark_current_law_rows_match_the_code(instrument, cls):
    factor, activation, held, below = _read(
        instrument, "Dark current against temperature, set in the code",
        r"the 293 K value times NUM T\^3 exp\(-NUM/T\), held at its NUM K value below NUM K")
    rate = cls.dark_current_293k
    for kelvin in (290.0, 250.0, below, below - 20):
        used = max(kelvin, held)
        got = cls.calculate_dark_current(kelvin * u.K, rate).value
        assert got == pytest.approx(rate.value * factor * used**3 * np.exp(-activation / used),
                                    rel=1e-12), (
            f"Edit the {instrument} dark current row of {tool.PAGE_NAME} to the law in "
            f"calculate_dark_current.")


def test_aperture_share_row_matches_the_code():
    share, = _read("SWC", "Half the aperture feeding the SW channel, set in the code", "NUM")
    telescope = Telescope_EUVST()
    full = np.pi * (telescope.D_ap.to_value(u.m) / 2) ** 2
    assert telescope.collecting_area.to_value(u.m**2) == pytest.approx(share * full), (
        f"Edit the aperture share row of {tool.PAGE_NAME} to Telescope_EUVST.collecting_area "
        f"over the aperture's area.")


def test_check_reads_items_in_order():
    """Words, numbers and "none" must start the cell in the order of the settings."""
    def found(instrument, key, value):
        return (key, tool.setting(instrument, key)[0], value)

    ground = [found("EIS", "telescope.calibration", "ground")]
    assert tool._shows("`ground`, as calibrated before launch", ground)
    assert not tool._shows("before launch; not `ground`", ground)
    slit = [found("SWC", "telescope.fiducial_centres", []),
            found("SWC", "telescope.slit_length", 300.0 * u.arcsec)]
    assert tool._shows("none; 300 arcsec", slit)
    assert not tool._shows("300 arcsec; none", slit)
    assert not tool._shows("300 arcsec", slit)


def test_script_keeps_hand_written_cells_and_refuses_repeats(monkeypatch):
    tilt, _ = tool.setting("SWC", "detector.slit_image_tilt")
    monkeypatch.setattr(tilt, "default", (0.0, 0.0))
    page = SYNTHETIC.replace(
        "| Something set in the code",
        "| `detector.slit_image_tilt` | none | old |\n"
        "| `detector.dispersion` | quadratic | old |\n"
        "| `detector.ccd_gap` |  | old |\n"
        "| Something set in the code")
    lines = tool.updated_page(page).split("\n")
    assert lines[8].startswith("| `detector.slit_image_tilt` | none | ")
    assert lines[9].startswith("| `detector.dispersion` | quadratic | ")
    assert lines[10].startswith(
        f"| `detector.ccd_gap` | {tool.plain_default('', Detector_SWC().ccd_gap, 'mm')} | ")
    assert not [p for p in tool.hand_written_problems(page) if "slit_image_tilt" in p
                or "dispersion" in p]
    empty = page.replace("| `detector.dispersion` | quadratic |", "| `detector.dispersion` |  |")
    assert any("Write the Default cell" in p for p in tool.hand_written_problems(empty))
    with pytest.raises(ValueError, match="named more than once"):
        tool.updated_page(page.replace("`detector.ccd_gap`", "`detector.dispersion`"))
