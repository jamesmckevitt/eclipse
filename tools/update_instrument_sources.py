"""
Write the sources in docs/instrument-sources.md, and simple defaults, from the code.

The tables this script writes follow a comment line naming their instrument,
`<!-- SWC settings: ... -->` or `<!-- EIS settings: ... -->`, and a blank
line. In those tables, a row whose first cell names settings in backticks
gets their `source` from the field metadata in `euvst_response.config`. Its
Default cell is written too when the row names one setting and the cell holds
only a value, such as `65535 DN`. Other Default cells are worded by hand, and
`hand_written_problems` lists those that no longer show the code's defaults.
Run it as `python tools/update_instrument_sources.py`.
"""

import dataclasses
import math
import re
import sys
from pathlib import Path

import astropy.units as u
import numpy as np

if __name__ == "__main__":
    # This checkout's package, not one installed elsewhere.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from euvst_response.config import (DETECTOR_MATERIALS, AluminiumFilter,  # noqa: E402
                                   Detector_EIS, Detector_SWC, Simulation, Telescope_EIS,
                                   Telescope_EUVST)

PAGE = Path(__file__).resolve().parents[1] / "docs" / "instrument-sources.md"
PAGE_NAME = "docs/instrument-sources.md"

# The class behind each configuration section, for each instrument.
INSTRUMENTS = {
    "SWC": {"detector": Detector_SWC, "telescope": Telescope_EUVST, "filter": AluminiumFilter,
            "simulation": Simulation},
    "EIS": {"detector": Detector_EIS, "telescope": Telescope_EIS},
}

TABLE_MARK = re.compile(r"<!-- (\w+) settings: [^\n]*-->")
SETTING = re.compile(r"`([a-z]+\.\w+)`")
NUMBER = r"[-+]?\d+(?:\.\d+)?"

# Words of the `unit` metadata, and the units they stand for.
UNIT_WORDS = {
    "Angstrom": u.AA, "arcsec": u.arcsec, "Celsius": u.deg_C, "DN": u.DN,
    "electrons": u.electron, "eV": u.eV, "K": u.K, "m": u.m, "mA": u.mAA, "micron": u.um,
    "mm": u.mm, "nm": u.nm, "ns": u.ns, "pixel": u.pix, "pixels": u.pix, "second": u.s,
    "us": u.us,
}

# Hand-written Default cells that do not show the setting's value, and why.
# tests/test_instrument_sources.py checks what they show instead.
NOT_SHOWN = {
    ("SWC", "telescope.psf_params"): "None, for the design values of field_of_view",
    ("SWC", "telescope.field_of_view"): "None, for 280x280; the cell gives the design values",
    ("SWC", "detector.dispersion"): "coefficients, shown as the polynomial's degree",
}


def _number(value):
    """Number as the page shows it, without exponents: 13.5, 2048, -0.0006."""
    text = np.format_float_positional(float(f"{value:.12g}"), trim="-")
    return "0" if text == "-0" else text


def _unit(label, name):
    """Unit for a label such as "electrons per pixel per second"."""
    parts = label.split(" per ")
    unknown = [part for part in parts if part not in UNIT_WORDS]
    if unknown:
        raise ValueError(f"{name} has unit {label!r}, with words this script does not know: "
                         f"{unknown}. Add them to UNIT_WORDS in tools/"
                         f"update_instrument_sources.py.")
    unit = UNIT_WORDS[parts[0]]
    for part in parts[1:]:
        unit = unit / UNIT_WORDS[part]
    return unit


def display_numbers(name, value, label):
    """
    Numbers of a default in its display unit, or None if it holds no numbers.

    Parameters
    ----------
    name : str
        Setting, for error messages.
    value : object
        Default.
    label : str
        Display unit from the `unit` metadata, such as "micron", or "".

    Raises
    ------
    ValueError
        If a quantity has no unit, a plain number has one, or the unit does
        not fit.
    """
    if value is None or isinstance(value, (bool, str)) or hasattr(value, "is_file"):
        return None
    values = list(value) if isinstance(value, (list, tuple)) or np.ndim(value) else [value]
    quantity = any(isinstance(v, u.Quantity) for v in values)
    if quantity and not label:
        raise ValueError(f"{name} is a quantity, so needs a `unit` in its field metadata.")
    if not quantity:
        if label:
            raise ValueError(f"{name} is a plain number, so takes no `unit` in its field "
                             f"metadata.")
        return [float(v) for v in values]
    unit = _unit(label, name)
    numbers = []
    for v in values:
        try:
            numbers.append(v.to_value(unit, equivalencies=u.temperature()))
        except u.UnitConversionError:
            # A value per pixel shown without "per pixel", such as 10 electrons.
            try:
                numbers.append((v * u.pix).to_value(unit))
            except u.UnitConversionError:
                raise ValueError(f"{name} is in {v.unit}, which cannot be shown in "
                                 f"{label!r}.") from None
    return numbers


def _words(value):
    """How the page shows a default that holds no numbers."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return f"`{value}`"
    return "packaged table"


def plain_default(name, value, label):
    """Default as a Default cell holding only a value: "13.5 micron", "true"."""
    numbers = display_numbers(name, value, label)
    if numbers is None:
        return _words(value)
    texts = [_number(n) for n in numbers]
    text = texts[0] if len(texts) == 1 else ", ".join(texts[:-1]) + " and " + texts[-1]
    return f"{text} {label}" if label else text


def _is_plain(cell, value, label):
    """Whether a Default cell holds only a value of the kind of `value`."""
    if not cell:
        return True
    if display_numbers("", value, label) is None:
        return cell in ("none", "true", "false", "packaged table") or bool(
            re.fullmatch(r"`[^`]*`", cell))
    unit = " " + re.escape(label) if label else ""
    return bool(re.fullmatch(rf"{NUMBER}(?:(?:, | and ){NUMBER})*{unit}", cell))


def setting(instrument, key):
    """
    Field and default of a setting.

    Returns
    -------
    field : dataclasses.Field
        Its field, whose metadata gives `source` and `unit`.
    default : object
        Its default, or its material's value where the default is None.

    Raises
    ------
    ValueError
        If the instrument is unknown, or the key is not one of its settings
        with a source.
    """
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument {instrument!r} in a table mark of {PAGE_NAME}; "
                         f"use {', '.join(INSTRUMENTS)}.")
    section, _, name = key.partition(".")
    cls = INSTRUMENTS[instrument].get(section)
    fields = {f.name: f for f in dataclasses.fields(cls)} if cls else {}
    if name not in fields or "source" not in fields[name].metadata:
        raise ValueError(f"{key}, in an {instrument} table of {PAGE_NAME}, is not an "
                         f"{instrument} setting with a source in its field metadata.")
    f = fields[name]
    if f.default_factory is not dataclasses.MISSING:
        default = f.default_factory()
    else:
        default = f.default
    material = DETECTOR_MATERIALS.get(getattr(cls, "material", None), {})
    if default is None and name in material:
        default = material[name]
    return f, default


def table_rows(lines):
    """
    Rows of the tables that follow a table mark.

    Yields
    ------
    index : int
        Line of the row.
    instrument : str
        Instrument the mark names.
    cells : list of str
        Its cells.

    Raises
    ------
    ValueError
        If a mark is malformed or is not followed by a blank line and a table.
    """
    instrument, pending, row = None, None, None
    for index, line in enumerate(lines):
        line = line.rstrip()
        if line.startswith("<!--") and "settings" in line:
            mark = TABLE_MARK.fullmatch(line)
            if not mark:
                raise ValueError(f"Line {index + 1} of {PAGE_NAME} looks like a table mark "
                                 f"but is not one: {line!r}.")
            if pending or row is not None:
                raise ValueError(f"The table mark on line {index + 1} of {PAGE_NAME} must "
                                 f"follow a blank line and be followed by one.")
            pending, instrument = mark.group(1), None
            if index + 1 >= len(lines) or lines[index + 1].strip():
                raise ValueError(f"The table mark on line {index + 1} of {PAGE_NAME} must be "
                                 f"followed by a blank line and its table.")
        elif line.startswith("|"):
            if row is None:
                instrument, pending, row = pending, None, 0
            else:
                row += 1
            # The first two lines of a table are its header and separator.
            if instrument and row >= 2:
                yield index, instrument, [cell.strip() for cell in line.strip("|").split("|")]
        elif line or row is not None:
            if pending and line:
                raise ValueError(f"The table mark before line {index + 1} of {PAGE_NAME} "
                                 f"must be followed by a blank line and its table.")
            instrument, row = None, None
    if pending:
        raise ValueError(f"The last table mark of {PAGE_NAME} has no table.")


def _setting_rows(text):
    """Rows naming settings: line, instrument, cells and keys."""
    lines = text.split("\n")
    for index, instrument, cells in table_rows(lines):
        keys = SETTING.findall(cells[0])
        if keys:
            if len(cells) != 3:
                raise ValueError(f"The row for {cells[0]} in {PAGE_NAME} has {len(cells)} "
                                 f"cells, not 3.")
            yield index, instrument, cells, keys


def _check_cell(text, key):
    """Refuse text that would break a table row."""
    if "\n" in text or "|" in text or text.endswith("\\"):
        raise ValueError(f"The text for {key} has a newline, a '|' or a final backslash, "
                         f"which would break its row in {PAGE_NAME}: {text!r}.")


def updated_page(text):
    """
    Page `text` with the sources, and the Default cells holding only a value, rewritten.

    Raises
    ------
    ValueError
        If a row names settings with different sources, or text would break
        a row.
    """
    lines = text.split("\n")
    for index, instrument, cells, keys in list(_setting_rows(text)):
        found = [(key, *setting(instrument, key)) for key in keys]
        sources = {f.metadata["source"] for _, f, _ in found}
        if len(sources) > 1:
            raise ValueError(f"{', '.join(keys)} have different sources, so need a row each "
                             f"in {PAGE_NAME}.")
        source = sources.pop()
        _check_cell(source, keys[0])
        default = cells[1]
        if len(keys) == 1:
            key, f, value = found[0]
            label = f.metadata.get("unit", "")
            if _is_plain(default, value, label):
                default = plain_default(key, value, label)
                _check_cell(default, key)
        lines[index] = f"| {cells[0]} | {default} | {source} |"
    return "\n".join(lines)


def _shows(cell, numbers, words, label):
    """Whether a hand-written Default cell shows a default's numbers or words, and unit."""
    if numbers is None:
        return words in cell
    if words == "none" and not any(numbers) and "none" in cell:
        return True
    shown = [float(token) for token in re.findall(NUMBER, cell)]
    return (all(any(math.isclose(n, s, rel_tol=1e-9, abs_tol=1e-12) for s in shown)
                for n in numbers)
            and (not label or label in cell))


def hand_written_problems(text):
    """Settings whose hand-written Default cell does not show the code's default."""
    problems = []
    for _, instrument, cells, keys in _setting_rows(text):
        for key in keys:
            f, value = setting(instrument, key)
            label = f.metadata.get("unit", "")
            if (instrument, key) in NOT_SHOWN:
                continue
            if len(keys) == 1 and _is_plain(cells[1], value, label):
                continue
            numbers = display_numbers(key, value, label)
            words = "none" if numbers is not None and not any(numbers) else _words(value)
            if not _shows(cells[1], numbers, words, label):
                problems.append(f"Edit the Default cell of the {instrument} row for {cells[0]} "
                                f"in {PAGE_NAME} by hand: it should show {key} as "
                                f"{plain_default(key, value, label)}.")
    return problems


def main():
    text = PAGE.read_text()
    updated = updated_page(text)
    if updated != text:
        PAGE.write_text(updated)
        print(f"Updated {PAGE_NAME}.")
    problems = hand_written_problems(updated)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
