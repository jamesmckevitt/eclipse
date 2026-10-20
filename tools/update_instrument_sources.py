"""
Write the sources and plain defaults in docs/instrument-sources.md from the code.

CONTRIBUTING.md describes the page's tables and which cells this script
writes. Run it as `python tools/update_instrument_sources.py`.
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
# A comment that names settings like a table mark, and so must be one.
MARK_LIKE = re.compile(rf"<!--\s*(?:\w+\s+settings\s*:|(?:{'|'.join(INSTRUMENTS)})\b)",
                       re.IGNORECASE)
FENCE = re.compile(r" {0,3}(```|~~~)")
SETTING = re.compile(r"`([a-z]+\.\w+)`")
NUMBER = r"[-+]?\d+(?:\.\d+)?"

# Words of the `unit` metadata, and the units they stand for.
UNIT_WORDS = {
    "Angstrom": u.AA, "arcsec": u.arcsec, "Celsius": u.deg_C, "DN": u.DN,
    "electrons": u.electron, "eV": u.eV, "K": u.K, "m": u.m, "mA": u.mAA, "micron": u.um,
    "mm": u.mm, "nm": u.nm, "ns": u.ns, "pixel": u.pix, "pixels": u.pix, "second": u.s,
    "us": u.us,
}
_WORD = "|".join(sorted(map(re.escape, UNIT_WORDS), key=len, reverse=True))
# Text between two numbers of one list, such as " and " or " micron, ".
LIST_JOIN = re.compile(rf"(?: (?:{_WORD})(?: per (?:{_WORD}))*)?(?:, | and | by | to )")
# A Default cell holding only a value: numbers and a unit, or one word.
PLAIN = re.compile(rf"{NUMBER}(?:(?:, | and ){NUMBER})*(?: (?:{_WORD})(?: per (?:{_WORD}))*)?"
                   r"|none|true|false|packaged table|`[^`|]*`")

# Settings whose hand-written Default cell does not give the default itself.
# The PSF row gives the design values used when the defaults are None, and
# the dispersion row gives the polynomial's degree.
# tests/test_instrument_sources.py checks these cells.
NOT_SHOWN = {("SWC", "telescope.psf_params"), ("SWC", "telescope.field_of_view"),
             ("SWC", "detector.dispersion")}


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
        if label and values:
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
    """Default written as a plain Default cell: "13.5 micron", "true"."""
    numbers = display_numbers(name, value, label)
    if not numbers:
        return _words(value if numbers is None else None)
    texts = [_number(n) for n in numbers]
    text = texts[0] if len(texts) == 1 else ", ".join(texts[:-1]) + " and " + texts[-1]
    return f"{text} {label}" if label else text


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
    if name not in fields or not fields[name].metadata.get("source"):
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


def _split_row(line, code_spans=True):
    """Cells of a table row, split at pipes that are not escaped or in code."""
    cells, cell, index, code = [], "", 0, None
    text = line.strip()
    text = text[1:] if text.startswith("|") else text
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            cell += text[index:index + 2]
            index += 2
            continue
        if char == "`" and code_spans:
            run = len(re.match(r"`+", text[index:]).group(0))
            code = None if code == run else (run if code is None else code)
            cell += "`" * run
            index += run
            continue
        if char == "|" and code is None:
            cells.append(cell.strip())
            cell = ""
        else:
            cell += char
        index += 1
    if code is not None:
        # An unclosed backtick is a literal one, as in Markdown.
        return _split_row(line, code_spans=False)
    if cell.strip():
        cells.append(cell.strip())
    return cells


def table_rows(lines):
    """
    Body rows of the tables that follow a table mark.

    A mark applies to the table after it and the blank line below it, and
    stops at the end of that table.

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
    instrument, pending, row, fence = None, None, None, None
    for index, line in enumerate(lines):
        line = line.rstrip()
        opening = FENCE.match(line)
        if fence or opening:
            if pending:
                raise ValueError(f"The table mark before line {index + 1} of {PAGE_NAME} "
                                 f"must be followed by a blank line and its table.")
            if fence is None:
                fence = opening.group(1)
            elif opening and opening.group(1) == fence:
                fence = None
            instrument, row = None, None
            continue
        if MARK_LIKE.match(line):
            mark = TABLE_MARK.fullmatch(line)
            if not mark or mark.group(1) not in INSTRUMENTS:
                raise ValueError(f"Line {index + 1} of {PAGE_NAME} looks like a table mark "
                                 f"but is not one: {line!r}. Marks read <!-- SWC settings: "
                                 f"... --> or <!-- EIS settings: ... -->.")
            if pending or (index > 0 and lines[index - 1].strip()):
                raise ValueError(f"The table mark on line {index + 1} of {PAGE_NAME} must "
                                 f"follow a blank line.")
            if index + 1 >= len(lines) or lines[index + 1].strip():
                raise ValueError(f"The table mark on line {index + 1} of {PAGE_NAME} must be "
                                 f"followed by a blank line and its table.")
            pending, instrument, row = mark.group(1), None, None
        elif line.startswith("|"):
            if row is None:
                instrument, pending, row = pending, None, 0
            else:
                row += 1
            # The first two lines of a table are its header and separator.
            if instrument and row >= 2:
                yield index, instrument, _split_row(line)
        elif line or row is not None:
            if pending:
                raise ValueError(f"The table mark before line {index + 1} of {PAGE_NAME} "
                                 f"must be followed by a blank line and its table.")
            instrument, row = None, None
    if pending:
        raise ValueError(f"The last table mark of {PAGE_NAME} has no table.")


def _setting_rows(text):
    """Rows naming settings: line, instrument, cells and settings."""
    for index, instrument, cells in table_rows(text.split("\n")):
        keys = SETTING.findall(cells[0]) if cells else []
        if keys:
            if len(cells) != 3:
                raise ValueError(f"The row for {cells[0]} in {PAGE_NAME} has {len(cells)} "
                                 f"cells, not 3.")
            found = [(key, *setting(instrument, key)) for key in keys]
            yield index, instrument, cells, found


def _check_cell(text, key):
    """Refuse text that would break a table row."""
    if "\n" in text or "|" in text or text.endswith("\\"):
        raise ValueError(f"The text for {key} has a newline, a '|' or a final backslash, "
                         f"which would break its row in {PAGE_NAME}: {text!r}.")


def updated_page(text):
    """
    Page with its sources and plain Default cells rewritten from the code.

    Raises
    ------
    ValueError
        If a row names settings with different sources, or text would break
        a row.
    """
    lines = text.split("\n")
    for index, _, cells, found in list(_setting_rows(text)):
        keys = [key for key, _, _ in found]
        sources = {f.metadata["source"] for _, f, _ in found}
        if len(sources) > 1:
            raise ValueError(f"{', '.join(keys)} have different sources, so need a row each "
                             f"in {PAGE_NAME}.")
        source = sources.pop()
        _check_cell(source, keys[0])
        default = cells[1]
        if len(found) == 1 and (not default or PLAIN.fullmatch(default)):
            key, f, value = found[0]
            default = plain_default(key, value, f.metadata.get("unit", ""))
            _check_cell(default, key)
        lines[index] = f"| {cells[0]} | {default} | {source} |"
    return "\n".join(lines)


def _collapse(numbers):
    """Numbers with each run of equal ones kept once."""
    out = []
    for number in numbers:
        if not out or not math.isclose(out[-1], number, rel_tol=1e-9, abs_tol=1e-12):
            out.append(number)
    return out


def _starts_with(cell, wanted):
    """Whether the numbers in `cell` start with `wanted` and list no more after them."""
    tokens = list(re.finditer(NUMBER, cell))
    if len(tokens) < len(wanted) or not all(
            math.isclose(float(t.group(0)), n, rel_tol=1e-9, abs_tol=1e-12)
            for t, n in zip(tokens, wanted)):
        return False
    if wanted and len(tokens) > len(wanted):
        between = cell[tokens[len(wanted) - 1].end():tokens[len(wanted)].start()]
        return not LIST_JOIN.fullmatch(between)
    return True


def _shows(cell, found):
    """
    Whether a hand-written Default cell shows the defaults of its settings.

    A cell starting with "none" shows defaults that are None, empty, or a
    list of zeros.
    Otherwise the cell starts with the defaults' numbers, in order, and lists
    no more after them. Equal numbers in a row may be given once, either
    within a setting or, when all are equal, for the whole row. The cell
    must also give their units and words.
    """
    raw, collapsed, labels, words = [], [], set(), []
    for key, f, value in found:
        label = f.metadata.get("unit", "")
        shown = display_numbers(key, value, label)
        if not shown or (len(shown) > 1 and not any(shown)):
            words.append("none" if shown is not None else _words(value))
        else:
            raw += shown
            collapsed += _collapse(shown)
            labels.add(label)
    starts_none = cell.startswith("none")
    if starts_none or (not raw and "none" in words):
        return starts_none and not raw and all(word == "none" for word in words)
    merged = _collapse(raw)
    candidates = [raw, collapsed] + ([merged] if len(merged) == 1 else [])
    return (any(_starts_with(cell, wanted) for wanted in candidates)
            and all(label in cell for label in labels if label)
            and all(word in cell for word in words))


def hand_written_problems(text):
    """Messages for the hand-written Default cells that do not show the code's defaults."""
    problems = []
    for _, instrument, cells, found in _setting_rows(text):
        if len(found) == 1 and (not cells[1] or PLAIN.fullmatch(cells[1])):
            continue
        found = [item for item in found if (instrument, item[0]) not in NOT_SHOWN]
        if found and not _shows(cells[1], found):
            defaults = "; ".join(
                f"{key} as {plain_default(key, value, f.metadata.get('unit', ''))}"
                for key, f, value in found)
            problems.append(f"Edit the Default cell of the {instrument} row for {cells[0]} in "
                            f"{PAGE_NAME} by hand: it should show {defaults}.")
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
