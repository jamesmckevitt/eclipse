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
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eclipse.config import (DETECTOR_MATERIALS, AluminiumFilter,  # noqa: E402
                            Detector_EIS, Detector_LW, Detector_SWC, Simulation,
                            Telescope_EIS, Telescope_EUVST, Telescope_EUVST_LW)

PAGE = Path(__file__).resolve().parents[1] / "docs" / "instrument-sources.md"
PAGE_NAME = "docs/instrument-sources.md"

# The class behind each configuration section, for each instrument.
INSTRUMENTS = {
    "SWC": {"detector": Detector_SWC, "telescope": Telescope_EUVST, "filter": AluminiumFilter,
            "simulation": Simulation},
    "LW": {"detector": Detector_LW, "telescope": Telescope_EUVST_LW},
    "EIS": {"detector": Detector_EIS, "telescope": Telescope_EIS},
}

TABLE_MARK = re.compile(r"<!-- (\w+) settings: [^\n]*-->")
# A comment that starts like a table mark, and so must be one: an
# instrument's name and "settings", or any word, "settings" and a colon.
MARK_LIKE = re.compile(rf" {{0,3}}<!--\s*(?:(?:{'|'.join(INSTRUMENTS)})\s+settings\b"
                       rf"|\w+\s+settings\s*:)", re.IGNORECASE)
FENCE = re.compile(r" {0,3}(```|~~~)")
SETTING = re.compile(r"`([a-z]+\.\w+)`")
NUMBER = r"[-+]?\d+(?:\.\d+)?"
NUMBER_E = rf"{NUMBER}(?:e[-+]?\d+)?"

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
# A Default cell holding only a value: numbers, with or without a unit,
# "none", "true", "false", "packaged table" or a backticked word.
PLAIN = re.compile(rf"{NUMBER}(?:(?:, | and ){NUMBER})*(?: (?:{_WORD})(?: per (?:{_WORD}))*)?"
                   r"|none|true|false|packaged table|`[^`|]*`")

# Settings whose hand-written Default cell does not give the default itself.
# The PSF row gives the design values used when the defaults are None, the
# dispersion row gives the polynomial's degree, and the fiducial row gives
# one length for the narrow slits and one for the widest. The LW rows give
# the values each band takes when the defaults are None.
# tests/test_instrument_sources.py checks these cells.
NOT_SHOWN = {("SWC", "telescope.psf_params"), ("SWC", "telescope.field_of_view"),
             ("SWC", "detector.dispersion"), ("SWC", "telescope.fiducial_lengths"),
             ("LW", "telescope.psf_params"), ("LW", "detector.wvl_res"),
             ("LW", "detector.plate_scale_angle"), ("LW", "detector.dark_current")}


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


def _reads_as_none(numbers):
    """Whether numbers are a list of zeros, which the page gives as none."""
    return len(numbers) > 1 and not any(numbers)


def plain_default(name, value, label):
    """Default written as a plain Default cell: "13.5 micron", "true"."""
    numbers = display_numbers(name, value, label)
    if not numbers or _reads_as_none(numbers):
        # A list of zeros, such as no slit image tilt, reads as none.
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
            mark = TABLE_MARK.fullmatch(line.strip())
            if not mark or mark.group(1) not in INSTRUMENTS:
                raise ValueError(f"Line {index + 1} of {PAGE_NAME} looks like a table mark "
                                 f"but is not one: {line!r}. Marks read <!-- SWC settings: "
                                 f"... -->, <!-- LW settings: ... --> or <!-- EIS "
                                 f"settings: ... -->.")
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
    """
    Rows naming settings: line, instrument, cells and settings.

    Raises
    ------
    ValueError
        If a row does not have three cells, or a setting has two rows.
    """
    seen = set()
    for index, instrument, cells in table_rows(text.split("\n")):
        keys = SETTING.findall(cells[0]) if cells else []
        for key in keys:
            if (instrument, key) in seen:
                raise ValueError(f"{key} is named more than once in the {instrument} tables "
                                 f"of {PAGE_NAME}; name it in one row, once.")
            seen.add((instrument, key))
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


def _written_by_script(instrument, cell, found):
    """
    Whether the script writes this Default cell.

    It does for a row naming one setting, not in `NOT_SHOWN`, whose cell is
    empty or holds only a value.
    """
    if len(found) != 1 or (instrument, found[0][0]) in NOT_SHOWN:
        return False
    return not cell or bool(PLAIN.fullmatch(cell))


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
    for index, instrument, cells, found in list(_setting_rows(text)):
        keys = [key for key, _, _ in found]
        sources = {f.metadata["source"] for _, f, _ in found}
        if len(sources) > 1:
            raise ValueError(f"{', '.join(keys)} have different sources, so need a row each "
                             f"in {PAGE_NAME}.")
        source = sources.pop()
        _check_cell(source, keys[0])
        default = cells[1]
        if _written_by_script(instrument, default, found):
            key, f, value = found[0]
            default = plain_default(key, value, f.metadata.get("unit", ""))
            _check_cell(default, key)
        lines[index] = f"| {cells[0]} | {default} | {source} |"
    return "\n".join(lines)


# Items a hand-written Default cell gives, in order: numbers and words.
ITEM = re.compile(rf"(?<![\w.])(?P<number>{NUMBER_E})(?!\w|\.\d)"
                  r"|\b(?:none|true|false|packaged tables?)\b|`[^`]*`")


def _same(a, b):
    """Whether two items are the same number or word."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _expected(key, f, value):
    """
    Items that show one setting's default, as alternatives.

    Returns
    -------
    list of list
        Default's items in full, and the number once for a list of equal
        numbers.
    """
    shown = display_numbers(key, value, f.metadata.get("unit", ""))
    if shown is None:
        return [[_words(value)]]
    if not shown or _reads_as_none(shown):
        return [["none"]]
    if len(shown) > 1 and all(_same(n, shown[0]) for n in shown):
        return [shown, shown[:1]]
    return [shown]


def _cell_items(cell):
    """Numbers and words of a cell, each with its start and end."""
    items = []
    for match in ITEM.finditer(cell):
        text = match.group(0)
        item = "packaged table" if text.startswith("packaged") else text
        if match.group("number"):
            item = float(text)
        items.append((item, match.start(), match.end()))
    return items


def _shows(cell, found):
    """
    Whether a hand-written Default cell shows the defaults of its settings.

    The cell starts with each setting's default, in the order of the
    settings, and lists no more after them. A default reads as numbers, or
    as "none" for None, empty or a list of zeros, "true", "false", a
    backticked string or "packaged table". A list of equal numbers may be
    given once, and so may a default shared by every setting of the row or
    by neighbouring settings that are not numbers. The cell must also give
    the units.
    """
    expected = [_expected(key, f, value) for key, f, value in found]
    options = [[]]
    for alternatives in expected:
        options = [done + more for done in options for more in alternatives]
    for wanted in list(options):
        merged = [item for index, item in enumerate(wanted)
                  if index == 0 or isinstance(item, float) or not _same(item, wanted[index - 1])]
        options.append(merged)
        if all(_same(item, wanted[0]) for item in wanted):
            options.append(wanted[:1])
    options = [list(option) for option in dict.fromkeys(map(tuple, options))]
    items = _cell_items(cell)
    labels = {f.metadata.get("unit", "") for (_, f, _), alternatives in zip(found, expected)
              if isinstance(alternatives[0][0], float)} - {""}
    if not all(re.search(rf"(?<!\w){re.escape(label)}(?!\w)", cell) for label in labels):
        return False
    for wanted in options:
        if not items or items[0][1] > 0 or len(items) < len(wanted):
            continue
        if not all(_same(item[0], w) for item, w in zip(items, wanted)):
            continue
        if len(items) > len(wanted):
            between = cell[items[len(wanted) - 1][2]:items[len(wanted)][1]]
            if LIST_JOIN.fullmatch(between):
                continue
        return True
    return False


def hand_written_problems(text):
    """Messages for the hand-written Default cells that do not show the code's defaults."""
    problems = []
    for _, instrument, cells, found in _setting_rows(text):
        if _written_by_script(instrument, cells[1], found):
            continue
        if not cells[1]:
            problems.append(f"Write the Default cell of the {instrument} row for {cells[0]} in "
                            f"{PAGE_NAME} by hand.")
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
