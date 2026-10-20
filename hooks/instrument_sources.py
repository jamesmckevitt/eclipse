"""
MkDocs hook that fills in the tables of docs/instrument-sources.md from the code.

Each line of the page such as::

    <!-- setting SWC: detector.pix_size, detector.n_rows -->

becomes a table row with the settings, their defaults and their source, read
from the metadata of the settings' fields in `euvst_response.config`.
"""

import dataclasses
import re

import astropy.units as u

from euvst_response.config import (DETECTOR_MATERIALS, AluminiumFilter, Detector_EIS,
                                   Detector_SWC, Simulation, Telescope_EIS, Telescope_EUVST)

PAGE = "instrument-sources.md"

# The class behind each configuration section, for each instrument.
INSTRUMENTS = {
    "SWC": {"detector": Detector_SWC, "telescope": Telescope_EUVST, "filter": AluminiumFilter,
            "simulation": Simulation},
    "EIS": {"detector": Detector_EIS, "telescope": Telescope_EIS, "simulation": Simulation},
}

PLACEHOLDER = re.compile(r"^<!-- setting (\w+): (.+?) -->[ \t]*$", re.MULTILINE)

# Words of the "unit" metadata, and the units they stand for.
_UNIT_WORDS = {
    "Angstrom": u.AA, "arcsec": u.arcsec, "Celsius": u.deg_C, "DN": u.DN,
    "electrons": u.electron, "eV": u.eV, "K": u.K, "m": u.m, "mA": u.mAA, "micron": u.um,
    "mm": u.mm, "nm": u.nm, "ns": u.ns, "pixel": u.pix, "pixels": u.pix, "rows": None,
    "second": u.s, "us": u.us,
}


def _unit(label):
    """Unit for a label such as "electrons per pixel per second", or None for a plain number."""
    parts = label.split(" per ")
    unknown = [part for part in parts if part not in _UNIT_WORDS]
    if unknown:
        raise ValueError(f"Unit {label!r} has words the docs do not know: {unknown}.")
    words = [_UNIT_WORDS[part] for part in parts]
    if words[0] is None and len(words) == 1:
        return None
    if None in words:
        raise ValueError(f"Unit {label!r} mixes a plain count with units.")
    unit = words[0]
    for word in words[1:]:
        unit = unit / word
    return unit


def _number(value):
    """Number as it reads in the docs: 13.5, 2048, -0.0006."""
    return f"{value:.9g}"


def _default(cls, f):
    """Default of field `f` of `cls`, or its material's where the default is None."""
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    value = f.default
    material = DETECTOR_MATERIALS.get(getattr(cls, "material", None), {})
    return material.get(f.name, value) if value is None else value


def _join(texts):
    """Texts joined with commas and "and", or with semicolons if they hold either."""
    if len(texts) == 1:
        return texts[0]
    if any(" and " in text or ", " in text for text in texts):
        return "; ".join(texts)
    return ", ".join(texts[:-1]) + " and " + texts[-1]


def default_value(cls, f):
    """
    Default of field `f` of `cls` as the page shows it.

    Returns
    -------
    value : str
        Default, such as "13.5" or "`ground`".
    unit : str
        Unit it is in, or "" for none.
    """
    meta = f.metadata
    value = _default(cls, f)
    if "shown" in meta:
        return meta["shown"], ""
    if value is None:
        return "none", ""
    if isinstance(value, bool):
        return str(value).lower(), ""
    if isinstance(value, str):
        return f"`{value}`", ""
    if hasattr(value, "is_file"):
        return "packaged table", ""
    label = meta.get("unit", "")
    unit = _unit(label) if label else None
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    if unit is not None:
        converted = []
        for v in values:
            try:
                converted.append(v.to_value(unit, equivalencies=u.temperature()))
            except u.UnitConversionError:
                # A value per pixel shown without "per pixel", such as 10 electrons.
                converted.append((v * u.pix).to_value(unit))
        values = converted
    numbers = [_number(v) for v in values]
    if meta.get("range"):
        if len(numbers) != 2:
            raise ValueError(f"{f.name} is shown as a range, so needs two values.")
        return f"{numbers[0]} to {numbers[1]}", label
    if len(set(numbers)) == 1:
        numbers = numbers[:1]
    return _join(numbers), label


def setting_row(instrument, keys):
    """
    Table row for settings that share a source.

    Parameters
    ----------
    instrument : str
        `"SWC"` or `"EIS"`.
    keys : list of str
        Configuration keys, such as `"detector.pix_size"`.

    Returns
    -------
    str
        Markdown row: the settings, their defaults and their source.

    Raises
    ------
    ValueError
        If the instrument is unknown, a key is not one of its settings with a
        source, or the settings' sources differ.
    """
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument {instrument!r}; use {', '.join(INSTRUMENTS)}.")
    found = []
    for key in keys:
        section, _, name = key.partition(".")
        cls = INSTRUMENTS[instrument].get(section)
        fields = {f.name: f for f in dataclasses.fields(cls)} if cls else {}
        if name not in fields or "source" not in fields[name].metadata:
            raise ValueError(f"{key} is not an {instrument} setting with a source.")
        found.append((cls, fields[name]))
    sources = {f.metadata["source"] for _, f in found}
    if len(sources) > 1:
        raise ValueError(f"{', '.join(keys)} have different sources, so need a row each.")

    values = [default_value(cls, f) for cls, f in found]
    notes = [f.metadata.get("note", "") for _, f in found]
    shared_note = notes[0] if len(set(notes)) == 1 else ""
    if shared_note:
        notes = [""] * len(notes)
    units = {unit for _, unit in values}
    if len(set(zip(values, notes))) == 1:
        # The same default for each, such as 1 for both transfer efficiencies.
        value, unit = values[0]
        default = f"{value} {unit}".strip() + notes[0]
    elif (len(units) == 1 and not any(notes)
          and not any(" and " in v or " to " in v for v, _ in values)):
        # One unit, given once: 1485, 95 and 0 Angstrom.
        default = f"{_join([v for v, _ in values])} {units.pop()}".strip()
    else:
        default = _join([f"{value} {unit}".strip() + note
                         for (value, unit), note in zip(values, notes)])
    settings = ", ".join(f"`{key}`" for key in keys)
    cells = [settings, default + shared_note, sources.pop()]
    return "| " + " | ".join(_escape_pipes(cell) for cell in cells) + " |"


def _escape_pipes(text):
    """`text` with each `|` outside code spans escaped, so it stays in its table cell."""
    pieces = text.split("`")
    pieces[::2] = [piece.replace("|", "\\|") for piece in pieces[::2]]
    return "`".join(pieces)


def render(markdown):
    """Page with each placeholder line replaced by its table row."""
    def row(match):
        keys = [key.strip() for key in match.group(2).split(",")]
        return setting_row(match.group(1), keys)
    return PLACEHOLDER.sub(row, markdown)


def on_page_markdown(markdown, page, **kwargs):
    """Fill in the instrument sources page."""
    if page.file.src_uri == PAGE:
        return render(markdown)
    return markdown
