"""
Write the defaults and sources in docs/instrument-sources.md from the code.

Each table of settings follows a comment line naming its instrument, such as
`<!-- SWC settings: ... -->`. A row whose first cell names settings, such as
`detector.pix_size`, gets the settings' defaults, in the `unit` of their field
metadata, and their `source`. Other rows are left as they are.
"""

import dataclasses
import re
import string
import sys
from pathlib import Path

import astropy.units as u
import numpy as np

from euvst_response import resolution_budget
from euvst_response.config import (MEASURED_SLIT_IMAGE_TILT, TILT_REFERENCE_ANGLE,
                                   AluminiumFilter, Detector_EIS, Detector_SWC, Simulation,
                                   Telescope_EIS, Telescope_EUVST, _material_value)

PAGE = Path(__file__).resolve().parents[1] / "docs" / "instrument-sources.md"

# The class behind each configuration section, for each instrument.
INSTRUMENTS = {
    "SWC": {"detector": Detector_SWC, "telescope": Telescope_EUVST, "filter": AluminiumFilter,
            "simulation": Simulation},
    "EIS": {"detector": Detector_EIS, "telescope": Telescope_EIS},
}

TABLE_MARK = re.compile(r"^<!-- (\w+) settings: .*-->$")
SETTING = re.compile(r"`([a-z]+\.\w+)`")

# Words of the "unit" metadata, and the units they stand for.
UNIT_WORDS = {
    "Angstrom": u.AA, "arcsec": u.arcsec, "Celsius": u.deg_C, "DN": u.DN,
    "electrons": u.electron, "eV": u.eV, "K": u.K, "m": u.m, "mA": u.mAA, "micron": u.um,
    "mm": u.mm, "nm": u.nm, "ns": u.ns, "pixel": u.pix, "pixels": u.pix, "second": u.s,
    "us": u.us,
}

# The RSC-2022021C design values that `Telescope_EUVST.FIELDS_OF_VIEW` gives in
# pixels. tests/test_instrument_sources.py checks the two agree.
EUVST_PSF = ("0.423 arcsec along the slit for the default 280 x 280 arcsec field (0.195 and "
             "0.161 for the smaller fields); 43.00 mA in wavelength with the 0.2 arcsec slit")

# Wording of rows whose default needs more than the values, keyed by the row's
# first setting. `{name}` is a setting of the row, `{all}` all of them as a
# row without wording shows them, and the other names are those of
# `_constants`. The format specs are those of `Value`.
WORDING = {
    "SWC": {
        "telescope.psf_params": EUVST_PSF,
        "telescope.psf_slit_width": "{psf_slit_width}, the slit the spectral PSF above is "
                                    "given for",
        "telescope.psf_across_slit": "{psf_across_slit}: across the slit only the budget terms "
                                     "below",
        "telescope.pointing_jitter": "{pointing_jitter}: the budget's {budget_jitter:.3f} arcsec, "
                                     "or no jitter with `resolution_budget: none`",
        "simulation.slit_width": "{slit_width}; the slits are 0.2, 0.4, 0.8 and 1.6 arcsec wide",
        "telescope.fiducial_centres": "{fiducial_centres[0]} and {fiducial_centres[1]:+} arcsec; "
                                      "{fiducial_lengths[0]} arcsec, {fiducial_lengths[3]} "
                                      "arcsec for the widest slit",
        "filter.al_table": "packaged tables for {table_thickness} layers",
        "detector.pix_size": "{pix_size}, {n_rows} by {n_columns}",
        "detector.serial_prescan": "{all}; each settable from 0 to 200",
        "detector.cte_parallel": "{all}, a perfect transfer",
        "detector.dispersion": "{dispersion:degree} in position along the dispersion",
        "detector.lit_band": "{lit_band:range}",
        "detector.wavelength_offset": "{wavelength_offset}; the alignment of the camera to the "
                                      "beam is known to +/-0.79 Angstrom",
        "detector.slit_image_tilt": "{slit_image_tilt:none}; measured as a drift along the "
                                    "dispersion of {tilt[0]} f + {tilt[1]} f^2 rows, where f "
                                    "is the angle along the slit over {tilt_angle}",
        "detector.fano_factor": "{fano_factor}, for {material}",
        "detector.pair_energy_300k": "{pair_energy_300k} and {pair_energy_slope}, for "
                                     "{material}",
        "filter.visible_decade_thickness": "{visible_decade_thickness}: visible light falls by a "
                                           "factor 10 per {visible_decade_thickness} of "
                                           "aluminium, times the mesh",
    },
    "EIS": {
        "telescope.psf_params": "{psf_params:each} FWHM",
    },
}

DEGREES = {1: "linear", 2: "quadratic", 3: "cubic"}


def _number(value):
    """Number as the page shows it, without exponents: 13.5, 2048, -0.0006."""
    text = np.format_float_positional(float(f"{value:.12g}"), trim="-")
    return "0" if text == "-0" else text


def _join(texts):
    """Texts joined with commas and a final "and"."""
    return texts[0] if len(texts) == 1 else ", ".join(texts[:-1]) + " and " + texts[-1]


def _unit(label, name):
    """Unit for a label such as "electrons per pixel per second"."""
    parts = label.split(" per ")
    unknown = [part for part in parts if part not in UNIT_WORDS]
    if unknown:
        raise ValueError(f"{name} has unit {label!r}, with words this script does not know: "
                         f"{unknown}.")
    unit = UNIT_WORDS[parts[0]]
    for part in parts[1:]:
        unit = unit / UNIT_WORDS[part]
    return unit


class Number(float):
    """A number formatted as the page shows it; `+` adds a plus sign."""

    def __format__(self, spec):
        if spec in ("", "+"):
            text = _number(self)
            return "+" + text if spec == "+" and self > 0 else text
        return float.__format__(self, spec)


class Value:
    """
    Default of one setting, formatted for the page.

    Format specs: none for the default with its unit, `n` without its unit,
    `each` for a list shown once when its values are equal, `range` for a
    pair as "a to b", `none` for "none" when every value is zero, and
    `degree` for the degree of a polynomial's coefficients. Indexing gives
    one value of a list as a `Number`.
    """

    def __init__(self, name, value, label=""):
        self.name, self.label, self.value = name, label, value
        self.numbers = None
        if value is None or isinstance(value, (bool, str)) or hasattr(value, "is_file"):
            return
        values = list(value) if isinstance(value, (list, tuple)) or np.ndim(value) else [value]
        quantity = any(isinstance(v, u.Quantity) for v in values)
        if quantity and not label:
            raise ValueError(f"{name} is a quantity, so needs a `unit` in its metadata.")
        if not quantity and label:
            raise ValueError(f"{name} is a plain number, so takes no `unit` in its metadata.")
        if quantity:
            unit = _unit(label, name)
            converted = []
            for v in values:
                try:
                    converted.append(v.to_value(unit, equivalencies=u.temperature()))
                except u.UnitConversionError:
                    # A value per pixel shown without "per pixel", such as 10 electrons.
                    try:
                        converted.append((v * u.pix).to_value(unit))
                    except u.UnitConversionError:
                        raise ValueError(f"{name} is in {v.unit}, which cannot be shown in "
                                         f"{label!r}.") from None
            values = converted
        self.numbers = [Number(v) for v in values]

    def __getitem__(self, index):
        return self.numbers[index]

    def _with_unit(self, text):
        return f"{text} {self.label}" if self.label else text

    def __format__(self, spec):
        if self.numbers is None:
            if spec:
                raise ValueError(f"{self.name} has no numbers for the format {spec!r}.")
            if self.value is None:
                return "none"
            if isinstance(self.value, bool):
                return str(self.value).lower()
            if isinstance(self.value, str):
                return f"`{self.value}`"
            return "packaged table"
        numbers = [format(n) for n in self.numbers]
        if spec == "each" and len(set(numbers)) == 1:
            return self._with_unit(numbers[0])
        if spec == "range":
            if len(numbers) != 2:
                raise ValueError(f"{self.name} is shown as a range, so needs two values.")
            return self._with_unit(f"{numbers[0]} to {numbers[1]}")
        if spec == "none":
            return "none" if not any(self.numbers) else self._with_unit(_join(numbers))
        if spec == "degree":
            return DEGREES[len(numbers) - 1]
        if spec == "n":
            return _join(numbers)
        if spec in ("", "each"):
            return self._with_unit(_join(numbers))
        raise ValueError(f"Unknown format {spec!r} for {self.name}.")


def _default(cls, f):
    """Default of field `f` of `cls`, or its material's where the default is None."""
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    if f.default is None and hasattr(cls, "material"):
        return _material_value(cls, f.name)
    return f.default


def _constants():
    """Values from the code that the wording quotes besides the settings."""
    column = resolution_budget._column(Telescope_EUVST.resolution_budget)
    budget = resolution_budget.SPATIAL_ROWS[resolution_budget.JITTER_ROW][column]
    return {"budget_jitter": Number(budget), "material": Detector_SWC.material,
            "tilt": Value("MEASURED_SLIT_IMAGE_TILT", MEASURED_SLIT_IMAGE_TILT),
            "tilt_angle": Value("TILT_REFERENCE_ANGLE", TILT_REFERENCE_ANGLE, "arcsec")}


def _plain(values):
    """Default cell for settings with no wording of their own."""
    texts = [format(v) for v in values]
    if len(set(texts)) == 1:
        return texts[0]
    if (len({v.label for v in values}) == 1
            and all(v.numbers and len(v.numbers) == 1 for v in values)):
        # One unit, given once: 1485, 95 and 0 Angstrom.
        return values[0]._with_unit(_join([format(v, "n") for v in values]))
    return ", ".join(texts)


def cells(instrument, keys):
    """
    Default and Source cells of a row naming settings that share a source.

    Parameters
    ----------
    instrument : str
        `"SWC"` or `"EIS"`.
    keys : list of str
        Configuration keys, such as `"detector.pix_size"`.

    Returns
    -------
    tuple of str
        Default and source.

    Raises
    ------
    ValueError
        If the instrument is unknown, a key is not one of its settings with a
        source, the settings' sources differ, or the wording leaves out a
        setting.
    """
    if instrument not in INSTRUMENTS:
        raise ValueError(f"Unknown instrument {instrument!r}; use {', '.join(INSTRUMENTS)}.")
    values, sources, hidden = {}, set(), []
    for key in keys:
        section, _, name = key.partition(".")
        cls = INSTRUMENTS[instrument].get(section)
        fields = {f.name: f for f in dataclasses.fields(cls)} if cls else {}
        if name not in fields or "source" not in fields[name].metadata:
            raise ValueError(f"{key} is not an {instrument} setting with a source.")
        f = fields[name]
        default = _default(cls, f)
        values[name] = Value(key, default, f.metadata.get("unit", ""))
        sources.add(f.metadata["source"])
        if default is not None and not hasattr(default, "is_file"):
            hidden.append(name)
    if len(sources) > 1:
        raise ValueError(f"{', '.join(keys)} have different sources, so need a row each.")

    wording = WORDING[instrument].get(keys[0])
    if wording is None:
        default = _plain(list(values.values()))
    else:
        used = {re.split(r"[.\[]", name)[0]
                for _, name, _, _ in string.Formatter().parse(wording) if name}
        missing = [name for name in hidden if name not in used and "all" not in used]
        if missing:
            raise ValueError(f"The wording for {keys[0]} leaves out {', '.join(missing)}.")
        unknown = used - set(values) - set(_constants()) - {"all"}
        if unknown:
            raise ValueError(f"The wording for {keys[0]} names {', '.join(sorted(unknown))}, "
                             f"which are not in its row.")
        default = wording.format(all=_plain(list(values.values())), **values, **_constants())
    row = (default, sources.pop())
    if any("|" in cell for cell in row):
        raise ValueError(f"A '|' in the row for {', '.join(keys)} would split its table cell.")
    return row


def updated_page(text):
    """Page `text` with the Default and Source cells of each setting's row rewritten."""
    lines = text.split("\n")
    instrument, in_table = None, False
    for index, line in enumerate(lines):
        line = line.rstrip()
        mark = TABLE_MARK.match(line)
        if mark:
            instrument, in_table = mark.group(1), False
        elif not line.startswith("|"):
            if in_table:
                instrument, in_table = None, False
        elif instrument:
            in_table = True
            row = [cell.strip() for cell in line.strip().strip("|").split("|")]
            keys = SETTING.findall(row[0])
            if keys:
                if len(row) != 3:
                    raise ValueError(f"Row for {row[0]} has {len(row)} cells, not 3.")
                lines[index] = "| " + " | ".join([row[0], *cells(instrument, keys)]) + " |"
    return "\n".join(lines)


def main():
    text = PAGE.read_text()
    updated = updated_page(text)
    if updated == text:
        print(f"{PAGE.name} is up to date.")
    else:
        PAGE.write_text(updated)
        print(f"Updated {PAGE.name}.")


if __name__ == "__main__":
    sys.exit(main())
