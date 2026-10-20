"""Every instrument setting has a source, and docs/instrument-sources.md shows them all."""
import dataclasses
import importlib.util
from pathlib import Path

import pytest

from euvst_response.config import (AluminiumFilter, Detector_EIS, Detector_SWC, Simulation,
                                   Telescope_EIS, Telescope_EUVST)

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "instrument-sources.md"

_spec = importlib.util.spec_from_file_location("instrument_sources",
                                               ROOT / "hooks" / "instrument_sources.py")
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

INSTRUMENT_CLASSES = (AluminiumFilter, Detector_SWC, Detector_EIS, Telescope_EUVST, Telescope_EIS)

# Settings that are not facts about the instrument, or that EIS leaves off or
# shares with SWC, so have no source of their own.
WITHOUT_SOURCE = {
    "AluminiumFilter": set(),
    "Detector_SWC": {"material"},
    "Detector_EIS": {"material", "fano_factor", "pair_energy_300k", "pair_energy_slope"},
    "Telescope_EUVST": {"psf_type", "filter"},
    "Telescope_EIS": {"psf_type", "psf_slit_width", "psf_across_slit", "date"},
}


def _sourced(cls):
    return {f.name for f in dataclasses.fields(cls) if "source" in f.metadata}


@pytest.mark.parametrize("cls", INSTRUMENT_CLASSES, ids=lambda cls: cls.__name__)
def test_every_instrument_setting_has_a_source(cls):
    """A new setting needs a source in its field metadata, shown on the sources page."""
    settings = {f.name for f in dataclasses.fields(cls) if f.init}
    assert settings - _sourced(cls) == WITHOUT_SOURCE[cls.__name__]
    for f in dataclasses.fields(cls):
        if f.name in _sourced(cls):
            assert isinstance(f.metadata["source"], str) and f.metadata["source"]


def test_page_shows_every_setting_with_a_source():
    """Each setting with a source has a row for its instrument, and no placeholder is left."""
    markdown = PAGE.read_text()
    shown = set()
    for instrument, keys in hook.PLACEHOLDER.findall(markdown):
        for key in keys.split(","):
            section, _, name = key.strip().partition(".")
            shown.add((hook.INSTRUMENTS[instrument][section], name))
    for cls in INSTRUMENT_CLASSES + (Simulation,):
        missing = {name for name in _sourced(cls) if (cls, name) not in shown}
        assert not missing, f"{cls.__name__} settings missing from {PAGE.name}: {missing}"

    rendered = hook.render(markdown)
    assert "<!-- setting" not in rendered
    for cls, name in shown:
        section = next(s for table in hook.INSTRUMENTS.values()
                       for s, c in table.items() if c is cls)
        assert f"`{section}.{name}`" in rendered


@pytest.mark.parametrize("instrument, keys, row", [
    ("SWC", ["detector.pix_size", "detector.n_rows", "detector.n_columns"],
     "13.5 micron, 2048 and 2048"),
    ("SWC", ["detector.wvl_res"], "16.9 mA per pixel"),
    ("SWC", ["detector.ccd_temperature"], "-60 Celsius"),
    ("SWC", ["detector.read_noise_rms"], "10 electrons"),
    ("SWC", ["detector.dark_current_293k"], "20000 electrons per pixel per second"),
    ("SWC", ["filter.al_thickness", "filter.oxide_thickness", "filter.c_thickness"],
     "1485, 95 and 0 Angstrom"),
    ("SWC", ["detector.serial_prescan", "detector.serial_overscan"],
     "50 and 20; each settable from 0 to 200"),
    ("SWC", ["detector.cte_parallel", "detector.cte_serial"], "1, a perfect transfer"),
    ("SWC", ["detector.pair_energy_300k", "detector.pair_energy_slope"],
     "3.71 eV and -0.0006 eV per K, for silicon"),
    ("SWC", ["detector.shutter"], "true"),
    ("SWC", ["detector.lit_band"], "170 to 212.3 Angstrom"),
    ("EIS", ["telescope.psf_params"], "3 and 3 pixels FWHM"),
    ("EIS", ["telescope.calibration"], "`ground`"),
])
def test_defaults_in_their_display_units(instrument, keys, row):
    """Stored units are converted for the page, e.g. pix_size from cm per pixel to micron."""
    assert hook.setting_row(instrument, keys).split(" | ")[1] == row


def test_row_refuses_unknown_settings_and_mixed_sources():
    with pytest.raises(ValueError, match="not an EIS setting"):
        hook.setting_row("EIS", ["detector.shutter"])
    with pytest.raises(ValueError, match="different sources"):
        hook.setting_row("SWC", ["detector.qe_euv", "detector.max_dn"])
    with pytest.raises(ValueError, match="Unknown instrument"):
        hook.setting_row("SW", ["detector.qe_euv"])
    with pytest.raises(ValueError, match="do not know"):
        hook._unit("arcseconds")


def test_row_escapes_pipes(monkeypatch):
    """A `|` in a source would otherwise start a new table column."""
    f = next(f for f in dataclasses.fields(Detector_SWC) if f.name == "max_dn")
    monkeypatch.setattr(f, "metadata", {**f.metadata, "source": "Table 6-1 | 6-2"})
    assert hook.setting_row("SWC", ["detector.max_dn"]).endswith("| Table 6-1 \\| 6-2 |")
