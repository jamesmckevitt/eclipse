"""Every instrument setting has a source, and docs/instrument-sources.md shows the code's."""
import dataclasses
import importlib.util
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from euvst_response.config import (AluminiumFilter, Detector_EIS, Detector_SWC, Simulation,
                                   Telescope_EIS, Telescope_EUVST)

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "update_instrument_sources", ROOT / "tools" / "update_instrument_sources.py")
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

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


def test_page_is_up_to_date():
    page = tool.PAGE.read_text()
    assert tool.updated_page(page) == page, (
        "docs/instrument-sources.md differs from the defaults and sources in the code; run "
        "python tools/update_instrument_sources.py")


def test_page_has_a_row_for_every_setting_with_a_source():
    shown = set()
    for _, instrument, _, keys in tool.setting_rows(tool.PAGE.read_text().split("\n")):
        for key in keys:
            section, _, name = key.partition(".")
            shown.add((tool.INSTRUMENTS[instrument][section], name))
    for cls in INSTRUMENT_CLASSES + (Simulation,):
        missing = {name for name in _sourced(cls) if (cls, name) not in shown}
        assert not missing, f"{cls.__name__} settings missing from the page: {missing}"


def test_hand_written_rows_are_kept(monkeypatch):
    monkeypatch.setattr(tool, "WORDING", {"SWC": {}, "EIS": {}})
    page = ("<!-- SWC settings: written by the script -->\n\n"
            "| Setting | Default | Source |\n| --- | --- | --- |\n"
            "| `detector.max_dn` | 1 DN | old |\n"
            "| Something set in the code | 0.5 | Not recorded |\n\n"
            "| `detector.max_dn` | outside a marked table | kept |\n")
    lines = tool.updated_page(page).split("\n")
    assert lines[4] == ("| `detector.max_dn` | 65535 DN | SOLC-EUVST-MSSL-RP-0012 v2.0, "
                        "Table 5-5: a 16-bit converter |")
    assert lines[5] == "| Something set in the code | 0.5 | Not recorded |"
    assert lines[7] == "| `detector.max_dn` | outside a marked table | kept |"


@pytest.mark.parametrize("instrument, keys, default", [
    ("SWC", ["detector.pix_size", "detector.n_rows", "detector.n_columns"],
     "13.5 micron, 2048 by 2048"),
    ("SWC", ["detector.wvl_res"], "16.9 mA per pixel"),
    ("SWC", ["detector.ccd_temperature"], "-60 Celsius"),
    ("SWC", ["detector.read_noise_rms"], "10 electrons"),
    ("SWC", ["detector.dark_current_293k"], "20000 electrons per pixel per second"),
    ("SWC", ["filter.al_thickness", "filter.oxide_thickness", "filter.c_thickness"],
     "1485, 95 and 0 Angstrom"),
    ("SWC", ["telescope.fiducial_centres", "telescope.fiducial_lengths"],
     "-25 and +70 arcsec; 0.8 arcsec, 1.6 arcsec for the widest slit"),
    ("SWC", ["detector.lit_band"], "170 to 212.3 Angstrom"),
    ("SWC", ["detector.slit_image_tilt"], "none; measured as a drift along the dispersion of "
     "0.991 f + 0.605 f^2 rows, where f is the angle along the slit over 140 arcsec"),
    ("SWC", ["telescope.pointing_jitter"], "none: the budget's 0.110 arcsec, or no jitter with "
     "`resolution_budget: none`"),
    ("EIS", ["telescope.psf_params"], "3 pixels FWHM"),
    ("EIS", ["detector.wvl_res", "detector.plate_scale_angle"],
     "22.3 mA per pixel, 1 arcsec per pixel"),
])
def test_defaults_in_their_display_units(instrument, keys, default):
    """Stored units are converted for the page, e.g. `pix_size` from cm per pixel to micron."""
    assert tool.cells(instrument, keys)[0] == default


def test_numbers_have_no_exponent():
    assert format(tool.Number(1e6)) == "1000000"
    assert format(tool.Number(2e-7)) == "0.0000002"
    assert format(tool.Number((13.5 * u.um).cgs.to_value(u.um))) == "13.5"


def test_refuses_bad_settings_units_and_rows(monkeypatch):
    with pytest.raises(ValueError, match="not an EIS setting"):
        tool.cells("EIS", ["detector.shutter"])
    with pytest.raises(ValueError, match="not an EIS setting"):
        tool.cells("EIS", ["simulation.slit_width"])
    with pytest.raises(ValueError, match="different sources"):
        tool.cells("SWC", ["detector.qe_euv", "detector.max_dn"])
    with pytest.raises(ValueError, match="Unknown instrument"):
        tool.cells("SW", ["detector.qe_euv"])
    with pytest.raises(ValueError, match="does not know"):
        tool.Value("x", 1 * u.arcsec, "arcseconds")
    with pytest.raises(ValueError, match="x is a quantity"):
        tool.Value("x", 1 * u.arcsec)
    with pytest.raises(ValueError, match="x is a plain number"):
        tool.Value("x", 0.5, "arcsec")
    with pytest.raises(ValueError, match="needs two values"):
        format(tool.Value("x", (1.0, 2.0, 3.0)), "range")
    with pytest.raises(ValueError, match="leaves out pix_size"):
        monkeypatch.setitem(tool.WORDING["SWC"], "detector.pix_size", "{n_rows} by {n_columns}")
        tool.cells("SWC", ["detector.pix_size", "detector.n_rows", "detector.n_columns"])
    f = next(f for f in dataclasses.fields(Detector_SWC) if f.name == "max_dn")
    monkeypatch.setattr(f, "metadata", {**f.metadata, "source": "Table 6-1 | 6-2"})
    with pytest.raises(ValueError, match="split its table cell"):
        tool.cells("SWC", ["detector.max_dn"])


def test_euvst_psf_design_values_match_the_code():
    """The design FWHMs, in pixels rounded to 0.01 as the code holds them."""
    fields = Telescope_EUVST.FIELDS_OF_VIEW
    plate_scale = Detector_SWC.plate_scale_angle.to_value(u.arcsec / u.pix)
    spectral_step = Detector_SWC.wvl_res.to_value(u.mAA / u.pix)
    for field, arcsec in (("280x280", 0.423), ("100x100", 0.195), ("70x70", 0.161)):
        assert round(arcsec / plate_scale, 2) == fields[field][0].to_value(u.pix)
    for field in fields:
        assert round(43.00 / spectral_step, 2) == fields[field][1].to_value(u.pix)
    assert "0.423" in tool.EUVST_PSF and "0.195 and 0.161" in tool.EUVST_PSF
    assert "43.00 mA" in tool.EUVST_PSF


def test_wording_matches_the_code():
    """Facts the page's wording states, beyond the values it takes from the code."""
    # "each settable from 0 to 200"
    for name in ("serial_prescan", "serial_overscan"):
        Detector_SWC(**{name: 0})
        Detector_SWC(**{name: 200})
        for bad in (-1, 201):
            with pytest.raises(ValueError):
                Detector_SWC(**{name: bad})
    # "a perfect transfer", one value for both
    assert Detector_SWC.cte_parallel == Detector_SWC.cte_serial == 1
    # "0.8 arcsec, 1.6 arcsec for the widest slit", two fiducials
    telescope = Telescope_EUVST()
    assert len(telescope.fiducial_centres) == 2
    assert len(set(telescope.fiducial_lengths[:3])) == 1 and len(telescope.fiducial_lengths) == 4
    # "the slits are 0.2, 0.4, 0.8 and 1.6 arcsec wide"
    for width in (0.2, 0.4, 0.8, 1.6):
        Simulation(slit_width=width * u.arcsec)
    with pytest.raises(ValueError):
        Simulation(slit_width=0.3 * u.arcsec)


def test_every_wording_has_a_row():
    page = tool.PAGE.read_text()
    assert tool.unused_wording(page) == []
    reordered = page.replace("`detector.pix_size`, `detector.n_rows`",
                             "`detector.n_rows`, `detector.pix_size`")
    assert tool.updated_page(reordered) == reordered
    with pytest.raises(ValueError, match="SWC detector.lit_band"):
        tool.updated_page(page.replace("`detector.lit_band`", "`detector.ccd_gap`"))


def test_wording_names_only_its_row(monkeypatch):
    monkeypatch.setitem(tool.WORDING["SWC"], "detector.pix_size",
                        "{pix_size}, {n_rows} by {n_columns}")
    with pytest.raises(ValueError, match="n_columns, n_rows, which are not in its row"):
        tool.cells("SWC", ["detector.pix_size"])


def test_hand_written_rows_match_the_code():
    """The dark current laws and the aperture share, written by hand on the page."""
    page = tool.PAGE.read_text()
    assert ("the 293 K value times 122 T^3 exp(-6400/T), held at its 198 K value below "
            "198 K") in page
    assert ("the 293 K value times 1.14e6 T^3 exp(-9080/T), held at its 198 K value below "
            "198 K") in page
    for cls, law in ((Detector_SWC, lambda t: 122 * t**3 * np.exp(-6400 / t)),
                     (Detector_EIS, lambda t: 1.14e6 * t**3 * np.exp(-9080 / t))):
        rate = cls.dark_current_293k
        for kelvin, used in ((250.0, 250.0), (198.0, 198.0), (180.0, 198.0)):
            got = cls.calculate_dark_current(kelvin * u.K, rate)
            assert got.value == pytest.approx(rate.value * law(used), rel=1e-12)
    assert "| Half the aperture feeding the SW channel, set in the code | 0.5 |" in page
    telescope = Telescope_EUVST()
    assert telescope.collecting_area.to_value(u.m**2) == pytest.approx(
        0.5 * np.pi * (telescope.D_ap.to_value(u.m) / 2) ** 2)
