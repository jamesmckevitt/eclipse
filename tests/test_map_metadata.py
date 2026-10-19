"""Maps carry an observation date and an observer position.

Without them, sunpy stamps maps with the time the code ran and assumes an
Earth-based observer.
"""
import warnings

from datetime import date as date_type

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
import sunpy
from astropy.wcs import WCS
from ndcube import NDCube
from sunpy.util.exceptions import SunpyMetadataWarning

from euvst_response.analysis import create_sunpy_maps_from_combo

REST = 195.119 * u.Angstrom
NX, NY, NWAVE = 3, 5, 4
WHEN = "2024-03-20T09:30:00"


def _combination_results(telescope=None):
    """Minimal results dict in the form create_sunpy_maps_from_combo expects."""
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLN-TAN", "HPLT-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [0.02, 0.2, 0.16]
    wcs.wcs.crpix = [NWAVE / 2.0, NX / 2.0, NY / 2.0]
    wcs.wcs.crval = [REST.to_value(u.Angstrom), 0.0, 0.0]

    data = np.ones((NY, NX, NWAVE))
    fits = np.zeros((NY, NX, 4))
    fits[..., 0] = 1.0
    fits[..., 1] = REST.to_value(u.Angstrom)
    fits[..., 2] = 0.06
    units = [u.DN / u.pix, u.Angstrom, u.Angstrom, u.DN / u.pix]

    combo = {
        "first_signal_wcs": wcs,
        "first_photon_signal": NDCube(data, wcs=wcs, unit=u.photon / u.pix),
        "first_dn_signal": NDCube(data, wcs=wcs, unit=u.DN / u.pix),
        "dn_fit_stats": {"first_fit_data": fits, "mean_data": fits,
                         "std_data": np.zeros_like(fits), "units": units},
        "ground_truth": {"fit_truth_data": fits, "fit_truth_units": units},
    }
    if telescope is not None:
        combo["config_objects"] = {"telescope": telescope}
    return combo


def _make(**kwargs):
    return create_sunpy_maps_from_combo(
        _combination_results(kwargs.pop("telescope", None)),
        rest_wavelength=REST, data_type="dn", **kwargs)


def test_every_map_carries_the_date_it_was_given():
    maps = _make(date_obs=WHEN)
    assert maps  # the loop below is worthless if this is empty
    for name, map_obj in maps.items():
        assert map_obj.meta["date-obs"].startswith("2024-03-20"), name


def test_every_map_states_the_observer():
    """On the disc-centre line, one au from the emitting surface, as the chain assumes."""
    maps = _make(date_obs=WHEN)
    for name, map_obj in maps.items():
        assert map_obj.meta["dsun_obs"] == pytest.approx(
            (const.au + const.R_sun).to_value(u.m)), name
        assert map_obj.meta["hgln_obs"] == 0.0, name
        assert map_obj.meta["hglt_obs"] == 0.0, name


def test_the_observer_survives_into_the_coordinate_frame():
    """The keywords are only worth writing if sunpy reads them back."""
    map_obj = _make(date_obs=WHEN)["total_dn"]
    observer = map_obj.observer_coordinate
    assert observer.radius.to_value(u.au) == pytest.approx(
        1.0 + (const.R_sun / const.au).decompose().value, rel=1e-9)
    assert observer.lat.to_value(u.deg) == pytest.approx(0.0, abs=1e-9)


def test_no_sunpy_metadata_warnings_are_raised():
    """The warnings are the visible symptom of missing metadata."""
    maps = _make(date_obs=WHEN)
    with warnings.catch_warnings(), sunpy.log.log_to_list() as messages:
        warnings.simplefilter("error", SunpyMetadataWarning)
        for map_obj in maps.values():
            # sunpy only checks the date, observer and solar radius when they
            # are read, as in plotting, so building the maps alone would pass
            # even without them.
            map_obj.wcs
    assert [m.getMessage() for m in messages] == []


def test_a_date_object_is_accepted():
    maps = _make(date_obs=date_type(2024, 3, 20))
    assert maps["total_dn"].meta["date-obs"].startswith("2024-03-20")


def test_an_eis_calibration_date_is_used_when_no_date_is_given():
    """A time-dependent EIS calibration carries a real observing date."""
    from euvst_response.config import Telescope_EIS

    telescope = Telescope_EIS(calibration="dz2025", date="2012-06-03")
    maps = _make(telescope=telescope)
    assert maps["total_dn"].meta["date-obs"].startswith("2012-06-03")


def test_an_explicit_date_beats_the_calibration_date():
    from euvst_response.config import Telescope_EIS

    telescope = Telescope_EIS(calibration="dz2025", date="2012-06-03")
    maps = _make(telescope=telescope, date_obs=WHEN)
    assert maps["total_dn"].meta["date-obs"].startswith("2024-03-20")


def test_no_date_anywhere_raises_rather_than_inventing_one():
    with pytest.raises(ValueError, match="no observation date"):
        _make()


def test_units_are_still_attached():
    """The header rewrite must keep the map units."""
    maps = _make(date_obs=WHEN)
    assert maps["total_dn"].meta["bunit"] == "DN"
    assert maps["velocity_mean"].meta["bunit"] == "km / s"
    assert maps["line_width_mean"].meta["bunit"] == "Angstrom"


def test_the_exposure_time_map_holds_the_exposure_times_in_seconds():
    """The map holds exposure times, not each exposure's index (0, 1, 2) labelled as seconds."""
    runs = []
    # Pixel (0, 0) meets 2 km/s from 1 s, (0, 1) from 2 s, (0, 2) from 5 s,
    # and the rest not at all.
    for expos, meeting in ((1.0, 1), (2.0, 2), (5.0, 3)):
        combo = _combination_results()
        std = np.zeros((NY, NX, 4))
        std[..., 1] = 1e-2  # Angstrom: some 15 km/s
        std[0, :meeting, 1] = 1e-4
        combo["dn_fit_stats"]["std_data"] = std
        combo["parameters"] = {"simulation.expos": expos * u.s}
        runs.append(combo)
    exposure = _make(date_obs=WHEN, exposure_time_results=runs)["exposure_time"]
    assert exposure.meta["bunit"] == "s"
    finite = exposure.data[np.isfinite(exposure.data)]
    assert sorted(finite.tolist()) == [1.0, 2.0, 5.0]
    assert np.isinf(exposure.data).sum() == NX * NY - 3
