"""Maps carry an observation date and a stated vantage point.

Without them sunpy stamps every map with the time the code happened to run
and assumes an Earth-based observer, both silently. The date one is the
serious one: it makes two runs of the same analysis produce different maps.
"""
import warnings

from datetime import date as date_type

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS
from ndcube import NDCube
from sunpy.util.exceptions import SunpyMetadataWarning

from euvst_response.analysis import create_sunpy_maps_from_combo

REST = 195.119 * u.Angstrom
NX, NY, NWAVE = 3, 5, 4
WHEN = "2024-03-20T09:30:00"


def _combination_results(telescope=None):
    """A minimal results dict of the shape create_sunpy_maps_from_combo wants."""
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["WAVE", "HPLT-TAN", "HPLN-TAN"]
    wcs.wcs.cunit = ["Angstrom", "arcsec", "arcsec"]
    wcs.wcs.cdelt = [0.02, 0.16, 0.2]
    wcs.wcs.crpix = [NWAVE / 2.0, NY / 2.0, NX / 2.0]
    wcs.wcs.crval = [REST.to_value(u.Angstrom), 0.0, 0.0]

    data = np.ones((NX, NY, NWAVE))
    fits = np.zeros((NX, NY, 4))
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
    """One au on the disc-centre line, which is what the chain assumes."""
    maps = _make(date_obs=WHEN)
    for name, map_obj in maps.items():
        assert map_obj.meta["dsun_obs"] == pytest.approx(
            const.au.to_value(u.m)), name
        assert map_obj.meta["hgln_obs"] == 0.0, name
        assert map_obj.meta["hglt_obs"] == 0.0, name


def test_the_observer_survives_into_the_coordinate_frame():
    """The keywords are only worth writing if sunpy reads them back."""
    map_obj = _make(date_obs=WHEN)["total_dn"]
    observer = map_obj.observer_coordinate
    assert observer.radius.to_value(u.au) == pytest.approx(1.0, rel=1e-9)
    assert observer.lat.to_value(u.deg) == pytest.approx(0.0, abs=1e-9)


def test_no_sunpy_metadata_warnings_are_raised():
    """The warnings were the visible symptom, so they are worth asserting on."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", SunpyMetadataWarning)
        _make(date_obs=WHEN)


def test_two_runs_produce_the_same_date():
    """The point of the change: the maps stop depending on the clock."""
    first = _make(date_obs=WHEN)["total_dn"].meta["date-obs"]
    second = _make(date_obs=WHEN)["total_dn"].meta["date-obs"]
    assert first == second


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
    """The header rewrite must not lose what the maps are measured in."""
    maps = _make(date_obs=WHEN)
    assert maps["total_dn"].meta["bunit"] == "DN"
    assert maps["velocity_mean"].meta["bunit"] == "km / s"
    assert maps["line_width_mean"].meta["bunit"] == "Angstrom"


def test_the_wcs_still_describes_the_same_pixels():
    """Going through a FITS header must not disturb the plate scale."""
    maps = _make(date_obs=WHEN)
    map_obj = maps["total_dn"]
    assert map_obj.data.shape == (NY, NX)
    assert map_obj.scale[0].to_value(u.arcsec / u.pix) == pytest.approx(0.2)
    assert map_obj.scale[1].to_value(u.arcsec / u.pix) == pytest.approx(0.16)
