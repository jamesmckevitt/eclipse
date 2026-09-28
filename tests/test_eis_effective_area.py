"""
Tests for the Hinode/EIS effective area.

The numerical anchors are IDL: eis_ea.pro on the pre-flight tables returns
0.301803 cm^2 at 195 Angstrom and 0.0413185 cm^2 at 284 Angstrom. Those two
numbers pin down three things at once - that the short- and long-wavelength
tables are assigned to the right channels, that the interpolation convention
is a natural cubic spline through log(EA) rather than a linear interpolation,
and that the quantum efficiency is handled the way the tables assume.

Everything else here is structural: that the epoch changes the answer, that
the two bands stay separate, and that the quantum efficiency is divided out
exactly once on the way into ECLIPSE's radiometric chain.
"""

import numpy as np
import pytest
import astropy.units as u

from euvst_response import eis_calibration
from euvst_response.config import Detector_EIS, Telescope_EIS


# IDL eis_ea.pro, pre-flight tables.
IDL_GROUND_EA = {195.0: 0.301803, 284.0: 0.0413185}


@pytest.mark.parametrize("wavelength,expected", sorted(IDL_GROUND_EA.items()))
def test_ground_matches_idl(wavelength, expected):
    area = eis_calibration.effective_area(wavelength, method="ground")
    assert area.shape == (1,)
    assert area[0] == pytest.approx(expected, rel=1e-5)


def test_ground_needs_no_date():
    """The pre-flight calibration has no epoch, so it must not demand one."""
    with_date = eis_calibration.effective_area(
        195.0, date="2012-06-03", method="ground")
    without = eis_calibration.effective_area(195.0, method="ground")
    assert with_date[0] == without[0]


@pytest.mark.parametrize("method", eis_calibration.TIME_DEPENDENT_CALIBRATIONS)
def test_time_dependent_calibrations_require_a_date(method):
    with pytest.raises(ValueError, match="time-dependent"):
        eis_calibration.effective_area(195.0, method=method)


@pytest.mark.parametrize("method", eis_calibration.TIME_DEPENDENT_CALIBRATIONS)
def test_epoch_changes_the_answer(method):
    """The whole point of an in-flight calibration is that the area moves.

    Asserted on the long-wavelength channel, which every published
    calibration agrees decayed substantially; the direction is deliberately
    not asserted for the short-wavelength channel, where the in-flight
    calibrations revise the pre-flight peak upwards rather than downwards.
    """
    early = eis_calibration.effective_area(284.0, date="2008-01-01", method=method)
    late = eis_calibration.effective_area(284.0, date="2018-01-01", method=method)
    assert late[0] < early[0]


def test_area_varies_strongly_across_a_band():
    """A flat effective area is the thing this module exists to replace."""
    wavelengths = np.array([181.9, 188.2, 195.119, 202.0, 211.3])
    area = eis_calibration.effective_area(wavelengths, method="ground")
    assert np.all(np.isfinite(area))
    assert area.max() / area.min() > 10.0


def test_no_area_between_the_bands():
    wavelengths = np.array([195.119, 225.0, 284.16])
    area = eis_calibration.effective_area(wavelengths, method="ground")
    assert np.isfinite(area[0])
    assert np.isnan(area[1])
    assert np.isfinite(area[2])


def test_both_bands_in_one_call():
    """Each band must use its own table, whatever order they arrive in."""
    together = eis_calibration.effective_area(
        np.array([284.0, 195.0]), method="ground")
    separately = [eis_calibration.effective_area(w, method="ground")[0]
                  for w in (284.0, 195.0)]
    assert together == pytest.approx(separately, rel=1e-12)


def test_unknown_calibration_is_rejected():
    with pytest.raises(ValueError, match="Unknown EIS calibration"):
        eis_calibration.effective_area(195.0, method="dz2099")


def test_offset_dates_are_converted_not_relabelled():
    """An ISO offset must shift the instant, not just the label.

    replace(tzinfo=utc) would keep the wall clock, so a -05:00 date would
    enter the degradation calculation five hours early.
    """
    utc = eis_calibration._parse_date("2012-06-03T00:00:00+00:00")
    minus5 = eis_calibration._parse_date("2012-06-03T00:00:00-05:00")
    assert (minus5 - utc).total_seconds() == 5 * 3600


@pytest.mark.parametrize("date", ["2012-06-03", "2012-06-03T00:00:00"])
def test_date_forms_agree(date):
    from datetime import date as date_type

    as_string = eis_calibration.effective_area(195.0, date=date, method="dz2025")
    as_object = eis_calibration.effective_area(
        195.0, date=date_type(2012, 6, 3), method="dz2025")
    assert as_string[0] == pytest.approx(as_object[0], rel=1e-12)


class TestTelescope:

    def test_default_is_the_preflight_curve(self):
        tel = Telescope_EIS()
        assert tel.calibration == "ground"
        area = tel.effective_area(195.0 * u.AA)
        assert area.unit.is_equivalent(u.cm**2)
        assert area.to_value(u.cm**2) == pytest.approx(
            IDL_GROUND_EA[195.0], rel=1e-5)

    def test_quantum_efficiency_is_divided_out_once(self):
        """The tables include QE; ECLIPSE applies it separately downstream."""
        tel = Telescope_EIS()
        area = tel.effective_area(195.0 * u.AA)
        throughput = tel.ea_and_throughput(195.0 * u.AA)
        assert throughput.to_value(u.cm**2) == pytest.approx(
            area.to_value(u.cm**2) / eis_calibration.QE_IN_TABLES, rel=1e-12)

    def test_table_qe_is_not_a_telescope_field(self):
        """The QE the tables are quoted against is a calibration constant.

        Exposing it on the telescope would let a caller divide by one value
        while ``to_electrons`` applied ``Detector_EIS.qe_euv``, silently
        scaling the whole response.
        """
        assert not hasattr(Telescope_EIS(), "qe_euv")
        with pytest.raises(TypeError):
            Telescope_EIS(qe_euv=0.5)

    def test_time_dependent_calibration_needs_a_date(self):
        with pytest.raises(ValueError, match="time-dependent"):
            Telescope_EIS(calibration="dz2025")

    def test_unknown_calibration_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown EIS calibration"):
            Telescope_EIS(calibration="dz2099", date="2012-06-03")

    def test_yaml_style_date_is_accepted(self):
        """YAML turns an unquoted 2012-06-03 into a datetime.date."""
        from datetime import date as date_type

        tel = Telescope_EIS(calibration="dz2025", date=date_type(2012, 6, 3))
        assert tel.date == "2012-06-03"
        assert np.isfinite(tel.effective_area(195.0 * u.AA).to_value(u.cm**2))

    def test_out_of_band_wavelength_raises(self):
        """Silently returning NaN would show up as an unexplained zero later."""
        tel = Telescope_EIS()
        with pytest.raises(ValueError, match="no effective area"):
            tel.ea_and_throughput(225.0 * u.AA)

    def test_scalar_in_scalar_out(self):
        """radiometric.add_telescope_throughput feeds one wavelength at a time."""
        tel = Telescope_EIS()
        assert tel.ea_and_throughput(195.0 * u.AA).isscalar
        assert not tel.ea_and_throughput([195.0, 196.0] * u.AA).isscalar


# Areas from SolarSoft itself, IDL 8.8: eis_ea (ground), eis_ea_gdz, which is
# eis_ea over eis_ltds's correction (dz2013), eis_ea_nrl (warren2014) and
# interpol_eis_ea with the bundled smooth fit (dz2025). Between the tables'
# nodes, where each routine's own interpolation decides the value: a linear
# one in place of eis_ltds's splines put dz2013 over twice too bright near
# 168 A. IDL works in single precision, which the tolerance allows for.
IDL_BETWEEN_NODES = [
    ("ground", None, 172.5, 8.67057301e-04),
    ("ground", None, 203.75, 4.46710475e-02),
    ("ground", None, 212.5, 8.83354433e-03),
    ("ground", None, 268.25, 1.07728504e-01),
    ("dz2013", "2010-01-01T00:00:00", 172.5, 3.56071861e-04),
    ("dz2013", "2010-01-01T00:00:00", 203.75, 4.39554863e-02),
    ("dz2013", "2010-01-01T00:00:00", 268.25, 6.07873648e-02),
    ("warren2014", "2015-06-03T00:00:00", 172.5, 7.00180070e-04),
    ("warren2014", "2015-06-03T00:00:00", 203.75, 8.20269734e-02),
    ("warren2014", "2015-06-03T00:00:00", 212.5, 2.37968583e-02),
    ("warren2014", "2015-06-03T00:00:00", 268.25, 6.13164417e-02),
    ("dz2025", "2020-01-01T00:00:00", 172.5, 2.16257467e-04),
    ("dz2025", "2020-01-01T00:00:00", 203.75, 3.23750749e-02),
    ("dz2025", "2020-01-01T00:00:00", 212.5, 8.24429933e-03),
    ("dz2025", "2020-01-01T00:00:00", 268.25, 3.77405398e-02),
]


@pytest.mark.parametrize("method, date, wavelength, expected", IDL_BETWEEN_NODES)
def test_areas_between_the_nodes_match_solarsoft(method, date, wavelength, expected):
    area = eis_calibration.effective_area(wavelength, date, method)[0]
    assert area == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("method, date, message", [
    ("dz2013", "2006-01-01", "before 22 September 2006.*ground calibration is used"),
    ("dz2013", "2020-01-01", "after 14 September 2012.*decay is held"),
    ("dz2025", "2024-01-01", "outside 2007-04-01.*nearer end of the fit"),
])
def test_a_date_beyond_a_fitted_range_is_said_to_be(method, date, message):
    """SolarSoft prints the same; the value is SolarSoft's too."""
    eis_calibration._interpolator.cache_clear()
    with pytest.warns(UserWarning, match=message):
        area = eis_calibration.effective_area(268.25, date, method)[0]
    if method == "dz2013" and date.startswith("2006"):
        assert area == pytest.approx(eis_calibration.effective_area(268.25)[0], rel=1e-12)
    if method == "dz2013" and date.startswith("2020"):
        assert area == pytest.approx(4.62971665e-02, rel=1e-4)
    if method == "dz2025":
        assert eis_calibration.effective_area(203.75, date, method)[0] == pytest.approx(
            2.80065313e-02, rel=1e-4)
