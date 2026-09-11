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
