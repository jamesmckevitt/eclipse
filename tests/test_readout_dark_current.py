"""Dark current collected during the clear and the read-out, in runs."""
import dataclasses

import astropy.units as u
import numpy as np
import pytest

from eclipse.config import Detector_EIS, Detector_SWC, Simulation, Telescope_EIS, Telescope_EUVST
from eclipse.data_processing import create_uniform_intensity_cube
from eclipse.monte_carlo import _dark_time, expected_dn_uncertainty, simulate_once
from eclipse.radiometric import dn_variance
from eclipse.readout import (FocalPlane_SWC, ReadoutSequence, dark_current_time,
                             windows_from_wavelengths)

DET = Detector_SWC()
TEL = Telescope_EUVST()
INTENSITY = 1000 * u.erg / (u.s * u.cm**2 * u.sr)
EXPOSURE = 10 * u.s


def _cube(rest, det=DET, tel=TEL, sim=None):
    sim = sim or Simulation(slit_width=0.2 * u.arcsec, noise=False)
    return create_uniform_intensity_cube(INTENSITY, rest, 20 * u.km / u.s, det, sim, tel=tel)


FOCAL_PLANE = FocalPlane_SWC.from_detector(DET)


def _per_row(windows):
    """Dark time of each image row, with `windows` read and the other rows dumped."""
    n_rows = FOCAL_PLANE.n_rows
    sequence = ReadoutSequence.from_detector(DET, windows=windows)
    return dark_current_time(EXPOSURE, sequence, n_rows)[:n_rows]


def _window(cube):
    """Rows the cube's wavelengths reach, as a read-out window."""
    wavelength = cube.axis_world_coords_values(2)[0]
    half = 0.5 * (wavelength[1] - wavelength[0])
    return windows_from_wavelengths(FOCAL_PLANE, [(wavelength[0] - half, wavelength[-1] + half)])


@pytest.mark.parametrize("ccd, row", [("left", 0), ("left", 1500), ("left", 2047),
                                      ("right", 0), ("right", 900)])
def test_a_pixel_takes_the_dark_time_of_the_row_recording_its_wavelength(ccd, row):
    """Only the rows of the cube's own wavelengths are read."""
    cube = _cube(FOCAL_PLANE.wavelength(row, ccd))
    time = _dark_time(cube, EXPOSURE, DET, Simulation(slit_width=0.2 * u.arcsec, noise=False))
    middle = cube.data.shape[-1] // 2
    expected = _per_row(_window(cube))[row]
    assert time[middle].to_value(u.s) == pytest.approx(expected.to_value(u.s), rel=1e-7)
    assert np.all(time > EXPOSURE)


def test_reading_only_the_window_waits_far_less_than_reading_every_row():
    """
    Read alone, the window of a line adds a few hundredths of a second with
    the default timing, wherever it is. With every row read, the last row
    would wait 1.15 s.
    """
    assert (_per_row([])[-1] - EXPOSURE).to_value(u.s) == pytest.approx(1.15, abs=0.01)
    for wavelength in (172 * u.AA, 197 * u.AA, 205 * u.AA):
        extra = (_dark_time(_cube(wavelength), EXPOSURE, DET, Simulation(noise=False))
                 - EXPOSURE).to_value(u.s)
        assert np.all((extra > 0.02) & (extra < 0.1)), wavelength


def test_a_wavelength_in_the_gap_takes_the_last_row():
    time = _dark_time(_cube(198.9 * u.AA), EXPOSURE, DET, Simulation(noise=False))
    expected = _per_row([(FOCAL_PLANE.n_rows - 1, FOCAL_PLANE.n_rows - 1)])[-1]
    assert np.allclose(time.to_value(u.s), expected.to_value(u.s), rtol=1e-12, atol=0.0)


def test_without_readout_dark_current_or_for_eis_it_is_the_exposure():
    off = Simulation(noise=False, readout_dark_current=False)
    assert _dark_time(_cube(195.119 * u.AA), EXPOSURE, DET, off) == EXPOSURE
    eis = Simulation(slit_width=1 * u.arcsec, instrument="EIS", noise=False)
    cube = _cube(195.119 * u.AA, Detector_EIS(), Telescope_EIS(), eis)
    assert _dark_time(cube, EXPOSURE, Detector_EIS(), eis) == EXPOSURE


def test_a_run_adds_the_dark_current_of_the_extra_time():
    """Without noise, the electrons differ by the dark current over the extra time alone."""
    on = Simulation(slit_width=0.2 * u.arcsec, noise=False)
    off = dataclasses.replace(on, readout_dark_current=False)
    cube = _cube(195.119 * u.AA, sim=on)
    with_readout = simulate_once(cube, EXPOSURE, DET, TEL, on, uniform_mode=True)[6]
    exposure_only = simulate_once(cube, EXPOSURE, DET, TEL, off, uniform_mode=True)[6]
    extra = (DET.dark_current * (_dark_time(cube, EXPOSURE, DET, on) - EXPOSURE)).to_value(
        u.electron / u.pixel)
    assert np.allclose(with_readout.data - exposure_only.data, extra, rtol=1e-10, atol=1e-10)
    assert np.all(extra > 0)


def test_the_expected_uncertainty_counts_the_same_dark_current():
    on = Simulation(slit_width=0.2 * u.arcsec, noise=False)
    off = dataclasses.replace(on, readout_dark_current=False)
    cube = _cube(195.119 * u.AA, sim=on)
    gain = DET.gain_e_per_dn.to_value(u.electron / u.DN)
    extra = (DET.dark_current * (_dark_time(cube, EXPOSURE, DET, on) - EXPOSURE)).to_value(
        u.electron / u.pixel)
    variance = {name: expected_dn_uncertainty(cube, EXPOSURE, DET, TEL, sim)[0, 0] ** 2
                for name, sim in (("on", on), ("off", off))}
    # The extra dark current adds its Poisson variance, and its mean is not
    # taken for EUV, so the EUV variance is unchanged.
    assert np.allclose(variance["on"] - variance["off"], extra / gain**2, rtol=1e-6, atol=0.0)


def test_dn_variance_takes_a_dark_time_per_pixel():
    dn = np.full(3, 100.0)
    times = np.array([10.0, 10.5, 11.0]) * u.s
    each = [dn_variance(dn[:1], 195.119 * u.AA, t, DET)[0] for t in times]
    together = dn_variance(dn, 195.119 * u.AA, EXPOSURE, DET, dark_time=times)
    assert np.allclose(together, each, rtol=1e-12, atol=0.0)


def test_charge_transfer_is_left_out_with_a_warning():
    leaky = Detector_SWC(cte_parallel=0.99999)
    sim = Simulation(noise=False)
    cube = _cube(195.119 * u.AA)
    with pytest.warns(UserWarning, match="does not model charge transfer"):
        time = _dark_time(cube, EXPOSURE, leaky, sim)
    assert np.array_equal(time, _dark_time(cube, EXPOSURE, DET, sim))
