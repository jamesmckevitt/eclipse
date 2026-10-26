"""Dark current collected during the clear and the read-out, in runs."""
import dataclasses

import astropy.units as u
import numpy as np
import pytest

from eclipse.config import Detector_EIS, Detector_SWC, Simulation, Telescope_EIS, Telescope_EUVST
from eclipse.data_processing import create_uniform_intensity_cube
from eclipse.monte_carlo import _dark_time, expected_dn_uncertainty, simulate_once
from eclipse.radiometric import dn_variance
from eclipse.readout import FocalPlane_SWC, ReadoutSequence, dark_current_time

DET = Detector_SWC()
TEL = Telescope_EUVST()
INTENSITY = 1000 * u.erg / (u.s * u.cm**2 * u.sr)
EXPOSURE = 10 * u.s


def _cube(rest, det=DET, tel=TEL, sim=None):
    sim = sim or Simulation(slit_width=0.2 * u.arcsec, noise=False)
    return create_uniform_intensity_cube(INTENSITY, rest, 20 * u.km / u.s, det, sim, tel=tel)


def _per_row():
    focal_plane = FocalPlane_SWC.from_detector(DET)
    n_rows = focal_plane.n_rows
    return focal_plane, dark_current_time(EXPOSURE, ReadoutSequence.from_detector(DET),
                                          n_rows)[:n_rows]


@pytest.mark.parametrize("ccd, row", [("left", 0), ("left", 1500), ("left", 2047),
                                      ("right", 0), ("right", 900)])
def test_a_pixel_takes_the_dark_time_of_the_row_recording_its_wavelength(ccd, row):
    focal_plane, per_row = _per_row()
    rest = focal_plane.wavelength(row, ccd)
    cube = _cube(rest)
    sim = Simulation(slit_width=0.2 * u.arcsec, noise=False)
    time = _dark_time(cube, EXPOSURE, DET, sim)
    middle = cube.data.shape[-1] // 2
    assert time[middle].to_value(u.s) == pytest.approx(per_row[row].to_value(u.s), rel=1e-7)
    assert np.all(time > EXPOSURE)


def test_rows_near_the_gap_wait_longest():
    focal_plane, per_row = _per_row()
    assert (per_row[-1] - EXPOSURE).to_value(u.s) == pytest.approx(1.15, abs=0.05)
    outer, inner = (_dark_time(_cube(w), EXPOSURE, DET, Simulation(noise=False))
                    for w in (172 * u.AA, 197 * u.AA))
    assert outer.max() < inner.min()


def test_a_wavelength_in_the_gap_takes_the_last_row():
    _, per_row = _per_row()
    time = _dark_time(_cube(198.9 * u.AA), EXPOSURE, DET, Simulation(noise=False))
    middle = len(time) // 2
    assert time[middle].to_value(u.s) == pytest.approx(per_row[-1].to_value(u.s), rel=1e-12)


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
