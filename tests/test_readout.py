"""Check the SW read-out model: where each row sits, and what it collects.

Two things are tested here. The first is the geometry: which wavelength a CCD
row records, given that the two devices are butted along the dispersion with
their registers on the outer edges. The expected wavelengths are the ones the
optical design gives for the focal plane edges and for the key lines, so a
failure means the row map moved rather than that two spellings of the same
polynomial disagree.

The second is the smear a frame collects when the chip stays illuminated while
it is cleared and read. Every expected value is written out from the clocking
rather than taken from the module:

    a packet read out of row r sits at row r - k while image row k - 1 is in
    the serial register, so it collects rate[r - k] * dwell[k], where

    dwell[k] = row_transfer_time + line_read_time if row k - 1 is read
             = row_transfer_time                  otherwise

and, before the exposure, the packet that ends the clear at row r has been
clocked down from row r + dump_rows, collecting one row transfer at each row on
the way.

The third is the charge each transfer leaves behind. The expected values come
from moving the charge one transfer at a time, each transfer keeping a fraction
cte of every well's charge and leaving the rest for the packet behind, rather
than from the closed form the module uses.
"""
import warnings

import astropy.units as u
import numpy as np
import pytest

from euvst_response.config import Detector_SWC
from euvst_response.readout import (
    MEASURED_SLIT_IMAGE_TILT,
    FocalPlane_SWC,
    ReadoutSequence,
    dark_current_time,
    expose,
    expose_variance,
    smear_photons,
    transfer_probabilities,
    windows_from_wavelengths,
)

# The design wavelengths at the four imaging-area edges, in Angstrom, from the
# fit to RSC-2022021C. Rows count from each CCD's serial register, which is on
# the outer edge, so row 0 is the outermost row and row 2047 is at the butt.
LEFT_ROW_0, LEFT_ROW_2047 = 163.5549, 198.2516
RIGHT_ROW_0, RIGHT_ROW_2047 = 234.0014, 199.5202

# Rows of the lines that matter in a flare, at the centre of the slit.
KEY_LINES = {
    "Fe IX 171.073": (171.073, "left", 442.5),
    "Fe XXIV 192.030": (192.030, "left", 1679.0),
    "Ca XVII 192.858": (192.858, "left", 1728.0),
    "Fe XII 195.119": (195.119, "left", 1861.7),
    "Ca XV 200.972": (200.972, "right", 1961.1),
    "Fe XIV 211.317": (211.317, "right", 1348.1),
}


# The small frames here are mostly not cleared, which ECLIPSE warns of; the
# warning has its own test below.
pytestmark = pytest.mark.filterwarnings("ignore:dump_rows is")


def small_sequence(**kwargs):
    """A short frame, so that a test can write the clocking out by hand."""
    defaults = dict(shutter=False, row_transfer_time=15.0 * u.us,
                    pixel_period=500.0 * u.ns, serial_prescan=2,
                    serial_image_pixels=6, serial_overscan=2,
                    parallel_overscan_rows=2, dump_rows=0, windows=[])
    defaults.update(kwargs)
    return ReadoutSequence(**defaults)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_the_outermost_row_of_each_ccd_is_the_one_at_the_register():
    fp = FocalPlane_SWC()
    assert fp.wavelength(0, "left").to_value(u.Angstrom) == pytest.approx(LEFT_ROW_0, abs=1e-3)
    assert fp.wavelength(2047, "left").to_value(u.Angstrom) == pytest.approx(LEFT_ROW_2047, abs=1e-3)
    assert fp.wavelength(0, "right").to_value(u.Angstrom) == pytest.approx(RIGHT_ROW_0, abs=1e-3)
    assert fp.wavelength(2047, "right").to_value(u.Angstrom) == pytest.approx(RIGHT_ROW_2047, abs=1e-3)


def test_wavelength_rises_with_row_on_the_left_ccd_and_falls_on_the_right():
    fp = FocalPlane_SWC()
    rows = np.arange(2048)
    assert np.all(np.diff(fp.wavelength(rows, "left").to_value(u.Angstrom)) > 0)
    assert np.all(np.diff(fp.wavelength(rows, "right").to_value(u.Angstrom)) < 0)


def test_the_rows_are_not_evenly_spaced_in_wavelength():
    # 17.0 mA per row at the short-wavelength end against 16.8 at the long one.
    # A single spacing would misplace a line by several rows at the gap.
    fp = FocalPlane_SWC()
    first = fp.wavelength(1, "left") - fp.wavelength(0, "left")
    last = fp.wavelength(1, "right") - fp.wavelength(0, "right")
    assert first.to_value(u.mAA) == pytest.approx(17.00, abs=0.02)
    assert abs(last.to_value(u.mAA)) == pytest.approx(16.79, abs=0.02)


def test_key_lines_land_where_the_design_puts_them():
    fp = FocalPlane_SWC()
    for name, (wavelength, ccd, row) in KEY_LINES.items():
        where, at = fp.row_of_wavelength(wavelength * u.Angstrom)
        assert where == ccd, name
        assert at == pytest.approx(row, abs=0.5), name


def test_row_of_wavelength_inverts_wavelength():
    fp = FocalPlane_SWC()
    for ccd in ("left", "right"):
        for row in (0, 17, 600, 1861, 2047):
            lam = fp.wavelength(row, ccd)
            where, back = fp.row_of_wavelength(lam)
            assert where == ccd
            assert back == pytest.approx(row, abs=1e-3)


def test_the_gap_between_the_ccds_records_nothing():
    fp = FocalPlane_SWC()
    with pytest.raises(ValueError, match="not on either CCD"):
        fp.row_of_wavelength(198.9 * u.Angstrom)


def test_a_wavelength_calibration_offset_moves_every_row():
    fp = FocalPlane_SWC(wavelength_offset=0.1 * u.Angstrom)
    plain = FocalPlane_SWC()
    for ccd in ("left", "right"):
        shift = fp.wavelength(1000, ccd) - plain.wavelength(1000, ccd)
        assert shift.to_value(u.Angstrom) == pytest.approx(0.1)


def test_row_edges_bracket_the_row_centres():
    fp = FocalPlane_SWC()
    edges = fp.row_edges("left").to_value(u.Angstrom)
    centres = fp.wavelength(np.arange(2048), "left").to_value(u.Angstrom)
    assert len(edges) == 2049
    assert np.all(edges[:-1] < centres)
    assert np.all(centres < edges[1:])


def test_light_stops_at_the_band_limits():
    # The baffle vignettes the beam at 170.0 and 212.3 Angstrom, which leaves
    # 380 dark rows on the left CCD and 1290 on the right. Charge from the lit
    # rows is clocked across all of them.
    fp = FocalPlane_SWC()
    assert fp.lit_rows("left") == (380, 2047)
    assert fp.lit_rows("right") == (1290, 2047)


def test_a_line_lands_on_one_row_all_along_the_slit_by_default():
    fp = FocalPlane_SWC()
    assert fp.wavelength(1861, "left", column=0) == fp.wavelength(1861, "left", column=2047)
    assert fp.row_of_wavelength(195.119 * u.Angstrom, column=0) == \
        fp.row_of_wavelength(195.119 * u.Angstrom, column=2047)


def test_the_slit_image_tilt_drifts_a_line_along_the_dispersion():
    # Switched on, a line drifts by the measured 0.991 rows of tilt plus 0.605
    # of curvature at one end of the slit, and by the difference of the two at
    # the other, which is the two rows end to end the design shows.
    fp = FocalPlane_SWC(slit_image_tilt=MEASURED_SLIT_IMAGE_TILT)
    centre_column = (fp.n_columns - 1) / 2
    end_column = centre_column + (140.0 / 0.159)
    _, middle = fp.row_of_wavelength(195.119 * u.Angstrom, column=centre_column)
    _, top = fp.row_of_wavelength(195.119 * u.Angstrom, column=end_column)
    _, bottom = fp.row_of_wavelength(195.119 * u.Angstrom, column=2 * centre_column - end_column)
    assert top - middle == pytest.approx(0.991 + 0.605, abs=0.01)
    assert bottom - middle == pytest.approx(-0.991 + 0.605, abs=0.01)


def test_the_tilt_leaves_the_centre_of_the_field_alone():
    fp = FocalPlane_SWC(slit_image_tilt=MEASURED_SLIT_IMAGE_TILT)
    plain = FocalPlane_SWC()
    centre_column = (fp.n_columns - 1) / 2
    assert fp.wavelength(1861, "left", column=centre_column).to_value(u.Angstrom) == \
        pytest.approx(plain.wavelength(1861, "left").to_value(u.Angstrom))


def test_charge_from_fe_xii_crosses_the_flare_lines_on_its_way_out():
    # The reason a flare smears: on the left CCD the register is at the
    # short-wavelength end, so the packets recording Fe XII 195.119 are clocked
    # through Fe XXIV 192.030 and Ca XVII 192.858.
    fp = FocalPlane_SWC()
    _, fe12 = fp.row_of_wavelength(195.119 * u.Angstrom)
    _, fe24 = fp.row_of_wavelength(192.030 * u.Angstrom)
    _, ca17 = fp.row_of_wavelength(192.858 * u.Angstrom)
    assert fe24 < fe12 and ca17 < fe12


# ---------------------------------------------------------------------------
# Clocking
# ---------------------------------------------------------------------------

def test_a_read_row_costs_the_whole_register():
    # 50 prescan + 1024 image + 20 overscan samples at 500 ns per sample.
    sequence = ReadoutSequence()
    assert sequence.line_read_time.to_value(u.us) == pytest.approx(547.0)


def test_reading_every_row_takes_as_long_as_the_rows_cost():
    # 2048 image rows plus 20 parallel overscan rows, each moved and read.
    sequence = ReadoutSequence(windows=[])
    expected = 2068 * (15.0e-6 + 547.0e-6)
    assert sequence.readout_duration(2048).to_value(u.s) == pytest.approx(expected)


def test_windows_pay_for_the_rows_they_read_and_dump_the_rest():
    # Sixteen 40-row windows: every row is still moved, but only the windowed
    # rows and the overscan go through the register.
    windows = [(100 * i, 100 * i + 39) for i in range(16)]
    sequence = ReadoutSequence(windows=windows)
    read_rows = 16 * 40 + 20
    expected = 2068 * 15.0e-6 + read_rows * 547.0e-6
    assert sequence.readout_duration(2048).to_value(u.s) == pytest.approx(expected)


def test_windows_from_wavelengths_covers_the_line():
    fp = FocalPlane_SWC()
    windows = windows_from_wavelengths(fp, [(194.9 * u.Angstrom, 195.3 * u.Angstrom)])
    (first, last), = windows
    _, row = fp.row_of_wavelength(195.119 * u.Angstrom)
    assert first <= row <= last
    assert last - first == pytest.approx(0.4 / 0.0169, abs=2)


@pytest.mark.parametrize("low, high", [
    (194.9, 195.3),       # one CCD
    (195.119, 200.972),   # Fe XII on the left CCD to Ca XV on the right
    (200.972, 195.119),   # the same, given the other way round
])
def test_a_window_holds_every_row_its_wavelengths_land_on(low, high):
    # Across the gap, each end of the range runs out to the butted edge, so
    # the rows it needs reach row 2047 on both CCDs.
    fp = FocalPlane_SWC()
    (first, last), = windows_from_wavelengths(fp, [(low * u.Angstrom, high * u.Angstrom)])
    rows = np.arange(fp.n_rows)
    for ccd in ("left", "right"):
        lam = fp.wavelength(rows, ccd).to_value(u.Angstrom)
        wanted = rows[(lam >= min(low, high)) & (lam <= max(low, high))]
        if wanted.size:
            assert first <= wanted.min() and wanted.max() <= last
    if min(low, high) < 199.0 < max(low, high):
        assert last == fp.n_rows - 1


# ---------------------------------------------------------------------------
# Smear
# ---------------------------------------------------------------------------

def test_a_shutter_leaves_no_smear():
    rate = np.ones((6, 3))
    smear = smear_photons(rate, small_sequence(shutter=True))
    assert smear.shape == (8, 3)
    assert np.all(smear == 0)


def test_uniform_light_smears_by_the_time_the_charge_spends_on_the_way_out():
    # Every row is read, so each transfer costs the same: 15 us to move plus
    # 10 samples at 500 ns to read. The packet from row r waits through r of
    # those intervals before it reaches the register.
    rate = np.full((6, 2), 3.0)
    sequence = small_sequence()
    dwell = 15.0e-6 + 10 * 500e-9
    smear = smear_photons(rate, sequence)
    expected = 3.0 * dwell * np.arange(6)
    assert smear[:6, 0] == pytest.approx(expected)


def test_clearing_the_image_area_smears_from_the_rows_above():
    # With an empty register a read costs nothing, so the two contributions can
    # be written out separately: during the clear row r collects the rows above
    # it, and during the read-out it collects the rows below it, one transfer
    # each way.
    rate = np.full((6, 1), 2.0)
    sequence = small_sequence(dump_rows=6, row_transfer_time=10.0 * u.us,
                              serial_prescan=0, serial_image_pixels=0,
                              serial_overscan=0)
    smear = smear_photons(rate, sequence)
    clear = 2.0 * 10.0e-6 * np.array([5, 4, 3, 2, 1, 0])
    readout = 2.0 * 10.0e-6 * np.arange(6)
    assert smear[:6, 0] == pytest.approx(clear + readout)


def test_no_clear_means_no_smear_from_the_rows_above():
    rate = np.zeros((6, 1))
    rate[4] = 5.0
    sequence = small_sequence(dump_rows=0)
    smear = smear_photons(rate, sequence)
    # Exactly: the rounding noise of the FFT is not left where no light went.
    assert np.all(smear[:4, 0] == 0.0)


def test_one_bright_row_streaks_toward_the_register():
    # A single bright row lands in every row read out after it, once per
    # transfer interval, and in every row below it once per clear transfer.
    rate = np.zeros((6, 1))
    rate[4, 0] = 100.0
    sequence = small_sequence(dump_rows=6)
    dwell = 15.0e-6 + 10 * 500e-9
    smear = smear_photons(rate, sequence)
    assert smear[5, 0] == pytest.approx(100.0 * dwell)          # crossed it once
    assert smear[3, 0] == pytest.approx(100.0 * 15.0e-6)        # clear only
    assert smear[4, 0] == pytest.approx(0.0, abs=1e-15)         # its own row


def test_dumped_rows_cost_only_a_transfer():
    # Reading one row of a six-row chip: the dumped rows still move, so a
    # packet crossing them collects a transfer each, not a read.
    rate = np.zeros((6, 1))
    rate[0, 0] = 10.0
    sequence = small_sequence(windows=[(0, 0)], parallel_overscan_rows=0)
    smear = smear_photons(rate, sequence)
    read_dwell = 15.0e-6 + 10 * 500e-9
    assert smear[1, 0] == pytest.approx(10.0 * read_dwell)   # row 0 was read
    assert smear[2, 0] == pytest.approx(10.0 * 15.0e-6)      # row 1 was dumped


def test_the_parallel_overscan_holds_smear_and_nothing_else():
    rate = np.full((6, 1), 4.0)
    sequence = small_sequence(parallel_overscan_rows=2)
    dwell = 15.0e-6 + 10 * 500e-9
    signal = expose(rate, 2.0 * u.s, sequence)
    assert signal.shape == (8, 1)
    # The first overscan packet enters at the top and crosses all six rows.
    assert signal[6, 0] == pytest.approx(4.0 * dwell * 6)
    assert signal[7, 0] == pytest.approx(4.0 * dwell * 6)


def test_the_exposure_lands_only_on_the_image_rows():
    rate = np.full((6, 1), 4.0)
    shuttered = expose(rate, 2.0 * u.s, small_sequence(shutter=True))
    assert shuttered[:6, 0] == pytest.approx(8.0)
    assert shuttered[6:, 0] == pytest.approx(0.0)


def test_smear_is_linear_in_the_light():
    rng = np.random.default_rng(3)
    rate = rng.random((12, 4))
    sequence = small_sequence(dump_rows=12, windows=[(2, 5)])
    doubled = smear_photons(2 * rate, sequence)
    assert doubled == pytest.approx(2 * smear_photons(rate, sequence))


def test_the_smear_depends_on_which_row_is_being_read_at_the_time():
    # A packet crossing a bright row picks up that row's light for as long as
    # the transfer it is in the middle of. The packet read out of row 11 crosses
    # the bright row 6 while image row 4 is in the register, so reading row 4
    # costs it a register read and reading row 5 instead does not, even though
    # the same number of rows is read either way.
    rate = np.zeros((12, 1))
    rate[6, 0] = 50.0
    reading_row_4 = smear_photons(rate, small_sequence(windows=[(4, 4)], parallel_overscan_rows=0))
    reading_row_5 = smear_photons(rate, small_sequence(windows=[(5, 5)], parallel_overscan_rows=0))
    assert reading_row_4[11, 0] == pytest.approx(50.0 * (15.0e-6 + 10 * 500e-9))
    assert reading_row_5[11, 0] == pytest.approx(50.0 * 15.0e-6)


def test_a_flare_line_smears_into_the_rows_beyond_it():
    # The whole point, on the real focal plane: put Fe XXIV 192.030 on the left
    # CCD and read a window on Fe XII 195.119. The Fe XII rows collect Fe XXIV
    # light they never saw during the exposure.
    fp = FocalPlane_SWC()
    _, fe24 = fp.row_of_wavelength(192.030 * u.Angstrom)
    _, fe12 = fp.row_of_wavelength(195.119 * u.Angstrom)
    rate = np.zeros((fp.n_rows, 4))
    rate[int(round(fe24))] = 1.0e4
    windows = windows_from_wavelengths(fp, [(194.9 * u.Angstrom, 195.3 * u.Angstrom)])
    sequence = ReadoutSequence(shutter=False, windows=windows)
    smeared = expose(rate, 1.0 * u.s, sequence)[int(round(fe12)), 0]
    clean = expose(rate, 1.0 * u.s, ReadoutSequence(shutter=True, windows=windows))[int(round(fe12)), 0]
    assert clean == 0.0
    assert smeared > 0.0


def test_dark_current_time_grows_down_the_frame():
    sequence = small_sequence(dump_rows=6)
    times = dark_current_time(2.0 * u.s, sequence, 6).to_value(u.s)
    # The clear brings row 0's packet down past the five rows above it, and
    # row 0 is read first. Each later row waits for one more read, 15 us of
    # transfer and 10 samples of 500 ns, and spends one transfer less in the
    # clear.
    assert times[0] == pytest.approx(2.0 + 5 * 15.0e-6)
    assert np.diff(times[:6]) == pytest.approx(np.full(5, 10 * 500e-9))
    # The overscan packets come in after the exposure and cross all six rows.
    assert times[6:] == pytest.approx(np.full(2, 6 * (15.0e-6 + 10 * 500e-9)))


@pytest.mark.parametrize("windows", [[], [(2, 3)]])
@pytest.mark.parametrize("dump_rows", [0, 3, 6, 10])
def test_a_packet_collects_dark_current_wherever_it_collects_light(windows, dump_rows):
    # Lit evenly at one photon a second, a packet collects outside the
    # exposure one photon for each second it spends in the image area then,
    # which is its dark time less the exposure.
    sequence = small_sequence(windows=windows, dump_rows=dump_rows)
    n_rows, exposure = 6, 2.0
    smear = smear_photons(np.ones((n_rows, 1)), sequence)[:, 0]
    dark = dark_current_time(exposure * u.s, sequence, n_rows).to_value(u.s)
    exposed = np.where(np.arange(dark.size) < n_rows, exposure, 0.0)
    np.testing.assert_allclose(dark, smear + exposed, rtol=1e-9, atol=1e-15)


def test_a_sequence_rejects_a_backwards_window():
    with pytest.raises(ValueError, match="first row to its last"):
        ReadoutSequence(windows=[(100, 50)])


# ---------------------------------------------------------------------------
# The edges of the CCDs, the windows, and the rows no light reaches
# ---------------------------------------------------------------------------

def test_row_of_wavelength_inverts_wavelength_exactly():
    fp = FocalPlane_SWC()
    for ccd in ("left", "right"):
        for row in (-0.5, 0, 17.3, 1861, 2047, 2047.5):
            where, back = fp.row_of_wavelength(fp.wavelength(row, ccd))
            assert where == ccd
            assert back == pytest.approx(row, abs=1e-9)


@pytest.mark.parametrize("wavelength, ccd, rows", [
    (198.2550, "left", (2047.0, 2047.5)),    # the butted half of the left CCD's last row
    (199.5150, "right", (2047.0, 2047.5)),   # and of the right CCD's
    (163.5500, "left", (-0.5, 0.0)),         # the register half of the left CCD's first row
])
def test_the_half_of_an_edge_row_beyond_its_centre_is_on_the_ccd(wavelength, ccd, rows):
    """Only row centres were looked at, so these were said to be in the gap or off the focal plane."""
    where, row = FocalPlane_SWC().row_of_wavelength(wavelength * u.Angstrom)
    assert where == ccd and rows[0] < row < rows[1]


def _rows_overlapping(fp, low, high, column=None):
    """Every row, on either CCD, whose span between its edges overlaps low to high."""
    rows = set()
    for ccd in ("left", "right"):
        edges = fp.row_edges(ccd, column).to_value(u.Angstrom)
        below, above = np.minimum(edges[:-1], edges[1:]), np.maximum(edges[:-1], edges[1:])
        rows |= set(np.flatnonzero((above > low) & (below < high)).tolist())
    return rows


@pytest.mark.parametrize("low, high", [(194.9, 195.3), (192.0, 192.06), (195.119, 200.972),
                                       (198.0, 198.6), (160.0, 164.0)])
def test_a_window_is_the_rows_its_wavelengths_land_on_and_no_more(low, high):
    """Rounding each end outward added a row whenever an end fell in the inner half of its row."""
    fp = FocalPlane_SWC()
    (first, last), = windows_from_wavelengths(fp, [(low * u.Angstrom, high * u.Angstrom)])
    wanted = _rows_overlapping(fp, low, high)
    assert (first, last) == (min(wanted), max(wanted))


def test_a_window_follows_a_tilted_line_along_the_whole_slit():
    """Placed at the centre of the field, the window missed the row the line reached at one end."""
    fp = FocalPlane_SWC(slit_image_tilt=MEASURED_SLIT_IMAGE_TILT)
    low, high = 194.9, 195.3
    (first, last), = windows_from_wavelengths(fp, [(low * u.Angstrom, high * u.Angstrom)])
    wanted = set()
    for column in range(0, fp.n_columns, 7):
        wanted |= _rows_overlapping(fp, low, high, column)
    wanted |= _rows_overlapping(fp, low, high, fp.n_columns - 1)
    assert (first, last) == (min(wanted), max(wanted))
    assert last > max(_rows_overlapping(fp, low, high, (fp.n_columns - 1) / 2))


def test_a_range_on_neither_ccd_has_no_window():
    with pytest.raises(ValueError, match="No part of"):
        windows_from_wavelengths(FocalPlane_SWC(), [(198.4 * u.Angstrom, 198.8 * u.Angstrom)])


def test_a_partial_clear_is_said_to_leave_out_the_charge_it_leaves():
    rate = np.ones((6, 1))
    with pytest.warns(UserWarning, match="dump_rows is 2, fewer than the 6 rows"):
        smear_photons(rate, small_sequence(dump_rows=2))
    with pytest.warns(UserWarning, match="The dark current here starts from an empty chip"):
        dark_current_time(1.0 * u.s, small_sequence(dump_rows=2, shutter=True), 6)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        smear_photons(rate, small_sequence(dump_rows=6))
        dark_current_time(1.0 * u.s, small_sequence(dump_rows=6), 6)
        # With a shutter the smear is none whatever the clear.
        smear_photons(rate, small_sequence(dump_rows=2, shutter=True))


class _RowsAreWavelengths:
    """A focal plane of ten rows on the left CCD only, each row at its own wavelength in Angstrom."""
    n_rows = 10

    def _extreme_columns(self):
        return [0]

    def _row_on(self, ccd, target, column):
        return target if ccd == "left" else np.inf


@pytest.mark.parametrize("low, high, window", [
    (-3.0, 0.2, (0, 0)),
    (2.0, 2.0, (2, 2)),
    (3.5, 3.5, (3, 3)),     # a boundary between rows is the lower one's
    (3.5, 4.5, (4, 4)),
    (8.7, 12.0, (9, 9)),
])
def test_a_window_is_the_rows_a_range_overlaps(low, high, window):
    assert windows_from_wavelengths(_RowsAreWavelengths(), [(low * u.AA, high * u.AA)]) == [window]


@pytest.mark.parametrize("low, high", [(-3.0, -0.5), (9.5, 12.0), (-0.5, -0.5), (-3.0, -1.0)])
def test_a_range_that_only_reaches_the_edge_of_the_ccd_is_not_on_it(low, high):
    """Reaching row 0's outer edge from below gave the window (-1, -1)."""
    with pytest.raises(ValueError, match="is on either CCD"):
        windows_from_wavelengths(_RowsAreWavelengths(), [(low * u.AA, high * u.AA)])


# ---------------------------------------------------------------------------
# The camera's settings come from its configuration
# ---------------------------------------------------------------------------
def test_the_read_out_and_focal_plane_take_their_camera_from_the_detector():
    """The format, the pixels and the timing are set once, in Detector_SWC."""
    det = Detector_SWC(n_rows=100, n_columns=40, ccd_gap=2.0 * u.mm,
                       pix_size=(10.0 * u.um).cgs / u.pixel,
                       plate_scale_angle=0.2 * u.arcsec / u.pixel,
                       row_transfer_time=20.0 * u.us, pixel_period=1.0 * u.us,
                       serial_prescan=0, serial_overscan=3, parallel_overscan_rows=5)

    fp = FocalPlane_SWC.from_detector(det, lit_band=(180.0 * u.Angstrom, 190.0 * u.Angstrom))
    # The camera's own settings are the detector's, and cannot be given here.
    with pytest.raises(ValueError, match="gap, n_rows come from the detector"):
        FocalPlane_SWC.from_detector(det, n_rows=50, gap=3.0 * u.mm)
    assert (fp.n_rows, fp.n_columns) == (100, 40)
    assert fp.pixel_size.to_value(u.um) == pytest.approx(10.0)
    assert fp.gap == 2.0 * u.mm
    assert fp.plate_scale.to_value(u.arcsec) == pytest.approx(0.2)
    assert fp.lit_band == (180.0 * u.Angstrom, 190.0 * u.Angstrom)

    sequence = ReadoutSequence.from_detector(det, windows=[(0, 9)])
    assert sequence.shutter is True
    assert (sequence.cte_parallel, sequence.cte_serial) == (1.0, 1.0)
    leaky = ReadoutSequence.from_detector(Detector_SWC(cte_parallel=0.99999, cte_serial=0.9999))
    assert (leaky.cte_parallel, leaky.cte_serial) == (0.99999, 0.9999)
    assert sequence.row_transfer_time == 20.0 * u.us
    assert sequence.pixel_period == 1.0 * u.us
    assert (sequence.serial_prescan, sequence.serial_image_pixels,
            sequence.serial_overscan) == (0, 20, 3)
    assert sequence.parallel_overscan_rows == 5
    assert sequence.dump_rows == 100
    assert sequence.windows == [(0, 9)]
    assert sequence.line_read_time == 23.0 * u.us

    # The observation's own choices override the camera's.
    other = ReadoutSequence.from_detector(det, shutter=False, dump_rows=10)
    assert other.shutter is False and other.dump_rows == 10 and other.windows == []

    # An odd number of columns gives one output a pixel more, and a row takes
    # as long as that output.
    odd = ReadoutSequence.from_detector(Detector_SWC(n_columns=41, serial_prescan=0,
                                                     serial_overscan=0))
    assert odd.serial_image_pixels == 21
    assert odd.line_read_time == 21 * Detector_SWC.pixel_period


def test_the_defaults_are_those_of_the_default_camera():
    """Made on their own, the two classes describe the camera as Detector_SWC does by default."""
    det = Detector_SWC()
    fp, from_det = FocalPlane_SWC(), FocalPlane_SWC.from_detector(det)
    for name in ("n_rows", "n_columns", "pixel_size", "gap", "plate_scale"):
        assert np.all(getattr(fp, name) == getattr(from_det, name)), name
    sequence, from_det = ReadoutSequence(), ReadoutSequence.from_detector(det)
    for name in ("shutter", "row_transfer_time", "pixel_period", "serial_prescan",
                 "serial_image_pixels", "serial_overscan", "parallel_overscan_rows",
                 "dump_rows", "windows", "cte_parallel", "cte_serial"):
        assert np.all(getattr(sequence, name) == getattr(from_det, name)), name
    # The numbers the documents give, so that a change to either class shows.
    assert sequence.line_read_time.to_value(u.us) == pytest.approx(547.0)
    assert fp.pixel_size.to_value(u.um) == pytest.approx(13.5)


def _transfer(wells, cte):
    """One transfer: the front well's charge moved out, and every well keeping what it is left."""
    moved = cte * wells
    left = wells - moved
    after = np.empty_like(wells)
    after[:-1] = moved[1:] + left[:-1]
    after[-1] = left[-1]
    return moved[0], after


def _read_one_transfer_at_a_time(rate, exposure, sequence):
    """The charge each packet brings to the register, moving it one transfer at a time."""
    n_rows = rate.size
    wells = np.zeros(n_rows)
    lit = not sequence.shutter
    for _ in range(sequence.dump_rows):
        if lit:
            wells = wells + rate * sequence.row_transfer_time.to_value(u.s)
        _, wells = _transfer(wells, sequence.cte_parallel)
    wells = wells + rate * exposure
    dwell = sequence.dwell(n_rows)
    read = np.zeros(n_rows + sequence.parallel_overscan_rows)
    for step in range(read.size):
        read[step], wells = _transfer(wells, sequence.cte_parallel)
        if lit:
            wells = wells + rate * dwell[step]
    return read


def test_transfer_probabilities_count_the_ways_charge_can_be_held_back():
    """Arriving k packets late after m transfers is m moves and k waits, the last a move."""
    m, cte = np.array([1, 2, 5]), 0.8
    probability = transfer_probabilities(m, cte)
    assert probability[0] == pytest.approx(cte ** m)
    assert probability[1] == pytest.approx(m * cte ** m * (1 - cte))
    assert probability[2] == pytest.approx(m * (m + 1) / 2 * cte ** m * (1 - cte) ** 2)
    assert probability.sum(axis=0) == pytest.approx(1.0, abs=1e-14)
    assert transfer_probabilities(m, 1.0).tolist() == [[1.0, 1.0, 1.0]]
    # Stopping at the latest arrival wanted.
    assert transfer_probabilities(m, cte, most=2).shape == (3, 3)
    with pytest.raises(ValueError, match="at least one transfer"):
        transfer_probabilities([0], cte)


@pytest.mark.parametrize("shutter", [True, False])
@pytest.mark.parametrize("windows", [[], [(2, 4), (8, 9)]])
@pytest.mark.parametrize("dump_rows", [12, 5, 0])
@pytest.mark.parametrize("cte", [0.99, 0.9])
def test_charge_left_behind_arrives_as_moving_it_one_transfer_at_a_time(
        shutter, windows, dump_rows, cte):
    """The clear, the exposure, every read and dumped row, and the packets dumped in the clear."""
    rate = np.random.default_rng(1).uniform(0.0, 5.0, 12)
    rate[3] = 200.0
    sequence = ReadoutSequence(shutter=shutter, windows=windows, dump_rows=dump_rows,
                               parallel_overscan_rows=3, cte_parallel=cte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = expose(rate[:, np.newaxis], 0.7 * u.s, sequence)[:, 0]
    expected = _read_one_transfer_at_a_time(rate, 0.7, sequence)
    assert frame == pytest.approx(expected, rel=1e-12, abs=1e-12 * expected.max())


def test_charge_left_behind_along_the_register_trails_towards_its_middle():
    """Each half is read from its own end, through the prescan."""
    rate = np.random.default_rng(2).uniform(0.0, 3.0, (4, 10))
    rate[1, 2] = 100.0
    sequence = ReadoutSequence(serial_prescan=3, serial_image_pixels=5, parallel_overscan_rows=0,
                               dump_rows=4, cte_serial=0.9)
    frame = expose(rate, 1.0 * u.s, sequence)
    expected = np.zeros_like(rate)
    for columns in (np.arange(5), np.arange(9, 4, -1)):
        for row in range(rate.shape[0]):
            register = np.concatenate([np.zeros(3), rate[row, columns]])
            read = []
            for _ in range(register.size):
                out, register = _transfer(register, 0.9)
                read.append(out)
            expected[row, columns] = read[3:]
    assert frame == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="needs the frame's every column"):
        expose(rate[:, :8], 1.0 * u.s, sequence)


def test_a_perfect_transfer_leaves_the_frame_as_it_was():
    """The defaults change nothing: the smear and the exposure, as without the transfer."""
    rate = np.random.default_rng(3).uniform(0.0, 5.0, (40, 6))
    sequence = ReadoutSequence(shutter=False, windows=[(5, 9)], parallel_overscan_rows=4)
    smear = smear_photons(rate, sequence)
    expected = smear.copy()
    expected[:40] += rate * 2.0
    assert np.array_equal(expose(rate, 2.0 * u.s, sequence), expected)
    assert np.array_equal(expose_variance(rate, rate, 2.0 * u.s, sequence),
                          expose(rate, 2.0 * u.s, sequence))


def test_a_bright_row_trails_into_the_rows_read_after_it():
    """What is left behind is picked up by the packets behind, and none of it goes the other way."""
    rate = np.zeros((20, 1))
    rate[5] = 1000.0
    sequence = ReadoutSequence(shutter=True, parallel_overscan_rows=0, cte_parallel=0.999)
    frame = expose(rate, 1.0 * u.s, sequence)[:, 0]
    assert np.all(frame[:5] == 0.0)
    assert frame[5] == pytest.approx(1000.0 * 0.999 ** 6)
    assert np.all(np.diff(frame[6:]) <= 0.0) and frame[6] > 0.0
    assert frame.sum() == pytest.approx(1000.0, rel=1e-12)


def test_the_variance_of_charge_left_behind_is_that_of_draws_of_every_electron():
    """Photons of three electrons each, every electron left behind or not on its own."""
    rng = np.random.default_rng(4)
    rate, cte, per_photon, draws = np.array([4.0, 0.5, 9.0, 1.0, 2.0, 0.2]), 0.8, 3, 40000
    sequence = ReadoutSequence(shutter=False, parallel_overscan_rows=2, cte_parallel=cte,
                               row_transfer_time=0.05 * u.s, pixel_period=1e-3 * u.s,
                               serial_prescan=0, serial_image_pixels=1, serial_overscan=0)

    def collect(wells, seconds):
        return wells + per_photon * rng.poisson(rate * seconds, size=wells.shape)

    def move(wells):
        kept = rng.binomial(wells, cte)
        after = np.empty_like(wells)
        after[:, :-1] = kept[:, 1:] + wells[:, :-1] - kept[:, :-1]
        after[:, -1] = wells[:, -1] - kept[:, -1]
        return kept[:, 0], after

    wells = np.zeros((draws, rate.size), dtype=np.int64)
    for _ in range(sequence.dump_rows):
        _, wells = move(collect(wells, sequence.row_transfer_time.to_value(u.s)))
    wells = collect(wells, 0.6)
    dwell = sequence.dwell(rate.size)
    read = np.zeros((draws, rate.size + 2))
    for step in range(read.shape[1]):
        read[:, step], wells = move(wells)
        wells = collect(wells, dwell[step])
    variance = expose_variance((per_photon ** 2 * rate)[:, np.newaxis],
                               (per_photon * rate)[:, np.newaxis], 0.6 * u.s, sequence)[:, 0]
    # Each drawn variance is good to about sqrt(2 / draws), 0.7 percent.
    assert read.var(axis=0) == pytest.approx(variance, rel=0.035)


def test_dark_charge_is_left_behind_like_any_other():
    """The dark current time is that of an even light, with or without a shutter."""
    sequence = ReadoutSequence(windows=[(10, 19)], parallel_overscan_rows=5, cte_parallel=0.99)
    time = dark_current_time(2.0 * u.s, sequence, 60).to_value(u.s)
    lit = ReadoutSequence(shutter=False, windows=[(10, 19)], parallel_overscan_rows=5,
                          cte_parallel=0.99)
    assert time == pytest.approx(expose(np.ones((60, 1)), 2.0 * u.s, lit)[:, 0], rel=1e-12)
    assert time[0] < dark_current_time(2.0 * u.s, ReadoutSequence(
        windows=[(10, 19)], parallel_overscan_rows=5), 60).to_value(u.s)[0]


def test_a_transfer_efficiency_is_a_fraction():
    with pytest.raises(ValueError, match="cte_parallel is the fraction"):
        ReadoutSequence(cte_parallel=0.0)
    with pytest.raises(ValueError, match="cte_serial is the fraction"):
        ReadoutSequence(cte_serial=1.5)
