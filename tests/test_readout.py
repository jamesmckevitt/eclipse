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
"""
import astropy.units as u
import numpy as np
import pytest

from euvst_response.readout import (
    MEASURED_SLIT_IMAGE_TILT,
    FocalPlane_SWC,
    ReadoutSequence,
    dark_current_time,
    expose,
    smear_photons,
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
    # The tolerance is for the FFT the convolution uses, which leaves rounding
    # noise around 1e-20 photons where the answer is zero.
    rate = np.zeros((6, 1))
    rate[4] = 5.0
    sequence = small_sequence(dump_rows=0)
    smear = smear_photons(rate, sequence)
    assert smear[:4, 0] == pytest.approx(0.0, abs=1e-15)


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
    sequence = small_sequence()
    times = dark_current_time(2.0 * u.s, sequence, 6).to_value(u.s)
    assert np.all(np.diff(times) > 0)
    assert times[0] == pytest.approx(2.0 + 15.0e-6 + 10 * 500e-9)


def test_a_sequence_rejects_a_backwards_window():
    with pytest.raises(ValueError, match="first row to its last"):
        ReadoutSequence(windows=[(100, 50)])
