"""
Reading out the SW CCDs, with and without a shutter.

The SW focal plane is two Teledyne e2v CCD42-40 devices butted along the
dispersion, each 2048 x 2048 pixels of 13.5 micron, with a 1 mm gap between
them.  Their serial registers are on the two outer edges, so charge is clocked
*along the dispersion*: a row of the CCD is one wavelength, and the columns of a
row run along the slit.  Rows are numbered from the register, so row 0 is the
outermost row of each device and row 2047 sits at the butted edge.

A frame is taken in three steps (SOLC-EUVST-MSSL-ICD-0003 v3.1 {SWI-IRD}-032):
the FEE dumps a number of rows to clear the image area, waits for the exposure,
then reads out row by row.  Rows inside a read-out window go through the serial
register; the rest are dumped through the dump drain, which is much quicker.

With a mechanical shutter the chip is dark for the first and third steps, and a
pixel records only what fell on it during the exposure.  Without one, every
charge packet keeps collecting light while it is clocked, and so carries a
sample of every row it crossed on its way to the register.  That is what
:func:`smear_photons` computes, and why a bright line contaminates the rows
between it and the register.

Sources
-------
RSC-2022021C   SOLAR-C EUVST Optical Design Summary (ver.20230909): focal plane
               layout, wavelength coverage and dispersion.
SOLC-EUVST-MSSL-SP-0001 v1.4: CCD format, registers and dump drain.
SOLC-EUVST-MSSL-RS-0002 v2.0: windowing, the dump-and-wait sequence, and the
               500 ns pixel period.
SOLC-EUVST-MSSL-ICD-0003 v3.1: the same sequence at the FEE to SEB interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import astropy.units as u
import numpy as np
from scipy.signal import fftconvolve

# Wavelength in Angstrom as a function of position along the dispersion, in mm
# from the centre of the focal plane (the middle of the gap), positive toward
# longer wavelength.  Fitted to the positions RSC-2022021C gives for the two
# detector edges, the focal plane centre, the chief ray and the dispersion
# there, together with the ray-traced line positions drawn on its SW footprint
# figure.  A single linear scale is not good enough: the design runs from 17.0
# mA per row at the short-wavelength end to 16.8 at the long one, so the
# nominal 16.9 misplaces a line by up to nine rows.
DISPERSION_COEFFICIENTS = (198.885944, 1.2516577, -1.361114e-4)

# A spectral line is an image of the slit, and the optics do not lay that image
# down on exactly one row: it drifts along the dispersion from one end of the
# slit to the other.  These are the row offsets at 140 arcsec from the centre of
# the field, the linear term first and then the quadratic, measured from the
# ray-traced line positions on the SW footprint figure of RSC-2022021C.  All
# eleven lines drawn there agree on them to a few tenths of a row.  Pass them to
# FocalPlane_SWC to model the drift; by default a line lands on one row all the
# way along the slit.
MEASURED_SLIT_IMAGE_TILT = (0.991, 0.605)

# The field angle those offsets are quoted at.
TILT_REFERENCE_ANGLE = 140.0 * u.arcsec


@dataclass
class FocalPlane_SWC:
    """
    Geometry of the SW focal plane, and the wavelength each row records.

    Parameters
    ----------
    n_rows, n_columns : int
        Pixels along the dispersion and along the slit, per CCD.
    pixel_size : u.Quantity
        Pixel pitch.
    gap : u.Quantity
        Space between the two imaging areas.
    dispersion : tuple of float
        Coefficients of the wavelength scale, lowest order first, in Angstrom
        for a position in mm.  See :data:`DISPERSION_COEFFICIENTS`.
    wavelength_offset : u.Quantity
        Added to every row's wavelength.  The design positions above are good to
        about a row, but the camera's alignment to the beam is quoted at
        +/-0.79 Angstrom, some 47 rows (SOLC-EUVST-MSSL-RP-0007 v1.2 p45), so an
        as-built or in-flight wavelength calibration belongs here.
    lit_band : tuple of u.Quantity
        The wavelengths between which light reaches the detectors; outside them
        a baffle vignettes the beam.  Rows outside this band record nothing, but
        charge still crosses them on its way out.
    plate_scale : u.Quantity
        Angle a pixel covers along the slit.
    slit_image_tilt : tuple of float
        How far a line drifts along the dispersion between the centre of the
        field and 140 arcsec from it, in rows: a linear term and a quadratic
        one.  The default ``(0.0, 0.0)`` lays each line on one row for the whole
        slit, and the methods below then ignore their *column* argument.  Pass
        :data:`MEASURED_SLIT_IMAGE_TILT` to model the drift the optical design
        shows, which is about two rows end to end.  Which end of the slit it
        runs toward is not documented, so the sign may need flipping against a
        calibration.
    """

    n_rows: int = 2048
    n_columns: int = 2048
    pixel_size: u.Quantity = 13.5 * u.micron
    gap: u.Quantity = 1.0 * u.mm
    dispersion: Tuple[float, ...] = DISPERSION_COEFFICIENTS
    wavelength_offset: u.Quantity = 0.0 * u.Angstrom
    lit_band: Tuple[u.Quantity, u.Quantity] = (170.0 * u.Angstrom, 212.3 * u.Angstrom)
    plate_scale: u.Quantity = 0.159 * u.arcsec
    slit_image_tilt: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if self.n_rows < 1 or self.n_columns < 1:
            raise ValueError(
                f"A CCD needs at least one row and one column, got "
                f"{self.n_rows} x {self.n_columns}."
            )
        if len(self.dispersion) < 2:
            raise ValueError(
                "dispersion needs at least a constant and a linear coefficient, "
                f"got {len(self.dispersion)} coefficient(s)."
            )

    @property
    def outer_edge(self) -> u.Quantity:
        """Distance from the focal plane centre to the outer edge of a CCD."""
        return self.n_rows * self.pixel_size.to(u.mm) + self.gap.to(u.mm) / 2

    def _check_ccd(self, ccd: str) -> str:
        if ccd not in ("left", "right"):
            raise ValueError(
                f"ccd must be 'left' (short wavelength) or 'right' (long "
                f"wavelength), got {ccd!r}."
            )
        return ccd

    def position(self, row, ccd: str) -> u.Quantity:
        """
        Position along the dispersion of the centre of *row*, in mm from the
        focal plane centre.  Rows count from the serial register, which is on
        the outer edge of each CCD, so the position runs inward as the row
        number rises.
        """
        self._check_ccd(ccd)
        offset = (np.asarray(row, dtype=float) + 0.5) * self.pixel_size.to_value(u.mm)
        edge = self.outer_edge.to_value(u.mm)
        return (-edge + offset if ccd == "left" else edge - offset) * u.mm

    def field_angle(self, column) -> u.Quantity:
        """Angle along the slit of a column, from the centre of the field."""
        middle = (self.n_columns - 1) / 2
        return (np.asarray(column, dtype=float) - middle) * self.plate_scale

    def line_shift(self, column=None) -> u.Quantity:
        """
        How far a line lands from where it would at the centre of the field, in
        mm along the dispersion, positive toward longer wavelength.  Zero
        everywhere unless :attr:`slit_image_tilt` says otherwise.
        """
        linear, quadratic = self.slit_image_tilt
        if column is None or (linear == 0.0 and quadratic == 0.0):
            return 0.0 * u.mm
        f = (self.field_angle(column) / TILT_REFERENCE_ANGLE).to_value(u.dimensionless_unscaled)
        return (linear * f + quadratic * f**2) * self.pixel_size.to(u.mm)

    def wavelength(self, row, ccd: str, column=None) -> u.Quantity:
        """
        Wavelength recorded by the centre of *row*.

        Give *column* to include the drift of a line along the slit; without
        :attr:`slit_image_tilt` it makes no difference.
        """
        s = (self.position(row, ccd) - self.line_shift(column)).to_value(u.mm)
        lam = np.polynomial.polynomial.polyval(s, self.dispersion) * u.Angstrom
        return lam + self.wavelength_offset

    def row_edges(self, ccd: str, column=None) -> u.Quantity:
        """
        Wavelengths at the boundaries between rows, ``n_rows + 1`` of them.

        Binning a spectrum onto the detector conserves flux when it is
        integrated between these, rather than sampled at the row centres.  They
        decrease with row number on the right CCD, since its register is at the
        long-wavelength end.
        """
        s = (self.position(np.arange(self.n_rows + 1) - 0.5, ccd)
             - self.line_shift(column)).to_value(u.mm)
        lam = np.polynomial.polynomial.polyval(s, self.dispersion) * u.Angstrom
        return lam + self.wavelength_offset

    def row_of_wavelength(self, wavelength: u.Quantity, column=None) -> Tuple[str, float]:
        """
        Which CCD records a wavelength, and at which row.

        Returns the row as a float, where an integer value is the centre of that
        row.  Raises if the wavelength falls in the gap or off the focal plane.
        Give *column* to include the drift of a line along the slit.
        """
        target = u.Quantity(wavelength).to_value(u.Angstrom)
        rows = np.arange(self.n_rows)
        for ccd in ("left", "right"):
            lam = self.wavelength(rows, ccd, column).to_value(u.Angstrom)
            if min(lam) <= target <= max(lam):
                order = np.argsort(lam)
                return ccd, float(np.interp(target, lam[order], rows[order]))
        raise ValueError(
            f"{u.Quantity(wavelength)} is not on either CCD. The left CCD covers "
            f"{self.wavelength(0, 'left'):.2f} to "
            f"{self.wavelength(self.n_rows - 1, 'left'):.2f}, the right CCD "
            f"{self.wavelength(self.n_rows - 1, 'right'):.2f} to "
            f"{self.wavelength(0, 'right'):.2f}, and the gap between them is not "
            f"recorded."
        )

    def lit_rows(self, ccd: str, column=None) -> Tuple[int, int]:
        """
        First and last row of *ccd* whose centre receives light, inclusive.

        The edge is treated as sharp at :attr:`lit_band`.  The documents put it
        within a few rows of there but do not dimension the baffle, so a row or
        two either side of these two values is a modelling choice rather than a
        measurement.  Rows outside the range still pass charge, and on the right
        CCD there are some 1290 of them between the band and the register.
        """
        self._check_ccd(ccd)
        lam = self.wavelength(np.arange(self.n_rows), ccd, column).to_value(u.Angstrom)
        low, high = sorted(u.Quantity(limit).to_value(u.Angstrom) for limit in self.lit_band)
        lit = np.flatnonzero((lam >= low) & (lam <= high))
        if lit.size == 0:
            raise ValueError(
                f"No row of the {ccd} CCD sees light: it covers "
                f"{lam.min():.2f} to {lam.max():.2f} Angstrom and the band is "
                f"{low:.2f} to {high:.2f}."
            )
        return int(lit[0]), int(lit[-1])


@dataclass
class ReadoutSequence:
    """
    How the FEE clocks a frame out of the SW CCDs.

    Both CCDs are clocked together, and a row wanted on either of them is read
    on both, so one row timeline covers the pair
    (SOLC-EUVST-MSSL-RS-0002 v2.0 SolC-FPGA-RS-033).

    Parameters
    ----------
    shutter : bool
        True keeps the chip dark outside the exposure, as a mechanical shutter
        does.  False leaves it illuminated while the image area is cleared and
        read, which is what smears the frame.
    row_transfer_time : u.Quantity
        Time to move every row one step toward the register.  Configurable in
        flight; 15 us is the value in the SWC modes table and in the read-out
        times of RP-0007 v1.2 Table 5-11.
    pixel_period : u.Quantity
        Time to move one pixel along the serial register and digitise it.
    serial_prescan, serial_image_pixels, serial_overscan : int
        Samples read per output for each row that goes through the register.
        The register is split, so each half carries 1024 image pixels, with the
        prescan at the outer end and the overscan at the middle.  Both scans are
        configurable in flight between 0 and 200.
    parallel_overscan_rows : int
        Rows clocked and read after the last image row.  Without a shutter these
        hold pure smear, since their charge crosses the whole illuminated image
        area on the way out.
    dump_rows : int
        Row transfers used to clear the image area before the exposure.  The
        default clears a whole CCD.
    windows : list of tuple of int
        Inclusive row ranges that are read out.  An empty list reads every row,
        which is the slowest case.
    """

    shutter: bool = True
    row_transfer_time: u.Quantity = 15.0 * u.us
    pixel_period: u.Quantity = 500.0 * u.ns
    serial_prescan: int = 50
    serial_image_pixels: int = 1024
    serial_overscan: int = 20
    parallel_overscan_rows: int = 20
    dump_rows: int = 2048
    windows: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        for first, last in self.windows:
            if first > last:
                raise ValueError(
                    f"A window runs from its first row to its last, got "
                    f"({first}, {last})."
                )
            if first < 0:
                raise ValueError(f"Window rows start at 0, got {first}.")
        if self.dump_rows < 0:
            raise ValueError(f"dump_rows cannot be negative, got {self.dump_rows}.")
        if self.parallel_overscan_rows < 0:
            raise ValueError(
                f"parallel_overscan_rows cannot be negative, got "
                f"{self.parallel_overscan_rows}."
            )

    @property
    def samples_per_row(self) -> int:
        """Samples each output digitises for one row that is read."""
        return self.serial_prescan + self.serial_image_pixels + self.serial_overscan

    @property
    def line_read_time(self) -> u.Quantity:
        """Time to read one row through the register, both halves at once."""
        return (self.samples_per_row * self.pixel_period).to(u.s)

    def is_read(self, n_rows: int) -> np.ndarray:
        """
        Which rows go through the serial register, as a boolean array covering
        the image rows and then the parallel overscan rows.  The overscan is
        always read; with no windows, so is everything else.
        """
        total = n_rows + self.parallel_overscan_rows
        if not self.windows:
            return np.ones(total, dtype=bool)
        read = np.zeros(total, dtype=bool)
        for first, last in self.windows:
            read[first:min(last, n_rows - 1) + 1] = True
        read[n_rows:] = True
        return read

    def dwell(self, n_rows: int) -> np.ndarray:
        """
        Time in seconds between one row transfer and the next, for every
        transfer of a read-out.

        Element ``k`` covers the interval after transfer ``k + 1``, during which
        image row ``k`` sits in the serial register and is either read or
        dumped.  Every charge packet in the image area is stationary for that
        whole interval, which is what makes the smear depend on the window
        layout rather than only on the number of rows.
        """
        read = self.is_read(n_rows)
        return (self.row_transfer_time.to_value(u.s)
                + read * self.line_read_time.to_value(u.s))

    def readout_duration(self, n_rows: int) -> u.Quantity:
        """How long one read-out takes, from the first transfer to the last."""
        return self.dwell(n_rows).sum() * u.s

    def dump_duration(self) -> u.Quantity:
        """How long clearing the image area takes."""
        return (self.dump_rows * self.row_transfer_time).to(u.s)

    def time_until_read(self, n_rows: int) -> np.ndarray:
        """
        Seconds from the end of the exposure until each row reaches the serial
        register, for the image rows and then the parallel overscan rows.  A row
        collects dark current for this long after its exposure ends, whether or
        not there is a shutter.
        """
        return np.cumsum(self.dwell(n_rows))


def windows_from_wavelengths(focal_plane: FocalPlane_SWC,
                             ranges: Sequence[Tuple[u.Quantity, u.Quantity]],
                             ) -> List[Tuple[int, int]]:
    """
    Turn wavelength ranges into the row ranges a read-out window covers.

    Both CCDs share one row timeline, so the returned ranges are row numbers
    without a CCD attached: a window over rows 1850 to 1880 makes those rows
    slow on both devices, whatever wavelength they are on the other one.
    """
    windows = []
    for low, high in ranges:
        rows = sorted(focal_plane.row_of_wavelength(w)[1] for w in (low, high))
        windows.append((int(np.floor(rows[0])), int(np.ceil(rows[1]))))
    return windows


def smear_photons(rate: np.ndarray, sequence: ReadoutSequence) -> np.ndarray:
    """
    Photons a shutterless frame collects outside its exposure.

    Charge is clocked toward the register while the chip is still illuminated,
    so a packet collects light from every row it crosses.  For the packet read
    out of row ``r``, the read-out contributes ``sum_k rate[r - k] * dwell[k]``,
    and clearing the image area beforehand contributes ``row_transfer_time``
    times the rows above it, since the packet that ends at row ``r`` was clocked
    down from ``r + dump_rows``.

    Parameters
    ----------
    rate : np.ndarray
        Photons per second reaching each pixel of one CCD, shaped
        ``(n_rows, n_columns)`` with row 0 at the serial register.
    sequence : ReadoutSequence
        The clocking.  With ``shutter`` True this returns zeros.

    Returns
    -------
    np.ndarray
        Photons collected outside the exposure, shaped
        ``(n_rows + parallel_overscan_rows, n_columns)``.  The extra rows are
        the parallel overscan, which sees nothing but smear.
    """
    rate = np.asarray(rate, dtype=float)
    if rate.ndim != 2:
        raise ValueError(
            f"rate is one CCD's pixels, shaped (n_rows, n_columns), got shape "
            f"{rate.shape}."
        )
    n_rows, _ = rate.shape
    total_rows = n_rows + sequence.parallel_overscan_rows
    if sequence.shutter:
        return np.zeros((total_rows, rate.shape[1]))

    # Read-out.  The packet leaving row r is at row r - k while image row k - 1
    # is in the register, so the smear is a convolution of the rate with the
    # dwell times, and the parallel overscan rows are the terms past the last
    # image row.
    kernel = np.concatenate([[0.0], sequence.dwell(n_rows)])
    smear = fftconvolve(rate, kernel[:, np.newaxis], mode="full", axes=0)[:total_rows]

    # Clearing the image area.  The packet that ends the clear at row r has
    # come down from row r + dump_rows, collecting one row transfer at each row
    # on the way.  Rows above the image area are empty, so the sum stops there.
    if sequence.dump_rows:
        above = np.zeros((n_rows + 1, rate.shape[1]))
        above[:-1] = np.cumsum(rate[::-1], axis=0)[::-1]
        first = np.arange(1, n_rows + 1)
        last = np.minimum(np.arange(n_rows) + sequence.dump_rows + 1, n_rows)
        smear[:n_rows] += (above[first] - above[last]) * sequence.row_transfer_time.to_value(u.s)

    # The sum above cannot be negative, but the convolution is done by FFT and
    # leaves rounding noise of order 1e-20 where the answer is zero, which would
    # otherwise reach the Poisson draw downstream as a negative mean.
    return np.maximum(smear, 0.0)


def expose(rate: np.ndarray, exposure: u.Quantity, sequence: ReadoutSequence) -> np.ndarray:
    """
    Photons in each pixel of a frame, exposure and smear together.

    Parameters
    ----------
    rate : np.ndarray
        Photons per second reaching each pixel of one CCD, ``(n_rows, n_columns)``.
    exposure : u.Quantity
        Time between clearing the image area and starting the read-out.
    sequence : ReadoutSequence
        The clocking.

    Returns
    -------
    np.ndarray
        Photons per pixel, shaped ``(n_rows + parallel_overscan_rows, n_columns)``.
        Feed this to the detector stages in :mod:`euvst_response.radiometric` in
        place of the exposure-only photon count.
    """
    rate = np.asarray(rate, dtype=float)
    signal = smear_photons(rate, sequence)
    signal[:rate.shape[0]] += rate * u.Quantity(exposure).to_value(u.s)
    return signal


def dark_current_time(exposure: u.Quantity, sequence: ReadoutSequence,
                      n_rows: int) -> u.Quantity:
    """
    How long each row accumulates dark current, from the clear to its read-out.

    This is the exposure plus the time the row waits while the rows before it
    are read, and it applies with a shutter as well as without one: the chip is
    dark then, but not cold.  Rows read late in a windowed frame wait longest.
    """
    waiting = sequence.time_until_read(n_rows)
    return (u.Quantity(exposure).to_value(u.s) + waiting) * u.s
