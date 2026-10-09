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

No transfer is perfect: each leaves a little of a packet's charge behind, which
joins the packet that follows. The detector's ``cte_parallel`` and
``cte_serial`` say how little, and :func:`transfer_probabilities` where the
charge ends up.

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

import warnings
from dataclasses import dataclass, field, replace
from typing import List, Sequence, Tuple

import astropy.units as u
import numpy as np
from scipy.optimize import brentq
from scipy.signal import fftconvolve
from scipy.special import gammaln

from .config import Detector_SWC

# Wavelength in Angstrom as a function of position along the dispersion, in mm
# from the centre of the focal plane (the middle of the gap), positive toward
# longer wavelength.  Fitted to the positions RSC-2022021C gives for the two
# detector edges, the focal plane centre, the chief ray and the dispersion
# there, together with the ray-traced line positions drawn on its SW footprint
# figure.  A single linear scale is not good enough: the design runs from 17.0
# mA per row at the short-wavelength end to 16.8 at the long one, so the
# nominal 16.9 misplaces a line by up to about seven rows.
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

    The CCDs' format, pixel size, gap and plate scale are the camera's, and
    default to those of :class:`euvst_response.config.Detector_SWC`;
    :meth:`from_detector` takes them from a configured one.

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

    n_rows: int = Detector_SWC.n_rows
    n_columns: int = Detector_SWC.n_columns
    pixel_size: u.Quantity = (Detector_SWC.pix_size * u.pixel).to(u.micron)
    gap: u.Quantity = Detector_SWC.ccd_gap
    dispersion: Tuple[float, ...] = DISPERSION_COEFFICIENTS
    wavelength_offset: u.Quantity = 0.0 * u.Angstrom
    lit_band: Tuple[u.Quantity, u.Quantity] = (170.0 * u.Angstrom, 212.3 * u.Angstrom)
    plate_scale: u.Quantity = (Detector_SWC.plate_scale_angle * u.pixel).to(u.arcsec)
    slit_image_tilt: Tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_detector(cls, det: Detector_SWC, **settings) -> "FocalPlane_SWC":
        """
        The focal plane of the camera *det* describes: its CCDs' format, pixel
        size, gap and plate scale, with the rest as the defaults or as
        *settings* gives them.  The camera's own settings are refused here,
        so that they are set in one place, on the detector.
        """
        camera = dict(n_rows=det.n_rows, n_columns=det.n_columns,
                      pixel_size=(det.pix_size * u.pixel).to(u.micron), gap=det.ccd_gap,
                      plate_scale=(det.plate_scale_angle * u.pixel).to(u.arcsec))
        taken = sorted(set(settings) & set(camera))
        if taken:
            raise ValueError(f"{', '.join(taken)} come from the detector; set them there, as "
                             f"Detector_SWC.n_rows, n_columns, pix_size, ccd_gap and "
                             f"plate_scale_angle.")
        return cls(**camera, **settings)

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
        row, and the outer edges of the first and last rows are -0.5 and
        ``n_rows - 0.5``.  Raises if the wavelength falls in the gap or off the
        focal plane.  Give *column* to include the drift of a line along the
        slit.
        """
        target = u.Quantity(wavelength).to_value(u.Angstrom)
        for ccd in ("left", "right"):
            row = self._row_on(ccd, target, column)
            if -0.5 <= row <= self.n_rows - 0.5:
                return ccd, row
        ends = {ccd: self.row_edges(ccd, column)[[0, -1]] for ccd in ("left", "right")}
        raise ValueError(
            f"{u.Quantity(wavelength)} is not on either CCD. The left CCD covers "
            f"{ends['left'][0]:.4f} to {ends['left'][1]:.4f}, the right CCD "
            f"{ends['right'][1]:.4f} to {ends['right'][0]:.4f}, and the gap "
            f"between them is not recorded."
        )

    def _row_on(self, ccd: str, target: float, column=None) -> float:
        """
        The row of *ccd* at which *target*, in Angstrom, lands, counted as
        :meth:`row_of_wavelength` counts them: below -0.5 or above
        ``n_rows - 0.5``, and infinite if it is beyond every row, when it is
        off the CCD.
        """
        ends = (-0.5, self.n_rows - 0.5)

        def beyond(row):
            return self.wavelength(row, ccd, column).to_value(u.Angstrom) - target

        at_ends = [beyond(end) for end in ends]
        if at_ends[0] == 0.0:
            return ends[0]
        if at_ends[1] == 0.0:
            return ends[1]
        if np.sign(at_ends[0]) == np.sign(at_ends[1]):
            # Off the CCD, on the side of the end nearer to it in wavelength.
            return -np.inf if abs(at_ends[0]) < abs(at_ends[1]) else np.inf
        # The dispersion is monotonic across a CCD, so there is one root.
        return float(brentq(beyond, *ends, xtol=1e-12, rtol=4 * np.finfo(float).eps))

    def _extreme_columns(self) -> List[float]:
        """
        The columns at which a line lands furthest each way along the
        dispersion: the two ends of the slit and, when the curvature turns the
        drift back within them, the columns either side of the turn.  With no
        :attr:`slit_image_tilt` a line lands on the same row at all of them.
        """
        columns = [0.0, float(self.n_columns - 1)]
        linear, quadratic = self.slit_image_tilt
        if quadratic != 0.0:
            # line_shift is linear * f + quadratic * f**2 in the field angle f
            # in units of TILT_REFERENCE_ANGLE, which turns at -linear / 2 quadratic.
            turn = (-linear / (2 * quadratic) * TILT_REFERENCE_ANGLE
                    / self.plate_scale).to_value(u.dimensionless_unscaled) + (self.n_columns - 1) / 2
            columns += [float(c) for c in (np.floor(turn), np.ceil(turn)) if 0 <= c <= self.n_columns - 1]
        return columns

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

    The shutter, the timing and the register layout are the camera's, and
    default to those of :class:`euvst_response.config.Detector_SWC`;
    :meth:`from_detector` takes them from a configured one.  The windows and
    the clear belong to an observation, and are given here.

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
        The register is split, so each half carries half the columns, 1024
        image pixels, with the prescan at the outer end and the overscan at
        the middle.  Both scans are configurable in flight between 0 and 200.
        A row takes as long as the output with more pixels, so an odd number
        of columns counts the larger half.
    parallel_overscan_rows : int
        Rows clocked and read after the last image row.  Without a shutter these
        hold pure smear, since their charge crosses the whole illuminated image
        area on the way out.
    dump_rows : int
        Row transfers used to clear the image area before the exposure.  The
        default clears a whole CCD.  A frame here always starts from an empty
        chip, so fewer rows leave out the charge a partial clear leaves
        behind, which :func:`dark_current_time` warns of, and
        :func:`smear_photons` too without a shutter, with which there is no
        smear to leave out.
    windows : list of tuple of int
        Inclusive row ranges that are read out.  An empty list reads every row,
        which is the slowest case.
    cte_parallel, cte_serial : float
        The charge transfer efficiency of one row transfer and of one transfer
        along the serial register: the fraction of a packet's charge that moves
        with it, the rest joining the packet behind.  1 is a perfect transfer.
    """

    shutter: bool = Detector_SWC.shutter
    row_transfer_time: u.Quantity = Detector_SWC.row_transfer_time
    pixel_period: u.Quantity = Detector_SWC.pixel_period
    serial_prescan: int = Detector_SWC.serial_prescan
    serial_image_pixels: int = (Detector_SWC.n_columns + 1) // 2
    serial_overscan: int = Detector_SWC.serial_overscan
    parallel_overscan_rows: int = Detector_SWC.parallel_overscan_rows
    dump_rows: int = Detector_SWC.n_rows
    windows: List[Tuple[int, int]] = field(default_factory=list)
    cte_parallel: float = Detector_SWC.cte_parallel
    cte_serial: float = Detector_SWC.cte_serial

    @classmethod
    def from_detector(cls, det: Detector_SWC, *, windows: Sequence[Tuple[int, int]] = (),
                      dump_rows: int | None = None, shutter: bool | None = None,
                      ) -> "ReadoutSequence":
        """
        The read-out of the camera *det* describes, for one observation.

        Parameters
        ----------
        det : Detector_SWC
            The camera, whose shutter, timing and register layout are used.
        windows : sequence of tuple of int, optional
            The row ranges read out, as for the constructor.  Default every row.
        dump_rows : int, optional
            The row transfers of the clear.  Default a whole CCD.
        shutter : bool, optional
            Whether the frame is taken with the shutter.  Default as the
            camera is configured.

        The charge transfer efficiencies are the camera's too.
        """
        return cls(shutter=det.shutter if shutter is None else shutter,
                   row_transfer_time=det.row_transfer_time, pixel_period=det.pixel_period,
                   serial_prescan=det.serial_prescan,
                   serial_image_pixels=(det.n_columns + 1) // 2,
                   serial_overscan=det.serial_overscan,
                   parallel_overscan_rows=det.parallel_overscan_rows,
                   dump_rows=det.n_rows if dump_rows is None else dump_rows,
                   windows=list(windows), cte_parallel=det.cte_parallel,
                   cte_serial=det.cte_serial)

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
        for name in ("cte_parallel", "cte_serial"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(
                    f"{name} is the fraction of a packet's charge one transfer moves, above "
                    f"0 and at most 1, got {value}."
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


def windows_from_wavelengths(focal_plane: FocalPlane_SWC,
                             ranges: Sequence[Tuple[u.Quantity, u.Quantity]],
                             ) -> List[Tuple[int, int]]:
    """
    Turn wavelength ranges into the row ranges a read-out window covers.

    A window holds every row any part of the range lands on, at any column:
    with a :attr:`~FocalPlane_SWC.slit_image_tilt` a line drifts along the
    dispersion from one end of the slit to the other, and the window follows
    it.  A row the range only touches at its edge is left out.

    Both CCDs share one row timeline, so the returned ranges are row numbers
    without a CCD attached: a window over rows 1850 to 1880 makes those rows
    slow on both devices, whatever wavelength they are on the other one.  A
    range that crosses the gap runs from each of its ends to the butted edge,
    so its window goes on to the last row.  The parts of a range in the gap or
    beyond the ends of the focal plane are not recorded and need no rows.
    """
    n_rows = focal_plane.n_rows
    windows = []
    for low, high in ranges:
        targets = [u.Quantity(w).to_value(u.Angstrom) for w in (low, high)]
        first, last = np.inf, -np.inf
        for column in focal_plane._extreme_columns():
            for ccd in ("left", "right"):
                rows = [focal_plane._row_on(ccd, target, column) for target in targets]
                # The rows the range covers on this CCD, cut to its edges. A
                # range that reaches the CCD only at its outer edge is not on
                # it; one of no width is on the row below the boundary it is
                # on, as between two rows, which the lower edge has none of.
                lo, hi = min(rows), max(rows)
                start, end = max(lo, -0.5), min(hi, n_rows - 0.5)
                if start < end or (lo == hi and -0.5 < lo <= n_rows - 0.5):
                    first, last = min(first, start), max(last, end)
        if first > last:
            raise ValueError(
                f"No part of {u.Quantity(low)} to {u.Quantity(high)} is on either CCD.")
        # The rows whose span, from half a row below their centre to half a
        # row above, overlaps the range; a range of no width is on one row.
        top = int(np.ceil(last + 0.5)) - 1
        windows.append((min(int(np.floor(first - 0.5)) + 1, top), top))
    return windows


def _warn_of_a_partial_clear(sequence: ReadoutSequence, n_rows: int, what: str) -> None:
    """Say that a clear of fewer rows than the CCD has leaves out the charge it does not clear."""
    if sequence.dump_rows < n_rows:
        warnings.warn(
            f"dump_rows is {sequence.dump_rows}, fewer than the {n_rows} rows of the CCD, so "
            f"the clear leaves charge from before it in the image area. The {what} here "
            f"starts from an empty chip and leaves that charge out.", UserWarning, stacklevel=3)


# What is left over of a charge past the last of the terms that
# transfer_probabilities gives, at most, as a fraction of it.
_TAIL = 1e-15


def transfer_probabilities(transfers, cte: float, most: int | None = None) -> np.ndarray:
    """
    Where charge arrives after transfers that each leave some of it behind.

    Each transfer moves a fraction ``cte`` of a packet's charge and leaves the
    rest in the well it vacated, where the packet behind picks it up.  Every
    electron is left behind independently, so charge with ``m`` transfers to
    go arrives ``k`` packets late with the negative binomial probability
    ``C(m - 1 + k, k) * cte**m * (1 - cte)**k``: it moves ``m`` times and is
    held back ``k`` times on the way, each time joining a packet that has one
    transfer more to go.

    Parameters
    ----------
    transfers : array_like of int
        The transfers each charge has to go, 1 or more.
    cte : float
        The fraction of a packet's charge one transfer moves.
    most : int, optional
        The latest arrival wanted, past which the charge would leave the frame
        anyway.  Default no limit.

    Returns
    -------
    np.ndarray
        Shaped ``(n_terms, *transfers.shape)``: element ``[k, ...]`` is the
        fraction arriving ``k`` packets late.  The terms run until less than
        1e-15 of every charge is left over, or to *most*.  A perfect transfer
        has one term, of ones.
    """
    m = np.asarray(transfers, dtype=float)
    if np.any(m < 1):
        raise ValueError("A charge has at least one transfer to go.")
    if not 0 < cte <= 1:
        raise ValueError(f"cte is a fraction above 0 and at most 1, got {cte}.")
    if cte == 1:
        return np.ones((1,) + m.shape)
    log_kept, log_left = np.log(cte), np.log1p(-cte)
    terms = []
    k = 0
    while True:
        terms.append(np.exp(gammaln(m + k) - gammaln(k + 1) - gammaln(m)
                            + m * log_kept + k * log_left))
        # Past the most likely arrival the terms fall by at least this ratio
        # from one to the next, so what is left is below a geometric series.
        ratio = (m + k) / (k + 1) * (1.0 - cte)
        with np.errstate(divide="ignore"):
            left = np.where(ratio < 1, terms[-1] * ratio / (1.0 - ratio), np.inf)
        if np.all(left < _TAIL) or (most is not None and k >= most):
            return np.array(terms)
        k += 1


def _check_rate(rate: np.ndarray) -> np.ndarray:
    rate = np.asarray(rate, dtype=float)
    if rate.ndim != 2:
        raise ValueError(
            f"rate is one CCD's pixels, shaped (n_rows, n_columns), got shape "
            f"{rate.shape}."
        )
    return rate


def _cleared(rate: np.ndarray, sequence: ReadoutSequence, packets: np.ndarray) -> np.ndarray:
    """
    What each of *packets* collects while the image area is cleared, in units
    of *rate* times seconds.  The packet that ends the clear at row ``r`` has
    come down from row ``r + dump_rows``, collecting one row transfer at each
    row on the way, ``r + 1`` to ``r + dump_rows``.  Rows above the image area
    are empty, so the sum stops there; a packet with ``r`` below zero went into
    the dump drain before the clear ended.
    """
    n_rows = rate.shape[0]
    cleared = np.zeros((packets.size, rate.shape[1]))
    if not sequence.dump_rows:
        return cleared
    above = np.zeros((n_rows + 1, rate.shape[1]))
    above[:-1] = np.cumsum(rate[::-1], axis=0)[::-1]
    first = np.clip(packets + 1, 0, n_rows)
    last = np.clip(packets + sequence.dump_rows + 1, 0, n_rows)
    some = first < last
    cleared[some] = ((above[first[some]] - above[last[some]])
                     * sequence.row_transfer_time.to_value(u.s))
    return cleared


def _along_columns(rate: np.ndarray, exposure, sequence: ReadoutSequence,
                   power: int = 1) -> np.ndarray:
    """
    What each packet of a frame brings to the serial register, of a quantity
    every pixel's light adds to per second: during the exposure if there is
    one (*exposure* None leaves it out), and without a shutter while the image
    area is cleared and read.

    Without a shutter, the packet read out of row ``r`` adds
    ``sum_k rate[r - k] * dwell[k - 1]`` over ``k >= 1`` while the rows before
    it are read, and :func:`_cleared` during the clear.  Each addition, at row
    ``x``, then has ``x + 1`` row transfers to go, and arrives in the packet
    :func:`transfer_probabilities` says, so it is weighted by that
    probability, to the power *power*.  Packets that went into the dump drain
    during the clear leave some of their charge behind too, which arrives in
    the first rows read.
    """
    n_rows, n_columns = rate.shape
    total_rows = n_rows + sequence.parallel_overscan_rows
    probability = transfer_probabilities(np.arange(1, n_rows + 1), sequence.cte_parallel,
                                         most=total_rows) ** power
    lit = not sequence.shutter
    if lit:
        kernel = np.concatenate([[0.0], sequence.dwell(n_rows)])
        # The FFT leaves rounding noise, of order 1e-16 of the brightest row, in
        # packets no light reached, which would give them a mean photon energy
        # made of noise.  Those are the packets whose path crossed no lit
        # pixel: a count of lit pixels, which the same convolution gives to far
        # better than a half.
        crossed = fftconvolve((rate > 0).astype(float),
                              (kernel > 0).astype(float)[:, np.newaxis],
                              mode="full", axes=0)[:total_rows]
    arrived = np.zeros((total_rows, n_columns))
    for k, weight in enumerate(probability):
        weighted = rate * weight[:, np.newaxis]
        # Indexed by where the charge arrives, k packets after the one it was
        # collected in, so the first k are packets dumped during the clear.
        packets = np.zeros((total_rows, n_columns))
        if lit:
            # The read-out.  The packet leaving row r is at row r - k while image
            # row k - 1 is in the register, so this is a convolution of the
            # rate with the dwell times, and the parallel overscan rows are the
            # terms past the last image row.
            reading = fftconvolve(weighted, kernel[:, np.newaxis], mode="full",
                                  axes=0)[:total_rows]
            reading[crossed < 0.5] = 0.0
            packets[k:] += reading[:total_rows - k]
            packets += _cleared(weighted, sequence, np.arange(total_rows) - k)
            # The sums cannot be negative, but the convolution is done by FFT
            # and leaves rounding noise of order 1e-20 where the answer is
            # zero, which would otherwise reach the Poisson draw downstream as
            # a negative mean.
            np.maximum(packets, 0.0, out=packets)
        if exposure is not None:
            # What image row r collects in the exposure arrives in packet r + k,
            # which for the last rows is in the parallel overscan.
            end = min(n_rows + k, total_rows)
            packets[k:end] += (weighted * u.Quantity(exposure).to_value(u.s))[:end - k]
        arrived += packets
    return arrived


def _along_register(frame: np.ndarray, sequence: ReadoutSequence, power: int = 1) -> np.ndarray:
    """
    The charge each pixel of a frame brings to its output, through the half
    of the serial register it is read along, weighted by the probability of
    arriving there to the power *power*.  A pixel ``j`` image pixels from the
    end of its half has ``serial_prescan + j + 1`` transfers to go, and what it
    leaves behind arrives in the pixels read after it, towards the middle of
    the register; past the last image pixel it goes into the overscan.
    """
    if sequence.cte_serial == 1:
        return frame
    n_columns = frame.shape[1]
    half = sequence.serial_image_pixels
    if n_columns not in (2 * half, 2 * half - 1):
        raise ValueError(
            f"A transfer along the serial register needs the frame's every column, "
            f"{2 * half} for two halves of {half}, to know how far each is from its "
            f"output; got {n_columns}."
        )
    arrived = np.zeros_like(frame)
    # Each half in the order its output reads it: the first from column 0, the
    # second from the last column.
    for columns in (np.arange(half), np.arange(n_columns - 1, half - 1, -1)):
        charge = frame[:, columns]
        probability = transfer_probabilities(sequence.serial_prescan + 1 + np.arange(columns.size),
                                             sequence.cte_serial, most=columns.size) ** power
        reached = np.zeros_like(charge)
        for k, weight in enumerate(probability[:columns.size]):
            reached[:, k:] += (charge * weight)[:, :columns.size - k]
        arrived[:, columns] = reached
    return arrived


def smear_photons(rate: np.ndarray, sequence: ReadoutSequence) -> np.ndarray:
    """
    Photons a shutterless frame collects outside its exposure.

    Charge is clocked toward the register while the chip is still illuminated,
    so a packet collects light from every row it crosses.  For the packet read
    out of row ``r``, the read-out contributes ``sum_k rate[r - k] * dwell[k - 1]``
    over ``k >= 1``, and clearing the image area beforehand contributes
    ``row_transfer_time`` times the rows above it, since the packet that ends
    at row ``r`` was clocked down from ``r + dump_rows``.  With a transfer
    efficiency below 1, some of what each packet collects arrives in the
    packets after it, as :func:`transfer_probabilities` says, along the
    columns and then along the serial register.

    Parameters
    ----------
    rate : np.ndarray
        Photons per second reaching each pixel of one CCD, shaped
        ``(n_rows, n_columns)`` with row 0 at the serial register.  With
        ``cte_serial`` below 1, ``n_columns`` must be the CCD's, so that each
        column's distance from its output is known.
    sequence : ReadoutSequence
        The clocking.  With ``shutter`` True this returns zeros.

    Returns
    -------
    np.ndarray
        Photons collected outside the exposure, shaped
        ``(n_rows + parallel_overscan_rows, n_columns)``.  The extra rows are
        the parallel overscan, which sees nothing but smear.
    """
    rate = _check_rate(rate)
    n_rows, _ = rate.shape
    total_rows = n_rows + sequence.parallel_overscan_rows
    if sequence.shutter:
        return np.zeros((total_rows, rate.shape[1]))
    _warn_of_a_partial_clear(sequence, n_rows, "smear")
    return _along_register(_along_columns(rate, None, sequence), sequence)


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
        The mean photons per pixel, shaped
        ``(n_rows + parallel_overscan_rows, n_columns)``. Without a shutter a
        pixel holds photons from every row its charge crossed, so for a frame
        :func:`euvst_response.frame.expose_with_wavelength` gives these with
        the wavelength that carries each pixel's mean photon energy. Drawn as
        whole photons, a Poisson draw of each pixel, they go to
        :func:`euvst_response.frame.detect` with that wavelength and
        :func:`dark_current_time`; or the means go with ``noise=False``, for
        the frame's mean.

        With a transfer efficiency below 1 these are the photons whose charge
        arrives in each pixel, on average.  The transfer is not drawn: a frame
        drawn from them has the right mean, and its noise leaves out how the
        charge left behind varies, which :func:`expose_variance` includes.
    """
    rate = _check_rate(rate)
    if not sequence.shutter:
        _warn_of_a_partial_clear(sequence, rate.shape[0], "smear")
    return _along_register(_along_columns(rate, exposure, sequence), sequence)


def expose_variance(variance_rate: np.ndarray, mean_rate: np.ndarray, exposure: u.Quantity,
                    sequence: ReadoutSequence) -> np.ndarray:
    """
    The variance of the charge in each pixel of a frame, exposure and smear
    together, from what each pixel's light frees per second.

    Every electron is left behind independently at each transfer, so a pixel
    that receives a fraction ``P`` of a charge ``Q`` gets a variance of
    ``P**2 * var(Q) + P * (1 - P) * mean(Q)``.  Summed over all that the frame
    collects, with ``P`` from :func:`transfer_probabilities`, that is the
    transfer of ``variance_rate - mean_rate`` weighted by ``P**2`` and of
    ``mean_rate`` weighted by ``P``.  With a perfect transfer it is
    :func:`expose` of *variance_rate*.

    Parameters
    ----------
    variance_rate, mean_rate : np.ndarray
        The variance and the mean of the charge each pixel's light frees per
        second, ``(n_rows, n_columns)``, in electrons squared and electrons.
        The variance is at least the mean, as it is for charge that photons
        free, which comes in whole photons' worth.
    exposure, sequence
        As for :func:`expose`.

    Returns
    -------
    np.ndarray
        The variance, shaped ``(n_rows + parallel_overscan_rows, n_columns)``.
    """
    variance_rate, mean_rate = _check_rate(variance_rate), _check_rate(mean_rate)
    if variance_rate.shape != mean_rate.shape:
        raise ValueError(
            f"variance_rate and mean_rate describe the same pixels, got shapes "
            f"{variance_rate.shape} and {mean_rate.shape}."
        )
    if sequence.cte_parallel == 1 and sequence.cte_serial == 1:
        return expose(variance_rate, exposure, sequence)
    if not sequence.shutter:
        _warn_of_a_partial_clear(sequence, variance_rate.shape[0], "smear")
    excess = variance_rate - mean_rate
    if np.any(excess < -1e-12 * np.abs(variance_rate).max()):
        raise ValueError("variance_rate must be at least mean_rate in every pixel.")
    return (_along_register(_along_columns(np.maximum(excess, 0.0), exposure, sequence, 2),
                            sequence, 2)
            + _along_register(_along_columns(mean_rate, exposure, sequence), sequence))


def dark_current_time(exposure: u.Quantity, sequence: ReadoutSequence,
                      n_rows: int) -> u.Quantity:
    """
    How long each packet of a frame collects dark current, for the image rows
    and then the parallel overscan rows.

    A packet collects dark current in whichever pixel it sits, so this is the
    time it spends in the image area, the same time :func:`smear_photons`
    gives it light for.  An image row's packet is clocked into place during
    the clear, sits through the exposure, and waits while the rows before it
    are read; a parallel overscan packet only crosses the image area during
    the read-out.  None of that needs light, so it is the same with a shutter
    as without one.  The serial register's own dark current is not included.

    With ``cte_parallel`` below 1, dark charge is left behind and picked up
    like any other, and this is the time each packet's dark charge is worth,
    the dark current collected in it weighted as :func:`expose` weights light.
    Along the serial register that is not followed: for a dark current the
    same in every column it changes only the first pixels each output reads,
    by less than ``(serial_prescan + 1) * (1 - cte_serial)`` of their dark
    charge.
    """
    _warn_of_a_partial_clear(sequence, n_rows, "dark current")
    if sequence.cte_parallel != 1:
        unlit = replace(sequence, shutter=False)
        return _along_columns(np.ones((n_rows, 1)), exposure, unlit)[:, 0] * u.s
    dwell = sequence.dwell(n_rows)
    packets = np.arange(n_rows + sequence.parallel_overscan_rows)
    image = packets < n_rows
    # The read-out: every dwell before the packet's own, back to the one in
    # which it came in at the top of the image area.
    elapsed = np.concatenate([[0.0], np.cumsum(dwell)])
    reading = elapsed[packets] - elapsed[np.maximum(packets - n_rows, 0)]
    # The clear: one row transfer at each row above its own on the way down,
    # as far as the clear reaches.
    above = np.where(image, n_rows - 1 - packets, 0)
    clear = np.minimum(sequence.dump_rows, above) * sequence.row_transfer_time.to_value(u.s)
    exposed = np.where(image, u.Quantity(exposure).to_value(u.s), 0.0)
    return (clear + exposed + reading) * u.s
