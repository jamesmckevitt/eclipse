"""
Read-out of the SW CCDs, with and without a shutter.

The SW focal plane is two Teledyne e2v CCD42-40s, 2048 x 2048 pixels of
13.5 micron, butted along the dispersion with a 1 mm gap. Each serial
register is on an outer edge, so a CCD row is one wavelength and row 0 is
the row next to the register.

A frame is a dump of the image area, the exposure, and a row-by-row
read-out, in which rows outside the read-out windows are dumped
(SOLC-EUVST-MSSL-ICD-0003 v3.1 {SWI-IRD}-032). Without a shutter, each
charge packet also collects light from every row it crosses on its way to
the register (:func:`smear_photons`).

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
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import astropy.units as u
import numpy as np
from scipy.optimize import brentq
from scipy.signal import fftconvolve

from .config import Detector_SWC

# Wavelength in Angstrom against position along the dispersion, in mm from the
# focal plane centre (the middle of the gap), positive toward longer
# wavelength. Fitted to the RSC-2022021C positions of the two detector edges,
# the focal plane centre, the chief ray and the dispersion there, and to the
# ray-traced line positions on its SW footprint figure. A linear scale is not
# good enough: the design runs from 17.0 mA per row at the short-wavelength
# end to 16.8 at the long end, so the nominal 16.9 misplaces a line by up to
# about seven rows.
DISPERSION_COEFFICIENTS = (198.885944, 1.2516577, -1.361114e-4)

# A spectral line is an image of the slit, and the optics do not put it on
# exactly one row: it drifts along the dispersion from one end of the slit to
# the other. These are the row offsets at 140 arcsec from the field centre,
# linear term then quadratic, measured from the ray-traced line positions on
# the SW footprint figure of RSC-2022021C. All eleven lines drawn there agree
# on them to a few tenths of a row. Pass them to FocalPlane_SWC to model the
# drift; by default a line lands on one row along the whole slit.
MEASURED_SLIT_IMAGE_TILT = (0.991, 0.605)

# The field angle those offsets are quoted at.
TILT_REFERENCE_ANGLE = 140.0 * u.arcsec


@dataclass
class FocalPlane_SWC:
    """
    Geometry of the SW focal plane, and the wavelength each row records.

    The CCD format, pixel size, gap and plate scale belong to the camera, and
    default to those of :class:`euvst_response.config.Detector_SWC`;
    :meth:`from_detector` takes them from a configured detector.

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
        for a position in mm. See :data:`DISPERSION_COEFFICIENTS`.
    wavelength_offset : u.Quantity
        Added to every row's wavelength. The design positions are good to
        about a row, but the camera's alignment to the beam is quoted at
        +/-0.79 Angstrom, some 47 rows (SOLC-EUVST-MSSL-RP-0007 v1.2 p45), so
        an as-built or in-flight wavelength calibration goes here.
    lit_band : tuple of u.Quantity
        Wavelengths between which light reaches the detectors; outside them a
        baffle vignettes the beam. Rows outside this band record nothing, but
        charge still crosses them on its way out.
    plate_scale : u.Quantity
        Angle a pixel covers along the slit.
    slit_image_tilt : tuple of float
        Drift of a line along the dispersion between the field centre and
        140 arcsec from it, in rows: a linear and a quadratic term. The
        default `(0.0, 0.0)` puts each line on one row along the whole slit,
        and the methods then ignore their `column` argument. Pass
        :data:`MEASURED_SLIT_IMAGE_TILT` to model the drift in the optical
        design, about two rows end to end. Which end of the slit it runs
        toward is not documented, so the sign may need flipping against a
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
        Focal plane of the camera `det`.

        The CCD format, pixel size, gap and plate scale come from `det`, and
        the rest are the defaults or given in `settings`. Camera settings in
        `settings` are refused, so they are set in one place, the detector.
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
        Position along the dispersion of the centre of `row`, in mm from the
        focal plane centre. Rows count from the serial register on the outer
        edge of each CCD, so the position moves inward as the row number
        rises.
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
        Shift of a line from where it lands at the field centre, in mm along
        the dispersion, positive toward longer wavelength. Zero everywhere
        unless :attr:`slit_image_tilt` is non-zero.
        """
        linear, quadratic = self.slit_image_tilt
        if column is None or (linear == 0.0 and quadratic == 0.0):
            return 0.0 * u.mm
        f = (self.field_angle(column) / TILT_REFERENCE_ANGLE).to_value(u.dimensionless_unscaled)
        return (linear * f + quadratic * f**2) * self.pixel_size.to(u.mm)

    def wavelength(self, row, ccd: str, column=None) -> u.Quantity:
        """
        Wavelength recorded by the centre of `row`.

        Give `column` to include the drift of a line along the slit; without
        :attr:`slit_image_tilt` it makes no difference.
        """
        s = (self.position(row, ccd) - self.line_shift(column)).to_value(u.mm)
        lam = np.polynomial.polynomial.polyval(s, self.dispersion) * u.Angstrom
        return lam + self.wavelength_offset

    def row_edges(self, ccd: str, column=None) -> u.Quantity:
        """
        Wavelengths at the `n_rows + 1` boundaries between rows.

        Binning a spectrum onto the detector conserves flux when it is
        integrated between these, rather than sampled at the row centres.
        They decrease with row number on the right CCD, whose register is at
        the long-wavelength end.
        """
        s = (self.position(np.arange(self.n_rows + 1) - 0.5, ccd)
             - self.line_shift(column)).to_value(u.mm)
        lam = np.polynomial.polynomial.polyval(s, self.dispersion) * u.Angstrom
        return lam + self.wavelength_offset

    def row_of_wavelength(self, wavelength: u.Quantity, column=None) -> Tuple[str, float]:
        """
        Which CCD records a wavelength, and at which row.

        The row is a float: an integer is the centre of that row, and the
        outer edges of the first and last rows are -0.5 and `n_rows - 0.5`.
        Raises ValueError if the wavelength falls in the gap or off the focal
        plane. Give `column` to include the drift of a line along the slit.
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
        Row of `ccd` where `target`, in Angstrom, lands, counted as in
        :meth:`row_of_wavelength`. Off the CCD it is below -0.5 or above
        `n_rows - 0.5`, and infinite if beyond every row.
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
        Columns where a line lands furthest each way along the dispersion.

        These are the two ends of the slit and, when the curvature turns the
        drift back between them, the columns either side of the turn. With
        no :attr:`slit_image_tilt` a line lands on the same row at all of
        them.
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
        First and last row of `ccd` whose centre receives light, inclusive.

        The edge is taken as sharp at :attr:`lit_band`. The documents put it
        within a few rows of there but do not dimension the baffle, so a row
        or two either side of these values is a modelling choice, not a
        measurement.
        Rows outside the range still pass charge; on the right CCD some 1290
        of them lie between the band and the register.
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

    Both CCDs are clocked together, and a row wanted on either is read on
    both, so one row timeline covers the pair
    (SOLC-EUVST-MSSL-RS-0002 v2.0 SolC-FPGA-RS-033).

    The shutter, timing and register layout belong to the camera, and
    default to those of :class:`euvst_response.config.Detector_SWC`;
    :meth:`from_detector` takes them from a configured detector. The windows
    and the clear belong to an observation, and are given here.

    Parameters
    ----------
    shutter : bool
        True keeps the chip dark outside the exposure, as a mechanical
        shutter does. False leaves it illuminated while the image area is
        cleared and read, which smears the frame.
    row_transfer_time : u.Quantity
        Time to move every row one step toward the register. Configurable in
        flight; 15 us is the value in the SWC modes table and in the read-out
        times of RP-0007 v1.2 Table 5-11.
    pixel_period : u.Quantity
        Time to move one pixel along the serial register and digitise it.
    serial_prescan, serial_image_pixels, serial_overscan : int
        Samples read per output for each row that goes through the register.
        The register is split, so each half carries half the columns, 1024
        image pixels, with the prescan at the outer end and the overscan at
        the middle. Both scans are configurable in flight from 0 to 200. A
        row takes as long as the output with more pixels, so for an odd
        number of columns the larger half counts.
    parallel_overscan_rows : int
        Rows clocked and read after the last image row. Without a shutter
        they hold pure smear, as their charge crosses the whole illuminated
        image area on the way out.
    dump_rows : int
        Row transfers used to clear the image area before the exposure.
        Default a whole CCD. A frame here always starts from an empty chip,
        so with fewer rows the charge a partial clear leaves behind is not
        modelled. :func:`dark_current_time` warns of this, and so does
        :func:`smear_photons` without a shutter (with one there is no smear
        to leave out).
    windows : list of tuple of int
        Inclusive row ranges that are read out. An empty list reads every
        row, the slowest case.
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

    @classmethod
    def from_detector(cls, det: Detector_SWC, *, windows: Sequence[Tuple[int, int]] = (),
                      dump_rows: int | None = None, shutter: bool | None = None,
                      ) -> "ReadoutSequence":
        """
        Read-out of the camera `det` for one observation.

        Parameters
        ----------
        det : Detector_SWC
            Camera, whose shutter, timing and register layout are used.
        windows : sequence of tuple of int, optional
            Row ranges read out, as for the constructor. Default every row.
        dump_rows : int, optional
            Row transfers of the clear. Default a whole CCD.
        shutter : bool, optional
            Whether the frame is taken with the shutter. Default as the
            camera is configured.
        """
        return cls(shutter=det.shutter if shutter is None else shutter,
                   row_transfer_time=det.row_transfer_time, pixel_period=det.pixel_period,
                   serial_prescan=det.serial_prescan,
                   serial_image_pixels=(det.n_columns + 1) // 2,
                   serial_overscan=det.serial_overscan,
                   parallel_overscan_rows=det.parallel_overscan_rows,
                   dump_rows=det.n_rows if dump_rows is None else dump_rows,
                   windows=list(windows))

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
        Which rows go through the serial register.

        A boolean array over the image rows and then the parallel overscan
        rows. The overscan is always read; with no windows, so is every
        other row.
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
        Time in seconds between successive row transfers of a read-out.

        Element `k` is the interval after transfer `k + 1`, while image row
        `k` is in the serial register being read or dumped. Every charge
        packet in the image area is stationary for that whole interval,
        which is why the smear depends on the window layout and not only on
        the number of rows.
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
    Convert wavelength ranges to the row ranges of read-out windows.

    A window holds every row that any part of the range lands on, at any
    column. With a :attr:`~FocalPlane_SWC.slit_image_tilt` a line drifts
    along the dispersion from one end of the slit to the other, and the
    window follows it. A row the range only touches at its edge is left
    out.

    Both CCDs share one row timeline, so the returned ranges are row numbers
    with no CCD attached: a window over rows 1850 to 1880 makes those rows
    slow on both devices, though they record a different wavelength on the
    other one. A range that crosses the gap runs from each of its ends to
    the butted edge, so its window extends to the last row. Parts of a range
    in the gap or beyond the ends of the focal plane are not recorded and
    need no rows.
    """
    n_rows = focal_plane.n_rows
    windows = []
    for low, high in ranges:
        targets = [u.Quantity(w).to_value(u.Angstrom) for w in (low, high)]
        first, last = np.inf, -np.inf
        for column in focal_plane._extreme_columns():
            for ccd in ("left", "right"):
                rows = [focal_plane._row_on(ccd, target, column) for target in targets]
                # Rows the range covers on this CCD, cut to its edges. A range
                # that reaches the CCD only at its outer edge is not on it. A
                # range of no width on a row boundary is on the row below, so
                # at the lower edge, with no row below, it is not on the CCD.
                lo, hi = min(rows), max(rows)
                start, end = max(lo, -0.5), min(hi, n_rows - 0.5)
                if start < end or (lo == hi and -0.5 < lo <= n_rows - 0.5):
                    first, last = min(first, start), max(last, end)
        if first > last:
            raise ValueError(
                f"No part of {u.Quantity(low)} to {u.Quantity(high)} is on either CCD.")
        # Rows whose span, half a row either side of the centre, overlaps the
        # range; a range of no width is on one row.
        top = int(np.ceil(last + 0.5)) - 1
        windows.append((min(int(np.floor(first - 0.5)) + 1, top), top))
    return windows


def _warn_of_a_partial_clear(sequence: ReadoutSequence, n_rows: int, what: str) -> None:
    """Warn that the charge left by a clear of fewer rows than the CCD has is not modelled."""
    if sequence.dump_rows < n_rows:
        warnings.warn(
            f"dump_rows is {sequence.dump_rows}, fewer than the {n_rows} rows of the CCD, so "
            f"the clear leaves charge from before it in the image area. The {what} here "
            f"starts from an empty chip and leaves that charge out.", UserWarning, stacklevel=3)


def smear_photons(rate: np.ndarray, sequence: ReadoutSequence) -> np.ndarray:
    """
    Photons a shutterless frame collects outside its exposure.

    Charge is clocked toward the register while the chip is still lit, so a
    packet collects light from every row it crosses. For the packet read out
    of row `r`, the read-out adds `sum_k rate[r - k] * dwell[k - 1]` over
    `k >= 1`. Clearing the image area beforehand adds `row_transfer_time`
    times the rows above it, since the packet that ends at row `r` was
    clocked down from `r + dump_rows`.

    Parameters
    ----------
    rate : np.ndarray
        Photons per second reaching each pixel of one CCD, shaped
        `(n_rows, n_columns)` with row 0 at the serial register.
    sequence : ReadoutSequence
        Clocking. With `shutter` True this returns zeros.

    Returns
    -------
    np.ndarray
        Photons collected outside the exposure, shaped
        `(n_rows + parallel_overscan_rows, n_columns)`. The extra rows are
        the parallel overscan, which sees only smear.
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
    _warn_of_a_partial_clear(sequence, n_rows, "smear")

    # Read-out. The packet leaving row r is at row r - k while image row k - 1
    # is in the register, so the smear is the rate convolved with the dwell
    # times. The parallel overscan rows are the terms past the last image row.
    kernel = np.concatenate([[0.0], sequence.dwell(n_rows)])
    smear = fftconvolve(rate, kernel[:, np.newaxis], mode="full", axes=0)[:total_rows]
    # The FFT leaves rounding noise, of order 1e-16 of the brightest row, in
    # packets no light reached, which would give them a mean photon energy
    # made of noise. Zero the packets whose path crossed no lit pixel, found
    # by convolving a count of lit pixels, which is good to far better than
    # a half.
    crossed = fftconvolve((rate > 0).astype(float), (kernel > 0).astype(float)[:, np.newaxis],
                          mode="full", axes=0)[:total_rows]
    smear[crossed < 0.5] = 0.0

    # Clearing the image area. The packet that ends the clear at row r came
    # down from row r + dump_rows, collecting light for one row transfer at
    # each row on the way. Rows above the image area are empty, so the sum
    # stops there.
    if sequence.dump_rows:
        above = np.zeros((n_rows + 1, rate.shape[1]))
        above[:-1] = np.cumsum(rate[::-1], axis=0)[::-1]
        first = np.arange(1, n_rows + 1)
        last = np.minimum(np.arange(n_rows) + sequence.dump_rows + 1, n_rows)
        smear[:n_rows] += (above[first] - above[last]) * sequence.row_transfer_time.to_value(u.s)

    # The sum cannot be negative, but the FFT convolution leaves rounding noise
    # of order 1e-20 where it is zero, which would reach the Poisson draw
    # downstream as a negative mean.
    return np.maximum(smear, 0.0)


def expose(rate: np.ndarray, exposure: u.Quantity, sequence: ReadoutSequence) -> np.ndarray:
    """
    Photons in each pixel of a frame, exposure and smear together.

    Parameters
    ----------
    rate : np.ndarray
        Photons per second reaching each pixel of one CCD, `(n_rows, n_columns)`.
    exposure : u.Quantity
        Time between clearing the image area and starting the read-out.
    sequence : ReadoutSequence
        Clocking.

    Returns
    -------
    np.ndarray
        Mean photons per pixel, shaped
        `(n_rows + parallel_overscan_rows, n_columns)`. Without a shutter a
        pixel holds photons from every row its charge crossed, so for a frame
        :func:`euvst_response.frame.expose_with_wavelengths` gives these with
        the wavelengths of each pixel's mean and root mean square photon
        energies. A Poisson draw of whole photons in each pixel goes to
        :func:`euvst_response.frame.detect` with those wavelengths and
        :func:`dark_current_time`. For the frame's mean, pass the means
        instead, with `noise=False`.
    """
    rate = np.asarray(rate, dtype=float)
    signal = smear_photons(rate, sequence)
    signal[:rate.shape[0]] += rate * u.Quantity(exposure).to_value(u.s)
    return signal


def dark_current_time(exposure: u.Quantity, sequence: ReadoutSequence,
                      n_rows: int) -> u.Quantity:
    """
    How long each packet of a frame collects dark current, for the image rows
    and then the parallel overscan rows.

    A packet collects dark current in any pixel it sits in, so this is its
    time in the image area, the same time :func:`smear_photons` gives it
    light for. An image row's packet is clocked into place during the clear,
    sits through the exposure, and waits while the rows before it are read.
    A parallel overscan packet only crosses the image area during the
    read-out. None of this needs light, so it is the same with a shutter as
    without one. The serial register's own dark current is not included.
    """
    _warn_of_a_partial_clear(sequence, n_rows, "dark current")
    dwell = sequence.dwell(n_rows)
    packets = np.arange(n_rows + sequence.parallel_overscan_rows)
    image = packets < n_rows
    # Read-out: every dwell before the packet's own, back to the one in which
    # it entered the top of the image area.
    elapsed = np.concatenate([[0.0], np.cumsum(dwell)])
    reading = elapsed[packets] - elapsed[np.maximum(packets - n_rows, 0)]
    # Clear: one row transfer at each row above its own, as far as the clear
    # reaches.
    above = np.where(image, n_rows - 1 - packets, 0)
    clear = np.minimum(sequence.dump_rows, above) * sequence.row_transfer_time.to_value(u.s)
    exposed = np.where(image, u.Quantity(exposure).to_value(u.s), 0.0)
    return (clear + exposed + reading) * u.s
