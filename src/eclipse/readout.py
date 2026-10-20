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

Each transfer leaves a fraction 1 - CTE of a packet's charge to the packet
behind it, along the columns (`cte_parallel`) and along the serial register
(`cte_serial`); :func:`transfer_probabilities` gives where the charge ends
up.

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

# The focal plane constants are set in config and can be imported from here too.
from .config import (DISPERSION_COEFFICIENTS, MEASURED_SLIT_IMAGE_TILT,  # noqa: F401
                     TILT_REFERENCE_ANGLE, Detector_SWC, _check_focal_plane)


@dataclass
class FocalPlane_SWC:
    """
    Geometry of the SW focal plane, and the wavelength each row records.

    Every setting defaults to that of :class:`eclipse.config.Detector_SWC`,
    and :meth:`from_detector` takes them from a configured detector.

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
        for a position in mm. See
        :data:`~eclipse.config.DISPERSION_COEFFICIENTS`.
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
        :data:`~eclipse.config.MEASURED_SLIT_IMAGE_TILT` to model the
        drift in the optical design, about two rows end to end. Which end of
        the slit it runs toward is not documented, so the sign may need
        flipping against a calibration. `full_ccd_frame` does not yet take a
        tilted slit image.
    """

    n_rows: int = Detector_SWC.n_rows
    n_columns: int = Detector_SWC.n_columns
    pixel_size: u.Quantity = (Detector_SWC.pix_size * u.pixel).to(u.micron)
    gap: u.Quantity = Detector_SWC.ccd_gap
    dispersion: Tuple[float, ...] = Detector_SWC.dispersion
    wavelength_offset: u.Quantity = Detector_SWC.wavelength_offset
    lit_band: Tuple[u.Quantity, u.Quantity] = Detector_SWC.lit_band
    plate_scale: u.Quantity = (Detector_SWC.plate_scale_angle * u.pixel).to(u.arcsec)
    slit_image_tilt: Tuple[float, float] = Detector_SWC.slit_image_tilt

    @classmethod
    def from_detector(cls, det: Detector_SWC, **settings) -> "FocalPlane_SWC":
        """
        Focal plane of the camera `det`.

        Every setting comes from `det`. `dispersion`, `wavelength_offset`,
        `lit_band` and `slit_image_tilt` given in `settings` replace the
        detector's. The CCD format, pixel size, gap and plate scale are
        refused in `settings`, so they are set in one place, the detector.
        """
        camera = dict(n_rows=det.n_rows, n_columns=det.n_columns,
                      pixel_size=(det.pix_size * u.pixel).to(u.micron), gap=det.ccd_gap,
                      plate_scale=(det.plate_scale_angle * u.pixel).to(u.arcsec))
        taken = sorted(set(settings) & set(camera))
        if taken:
            raise ValueError(f"{', '.join(taken)} come from the detector; set them there, as "
                             f"Detector_SWC.n_rows, n_columns, pix_size, ccd_gap and "
                             f"plate_scale_angle.")
        optics = dict(dispersion=det.dispersion, wavelength_offset=det.wavelength_offset,
                      lit_band=det.lit_band, slit_image_tilt=det.slit_image_tilt)
        return cls(**camera, **{**optics, **settings})

    def __post_init__(self):
        if self.n_rows < 1 or self.n_columns < 1:
            raise ValueError(
                f"A CCD needs at least one row and one column, got "
                f"{self.n_rows} x {self.n_columns}."
            )
        _check_focal_plane(self, "focal_plane")

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
    default to those of :class:`eclipse.config.Detector_SWC`;
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
    cte_parallel, cte_serial : float
        Charge transfer efficiency of one row transfer and of one transfer
        along the serial register: the fraction of a packet's charge that
        moves, the rest joining the packet behind. 1 is a perfect transfer.
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
        Read-out of the camera `det` for one observation.

        Parameters
        ----------
        det : Detector_SWC
            Camera, whose shutter, timing, register layout and charge
            transfer efficiencies are used.
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
        if self.cte_serial < 1:
            # Each row is read from an empty serial register, so charge the
            # last image pixel leaves behind past the end of the overscan,
            # which would be read with the next row, is left out.
            transfers = self.serial_prescan + self.serial_image_pixels
            arrived = transfer_probabilities([transfers], self.cte_serial,
                                             most=self.serial_overscan)[:, 0].sum()
            if 1 - arrived > _LEFT_IN_REGISTER:
                warnings.warn(
                    f"With cte_serial {self.cte_serial} and {self.serial_overscan} serial "
                    f"overscan samples, {1 - arrived:.1e} of the last image pixel's charge is "
                    f"still in the serial register when the next row is read. Each row is "
                    f"read from an empty register, so that charge is left out.", UserWarning,
                    stacklevel=3)

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


# Largest fraction of a charge that transfer_probabilities leaves out past its
# last term.
_TAIL = 1e-15

# Largest fraction of the last image pixel's charge that may still be in the
# serial register after the overscan without a warning.
_LEFT_IN_REGISTER = 1e-6


def transfer_probabilities(transfers, cte: float, most: int | None = None) -> np.ndarray:
    """
    Fraction of a charge arriving 0, 1, 2, ... packets late after its transfers.

    Each transfer moves a fraction `cte` of a packet's charge and leaves the
    rest in the well it vacated, where the packet behind picks it up. Every
    electron is left behind independently, so charge with `m` transfers to
    go arrives `k` packets late with the negative binomial probability
    `C(m - 1 + k, k) * cte**m * (1 - cte)**k`: it moves `m` times and is
    held back `k` times, each time joining a packet with one more transfer
    to go.

    Parameters
    ----------
    transfers : array_like of int
        Transfers each charge has to go, 1 or more.
    cte : float
        Fraction of a packet's charge one transfer moves, above 0 and at
        most 1.
    most : int, optional
        Latest arrival wanted; later charge would leave the frame anyway.
        Default no limit.

    Returns
    -------
    np.ndarray
        Shape `(n_terms, *transfers.shape)`; element `[k, ...]` is the
        fraction arriving `k` packets late. Terms run until less than 1e-15
        of every charge is left over, or to `most`. A perfect transfer gives
        one term of ones.

    Raises
    ------
    ValueError
        If a transfer count is not finite, a charge has no transfer to go,
        or `cte` is not above 0 and at most 1.
    """
    m = np.asarray(transfers, dtype=float)
    if not np.all(np.isfinite(m)):
        raise ValueError("Transfers must be finite.")
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
        # Past the most likely arrival each term is at most `ratio` times the
        # last, so the rest is bounded by a geometric series.
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
    Charge each of `packets` collects while the image area is cleared, in
    units of `rate` times seconds.

    The packet that ends the clear at row `r` came down from row
    `r + dump_rows`, collecting one row transfer at each of rows `r + 1` to
    `r + dump_rows`. Rows above the image area are empty. A packet with `r`
    below zero went into the dump drain before the clear ended.
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
    Charge each packet of a frame brings to the serial register.

    `rate` is any quantity each pixel's light adds per second. It is
    collected during the exposure, unless `exposure` is None, and without a
    shutter also while the image area is cleared and read. Without a
    shutter, the packet read out of row `r` adds
    `sum_k rate[r - k] * dwell[k - 1]` over `k >= 1` during the read-out,
    and :func:`_cleared` during the clear. Each addition, made at row `x`,
    has `x + 1` row transfers to go and is shared among the packets by
    :func:`transfer_probabilities`, with the probabilities raised to
    `power`. Packets dumped during the clear also leave charge behind, which
    arrives in the first rows read.
    """
    n_rows, n_columns = rate.shape
    total_rows = n_rows + sequence.parallel_overscan_rows
    lit = not sequence.shutter
    # A packet dumped during the clear can start up to dump_rows rows before
    # the first one read, so its charge can arrive that much later.
    latest = total_rows - 1 + (sequence.dump_rows if lit else 0)
    probability = transfer_probabilities(np.arange(1, n_rows + 1), sequence.cte_parallel,
                                         most=latest) ** power
    if lit:
        kernel = np.concatenate([[0.0], sequence.dwell(n_rows)])
        # The FFT leaves rounding noise, of order 1e-16 of the brightest row, in
        # packets no light reached, which would give them a mean photon energy
        # made of noise. Zero the packets whose path crossed no lit pixel,
        # found by convolving a count of lit pixels, which is good to far
        # better than a half.
        crossed = fftconvolve((rate > 0).astype(float),
                              (kernel > 0).astype(float)[:, np.newaxis],
                              mode="full", axes=0)[:total_rows]
    arrived = np.zeros((total_rows, n_columns))
    for k, weight in enumerate(probability):
        weighted = rate * weight[:, np.newaxis]
        # Indexed by arrival, k packets after collection, so the first k rows
        # come from packets dumped during the clear.
        packets = np.zeros((total_rows, n_columns))
        if lit:
            if k < total_rows:
                # Read-out. The packet leaving row r is at row r - j while image
                # row j - 1 is in the register, so the smear is the rate
                # convolved with the dwell times. The parallel overscan rows are
                # the terms past the last image row.
                reading = fftconvolve(weighted, kernel[:, np.newaxis], mode="full",
                                      axes=0)[:total_rows]
                reading[crossed < 0.5] = 0.0
                packets[k:] += reading[:total_rows - k]
            packets += _cleared(weighted, sequence, np.arange(total_rows) - k)
            # The sum cannot be negative, but the FFT convolution leaves rounding
            # noise of order 1e-20 where it is zero, which would reach the
            # Poisson draw downstream as a negative mean.
            np.maximum(packets, 0.0, out=packets)
        if exposure is not None and k < total_rows:
            # Exposure. Image row r's charge arrives in packet r + k, in the
            # parallel overscan for the last rows.
            end = min(n_rows + k, total_rows)
            packets[k:end] += (weighted * u.Quantity(exposure).to_value(u.s))[:end - k]
        arrived += packets
    return arrived


def _along_register(frame: np.ndarray, sequence: ReadoutSequence, power: int = 1,
                    overscan: int = 0):
    """
    Charge each pixel of a frame brings to its output along the serial
    register, with the arrival probabilities raised to `power`.

    Each half of the register is read from its outer end. A pixel `j` image
    pixels from the outer end of its half has `serial_prescan + j + 1`
    transfers to go. What it leaves behind arrives in the pixels read after
    it, toward the middle of the register, and past the last image pixel
    goes into the overscan. With `overscan` above zero, the charge reaching
    the first `overscan` overscan samples of each half is returned too,
    shaped `(rows, 2 * overscan)` and laid out across the CCD: the left
    output's samples in the order read, then the right output's in reverse,
    so that each half's first sample sits next to its image pixels.
    """
    n_columns = frame.shape[1]
    if sequence.cte_serial == 1:
        trail = np.zeros((frame.shape[0], 2 * overscan))
        return (frame, trail) if overscan else frame
    half = sequence.serial_image_pixels
    if n_columns not in (2 * half, 2 * half - 1):
        raise ValueError(
            f"A transfer along the serial register needs the frame's every column, "
            f"{2 * half} for two halves of {half}, to know how far each is from its "
            f"output; got {n_columns}."
        )
    arrived = np.zeros_like(frame)
    trails = []
    # Each half in the order its output reads it: the first from column 0, the
    # second from the last column.
    for columns in (np.arange(half), np.arange(n_columns - 1, half - 1, -1)):
        charge = frame[:, columns]
        size = columns.size
        probability = transfer_probabilities(sequence.serial_prescan + 1 + np.arange(size),
                                             sequence.cte_serial, most=size + overscan) ** power
        # Samples in the order read: the image pixels, then the overscan.
        reached = np.zeros((frame.shape[0], size + overscan))
        for k, weight in enumerate(probability[:size + overscan]):
            n = min(size, size + overscan - k)
            reached[:, k:k + n] += (charge * weight)[:, :n]
        arrived[:, columns] = reached[:, :size]
        trails.append(reached[:, size:])
    if not overscan:
        return arrived
    return arrived, np.concatenate([trails[0], trails[1][:, ::-1]], axis=1)


def smear_photons(rate: np.ndarray, sequence: ReadoutSequence) -> np.ndarray:
    """
    Photons a shutterless frame collects outside its exposure.

    Charge is clocked toward the register while the chip is still lit, so a
    packet collects light from every row it crosses. For the packet read out
    of row `r`, the read-out adds `sum_k rate[r - k] * dwell[k - 1]` over
    `k >= 1`. Clearing the image area beforehand adds `row_transfer_time`
    times the rows above it, since the packet that ends at row `r` was
    clocked down from `r + dump_rows`.

    With a transfer efficiency below 1, part of what each packet collects
    arrives in the packets after it (:func:`transfer_probabilities`), along
    the columns and then along the serial register.

    Parameters
    ----------
    rate : np.ndarray
        Photons per second reaching each pixel of one CCD, shaped
        `(n_rows, n_columns)` with row 0 at the serial register. With
        `cte_serial` below 1, `n_columns` must be the whole CCD's, so that
        each column's distance from its output is known.
    sequence : ReadoutSequence
        Clocking. With `shutter` True this returns zeros.

    Returns
    -------
    np.ndarray
        Photons collected outside the exposure, shaped
        `(n_rows + parallel_overscan_rows, n_columns)`. The extra rows are
        the parallel overscan, which sees only smear.
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
        :func:`eclipse.frame.expose_with_wavelengths` gives these with
        the wavelengths of each pixel's mean and root mean square photon
        energies. A Poisson draw of whole photons in each pixel goes to
        :func:`eclipse.frame.detect` with those wavelengths and
        :func:`dark_current_time`. For the frame's mean, pass the means
        instead, with `noise=False`.

        With a transfer efficiency below 1, these are the mean photons whose
        charge arrives in each pixel. The transfer itself is not drawn, so a
        frame drawn from them has the right mean, but its noise leaves out
        the spread of the charge left behind; :func:`expose_variance`
        includes it.
    """
    rate = _check_rate(rate)
    if not sequence.shutter:
        _warn_of_a_partial_clear(sequence, rate.shape[0], "smear")
    return _along_register(_along_columns(rate, exposure, sequence), sequence)


def expose_serial_overscan(rate: np.ndarray, exposure: u.Quantity,
                           sequence: ReadoutSequence) -> np.ndarray:
    """
    Photons whose charge the serial register carries past the image into the overscan.

    With `cte_serial` below 1, each pixel leaves charge behind at every
    transfer along the register, and that charge arrives in the samples read
    after it. For the last image pixels of each half these are the serial
    overscan samples. The register's own dark current is not included, and
    the image's dark charge is in :func:`serial_dark_shares`.

    Parameters
    ----------
    rate, exposure, sequence
        As for :func:`expose`. `rate` must have the whole CCD's columns.

    Returns
    -------
    np.ndarray
        Mean photons in each serial overscan sample, shaped
        `(n_rows + parallel_overscan_rows, 2 * serial_overscan)`: the left
        output's samples in the order read, then the right output's in
        reverse, as they sit across the CCD. Zeros with `cte_serial` 1.
    """
    rate = _check_rate(rate)
    n_out = (rate.shape[0] + sequence.parallel_overscan_rows, 2 * sequence.serial_overscan)
    if sequence.cte_serial == 1 or sequence.serial_overscan == 0:
        return np.zeros(n_out)
    return _along_register(_along_columns(rate, exposure, sequence), sequence,
                           overscan=sequence.serial_overscan)[1]


def serial_dark_shares(sequence: ReadoutSequence, n_columns: int) -> tuple:
    """
    Share of a row's dark charge per pixel that each sample receives along the register.

    Dark current is the same in every column of a row, so with `cte_serial`
    below 1 each sample gets the row's dark charge per pixel times a share
    that is the same for every row. Most image pixels get 1. The first
    pixels each output reads get a little less, as their charge left behind
    is not made up by pixels read before them, and the serial overscan
    samples get the charge left behind by the last image pixels.

    Parameters
    ----------
    sequence : ReadoutSequence
        Clocking.
    n_columns : int
        Columns of the whole CCD.

    Returns
    -------
    image : np.ndarray
        Share for each image column, `(n_columns,)`.
    overscan : np.ndarray
        Share for each serial overscan sample, `(2 * serial_overscan,)`, laid
        out as from :func:`expose_serial_overscan`. Ones and zeros with
        `cte_serial` 1.
    """
    ones = np.ones((1, n_columns))
    if not sequence.serial_overscan:
        return _along_register(ones, sequence)[0], np.zeros(0)
    image, overscan = _along_register(ones, sequence, overscan=sequence.serial_overscan)
    return image[0], overscan[0]


def expose_variance(variance_rate: np.ndarray, mean_rate: np.ndarray, exposure: u.Quantity,
                    sequence: ReadoutSequence) -> np.ndarray:
    """
    Variance of the charge in each pixel of a frame, exposure and smear
    together.

    Every electron is left behind independently at each transfer, so a
    pixel receiving a fraction `P` of a charge `Q` gets a variance of
    `P**2 * var(Q) + P * (1 - P) * mean(Q)`. Summed over the frame, with `P`
    from :func:`transfer_probabilities`, this is the transfer of
    `variance_rate - mean_rate` weighted by `P**2` plus that of `mean_rate`
    weighted by `P`. With a perfect transfer it equals :func:`expose` of
    `variance_rate`.

    Parameters
    ----------
    variance_rate, mean_rate : np.ndarray
        Variance (electrons^2) and mean (electrons) of the charge each
        pixel's light frees per second, `(n_rows, n_columns)`. The variance
        is at least the mean, as for charge freed by whole photons.
    exposure, sequence
        As for :func:`expose`.

    Returns
    -------
    np.ndarray
        Variance in electrons^2, shape
        `(n_rows + parallel_overscan_rows, n_columns)`.

    Raises
    ------
    ValueError
        If the two rates differ in shape, or the variance is below the mean
        in any pixel.
    """
    variance_rate, mean_rate = _check_rate(variance_rate), _check_rate(mean_rate)
    if variance_rate.shape != mean_rate.shape:
        raise ValueError(
            f"variance_rate and mean_rate describe the same pixels, got shapes "
            f"{variance_rate.shape} and {mean_rate.shape}."
        )
    excess = variance_rate - mean_rate
    if np.any(excess < -1e-12 * np.abs(variance_rate).max()):
        raise ValueError("variance_rate must be at least mean_rate in every pixel.")
    if sequence.cte_parallel == 1 and sequence.cte_serial == 1:
        return expose(variance_rate, exposure, sequence)
    if not sequence.shutter:
        _warn_of_a_partial_clear(sequence, variance_rate.shape[0], "smear")
    return (_along_register(_along_columns(np.maximum(excess, 0.0), exposure, sequence, 2),
                            sequence, 2)
            + _along_register(_along_columns(mean_rate, exposure, sequence), sequence))


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

    With `cte_parallel` below 1, dark charge is left behind and picked up
    like any other, so this is the time each packet's dark charge is worth,
    weighted as :func:`expose` weights light. Along the serial register,
    :func:`serial_dark_shares` gives the share of a row's dark charge each
    sample receives.
    """
    _warn_of_a_partial_clear(sequence, n_rows, "dark current")
    if sequence.cte_parallel != 1:
        unlit = replace(sequence, shutter=False)
        return _along_columns(np.ones((n_rows, 1)), exposure, unlit)[:, 0] * u.s
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
