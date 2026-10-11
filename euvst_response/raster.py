"""
Slit observations of a time series of atmospheres or of their spectra.

A slit spectrograph sees one strip of the Sun at a time. In a raster each
exposure sees a different strip at a later time; in a sit-and-stare it sees
the same strip repeatedly. Given a series of atmosphere files, each with
its time, this module synthesises only the columns under the slit for each
exposure, from the snapshots that overlap it. A long series then costs
about one snapshot's worth of columns rather than every snapshot in full.
Given a series of synthesis files instead, from ECLIPSE or another code, it
reads those columns' spectra.

The observing plan is a :class:`RasterPlan`: the start time of the first
exposure, the number of slit positions per raster and their spacing, the
exposure cadence, and the number of rasters. Each raster is observed
separately, as one cube whose columns are its slit positions. A
sit-and-stare is a raster of one position, so each of its exposures is a
one-column cube. The slit width and exposure time are instrument settings
that can be swept, so the cubes are built for each combination inside the
instrument run, and the columns already obtained from each snapshot are
kept for the next combination.

An exposure spanning more than one snapshot gets the average of their
spectra, weighted by the fraction of the exposure each covers. A snapshot
represents the atmosphere from its time until the next snapshot's time, and
the last one for as long again as the gap before it. An exposure outside
that range raises an error rather than using the nearest snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.wcs import WCS
from ndcube import NDCube

from .atmosphere import (Atmosphere, read_atmosphere, read_edges, read_time,
                         require_mass_per_electron)
from .data_processing import sum_line_cubes
from .synthesis import (
    _LOGT_MAX,
    _LOGT_MIN,
    _N_N,
    _N_T,
    _density_point,
    _density_points,
    compute_goft_fiasco,
    line_of_sight_velocity,
    synthesise_cubes,
)
from .synthesis_file import (RADIANCE_UNIT, SpectralLine, Synthesis, _same_grid, _to_length,
                             read_synthesis, read_synthesis_layout, read_synthesis_products)
from .sampling import pixel_weights
from .utils import (VELOCITY_CONVENTION, _fwhm_to_sigma, angle_to_distance, require_uniform_grid,
                    velocity_grid)

__all__ = ["SynthesisSettings", "RasterPlan", "Exposure", "AtmosphereSeries",
           "RasterSynthesiser", "SynthesisSeries", "SynthesisRaster"]

INTENSITY_UNIT = u.erg / u.s / u.cm**2 / u.sr / u.cm


@dataclass(frozen=True)
class SynthesisSettings:
    """
    Synthesis settings, as given to `synthesise-spectra` on the command line.

    Parameters
    ----------
    lines : sequence of str
        Line names such as `"Fe12_195.1190"`.
    abundance : str
        CHIANTI abundance set.
    vel_res, vel_lim : u.Quantity
        Spacing and half range of the velocity grid.
    crop_y, crop_z : tuple of two quantities, optional
        Ranges to keep along y and z, in the atmosphere's own coordinates.
    precision : type
        np.float32 or np.float64.
    mass_per_electron : float, optional
        Atomic mass units per free electron, for atmospheres that give only
        a mass density. None derives it from the abundance set.
    hdf5_dbase_root : str, optional
        CHIANTI database for fiasco. None uses fiasco's default.
    n_workers : int
        Workers for computing the contribution functions; 0 uses all CPUs.
    goft_temperature_chunk : int, optional
        Number of temperatures passed to fiasco at a time when computing the
        contribution functions, to reduce memory use. None passes the whole
        grid at once.
    """

    lines: Tuple[str, ...]
    abundance: str = "sun_coronal_2021_chianti"
    vel_res: u.Quantity = 5.0 * u.km / u.s
    vel_lim: u.Quantity = 300.0 * u.km / u.s
    crop_y: Optional[Tuple[u.Quantity, u.Quantity]] = None
    crop_z: Optional[Tuple[u.Quantity, u.Quantity]] = None
    precision: type = np.float64
    mass_per_electron: Optional[float] = None
    hdf5_dbase_root: Optional[str] = None
    n_workers: int = 0
    goft_temperature_chunk: Optional[int] = None

    def __post_init__(self):
        if not self.lines:
            raise ValueError("At least one line is needed.")
        for name in ("vel_res", "vel_lim"):
            value = getattr(self, name)
            if not isinstance(value, u.Quantity) or not value.unit.is_equivalent(u.km / u.s):
                raise ValueError(f"{name} must be a velocity, got {value!r}.")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.precision not in (np.float32, np.float64):
            raise ValueError(f"precision must be np.float32 or np.float64, got {self.precision!r}.")
        if self.mass_per_electron is not None:
            require_mass_per_electron(self.mass_per_electron)

    def velocity_grid(self) -> u.Quantity:
        """Velocity bin centres, as built by `synthesise-spectra`."""
        return velocity_grid(self.vel_res, self.vel_lim)


@dataclass(frozen=True)
class Exposure:
    """One exposure of a plan: where the slit is and when it is open."""

    index: int
    position: u.Quantity  # heliocentric x of the slit's centre
    start: u.Quantity     # simulation time
    end: u.Quantity


@dataclass(frozen=True)
class RasterPlan:
    """
    Observing plan: rasters of slit positions, one exposure after another.

    Parameters
    ----------
    start : u.Quantity
        Simulation time at which the first exposure starts.
    steps : int
        Slit positions per raster; 1 is a sit-and-stare.
    step : u.Quantity, optional
        Angle between neighbouring slit positions. None uses the slit width,
        so the positions abut.
    repeats : int
        Number of rasters, one after another.
    cadence : u.Quantity, optional
        Time between the starts of consecutive exposures. None uses the
        exposure time, so each exposure starts as the previous one ends.
    centre : u.Quantity, optional
        Heliocentric x of the raster centre, as a length. None centres it on
        the atmosphere.
    direction : str
        Direction the slit steps in x: `"increasing"` (default) or
        `"decreasing"`, as the deprecated dynamic mode scanned. Either way
        the image columns run along increasing x, each observed at its own
        exposure's time.
    """

    start: u.Quantity
    steps: int = 1
    step: Optional[u.Quantity] = None
    repeats: int = 1
    cadence: Optional[u.Quantity] = None
    centre: Optional[u.Quantity] = None
    direction: str = "increasing"

    def __post_init__(self):
        if not isinstance(self.start, u.Quantity) or not self.start.unit.is_equivalent(u.s):
            raise ValueError(f"start must be a time, got {self.start!r}.")
        for name in ("steps", "repeats"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a whole number of 1 or more, got {value!r}.")
        if self.step is not None:
            if not isinstance(self.step, u.Quantity) or not self.step.unit.is_equivalent(u.arcsec):
                raise ValueError(f"step must be an angle, got {self.step!r}.")
            if self.step <= 0:
                raise ValueError(f"step must be positive, got {self.step}.")
        if self.cadence is not None:
            if not isinstance(self.cadence, u.Quantity) or not self.cadence.unit.is_equivalent(u.s):
                raise ValueError(f"cadence must be a time, got {self.cadence!r}.")
            if self.cadence <= 0:
                raise ValueError(f"cadence must be positive, got {self.cadence}.")
        if self.centre is not None and (not isinstance(self.centre, u.Quantity)
                                        or not self.centre.unit.is_equivalent(u.Mm)):
            raise ValueError(f"centre must be a length, got {self.centre!r}.")
        if self.direction not in ("increasing", "decreasing"):
            raise ValueError(f"direction must be 'increasing' or 'decreasing' x, got "
                             f"{self.direction!r}.")

    def exposures(self, slit_width: u.Quantity, expos: u.Quantity,
                  atmosphere_centre: u.Quantity,
                  repeat: Optional[int] = None) -> List[Exposure]:
        """
        Exposures of one raster of the plan, for one slit width and exposure time.

        Parameters
        ----------
        slit_width : u.Quantity
            Slit width as an angle. Also the step if none was given.
        expos : u.Quantity
            Exposure time. Also the cadence if none was given.
        atmosphere_centre : u.Quantity
            Heliocentric x of the centre of the atmosphere, or of the image
            of a series of synthesis files. Also the raster centre if none
            was given.
        repeat : int, optional
            Raster index, from 0. Not needed for a plan of one raster, but
            required for several, since each raster is observed separately.
        """
        if expos <= 0:
            raise ValueError(f"The exposure time must be positive, got {expos}.")
        if repeat is None:
            if self.repeats > 1:
                raise ValueError(f"The plan has {self.repeats} rasters; say which one "
                                 f"with repeat=0 to {self.repeats - 1}.")
            repeat = 0
        if isinstance(repeat, bool) or not isinstance(repeat, (int, np.integer)):
            raise ValueError(f"repeat must be a whole number, got {repeat!r}.")
        if not 0 <= repeat < self.repeats:
            raise ValueError(f"repeat must be between 0 and {self.repeats - 1}, got {repeat}.")
        cadence = expos if self.cadence is None else self.cadence
        if cadence < expos:
            raise ValueError(f"The cadence, {cadence}, is shorter than the exposure, "
                             f"{expos}, so exposures would overlap.")
        step = angle_to_distance(slit_width if self.step is None else self.step).to(u.Mm)
        if self.direction == "decreasing":
            step = -step
        centre = (atmosphere_centre if self.centre is None else self.centre).to(u.Mm)
        first = centre - (self.steps - 1) / 2 * step
        exposures = []
        for position in range(self.steps):
            index = repeat * self.steps + position
            start = self.start + index * cadence
            exposures.append(Exposure(index, first + position * step, start, start + expos))
        return exposures


class _Series:
    """
    Snapshot files sorted by their recorded times.

    Each snapshot represents the Sun from its time until the next one's,
    and the last for as long again as the gap before it.
    """

    # Suggestion for a single snapshot, used in the error for a series of one.
    _alone = "observe a single snapshot as a static atmosphere"

    def __init__(self, timed: Sequence[Tuple[float, Path]]):
        timed = sorted(timed, key=lambda item: item[0])
        times = np.array([t for t, _ in timed])
        if np.any(np.diff(times) <= 0):
            same = [str(p) for (t, p), (t2, _) in zip(timed[:-1], timed[1:]) if t == t2]
            raise ValueError(f"Two files of the series record the same time: {same[0]}")
        self.paths: List[Path] = [p for _, p in timed]
        self.times: u.Quantity = times * u.s

    def __len__(self) -> int:
        return len(self.paths)

    def valid_until(self) -> u.Quantity:
        """
        Time at which each snapshot stops representing the atmosphere.

        This is the next snapshot's time, and for the last snapshot its own
        time plus the gap before it.
        """
        times = self.times.to_value(u.s)
        if times.size == 1:
            raise ValueError(f"A series of one snapshot cannot say how long it stands for; "
                             f"give at least two, or {self._alone}.")
        ends = np.append(times[1:], times[-1] + (times[-1] - times[-2]))
        return ends * u.s

    def coverage(self, start: u.Quantity, end: u.Quantity) -> List[Tuple[int, float]]:
        """
        Snapshots an exposure from `start` to `end` falls in, with the fraction each covers.

        The fractions sum to one. An exposure reaching before the first
        snapshot or beyond the last one's span raises a ValueError.
        """
        t0, t1 = start.to_value(u.s), end.to_value(u.s)
        if t1 <= t0:
            raise ValueError(f"An exposure must end after it starts, got {start} to {end}.")
        begins = self.times.to_value(u.s)
        ends = self.valid_until().to_value(u.s)
        # Allow for rounding, so a plan that exactly fills the series is not
        # rejected. Exposure times are a start plus whole cadences plus an
        # exposure, and the series end is the last time plus the last gap.
        # Each of these few steps, and each unit change, rounds by at most
        # half a unit in the last place of the largest time; 16 units bound
        # them all.
        rounding = 16 * np.spacing(max(abs(begins[0]), abs(ends[-1]), abs(t0), abs(t1)))
        early, late = begins[0] - t0, t1 - ends[-1]
        if early > rounding or late > rounding:
            where = (f"starting {early:.3g} s before it" if early > rounding
                     else f"ending {late:.3g} s after it")
            raise ValueError(
                f"An exposure from {t0:.10g} to {t1:.10g} s lies outside the series, {where}; "
                f"the series runs from {begins[0]:.10g} to {ends[-1]:.10g} s (the last "
                f"snapshot, at {begins[-1]:.10g} s, stands for as long as the gap before it).")
        inside = max(t0, begins[0]), min(t1, ends[-1])
        if inside[1] <= inside[0]:
            raise ValueError(
                f"An exposure from {t0:.10g} to {t1:.10g} s lies outside the series, which "
                f"runs from {begins[0]:.10g} to {ends[-1]:.10g} s.")
        t0, t1 = inside
        overlap = np.clip(np.minimum(ends, t1) - np.maximum(begins, t0), 0.0, None)
        fractions = overlap / (t1 - t0)
        return [(int(k), float(f)) for k, f in enumerate(fractions) if f > 0.0]


class AtmosphereSeries(_Series):
    """
    Atmosphere files sorted by their recorded times.

    Every file must record a time, and all must share one grid, since an
    exposure combines columns from different snapshots.
    """

    def __init__(self, paths: Sequence[str | Path]):
        if not paths:
            raise ValueError("An atmosphere series needs at least one file.")
        timed = []
        for path in paths:
            time = read_time(path)
            if time is None:
                raise ValueError(f"{path} records no time, which a series needs to place "
                                 f"its snapshots; write one into the file.")
            timed.append((time.to_value(u.s), Path(path)))
        super().__init__(timed)
        # The grid comes from the first file; the others are checked as they
        # are read.
        edges = read_edges(self.paths[0])
        self.x_edges = edges["x"]
        self.y_edges = edges["y"]
        self.z_edges = edges["z"]


class SynthesisSeries(_Series):
    """
    Synthesis files sorted by their recorded times, for one line's window.

    Every file must record a time, and all must share one image, seen along
    one axis, and the same lines on the same wavelengths, since an exposure
    combines columns from different snapshots and adds up their spectra.
    Only the lines reaching the window of `reference_line` are read, as for
    a single synthesis file.

    Parameters
    ----------
    paths : sequence of str or Path
    reference_line : str
        Line the instrument measures.
    """

    _alone = "observe the one file as a single snapshot, with 'synthesis_file'"

    def __init__(self, paths: Sequence[str | Path], reference_line: str):
        if not paths:
            raise ValueError("A synthesis series needs at least one file.")
        layouts = {}
        timed = []
        for path in paths:
            path = Path(path)
            layout = read_synthesis_layout(path, reference_line)
            if layout["time"] is None:
                raise ValueError(f"{path} records no time, which a series needs to place "
                                 f"its snapshots; write one into the file.")
            dynamic = read_synthesis_products(path, keys=("dynamic_mode",)).get("dynamic_mode")
            if (dynamic or {}).get("enabled"):
                raise ValueError(f"{path} was synthesised in dynamic mode, which already "
                                 f"scanned a slit over time; a series takes one snapshot "
                                 f"per file.")
            layouts[path] = layout
            timed.append((layout["time"].to_value(u.s), path))
        super().__init__(timed)
        self.reference_line = reference_line
        first = self.paths[0]
        layout = layouts[first]
        for axis in ("x", "y"):
            try:
                require_uniform_grid(layout[f"{axis}_edges"].value, f"{axis}_edges")
            except ValueError as error:
                raise ValueError(f"{first}: {error}") from None
        # x in Mm, like the slit positions; y in the files' own unit.
        self.x_edges: u.Quantity = _to_length(layout["x_edges"])
        self.y_edges: u.Quantity = layout["y_edges"]
        self.integration_axis: Optional[str] = layout["integration_axis"]
        self.wavelengths: Dict[str, u.Quantity] = {
            name: info["wavelength"] for name, info in layout["lines"].items()}
        self.rest_wavelengths: Dict[str, u.Quantity] = {
            name: info["rest_wavelength"] for name, info in layout["lines"].items()}
        self.identities: Dict[str, dict] = {
            name: info["identity"] for name, info in layout["lines"].items()}
        shape = (self.y_edges.size - 1, self.x_edges.size - 1)
        for name, info in layout["lines"].items():
            if tuple(info["shape"]) != shape + (info["wavelength"].size,):
                raise ValueError(
                    f"{first}, line {name!r}: the intensity is {tuple(info['shape'])}, but "
                    f"the edges and wavelengths give (ny, nx, n_wavelength) = "
                    f"{shape + (info['wavelength'].size,)}.")
        for path in self.paths[1:]:
            other = layouts[path]
            for axis, edges in (("x", self.x_edges), ("y", self.y_edges)):
                found = other[f"{axis}_edges"]
                if not _same_grid(_to_length(found), _to_length(edges)):
                    raise ValueError(f"{path} has a different {axis} grid from {first}; a "
                                     f"series must share one image.")
            if (other["integration_axis"] or "z") != (self.integration_axis or "z"):
                raise ValueError(f"{path} was seen along {other['integration_axis'] or 'z'}, "
                                 f"and {first} along {self.integration_axis or 'z'}; a series "
                                 f"must share one view.")
            if set(other["lines"]) != set(layout["lines"]):
                raise ValueError(f"{path} has the lines {list(other['lines'])} in the window "
                                 f"of {reference_line}, and {first} has "
                                 f"{list(layout['lines'])}; a series must have the same.")
            for name, info in other["lines"].items():
                if not _same_grid(info["wavelength"], self.wavelengths[name]):
                    raise ValueError(f"{path} has different wavelengths for {name} from "
                                     f"{first}; a series must keep each line on one grid.")
                if not _same_grid(info["rest_wavelength"], self.rest_wavelengths[name]):
                    raise ValueError(f"{path} gives {name} a different rest wavelength from "
                                     f"{first}.")
                if info["identity"] != self.identities[name]:
                    raise ValueError(f"{path} gives {name} the atom and ion "
                                     f"{info['identity']}, and {first} "
                                     f"{self.identities[name]}.")
                if tuple(info["shape"]) != tuple(layout["lines"][name]["shape"]):
                    raise ValueError(f"{path}, line {name!r}: the intensity is "
                                     f"{tuple(info['shape'])}, not "
                                     f"{tuple(layout['lines'][name]['shape'])} as in {first}.")

    @property
    def lines(self) -> List[str]:
        """Lines read: those reaching the window of the reference line."""
        return list(self.wavelengths)


def _runs(indices: Sequence[int]) -> List[Tuple[int, int]]:
    """Runs of consecutive values in ascending `indices`, as (first, last + 1) pairs."""
    runs: List[Tuple[int, int]] = []
    for index in indices:
        if runs and index == runs[-1][1]:
            runs[-1] = (runs[-1][0], index + 1)
        else:
            runs.append((index, index + 1))
    return runs


class _SlitRaster:
    """
    Spectra a slit sees over a plan, from the columns of a snapshot series.

    Each column of each snapshot is fetched when an exposure first needs it
    and then cached, so a sweep over exposure times or slit widths reuses
    most of the work. Subclasses define how a strip of columns is fetched.
    """

    # What the x grid spans, named in the error when a slit reaches beyond it.
    _extent = "atmosphere"

    def __init__(self, series: _Series):
        self.series = series
        self._columns: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}

    def _load_strip(self, snapshot: int, first: int, last: int) -> None:
        """Put columns `first` to `last` - 1 of one snapshot into the cache."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def atmosphere_centre(self) -> u.Quantity:
        """Heliocentric x of the centre of the series' x grid."""
        edges = self.series.x_edges
        return 0.5 * (edges[0] + edges[-1])

    @classmethod
    def check_plan(cls, series: _Series, plan: RasterPlan, slit_widths: Sequence[u.Quantity],
                   exposure_times: Sequence[u.Quantity]) -> None:
        """
        Check that `plan` can observe `series` with every slit width and exposure time.

        Every exposure of every raster must lie within the series' times and
        put the slit within its x grid, or a ValueError is raised. Only the
        series' times and edges are used, so the check runs before anything
        is synthesised or read and a sweep does not fail part-way through.
        """
        geometry = _SlitRaster(series)
        geometry._extent = cls._extent
        repeats = range(plan.repeats) if plan.repeats > 1 else [None]
        for slit_width in slit_widths:
            for expos in exposure_times:
                for repeat in repeats:
                    for exposure in plan.exposures(slit_width, expos,
                                                   geometry.atmosphere_centre(), repeat):
                        series.coverage(exposure.start, exposure.end)
                        geometry.columns_under(exposure.position, slit_width)

    def columns_under(self, position: u.Quantity, slit_width: u.Quantity) -> Tuple[int, int, np.ndarray]:
        """
        Columns covered by a slit at `position`.

        Returns the first index, the last index plus one, and the fraction of
        the slit each column covers, summing to one.
        """
        width = angle_to_distance(slit_width).to_value(u.Mm)
        edges = self.series.x_edges.to_value(u.Mm)
        low = position.to_value(u.Mm) - width / 2
        high = position.to_value(u.Mm) + width / 2
        # A slit edge on a cell boundary (as when the slit is a whole number
        # of cells wide) must not, through rounding, pull in the next cell or
        # reach past the edge of the box. A column counts only when more than
        # a millionth of a cell of it is under the slit.
        tolerance = 1e-6 * np.diff(edges).min()
        if low < edges[0] - tolerance or high > edges[-1] + tolerance:
            where = (f"{edges[0] - low:.3g} Mm below it" if low < edges[0] - tolerance
                     else f"{high - edges[-1]:.3g} Mm beyond it")
            raise ValueError(
                f"A slit {width:.4g} Mm wide at x = {position.to_value(u.Mm):.4g} Mm reaches "
                f"outside the {self._extent}, {where}; the {self._extent} spans x = "
                f"{edges[0]:.4g} to {edges[-1]:.4g} Mm.")
        overlap = np.clip(np.minimum(edges[1:], high) - np.maximum(edges[:-1], low), 0.0, None)
        inside = np.flatnonzero(overlap > tolerance)
        if inside.size == 0:
            raise ValueError(
                f"A slit {width:.4g} Mm wide at x = {position.to_value(u.Mm):.4g} Mm covers no "
                f"cell of the {self._extent}, which spans x = {edges[0]:.4g} to "
                f"{edges[-1]:.4g} Mm.")
        first, last = int(inside[0]), int(inside[-1]) + 1
        return first, last, overlap[first:last] / overlap[first:last].sum()

    # ------------------------------------------------------------------
    # Exposures
    # ------------------------------------------------------------------
    def column_spectra(self, snapshot: int, column: int) -> Dict[str, np.ndarray]:
        """Spectra of one column of one snapshot, `(rows, wavelength)` per line."""
        if (snapshot, column) not in self._columns:
            self._load_strip(snapshot, column, column + 1)
        return self._columns[(snapshot, column)]

    def columns_seen(self, position: u.Quantity, slit_width: u.Quantity,
                     across_slit: Optional[u.Quantity] = None,
                     extend: bool = True) -> Tuple[int, int, np.ndarray]:
        """
        Columns whose light enters a slit at `position`.

        Returned in the same form as :meth:`columns_under`. With
        `across_slit`, the telescope blurs the image across the slit by a
        Gaussian of that FWHM, so light from columns beside the slit enters
        it too. With `extend` the edge columns stand in for the Sun beyond
        them; without it, the Sun beyond is dark. Without `across_slit`,
        these are the columns under the slit.
        """
        first, last, fractions = self.columns_under(position, slit_width)
        if across_slit is None:
            return first, last, fractions
        width = angle_to_distance(slit_width).to_value(u.Mm)
        sigma = angle_to_distance(_fwhm_to_sigma(across_slit)).to_value(u.Mm)
        centre = position.to_value(u.Mm)
        weights = pixel_weights(self.series.x_edges.to_value(u.Mm),
                                [centre - width / 2, centre + width / 2], sigma, extend=extend)[0]
        seen = np.flatnonzero(weights > 0)
        first, last = int(seen[0]), int(seen[-1]) + 1
        return first, last, weights[first:last]

    def exposure_spectra(self, exposure: Exposure, slit_width: u.Quantity,
                         across_slit: Optional[u.Quantity] = None,
                         extend: bool = True) -> Dict[str, np.ndarray]:
        """
        Spectra collected by the slit in one exposure.

        The columns under the slit are averaged over the slit, and the
        snapshots the exposure spans are weighted by the time each covers.
        With `across_slit`, light blurred into the slit from beside it is
        included too (:meth:`columns_seen`).
        """
        first, last, fractions = self.columns_seen(exposure.position, slit_width,
                                                   across_slit, extend)
        spectra: Dict[str, np.ndarray] = {}
        for snapshot, weight in self.series.coverage(exposure.start, exposure.end):
            missing = [c for c in range(first, last) if (snapshot, c) not in self._columns]
            # Fetch each run of missing neighbouring columns as one strip, so
            # a wider slit only fetches the columns beside those a narrower
            # slit already fetched.
            for run_first, run_last in _runs(missing):
                self._load_strip(snapshot, run_first, run_last)
            for column, fraction in zip(range(first, last), fractions):
                for name, si in self._columns[(snapshot, column)].items():
                    contribution = weight * fraction * si
                    spectra[name] = contribution if name not in spectra else spectra[name] + contribution
        return spectra

    def _observe(self, plan: RasterPlan, slit_width: u.Quantity, expos: u.Quantity,
                 repeat: Optional[int], across_slit: Optional[u.Quantity] = None,
                 extend: bool = True) -> Tuple[List[Dict[str, np.ndarray]], u.Quantity, u.Quantity, dict]:
        """
        Observe one raster of the plan.

        Returns each exposure's spectra, the slit positions, the spacing
        between them, and the raster metadata for the cube.

        The columns are the raster's exposures in order, at slit positions
        advancing by the step. A sit-and-stare is a raster of one exposure,
        so its cube has one column, as wide as the slit. `across_slit` and
        `extend` are as for :meth:`columns_seen`.
        """
        exposures = plan.exposures(slit_width, expos, self.atmosphere_centre(), repeat)
        # Order the columns along increasing x whichever way the slit stepped,
        # so the image has the same orientation; each keeps its own times.
        if plan.direction == "decreasing":
            exposures = exposures[::-1]
        collected = [self.exposure_spectra(exposure, slit_width, across_slit, extend)
                     for exposure in exposures]
        positions = u.Quantity([e.position for e in exposures]).to(u.Mm)
        pitch = (positions[1] - positions[0] if plan.steps > 1
                 else angle_to_distance(slit_width).to(u.Mm))
        meta_raster = {
            "raster": True,
            "repeat": 0 if repeat is None else repeat,
            "positions": positions,
            "starts": u.Quantity([e.start for e in exposures]).to(u.s),
            "ends": u.Quantity([e.end for e in exposures]).to(u.s),
            "steps": plan.steps,
            "repeats": plan.repeats,
            "slit_width": slit_width,
            "expos": expos,
        }
        return collected, positions, pitch, meta_raster


class RasterSynthesiser(_SlitRaster):
    """
    Synthesises the spectra a slit sees over a plan, one exposure at a time.

    The contribution functions are computed for the densities of the first
    strip synthesised. Later strips add only the densities they need beyond
    those, which can take as long as the first if the plasma reaches far
    beyond them. Each column of each snapshot is synthesised when an
    exposure first needs it and then cached, so a sweep over exposure times
    or slit widths reuses most of the work.

    Parameters
    ----------
    series : AtmosphereSeries
    settings : SynthesisSettings
    """

    def __init__(self, series: AtmosphereSeries, settings: SynthesisSettings):
        super().__init__(series)
        self.settings = settings
        self.vel_grid = settings.velocity_grid()
        # The contribution functions are computed for the first strip's
        # densities, and extended when a later strip needs others.
        self.goft: Optional[Dict[str, dict]] = None
        self.logT_grid = np.linspace(_LOGT_MIN, _LOGT_MAX, _N_T).astype(settings.precision)
        self.logN_grid: Optional[np.ndarray] = None
        self.goft_dbase_root: Optional[str] = None
        self._density_points: Optional[Tuple[int, int]] = None
        self._mass_per_electron: Optional[Tuple[float, str]] = None
        self._wl_grids: Dict[str, u.Quantity] = {}
        self._rows: Optional[u.Quantity] = None
        self._row_pitch: Optional[u.Quantity] = None
        self.strips_synthesised = 0

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def _mass_per_electron_amu(self) -> Tuple[float, str]:
        from .atmosphere import mass_per_electron
        if self._mass_per_electron is None:
            if self.settings.mass_per_electron is not None:
                self._mass_per_electron = (self.settings.mass_per_electron,
                                           "given in the configuration")
            else:
                self._mass_per_electron = (
                    mass_per_electron(self.settings.abundance, self.settings.hdf5_dbase_root),
                    f"fully ionised plasma with {self.settings.abundance} abundances")
        return self._mass_per_electron

    def _load_strip(self, snapshot: int, first: int, last: int) -> None:
        self._synthesise_strip(snapshot, first, last)

    def _cover_densities(self, temperature: np.ndarray, electron_density: np.ndarray) -> None:
        """Compute the contribution functions at the missing densities these cells need."""
        settings = self.settings
        needed = _density_points(temperature, electron_density, self.logT_grid,
                                 settings.precision)
        if needed is None:
            if self.goft is not None:
                return
            needed = (0, _N_N - 1)
        have = self._density_points
        if have is None:
            missing = [needed]
        else:
            missing = ([(needed[0], have[0] - 1)] if needed[0] < have[0] else []) + (
                [(have[1] + 1, needed[1])] if needed[1] > have[1] else [])
        for first, last in missing:
            print(f"Computing contribution functions via fiasco for {len(settings.lines)} lines "
                  f"at 10^{_density_point(first):.1f} to 10^{_density_point(last):.1f} cm^-3")
            goft, self.logT_grid, logN_grid = compute_goft_fiasco(
                list(settings.lines), abundance=settings.abundance,
                logN_min=_density_point(first), logN_max=_density_point(last),
                nN=last - first + 1, precision=settings.precision,
                n_workers=settings.n_workers, hdf5_dbase_root=settings.hdf5_dbase_root,
                temperature_chunk=settings.goft_temperature_chunk)
            if self.goft is None:
                self.goft, self.logN_grid = goft, logN_grid
                self.goft_dbase_root = next(iter(goft.values()))["hdf5_dbase_root"]
                print(f"  CHIANTI database: {self.goft_dbase_root}")
                self._density_points = (first, last)
                continue
            # Each density is computed independently, so the new ones are
            # simply joined onto either end of the table.
            below = first < self._density_points[0]
            for name, info in goft.items():
                parts = [info["g_tn"], self.goft[name]["g_tn"]]
                self.goft[name]["g_tn"] = np.concatenate(parts if below else parts[::-1])
            parts = [logN_grid, self.logN_grid]
            self.logN_grid = np.concatenate(parts if below else parts[::-1])
            self._density_points = (min(first, self._density_points[0]),
                                    max(last, self._density_points[1]))

    def _synthesise_strip(self, snapshot: int, first: int, last: int) -> None:
        """Synthesise columns `first` to `last` - 1 of one snapshot into the cache."""
        settings = self.settings
        strip = read_atmosphere(self.series.paths[snapshot], velocities=("z",),
                                columns=slice(first, last))
        expected = {"x": self.series.x_edges[first:last + 1], "y": self.series.y_edges,
                    "z": self.series.z_edges}
        for axis, edges in expected.items():
            found = strip.edges(axis)
            if found.size != edges.size or not np.allclose(found.to_value(u.Mm),
                                                           edges.to_value(u.Mm)):
                raise ValueError(f"{self.series.paths[snapshot]} has a different {axis} grid "
                                 f"from {self.series.paths[0]}; a series must share one grid.")
        if settings.crop_y or settings.crop_z:
            strip = strip.cropped(y=settings.crop_y, z=settings.crop_z)
        precision = settings.precision
        temperature = strip.temperature.astype(precision).to_value(u.K)
        if strip.electron_density is not None:
            electron_density = strip.electron_density.astype(precision).to_value(u.cm**-3)
        else:
            amu, _ = self._mass_per_electron_amu()
            electron_density = (strip.mass_density.astype(precision)
                                / (amu * const.u)).to_value(u.cm**-3)
        velocity = line_of_sight_velocity(
            strip.velocity("z").astype(precision).to_value(u.cm / u.s), "z")
        dh_cm = strip.cell_thickness("z").to_value(u.cm)

        self._cover_densities(temperature, electron_density)
        lines, _, _ = synthesise_cubes(
            temperature, electron_density, velocity, dh_cm, self.goft,
            self.logT_grid, self.logN_grid, self.vel_grid, "z", precision)
        self.strips_synthesised += 1

        if self._rows is None:
            self._rows = strip.coordinate("y")
            self._row_pitch = strip.spacing("y")
            self._wl_grids = {name: info["wl_grid"] for name, info in lines.items()}
        for offset in range(last - first):
            self._columns[(snapshot, first + offset)] = {
                name: info["si"][:, offset, :] for name, info in lines.items()}

    def line_cubes(self, plan: RasterPlan, slit_width: u.Quantity,
                   expos: u.Quantity, repeat: Optional[int] = None, *,
                   across_slit: Optional[u.Quantity] = None,
                   extend: bool = True) -> Dict[str, NDCube]:
        """
        One cube per line for one raster of the plan, `(rows, exposures, wavelength)`.

        The columns are the raster's exposures in order, at slit positions
        advancing by the step. A sit-and-stare is a raster of one exposure,
        so its cube has one column, as wide as the slit, and its repeats are
        separate cubes, one per exposure. `across_slit` and `extend` are as
        for :meth:`columns_seen`.
        """
        collected, positions, pitch, meta_raster = self._observe(plan, slit_width, expos, repeat,
                                                                 across_slit, extend)
        rows = self._rows.to(u.Mm)
        row_pitch = self._row_pitch.to(u.Mm)
        n_columns, n_rows = positions.size, rows.size
        cubes = {}
        for name in self.settings.lines:
            data = np.stack([c[name] for c in collected], axis=1)
            wl_grid = self._wl_grids[name].to(u.cm)
            # Reference pixel at the middle of each spatial axis, with the grid
            # value there, as in the line cubes of a static synthesis.
            wcs = WCS(naxis=3)
            wcs.wcs.ctype = ["WAVE", "SOLX", "SOLY"]
            wcs.wcs.cunit = ["cm", "Mm", "Mm"]
            wcs.wcs.crpix = [1, (n_columns + 1) / 2, (n_rows + 1) / 2]
            wcs.wcs.crval = [wl_grid[0].value,
                             positions[0].value + (n_columns - 1) / 2 * pitch.value,
                             rows[0].value + (n_rows - 1) / 2 * row_pitch.value]
            wcs.wcs.cdelt = [(wl_grid[1] - wl_grid[0]).value, pitch.value, row_pitch.value]
            info = self.goft[name]
            cubes[name] = NDCube(data, wcs=wcs, unit=INTENSITY_UNIT, meta={
                "line_name": name,
                "rest_wav": info["wl0"],
                "atom": info["atom"],
                "ion": info["ion"],
                "integration_axis": "z",
                "velocity_convention": VELOCITY_CONVENTION,
                **meta_raster,
            })
        return cubes

    def summed_cube(self, plan: RasterPlan, slit_width: u.Quantity, expos: u.Quantity,
                    reference_line: str, repeat: Optional[int] = None, *,
                    across_slit: Optional[u.Quantity] = None, extend: bool = True) -> NDCube:
        """Lines summed onto the reference line's wavelength grid, for the instrument run."""
        if reference_line not in self.settings.lines:
            raise ValueError(f"The reference line {reference_line!r} is not among the "
                             f"synthesised lines {list(self.settings.lines)}.")
        return sum_line_cubes(self.line_cubes(plan, slit_width, expos, repeat,
                                              across_slit=across_slit, extend=extend),
                              reference_line)


class SynthesisRaster(_SlitRaster):
    """
    Reads the spectra a slit sees over a plan from a series of synthesis files.

    Each column of each snapshot is read when an exposure first needs it and
    cached in ECLIPSE's radiance unit, so a sweep over exposure times or
    slit widths reads each column once.

    Parameters
    ----------
    series : SynthesisSeries
    """

    _extent = "image"

    def __init__(self, series: SynthesisSeries):
        super().__init__(series)
        self.strips_read = 0

    def _load_strip(self, snapshot: int, first: int, last: int) -> None:
        """Read columns `first` to `last` - 1 of one snapshot into the cache."""
        strip = read_synthesis(self.series.paths[snapshot], self.series.reference_line,
                               columns=slice(first, last))
        radiance = {name: strip.lines[name].radiance().to_value(RADIANCE_UNIT)
                    for name in self.series.lines}
        self.strips_read += 1
        for offset in range(last - first):
            self._columns[(snapshot, first + offset)] = {
                name: values[:, offset, :] for name, values in radiance.items()}

    def synthesis(self, plan: RasterPlan, slit_width: u.Quantity, expos: u.Quantity,
                  repeat: Optional[int] = None, *, across_slit: Optional[u.Quantity] = None,
                  extend: bool = True) -> Tuple[Synthesis, dict]:
        """
        One raster of the plan as a synthesis, and the raster metadata for its cube.

        The synthesis columns are the raster's exposures, at slit positions
        advancing by the step. A sit-and-stare is a raster of one exposure,
        so its image has one column, as wide as the slit. The synthesis is
        put on the detector like a single snapshot, with the raster entries
        in the cube's metadata to keep one column per exposure. `across_slit`
        and `extend` are as for :meth:`columns_seen`.
        """
        collected, positions, pitch, meta_raster = self._observe(plan, slit_width, expos, repeat,
                                                                 across_slit, extend)
        x_edges = positions[0] - pitch / 2 + np.arange(positions.size + 1) * pitch
        lines = {name: SpectralLine(
                     intensity=np.stack([c[name] for c in collected], axis=1) * RADIANCE_UNIT,
                     wavelength=self.series.wavelengths[name],
                     rest_wavelength=self.series.rest_wavelengths[name],
                     **self.series.identities[name])
                 for name in self.series.lines}
        return Synthesis(lines=lines, x_edges=x_edges, y_edges=self.series.y_edges,
                         integration_axis=self.series.integration_axis), meta_raster
