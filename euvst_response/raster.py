"""
Observing a time series of atmospheres with a slit.

A slit spectrograph sees one strip of the Sun at a time. Over a raster each
exposure sees a different strip at a later time, and over a sit-and-stare
the same strip again and again. Given a series of atmosphere files, each
with its time, this module synthesises only the columns under the slit for
each exposure, from the snapshots that overlap it, so an observation of a
long series costs about one snapshot's worth of columns rather than every
snapshot in full.

The observing plan is a :class:`RasterPlan`: when the first exposure starts,
how many slit positions a raster has, how far apart they are, how often the
exposures start, and how many rasters follow. Each raster is observed on
its own, as one cube whose columns are its slit positions; a sit-and-stare
is a raster of one position, so each of its exposures is a cube of one
column. The slit width and the exposure time are instrument settings and
can be swept, so the cubes are built per combination inside the instrument
run, and the columns each snapshot has already given are kept for the next
combination.

An exposure that spans more than one snapshot gets the average of their
spectra, weighted by how much of the exposure each covers. A snapshot stands
for the atmosphere from its time until the next snapshot's time, and the
last one for as long again as the gap before it; an exposure outside that is
refused rather than filled from the nearest snapshot.
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
    compute_goft_fiasco,
    line_of_sight_velocity,
    synthesise_cubes,
)
from .utils import VELOCITY_CONVENTION, angle_to_distance

__all__ = ["SynthesisSettings", "RasterPlan", "Exposure", "AtmosphereSeries",
           "RasterSynthesiser"]

INTENSITY_UNIT = u.erg / u.s / u.cm**2 / u.sr / u.cm


@dataclass(frozen=True)
class SynthesisSettings:
    """
    What the synthesis needs to know, as ``synthesise-spectra`` takes it on the command line.

    Parameters
    ----------
    lines : sequence of str
        Line names such as ``"Fe12_195.1190"``.
    abundance : str
        The CHIANTI abundance set.
    vel_res, vel_lim : u.Quantity
        The velocity grid's spacing and half range.
    crop_y, crop_z : tuple of two quantities, optional
        Ranges to keep along y and z, in the atmosphere's own coordinates.
    precision : type
        np.float32 or np.float64.
    mass_per_electron : float, optional
        Atomic mass units per free electron for atmospheres that give only a
        mass density; None derives it from the abundance set.
    hdf5_dbase_root : str, optional
        The CHIANTI database for fiasco; None uses its default.
    n_workers : int
        Workers for the contribution functions; 0 uses every CPU.
    goft_temperature_chunk : int, optional
        Temperatures to pass fiasco at a time when computing the contribution
        functions, which lowers the memory they need; None passes the whole
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
        """The velocity bin centres, as synthesise-spectra builds them."""
        lim = self.vel_lim.to_value(u.cm / u.s)
        res = self.vel_res.to_value(u.cm / u.s)
        return np.arange(-lim, lim + res, res) * (u.cm / u.s)


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
    An observing plan: rasters of slit positions, exposure after exposure.

    Parameters
    ----------
    start : u.Quantity
        The simulation time at which the first exposure starts.
    steps : int
        Slit positions in one raster; 1 is a sit-and-stare.
    step : u.Quantity, optional
        The angle between neighbouring slit positions. None uses the slit
        width, so the positions abut.
    repeats : int
        How many rasters are made, one after another.
    cadence : u.Quantity, optional
        The time between the starts of consecutive exposures. None uses the
        exposure time, so one exposure starts as the last ends.
    centre : u.Quantity, optional
        The heliocentric x at which the raster is centred, as a length. None
        centres it on the atmosphere.
    """

    start: u.Quantity
    steps: int = 1
    step: Optional[u.Quantity] = None
    repeats: int = 1
    cadence: Optional[u.Quantity] = None
    centre: Optional[u.Quantity] = None

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

    def exposures(self, slit_width: u.Quantity, expos: u.Quantity,
                  atmosphere_centre: u.Quantity,
                  repeat: Optional[int] = None) -> List[Exposure]:
        """
        The exposures of one raster of the plan, for one slit width and exposure time.

        Parameters
        ----------
        slit_width : u.Quantity
            The slit width as an angle; the step when none was given.
        expos : u.Quantity
            The exposure time; the cadence when none was given.
        atmosphere_centre : u.Quantity
            The heliocentric x of the middle of the atmosphere; the raster's
            centre when none was given.
        repeat : int, optional
            Which raster, counting from 0. A plan of one raster needs none;
            a plan of several needs one, since each raster is observed on
            its own.
        """
        if expos <= 0:
            raise ValueError(f"The exposure time must be positive, got {expos}.")
        if repeat is None:
            if self.repeats > 1:
                raise ValueError(f"The plan has {self.repeats} rasters; say which one "
                                 f"with repeat=0 to {self.repeats - 1}.")
            repeat = 0
        if not 0 <= repeat < self.repeats:
            raise ValueError(f"repeat must be between 0 and {self.repeats - 1}, got {repeat}.")
        cadence = expos if self.cadence is None else self.cadence
        if cadence < expos:
            raise ValueError(f"The cadence, {cadence}, is shorter than the exposure, "
                             f"{expos}, so exposures would overlap.")
        step = angle_to_distance(slit_width if self.step is None else self.step).to(u.Mm)
        centre = (atmosphere_centre if self.centre is None else self.centre).to(u.Mm)
        first = centre - (self.steps - 1) / 2 * step
        exposures = []
        for position in range(self.steps):
            index = repeat * self.steps + position
            start = self.start + index * cadence
            exposures.append(Exposure(index, first + position * step, start, start + expos))
        return exposures


class AtmosphereSeries:
    """
    Atmosphere files ordered by the time each records.

    Every file must carry a time, and all must share one grid, since the
    columns of one snapshot stand in for those of another in an exposure.
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
        timed.sort(key=lambda item: item[0])
        times = np.array([t for t, _ in timed])
        if np.any(np.diff(times) <= 0):
            same = [str(p) for (t, p), (t2, _) in zip(timed[:-1], timed[1:]) if t == t2]
            raise ValueError(f"Two files of the series record the same time: {same[0]}")
        self.paths: List[Path] = [p for _, p in timed]
        self.times: u.Quantity = times * u.s
        # The grid comes from the first file; the others are checked as they
        # are read.
        edges = read_edges(self.paths[0])
        self.x_edges = edges["x"]
        self.y_edges = edges["y"]
        self.z_edges = edges["z"]

    def __len__(self) -> int:
        return len(self.paths)

    def valid_until(self) -> u.Quantity:
        """When each snapshot stops standing for the atmosphere: the next one's time,
        and for the last, as long again as the gap before it."""
        times = self.times.to_value(u.s)
        if times.size == 1:
            raise ValueError("A series of one snapshot cannot say how long it stands for; "
                             "give at least two, or observe a single snapshot as a static "
                             "atmosphere.")
        ends = np.append(times[1:], times[-1] + (times[-1] - times[-2]))
        return ends * u.s

    def coverage(self, start: u.Quantity, end: u.Quantity) -> List[Tuple[int, float]]:
        """
        Which snapshots an exposure from *start* to *end* falls in, and what fraction of it each covers.

        The fractions sum to one. An exposure reaching before the first
        snapshot or beyond the last one's span is refused.
        """
        t0, t1 = start.to_value(u.s), end.to_value(u.s)
        if t1 <= t0:
            raise ValueError(f"An exposure must end after it starts, got {start} to {end}.")
        begins = self.times.to_value(u.s)
        ends = self.valid_until().to_value(u.s)
        if t0 < begins[0] or t1 > ends[-1]:
            raise ValueError(
                f"An exposure from {t0:.3f} to {t1:.3f} s lies outside the series, which "
                f"runs from {begins[0]:.3f} to {ends[-1]:.3f} s (the last snapshot, at "
                f"{begins[-1]:.3f} s, stands for as long as the gap before it).")
        overlap = np.clip(np.minimum(ends, t1) - np.maximum(begins, t0), 0.0, None)
        fractions = overlap / (t1 - t0)
        return [(int(k), float(f)) for k, f in enumerate(fractions) if f > 0.0]


class RasterSynthesiser:
    """
    Synthesises the spectra a slit sees over a plan, one exposure at a time.

    The contribution functions are computed once. Each column of each
    snapshot is synthesised the first time an exposure needs it and kept,
    so a sweep over exposure times or slit widths reuses most of the work.

    Parameters
    ----------
    series : AtmosphereSeries
    settings : SynthesisSettings
    """

    def __init__(self, series: AtmosphereSeries, settings: SynthesisSettings):
        self.series = series
        self.settings = settings
        self.vel_grid = settings.velocity_grid()
        print(f"Computing contribution functions via fiasco for {len(settings.lines)} lines")
        self.goft, self.logT_grid, self.logN_grid = compute_goft_fiasco(
            list(settings.lines), abundance=settings.abundance,
            precision=settings.precision, n_workers=settings.n_workers,
            hdf5_dbase_root=settings.hdf5_dbase_root,
            temperature_chunk=settings.goft_temperature_chunk)
        self.goft_dbase_root = next(iter(self.goft.values()))["hdf5_dbase_root"]
        self._mass_per_electron: Optional[Tuple[float, str]] = None
        self._columns: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        self._wl_grids: Dict[str, u.Quantity] = {}
        self._rows: Optional[u.Quantity] = None
        self._row_pitch: Optional[u.Quantity] = None
        self.strips_synthesised = 0

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def atmosphere_centre(self) -> u.Quantity:
        edges = self.series.x_edges
        return 0.5 * (edges[0] + edges[-1])

    def columns_under(self, position: u.Quantity, slit_width: u.Quantity) -> Tuple[int, int, np.ndarray]:
        """
        The columns a slit at *position* covers: first index, last index plus one, and
        what fraction of the slit each covers, summing to one.
        """
        width = angle_to_distance(slit_width).to_value(u.Mm)
        edges = self.series.x_edges.to_value(u.Mm)
        low = position.to_value(u.Mm) - width / 2
        high = position.to_value(u.Mm) + width / 2
        if low < edges[0] or high > edges[-1]:
            raise ValueError(
                f"A slit {width:.4g} Mm wide at x = {position.to_value(u.Mm):.4g} Mm reaches "
                f"outside the atmosphere, which spans x = {edges[0]:.4g} to {edges[-1]:.4g} Mm.")
        overlap = np.clip(np.minimum(edges[1:], high) - np.maximum(edges[:-1], low), 0.0, None)
        # A slit edge that lands on a cell boundary, as it does when the slit
        # is a whole number of cells wide, must not pull in the cell beyond
        # on rounding: a column counts only when more than a millionth of a
        # cell of it is under the slit.
        tolerance = 1e-6 * np.diff(edges).min()
        inside = np.flatnonzero(overlap > tolerance)
        first, last = int(inside[0]), int(inside[-1]) + 1
        return first, last, overlap[first:last] / overlap[first:last].sum()

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

    def _synthesise_strip(self, snapshot: int, first: int, last: int) -> None:
        """Synthesise columns *first* to *last* of one snapshot into the cache."""
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

    def column_spectra(self, snapshot: int, column: int) -> Dict[str, np.ndarray]:
        """The spectra of one column of one snapshot, ``(rows, wavelength)`` per line."""
        if (snapshot, column) not in self._columns:
            self._synthesise_strip(snapshot, column, column + 1)
        return self._columns[(snapshot, column)]

    def exposure_spectra(self, exposure: Exposure, slit_width: u.Quantity) -> Dict[str, np.ndarray]:
        """
        What the slit collects in one exposure: the columns under it, averaged over
        the slit, and the snapshots it spans, weighted by the time each covers.
        """
        first, last, fractions = self.columns_under(exposure.position, slit_width)
        spectra: Dict[str, np.ndarray] = {}
        for snapshot, weight in self.series.coverage(exposure.start, exposure.end):
            missing = [c for c in range(first, last) if (snapshot, c) not in self._columns]
            if missing:
                self._synthesise_strip(snapshot, min(missing), max(missing) + 1)
            for column, fraction in zip(range(first, last), fractions):
                for name, si in self._columns[(snapshot, column)].items():
                    contribution = weight * fraction * si
                    spectra[name] = contribution if name not in spectra else spectra[name] + contribution
        return spectra

    def line_cubes(self, plan: RasterPlan, slit_width: u.Quantity,
                   expos: u.Quantity, repeat: Optional[int] = None) -> Dict[str, NDCube]:
        """
        One cube per line for one raster of the plan: ``(rows, exposures, wavelength)``.

        The columns are the raster's exposures in order, at the slit
        positions, which advance by the step. A sit-and-stare is a raster of
        one exposure, so its cube has one column, as wide as the slit; its
        repeats are separate cubes, one per exposure.
        """
        exposures = plan.exposures(slit_width, expos, self.atmosphere_centre(), repeat)
        collected = [self.exposure_spectra(exposure, slit_width) for exposure in exposures]
        positions = u.Quantity([e.position for e in exposures]).to(u.Mm)
        pitch = (positions[1] - positions[0] if plan.steps > 1
                 else angle_to_distance(slit_width).to(u.Mm))
        rows = self._rows.to(u.Mm)
        row_pitch = self._row_pitch.to(u.Mm)
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
        n_columns, n_rows = positions.size, rows.size
        cubes = {}
        for name in self.settings.lines:
            data = np.stack([c[name] for c in collected], axis=1)
            wl_grid = self._wl_grids[name].to(u.cm)
            # The reference pixel sits at the middle of each spatial axis, as
            # the line cubes of a static synthesis have it, with the value
            # the grid has there.
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
                    reference_line: str, repeat: Optional[int] = None) -> NDCube:
        """The lines summed onto the reference line's grid, as the instrument run takes it."""
        if reference_line not in self.settings.lines:
            raise ValueError(f"The reference line {reference_line!r} is not among the "
                             f"synthesised lines {list(self.settings.lines)}.")
        return sum_line_cubes(self.line_cubes(plan, slit_width, expos, repeat), reference_line)
