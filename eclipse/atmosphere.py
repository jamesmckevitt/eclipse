"""
Atmosphere input for the synthesis, and its HDF5 file.

An `Atmosphere` holds the temperature, density and velocity of every cell
and the cell edges, from any simulation code. This module reads and writes
it, and converts mass density to electron density with the synthesis's
abundances. The file layout is described on the Files page of the docs.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence

import astropy.constants as const
import astropy.units as u
import h5py
import numpy as np
from astropy.wcs import WCS
from ndcube import NDCube

from .utils import element_data, require_downsample_divides, require_uniform_grid

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "AXES",
    "Atmosphere",
    "edges_from_centres",
    "read_atmosphere",
    "read_time",
    "read_edges",
    "write_atmosphere",
    "describe_atmosphere_file",
    "mass_per_electron",
    "mass_per_electron_from_abundances",
    "require_mass_per_electron",
]

FORMAT_NAME = "eclipse-atmosphere"
FORMAT_VERSION = 1

# The cubes are (z, y, x), so numpy axis 0 is z.
AXES = ("x", "y", "z")
NUMPY_AXIS = {"x": 2, "y": 1, "z": 0}
CTYPE = {"x": "SOLX", "y": "SOLY", "z": "SOLZ"}
EDGES = {axis: f"{axis}_edges" for axis in AXES}

CUBES = ("temperature", "mass_density", "electron_density",
         "velocity_x", "velocity_y", "velocity_z")
UNITS = {
    "temperature": u.K,
    "mass_density": u.g / u.cm**3,
    "electron_density": u.cm**-3,
    "velocity_x": u.cm / u.s,
    "velocity_y": u.cm / u.s,
    "velocity_z": u.cm / u.s,
    "x_edges": u.Mm, "y_edges": u.Mm, "z_edges": u.Mm,
    "time": u.s,
}

# fiasco needs a temperature to build an Ion, but Ion.abundance reads only
# the abundance table, so any fixed value will do.
_ABUNDANCE_PROBE_TEMPERATURE = 1.0e6 * u.K


def _quantity(value, name: str, ndim: Optional[int] = None) -> u.Quantity:
    """
    `value`, checked to be a finite Quantity of the right kind for `name`.

    `ndim`, if given, is the number of axes it must have.
    """
    unit = UNITS[name]
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an astropy Quantity in a unit "
                        f"convertible to {unit}, got {type(value).__name__}.")
    if not value.unit.is_equivalent(unit):
        raise u.UnitConversionError(
            f"{name} must be in a unit convertible to {unit}, got {value.unit}.")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}.")
    if not np.all(np.isfinite(value.value)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return value


def _check_edges(edges, name: str) -> u.Quantity:
    """Check that cell edges increase and bound at least one cell."""
    edges = _quantity(edges, name, ndim=1)
    if edges.size < 2:
        raise ValueError(f"{name} must bound at least 1 cell, so have "
                         f"at least 2 values, got {edges.size}.")
    if np.any(np.diff(edges.value) <= 0):
        raise ValueError(f"{name} must increase; ECLIPSE's cubes are "
                         f"(z, y, x) with every axis ascending.")
    return edges


def edges_from_centres(centres: u.Quantity) -> u.Quantity:
    """
    Cell edges from cell centres.

    Each inner edge is halfway between two centres. Each outer edge is as
    far from the end centre as that cell's inner edge. On an even grid every
    cell then has the grid spacing. On an uneven grid this is only one
    reasonable choice, so give the true edges if your code knows them.

    Parameters
    ----------
    centres : u.Quantity
        Cell centres along one axis, two or more.

    Returns
    -------
    u.Quantity
        Edges, one more than the centres.
    """
    centres = np.atleast_1d(centres)
    if centres.size < 2:
        raise ValueError("At least 2 centres are needed to place edges "
                         "between them.")
    values = centres.value
    inner = 0.5 * (values[1:] + values[:-1])
    first = values[0] - (inner[0] - values[0])
    last = values[-1] + (values[-1] - inner[-1])
    return np.concatenate([[first], inner, [last]]) * centres.unit


@dataclass(frozen=True)
class Atmosphere:
    """
    One snapshot of a simulated atmosphere, for the synthesis.

    Every cube is `(nz, ny, nx)`. The edge arrays `z_edges`, `y_edges` and
    `x_edges` give the cell boundaries and are one longer than the cubes
    along their axis. Velocities are positive towards increasing coordinate.
    Either density may be given; if only the mass density is, the synthesis
    derives the electron density from it.

    Parameters
    ----------
    temperature : u.Quantity
        Temperature of every cell, `(nz, ny, nx)`.
    x_edges, y_edges, z_edges : u.Quantity
        Cell boundaries along each axis, 1D and increasing.
    mass_density, electron_density : u.Quantity, optional
        At least one is needed.
    velocity_x, velocity_y, velocity_z : u.Quantity, optional
        Velocity components. The synthesis needs the one along its line of
        sight.
    time : u.Quantity, optional
        Simulation time of this snapshot.
    source : str, optional
        Free text naming the simulation, kept in the file and the results.
    """

    temperature: u.Quantity
    x_edges: u.Quantity
    y_edges: u.Quantity
    z_edges: u.Quantity
    mass_density: Optional[u.Quantity] = None
    electron_density: Optional[u.Quantity] = None
    velocity_x: Optional[u.Quantity] = None
    velocity_y: Optional[u.Quantity] = None
    velocity_z: Optional[u.Quantity] = None
    time: Optional[u.Quantity] = None
    source: str = ""

    def __post_init__(self):
        if self.temperature is None:
            raise ValueError("An atmosphere needs a temperature.")
        for axis in AXES:
            _check_edges(getattr(self, EDGES[axis]), EDGES[axis])
        shape = tuple(getattr(self, EDGES[axis]).size - 1 for axis in ("z", "y", "x"))
        for name in CUBES:
            cube = getattr(self, name)
            if cube is None:
                continue
            _quantity(cube, name, ndim=3)
            if cube.shape != shape:
                raise ValueError(
                    f"{name} has shape {cube.shape} but the edges bound "
                    f"(nz, ny, nx) = {shape} cells. The cubes are (z, y, x): "
                    f"the first axis is height.")
        # Cells at or below zero temperature would otherwise be dropped
        # silently from the synthesis, as log10 is undefined there. An empty
        # cell may have zero density.
        if np.any(self.temperature.value <= 0):
            raise ValueError("temperature must be above zero in every cell, got "
                             f"{self.temperature.min()} at least once.")
        for name in ("mass_density", "electron_density"):
            cube = getattr(self, name)
            if cube is not None and np.any(cube.value < 0):
                raise ValueError(f"{name} cannot be negative, got {cube.min()} at least once.")
        if self.mass_density is None and self.electron_density is None:
            raise ValueError("An atmosphere needs a mass_density or an "
                             "electron_density.")
        if self.time is not None:
            _quantity(self.time, "time", ndim=0)
        if not isinstance(self.source, str):
            raise TypeError(f"source must be a string, got "
                            f"{type(self.source).__name__}.")

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the cubes, `(nz, ny, nx)`."""
        return self.temperature.shape

    def edges(self, axis: str) -> u.Quantity:
        """Cell boundaries along `axis`."""
        return getattr(self, EDGES[_check_axis(axis)])

    def coordinate(self, axis: str) -> u.Quantity:
        """Cell centres along `axis`."""
        edges = self.edges(axis)
        return 0.5 * (edges[1:] + edges[:-1])

    def cell_thickness(self, axis: str) -> u.Quantity:
        """Size of every cell along `axis`."""
        return np.diff(self.edges(axis))

    def is_uniform(self, axis: str) -> bool:
        """Whether the cells along `axis` all have the same size."""
        try:
            require_uniform_grid(self.edges(axis), EDGES[axis])
        except ValueError:
            return False
        return True

    def spacing(self, axis: str) -> u.Quantity:
        """Cell size along `axis`, which must be evenly spaced."""
        edges = self.edges(axis)
        return require_uniform_grid(edges, EDGES[axis]) * edges.unit

    def velocity(self, axis: str) -> u.Quantity:
        """Velocity component along `axis`."""
        velocity = getattr(self, f"velocity_{_check_axis(axis)}")
        if velocity is None:
            raise ValueError(f"The atmosphere has no velocity_{axis}, which a "
                             f"line of sight along {axis} needs.")
        return velocity

    # ------------------------------------------------------------------
    # Density
    # ------------------------------------------------------------------
    def electron_density_from(self, mass_per_electron_amu: float) -> u.Quantity:
        """
        Electron density, derived from the mass density if not given.

        Parameters
        ----------
        mass_per_electron_amu : float
            Plasma mass per free electron, in atomic mass units, about 1.16
            for coronal abundances. Not used if the atmosphere has an
            electron density.

        Returns
        -------
        u.Quantity
            Electron density in every cell, in cm-3.
        """
        if self.electron_density is not None:
            return self.electron_density
        require_mass_per_electron(mass_per_electron_amu)
        return (self.mass_density / (mass_per_electron_amu * const.u)).to(u.cm**-3)

    # ------------------------------------------------------------------
    # Selecting part of the box
    # ------------------------------------------------------------------
    def cropped(self, x=None, y=None, z=None) -> "Atmosphere":
        """
        Part of the box within the given ranges.

        Each range is `(low, high)`, with units, in the atmosphere's own
        coordinates. A cell is kept if any part of it is inside the range. A
        bound on a cell edge does not keep the cell beyond it.
        """
        item = [slice(None)] * 3
        edges = {}
        for axis, bounds in (("x", x), ("y", y), ("z", z)):
            if bounds is None:
                continue
            all_edges = self.edges(axis)
            low, high = (u.Quantity(b).to_value(all_edges.unit) for b in bounds)
            if not low < high:
                raise ValueError(f"The {axis} range must have low < high, got "
                                 f"{bounds}.")
            values = all_edges.value
            # Round-number bounds often land on cell boundaries. So that
            # rounding does not decide whether the cell beyond is kept, a cell
            # counts as inside only if more than a millionth of it is.
            tolerance = 1e-6 * np.diff(values).min()
            inside = np.flatnonzero((values[1:] - low > tolerance)
                                    & (high - values[:-1] > tolerance))
            if inside.size == 0:
                raise ValueError(
                    f"No cells lie within {axis} = {bounds}; the atmosphere "
                    f"covers {all_edges[0]:.6g} to {all_edges[-1]:.6g}.")
            first, last = int(inside[0]), int(inside[-1]) + 1
            item[NUMPY_AXIS[axis]] = slice(first, last)
            edges[EDGES[axis]] = all_edges[first:last + 1]
        item = tuple(item)
        cubes = {name: None if getattr(self, name) is None
                 else getattr(self, name)[item] for name in CUBES}
        return replace(self, **edges, **cubes)

    def downsampled(self, factor: int) -> "Atmosphere":
        """
        Atmosphere with every `factor`-th cell along each axis.

        Each kept cell grows to cover the cells it replaces, so the box keeps
        its size. `factor` must divide every dimension.
        """
        require_downsample_divides(self.shape, factor)
        if factor == 1:
            return self
        item = (slice(None, None, factor),) * 3
        edges = {EDGES[axis]: self.edges(axis)[::factor] for axis in AXES}
        cubes = {name: None if getattr(self, name) is None
                 else getattr(self, name)[item] for name in CUBES}
        return replace(self, **edges, **cubes)

    # ------------------------------------------------------------------
    # Cubes for the synthesis
    # ------------------------------------------------------------------
    def nonuniform_axes(self) -> list[str]:
        """Axes whose cells are not all the same size."""
        return [axis for axis in AXES if not self.is_uniform(axis)]

    def to_ndcube(self, quantity: u.Quantity) -> NDCube:
        """
        `quantity` as an NDCube with the atmosphere's coordinates.

        `quantity` must have the atmosphere's shape. The coordinates are
        linear. An axis whose cells are not all the same size gets their mean
        size, and is listed in the cube's `meta["nonuniform_axes"]`.
        """
        # Only the line of sight may be uneven: it is integrated out with the
        # true size of every cell, and its coordinates are never used.
        if quantity.shape != self.shape:
            raise ValueError(f"Expected a cube of shape {self.shape}, got "
                             f"{quantity.shape}.")
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = [CTYPE[axis] for axis in AXES]
        wcs.wcs.cunit = ["Mm"] * 3
        wcs.wcs.crpix = [1, 1, 1]
        wcs.wcs.crval = [self.coordinate(axis)[0].to_value(u.Mm) for axis in AXES]
        wcs.wcs.cdelt = [self.cell_thickness(axis).mean().to_value(u.Mm) for axis in AXES]
        meta = {"source": self.source, "nonuniform_axes": self.nonuniform_axes()}
        return NDCube(quantity.value, wcs=wcs, unit=quantity.unit, meta=meta)

    def describe(self) -> str:
        """A few lines describing the atmosphere."""
        present = [name for name in CUBES if getattr(self, name) is not None]
        return _describe({axis: self.edges(axis) for axis in AXES}, present,
                         self.time, self.source)


def _check_axis(axis: str) -> str:
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}.")
    return axis


def _describe(edges: Dict[str, u.Quantity], cubes: Sequence[str],
              time: Optional[u.Quantity], source: str) -> str:
    """Description shared by an Atmosphere and a file on disk."""
    shape = tuple(edges[axis].size - 1 for axis in ("z", "y", "x"))
    lines = [f"  Shape (nz, ny, nx): {shape}"]
    for axis in AXES:
        axis_edges = edges[axis].to(u.Mm)
        extent = f"{axis_edges[0]:.3f} to {axis_edges[-1]:.3f}"
        thickness = np.diff(axis_edges)
        try:
            spacing = require_uniform_grid(axis_edges, EDGES[axis]) * u.Mm
        except ValueError:
            lines.append(f"  {axis}: {extent}, cells from {thickness.min():.4f} "
                         f"to {thickness.max():.4f}")
        else:
            lines.append(f"  {axis}: {extent}, cells of {spacing:.4f}")
    densities = [name for name in ("mass_density", "electron_density") if name in cubes]
    lines.append(f"  Density: {' and '.join(densities)}")
    velocities = [axis for axis in AXES if f"velocity_{axis}" in cubes]
    lines.append(f"  Velocities: {', '.join(velocities) if velocities else 'none'}")
    if time is not None:
        lines.append(f"  Time: {time.to(u.s):.3f}")
    if source:
        lines.append(f"  Source: {source}")
    return "\n".join(lines)


def require_mass_per_electron(value) -> float:
    """`value` as a mass per free electron in atomic mass units; must be finite and positive."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = np.nan
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"The mass per electron must be a finite, positive number "
                         f"of atomic mass units, got {value!r}.")
    return number


# ----------------------------------------------------------------------
# HDF5
# ----------------------------------------------------------------------
def _new_partial(path: Path) -> Path:
    """
    New empty file with a unique name beside `path`, to write `path` into.

    It is created normally, so the umask and the directory's default
    permissions apply to it, and so to `path`, as colleagues sharing a
    project directory expect.
    """
    while True:
        suffix = f".{secrets.token_hex(4)}.part"
        # Truncate the stem so the name fits the 255 bytes most filesystems allow.
        stem = path.name.encode()[:255 - len(suffix)].decode(errors="ignore")
        partial = path.with_name(stem + suffix)
        try:
            os.close(os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666))
        except FileExistsError:
            continue
        return partial


@contextmanager
def _replacing(path: Path):
    """
    New file beside `path` to write, which replaces `path` once written.

    A write cut short by an error, a full disk or the job being killed
    leaves any file already at `path` unchanged. Writing over it directly
    would leave a broken file and lose the old one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Follow a symlink to its target, as writing to the link would.
    destination = Path(os.path.realpath(path)) if path.is_symlink() else path
    # Refuse to replace a read-only file, as writing into it would fail.
    if destination.exists() and not os.access(destination, os.W_OK):
        raise PermissionError(f"{destination} is read-only, so it is not written over.")
    partial = _new_partial(destination)
    try:
        yield partial
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _is_pickle(path: Path) -> bool:
    """Whether `path` starts with dill's protocol marker, as ECLIPSE's pickles do."""
    with open(path, "rb") as handle:
        return handle.read(1) == b"\x80"


def write_atmosphere(atmosphere: Atmosphere, path: str | Path,
                     compression: Optional[str] = None) -> Path:
    """
    Write an atmosphere file.

    The file is written under a temporary name beside `path` and then moved
    into place, so a run stopped part way leaves any existing file unchanged.

    Parameters
    ----------
    atmosphere : Atmosphere
        Atmosphere to write.
    path : str or Path
        File to write. An existing file is replaced.
    compression : str, optional
        HDF5 compression filter for the cubes, such as `"gzip"`. Default
        None, uncompressed, which reads fastest.

    Returns
    -------
    Path
        Path of the file written.
    """
    path = Path(path)
    with _replacing(path) as partial, h5py.File(partial, "w") as f:
        f.attrs["format"] = FORMAT_NAME
        f.attrs["version"] = FORMAT_VERSION
        f.attrs["source"] = atmosphere.source
        for axis in AXES:
            _write_dataset(f, EDGES[axis], atmosphere.edges(axis))
        for name in CUBES:
            cube = getattr(atmosphere, name)
            if cube is not None:
                _write_dataset(f, name, cube, compression=compression)
        if atmosphere.time is not None:
            _write_dataset(f, "time", atmosphere.time)
    return path


def _write_dataset(f: h5py.File, name: str, quantity: u.Quantity,
                   compression: Optional[str] = None) -> None:
    dataset = f.create_dataset(name, data=quantity.value, compression=compression)
    dataset.attrs["unit"] = quantity.unit.to_string()


def read_atmosphere(path: str | Path,
                    velocities: Optional[Sequence[str]] = None,
                    columns: Optional[slice] = None) -> Atmosphere:
    """
    Read an atmosphere file.

    Parameters
    ----------
    path : str or Path
        Atmosphere file.
    velocities : sequence of str, optional
        Velocity components to read, such as `("z",)` for a view along z.
        Default None, for every component in the file. Asking for one the
        file lacks is an error.
    columns : slice, optional
        Cells along x to read, as a slice of their index. Only those columns
        are read from disk, so reading a few columns of a large box is quick.
        Default None, for all of them.

    Returns
    -------
    Atmosphere
        Atmosphere read from the file.
    """
    path = Path(path)
    wanted = AXES if velocities is None else tuple(_check_axis(axis) for axis in velocities)
    with _open_to_read(path) as f:
        _check_format(f, path)
        # Check for missing velocities before reading any cube, since the
        # cubes of a large simulation take gigabytes.
        if velocities is not None:
            for axis in wanted:
                if f"velocity_{axis}" not in f:
                    raise ValueError(f"{path} has no velocity_{axis}, which a line "
                                     f"of sight along {axis} needs.")
        fields = {EDGES[axis]: _read_dataset(f, EDGES[axis]) for axis in AXES}
        if columns is not None:
            nx = fields[EDGES["x"]].size - 1
            first, last, step = columns.indices(nx)
            if step != 1 or last <= first:
                raise ValueError(f"columns must select a contiguous, non-empty "
                                 f"range of the {nx} cells along x, got {columns}.")
            fields[EDGES["x"]] = fields[EDGES["x"]][first:last + 1]
            columns = slice(first, last)
        fields["temperature"] = _read_dataset(f, "temperature", columns)
        for name in ("mass_density", "electron_density"):
            if name in f:
                fields[name] = _read_dataset(f, name, columns)
        if "time" in f:
            fields["time"] = _read_dataset(f, "time")
        for axis in wanted:
            name = f"velocity_{axis}"
            if name in f:
                fields[name] = _read_dataset(f, name, columns)
        source = _text_attribute(f.attrs, "source", "")
    return Atmosphere(source=str(source), **fields)


def read_time(path: str | Path) -> Optional[u.Quantity]:
    """
    Snapshot time in an atmosphere file, or None if it has none.

    The time gets the same checks as when reading the whole file, so a time
    that is NaN, infinite or not a single value is refused here too.
    """
    path = Path(path)
    with _open_to_read(path) as f:
        _check_format(f, path)
        if "time" not in f:
            return None
        time = _read_dataset(f, "time")
    try:
        return _quantity(time, "time", ndim=0)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None


def read_edges(path: str | Path) -> Dict[str, u.Quantity]:
    """Cell boundaries in an atmosphere file, keyed by axis, without reading the cubes."""
    path = Path(path)
    with _open_to_read(path) as f:
        _check_format(f, path)
        return {axis: _read_dataset(f, EDGES[axis]) for axis in AXES}


def describe_atmosphere_file(path: str | Path) -> str:
    """
    Description of an atmosphere file, without reading its cubes.

    This is what `eclipse-atmosphere info` prints. Only the attributes, the
    edges, the time and the cubes' names, shapes and units are read, so it
    is quick for any size of file. A file whose layout or units cannot be
    read is refused. The values in the cubes are checked only when they are
    read.

    Parameters
    ----------
    path : str or Path
        Atmosphere file.

    Returns
    -------
    str
        A few lines describing the file.
    """
    path = Path(path)
    with _open_to_read(path) as f:
        _check_format(f, path)
        edges = {axis: _read_dataset(f, EDGES[axis]) for axis in AXES}
        try:
            for axis in AXES:
                _check_edges(edges[axis], EDGES[axis])
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from None
        shape = tuple(edges[axis].size - 1 for axis in ("z", "y", "x"))
        cubes = [name for name in CUBES if name in f]
        for name in cubes:
            if f[name].shape != shape:
                raise ValueError(f"{name} in {path} has shape {f[name].shape} but "
                                 f"the edges bound (nz, ny, nx) = {shape} cells.")
            unit = _dataset_unit(f, name)
            if not unit.is_equivalent(UNITS[name]):
                raise u.UnitConversionError(f"{name} in {path} must be in a unit convertible "
                                            f"to {UNITS[name]}, got {unit}.")
        if "temperature" not in cubes:
            raise ValueError(f"{path} has no 'temperature' dataset.")
        if "mass_density" not in cubes and "electron_density" not in cubes:
            raise ValueError(f"{path} has no density: an atmosphere needs a "
                             f"mass_density or an electron_density.")
        time = _read_dataset(f, "time") if "time" in f else None
        if time is not None:
            try:
                time = _quantity(time, "time", ndim=0)
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from None
        source = _text_attribute(f.attrs, "source", "")
    return _describe(edges, cubes, time, str(source))


def _text_attribute(attrs, name: str, default=None):
    """
    Text attribute as a string.

    It may be stored as a string, as bytes, or, as IDL and some other
    writers do, as a one-element array of either.
    """
    value = attrs.get(name, default)
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(()).item()
    if isinstance(value, bytes):
        value = value.decode()
    return value


def _open_to_read(path: Path, kind: str = "atmosphere") -> h5py.File:
    """
    Open `path` for reading.

    Every ECLIPSE `kind` file is HDF5. An existing file that is not is
    refused with a plain message, rather than h5py's "file signature not
    found".
    """
    path = Path(path)
    if path.is_file() and not h5py.is_hdf5(path):
        raise ValueError(f"{path} is not an HDF5 file, so it is not an ECLIPSE {kind} file.")
    return h5py.File(path, "r")


def _check_format(f: h5py.File, path: Path, kind: str = "atmosphere",
                  format_name: str = FORMAT_NAME,
                  format_version: int = FORMAT_VERSION,
                  documented: bool = True) -> None:
    """
    Check that `f` is an ECLIPSE `kind` file of the version this ECLIPSE reads.

    `documented` says whether the docs give the layout, as they do for a
    file users may write themselves.
    """
    name = _text_attribute(f.attrs, "format")
    if name != format_name:
        layout = (f" See ECLIPSE's documentation of the {kind} file for the layout."
                  if documented else "")
        raise ValueError(
            f"{path} is not an ECLIPSE {kind} file: its 'format' "
            f"attribute is {name!r}, not {format_name!r}.{layout}")
    version = f.attrs.get("version")
    if version is None:
        raise ValueError(f"{path} has no 'version' attribute; an ECLIPSE "
                         f"{kind} file has version {format_version}.")
    # The version must be an integer, which some HDF5 writers store as a
    # one-element array. A string or float is refused even if it reads as 1.
    value = np.asarray(version)
    if value.size != 1 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{path} has a 'version' attribute of {version} "
                         f"({type(version).__name__}); it must be the integer "
                         f"{format_version}.")
    if value.item() != format_version:
        raise ValueError(f"{path} is {kind} format version {value.item()}; "
                         f"this ECLIPSE reads version {format_version}.")


def _read_dataset(f: h5py.Group, name: str,
                  columns: Optional[slice] = None,
                  units: Dict[str, u.Unit] = UNITS,
                  axis: int = -1) -> u.Quantity:
    """
    Dataset of a file or group, with its unit.

    If given, `columns` reads only that slice of the x axis, `axis`.
    """
    unit = _dataset_unit(f, name, units)
    dataset = f[name]
    if columns is None:
        values = dataset[()]
    else:
        index = [slice(None)] * dataset.ndim
        index[axis] = columns
        values = dataset[tuple(index)]
    return u.Quantity(values, unit)


def _dataset_unit(f: h5py.Group, name: str, units: Dict[str, u.Unit] = UNITS) -> u.Unit:
    """Unit of a dataset in a file or group, from its 'unit' attribute, without reading it."""
    where = f.file.filename if f.name == "/" else f"{f.file.filename}, {f.name}"
    if name not in f:
        raise ValueError(f"{where} has no '{name}' dataset.")
    unit = _text_attribute(f[name].attrs, "unit")
    if unit is None:
        raise ValueError(f"'{name}' in {where} has no 'unit' attribute. "
                         f"Every dataset needs one, for instance "
                         f"'{units[name]}'.")
    return u.Unit(unit)


# ----------------------------------------------------------------------
# From mass density to electron density
# ----------------------------------------------------------------------
def mass_per_electron_from_abundances(abundances: Dict[str, float]) -> float:
    """
    Mass per free electron of a fully ionised plasma, in atomic mass units.

    This is the sum over elements of abundance times atomic weight, divided
    by the sum of abundance times atomic number.

    Parameters
    ----------
    abundances : dict
        Atoms of each element per hydrogen atom, keyed by symbol, including
        hydrogen itself.
    """
    if "H" not in abundances:
        raise ValueError("The abundances must include hydrogen ('H').")
    mass = 0.0
    electrons = 0.0
    for symbol, per_hydrogen in abundances.items():
        atomic_number, atomic_weight = element_data(symbol)
        mass += per_hydrogen * atomic_weight
        electrons += per_hydrogen * atomic_number
    return mass / electrons


def _offer_database_build(hdf5_dbase_root: Optional[str] = None) -> None:
    """
    Offer to build fiasco's CHIANTI database if it is missing.

    A new user's first run needs this. fiasco asks only when an `Ion` is
    made, so code that reads the database before that, or makes its ions in
    worker processes with no terminal to answer from, calls this first in
    the process the user runs.
    """
    import fiasco
    from fiasco.util import check_database

    root = fiasco.defaults["hdf5_dbase_root"] if hdf5_dbase_root is None else hdf5_dbase_root
    check_database(root)
    # The user declined: raise here rather than have each worker ask again.
    if not Path(root).is_file():
        raise FileNotFoundError(
            f"There is no CHIANTI database for fiasco at {root}, and it was not built. "
            f"Answer yes when asked to build it, or build it as fiasco's documentation "
            f"describes.")


@lru_cache(maxsize=None)
def mass_per_electron(abundance: str, hdf5_dbase_root: Optional[str] = None) -> float:
    """
    Mass per free electron for a CHIANTI abundance set, in atomic mass units.

    It converts a simulation's mass density to the electron density the
    contribution functions need, when the simulation gives none. Every
    element is assumed fully ionised. Hydrogen and helium, which give almost
    all the electrons, are fully ionised above about log T 5.2, so there the
    electron density is within about 1 per cent even where the metals are
    not. Cooler cells get too high an electron density, but emit few of the
    lines ECLIPSE synthesises. The value is about 1.16 for coronal
    abundances, against 1.29 for a neutral gas, so an atmosphere read with
    the neutral value has about 20 per cent too little emission measure.

    Parameters
    ----------
    abundance : str
        CHIANTI abundance set, as given to `--abundance`.
    hdf5_dbase_root : str, optional
        fiasco database to read it from; None uses fiasco's default.
    """
    import fiasco
    from fiasco.util.exceptions import MissingDatasetException

    _offer_database_build(hdf5_dbase_root)
    ion_kwargs = {} if hdf5_dbase_root is None else {"hdf5_dbase_root": hdf5_dbase_root}
    ions = fiasco.list_ions(hdf5_dbase_root)
    abundances = {}
    for symbol in fiasco.list_elements(hdf5_dbase_root):
        # The abundance belongs to the element but is read through an ion,
        # so any existing ion of it will do.
        ion_name = next(name for name in ions if name.startswith(f"{symbol} "))
        ion = fiasco.Ion(ion_name, _ABUNDANCE_PROBE_TEMPERATURE,
                         abundance=abundance, **ion_kwargs)
        try:
            abundances[symbol] = float(ion.abundance)
        except MissingDatasetException:
            # The set does not list this element, so it has none.
            continue
    if not abundances:
        raise ValueError(f"No element has an abundance in the set "
                         f"{abundance!r}; is the name right?")
    return mass_per_electron_from_abundances(abundances)


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for eclipse-atmosphere."""
    parser = argparse.ArgumentParser(
        prog="eclipse-atmosphere", description="Inspect ECLIPSE atmosphere files.")
    commands = parser.add_subparsers(dest="command", required=True)
    info = commands.add_parser("info", help="Describe an atmosphere file")
    info.add_argument("atmosphere", type=str, help="The file to describe")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point: `eclipse-atmosphere info FILE` describes a file without loading its cubes."""
    args = build_parser().parse_args(argv)
    print(args.atmosphere)
    print(describe_atmosphere_file(args.atmosphere))


if __name__ == "__main__":
    main(sys.argv[1:])
