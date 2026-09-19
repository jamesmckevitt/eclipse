"""
The atmosphere ECLIPSE synthesises from, and the HDF5 file it reads it from.

ECLIPSE needs little from a simulation: the temperature and density of every
cell, the velocity along the line of sight, and where the cells are. This
module holds those as one object whatever code produced them, reads and
writes them as HDF5, and turns a mass density into an electron density with
the abundances the synthesis itself uses. Anything that can write HDF5 can
feed ECLIPSE without ECLIPSE knowing that code's own output format.

File layout, version 1
----------------------
Root attributes:

- ``format``: ``"eclipse-atmosphere"``
- ``version``: ``1``
- ``source``: free text naming the simulation, optional

Datasets, each with a ``unit`` attribute astropy can parse. The cubes are
``(nz, ny, nx)`` in C order, so ``cube[k]`` is a horizontal slice indexed
``[y, x]``:

- ``x_edges``, ``y_edges``, ``z_edges``: the positions of the cell
  boundaries along each axis, 1D and increasing, one longer than the cubes
  along that axis. A code that knows only its cell centres can place the
  edges halfway between them with :func:`edges_from_centres`.
- ``temperature``: ``(nz, ny, nx)``
- ``mass_density`` and/or ``electron_density``: ``(nz, ny, nx)``
- ``velocity_x``, ``velocity_y``, ``velocity_z``: ``(nz, ny, nx)``, positive
  towards increasing coordinate; only the one along the line of sight is
  needed
- ``time``: scalar, optional; only a time series needs it

The two axes that become the image must be evenly spaced. The axis along
the line of sight may be stretched, since it is integrated out cell by cell.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Sequence

import astropy.constants as const
import astropy.units as u
import h5py
import numpy as np
from astropy.wcs import WCS
from mendeleev import element
from ndcube import NDCube

from .utils import require_downsample_divides, require_uniform_grid

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "AXES",
    "Atmosphere",
    "edges_from_centres",
    "read_atmosphere",
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

# fiasco needs a temperature to build an Ion, but an element's abundance does
# not depend on it: Ion.abundance reads the abundance table alone. Any value
# does, so one is fixed here rather than asked for.
_ABUNDANCE_PROBE_TEMPERATURE = 1.0e6 * u.K


def _quantity(value, name: str, ndim: Optional[int] = None) -> u.Quantity:
    """*value* as a Quantity of the right kind for *name*, with *ndim* axes."""
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


def edges_from_centres(centres: u.Quantity) -> u.Quantity:
    """
    Cell edges for cells whose centres are *centres*.

    Each edge is placed halfway between two centres, and the outer edges as
    far beyond the end cells as their inner edges are within them. On an
    even grid that gives every cell the grid spacing; on a stretched grid
    it is one reasonable choice, so a code that knows its edges should give
    those instead.
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
    One snapshot of a simulated atmosphere, in the form the synthesis needs.

    Every cube is ``(nz, ny, nx)`` against the edge arrays ``z_edges``,
    ``y_edges`` and ``x_edges``, which give the cell boundaries and are one
    longer than the cubes along their axis. Velocities are positive towards
    increasing coordinate. Either density may be given; when only the mass
    density is, the synthesis derives the electron density from it.

    Parameters
    ----------
    temperature : u.Quantity
        Temperature of every cell, ``(nz, ny, nx)``.
    x_edges, y_edges, z_edges : u.Quantity
        Positions of the cell boundaries along each axis, 1D and increasing.
    mass_density, electron_density : u.Quantity, optional
        At least one is needed.
    velocity_x, velocity_y, velocity_z : u.Quantity, optional
        The velocity components. The synthesis needs the one along its line
        of sight.
    time : u.Quantity, optional
        The simulation time of this snapshot.
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
            name = EDGES[axis]
            edges = _quantity(getattr(self, name), name, ndim=1)
            if edges.size < 3:
                raise ValueError(f"{name} must bound at least 2 cells, so have "
                                 f"at least 3 values, got {edges.size}: a "
                                 f"single cell has nothing to integrate over.")
            if np.any(np.diff(edges.value) <= 0):
                raise ValueError(f"{name} must increase; ECLIPSE's cubes are "
                                 f"(z, y, x) with every axis ascending.")
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
        """``(nz, ny, nx)``."""
        return self.temperature.shape

    def edges(self, axis: str) -> u.Quantity:
        """The cell boundaries along *axis*."""
        return getattr(self, EDGES[_check_axis(axis)])

    def coordinate(self, axis: str) -> u.Quantity:
        """The cell centres along *axis*."""
        edges = self.edges(axis)
        return 0.5 * (edges[1:] + edges[:-1])

    def cell_thickness(self, axis: str) -> u.Quantity:
        """The size of every cell along *axis*."""
        return np.diff(self.edges(axis))

    def is_uniform(self, axis: str) -> bool:
        """Whether the cells along *axis* all have the same size."""
        try:
            require_uniform_grid(self.edges(axis), EDGES[axis])
        except ValueError:
            return False
        return True

    def spacing(self, axis: str) -> u.Quantity:
        """The one cell size along *axis*, which must be evenly spaced."""
        edges = self.edges(axis)
        return require_uniform_grid(edges, EDGES[axis]) * edges.unit

    def velocity(self, axis: str) -> u.Quantity:
        """The velocity component along *axis*."""
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
        The electron density, derived from the mass density where it was not given.

        Parameters
        ----------
        mass_per_electron_amu : float
            Mass of the plasma per free electron, in atomic mass units; see
            :func:`mass_per_electron`. Ignored when the atmosphere already
            has an electron density.
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
        The part of the box within the given ranges.

        Each range is ``(low, high)`` in the coordinates of the file. A cell
        is kept when any part of it lies inside the range, which is what
        NDCube's own cropping keeps on an even grid; a bound that falls on a
        cell boundary does not keep the cell beyond it.
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
            # A bound often lands on a cell boundary, as round numbers do on
            # a round grid. Whether that boundary cell is kept must not turn
            # on rounding, so a cell counts as inside only when more than a
            # millionth of it is.
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
        Every *factor*-th cell along each axis, each standing for the block it starts.

        The kept cell keeps its value and takes the boundaries of the block
        of *factor* cells it stands for, so the box keeps its extent and the
        column its depth.
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
        """The axes whose cells are not all the same size."""
        return [axis for axis in AXES if not self.is_uniform(axis)]

    def to_ndcube(self, quantity: u.Quantity) -> NDCube:
        """
        *quantity*, a cube of this atmosphere's shape, with its coordinates as a WCS.

        The WCS is linear. An axis whose cells are not all the same size gets
        the mean size and is named in the cube's ``meta["nonuniform_axes"]``,
        so that nothing downstream mistakes its coordinates for exact. Only
        the line of sight may be such an axis: it is integrated out with the
        true size of every cell, and its coordinates are never used.
        """
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
        """A few lines saying what the atmosphere holds."""
        present = [name for name in CUBES if getattr(self, name) is not None]
        return _describe({axis: self.edges(axis) for axis in AXES}, present,
                         self.time, self.source)


def _check_axis(axis: str) -> str:
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}.")
    return axis


def _describe(edges: Dict[str, u.Quantity], cubes: Sequence[str],
              time: Optional[u.Quantity], source: str) -> str:
    """The description shared by an Atmosphere and a file on disk."""
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
    """*value* as the mass per free electron in atomic mass units, which must be finite and positive."""
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
def write_atmosphere(atmosphere: Atmosphere, path: str | Path,
                     compression: Optional[str] = None) -> Path:
    """
    Write *atmosphere* as an ECLIPSE atmosphere file.

    Parameters
    ----------
    atmosphere : Atmosphere
    path : str or Path
        The file to write; an existing file is replaced.
    compression : str, optional
        An h5py compression filter such as ``"gzip"`` for the cubes. None
        writes them uncompressed, which reads fastest.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
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
                    velocities: Optional[Sequence[str]] = None) -> Atmosphere:
    """
    Read an ECLIPSE atmosphere file.

    Parameters
    ----------
    path : str or Path
    velocities : sequence of str, optional
        Which velocity components to read, e.g. ``("z",)`` for a view along
        z. None reads every component the file has. A component asked for
        that the file lacks raises.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        _check_format(f, path)
        fields = {EDGES[axis]: _read_dataset(f, EDGES[axis]) for axis in AXES}
        fields["temperature"] = _read_dataset(f, "temperature")
        for name in ("mass_density", "electron_density", "time"):
            if name in f:
                fields[name] = _read_dataset(f, name)
        wanted = AXES if velocities is None else tuple(velocities)
        for axis in wanted:
            name = f"velocity_{_check_axis(axis)}"
            if name in f:
                fields[name] = _read_dataset(f, name)
            elif velocities is not None:
                raise ValueError(f"{path} has no {name}, which a line of sight "
                                 f"along {axis} needs.")
        source = f.attrs.get("source", "")
        if isinstance(source, bytes):
            source = source.decode()
    return Atmosphere(source=str(source), **fields)


def describe_atmosphere_file(path: str | Path) -> str:
    """
    What an atmosphere file holds, without loading any of its cubes.

    Reads the attributes, the edges, the time and the cubes' names and
    shapes, so it costs nothing on a file of many gigabytes.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        _check_format(f, path)
        edges = {axis: _read_dataset(f, EDGES[axis]) for axis in AXES}
        shape = tuple(edges[axis].size - 1 for axis in ("z", "y", "x"))
        cubes = [name for name in CUBES if name in f]
        for name in cubes:
            if f[name].shape != shape:
                raise ValueError(f"{name} in {path} has shape {f[name].shape} but "
                                 f"the edges bound (nz, ny, nx) = {shape} cells.")
        if "temperature" not in cubes:
            raise ValueError(f"{path} has no 'temperature' dataset.")
        time = _read_dataset(f, "time") if "time" in f else None
        source = f.attrs.get("source", "")
        if isinstance(source, bytes):
            source = source.decode()
    return _describe(edges, cubes, time, str(source))


def _check_format(f: h5py.File, path: Path) -> None:
    name = f.attrs.get("format")
    if isinstance(name, bytes):
        name = name.decode()
    if name != FORMAT_NAME:
        raise ValueError(
            f"{path} is not an ECLIPSE atmosphere file: its 'format' "
            f"attribute is {name!r}, not {FORMAT_NAME!r}. See the "
            f"euvst_response.atmosphere documentation for the layout.")
    version = f.attrs.get("version")
    if version is None:
        raise ValueError(f"{path} has no 'version' attribute; an ECLIPSE "
                         f"atmosphere file has version {FORMAT_VERSION}.")
    try:
        matches = np.ndim(version) == 0 and float(version) == FORMAT_VERSION
    except (TypeError, ValueError):
        matches = False
    if not matches:
        if isinstance(version, bytes):
            version = version.decode()
        raise ValueError(f"{path} is atmosphere format version {version}; "
                         f"this ECLIPSE reads version {FORMAT_VERSION}.")


def _read_dataset(f: h5py.File, name: str) -> u.Quantity:
    if name not in f:
        raise ValueError(f"{f.filename} has no '{name}' dataset.")
    dataset = f[name]
    unit = dataset.attrs.get("unit")
    if unit is None:
        raise ValueError(f"'{name}' in {f.filename} has no 'unit' attribute. "
                         f"Every dataset needs one, for instance "
                         f"'{UNITS[name]}'.")
    if isinstance(unit, bytes):
        unit = unit.decode()
    return u.Quantity(dataset[()], u.Unit(unit))


# ----------------------------------------------------------------------
# From mass density to electron density
# ----------------------------------------------------------------------
def mass_per_electron_from_abundances(abundances: Dict[str, float]) -> float:
    """
    Mass per free electron of a fully ionised plasma, in atomic mass units.

    With every element fully ionised, a plasma holding ``A_X`` atoms of
    element X per hydrogen atom has mass ``sum(A_X m_X)`` and ``sum(A_X Z_X)``
    electrons per hydrogen atom, so the ratio is the mass per electron.

    Parameters
    ----------
    abundances : dict
        Number of atoms of each element per hydrogen atom, keyed by symbol,
        with hydrogen itself included.
    """
    if "H" not in abundances:
        raise ValueError("The abundances must include hydrogen ('H').")
    mass = 0.0
    electrons = 0.0
    for symbol, per_hydrogen in abundances.items():
        atom = element(symbol)
        mass += per_hydrogen * atom.atomic_weight
        electrons += per_hydrogen * atom.atomic_number
    return mass / electrons


@lru_cache(maxsize=None)
def mass_per_electron(abundance: str, hdf5_dbase_root: Optional[str] = None) -> float:
    """
    Mass per free electron for a CHIANTI abundance set, in atomic mass units.

    This is what turns a simulation's mass density into the electron density
    the contribution functions need, when the simulation gives no electron
    density of its own. It assumes every element is fully ionised, which
    holds where the EUV lines ECLIPSE synthesises form. Cooler cells get an
    electron density higher than they have, but they emit none of those
    lines. It is about 1.16 for coronal abundances, against 1.29 for a
    neutral gas, so an atmosphere read with the neutral value has about 20
    per cent too little emission measure.

    Parameters
    ----------
    abundance : str
        The CHIANTI abundance set, as given to ``--abundance``.
    hdf5_dbase_root : str, optional
        The fiasco database to read it from; None uses fiasco's default.
    """
    import fiasco
    from fiasco.util.exceptions import MissingDatasetException

    ion_kwargs = {} if hdf5_dbase_root is None else {"hdf5_dbase_root": hdf5_dbase_root}
    ions = fiasco.list_ions(hdf5_dbase_root)
    abundances = {}
    for symbol in fiasco.list_elements(hdf5_dbase_root):
        # The abundance is a property of the element, read through any of
        # its ions; the table just has to be asked through one that exists.
        ion_name = next(name for name in ions if name.startswith(f"{symbol} "))
        ion = fiasco.Ion(ion_name, _ABUNDANCE_PROBE_TEMPERATURE,
                         abundance=abundance, **ion_kwargs)
        try:
            abundances[symbol] = float(ion.abundance)
        except MissingDatasetException:
            # The set does not list this element, so it holds none of it.
            continue
    if not abundances:
        raise ValueError(f"No element has an abundance in the set "
                         f"{abundance!r}; is the name right?")
    return mass_per_electron_from_abundances(abundances)
