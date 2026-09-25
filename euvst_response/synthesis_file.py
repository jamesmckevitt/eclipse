"""
The synthesis file: the spectra the instrument simulation observes, whoever synthesised them.

ECLIPSE's own synthesis writes one, and so can any other code, optically
thin or thick: the spectral radiance leaving the Sun at each pixel of an
image, line by line, as HDF5. The instrument run reads the lines near the
one it measures and puts them on the detector's grid with
:func:`euvst_response.data_processing.rebin_spectra`.

File layout, version 1
----------------------
Root attributes:

- ``format``: ``"eclipse-synthesis"``
- ``version``: ``1``
- ``source``: free text naming the code and the model, optional

Datasets, each with a ``unit`` attribute astropy can parse, and groups:

- ``x_edges``, ``y_edges``: the pixel boundaries of the image, 1D,
  increasing and evenly spaced. Lengths on the Sun, or angles as seen from
  1 AU. x runs across the slit, the way a raster steps, and y along it.
- ``time``: a scalar, the time of the snapshot, optional. A time series of
  synthesis files needs it to place each file.
- ``lines/<name>``: one group per line, named as ``reference_line`` names it
  in the instrument configuration, holding

  - ``intensity``: ``(ny, nx, n_wavelength)``, the spectral radiance. Any
    unit of spectral radiance does, per wavelength or per frequency, in
    energy or in photons, such as ``erg / (s cm2 sr Angstrom)`` or
    ``W / (m2 sr Hz)``.
  - ``wavelength``: ``(n_wavelength,)``, increasing; they need not be
    evenly spaced.
  - ``rest_wavelength``: a scalar, the wavelength the line's Doppler shifts
    are measured from.

  A group can hold a whole spectral window, blends and all, as another code
  often gives it; the instrument run adds up whatever lines reach the
  window of the one it measures.
- ``synthesis``: optional, what ECLIPSE's own synthesis worked out on the
  way, which the instrument run does not read; see
  :func:`read_synthesis_products`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import astropy.units as u
import h5py
import numpy as np

from .atmosphere import _check_format, _read_dataset, _write_dataset
from .utils import angle_to_distance, require_uniform_grid

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "RADIANCE_UNIT",
    "SpectralLine",
    "Synthesis",
    "read_synthesis",
    "read_synthesis_layout",
    "write_synthesis",
    "read_synthesis_products",
    "load_synthesis",
    "write_line_cubes",
    "convert_synthesis_pickle",
    "is_synthesis_file",
    "synthesis_line_names",
]

FORMAT_NAME = "eclipse-synthesis"
FORMAT_VERSION = 1

# The spectral radiance the instrument simulation works in, as ECLIPSE's own
# synthesis writes it.
RADIANCE_UNIT = u.erg / (u.s * u.cm**2 * u.sr * u.cm)

# What a spectral radiance is once the steradian is taken off: a flux density
# per wavelength or per frequency, in energy or in photons.
_SPECTRAL_FLUX_DENSITIES = (u.erg / (u.s * u.cm**2 * u.cm), u.erg / (u.s * u.cm**2 * u.Hz),
                            u.ph / (u.s * u.cm**2 * u.cm), u.ph / (u.s * u.cm**2 * u.Hz))

AXES = ("x", "y")
EDGES = {axis: f"{axis}_edges" for axis in AXES}
# A unit of the right kind for each dataset, for the messages that ask for one.
UNITS = {"intensity": u.erg / (u.s * u.cm**2 * u.sr * u.AA), "wavelength": u.AA,
         "rest_wavelength": u.AA, "x_edges": u.Mm, "y_edges": u.Mm, "time": u.s}
# The image axes of a view along each axis of a simulation, as ECLIPSE's line
# cubes name them.
_VIEW_CTYPES = {"z": ("SOLX", "SOLY"), "x": ("SOLY", "SOLZ"), "y": ("SOLX", "SOLZ")}


def to_radiance(intensity: u.Quantity, wavelength: u.Quantity) -> u.Quantity:
    """
    *intensity* in erg / (s cm2 sr cm), from any unit of spectral radiance.

    A radiance per frequency or in photons is converted at each of the
    *wavelength*, which runs along the last axis of *intensity*.
    """
    # spectral_density converts flux densities between per wavelength and
    # per frequency, in energy or in photons. It knows nothing of solid
    # angle, so the steradian is taken off for it and put back. It also
    # converts lambda F_lambda, whose unit is that of an intensity summed
    # over a line, so the flux density is checked to be one of the four
    # kinds first rather than left to it.
    flux = intensity * u.sr
    if not any(flux.unit.is_equivalent(kind) for kind in _SPECTRAL_FLUX_DENSITIES):
        raise u.UnitConversionError(
            f"The intensity must be a spectral radiance, per wavelength or per "
            f"frequency, such as erg / (s cm2 sr Angstrom) or W / (m2 sr Hz); "
            f"got {intensity.unit}.")
    return flux.to(RADIANCE_UNIT * u.sr, equivalencies=u.spectral_density(wavelength)) / u.sr


def _checked(value, name: str, ndim: int, kinds: Optional[tuple] = None) -> u.Quantity:
    """*value*, which must be a finite Quantity of *ndim* axes, in a unit of one of the physical *kinds* if given."""
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an astropy Quantity, got "
                        f"{type(value).__name__}.")
    if kinds is not None and value.unit.physical_type not in kinds:
        raise u.UnitConversionError(
            f"{name} must be in a unit of {' or '.join(kinds)}, got {value.unit}.")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {value.ndim}.")
    if not np.all(np.isfinite(value.value)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return value


def _to_length(value: u.Quantity) -> u.Quantity:
    """A position or size on the Sun, given as a length or as an angle seen from 1 AU, in Mm."""
    if value.unit.physical_type == "angle":
        return angle_to_distance(value).to(u.Mm)
    return value.to(u.Mm)


@dataclass(frozen=True)
class SpectralLine:
    """
    The spectra of one line over the image, or of one spectral window.

    Parameters
    ----------
    intensity : u.Quantity
        The spectral radiance at each pixel and wavelength,
        ``(ny, nx, n_wavelength)``, in any unit of spectral radiance.
    wavelength : u.Quantity
        The wavelength of each plane of *intensity*, increasing. They need
        not be evenly spaced.
    rest_wavelength : u.Quantity
        The wavelength the line's Doppler shifts are measured from, within
        the range of *wavelength*.
    """

    intensity: u.Quantity
    wavelength: u.Quantity
    rest_wavelength: u.Quantity

    def __post_init__(self):
        wavelength = _checked(self.wavelength, "wavelength", ndim=1, kinds=("length",))
        if wavelength.size < 2:
            raise ValueError(f"wavelength must have at least 2 values, got "
                             f"{wavelength.size}.")
        if np.any(np.diff(wavelength.value) <= 0):
            raise ValueError("wavelength must increase; a grid that decreases, as "
                             "one converted from increasing frequencies does, "
                             "needs it and the intensity reversed along it.")
        rest = _checked(self.rest_wavelength, "rest_wavelength", ndim=0, kinds=("length",))
        if not wavelength[0] <= rest <= wavelength[-1]:
            raise ValueError(f"rest_wavelength {rest} is outside the wavelengths, "
                             f"which run from {wavelength[0]} to {wavelength[-1]}.")
        intensity = _checked(self.intensity, "intensity", ndim=3)
        # Any spectral radiance will do, which one conversion tells apart
        # from everything else.
        to_radiance(1.0 * intensity.unit, wavelength[0])
        if intensity.shape[-1] != wavelength.size:
            raise ValueError(
                f"intensity has {intensity.shape[-1]} wavelengths along its last "
                f"axis but there are {wavelength.size}. The intensity is "
                f"(y, x, wavelength): y along the slit first.")
        if np.any(intensity.value < 0):
            raise ValueError("intensity has negative values, which no spectral "
                             "radiance can have.")

    def radiance(self) -> u.Quantity:
        """The intensity in erg / (s cm2 sr cm), as ECLIPSE's own synthesis gives it."""
        return to_radiance(self.intensity, self.wavelength)


@dataclass(frozen=True)
class Synthesis:
    """
    Spectra over an image, line by line: what the instrument simulation observes.

    Parameters
    ----------
    lines : mapping of str to SpectralLine
        The lines, by the names the instrument configuration's
        ``reference_line`` uses. All share the image.
    x_edges, y_edges : u.Quantity
        The pixel boundaries, evenly spaced, as lengths on the Sun or angles
        seen from 1 AU. x runs across the slit and y along it.
    source : str, optional
        Free text naming the code and the model, kept in the file.
    time : u.Quantity, optional
        The time of the snapshot, which a time series of synthesis files
        needs to place it.
    """

    lines: Mapping[str, SpectralLine]
    x_edges: u.Quantity
    y_edges: u.Quantity
    source: str = ""
    time: Optional[u.Quantity] = None

    def __post_init__(self):
        if self.time is not None:
            _checked(self.time, "time", ndim=0, kinds=("time",))
        for axis in AXES:
            name = EDGES[axis]
            edges = _checked(getattr(self, name), name, ndim=1, kinds=("length", "angle"))
            # The instrument grid is laid over the pixels with one pixel size
            # per axis, so they have to be evenly spaced.
            require_uniform_grid(edges.value, name)
        if not isinstance(self.lines, Mapping) or not self.lines:
            raise ValueError("A synthesis needs at least one line, as a mapping "
                             "from its name to a SpectralLine.")
        for name, line in self.lines.items():
            if not isinstance(name, str) or not name or "/" in name:
                raise ValueError(f"A line's name must be a string without '/', "
                                 f"got {name!r}.")
            if not isinstance(line, SpectralLine):
                raise TypeError(f"Line {name!r} must be a SpectralLine, got "
                                f"{type(line).__name__}.")
            if line.intensity.shape[:2] != self.shape:
                raise ValueError(
                    f"Line {name!r} has an image of {line.intensity.shape[:2]} pixels "
                    f"but the edges give (ny, nx) = {self.shape}. The intensity is "
                    f"(y, x, wavelength): y along the slit first.")
        if not isinstance(self.source, str):
            raise TypeError(f"source must be a string, got "
                            f"{type(self.source).__name__}.")

    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)``."""
        return (self.y_edges.size - 1, self.x_edges.size - 1)

    def pixel_size(self, axis: str) -> u.Quantity:
        """The size of every pixel along *axis* on the Sun, in Mm."""
        edges = getattr(self, EDGES[_check_axis(axis)])
        return _to_length(require_uniform_grid(edges.value, EDGES[axis]) * edges.unit)

    def centre(self, axis: str) -> u.Quantity:
        """The middle of the image along *axis* on the Sun, in Mm."""
        edges = getattr(self, EDGES[_check_axis(axis)])
        return _to_length(0.5 * (edges[0] + edges[-1]))

    def summed(self, reference: str) -> u.Quantity:
        """
        Every line's radiance on the wavelengths of *reference*, added up.

        This is what the instrument observes when it measures *reference*.
        Each line is interpolated onto the reference line's wavelengths and
        is zero beyond its own, so the lines that reach the reference line's
        window add to it, as blends do, and the others add nothing.
        """
        if reference not in self.lines:
            raise ValueError(f"No line is named {reference!r}; the lines are "
                             f"{list(self.lines)}.")
        wavelength = self.lines[reference].wavelength
        total = np.zeros(self.shape + (wavelength.size,))
        for line in self.lines.values():
            own = line.wavelength.to_value(wavelength.unit)
            spectra = line.radiance().to_value(RADIANCE_UNIT).reshape(-1, own.size)
            total += np.array([np.interp(wavelength.value, own, spectrum, left=0.0, right=0.0)
                               for spectrum in spectra]).reshape(total.shape)
        return total * RADIANCE_UNIT

    def evenly_spaced(self, name: str) -> bool:
        """Whether line *name* has evenly spaced wavelengths, which a WCS can describe."""
        try:
            require_uniform_grid(self.lines[name].wavelength.value, "wavelength")
        except ValueError:
            return False
        return True

    def line_cube(self, name: str, view: str = "z"):
        """
        Line *name* as an NDCube like those of ECLIPSE's synthesis, indexed ``[y, x, wavelength]``.

        The wavelengths have to be evenly spaced for a WCS to describe them.
        *view* is the simulation axis the image was seen along, which names
        the image axes.
        """
        line = self.lines[name]
        return self._cube(line.intensity, name, view)

    def summed_cube(self, reference: str, summed: Optional[u.Quantity] = None, view: str = "z"):
        """
        :meth:`summed` as an NDCube on the wavelengths of *reference*, which have to be evenly spaced.

        *summed*, if given, is :meth:`summed` already worked out.
        """
        return self._cube(self.summed(reference) if summed is None else summed, reference, view)

    def _cube(self, data: u.Quantity, name: str, view: str):
        from astropy.wcs import WCS
        from ndcube import NDCube

        from .utils import VELOCITY_CONVENTION

        line = self.lines[name]
        step = require_uniform_grid(line.wavelength.to_value(u.cm), f"the wavelengths of {name}")
        ny, nx = self.shape
        n_wavelength = line.wavelength.size
        wcs = WCS(naxis=3)
        wcs.wcs.ctype = ["WAVE", *_VIEW_CTYPES[view]]
        wcs.wcs.cunit = ["cm", "Mm", "Mm"]
        wcs.wcs.crpix = [(n_wavelength + 1) / 2, (nx + 1) / 2, (ny + 1) / 2]
        middle = 0.5 * (line.wavelength[0] + line.wavelength[-1])
        wcs.wcs.crval = [middle.to_value(u.cm), self.centre("x").to_value(u.Mm),
                         self.centre("y").to_value(u.Mm)]
        wcs.wcs.cdelt = [step, self.pixel_size("x").to_value(u.Mm),
                         self.pixel_size("y").to_value(u.Mm)]
        return NDCube(data.value, wcs=wcs, unit=data.unit,
                      meta={"line_name": name, "rest_wav": line.rest_wavelength,
                            "integration_axis": view, "source": self.source,
                            "velocity_convention": VELOCITY_CONVENTION})


def _check_axis(axis: str) -> str:
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}.")
    return axis


# ----------------------------------------------------------------------
# HDF5
# ----------------------------------------------------------------------
def is_synthesis_file(path: str | Path) -> bool:
    """Whether *path* is an HDF5 file, as a synthesis file is, rather than an older pickle."""
    return h5py.is_hdf5(str(path))


def synthesis_line_names(path: str | Path) -> list:
    """The names of the lines in a synthesis file, in its order, without reading their spectra."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        _check_format(f, path, kind="synthesis", format_name=FORMAT_NAME,
                      format_version=FORMAT_VERSION)
        return list(f["lines"]) if "lines" in f else []


def write_synthesis(synthesis: Synthesis, path: str | Path,
                    products: Optional[Mapping] = None,
                    compression: Optional[str] = None) -> Path:
    """
    Write *synthesis* as a synthesis file.

    Parameters
    ----------
    synthesis : Synthesis
    path : str or Path
        The file to write; an existing file is replaced.
    products : mapping, optional
        What the synthesis worked out on the way, kept in the file's
        ``synthesis`` group: arrays and quantities with dimensions as
        datasets, compressed, and everything else as attributes. See
        :func:`read_synthesis_products`.
    compression : str, optional
        An h5py compression filter such as ``"gzip"`` for the line spectra.
        None writes them uncompressed, which reads fastest.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = FORMAT_NAME
        f.attrs["version"] = FORMAT_VERSION
        f.attrs["source"] = synthesis.source
        for axis in AXES:
            _write_dataset(f, EDGES[axis], getattr(synthesis, EDGES[axis]))
        if synthesis.time is not None:
            _write_dataset(f, "time", synthesis.time)
        # The lines keep the order they were given in, which HDF5 would
        # otherwise sort by name.
        group = f.create_group("lines", track_order=True)
        for name, line in synthesis.lines.items():
            entry = group.create_group(name)
            _write_dataset(entry, "intensity", line.intensity, compression=compression)
            _write_dataset(entry, "wavelength", line.wavelength)
            _write_dataset(entry, "rest_wavelength", line.rest_wavelength)
        if products:
            _write_tree(f.create_group("synthesis"), products)
    return path


def read_synthesis(path: str | Path, reference_line: Optional[str] = None,
                   columns: Optional[slice] = None) -> Synthesis:
    """
    Read a synthesis file's spectra.

    Parameters
    ----------
    path : str or Path
    reference_line : str, optional
        The line the instrument will measure. Only the lines whose
        wavelengths reach its window are read, since the others add nothing
        to it, which keeps a synthesis of many lines cheap to observe one at
        a time. None reads them all.
    columns : slice, optional
        Which pixels along x to read, as a slice of neighbouring x indices,
        as a time series reads the strip under the slit. None reads the
        whole image.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        _check_format(f, path, kind="synthesis", format_name=FORMAT_NAME,
                      format_version=FORMAT_VERSION)
        edges = {EDGES[axis]: _read_dataset(f, EDGES[axis], units=UNITS) for axis in AXES}
        if columns is not None:
            first, last, step = columns.indices(edges["x_edges"].size - 1)
            if step != 1 or last <= first:
                raise ValueError(f"columns must be a slice of neighbouring pixels, got {columns}.")
            edges["x_edges"] = edges["x_edges"][first:last + 1]
            columns = slice(first, last)
        group, names = _lines_reaching(f, path, reference_line)
        lines = {}
        for name in names:
            entry = group[name]
            fields = {"intensity": _read_dataset(entry, "intensity", columns, units=UNITS, axis=1),
                      "wavelength": _read_dataset(entry, "wavelength", units=UNITS),
                      "rest_wavelength": _read_dataset(entry, "rest_wavelength", units=UNITS)}
            try:
                lines[name] = SpectralLine(**fields)
            except (TypeError, ValueError) as error:
                raise type(error)(f"{path}, line {name!r}: {error}") from None
        source = _source(f)
        time = _read_dataset(f, "time", units=UNITS) if "time" in f else None
    try:
        return Synthesis(lines=lines, source=source, time=time, **edges)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{path}: {error}") from None


def read_synthesis_layout(path: str | Path, reference_line: Optional[str] = None) -> dict:
    """
    What a synthesis file holds, without reading its spectra.

    Gives ``x_edges``, ``y_edges``, ``time`` (None if the file has none),
    ``source`` and ``lines``, which maps each line, or each one reaching the
    window of *reference_line* if given, to its ``wavelength``, its
    ``rest_wavelength`` and the ``shape`` of its intensity. A time series
    checks its files with it before reading any spectra.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        _check_format(f, path, kind="synthesis", format_name=FORMAT_NAME,
                      format_version=FORMAT_VERSION)
        layout = {EDGES[axis]: _read_dataset(f, EDGES[axis], units=UNITS) for axis in AXES}
        layout["time"] = None
        if "time" in f:
            try:
                layout["time"] = _checked(_read_dataset(f, "time", units=UNITS), "time",
                                          ndim=0, kinds=("time",))
            except (TypeError, ValueError) as error:
                raise type(error)(f"{path}: {error}") from None
        layout["source"] = _source(f)
        group, names = _lines_reaching(f, path, reference_line)
        layout["lines"] = {}
        for name in names:
            entry = group[name]
            if "intensity" not in entry:
                raise ValueError(f"{path}, line {name!r} has no 'intensity' dataset.")
            layout["lines"][name] = {
                "wavelength": _read_dataset(entry, "wavelength", units=UNITS),
                "rest_wavelength": _read_dataset(entry, "rest_wavelength", units=UNITS),
                "shape": entry["intensity"].shape}
    return layout


def _lines_reaching(f: h5py.File, path: Path, reference_line: Optional[str]):
    """The file's ``lines`` group, and the names of its lines that reach the window of *reference_line*, or all of them."""
    group = f.get("lines")
    if group is None or len(group) == 0:
        raise ValueError(f"{path} holds no lines.")
    names = list(group)
    if reference_line is not None:
        if reference_line not in group:
            raise ValueError(f"'reference_line' {reference_line!r} is not in {path}; "
                             f"its lines are {names}.")
        window = _read_dataset(group[reference_line], "wavelength", units=UNITS)
        names = [name for name in names
                 if _reaches(_read_dataset(group[name], "wavelength", units=UNITS), window)]
    return group, names


def _source(f: h5py.File) -> str:
    source = f.attrs.get("source", "")
    return str(source.decode() if isinstance(source, bytes) else source)


def _reaches(wavelength: u.Quantity, window: u.Quantity) -> bool:
    """Whether a line on *wavelength* has any part inside *window*."""
    own = wavelength.to_value(window.unit)
    return own.max() >= window.value.min() and own.min() <= window.value.max()


def read_synthesis_products(path: str | Path, keys: Optional[Iterable[str]] = None) -> dict:
    """
    What ECLIPSE's synthesis worked out on the way to the spectra, as it saved it.

    For a file from ``synthesise-spectra`` these are ``dem_map``, ``em_tv``,
    ``logT_grid``, ``vel_grid``, ``logN_grid``, ``goft`` (each line's
    contribution functions), ``voxel_sizes``, ``atmosphere``,
    ``dynamic_mode`` and ``config``, as the pickles of older versions held
    them. A file from another code has none, and gives an empty dict.

    Parameters
    ----------
    path : str or Path
    keys : iterable of str, optional
        Read only these, which spares reading the emission measure cube,
        often the largest part of the file. None reads everything.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        _check_format(f, path, kind="synthesis", format_name=FORMAT_NAME,
                      format_version=FORMAT_VERSION)
        if "synthesis" not in f:
            return {}
        return _read_tree(f["synthesis"], None if keys is None else set(keys))


def load_synthesis(path: str | Path) -> dict:
    """
    Everything in a synthesis file, as the pickles of older versions held it.

    ``line_cubes`` maps each line to an NDCube indexed ``[y, x, wavelength]``,
    and the rest is :func:`read_synthesis_products`. Handy for looking at a
    synthesis; the instrument run reads the file itself.
    """
    synthesis = read_synthesis(path)
    products = read_synthesis_products(path)
    view = (products.get("config") or {}).get("integration_axis") or "z"
    line_cubes = {}
    for name in synthesis.lines:
        cube = synthesis.line_cube(name, view)
        line_info = (products.get("goft") or {}).get(name, {})
        cube.meta.update({key: line_info[key] for key in ("atom", "ion") if key in line_info})
        line_cubes[name] = cube
    return {"line_cubes": line_cubes, **products}


# The products are nested dicts of arrays and plain values. Arrays and
# quantities with dimensions become datasets; everything else becomes a
# JSON attribute, with quantities written as their value and unit so that
# they come back as quantities.
def _write_tree(group: h5py.Group, tree: Mapping) -> None:
    for key, value in tree.items():
        key = str(key)
        if _is_array(value):
            data = value.value if isinstance(value, u.Quantity) else np.asarray(value)
            dataset = group.create_dataset(key, data=data, compression="gzip",
                                           compression_opts=1)
            if isinstance(value, u.Quantity):
                dataset.attrs["unit"] = value.unit.to_string()
        elif isinstance(value, Mapping) and any(_holds_array(v) for v in value.values()):
            _write_tree(group.create_group(key), value)
        else:
            group.attrs[key] = json.dumps(_jsonable(value))


def _read_tree(group: h5py.Group, keys: Optional[set]) -> dict:
    tree = {}
    for key, value in group.attrs.items():
        if keys is None or key in keys:
            tree[key] = json.loads(value, object_hook=_unjson)
    for key, item in group.items():
        if keys is not None and key not in keys:
            continue
        if isinstance(item, h5py.Group):
            tree[key] = _read_tree(item, None)
        else:
            data = item[()]
            unit = item.attrs.get("unit")
            tree[key] = data if unit is None else u.Quantity(data, u.Unit(
                unit.decode() if isinstance(unit, bytes) else unit))
    return tree


def _is_array(value) -> bool:
    return (isinstance(value, np.ndarray) and value.ndim > 0) or (
        isinstance(value, u.Quantity) and value.ndim > 0)


def _holds_array(value) -> bool:
    if _is_array(value):
        return True
    return isinstance(value, Mapping) and any(_holds_array(v) for v in value.values())


def _jsonable(value):
    if isinstance(value, u.Quantity):
        return {"__quantity__": _jsonable(value.value), "unit": value.unit.to_string()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _unjson(value: dict):
    if "__quantity__" in value:
        return u.Quantity(value["__quantity__"], u.Unit(value["unit"]))
    return value


# ----------------------------------------------------------------------
# From ECLIPSE's line cubes
# ----------------------------------------------------------------------
def write_line_cubes(line_cubes: Mapping, path: str | Path, source: str = "",
                     products: Optional[Mapping] = None,
                     time: Optional[u.Quantity] = None) -> Path:
    """
    Write line cubes, as ECLIPSE's synthesis builds them, as a synthesis file.

    Each cube keeps its values, its wavelengths and its rest wavelength; the
    image grid is read off the first cube's WCS, which every cube shares.

    Parameters
    ----------
    line_cubes : mapping of str to NDCube
        Cubes indexed ``[y, x, wavelength]`` with ``rest_wav`` in their
        metadata, as :func:`euvst_response.synthesis.create_line_cube` makes.
    path : str or Path
    source : str, optional
        Free text naming the simulation.
    products : mapping, optional
        What the synthesis worked out on the way; see :func:`write_synthesis`.
    time : u.Quantity, optional
        The time of the snapshot.
    """
    cubes = dict(line_cubes)
    if not cubes:
        raise ValueError("There are no line cubes to write.")
    first = next(iter(cubes.values()))
    edges = {"x_edges": _pixel_edges(first, 1), "y_edges": _pixel_edges(first, 2)}
    lines = {}
    for name, cube in cubes.items():
        if cube.data.shape[:2] != first.data.shape[:2]:
            raise ValueError(f"Line {name!r} has an image of {cube.data.shape[:2]} "
                             f"pixels, not {first.data.shape[:2]} as the others.")
        lines[name] = SpectralLine(intensity=np.asarray(cube.data) * cube.unit,
                                   wavelength=cube.axis_world_coords(-1)[0],
                                   rest_wavelength=cube.meta["rest_wav"])
    return write_synthesis(Synthesis(lines=lines, source=source, time=time, **edges), path,
                           products=products)


def _pixel_edges(cube, wcs_axis: int) -> u.Quantity:
    """The pixel boundaries along one image axis of *cube*, from its WCS, in Mm."""
    wcs = getattr(cube.wcs, "low_level_wcs", cube.wcs)
    n = cube.data.shape[::-1][wcs_axis]
    points = np.zeros((n + 1, wcs.pixel_n_dim))
    points[:, wcs_axis] = np.arange(n + 1) - 0.5
    world = wcs.pixel_to_world_values(*points.T)
    return (np.asarray(world[wcs_axis]) * u.Unit(wcs.world_axis_units[wcs_axis])).to(u.Mm)


def convert_synthesis_pickle(pickle_path: str | Path, path: str | Path) -> Path:
    """
    Rewrite a synthesis pickle, as older versions wrote them, as a synthesis file.

    Everything the pickle holds is kept: its line cubes as the spectra and
    the rest as the synthesis products. A pickle whose line cubes have the
    old axis order or the old Doppler sign is refused, as the instrument
    run refuses it.
    """
    import dill

    from .data_processing import check_old_line_cubes

    with open(pickle_path, "rb") as f:
        saved = dill.load(f)
    if "line_cubes" not in saved or not saved["line_cubes"]:
        raise ValueError(f"{pickle_path} holds no line cubes.")
    check_old_line_cubes(saved["line_cubes"], pickle_path)
    products = {key: value for key, value in saved.items() if key != "line_cubes"}
    goft = products.get("goft")
    if isinstance(goft, Mapping):
        # The spectra and wavelengths are the lines themselves.
        products["goft"] = {name: {key: value for key, value in info.items()
                                   if key not in ("si", "wl_grid")}
                            for name, info in goft.items()}
    atmosphere = products.get("atmosphere") or {}
    time = atmosphere.get("time")
    return write_line_cubes(saved["line_cubes"], path, source=str(atmosphere.get("source") or ""),
                            products=products,
                            time=time if isinstance(time, u.Quantity) else None)
