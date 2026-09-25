"""
Spectra that another code has already synthesised, and the HDF5 file ECLIPSE reads them from.

ECLIPSE's own synthesis is optically thin. A code that does the radiative
transfer itself, optically thick or thin, can hand its spectra straight to
the instrument simulation instead: the spectral radiance leaving the Sun at
each point of an image, at each wavelength. This module holds those spectra
and reads and writes them as HDF5; the instrument run puts them on the
detector's grid with :func:`euvst_response.data_processing.rebin_spectra`.

File layout, version 1
----------------------
Root attributes:

- ``format``: ``"eclipse-spectra"``
- ``version``: ``1``
- ``source``: free text naming the code and the model, optional

Datasets, each with a ``unit`` attribute astropy can parse:

- ``intensity``: ``(ny, nx, n_wavelength)`` in C order, the spectral
  radiance at each pixel and wavelength. Any unit of spectral radiance
  does, per wavelength or per frequency, in energy or in photons, such as
  ``erg / (s cm2 sr Angstrom)`` or ``W / (m2 sr Hz)``.
- ``wavelength``: ``(n_wavelength,)``, increasing. The wavelengths need not
  be evenly spaced.
- ``x_edges``, ``y_edges``: the pixel boundaries, 1D, increasing and evenly
  spaced, one longer than the intensity along that axis. Lengths on the
  Sun, or angles as seen from 1 AU. x runs across the slit, the way a
  raster steps, and y along it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import astropy.units as u
import h5py
import numpy as np

from .atmosphere import _check_format, _read_dataset, _write_dataset
from .utils import angle_to_distance, require_uniform_grid

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "RADIANCE_UNIT",
    "Spectra",
    "read_spectra",
    "write_spectra",
]

FORMAT_NAME = "eclipse-spectra"
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
DATASETS = ("intensity", "wavelength", "x_edges", "y_edges")
# A unit of the right kind for each dataset, for the messages that ask for one.
UNITS = {"intensity": u.erg / (u.s * u.cm**2 * u.sr * u.AA), "wavelength": u.AA,
         "x_edges": u.Mm, "y_edges": u.Mm}


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
class Spectra:
    """
    Spectra synthesised by another code, in the form the instrument simulation needs.

    Parameters
    ----------
    intensity : u.Quantity
        The spectral radiance at each pixel and wavelength,
        ``(ny, nx, n_wavelength)``, in any unit of spectral radiance.
    wavelength : u.Quantity
        The wavelength of each plane of *intensity*, increasing. They need
        not be evenly spaced.
    x_edges, y_edges : u.Quantity
        The pixel boundaries, evenly spaced, as lengths on the Sun or angles
        seen from 1 AU. x runs across the slit and y along it.
    source : str, optional
        Free text naming the code and the model, kept in the file.
    """

    intensity: u.Quantity
    wavelength: u.Quantity
    x_edges: u.Quantity
    y_edges: u.Quantity
    source: str = ""

    def __post_init__(self):
        wavelength = _checked(self.wavelength, "wavelength", ndim=1, kinds=("length",))
        if wavelength.size < 2:
            raise ValueError(f"wavelength must have at least 2 values, got "
                             f"{wavelength.size}.")
        if np.any(np.diff(wavelength.value) <= 0):
            raise ValueError("wavelength must increase; a grid that decreases, as "
                             "one converted from increasing frequencies does, "
                             "needs it and the intensity reversed along it.")
        for axis in AXES:
            name = EDGES[axis]
            edges = _checked(getattr(self, name), name, ndim=1, kinds=("length", "angle"))
            # The instrument grid is laid over the pixels with one pixel size
            # per axis, so they have to be evenly spaced.
            require_uniform_grid(edges.value, name)
        intensity = _checked(self.intensity, "intensity", ndim=3)
        # Any spectral radiance will do, which one conversion tells apart
        # from everything else.
        to_radiance(1.0 * intensity.unit, wavelength[0])
        shape = (self.y_edges.size - 1, self.x_edges.size - 1, wavelength.size)
        if intensity.shape != shape:
            raise ValueError(
                f"intensity has shape {intensity.shape} but the edges and the "
                f"wavelengths give (ny, nx, n_wavelength) = {shape}. The "
                f"intensity is (y, x, wavelength): y along the slit first.")
        if np.any(intensity.value < 0):
            raise ValueError("intensity has negative values, which no spectral "
                             "radiance can have.")
        if not isinstance(self.source, str):
            raise TypeError(f"source must be a string, got "
                            f"{type(self.source).__name__}.")

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(ny, nx, n_wavelength)``."""
        return self.intensity.shape

    def radiance(self) -> u.Quantity:
        """The intensity in erg / (s cm2 sr cm), as ECLIPSE's own synthesis gives it."""
        return to_radiance(self.intensity, self.wavelength)

    def pixel_size(self, axis: str) -> u.Quantity:
        """The size of every pixel along *axis* on the Sun, in Mm."""
        edges = getattr(self, EDGES[_check_axis(axis)])
        return _to_length(require_uniform_grid(edges.value, EDGES[axis]) * edges.unit)

    def centre(self, axis: str) -> u.Quantity:
        """The middle of the image along *axis* on the Sun, in Mm."""
        edges = getattr(self, EDGES[_check_axis(axis)])
        return _to_length(0.5 * (edges[0] + edges[-1]))


def _check_axis(axis: str) -> str:
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES}, got {axis!r}.")
    return axis


def write_spectra(spectra: Spectra, path: str | Path,
                  compression: Optional[str] = None) -> Path:
    """
    Write *spectra* as an ECLIPSE spectra file.

    Parameters
    ----------
    spectra : Spectra
    path : str or Path
        The file to write; an existing file is replaced.
    compression : str, optional
        An h5py compression filter such as ``"gzip"`` for the intensity.
        None writes it uncompressed, which reads fastest.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = FORMAT_NAME
        f.attrs["version"] = FORMAT_VERSION
        f.attrs["source"] = spectra.source
        _write_dataset(f, "intensity", spectra.intensity, compression=compression)
        for name in ("wavelength", "x_edges", "y_edges"):
            _write_dataset(f, name, getattr(spectra, name))
    return path


def read_spectra(path: str | Path) -> Spectra:
    """Read an ECLIPSE spectra file."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        _check_format(f, path, kind="spectra", format_name=FORMAT_NAME,
                      format_version=FORMAT_VERSION)
        fields = {name: _read_dataset(f, name, units=UNITS) for name in DATASETS}
        source = f.attrs.get("source", "")
        if isinstance(source, bytes):
            source = source.decode()
    try:
        return Spectra(source=str(source), **fields)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{path}: {error}") from None
