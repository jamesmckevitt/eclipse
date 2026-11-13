"""
Emission measure files: a DEM or VDEM over an image, as HDF5.

The layout is described on the Files page of the docs. The emission measure
is n_e^2 dh in each bin, in cm^-5, not per kelvin or per km/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import astropy.units as u
import h5py
import numpy as np

from .atmosphere import (_check_format, _open_to_read, _read_dataset, _replacing,
                         _text_attribute, _write_dataset)
from .utils import require_uniform_grid

__all__ = ["FORMAT_NAME", "FORMAT_VERSION", "EmissionMeasure", "read_emission_measure",
           "read_emission_measure_layout", "write_emission_measure",
           "is_emission_measure_file"]

FORMAT_NAME = "eclipse-emission-measure"
FORMAT_VERSION = 1

# Example unit for each dataset, for error messages that ask for one.
UNITS = {"emission_measure": u.cm**-5, "temperature": u.K, "velocity": u.km / u.s,
         "electron_density": u.cm**-3, "x_edges": u.Mm, "y_edges": u.Mm, "time": u.s}
EDGES = ("x_edges", "y_edges")


def _checked(value, name: str, kinds: tuple, ndim: Optional[tuple] = None) -> u.Quantity:
    """`value`, checked to be a finite Quantity of one of the physical types `kinds`."""
    if not isinstance(value, u.Quantity):
        raise TypeError(f"{name} must be an astropy Quantity, got {type(value).__name__}.")
    if not any(value.unit.is_equivalent(kind) for kind in kinds):
        raise u.UnitConversionError(
            f"{name} must be in a unit such as {UNITS[name]}, got {value.unit}.")
    if ndim is not None and value.ndim not in ndim:
        raise ValueError(f"{name} must have {' or '.join(str(n) for n in ndim)} dimensions, "
                         f"got {value.ndim}.")
    if not np.all(np.isfinite(value.value)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return value


@dataclass(frozen=True)
class EmissionMeasure:
    """
    Emission measure of every pixel by temperature: a DEM, or with velocities a VDEM.

    Parameters
    ----------
    emission_measure : u.Quantity
        Emission measure n_e^2 dh in each bin, such as in cm^-5, shape
        (ny, nx, n_temperatures) for a DEM or (ny, nx, n_temperatures,
        n_velocities) for a VDEM.
    temperature : u.Quantity
        Temperature bin centres, increasing and evenly spaced in log10 T.
    x_edges, y_edges : u.Quantity
        Pixel boundaries, evenly spaced, as lengths on the Sun or angles seen
        from 1 AU. x runs across the slit and y along it.
    velocity : u.Quantity, optional
        Velocity bin centres of a VDEM, increasing. A positive velocity is
        away from the observer, a redshift. It holds
        the bulk motion only: the thermal width is added in the synthesis.
    electron_density : u.Quantity, optional
        Electron density the contribution functions are taken at: one value,
        one for each pixel, (ny, nx), or one for each temperature of each
        pixel, (ny, nx, n_temperatures). Default None, for a density or a
        pressure given to the synthesis.
    source : str, optional
        Free text saying where the emission measure came from, kept in the file.
    time : u.Quantity, optional
        Time of the snapshot.
    """

    emission_measure: u.Quantity
    temperature: u.Quantity
    x_edges: u.Quantity
    y_edges: u.Quantity
    velocity: Optional[u.Quantity] = None
    electron_density: Optional[u.Quantity] = None
    source: str = ""
    time: Optional[u.Quantity] = None

    def __post_init__(self):
        em = _checked(self.emission_measure, "emission_measure", (u.cm**-5,), ndim=(3, 4))
        if np.any(em.value < 0):
            raise ValueError("emission_measure has negative values.")
        temperature = _checked(self.temperature, "temperature", (u.K,), ndim=(1,))
        if temperature.size < 2 or np.any(temperature.value <= 0):
            raise ValueError("temperature must have two or more values above zero.")
        require_uniform_grid(np.log10(temperature.to_value(u.K)), "log10 of temperature")
        for name in EDGES:
            edges = _checked(getattr(self, name), name, (u.Mm, u.arcsec), ndim=(1,))
            require_uniform_grid(edges.value, name)
        shape = (self.y_edges.size - 1, self.x_edges.size - 1, temperature.size)
        if em.shape[:3] != shape:
            raise ValueError(
                f"emission_measure is shaped {em.shape}, but the edges and temperatures give "
                f"(ny, nx, n_temperatures) = {shape}: y along the slit first.")
        if (self.velocity is None) != (em.ndim == 3):
            raise ValueError("A VDEM has a velocity axis, (ny, nx, n_temperatures, "
                             "n_velocities), and its velocity bin centres; a DEM has neither.")
        if self.velocity is not None:
            velocity = _checked(self.velocity, "velocity", (u.km / u.s,), ndim=(1,))
            if velocity.size != em.shape[3]:
                raise ValueError(f"emission_measure has {em.shape[3]} velocity bins, but "
                                 f"velocity has {velocity.size} values.")
            if np.any(np.diff(velocity.value) <= 0):
                raise ValueError("velocity must increase.")
        if self.electron_density is not None:
            density = _checked(self.electron_density, "electron_density", (u.cm**-3,))
            if density.shape not in ((), shape[:2], shape):
                raise ValueError(
                    f"electron_density must be one value, or shaped (ny, nx) = {shape[:2]} "
                    f"or (ny, nx, n_temperatures) = {shape}, got {density.shape}.")
            if np.any(density.value <= 0):
                raise ValueError("electron_density must be above zero.")
        if self.time is not None:
            _checked(self.time, "time", (u.s,), ndim=(0,))
        if not isinstance(self.source, str):
            raise TypeError(f"source must be a string, got {type(self.source).__name__}.")

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape, `(ny, nx)`."""
        return (self.y_edges.size - 1, self.x_edges.size - 1)


def is_emission_measure_file(path: str | Path) -> bool:
    """Whether `path` is an HDF5 file marked as an ECLIPSE emission measure file."""
    try:
        with h5py.File(path, "r") as f:
            return _text_attribute(f.attrs, "format") == FORMAT_NAME
    except OSError:
        return False


def write_emission_measure(emission_measure: EmissionMeasure, path: str | Path) -> Path:
    """
    Write an emission measure file.

    Parameters
    ----------
    emission_measure : EmissionMeasure
        DEM or VDEM to write.
    path : str or Path
        File to write. An existing file is replaced.

    Returns
    -------
    Path
        Path of the written file.
    """
    path = Path(path)
    with _replacing(path) as partial, h5py.File(partial, "w") as f:
        f.attrs["format"] = FORMAT_NAME
        f.attrs["version"] = FORMAT_VERSION
        f.attrs["source"] = emission_measure.source
        for name in ("emission_measure", "temperature", *EDGES, "velocity", "electron_density",
                     "time"):
            value = getattr(emission_measure, name)
            if value is not None:
                _write_dataset(f, name, value)
    return path


def read_emission_measure_layout(path: str | Path) -> dict:
    """
    Contents of an emission measure file, without its emission measure.

    Returns a dict of `x_edges`, `y_edges`, `temperature`, `velocity` (None
    for a DEM) and `electron_density` (None if absent), each as in the file.
    """
    path = Path(path)
    with _open_to_read(path, "emission measure") as f:
        _check_format(f, path, kind="emission measure", format_name=FORMAT_NAME,
                      format_version=FORMAT_VERSION)
        for name in ("emission_measure", "temperature", *EDGES):
            if name not in f:
                raise ValueError(f"{path} has no '{name}' dataset.")
        return {name: _read_dataset(f, name, units=UNITS) if name in f else None
                for name in (*EDGES, "temperature", "velocity", "electron_density")}


def read_emission_measure(path: str | Path, columns: Optional[slice] = None) -> EmissionMeasure:
    """
    Read an emission measure file.

    Parameters
    ----------
    path : str or Path
        File written by `write_emission_measure`.
    columns : slice, optional
        Contiguous slice of pixels along x to read. Default None reads the
        whole image.

    Returns
    -------
    EmissionMeasure
    """
    path = Path(path)
    layout = read_emission_measure_layout(path)
    if columns is not None:
        nx = layout["x_edges"].size - 1
        first = 0 if columns.start is None else columns.start
        last = nx if columns.stop is None else columns.stop
        if columns.step not in (None, 1) or not 0 <= first < last <= nx:
            raise ValueError(f"columns must be a slice of neighbouring pixels among the "
                             f"{nx} along x, got {columns}.")
        columns = slice(first, last)
        layout["x_edges"] = layout["x_edges"][first:last + 1]
    with _open_to_read(path, "emission measure") as f:
        fields = {"emission_measure": _read_dataset(f, "emission_measure", columns,
                                                    units=UNITS, axis=1)}
        density = layout.pop("electron_density")
        if density is not None and density.ndim >= 2 and columns is not None:
            density = density[:, columns]
        if "time" in f:
            fields["time"] = _read_dataset(f, "time", units=UNITS)
        source = str(_text_attribute(f.attrs, "source", ""))
    try:
        return EmissionMeasure(source=source, electron_density=density, **fields, **layout)
    except (TypeError, ValueError) as error:
        raise type(error)(f"{path}: {error}") from None
