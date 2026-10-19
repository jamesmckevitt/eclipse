"""
Results file: the instrument simulation's results as HDF5.

`eclipse --config run.yaml` writes `run/result/run.h5`, which
:func:`euvst_response.load_instrument_response_results` reads. Reading a
file runs no code from it, and any language can read it.

File layout, version 1
----------------------
Root attributes `format` (`"eclipse-results"`) and `version` (`1`). The
results are a tree. An array with dimensions is a dataset, with a `unit`
attribute if it is a quantity; a cube, or a mapping, list or tuple holding
such arrays, is a group; anything else is a JSON attribute of its group.

A mapping keyed by names lists them in order, as JSON, in `eclipse_order`.
Other groups have an `eclipse_type` (plain text, as are `eclipse_length`,
`target` and `unit`):

- `ndcube`: an NDCube's `data` with its `unit`, `wcs` and `meta`.
- `map`: a mapping whose keys cannot name members, such as results keyed
  by parameter tuples; the keys as JSON in `eclipse_keys`, the values as
  members `0`, `1`, ...
- `list`, `tuple`: the length in `eclipse_length`, the items as members.
- `ref`: a value found in more than one place, written once and elsewhere
  as a group whose `target` is the path it was written at.

Values JSON cannot hold are objects with an `__eclipse__` entry naming the
kind: `float` (not finite), `numpy`, `quantity`, `array`, `unit`, `tuple`,
`map`, `date`, `datetime`, `set`, `frozenset`, `bytes` (base 64), `path`,
`resource` (a path inside the installed package), `numpy_type`, `wcs` (a
FITS header plus the unrounded `cunit`, `crpix`, `crval`, and `cdelt` and
`pc` or `crota`, or `cd`) and `dataclass` (a configuration object's
`class`, `fields` and `derived` values). Only the classes in
:func:`_dataclass_registry` are rebuilt.

Results pickles from older versions still load, with a warning, and
:func:`convert_results_pickle` converts one.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime
import functools
import json
import math
import os
import re
import sys
import warnings
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path, PurePosixPath

import astropy.units as u
import h5py
import numpy as np
from astropy.io import fits
from astropy.utils.masked import Masked
from astropy.wcs import WCS
from ndcube import NDCube

from .atmosphere import _check_format, _is_pickle, _new_partial

__all__ = ["FORMAT_NAME", "FORMAT_VERSION", "save_results", "load_results",
           "convert_results_pickle", "is_results_file"]

FORMAT_NAME = "eclipse-results"
FORMAT_VERSION = 1

# Key that tags a JSON value this module must rebuild.
TAG = "__eclipse__"
# Group attributes that describe the group rather than hold one of its values.
TYPE, ORDER, KEYS, LENGTH = "eclipse_type", "eclipse_order", "eclipse_keys", "eclipse_length"
_RESERVED = {TYPE, ORDER, KEYS, LENGTH, "format", "version"}

# Suffixes older versions and scripts gave pickles, as accepted by the synthesis.
_PICKLE_SUFFIXES = (".pkl", ".pickle", ".dill")

# Arrays with at least this many elements are compressed.
_COMPRESS_FROM = 1024


@functools.lru_cache(maxsize=None)
def _dataclass_registry() -> dict:
    """Classes that may be rebuilt from a file, keyed by name.

    Imported when first needed, since the time series settings import the
    synthesis, which reading results does not otherwise need.
    """
    from .config import (AluminiumFilter, Detector_EIS, Detector_SWC,
                         Simulation, Telescope_EIS, Telescope_EUVST)
    from .fitting import FitComponent, FitConfig
    from .raster import RasterPlan, SynthesisSettings

    return {cls.__name__: cls for cls in (
        AluminiumFilter, Detector_EIS, Detector_SWC, Simulation,
        Telescope_EIS, Telescope_EUVST, FitComponent, FitConfig,
        RasterPlan, SynthesisSettings,
    )}


def _package_root() -> Path:
    """Directory of this ECLIPSE installation's package data."""
    return Path(os.fspath(files("euvst_response")))


# ----------------------------------------------------------------------
# WCS
# ----------------------------------------------------------------------
def _wcs_to_tree(wcs: WCS) -> dict:
    """
    A WCS as a FITS header, plus the numbers of its axes unchanged.

    A header gives the axes in SI units and to 14 digits, so a wavelength
    axis in cm would come back in m, and a step of 0.2 arcsec as
    0.2000000000000016. The coordinates agree to that precision, but not
    the numbers a caller reads from `wcs.wcs`. So the units, reference
    pixels and values, and the steps and rotation or CD matrix are kept
    beside the header as given, and restored on reading.
    """
    # Read these before to_header(), which normalises a WCS in place; it is
    # called on a copy so that saving a cube leaves the caller's WCS alone.
    tree = {"cunit": [str(c) for c in wcs.wcs.cunit], "crpix": wcs.wcs.crpix.tolist(),
            "crval": wcs.wcs.crval.tolist()}
    # The rotation in the form the WCS uses, checked in wcslib's order:
    # a PC matrix (the default), then a CD matrix, then CROTA.
    lin = wcs.wcs
    if lin.has_pc() or not (lin.has_cd() or lin.has_crota()):
        tree.update(cdelt=lin.cdelt.tolist(),
                    pc=lin.pc.tolist() if lin.has_pc() else np.eye(lin.naxis).tolist())
    elif lin.has_cd():
        tree["cd"] = lin.cd.tolist()
    else:
        tree.update(cdelt=lin.cdelt.tolist(), crota=lin.crota.tolist())
    tree["header"] = wcs.deepcopy().to_header().tostring(sep="\n")
    if wcs.has_distortion:
        _warn("A results file does not hold a WCS's distortion, so this WCS's is left out.")
    return tree


# Header cards a WCS is rebuilt from: its axes, and any frame, time and
# observer. The number of axes (wcslib allocates its square) and the steps
# and rotation or CD matrix come from the values kept beside the header,
# since a header gives a CD matrix as a PC matrix and steps, which would
# take precedence. Other cards are left out, such as a distortion's, which
# astropy tabulates at whatever size the header asks.
_WCS_CARDS = re.compile(
    r"WCSNAME|(CTYPE|CUNIT|CRPIX|CRVAL|CNAME|CRDER|CSYER|CZPHS|CPERI)\d{1,2}"
    r"|(PV|PS)\d{1,2}_\d{1,2}|LONPOLE|LATPOLE|RADESYS|EQUINOX|RESTFRQ|RESTWAV|VELREF"
    r"|SPECSYS|SSYSOBS|SSYSSRC|VELOSYS|ZSOURCE|VELANGL|TIMESYS|TREFPOS|TREFDIR|PLEPHEM"
    r"|TIMEUNIT|TIMEDEL|TIMEPIXR|TIMEOFFS|TIMSYER|TIMRDER|TSTART|TSTOP|TELAPSE|XPOSURE"
    r"|JEPOCH|BEPOCH|(DATE|MJD)-(OBS|BEG|AVG|END)|DATEREF|MJDREF[IF]?|OBSGEO-[XYZBLH]"
    r"|OBSORBIT|DSUN_OBS|HGLN_OBS|HGLT_OBS|CRLN_OBS|CRLT_OBS|RSUN_REF|[ABC]_RADIUS"
    r"|BLON_OBS|BLAT_OBS|BDIS_OBS")


def _wcs_from_tree(tree: dict) -> WCS:
    """Rebuild a WCS written by :func:`_wcs_to_tree`."""
    header = fits.Header.fromstring(tree["header"], sep="\n")
    wcs = WCS(fits.Header([card for card in header.cards if _WCS_CARDS.fullmatch(card.keyword)]))
    if wcs.naxis != len(tree["cunit"]):
        raise ValueError("The results file holds a WCS whose header does not have its axes.")
    for axis, unit in enumerate(tree["cunit"]):
        wcs.wcs.cunit[axis] = unit
    wcs.wcs.crpix, wcs.wcs.crval = tree["crpix"], tree["crval"]
    if "cd" in tree:
        wcs.wcs.cd = tree["cd"]
    elif "crota" in tree:
        wcs.wcs.cdelt, wcs.wcs.crota = tree["cdelt"], tree["crota"]
    else:
        wcs.wcs.cdelt, wcs.wcs.pc = tree["cdelt"], tree["pc"]
    return wcs


# ----------------------------------------------------------------------
# Values as JSON
# ----------------------------------------------------------------------
def _is_array(value) -> bool:
    """Whether `value` is a dataset: an array of numbers or booleans with dimensions.

    A masked array is not, as its mask would be lost; the JSON writer refuses it.
    """
    return (isinstance(value, np.ndarray) and value.ndim > 0 and value.dtype.kind in "biufc"
            and not isinstance(value, (np.ma.MaskedArray, Masked)))


def _holds_array(value) -> bool:
    if _is_array(value) or isinstance(value, NDCube):
        return True
    if isinstance(value, dict):
        return any(_holds_array(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_array(v) for v in value)
    return False


def _warn(message: str) -> None:
    """Warn with `message`, attributed to the first caller outside this module."""
    frame, level = sys._getframe(1), 2
    while frame is not None and frame.f_globals.get("__name__") == __name__:
        frame, level = frame.f_back, level + 1
    warnings.warn(message, UserWarning, stacklevel=level)


def _unit_text(unit) -> str:
    """Text that reads back as `unit`; TypeError if there is none."""
    if not isinstance(unit, u.UnitBase):
        raise TypeError(f"The unit {unit} cannot be written to a results file, as it is not a "
                        f"plain unit.")
    return _checked_unit_text(unit)


@functools.lru_cache(maxsize=None)
def _checked_unit_text(unit: u.UnitBase) -> str:
    # Write the whole scale, since astropy's text gives it to six digits.
    text = unit.to_string()
    if unit.scale != 1:
        text = (f"{float(unit.scale)!r} "
                f"{u.CompositeUnit(1, unit.bases, unit.powers).to_string()}").strip()
    try:
        same = u.Unit(text) == unit
    except ValueError:
        same = False
    # A unit defined only in a script, for example, is unknown to a reader.
    if not same:
        raise TypeError(f"The unit {unit} cannot be written to a results file, as it would not "
                        f"read back as itself.")
    return text


def _package_parts(text: str):
    """Parts of `text`, a path relative to the package, or None if it could lead outside it."""
    parts = PurePosixPath(text).parts
    if parts and all(part != ".." and re.fullmatch(r"[\w.-]+", part) for part in parts):
        return parts
    return None


def _jsonable(value):
    """`value` in a form json can write, with kinds JSON lacks tagged."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, np.generic):
        item = value.item()
        # A longdouble, for example, has no Python equivalent, and a NumPy
        # date or time has none that keeps its units.
        if isinstance(item, np.generic) or value.dtype.kind not in "biufS":
            raise TypeError(f"Values of type {type(value).__name__} cannot be written to a "
                            f"results file.")
        if value.dtype.kind == "S":
            return _jsonable(item)
        # Keep the type, as for a key of a cube's sampling or a setting given
        # in single precision.
        return {TAG: "numpy", "dtype": value.dtype.str, "value": _jsonable(item)}
    if isinstance(value, float):
        return value if math.isfinite(value) else {TAG: "float", "value": repr(value)}
    if isinstance(value, (np.ma.MaskedArray, Masked)):
        raise TypeError("A masked array cannot be written to a results file: its mask would be "
                        "lost.")
    if isinstance(value, u.Quantity):
        return {TAG: "quantity", "value": _jsonable(value.value), "unit": _unit_text(value.unit)}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufU":
            raise TypeError(f"An array of {value.dtype} cannot be written to a results file.")
        return {TAG: "array", "dtype": value.dtype.str, "shape": list(value.shape),
                "value": _jsonable(value.tolist())}
    if isinstance(value, u.UnitBase):
        return {TAG: "unit", "value": _unit_text(value)}
    if isinstance(value, WCS):
        return {TAG: "wcs", **_wcs_to_tree(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        registry = _dataclass_registry()
        if registry.get(type(value).__name__) is not type(value):
            raise TypeError(f"A {type(value).__name__} cannot be written to a results file: "
                            f"only {', '.join(sorted(registry))} are rebuilt when it is read.")
        # Also keep the fields the object derives, such as a detector's dark
        # current from its temperature, with the values the run used.
        derived = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)
                   if not field.init and hasattr(value, field.name)}
        return {TAG: "dataclass", "class": type(value).__name__,
                "fields": {name: _jsonable(field) for name, field in _init_fields(value).items()},
                "derived": {name: _jsonable(field) for name, field in derived.items()}}
    if isinstance(value, type) and issubclass(value, np.generic):
        try:
            name = _plain_dtype(value).name
        except (TypeError, ValueError):
            raise TypeError(f"The type {value.__name__} cannot be written to a results "
                            f"file.") from None
        return {TAG: "numpy_type", "value": name}
    # YAML reads !!set and !!binary as these, so a configuration can hold them.
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(v) for v in value]
        try:
            items = sorted(items)
        except TypeError:
            pass
        return {TAG: "frozenset" if isinstance(value, frozenset) else "set", "items": items}
    if isinstance(value, bytes):
        return {TAG: "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, datetime.datetime):
        return {TAG: "datetime", "value": value.isoformat()}
    if isinstance(value, datetime.date):
        return {TAG: "date", "value": value.isoformat()}
    # Paths, and anything else with a place on the filesystem, such as the
    # throughput tables found through importlib.resources, which need not be
    # a pathlib.Path. A path inside the package is written relative to it, so
    # the reader finds its own copy; any other is written as it is.
    if isinstance(value, Path) or hasattr(value, "__fspath__"):
        try:
            relative = Path(os.fspath(value)).relative_to(_package_root()).as_posix()
        except ValueError:
            relative = None
        if relative is not None and _package_parts(relative):
            return {TAG: "resource", "value": relative}
        return {TAG: "path", "value": os.fspath(value)}
    if isinstance(value, tuple):
        return {TAG: "tuple", "items": [_jsonable(v) for v in value]}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value) and TAG not in value:
            return {k: _jsonable(v) for k, v in value.items()}
        return {TAG: "map", "items": [[_jsonable(k), _jsonable(v)] for k, v in value.items()]}
    raise TypeError(f"Values of type {type(value).__name__} cannot be written to a results file.")


# Settings whose value in runs made before they existed differs from
# today's default. An object or file from then that lacks one gets the old
# value, not the default. Fits were unweighted before FitConfig.weighted.
_EARLIER_VALUES = {"FitConfig": {"weighted": False},
                   "Telescope_EUVST": {"resolution_budget": "none"}}


def _init_fields(value) -> dict:
    """Arguments a configuration object was made with, as far as it has them.

    Fields derived in __post_init__, such as a detector's dark current from
    its temperature, are written separately as `derived` and restored with
    the run's values. An object from an older version's pickle may lack a
    field added since. It then gets today's default, or, for fields in
    `_EARLIER_VALUES` such as FitConfig's `weighted`, the value runs had
    before the field existed. An object from another version may have a
    setting this version lacks. It is written too, so that reading the file
    reports that it is left out.
    """
    fields, missing, earlier = {}, [], []
    # Use what the instance itself holds, so a missing field is not silently
    # taken from the class default.
    own = getattr(value, "__dict__", None)
    before = _EARLIER_VALUES.get(type(value).__name__, {})
    for field in dataclasses.fields(value):
        if not field.init:
            continue
        if own is None:
            fields[field.name] = getattr(value, field.name)
        elif field.name in own:
            fields[field.name] = own[field.name]
        elif field.name in before:
            fields[field.name] = before[field.name]
            earlier.append(field.name)
        else:
            missing.append(field.name)
    for name in earlier:
        _warn(f"This {type(value).__name__} has no {name}, which ECLIPSE added after it "
              f"was made; it gets {before[name]!r}, as runs then had.")
    if own is not None:
        names = {field.name for field in dataclasses.fields(value)}
        fields.update({key: item for key, item in own.items() if key not in names
                       and not key.startswith("_") and not hasattr(type(value), key)})
    if missing:
        _warn(f"This {type(value).__name__} has no {', '.join(missing)}, which ECLIPSE added "
              f"after it was made; it gets today's default.")
    return fields


def _unjson(value, reading: _Reading):
    """Rebuild what :func:`_jsonable` wrote; `reading` is the state of the file read."""
    if isinstance(value, list):
        return [_unjson(v, reading) for v in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(TAG)
    if tag is None:
        return {k: _unjson(v, reading) for k, v in value.items()}
    if tag == "quantity":
        return u.Quantity(_unjson(value["value"], reading), u.Unit(value["unit"]), dtype=None)
    if tag == "numpy":
        return _plain_dtype(value["dtype"]).type(_unjson(value["value"], reading))
    if tag == "float":
        return float(value["value"])
    if tag == "array":
        dtype = _plain_dtype(value["dtype"])
        items = np.array(_unjson(value["value"], reading),
                         dtype=object if dtype.kind == "U" else dtype)
        # Text items must fit the width the file declares.
        if dtype.kind == "U" and not all(isinstance(item, str) and len(item) <= dtype.itemsize // 4
                                         for item in items.flat):
            raise ValueError("A text array in the results file holds more than text of its width.")
        # Use the shape as written, which an empty array's items do not give.
        shape = value["shape"]
        if (not all(isinstance(size, int) and size >= 0 for size in shape)
                or math.prod(shape) != items.size):
            raise ValueError("An array in the results file does not have the shape it gives.")
        return items.astype(dtype, copy=False).reshape(shape)
    if tag == "unit":
        return u.Unit(value["value"])
    if tag == "wcs":
        return _wcs_from_tree(value)
    if tag == "dataclass":
        registry = _dataclass_registry()
        name = value["class"]
        if name not in registry:
            raise ValueError(f"The results file names a class ECLIPSE will not construct: "
                             f"{name!r}. Only {', '.join(sorted(registry))} are rebuilt, so "
                             f"that reading a file cannot construct anything the file chooses.")
        return _rebuild(registry[name],
                        {k: _unjson(v, reading) for k, v in value["fields"].items()},
                        {k: _unjson(v, reading) for k, v in value["derived"].items()},
                        reading)
    if tag == "numpy_type":
        return _plain_dtype(value["value"]).type
    if tag in ("set", "frozenset"):
        return (set if tag == "set" else frozenset)(_unjson(v, reading) for v in value["items"])
    if tag == "bytes":
        return base64.b64decode(value["value"])
    if tag == "datetime":
        return datetime.datetime.fromisoformat(value["value"])
    if tag == "date":
        return datetime.date.fromisoformat(value["value"])
    if tag == "path":
        return Path(value["value"])
    if tag == "resource":
        # Check the path as written, before touching the filesystem, so a
        # file cannot make the reader look outside the package at all.
        parts = _package_parts(value["value"])
        if parts is None:
            raise ValueError(f"The results file names a package file, {value['value']!r}, "
                             f"outside the package.")
        return _package_root().joinpath(*parts)
    if tag == "tuple":
        return tuple(_unjson(v, reading) for v in value["items"])
    if tag == "map":
        return {_unjson(k, reading): _unjson(v, reading) for k, v in value["items"]}
    raise ValueError(f"The results file holds a value tagged {tag!r}, which this ECLIPSE "
                     f"does not know.")


def _plain_dtype(name) -> np.dtype:
    """Dtype `name`, which must be plain numbers, booleans or text, as ECLIPSE writes.

    A dtype can also give each element its own shape, which would let a few
    bytes of a crafted file fill any amount of memory.
    """
    dtype = np.dtype(name)
    if dtype.kind not in "biufcU" or dtype.subdtype is not None or dtype.names is not None:
        raise ValueError(f"The results file holds values of type {name!r}, which a results "
                         f"file does not.")
    return dtype


def _rebuild(cls, stored: dict, derived: dict, reading: _Reading):
    """
    Configuration object rebuilt from what a file stored.

    The object is constructed again, so it is checked and computes its
    derived fields, which are then set to the values the run had. If this
    version would not make the object, because a setting has gone or a
    check has been added since, the file still reads: the object is rebuilt
    as stored, unchecked, with a warning given once `reading` is done.
    """
    known = {field.name for field in dataclasses.fields(cls) if field.init}
    # A setting stored under an old name is read under its new one.
    stored = {getattr(cls, "_OLD_NAMES", {}).get(key, key): item for key, item in stored.items()}
    # A setting in _EARLIER_VALUES that the file lacks gets the value runs
    # had before it existed.
    before = {name: item for name, item in _EARLIER_VALUES.get(cls.__name__, {}).items()
              if name not in stored}
    for name, item in before.items():
        reading.notes[f"The {cls.__name__} in the results file has no {name}, which ECLIPSE "
                      f"added after it was made; it gets {item!r}, as runs then had."] = None
    stored = {**stored, **before}
    # A setting added since the file was made gets today's default, with a
    # warning, as the run did not have it. One with no default makes the
    # construction below fail, and is reported there.
    missing = sorted(field.name for field in dataclasses.fields(cls)
                     if field.init and field.name not in stored
                     and (field.default is not dataclasses.MISSING
                          or field.default_factory is not dataclasses.MISSING))
    if missing:
        reading.notes[f"The {cls.__name__} in the results file has no {', '.join(missing)}, "
                      f"which ECLIPSE added after it was made; "
                      f"{'it gets' if len(missing) == 1 else 'they get'} today's "
                      f"default{'' if len(missing) == 1 else 's'}."] = None
    gone = sorted(set(stored) - known)
    if gone:
        reading.notes[f"The {cls.__name__} in the results file has {', '.join(gone)}, which "
                      f"this version of ECLIPSE does not have, so "
                      f"{'it is' if len(gone) == 1 else 'they are'} left out."] = None
    arguments = {key: item for key, item in stored.items() if key in known}
    try:
        obj = cls(**arguments)
    except (TypeError, ValueError, AttributeError) as error:
        reading.notes[f"The {cls.__name__} in the results file is not one this version of "
                      f"ECLIPSE would make ({error}), so it is rebuilt as it was stored, "
                      f"unchecked."] = None
        obj = cls.__new__(cls)
        for field in dataclasses.fields(cls):
            if field.name in arguments:
                object.__setattr__(obj, field.name, arguments[field.name])
            elif field.default is not dataclasses.MISSING:
                object.__setattr__(obj, field.name, field.default)
            elif field.default_factory is not dataclasses.MISSING:
                object.__setattr__(obj, field.name, field.default_factory())
    # Set only the derived fields, so a file cannot set anything else past
    # the checks.
    worked_out = {field.name for field in dataclasses.fields(cls) if not field.init}
    for key, item in derived.items():
        if key in worked_out:
            object.__setattr__(obj, key, item)
    return obj


def _to_json(value) -> str:
    # Strict JSON, which any language reads; non-finite numbers are tagged.
    return json.dumps(_jsonable(value), allow_nan=False)


# ----------------------------------------------------------------------
# The tree as groups
# ----------------------------------------------------------------------
def _is_name(key) -> bool:
    """Whether `key` can name a group member, as a dataset or an attribute."""
    if (not isinstance(key, str) or key in ("", ".") or key in _RESERVED
            or "/" in key or "\0" in key):
        return False
    # HDF5 names are UTF-8, which cannot encode a lone surrogate, for example.
    try:
        key.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


class _Unwritable(TypeError):
    """A value a results file cannot hold, and its path in the results."""


def _put(group: h5py.Group, name: str, value, compression, written: dict) -> None:
    """
    Write `value` as member `name` of `group`.

    A value holding arrays becomes a dataset or group, and anything else a
    JSON attribute. `written` maps the id of each object written so far as a
    dataset or group to the object and its path, so an object that appears
    again is written as a reference rather than written again.
    """
    try:
        _put_value(group, name, value, compression, written)
    except _Unwritable:
        raise
    except (TypeError, RecursionError) as error:
        # Give the value's path, which a save failing at the end of a run must report.
        reason = ("A value that holds itself cannot be written to a results file."
                  if isinstance(error, RecursionError) else error)
        raise _Unwritable(f"{group.name.rstrip('/')}/{name}: {reason}") from error


def _put_value(group: h5py.Group, name: str, value, compression, written: dict) -> None:
    if not _holds_array(value):
        group.attrs[name] = _to_json(value)
        return
    if id(value) in written:
        reference = group.create_group(name)
        reference.attrs[TYPE] = "ref"
        reference.attrs["target"] = written[id(value)][1]
        return
    if _is_array(value):
        data = value.value if isinstance(value, u.Quantity) else value
        unit = _unit_text(value.unit) if isinstance(value, u.Quantity) else None
        options = {}
        if compression and data.size >= _COMPRESS_FROM:
            options = {"compression": compression, "shuffle": True,
                       **({"compression_opts": 1} if compression == "gzip" else {})}
        node = group.create_dataset(name, data=data, **options)
        if unit is not None:
            node.attrs["unit"] = unit
    elif isinstance(value, NDCube):
        left_out = [part for part in ("mask", "uncertainty", "psf")
                    if getattr(value, part, None) is not None]
        extra = getattr(value, "extra_coords", None)
        if extra is not None and not getattr(extra, "is_empty", True):
            left_out.append("extra coordinates")
        if len(getattr(value, "global_coords", None) or {}):
            left_out.append("global coordinates")
        if left_out:
            _warn(f"{group.name.rstrip('/')}/{name}: a results file does not hold a cube's "
                  f"{' or '.join(left_out)}, so this cube's are left out.")
        if value.unit is not None:
            _unit_text(value.unit)
        node = group.create_group(name, track_order=True)
        node.attrs[TYPE] = "ndcube"
        # Write the data with the cube's unit, as for any other quantity.
        data = np.asarray(value.data)
        if value.unit is not None:
            data = u.Quantity(data, value.unit, copy=False, dtype=None)
        _put(node, "data", data, compression, written)
        _put(node, "wcs", value.wcs, compression, written)
        # The cube's own meta; one holding arrays that another cube shares is
        # written once.
        meta = value.meta if isinstance(value.meta, dict) else dict(value.meta or {})
        _put(node, "meta", meta, compression, written)
    elif isinstance(value, dict):
        node = group.create_group(name, track_order=True)
        _put_dict(node, value, compression, written)
    else:
        node = group.create_group(name, track_order=True)
        node.attrs[TYPE] = "tuple" if isinstance(value, tuple) else "list"
        node.attrs[LENGTH] = len(value)
        for index, item in enumerate(value):
            _put(node, str(index), item, compression, written)
    # Keep the object itself with its path, so no object written meanwhile
    # can reuse its id.
    written[id(value)] = (value, node.name)


def _put_dict(group: h5py.Group, value: dict, compression, written: dict) -> None:
    if all(_is_name(key) for key in value):
        group.attrs[ORDER] = json.dumps(list(value))
        for key, item in value.items():
            _put(group, key, item, compression, written)
    else:
        group.attrs[TYPE] = "map"
        group.attrs[KEYS] = _to_json(list(value))
        for index, item in enumerate(value.values()):
            _put(group, str(index), item, compression, written)


@dataclasses.dataclass
class _Reading:
    """State of the reading of one file."""

    size: int  # file size in bytes
    stored: int = 0  # bytes stored by the datasets read so far
    seen: set = dataclasses.field(default_factory=set)  # objects reached
    inside: set = dataclasses.field(default_factory=set)  # groups being read
    done: dict = dataclasses.field(default_factory=dict)  # value read at each path
    notes: dict = dataclasses.field(default_factory=dict)  # warnings to give once read


def _attribute(node, name: str, path: Path):
    """Attribute `name` of `node`, or None if absent; one string or integer, as ECLIPSE writes."""
    if name not in node.attrs:
        return None
    # Check before reading: each element of an array attribute, or each field
    # of a compound one, can refer to the same text elsewhere in the file,
    # which reading would copy for each.
    attribute = node.attrs.get_id(name)
    dtype = attribute.dtype
    if (attribute.shape is None or math.prod(attribute.shape) > 1
            or not (h5py.check_string_dtype(dtype) or dtype.kind in "iu")):
        raise ValueError(f"{path}: attribute {name!r} of {node.name} is not one string or "
                         f"integer, as a results file writes them.")
    return node.attrs[name]


def _get(group: h5py.Group, name: str, path: Path, reading: _Reading):
    """Read member `name` of `group`: a dataset, a group or a JSON attribute."""
    # A name with a slash in it would be looked up through other groups.
    if not _is_name(name):
        raise ValueError(f"{path}: {group.name} has a member named {name!r}, which a results "
                         f"file does not.")
    if name in group.attrs:
        return _unjson(json.loads(_attribute(group, name, path)), reading)
    link = group.get(name, getlink=True)
    if link is None:
        raise ValueError(f"{path}: {group.name} has no member {name!r}.")
    # A link, or a dataset whose data lives elsewhere, would make the reader
    # open another file, which a results file never needs.
    if not isinstance(link, h5py.HardLink):
        raise ValueError(f"{path}: {group.name}/{name} is a link, which a results file "
                         f"does not hold.")
    return _read(group[name], path, reading)


def _read(node, path: Path, reading: _Reading):
    """Read `node`, only once however often the results refer to it."""
    if node.name in reading.done:
        return reading.done[node.name]
    # ECLIPSE writes each object once, so one reached twice other than by a
    # reference is not ECLIPSE's: it would be read twice, or without end if
    # reached from inside itself. Objects are tracked by their number in the
    # file rather than kept open, since an open dataset keeps what it read.
    number = h5py.h5g.get_objinfo(node.id).objno
    if number in reading.seen:
        raise ValueError(f"{path}: {node.name} is reached from more than one place, which a "
                         f"results file does not do.")
    reading.seen.add(number)
    if isinstance(node, h5py.Dataset):
        _check_dataset(node, path, reading)
        data = node[()]
        unit = _attribute(node, "unit", path)
        value = data if unit is None else u.Quantity(data, u.Unit(unit), copy=False, dtype=None)
    else:
        reading.inside.add(node.name)
        if _attribute(node, TYPE, path) == "ref":
            value = _follow(node, path, reading)
        else:
            value = _get_group(node, path, reading)
        reading.inside.discard(node.name)
    reading.done[node.name] = value
    return value


def _follow(reference: h5py.Group, path: Path, reading: _Reading):
    """Object `reference` refers to, read through hard links within the file."""
    target = _attribute(reference, "target", path)
    if not isinstance(target, str) or not target.startswith("/"):
        raise ValueError(f"{path}: {reference.name} refers to nothing a results file holds.")
    if target in reading.inside:
        raise ValueError(f"{path}: {reference.name} refers to {target}, which it is inside.")
    if target in reading.done:
        return reading.done[target]
    node = reference.file
    for part in target[1:].split("/"):
        if (not isinstance(node, h5py.Group) or not _is_name(part)
                or not isinstance(node.get(part, getlink=True), h5py.HardLink)):
            raise ValueError(f"{path}: {reference.name} refers to {target}, which the file "
                             f"does not hold.")
        node = node[part]
    return _read(node, path, reading)


# Filters ECLIPSE compresses arrays with; any other would make HDF5 look for
# a plugin to read them.
_FILTERS = {h5py.h5z.FILTER_DEFLATE, h5py.h5z.FILTER_SHUFFLE}
# Most bytes deflate can give back per byte stored: 258 per two-bit code.
_DEFLATE_MOST = 1032


def _check_dataset(node: h5py.Dataset, path: Path, reading: _Reading) -> None:
    """
    Refuse, before reading, a dataset that ECLIPSE would not have written.

    What the file declares of its data is checked against what it stores,
    since a crafted file could otherwise declare more than it holds, which
    reading would make up, or have one stored piece read as many.
    """
    if node.is_virtual or node.external:
        raise ValueError(f"{path}: {node.name} keeps its data in another file, which a "
                         f"results file does not do.")
    dtype = node.dtype
    if dtype.kind not in "biufc" or dtype.subdtype is not None or dtype.names is not None:
        raise ValueError(f"{path}: {node.name} holds values of type {dtype}, which a results "
                         f"file does not.")
    plist = node.id.get_create_plist()
    filters = {plist.get_filter(i)[0] for i in range(plist.get_nfilters())}
    if not filters <= _FILTERS:
        raise ValueError(f"{path}: {node.name} is compressed in a way a results file is not.")
    stored, whole = 0, True
    if node.chunks is None:
        stored = node.id.get_storage_size()
    else:
        # Every chunk of the grid must be stored, as an unwritten one would
        # read as made-up values. Count them first, so a grid declared far
        # larger than the file holds is not looped over.
        steps = node.chunks
        grid = [-(-size // step) for size, step in zip(node.shape, steps)]
        whole = node.id.get_num_chunks() == math.prod(grid)
        for index in np.ndindex(*grid) if whole else ():
            chunk = node.id.get_chunk_info_by_coord(tuple(i * step for i, step in zip(index, steps)))
            if chunk.byte_offset is None:
                whole = False
                break
            stored += chunk.size
    most = stored * (_DEFLATE_MOST if h5py.h5z.FILTER_DEFLATE in filters else 1)
    if not whole or node.nbytes > most:
        raise ValueError(f"{path}: {node.name} does not hold all of its data.")
    reading.stored += stored
    if reading.stored > reading.size:
        raise ValueError(f"{path}: {node.name} shares what it stores with another dataset, "
                         f"which a results file does not do.")


def _describing(group: h5py.Group, name: str, path: Path):
    """Attribute `name` describing `group`, which a results file always writes."""
    value = _attribute(group, name, path)
    if value is None:
        raise ValueError(f"{path}: {group.name} has no {name!r} attribute, which a results "
                         f"file gives it.")
    return value


def _get_group(group: h5py.Group, path: Path, reading: _Reading):
    kind = _attribute(group, TYPE, path)
    if kind == "ndcube":
        data = _get(group, "data", path, reading)
        unit = None
        if isinstance(data, u.Quantity):
            data, unit = data.value, data.unit
        return NDCube(data, wcs=_get(group, "wcs", path, reading), unit=unit,
                      meta=_get(group, "meta", path, reading))
    if kind == "map":
        keys = _unjson(json.loads(_describing(group, KEYS, path)), reading)
        return {key: _get(group, str(index), path, reading) for index, key in enumerate(keys)}
    if kind in ("list", "tuple"):
        length = int(_describing(group, LENGTH, path))
        items = [_get(group, str(index), path, reading) for index in range(length)]
        return items if kind == "list" else tuple(items)
    if kind is not None:
        raise ValueError(f"{path}: {group.name} is of a kind, {kind!r}, this ECLIPSE does "
                         f"not know.")
    order = json.loads(_describing(group, ORDER, path))
    members = (set(group) | set(group.attrs)) - _RESERVED
    if (not isinstance(order, list) or len(set(order)) != len(order)
            or set(order) != members):
        raise ValueError(f"{path}: the members of {group.name} are not those it lists.")
    return {key: _get(group, key, path, reading) for key in order}


# ----------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------
def is_results_file(path: str | Path) -> bool:
    """
    Whether `path` is HDF5, as a results file is, rather than an older pickle.

    :func:`load_results` checks which kind of HDF5 file it is.
    """
    return h5py.is_hdf5(os.fspath(Path(path).expanduser()))


def save_results(path: str | Path, payload: dict, *, compression: str | None = "gzip") -> Path:
    """
    Write `payload` as a results file.

    Parameters
    ----------
    path : str or Path
        Where to write. An existing file is replaced only once the new one
        is complete, so a failed write leaves it unchanged. A pickle suffix
        such as `.pkl` is replaced with `.h5`, since the file is not a
        pickle.
    payload : dict
        Tree to write, keyed by strings that can name an HDF5 member.
    compression : {"gzip", None}, optional
        Compress the larger arrays with gzip, which any HDF5 reader can
        undo; None writes them uncompressed, which is faster for a large run.

    Returns
    -------
    Path
        File written.
    """
    path = Path(path).expanduser()
    if path.suffix.lower() in _PICKLE_SUFFIXES:
        path = path.with_suffix(".h5")
        warnings.warn(f"The results were written to {path.name}, an HDF5 file, rather than "
                      f"a pickle's name that would misdescribe it.", UserWarning, stacklevel=2)
    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory; name the results file to write.")
    if compression not in ("gzip", None):
        raise ValueError(f"compression must be 'gzip' or None, got {compression!r}.")
    if not isinstance(payload, Mapping):
        raise TypeError(f"The results must be a mapping, got {type(payload).__name__}.")
    bad = [key for key in payload if not _is_name(key)]
    if bad:
        raise ValueError(f"The results cannot be keyed by {bad}: the keys have to name "
                         f"HDF5 members, and {sorted(_RESERVED)} are taken.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write through a symlink to its target, as writing to the link would.
    destination = Path(os.path.realpath(path)) if path.is_symlink() else path
    # Do not replace a read-only file, as writing into it would fail.
    if destination.exists() and not os.access(destination, os.W_OK):
        raise PermissionError(f"{destination} is read-only, so the results are not written "
                              f"over it.")
    partial = _new_partial(destination)
    try:
        with h5py.File(partial, "w", track_order=True) as f:
            f.attrs["format"] = FORMAT_NAME
            f.attrs["version"] = FORMAT_VERSION
            _put_dict(f, dict(payload), compression, {})
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path


# HDF5 file suffixes. A pickle with one is not read, since a reader would
# take such a file to be safe to open.
_HDF5_SUFFIXES = (".h5", ".hdf5", ".hdf", ".he5")


def load_results(path: str | Path, *, _stacklevel: int = 2) -> dict:
    """
    Read a results file, or a results pickle written by an older version.

    A pickle still loads, with a warning, which also mentions a later run's
    `.h5` beside it. A pickle named as an HDF5 file is refused, since a
    reader would take it to be safe to open, and unpickling runs whatever
    code the file holds. A `.pkl` path that does not exist, as a script
    written for an older version may ask for, reads the `.h5` the
    simulation writes in its place, also with a warning.

    Parameters
    ----------
    path : str or Path
        File to read.

    Returns
    -------
    dict
        Stored payload.
    """
    path = Path(path).expanduser()
    # The .h5 the simulation writes in place of a pickle of this name.
    written = path.with_suffix(".h5") if path.suffix.lower() in _PICKLE_SUFFIXES else None
    if written is not None and not path.exists() and written.is_file():
        warnings.warn(f"{path} does not exist, so {written}, which the instrument simulation "
                      f"now writes in its place, is read instead. Name the .h5 file to read it "
                      f"without this warning.", FutureWarning, stacklevel=_stacklevel)
        path, written = written, None

    if path.is_dir():
        raise IsADirectoryError(f"{path} is a directory; name the results file to read.")
    if not path.is_file():
        if path.suffix.lower() in _HDF5_SUFFIXES:
            for older in (path.with_suffix(suffix) for suffix in _PICKLE_SUFFIXES):
                if older.is_file():
                    raise FileNotFoundError(
                        f"No results file {path}. {older}, beside it, is a results pickle from "
                        f"an older version of ECLIPSE: load_instrument_response_results reads "
                        f"it, and euvst_response.convert_results_pickle rewrites it as a "
                        f"results file.")
        raise FileNotFoundError(f"No results file {path}.")
    if not is_results_file(path):
        if not _is_pickle(path):
            raise ValueError(f"{path} is neither a results file nor a results pickle.")
        if path.suffix.lower() in _HDF5_SUFFIXES:
            raise ValueError(f"{path} is named as an HDF5 file but is a pickle, which is not "
                             f"read under such a name, since unpickling runs whatever code the "
                             f"file holds: rename it to .pkl if it is a results pickle you "
                             f"trust.")
        later = ""
        if (written is not None and written.is_file()
                and written.stat().st_mtime > path.stat().st_mtime):
            later = (f" {written}, beside it, is newer: the results of a later run, as the "
                     f"simulation now writes them.")
        warnings.warn(
            f"{path} is a results pickle, as older versions of ECLIPSE wrote them. Pickles "
            f"are deprecated and will not be read in a future release: convert it with "
            f"euvst_response.convert_results_pickle, or re-run the simulation. Reading a "
            f"pickle runs whatever code it holds, so only read files you trust.{later}",
            FutureWarning, stacklevel=_stacklevel)
        payload = _load_pickle(path)
        _give_earlier_values(payload)
        if _from_before_weighting(payload):
            warnings.warn(_BEFORE_WEIGHTING.format(path=path), UserWarning,
                          stacklevel=_stacklevel)
        return payload

    # A damaged or hand-made file can fail anywhere in the reading. Errors are
    # raised again naming the file, as an OSError if the filesystem fails,
    # except a MemoryError.
    reading = _Reading(size=path.stat().st_size)
    try:
        with h5py.File(path, "r") as f:
            # Check these like any other attribute before the format check reads them.
            for name in ("format", "version"):
                _attribute(f, name, path)
            _check_format(f, path, kind="results", format_name=FORMAT_NAME,
                          format_version=FORMAT_VERSION, documented=False)
            results = _get_group(f, path, reading)
    except MemoryError:
        raise
    except OSError as error:
        if str(path) in str(error):
            raise
        raise OSError(f"{path} could not be read: {error}") from error
    except ValueError as error:
        if str(error).startswith(str(path)):
            raise
        raise ValueError(f"{path} is not a results file this ECLIPSE can read: "
                         f"{error}") from error
    except Exception as error:
        raise ValueError(f"{path} is not a results file this ECLIPSE can read: "
                         f"{type(error).__name__}: {error}") from error
    # Give each warning from the reading once, as the caller's, such as for a
    # configuration object this version would not make.
    for message in reading.notes:
        warnings.warn(message, UserWarning, stacklevel=_stacklevel)
    if not isinstance(results, dict):
        raise ValueError(f"{path} is not a results file this ECLIPSE can read: it holds a "
                         f"{type(results).__name__}, not a mapping.")
    if _from_before_weighting(results):
        warnings.warn(_BEFORE_WEIGHTING.format(path=path), UserWarning, stacklevel=_stacklevel)
    return results


_BEFORE_WEIGHTING = ("The results in {path} were made before ECLIPSE weighted its fits by "
                     "each pixel's uncertainty, so their fits are unweighted; fit_weighted is "
                     "set to False.")


def _from_before_weighting(payload) -> bool:
    """
    Whether the results were made before fits could be weighted.

    If so, their fits were unweighted, so `fit_weighted` is set to False. So
    is `weighted` of a pickled FitConfig, which lacks it and would otherwise
    read the class default.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, dict) or "all_combinations" not in results:
        return False
    if "fit_weighted" in results:
        return False
    results["fit_weighted"] = False
    own = getattr(results.get("fit_config"), "__dict__", None)
    if own is not None and "weighted" not in own:
        object.__setattr__(results["fit_config"], "weighted", False)
    return True


def _give_earlier_values(value, seen=None) -> None:
    """
    Give each configuration object in an unpickled payload the `_EARLIER_VALUES` it lacks.

    An object pickled before a setting existed would otherwise read the
    class default, such as a telescope's `resolution_budget`, which runs
    then did not have.
    """
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, dict):
        for item in value.values():
            _give_earlier_values(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _give_earlier_values(item, seen)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        own = getattr(value, "__dict__", None)
        if own is not None:
            for name, item in _EARLIER_VALUES.get(type(value).__name__, {}).items():
                if name not in own:
                    object.__setattr__(value, name, item)


def _load_pickle(path: Path) -> dict:
    import dill

    with open(path, "rb") as handle:
        try:
            return dill.load(handle)
        # Such as a pickle of a class only another version or branch of ECLIPSE had.
        except (AttributeError, ImportError) as error:
            raise ValueError(f"{path} cannot be unpickled by this version of ECLIPSE: "
                             f"{error}") from error


def convert_results_pickle(pickle_path: str | Path, path: str | Path | None = None,
                           overwrite: bool = False) -> Path:
    """
    Rewrite a results pickle from ECLIPSE 0.11.0 and earlier as a results file.

    Everything in the pickle is kept, and the new file is read back as a
    check. A setting that did not exist when the pickle was made gets
    today's default, with a warning, unless runs then had another value:
    their fits were unweighted, so the file records `fit_weighted` False,
    and `weighted` False for a FitConfig in it. A pickle holding no results,
    such as a synthesis pickle, is refused. Reading a pickle runs whatever
    code it holds, so only convert files you trust.

    Parameters
    ----------
    pickle_path : str or Path
        Pickle to convert.
    path : str or Path, optional
        Results file to write. Default None, which writes it beside the
        pickle, with the same name ending in `.h5`.
    overwrite : bool, optional
        Replace a file already at `path`, which is otherwise refused.
        Default False.

    Returns
    -------
    Path
        File written.
    """
    pickle_path = Path(pickle_path).expanduser()
    if is_results_file(pickle_path):
        raise ValueError(f"{pickle_path} is already an HDF5 file.")
    if not _is_pickle(pickle_path):
        raise ValueError(f"{pickle_path} is not a pickle.")
    target = pickle_path.with_suffix(".h5") if path is None else Path(path).expanduser()
    if target.suffix.lower() in _PICKLE_SUFFIXES:
        target = target.with_suffix(".h5")
    if target.is_dir():
        raise IsADirectoryError(f"{target} is a directory; name the results file to write.")
    if target.resolve() == pickle_path.resolve():
        raise ValueError(f"{target} is the pickle itself; name another file to write.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it.")
    # Write through a symlink to its target, as writing to the link would.
    requested = target
    if target.is_symlink():
        target = Path(os.path.realpath(target))
    payload = _load_pickle(pickle_path)
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, dict) or "all_combinations" not in results:
        raise ValueError(f"{pickle_path} holds no results. A synthesis pickle converts with "
                         f"euvst_response.convert_synthesis_pickle.")
    # Recorded in the new file, so reading it does not warn again.
    if _from_before_weighting(payload):
        warnings.warn(_BEFORE_WEIGHTING.format(path=pickle_path), UserWarning, stacklevel=2)
    # Write under a temporary name and read it back before replacing the
    # target, so the result is never an unreadable file or a lost one.
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = _new_partial(target)
    try:
        save_results(staging, payload)
        del payload, results
        try:
            load_results(staging, _stacklevel=3)
        except (MemoryError, OSError):
            raise
        except Exception as error:
            raise ValueError(f"{pickle_path} converts to a file that cannot be read back, "
                             f"so {target} is not written: {error}") from error
        os.replace(staging, target)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return requested
