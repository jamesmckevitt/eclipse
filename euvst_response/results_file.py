"""
The results file: what the instrument simulation works out, as HDF5.

``eclipse --config run.yaml`` writes ``run/result/run.h5``, which
:func:`euvst_response.load_instrument_response_results` reads back. HDF5
is what the atmosphere and synthesis files are too: reading one runs no
code, as unpickling does, and any language can read it.

File layout, version 1
----------------------
Root attributes ``format`` (``"eclipse-results"``) and ``version`` (``1``).
The results are a tree of mappings, the root group being the top one:

- an array of numbers with dimensions is a dataset, with a ``unit``
  attribute if it is a quantity;
- a mapping, list or tuple that holds such arrays is a group;
- anything else is an attribute of its group, holding JSON.

The group of a mapping lists its keys, in order, as JSON in an
``eclipse_order`` attribute. A few groups are tagged by an ``eclipse_type``
attribute:

``ndcube``
    An NDCube: its ``data``, with the cube's ``unit``, its ``wcs`` and its
    ``meta``.

``map``
    A mapping whose keys cannot name members, as the results of each
    combination are keyed by the tuple of parameters that produced them:
    the keys as JSON in ``eclipse_keys``, and the values as members ``0``,
    ``1``, ...

``list``, ``tuple``
    A sequence that holds arrays: its length in ``eclipse_length``, and its
    items as members ``0``, ``1``, ...

The JSON is plain but for values it has no form for, which are objects
with an ``__eclipse__`` entry naming what they are:

- ``quantity`` (``value`` and ``unit``), ``array`` (``dtype`` and
  ``value``) and ``unit``;
- ``tuple`` and ``map`` (``items``, a map's as key and value pairs);
- ``date`` and ``datetime``, in ISO 8601;
- ``path``, and ``resource``, a path inside the installed package, such as
  a throughput table, relative to the package so that it names the
  reader's copy rather than the writer's;
- ``numpy_type``, such as the precision a time series is synthesised in;
- ``wcs``, a FITS ``header`` with the ``cunit`` it was written in, since a
  header gives the units in SI and a wavelength axis set up in cm would
  otherwise come back in m: the same coordinates, but not the numbers a
  caller reads out of ``wcs.wcs.cdelt``;
- ``dataclass``, a configuration object, by its ``class``, the ``fields``
  it was made with and what it ``derived`` from them, such as a detector's
  dark current. Only the classes in :func:`_dataclass_registry` are
  written or rebuilt, so that a file cannot have anything else
  constructed, and one that a later version would not make as it was
  stored is rebuilt unchecked, with a warning.

Results written by older versions are pickles. They still load, with a
warning, until a future release stops reading them, and
:func:`convert_results_pickle` rewrites one as a results file.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import json
import os
import tempfile
import warnings
from importlib.resources import files
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ndcube import NDCube

from .atmosphere import _check_format

__all__ = ["FORMAT_NAME", "FORMAT_VERSION", "save_results", "load_results",
           "convert_results_pickle", "is_results_file"]

FORMAT_NAME = "eclipse-results"
FORMAT_VERSION = 1

# The tag of a JSON value this module has to rebuild.
TAG = "__eclipse__"
# Attributes of a group that describe it rather than hold one of its values.
TYPE, ORDER, KEYS, LENGTH = "eclipse_type", "eclipse_order", "eclipse_keys", "eclipse_length"
_RESERVED = {TYPE, ORDER, KEYS, LENGTH, "format", "version"}

# Arrays smaller than this are written as they are; larger ones compressed.
_COMPRESS_FROM = 1024


@functools.lru_cache(maxsize=None)
def _dataclass_registry() -> dict:
    """Classes that may be rebuilt from a file, by name.

    Imported lazily: config and fitting both import from utils, and utils is
    imported by this module's callers.
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
    """The directory this installation of ECLIPSE keeps its package data in."""
    return Path(os.fspath(files("euvst_response")))


# ----------------------------------------------------------------------
# WCS
# ----------------------------------------------------------------------
def _wcs_to_tree(wcs: WCS) -> dict:
    """A WCS as a FITS header, plus the units it was expressed in.

    ``WCS.to_header`` normalises the axis units to SI, so a wavelength axis
    set up in cm comes back in m and its CDELT is rescaled to match. The
    coordinates are the same either way, but the numbers a caller reads out
    of ``wcs.wcs.cdelt`` are not, so the original units are recorded and put
    back on the way in.
    """
    # Read the units from the caller's WCS: a copy already reports the SI
    # ones. to_header() normalises the WCS it is called on in place, so it
    # runs on a copy, and saving a cube leaves the caller's units alone.
    cunit = [str(c) for c in wcs.wcs.cunit]
    return {"header": wcs.deepcopy().to_header().tostring(sep="\n"), "cunit": cunit}


def _wcs_from_tree(tree: dict) -> WCS:
    """Rebuild a WCS written by :func:`_wcs_to_tree`."""
    wcs = WCS(fits.Header.fromstring(tree["header"], sep="\n"))
    for axis, wanted in enumerate(tree.get("cunit") or []):
        current = str(wcs.wcs.cunit[axis])
        if not wanted or current == wanted:
            continue
        # Axis units are plain scalings (cm to m, arcsec to deg), so the
        # reference value and the step scale by the same factor and the
        # reference pixel does not move.
        factor = u.Unit(current).to(u.Unit(wanted))
        wcs.wcs.cdelt[axis] *= factor
        wcs.wcs.crval[axis] *= factor
        wcs.wcs.cunit[axis] = wanted
    return wcs


# ----------------------------------------------------------------------
# Values as JSON
# ----------------------------------------------------------------------
def _is_array(value) -> bool:
    """Whether *value* is an array to write as a dataset: numbers or booleans, with dimensions."""
    return (isinstance(value, np.ndarray) and value.ndim > 0
            and value.dtype.kind in "biufc")


def _holds_array(value) -> bool:
    if _is_array(value) or isinstance(value, NDCube):
        return True
    if isinstance(value, dict):
        return any(_holds_array(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_holds_array(v) for v in value)
    return False


def _jsonable(value):
    """*value* as something json can write, with the kinds JSON lacks tagged."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        item = value.item()
        # A longdouble, say, has no Python number to stand for it.
        if isinstance(item, np.generic):
            raise TypeError(f"A {type(value).__name__} cannot be written to a results file.")
        return _jsonable(item)
    if isinstance(value, u.Quantity):
        return {TAG: "quantity", "value": _jsonable(value.value), "unit": value.unit.to_string()}
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _jsonable(value.item())
        if value.dtype.kind not in "biufcU":
            raise TypeError(f"An array of {value.dtype} cannot be written to a results file.")
        return {TAG: "array", "dtype": value.dtype.str, "value": value.tolist()}
    if isinstance(value, u.UnitBase):
        return {TAG: "unit", "value": value.to_string()}
    if isinstance(value, WCS):
        return {TAG: "wcs", **_wcs_to_tree(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        registry = _dataclass_registry()
        if registry.get(type(value).__name__) is not type(value):
            raise TypeError(f"A {type(value).__name__} cannot be written to a results file: "
                            f"only {', '.join(sorted(registry))} are rebuilt when it is read.")
        # What the object works out for itself, such as a detector's dark
        # current from its temperature, is kept too, as the run had it.
        derived = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)
                   if not field.init and hasattr(value, field.name)}
        return {TAG: "dataclass", "class": type(value).__name__,
                "fields": {name: _jsonable(field) for name, field in _init_fields(value).items()},
                "derived": {name: _jsonable(field) for name, field in derived.items()}}
    if isinstance(value, type) and issubclass(value, np.generic):
        return {TAG: "numpy_type", "value": np.dtype(value).name}
    if isinstance(value, datetime.datetime):
        return {TAG: "datetime", "value": value.isoformat()}
    if isinstance(value, datetime.date):
        return {TAG: "date", "value": value.isoformat()}
    # Path, and anything else that describes a filesystem location: the
    # throughput tables arrive as importlib.resources traversables, which are
    # not always a pathlib.Path.
    if isinstance(value, Path) or hasattr(value, "__fspath__"):
        try:
            relative = Path(os.fspath(value)).relative_to(_package_root())
        except ValueError:
            return {TAG: "path", "value": os.fspath(value)}
        return {TAG: "resource", "value": relative.as_posix()}
    if isinstance(value, tuple):
        return {TAG: "tuple", "items": [_jsonable(v) for v in value]}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        if all(isinstance(k, str) for k in value) and TAG not in value:
            return {k: _jsonable(v) for k, v in value.items()}
        return {TAG: "map", "items": [[_jsonable(k), _jsonable(v)] for k, v in value.items()]}
    raise TypeError(f"A {type(value).__name__} cannot be written to a results file.")


def _init_fields(value) -> dict:
    """The arguments a configuration object was made with, as far as it has them.

    Its other fields are worked out in __post_init__, such as a detector's
    dark current from its temperature, and are worked out again when it is
    read. An object from an older version's pickle can lack a field added
    since, which then takes today's default.
    """
    fields, missing = {}, []
    # What the object itself holds, so that a field it lacks is not taken
    # silently from the class's default.
    own = getattr(value, "__dict__", None)
    for field in dataclasses.fields(value):
        if not field.init:
            continue
        if own is None:
            fields[field.name] = getattr(value, field.name)
        elif field.name in own:
            fields[field.name] = own[field.name]
        else:
            missing.append(field.name)
    if missing:
        warnings.warn(f"This {type(value).__name__} has no {', '.join(missing)}, which it gained "
                      f"after it was written; it gets today's default.", UserWarning,
                      stacklevel=2)
    return fields


def _unjson(value):
    """Rebuild what :func:`_jsonable` wrote."""
    if isinstance(value, list):
        return [_unjson(v) for v in value]
    if not isinstance(value, dict):
        return value
    tag = value.get(TAG)
    if tag is None:
        return {k: _unjson(v) for k, v in value.items()}
    if tag == "quantity":
        return u.Quantity(_unjson(value["value"]), u.Unit(value["unit"]))
    if tag == "array":
        return np.array(value["value"], dtype=_plain_dtype(value["dtype"]))
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
        return _rebuild(registry[name], {k: _unjson(v) for k, v in value["fields"].items()},
                        {k: _unjson(v) for k, v in value.get("derived", {}).items()})
    if tag == "numpy_type":
        return _plain_dtype(value["value"]).type
    if tag == "datetime":
        return datetime.datetime.fromisoformat(value["value"])
    if tag == "date":
        return datetime.date.fromisoformat(value["value"])
    if tag == "path":
        return Path(value["value"])
    if tag == "resource":
        return _package_root() / value["value"]
    if tag == "tuple":
        return tuple(_unjson(v) for v in value["items"])
    if tag == "map":
        return {_unjson(k): _unjson(v) for k, v in value["items"]}
    raise ValueError(f"The results file holds a value tagged {tag!r}, which this ECLIPSE "
                     f"does not know.")


def _plain_dtype(name) -> np.dtype:
    """The dtype *name*, which must be one of plain numbers, booleans or text, as ECLIPSE writes.

    A dtype can also give each element a shape of its own, which would have
    a few bytes of a crafted file fill as much memory as it liked.
    """
    dtype = np.dtype(name)
    if dtype.kind not in "biufcU" or dtype.subdtype is not None or dtype.names is not None:
        raise ValueError(f"The results file holds values of type {name!r}, which a results "
                         f"file does not.")
    return dtype


def _rebuild(cls, stored: dict, derived: dict):
    """
    A configuration object from what a file stored.

    It is made again, so that it is checked and works out what it works out,
    and is then given what it worked out when the run made it. A file from a
    version whose objects this version would not make, because a setting has
    gone or a check has been added since, still reads, with a warning: the
    object is then rebuilt as it was stored, unchecked.
    """
    known = {field.name for field in dataclasses.fields(cls) if field.init}
    gone = sorted(set(stored) - known)
    if gone:
        warnings.warn(f"The {cls.__name__} in the results file has {', '.join(gone)}, which "
                      f"this version of ECLIPSE no longer has; they are left out.",
                      UserWarning, stacklevel=2)
    arguments = {key: item for key, item in stored.items() if key in known}
    try:
        obj = cls(**arguments)
    except (TypeError, ValueError) as error:
        warnings.warn(f"The {cls.__name__} in the results file is not one this version of "
                      f"ECLIPSE would make ({error}), so it is rebuilt as it was stored, "
                      f"unchecked.", UserWarning, stacklevel=2)
        obj = cls.__new__(cls)
        for key, item in arguments.items():
            object.__setattr__(obj, key, item)
    for key, item in derived.items():
        object.__setattr__(obj, key, item)
    return obj


def _to_json(value) -> str:
    return json.dumps(_jsonable(value))


# ----------------------------------------------------------------------
# The tree as groups
# ----------------------------------------------------------------------
def _is_name(key) -> bool:
    """Whether *key* can name a member of a group, as a dataset or an attribute."""
    return (isinstance(key, str) and key not in ("", ".") and "/" not in key
            and key not in _RESERVED)


def _put(group: h5py.Group, name: str, value, compression) -> None:
    """Write *value* as member *name* of *group*: a dataset or group if it holds arrays, else a JSON attribute."""
    if not _holds_array(value):
        group.attrs[name] = _to_json(value)
    elif _is_array(value):
        data = value.value if isinstance(value, u.Quantity) else value
        options = {}
        if compression and data.size >= _COMPRESS_FROM:
            options = {"compression": compression, "shuffle": True,
                       **({"compression_opts": 1} if compression == "gzip" else {})}
        dataset = group.create_dataset(name, data=data, **options)
        if isinstance(value, u.Quantity):
            dataset.attrs["unit"] = value.unit.to_string()
    elif isinstance(value, NDCube):
        left_out = [part for part in ("mask", "uncertainty", "psf")
                    if getattr(value, part, None) is not None]
        if left_out:
            warnings.warn(f"{group.name.rstrip('/')}/{name}: a results file does not hold a "
                          f"cube's {' or '.join(left_out)}, so this cube's are left out.",
                          UserWarning, stacklevel=2)
        child = group.create_group(name, track_order=True)
        child.attrs[TYPE] = "ndcube"
        # The data with the cube's unit, as any other quantity is written.
        data = np.asarray(value.data)
        _put(child, "data", data if value.unit is None else u.Quantity(data, value.unit, copy=False),
             compression)
        child.attrs["wcs"] = _to_json(value.wcs)
        _put(child, "meta", dict(value.meta or {}), compression)
    elif isinstance(value, dict):
        _put_dict(group.create_group(name, track_order=True), value, compression)
    else:
        child = group.create_group(name, track_order=True)
        child.attrs[TYPE] = "tuple" if isinstance(value, tuple) else "list"
        child.attrs[LENGTH] = len(value)
        for index, item in enumerate(value):
            _put(child, str(index), item, compression)


def _put_dict(group: h5py.Group, value: dict, compression) -> None:
    if all(_is_name(key) for key in value):
        group.attrs[ORDER] = json.dumps(list(value))
        for key, item in value.items():
            _put(group, key, item, compression)
    else:
        group.attrs[TYPE] = "map"
        group.attrs[KEYS] = _to_json(list(value))
        for index, item in enumerate(value.values()):
            _put(group, str(index), item, compression)


def _get(group: h5py.Group, name: str, path: Path):
    """Read member *name* of *group*, a dataset, a group or a JSON attribute."""
    if name in group.attrs:
        return _unjson(json.loads(group.attrs[name]))
    link = group.get(name, getlink=True)
    if link is None:
        raise ValueError(f"{path}: {group.name} has no member {name!r}.")
    # A link or a dataset whose data lives elsewhere would have the reader
    # open another file, which a results file never needs.
    if not isinstance(link, h5py.HardLink):
        raise ValueError(f"{path}: {group.name}/{name} is a link, which a results file "
                         f"does not hold.")
    node = group[name]
    # Each member is written once, so one reached twice, which would be read
    # twice or, reached from inside itself, without end, is not ECLIPSE's.
    if h5py.h5o.get_info(node.id).rc > 1:
        raise ValueError(f"{path}: {node.name} is reached from more than one place, which a "
                         f"results file does not do.")
    if isinstance(node, h5py.Dataset):
        if node.is_virtual or node.external:
            raise ValueError(f"{path}: {node.name} keeps its data in another file, which a "
                             f"results file does not do.")
        data = node[()]
        unit = node.attrs.get("unit")
        return data if unit is None else u.Quantity(data, u.Unit(unit), copy=False)
    return _get_group(node, path)


def _describing(group: h5py.Group, name: str, path: Path):
    """The attribute *name* that describes *group*, which a results file always writes."""
    if name not in group.attrs:
        raise ValueError(f"{path}: {group.name} has no {name!r} attribute, which a results "
                         f"file gives it.")
    return group.attrs[name]


def _get_group(group: h5py.Group, path: Path):
    kind = group.attrs.get(TYPE)
    if kind == "ndcube":
        data = _get(group, "data", path)
        unit = None
        if isinstance(data, u.Quantity):
            data, unit = data.value, data.unit
        return NDCube(data, wcs=_get(group, "wcs", path), unit=unit,
                      meta=_get(group, "meta", path))
    if kind == "map":
        keys = _unjson(json.loads(_describing(group, KEYS, path)))
        return {key: _get(group, str(index), path) for index, key in enumerate(keys)}
    if kind in ("list", "tuple"):
        length = int(_describing(group, LENGTH, path))
        items = [_get(group, str(index), path) for index in range(length)]
        return items if kind == "list" else tuple(items)
    if kind is not None:
        raise ValueError(f"{path}: {group.name} is of a kind, {kind!r}, this ECLIPSE does "
                         f"not know.")
    order = json.loads(_describing(group, ORDER, path))
    members = (set(group) | set(group.attrs)) - _RESERVED
    if set(order) != members:
        raise ValueError(f"{path}: the members of {group.name} are not those it lists.")
    return {key: _get(group, key, path) for key in order}


# ----------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------
def is_results_file(path: str | Path) -> bool:
    """
    Whether *path* is HDF5, as a results file is, rather than an older pickle.

    Which HDF5 file it is, :func:`load_results` checks.
    """
    return h5py.is_hdf5(str(path))


def save_results(path: str | Path, payload: dict, *, compression: str | None = "gzip") -> Path:
    """
    Write *payload* as a results file.

    Parameters
    ----------
    path : str or Path
        Where to write; a file there is replaced once the new one is
        complete, so a write that fails leaves it as it was. A ``.pkl``
        suffix is replaced with ``.h5``, since the file is not a pickle.
    payload : dict
        The tree to write, keyed by strings that can name an HDF5 member.
    compression : str or None, optional
        An h5py compression filter for the larger arrays; None writes them
        uncompressed, which is faster for a large run.

    Returns
    -------
    Path
        The file written.
    """
    path = Path(path)
    if path.suffix == ".pkl":
        path = path.with_suffix(".h5")
        warnings.warn(f"The results were written to {path.name}, an HDF5 file, rather than "
                      f"a .pkl name that would misdescribe it.", UserWarning, stacklevel=2)
    bad = [key for key in payload if not _is_name(key)]
    if bad:
        raise ValueError(f"The results cannot be keyed by {bad}: the keys have to name "
                         f"HDF5 members, and {sorted(_RESERVED)} are taken.")
    path.parent.mkdir(parents=True, exist_ok=True)
    # A name of its own, so that two runs saving the same name at once
    # cannot write into, or remove, each other's.
    descriptor, partial = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.",
                                           suffix=".part")
    os.close(descriptor)
    partial = Path(partial)
    try:
        with h5py.File(partial, "w", track_order=True) as f:
            f.attrs["format"] = FORMAT_NAME
            f.attrs["version"] = FORMAT_VERSION
            _put_dict(f, dict(payload), compression)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def load_results(path: str | Path, _stacklevel: int = 2) -> dict:
    """
    Read a results file, or a results pickle as older versions wrote them.

    The format is taken from the file itself rather than its name, so a
    pickle still loads, with a warning, which also says so when the ``.h5``
    of a later run is beside it. A ``.pkl`` name that no longer exists, as
    a script written for an older version asks for, reads the ``.h5`` the
    simulation now writes in its place, also with a warning.

    Parameters
    ----------
    path : str or Path
        File to read.

    Returns
    -------
    dict
        The stored payload.
    """
    path = Path(path)
    written = path.with_suffix(".h5")
    if path.suffix == ".pkl" and not path.exists() and written.is_file():
        warnings.warn(f"{path} does not exist, so {written}, which the instrument simulation "
                      f"now writes in its place, is read instead. Name the .h5 file to read it "
                      f"without this warning.", FutureWarning, stacklevel=_stacklevel)
        path = written

    if not path.is_file():
        older = path.with_suffix(".pkl")
        if path.suffix == ".h5" and older.is_file():
            raise FileNotFoundError(
                f"No results file {path}. {older}, beside it, is a results pickle from an older "
                f"version of ECLIPSE: load_results reads it, and "
                f"euvst_response.convert_results_pickle rewrites it as a results file.")
        raise FileNotFoundError(f"No results file {path}.")
    if not is_results_file(path):
        if not _is_pickle(path):
            raise ValueError(f"{path} is neither a results file nor a results pickle.")
        later = ""
        if (path.suffix == ".pkl" and written.is_file()
                and written.stat().st_mtime > path.stat().st_mtime):
            later = (f" {written}, beside it, is newer: the results of a later run, as the "
                     f"simulation now writes them.")
        warnings.warn(
            f"{path} is a results pickle, as older versions of ECLIPSE wrote them. Pickles "
            f"are deprecated and will not be read in a future release: convert it with "
            f"euvst_response.convert_results_pickle, or re-run the simulation. Reading a "
            f"pickle runs whatever code it holds, so only read files you trust.{later}",
            FutureWarning, stacklevel=_stacklevel)
        return _load_pickle(path)

    with h5py.File(path, "r") as f:
        _check_format(f, path, kind="results", format_name=FORMAT_NAME,
                      format_version=FORMAT_VERSION)
        return _get_group(f, path)


def _is_pickle(path: Path) -> bool:
    """Whether *path* starts as the pickles ECLIPSE wrote do, with dill's protocol marker."""
    with open(path, "rb") as handle:
        return handle.read(1) == b"\x80"


def _load_pickle(path: Path) -> dict:
    import dill

    with open(path, "rb") as handle:
        return dill.load(handle)


def convert_results_pickle(pickle_path: str | Path, path: str | Path | None = None,
                           overwrite: bool = False) -> Path:
    """
    Rewrite a results pickle, as older versions wrote them, as a results file.

    What the pickle holds is kept, and the file written is read back before
    it is returned. A configuration object from a version older than one of
    its settings gets today's default for it, with a warning, and a pickle
    that holds no results, such as a synthesis pickle, is refused. Reading a
    pickle runs whatever code it holds, so only convert files you trust.

    Parameters
    ----------
    pickle_path : str or Path
        The pickle.
    path : str or Path, optional
        The results file to write. None writes it beside the pickle, with
        the same name ending in ``.h5``.
    overwrite : bool, optional
        Replace a file already at *path*, which is otherwise refused.

    Returns
    -------
    Path
        The file written.
    """
    pickle_path = Path(pickle_path)
    if is_results_file(pickle_path):
        raise ValueError(f"{pickle_path} is already an HDF5 file.")
    target = pickle_path.with_suffix(".h5") if path is None else Path(path)
    if target.suffix == ".pkl":
        target = target.with_suffix(".h5")
    if target.resolve() == pickle_path.resolve():
        raise ValueError(f"{target} is the pickle itself; name another file to write.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass overwrite=True to replace it.")
    payload = _load_pickle(pickle_path)
    if not isinstance(payload, dict) or "all_combinations" not in (payload.get("results") or {}):
        raise ValueError(f"{pickle_path} holds no results. A synthesis pickle converts with "
                         f"euvst_response.convert_synthesis_pickle.")
    # What the writing and the reading back warn of, such as a setting an old
    # configuration object lacks, is passed on as the caller's, once.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        written = save_results(target, payload)
        # Read back, so that a file is not left that cannot be read.
        try:
            load_results(written)
        except Exception as error:
            written.unlink()
            raise ValueError(f"{pickle_path} converted to a file that could not be read back, "
                             f"which is removed: {error}") from error
    for message in dict.fromkeys((str(w.message), w.category) for w in caught):
        warnings.warn(message[0], message[1], stacklevel=2)
    return written
