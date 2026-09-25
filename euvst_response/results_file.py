"""
The results file: what the instrument simulation works out, as HDF5.

``eclipse --config run.yaml`` writes ``run/result/run.h5``, which
:func:`euvst_response.load_instrument_response_results` reads back. HDF5
is what the atmosphere and synthesis files are too: reading one runs no
code, as unpickling does, and any language can read it.

File layout, version 1
----------------------
Root attributes ``format`` (``"eclipse-results"``) and ``version`` (``1``).
The results are a tree of mappings, written as groups: the arrays in them
as datasets, with a ``unit`` attribute where they are quantities, and
everything else as an attribute holding JSON. Each group records the order
of its keys in an ``eclipse_order`` attribute. A few kinds of value are
tagged, a group by an ``eclipse_type`` attribute and a JSON value by an
``__eclipse__`` entry:

``ndcube``
    A group of the cube's ``data``, with its ``unit``, its ``wcs`` as a FITS
    header plus the units it was written in, and its ``meta``. The units
    because a WCS written as a header comes back in SI, so a wavelength
    axis set up in cm would come back in m: the same coordinates, but not
    the numbers a caller reads out of ``wcs.wcs.cdelt``.

``map``
    A mapping whose keys are not names, as the results of each combination
    are keyed by the tuple of parameters that produced them: the keys in an
    ``eclipse_keys`` attribute and the values as members ``0``, ``1``, ...

``list``, ``tuple``
    A sequence holding arrays, its items as members ``0``, ``1``, ...

``dataclass``
    The configuration objects, by class name and the arguments they were
    made with. Only the classes in :func:`_dataclass_registry` are rebuilt,
    so a file cannot have anything else constructed.

``resource``
    A path inside the installed package, such as a throughput table,
    relative to the package, so that it names the reader's copy rather than
    the writer's.

Results written by older versions are pickles. They still load, with a
warning, until a future release stops reading them, and
:func:`convert_results_pickle` rewrites one as a results file.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
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
        return _jsonable(value.item())
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
        return {TAG: "dataclass", "class": type(value).__name__,
                "fields": {name: _jsonable(field) for name, field in _init_fields(value).items()}}
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
        return np.array(value["value"], dtype=np.dtype(value["dtype"]))
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
        fields = {k: _unjson(v) for k, v in value["fields"].items()}
        try:
            return registry[name](**fields)
        except (TypeError, ValueError) as error:
            raise ValueError(f"The {name} in the results file cannot be rebuilt: {error}") from None
    if tag == "numpy_type":
        kind = np.dtype(value["value"]).type
        if not issubclass(kind, np.generic):
            raise ValueError(f"{value['value']!r} is not a NumPy type.")
        return kind
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
        # Only the array given is written, not the whole of the array it may
        # be a view of.
        options = {}
        if compression and data.size >= _COMPRESS_FROM:
            options = {"compression": compression, "shuffle": True,
                       **({"compression_opts": 1} if compression == "gzip" else {})}
        dataset = group.create_dataset(name, data=np.ascontiguousarray(data), **options)
        if isinstance(value, u.Quantity):
            dataset.attrs["unit"] = value.unit.to_string()
    elif isinstance(value, NDCube):
        child = group.create_group(name, track_order=True)
        child.attrs[TYPE] = "ndcube"
        _put(child, "data", np.asarray(value.data), compression)
        child.attrs["unit"] = _to_json(value.unit)
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
    if isinstance(node, h5py.Dataset):
        if node.is_virtual or node.external:
            raise ValueError(f"{path}: {node.name} keeps its data in another file, which a "
                             f"results file does not do.")
        data = node[()]
        unit = node.attrs.get("unit")
        return data if unit is None else u.Quantity(data, u.Unit(unit), copy=False)
    return _get_group(node, path)


def _get_group(group: h5py.Group, path: Path):
    kind = group.attrs.get(TYPE)
    if kind == "ndcube":
        return NDCube(_get(group, "data", path), wcs=_get(group, "wcs", path),
                      unit=_get(group, "unit", path), meta=_get(group, "meta", path))
    if kind == "map":
        keys = _unjson(json.loads(group.attrs[KEYS]))
        return {key: _get(group, str(index), path) for index, key in enumerate(keys)}
    if kind in ("list", "tuple"):
        items = [_get(group, str(index), path) for index in range(int(group.attrs[LENGTH]))]
        return items if kind == "list" else tuple(items)
    if kind is not None:
        raise ValueError(f"{path}: {group.name} is of a kind, {kind!r}, this ECLIPSE does "
                         f"not know.")
    order = json.loads(group.attrs[ORDER])
    members = (set(group) | set(group.attrs)) - _RESERVED
    if set(order) != members:
        raise ValueError(f"{path}: the members of {group.name} are not those it lists.")
    return {key: _get(group, key, path) for key in order}


# ----------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------
def is_results_file(path: str | Path) -> bool:
    """Whether *path* is an HDF5 file, as a results file is, rather than an older pickle."""
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
    partial = path.with_name(path.name + ".part")
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
    pickle still loads, with a warning. A ``.pkl`` name that a script
    written for an older version asks for, with the ``.h5`` the simulation
    now writes in its place beside it, reads the ``.h5`` if that is newer,
    as the results of the latest run are, also with a warning.

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
    if path.suffix == ".pkl" and written.is_file() and (
            not path.exists() or written.stat().st_mtime > path.stat().st_mtime):
        reason = ("does not exist" if not path.exists() else
                  "is older, left from a version of ECLIPSE that wrote pickles")
        warnings.warn(f"{path} {reason}, so {written}, which the instrument simulation now "
                      f"writes in its place, is read instead. Name the .h5 file to read it "
                      f"without this warning.", FutureWarning, stacklevel=_stacklevel)
        path = written

    if not path.is_file():
        raise FileNotFoundError(f"No results file {path}.")
    if not is_results_file(path):
        warnings.warn(
            f"{path} is a results pickle, as older versions of ECLIPSE wrote them. Pickles "
            f"are deprecated and will not be read in a future release: convert it with "
            f"euvst_response.convert_results_pickle, or re-run the simulation. Reading a "
            f"pickle runs whatever code it holds, so only read files you trust.",
            FutureWarning, stacklevel=_stacklevel)
        return _load_pickle(path)

    with h5py.File(path, "r") as f:
        _check_format(f, path, kind="results", format_name=FORMAT_NAME,
                      format_version=FORMAT_VERSION)
        return _get_group(f, path)


def _load_pickle(path: Path) -> dict:
    import dill

    with open(path, "rb") as handle:
        return dill.load(handle)


def convert_results_pickle(pickle_path: str | Path, path: str | Path | None = None,
                           overwrite: bool = False) -> Path:
    """
    Rewrite a results pickle, as older versions wrote them, as a results file.

    What the pickle holds is kept, but for what ECLIPSE works out again when
    it reads the file, such as a detector's dark current from its
    temperature. A configuration object from a version older than one of
    its settings gets today's default for it, with a warning. Reading a
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
    return save_results(target, _load_pickle(pickle_path))
