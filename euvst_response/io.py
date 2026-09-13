"""
Reading and writing ECLIPSE result files.

ECLIPSE writes ASDF. Loading a pickle runs whatever code the file asks for,
which is a poor property for a data format that gets emailed around and kept
for years; ASDF is a YAML tree with the arrays in binary blocks alongside, so
reading one cannot execute anything, other languages can read it, and the
metadata stays legible in a text editor.

Not everything ECLIPSE holds has an ASDF representation, so a few types are
written as tagged mappings and rebuilt on the way back in:

``ndcube``
    Data, unit, meta, and the WCS as a FITS header plus the units it was
    written in. A header rather than an ASDF-tagged WCS object because a
    header is the more portable thing to hand another language; the units
    alongside it because both routes normalise a WCS to SI, so a wavelength
    axis set up in cm would otherwise come back in m. The coordinates would
    be the same, but the numbers a caller reads out of ``wcs.wcs.cdelt``
    would not.

``dataclass``
    The configuration objects, by class name and field values. Only the
    classes in :func:`_dataclass_registry` can be rebuilt, so a file cannot
    name an arbitrary class and have it constructed.

``map``
    A dict that does not have string keys. ECLIPSE keys its results by the
    tuple of parameters that produced them, and ASDF only permits str, int
    and bool as mapping keys.

Files written by older versions are pickles. They still load: the reader
picks the format from the file's own magic bytes rather than its name.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from pathlib import Path

import asdf
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from ndcube import NDCube

# Bumped when the tree layout changes in a way a reader has to know about.
FORMAT_VERSION = 1

# Marks a mapping that this module wrote and has to decode.
TAG = "__eclipse__"

ASDF_MAGIC = b"#ASDF"


def _dataclass_registry() -> dict:
    """Classes that may be rebuilt from a file, by name.

    Imported lazily: config and fitting both import from utils, and utils is
    imported by this module's callers.
    """
    from .config import (AluminiumFilter, Detector_EIS, Detector_SWC,
                         Simulation, Telescope_EIS, Telescope_EUVST)
    from .fitting import FitComponent, FitConfig

    return {cls.__name__: cls for cls in (
        AluminiumFilter, Detector_EIS, Detector_SWC, Simulation,
        Telescope_EIS, Telescope_EUVST, FitComponent, FitConfig,
    )}


def _wcs_to_tree(wcs: WCS) -> dict:
    """A WCS as a FITS header, plus the units it was expressed in.

    ``WCS.to_header`` normalises the axis units to SI, so a wavelength axis
    set up in cm comes back in m and its CDELT is rescaled to match. The
    coordinates are the same either way, but the numbers a caller reads out
    of ``wcs.wcs.cdelt`` are not, so the original units are recorded and put
    back on the way in.
    """
    # Read the units first: to_header() normalises the WCS object in place,
    # so afterwards wcs.wcs.cunit already reports the SI ones.
    cunit = [str(c) for c in wcs.wcs.cunit]
    header = wcs.to_header()
    return {
        "header": {key: header[key] for key in header},
        "cunit": cunit,
    }


def _wcs_from_tree(tree: dict) -> WCS:
    """Rebuild a WCS written by :func:`_wcs_to_tree`."""
    header = fits.Header()
    for key, value in tree["header"].items():
        header[key] = value
    wcs = WCS(header)

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


def _encode(obj):
    """Convert *obj* into something ASDF can write."""
    if isinstance(obj, NDCube):
        return {
            TAG: "ndcube",
            "data": np.asarray(obj.data),
            "unit": None if obj.unit is None else obj.unit.to_string(),
            "wcs": _wcs_to_tree(obj.wcs),
            "meta": _encode(dict(obj.meta) if obj.meta else {}),
        }

    if isinstance(obj, WCS):
        return {TAG: "wcs", **_wcs_to_tree(obj)}

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Constructor arguments only. The rest are derived in __post_init__
        # (Detector.dark_current from the CCD temperature, for instance), so
        # storing the inputs and letting the code work them out again keeps
        # an old file consistent with the current derivation rather than
        # carrying a stale answer forward.
        return {
            TAG: "dataclass",
            "class": type(obj).__name__,
            "fields": {f.name: _encode(getattr(obj, f.name))
                       for f in dataclasses.fields(obj) if f.init},
        }

    if isinstance(obj, u.Quantity):
        # Left to asdf-astropy, which has a tag for it.
        return obj

    if isinstance(obj, u.UnitBase):
        return {TAG: "unit", "value": obj.to_string()}

    # Path, and anything else that describes a filesystem location: the
    # throughput tables arrive as importlib.resources traversables, which are
    # not always a pathlib.Path.
    if isinstance(obj, Path) or hasattr(obj, "__fspath__"):
        return {TAG: "path", "value": os.fspath(obj)}

    if isinstance(obj, tuple):
        return {TAG: "tuple", "items": [_encode(v) for v in obj]}

    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            return {k: _encode(v) for k, v in obj.items()}
        return {TAG: "map",
                "items": [[_encode(k), _encode(v)] for k, v in obj.items()]}

    if isinstance(obj, list):
        return [_encode(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj

    # numpy scalars (including np.bool_, which ASDF refuses) become the
    # Python number they stand for.
    if isinstance(obj, np.generic):
        return obj.item()

    return obj


def _decode(obj):
    """Rebuild what :func:`_encode` wrote."""
    if isinstance(obj, dict):
        tag = obj.get(TAG)

        if tag == "ndcube":
            unit = obj["unit"]
            return NDCube(
                np.asarray(obj["data"]),
                wcs=_wcs_from_tree(obj["wcs"]),
                unit=None if unit is None else u.Unit(unit),
                meta=_decode(obj["meta"]),
            )

        if tag == "wcs":
            return _wcs_from_tree(obj)

        if tag == "dataclass":
            registry = _dataclass_registry()
            name = obj["class"]
            if name not in registry:
                raise ValueError(
                    f"Result file names a class ECLIPSE will not construct: "
                    f"{name!r}. Only {', '.join(sorted(registry))} are "
                    f"rebuilt, so that reading a file cannot instantiate "
                    f"anything the file chooses."
                )
            fields = {k: _decode(v) for k, v in obj["fields"].items()}
            return registry[name](**fields)

        if tag == "unit":
            return u.Unit(obj["value"])

        if tag == "path":
            return Path(obj["value"])

        if tag == "tuple":
            return tuple(_decode(v) for v in obj["items"])

        if tag == "map":
            return {_decode(k): _decode(v) for k, v in obj["items"]}

        return {k: _decode(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_decode(v) for v in obj]

    # Before the ndarray branch: Quantity subclasses ndarray, so asarray()
    # would quietly return the numbers without their unit.
    if isinstance(obj, u.Quantity):
        return u.Quantity(np.asarray(obj.value), obj.unit)

    if isinstance(obj, np.ndarray):
        # Materialise, so the array outlives the open file.
        return np.asarray(obj)

    return obj


def is_asdf(path: str | Path) -> bool:
    """True if *path* is an ASDF file, read from its own first bytes."""
    with open(path, "rb") as handle:
        return handle.read(len(ASDF_MAGIC)) == ASDF_MAGIC


def save_results(path: str | Path, payload: dict, *,
                 compression: str | None = "zlib") -> Path:
    """
    Write *payload* to an ASDF file.

    Parameters
    ----------
    path : str or Path
        Where to write. A ``.pkl`` suffix is replaced with ``.asdf``, since
        the contents are no longer a pickle and a name that says otherwise
        is worse than a renamed file.
    payload : dict
        The tree to write. Keys must be strings.
    compression : str or None, optional
        Array compression, passed to ASDF. Default ``"zlib"``. ``None``
        writes the arrays uncompressed, which is faster for a large run.

    Returns
    -------
    Path
        The file actually written.
    """
    path = Path(path)
    if path.suffix == ".pkl":
        path = path.with_suffix(".asdf")
        warnings.warn(
            f"ECLIPSE writes ASDF, so the output was written to {path.name} "
            f"rather than a .pkl name that would misdescribe it.",
            UserWarning,
        )

    tree = {
        "eclipse_format_version": FORMAT_VERSION,
        **{key: _encode(value) for key, value in payload.items()},
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    af = asdf.AsdfFile(tree)
    af.write_to(str(path), all_array_compression=compression)
    return path


def load_results(path: str | Path) -> dict:
    """
    Read a result file, ASDF or pickle.

    The format is taken from the file's magic bytes rather than its name, so
    a file written before ECLIPSE moved to ASDF still loads whatever it is
    called.

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

    if not is_asdf(path):
        import dill

        warnings.warn(
            f"{path.name} is a pickle, written before ECLIPSE moved to ASDF. "
            f"Reading it executes whatever the file contains, so only do "
            f"this for files you produced. Re-running writes ASDF.",
            UserWarning,
        )
        with open(path, "rb") as handle:
            return dill.load(handle)

    # lazy_load and memmap off: the arrays have to outlive the open file.
    with asdf.open(path, lazy_load=False, memmap=False) as af:
        tree = dict(af.tree)

    tree.pop("asdf_library", None)
    tree.pop("history", None)
    tree.pop("eclipse_format_version", None)

    return {key: _decode(value) for key, value in tree.items()}
