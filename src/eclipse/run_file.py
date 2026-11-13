"""
Run files: one YAML file for a whole ECLIPSE job.

A run file is the configuration `eclipse observe` reads. It can also name an
atmosphere to synthesise first, with `atmosphere:` and a `synthesis:`
section; a full-CCD frame to make, with a `frame:` section; and a list of
`runs:`, each a variation on the rest of the file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import NamedTuple, Optional

import astropy.units as u
import numpy as np
import yaml

from .utils import _parse_section, check_config_keys, load_yaml_config

#: Keys of the `frame:` section.
FRAME_KEYS = {"slit_position", "exposure", "slit_width", "band", "band_step", "band_lines",
              "band_file", "continuum", "n_workers", "output"}

# Keys of the `frame:` section that set how the band is synthesised from an
# atmosphere.
_BAND_KEYS = {"band", "band_step", "band_lines", "band_file", "continuum", "n_workers"}

# Keys of the `synthesis:` section the frame's own band synthesis sets itself.
_FRAME_SYNTHESIS_KEYS = {"lines", "continuum", "band", "band_step", "band_lines", "image_x"}

# Synthesis options a run file gives elsewhere: the atmosphere at the top
# level, and where the synthesis goes with `synthesis_file`.
_NOT_SYNTHESIS_KEYS = {"help", "config", "atmosphere", "emission_measure", "output_dir",
                       "output_name"}

# What a run synthesises from: an atmosphere, or an emission measure (a DEM or
# VDEM).
_SOURCES = ("atmosphere", "emission_measure")

# The band of the SW detectors, for a frame's lines, in Angstrom.
_SW_BAND = (169.0, 213.5)


class RunOutput(NamedTuple):
    """Files one run wrote or used."""

    #: Name of the run, which names its files.
    name: str
    #: The synthesis observed, or None.
    synthesis: Optional[Path]
    #: The results of the observation, or None.
    results: Optional[Path]
    #: The full-CCD frame, or None.
    frame: Optional[Path]


# ----------------------------------------------------------------------
# Reading a run file
# ----------------------------------------------------------------------
def merge(base: dict, extra: dict) -> dict:
    """`base` with `extra` laid over it, section by section."""
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def apply_setting(config: dict, setting: str) -> dict:
    """
    `config` with one setting changed, from text such as `simulation.expos=[5 s, 20 s]`.

    The key is a dotted path through the sections, and the value is read as
    YAML, so a list or a number is written as in the file.
    """
    key, equals, value = setting.partition("=")
    if not equals or not key.strip():
        raise ValueError(f"A setting is written key=value, such as "
                         f"simulation.slit_width='0.4 arcsec', got {setting!r}.")
    path = key.strip().split(".")
    changed = copy.deepcopy(config)
    section = changed
    for part in path[:-1]:
        if not isinstance(section.get(part, {}), dict):
            raise ValueError(f"{'.'.join(path[:path.index(part) + 1])!r} is not a section, "
                             f"so {key!r} cannot be set.")
        section = section.setdefault(part, {})
    section[path[-1]] = yaml.safe_load(value)
    return changed


def load(source, name: Optional[str] = None) -> tuple[dict, str, str]:
    """
    A run file's configuration, its name and where it came from.

    Parameters
    ----------
    source : str, Path or dict
        A run file, or its configuration.
    name : str, optional
        Name of the run. Default the file's name without its suffix, or
        `"run"` for a configuration given as a dict.

    Returns
    -------
    config : dict
    name : str
    source : str
        The file, or `name`, for messages.
    """
    if isinstance(source, dict):
        config = copy.deepcopy(source)
        label = name or "run"
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Run file not found: {source}")
        config = load_yaml_config(path.read_text())
        label = str(source)
        name = name or path.stem
    if config is None:
        raise ValueError(f"Run file is empty: {label}")
    if not isinstance(config, dict):
        raise ValueError(f"A run file is a mapping of keys to values, got "
                         f"{type(config).__name__}: {label}")
    return config, name or "run", label


def adjust(config: dict, settings=(), sections: Optional[dict] = None) -> dict:
    """
    A run's configuration with sections laid over it, then settings changed.

    `sections` holds top-level keys or whole sections, laid over the
    configuration with :func:`merge`, and each of `settings` is `key=value`
    as for :func:`apply_setting`.
    """
    config = merge(config, sections or {})
    for setting in settings:
        config = apply_setting(config, setting)
    return config


def runs(config: dict, name: str) -> list[tuple[str, dict]]:
    """
    Name and configuration of each run of a run file.

    Without `runs:` the file is one run. Otherwise each entry of `runs:` is
    laid over the rest of the file, and named `<file name>_<entry name>`.
    """
    if "runs" not in config:
        return [(name, config)]
    entries = config["runs"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("'runs' is a list of runs, each a mapping with a 'name' and the "
                         "settings it changes.")
    base = {key: value for key, value in config.items() if key != "runs"}
    found, seen = [], set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise ValueError(f"'runs[{index}]' needs a 'name', a word naming its files, "
                             f"and the settings it changes.")
        if entry["name"] in seen:
            raise ValueError(f"Two runs are named {entry['name']!r}; each needs its own "
                             f"name, which names its files.")
        seen.add(entry["name"])
        if "runs" in entry:
            raise ValueError(f"'runs[{index}]' has runs of its own; runs do not nest.")
        settings = {key: value for key, value in entry.items() if key != "name"}
        found.append((f"{name}_{entry['name']}", merge(base, settings)))
    return found


# ----------------------------------------------------------------------
# The synthesis
# ----------------------------------------------------------------------
def synthesis_options() -> dict:
    """The keys of a single snapshot's `synthesis:` section, with their options."""
    from .synthesis import build_parser

    return {action.dest: action for action in build_parser()._actions
            if action.option_strings and action.dest not in _NOT_SYNTHESIS_KEYS
            and action.help != argparse.SUPPRESS}


def synthesises(config: dict) -> bool:
    """Whether a run synthesises an atmosphere or an emission measure before observing it."""
    return any(key in config for key in _SOURCES)


def synthesised_from(config: dict) -> tuple[str, str]:
    """The key naming what a run synthesises from, and the file it names."""
    given = [key for key in _SOURCES if key in config]
    if len(given) != 1:
        raise ValueError("A run synthesises one 'atmosphere' or one 'emission_measure'; "
                         f"this one names {' and '.join(given) or 'neither'}.")
    return given[0], str(config[given[0]])


def synthesis_path(config: dict, name: str) -> Path:
    """Where a run's synthesis is: `synthesis_file`, or `run/input/<name>.h5`."""
    given = config.get("synthesis_file")
    return Path(given) if given else Path(f"run/input/{name}.h5")


def _option_words(key: str, action, value) -> list[str]:
    """Command-line words giving a synthesis option its value."""
    flag = action.option_strings[0]
    if action.nargs == 0:
        if not isinstance(value, bool):
            raise ValueError(f"'synthesis.{key}' is true or false, got {value!r}.")
        return [flag] if value else []
    if action.nargs is not None:
        values = value if isinstance(value, (list, tuple)) else [value]
        if isinstance(action.nargs, int) and len(values) != action.nargs:
            raise ValueError(f"'synthesis.{key}' takes {action.nargs} values, such as "
                             f"['0 Mm', '20 Mm'], got {value!r}.")
        return [flag, *(str(item) for item in values)]
    if isinstance(value, (list, tuple, dict)):
        raise ValueError(f"'synthesis.{key}' takes one value, got {value!r}.")
    # Joined, so a value starting with a dash, such as -x, is read as one.
    return [f"{flag}={value}"]


def synthesis_arguments(config: dict, name: str, output: Optional[Path] = None) -> list[str]:
    """
    `synthesise-spectra` command-line words for a run's synthesis.

    The synthesis is written to `output`, or to the run's synthesis file.

    Raises
    ------
    ValueError
        If the run has no atmosphere, or its `synthesis:` section has a key
        that is not a synthesis option.
    """
    if not synthesises(config):
        raise ValueError("A run synthesises the 'atmosphere' it names; this one names none.")
    section = config.get("synthesis") or {}
    if not isinstance(section, dict):
        raise ValueError("The 'synthesis:' section must be a mapping of settings to values.")
    options = synthesis_options()
    check_config_keys(section, options, "'synthesis' section")
    if not section.get("lines") and not section.get("band"):
        raise ValueError("The 'synthesis:' section names the 'lines' to synthesise, such as "
                         "lines: [Fe12_195.1190], or a 'band'.")
    path = synthesis_path(config, name)
    if path.suffix.lower() != ".h5":
        raise ValueError(f"A run writes its synthesis to an HDF5 file ending in .h5, not "
                         f"{path}.")
    output = path if output is None else Path(output)
    key, source = synthesised_from(config)
    words = [f"--{key.replace('_', '-')}={source}", f"--output-dir={output.parent}",
             f"--output-name={output.name}"]
    for key, value in section.items():
        words += _option_words(key, options[key], value)
    return words


# Synthesis settings that change how it is computed, not what it gives.
_COMPUTING_KEYS = ("n_workers", "goft_temperature_chunk", "band_lines")


def _hash(made_from: dict) -> str:
    text = json.dumps(made_from, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _settings_key(config: dict) -> str:
    """
    What a run's synthesis is made with, as one string.

    The atmosphere file's path, the synthesis settings and the version of
    ECLIPSE, without the settings that change only how it is computed.
    """
    from . import __version__

    settings = {key: value for key, value in (config.get("synthesis") or {}).items()
                if key not in _COMPUTING_KEYS}
    key, source = synthesised_from(config)
    return _hash({key: os.path.abspath(source),
                  "synthesis": settings, "eclipse": __version__})


def _run_key(config: dict) -> str:
    """
    What a run's synthesis is made from, as one string.

    The settings, and the atmosphere file's size and the time it was last
    changed.
    """
    status = Path(synthesised_from(config)[1]).resolve().stat()
    return _hash({"settings": _settings_key(config), "size": status.st_size,
                  "modified": status.st_mtime_ns})


def _stored_key(path: Path, name: str = "run_key") -> Optional[str]:
    """A key a run file stored in an HDF5 file it wrote, or None."""
    import h5py

    try:
        with h5py.File(path, "r") as file:
            key = file.attrs.get(name)
    except OSError:
        return None
    return None if key is None else str(key)


# Synthesis files already warned about, once each.
_WARNED = set()


def _made_already(config: dict, name: str, warn: bool = True) -> bool:
    """
    Whether a run's synthesis file holds what it would synthesise.

    With its atmosphere file gone, only the settings can be checked, and a
    warning says so, once for each file.
    """
    path = synthesis_path(config, name)
    key, source = synthesised_from(config)
    if Path(source).is_file():
        return _stored_key(path) == _run_key(config)
    if _stored_key(path, "run_settings_key") != _settings_key(config):
        return False
    if not warn or path in _WARNED:
        return True
    _WARNED.add(path)
    warnings.warn(f"The {key.replace('_', ' ')} {source} is not there, so {path}, made from "
                  f"it with these settings, is used as it is.", UserWarning, stacklevel=2)
    return True


def _write_with_key(path: Path, keys: dict, write) -> None:
    """
    Write a file with `write(partial path)` and mark it with `keys`, attributes of its root.

    The file is written under a name of its own and moved into place only
    once it is complete, so a run stopped part way, or another run writing
    the same file at the same time, leaves no file that looks finished.
    """
    import h5py
    import uuid

    path = Path(os.path.realpath(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.partial{path.suffix}")
    try:
        write(partial)
        if not partial.is_file():
            raise FileNotFoundError(f"Nothing was written to {partial}.")
        with h5py.File(partial, "a") as file:
            for name, key in keys.items():
                file.attrs[name] = key
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _reusable(path: Path, key: str, what: str) -> bool:
    """
    Whether a file a run would write already holds what it would write.

    Raises
    ------
    ValueError
        If the file is there but was not written by a run file, so it is
        not overwritten.
    """
    if not path.exists():
        return False
    stored = _stored_key(path)
    if stored is None:
        raise ValueError(f"{path} is there already, and was not made by a run file. Move it, "
                         f"or name another file for the {what}.")
    if stored == key:
        print(f"Reusing {path}, made from the same files and settings.")
        return True
    print(f"Remaking {path}: its files or settings have changed.")
    return False


def _synthesise_one(config: dict, name: str) -> Path:
    """
    Synthesise a run's atmosphere, unless its synthesis file holds it already.

    Returns
    -------
    Path
        The synthesis file.
    """
    from .synthesis import main as run_synthesis, parse_arguments

    synthesis_arguments(config, name)
    path = synthesis_path(config, name)
    key, source = synthesised_from(config)
    if not Path(source).is_file():
        if _made_already(config, name):
            return path
        raise FileNotFoundError(f"{key.replace('_', ' ').capitalize()} file not found: "
                                f"{source}")
    key = _run_key(config)
    if not _reusable(path, key, "synthesis"):
        keys = {"run_key": key, "run_settings_key": _settings_key(config)}
        _write_with_key(path, keys, lambda partial: run_synthesis(
            parse_arguments(synthesis_arguments(config, name, output=partial))))
    return path


# ----------------------------------------------------------------------
# The observation
# ----------------------------------------------------------------------
def observation_config(config: dict, name: str) -> dict:
    """
    What a run observes, as the configuration `eclipse observe` reads.

    The run's synthesis takes the place of its atmosphere, and its first
    line is the reference line unless one is given.
    """
    observed = {key: value for key, value in config.items()
                if key not in ("frame", "runs")}
    if synthesises(config):
        for key in ("atmosphere_series", "synthesis_series", "uniform_intensity"):
            if key in config:
                raise ValueError(f"{synthesised_from(config)[0]!r} and {key!r} cannot both be "
                                 f"given: each says "
                                 f"what is observed.")
        lines = (config.get("synthesis") or {}).get("lines") or []
        lines = [lines] if isinstance(lines, str) else list(lines)
        del observed[synthesised_from(config)[0]]
        observed.pop("synthesis", None)
        observed["synthesis_file"] = str(synthesis_path(config, name))
        if "reference_line" not in observed and lines:
            observed["reference_line"] = lines[0]
    return observed


def check(config: dict, name: str, steps: tuple = ("synthesise", "observe", "frame")) -> None:
    """
    Check a run's settings for `steps` before anything is computed.

    So a mistake in the observation's or the frame's settings is found
    before an atmosphere is synthesised, which can take hours.
    """
    from .main import _validate_config_keys

    if synthesises(config):
        # A frame synthesises its own band, so it needs no lines.
        if "synthesise" in steps or "observe" in steps:
            synthesis_arguments(config, name)
        key, source = synthesised_from(config)
        if ("synthesise" in steps and not Path(source).is_file()
                and not _made_already(config, name, warn=False)):
            raise FileNotFoundError(f"{key.replace('_', ' ').capitalize()} file not found: "
                                    f"{source}")
    if "frame" in config and "frame" in steps:
        band = _frame_settings(config, name).get("band")
        if (band is not None and not Path(band["atmosphere"]).is_file()
                and not _made_already(band, name, warn=False)):
            raise FileNotFoundError(f"Atmosphere file not found: {band['atmosphere']}")
    observed = observation_config(config, name)
    if "observe" in steps:
        _validate_config_keys(observed, str(observed.get("instrument", "SWC")).upper())


def _observe_one(config: dict, name: str, source: Optional[str] = None,
                 debug: bool = False) -> Path:
    """
    Observe a run's synthesis, or what else it names, and write the results.

    Returns
    -------
    Path
        The results file, `run/result/<name>.h5`.
    """
    from .main import run_config

    observed = observation_config(config, name)
    if synthesises(config):
        path = Path(observed["synthesis_file"])
        if not path.is_file():
            raise FileNotFoundError(f"The synthesis {path} is not made yet: run 'eclipse "
                                    f"synthesise' first, or 'eclipse run'.")
        if not _made_already(config, name):
            key, made_from = synthesised_from(config)
            if not Path(made_from).is_file():
                raise ValueError(f"The synthesis {path} was not made with these settings, and "
                                 f"the {key.replace('_', ' ')} {made_from} it would be made "
                                 f"from again is not there.")
            raise ValueError(f"The synthesis {path} was not made from this "
                             f"{key.replace('_', ' ')} with "
                             f"these settings: run 'eclipse synthesise' or 'eclipse run' "
                             f"to make it again.")
    return run_config(observed, name, source=source, debug=debug)


# ----------------------------------------------------------------------
# The full-CCD frame
# ----------------------------------------------------------------------
# Settings of the simulation: section a frame uses, besides its exposure and
# slit width.
_FRAME_SIMULATION_KEYS = ("psf", "spectral_psf", "noise", "psf_boundary", "slit_mask")


def _one_value(section: str, settings: dict) -> dict:
    """A section's settings parsed, each with one value; a sweep is refused."""
    fixed, sweep = _parse_section(settings, section)
    if sweep:
        raise ValueError(f"A frame takes one value of each setting, but "
                         f"'{section}.{next(iter(sweep))}' has several. Give one, or give "
                         f"the frame's own in the 'frame:' section where it has one.")
    return fixed


def _frame_settings(config: dict, name: str) -> dict:
    """
    Everything a run's frame needs, read from its sections and checked.

    The instrument's settings come from the run's `simulation:`,
    `detector:`, `telescope:` and `filter:` sections, each with one value,
    and the frame's exposure and slit width from `frame:` if it gives them.
    A run with an atmosphere synthesises the band under the slit for its
    frame; one with a `synthesis_file` makes its frame from that file.
    """
    frame = config.get("frame")
    if not isinstance(frame, dict):
        raise ValueError("A run makes a frame from its 'frame:' section; this one has none.")
    check_config_keys(frame, FRAME_KEYS, "'frame' section")
    if str(config.get("instrument", "SWC")).upper() != "SWC":
        raise ValueError("A full-CCD frame is of the SW camera, so the instrument is SWC.")
    if "slit_position" not in frame:
        raise ValueError("The 'frame:' section needs a 'slit_position', such as 0 arcsec.")
    simulation = config.get("simulation") or {}
    used = {key: simulation[key] for key in _FRAME_SIMULATION_KEYS if key in simulation}
    for key, own in (("expos", "exposure"), ("slit_width", "slit_width")):
        if own not in frame and key in simulation:
            used[key] = simulation[key]
    used = _one_value("simulation", used)
    exposure = frame.get("exposure", used.pop("expos", None))
    if exposure is None:
        raise ValueError("A frame needs an exposure time: give 'frame.exposure', or one "
                         "'simulation.expos'.")
    if isinstance(exposure, (list, tuple)):
        if len(exposure) != 1:
            raise ValueError(f"A frame takes one exposure time, got {exposure!r}.")
        exposure = exposure[0]
    exposure = u.Quantity(exposure)
    slit_width = u.Quantity(frame.get("slit_width", used.pop("slit_width", 0.2 * u.arcsec)))
    settings = {
        "slit_position": u.Quantity(frame["slit_position"]),
        "exposure": exposure,
        "slit_width": slit_width,
        "options": used,
        "detector": _one_value("detector", config.get("detector") or {}),
        "telescope": _one_value("telescope", config.get("telescope") or {}),
        "filter": _one_value("filter", config.get("filter") or {}),
        "output": Path(frame.get("output", f"run/result/{name}_frame.h5")),
    }
    if not synthesises(config):
        # The frame holds what the synthesis file holds.
        ignored = sorted(_BAND_KEYS & set(frame))
        if ignored:
            raise ValueError(
                f"'frame.{ignored[0]}' is for a run with an 'atmosphere' or an "
                f"'emission_measure', whose band is synthesised for the frame. This run's "
                f"frame holds the spectra in its 'synthesis_file' as they are.")
        settings["synthesis"] = Path(config.get("synthesis_file", ""))
        return settings
    band = frame.get("band", [f"{edge} Angstrom" for edge in _SW_BAND])
    step = frame.get("band_step")
    try:
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            raise ValueError
        low, high = (u.Quantity(edge).to_value(u.Angstrom) for edge in band)
        if not low < high or (step is not None
                              and not u.Quantity(step).to_value(u.Angstrom) > 0):
            raise ValueError
    except (TypeError, ValueError, u.UnitConversionError):
        raise ValueError(
            f"'frame.band' takes two wavelengths, the shorter first, such as [169 Angstrom, "
            f"213.5 Angstrom], and 'frame.band_step' one, such as 0.002 Angstrom; got "
            f"{band!r} and {step!r}.") from None
    settings["synthesis"] = Path(frame.get("band_file", f"run/input/{name}_band.h5"))
    settings["band"] = _band_config(config, settings, frame, band)
    # Check the band's synthesis settings now, as a run's own are.
    synthesis_arguments(settings["band"], name)
    return settings


def _band_config(config: dict, settings: dict, frame: dict, band) -> dict:
    """
    The run with its synthesis replaced by that of the band under the slit.

    The band is synthesised with the run's own synthesis settings, over the
    image columns the slit takes light from, into the frame's own synthesis
    file.
    """
    from .config import AluminiumFilter, Telescope_EUVST
    from .full_frame import _angle
    from .radiometric import across_slit_fwhm
    from .sampling import _REACH_IN_SIGMA
    from .utils import _fwhm_to_sigma

    # The slit, and as far either side as the blur across it reaches.
    reach = settings["slit_width"].to_value(u.arcsec) / 2
    if settings["options"].get("psf", True):
        telescope = Telescope_EUVST(filter=AluminiumFilter(**settings["filter"]),
                                    **settings["telescope"])
        across = across_slit_fwhm(telescope)
        if across is not None:
            reach += _REACH_IN_SIGMA * _fwhm_to_sigma(across.to_value(u.arcsec))
    centre = float(_angle(settings["slit_position"][np.newaxis])[0])
    reach = float(reach)
    section = {key: value for key, value in (config.get("synthesis") or {}).items()
               if key not in _FRAME_SYNTHESIS_KEYS}
    section.update(band=[str(edge) for edge in band],
                   band_lines=str(frame.get("band_lines", "run/input/band_lines.h5")),
                   continuum=bool(frame.get("continuum", False)),
                   image_x=[f"{centre - reach!r} arcsec", f"{centre + reach!r} arcsec"])
    if "band_step" in frame:
        section["band_step"] = str(frame["band_step"])
    if "n_workers" in frame:
        section["n_workers"] = int(frame["n_workers"])
    return {**config, "synthesis": section, "synthesis_file": str(settings["synthesis"])}


def _frame_one(config: dict, name: str) -> Path:
    """
    Make the full-CCD frame a run's `frame:` section describes.

    Returns
    -------
    Path
        The frame file, `frame.output` or `run/result/<name>_frame.h5`.
    """
    from .config import AluminiumFilter, Detector_SWC, Telescope_EUVST
    from .full_frame import full_ccd_frame, write_full_frame

    settings = _frame_settings(config, name)
    if "band" in settings:
        _synthesise_one(settings["band"], name)
    elif not settings["synthesis"].is_file():
        raise FileNotFoundError(f"The synthesis {settings['synthesis']} is not there: a frame "
                                f"is made from a synthesis, given with 'synthesis_file', or "
                                f"made from the 'atmosphere'.")
    telescope = Telescope_EUVST(filter=AluminiumFilter(**settings["filter"]),
                                **settings["telescope"])
    detector = Detector_SWC(**settings["detector"])
    made = full_ccd_frame(settings["synthesis"], slit_position=settings["slit_position"],
                          exposure=settings["exposure"], slit_width=settings["slit_width"],
                          det=detector, telescope=telescope, **settings["options"])
    path = settings["output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_full_frame(made, path)
    print(f"Wrote the full-CCD frame to {path}")
    return path


# ----------------------------------------------------------------------
# Running every step
# ----------------------------------------------------------------------
class FirstRankFailed(RuntimeError):
    """A step run on the first MPI rank only failed there."""


def _first_rank(step, *args):
    """
    Run `step` on the first MPI rank only, while the others wait.

    The first rank may use every CPU of the job on its node meanwhile, and
    the others wait without using them. If the step fails on the first
    rank, every rank stops.
    """
    import time
    import traceback

    from .main import _allowed_cpus, _cpuset
    from .utils import _get_mpi_info

    comm, rank, size = _get_mpi_info()
    if size == 1:
        return step(*args)
    result, failure = None, None
    if rank == 0:
        pinned = sorted(os.sched_getaffinity(0))
        widest = _allowed_cpus(os.environ, _cpuset(), pinned)
        try:
            if widest:
                os.sched_setaffinity(0, widest)
            result = step(*args)
        except BaseException as error:
            failure = error
            # A refusal says why itself; anything else needs its traceback.
            if not isinstance(error, (ValueError, FileNotFoundError)):
                traceback.print_exc()
        finally:
            os.sched_setaffinity(0, pinned)
    waiting = comm.Ibarrier()
    while not waiting.Test():
        time.sleep(1.0)
    # Every rank reports the error, so it is seen whichever ends the job.
    message = comm.bcast(None if failure is None else f"{type(failure).__name__}: {failure}",
                         root=0)
    if failure is not None:
        raise failure
    if message is not None:
        raise FirstRankFailed(f"On the first MPI rank: {message}")
    return result


#: Steps of each command. `run` makes a frame only for a run with a `frame:`
#: section.
STEPS = {"run": ("synthesise", "observe", "frame"), "synthesise": ("synthesise",),
         "observe": ("observe",), "frame": ("frame",)}


def run_steps(config: dict, name: str, steps: tuple, source: Optional[str] = None,
              debug: bool = False) -> RunOutput:
    """
    One run's steps, from `"synthesise"`, `"observe"` and `"frame"`.

    The synthesis is made only if the run names an atmosphere, and reused
    if its file already holds it. The frame is made only if the run has a
    `frame:` section. The run is checked first by :func:`run_all`.
    """
    synthesis = results = frame = None
    if synthesises(config):
        synthesis = synthesis_path(config, name)
        if "synthesise" in steps:
            _first_rank(_synthesise_one, config, name)
    elif "synthesis_file" in config:
        synthesis = Path(config["synthesis_file"])
    if "observe" in steps:
        results = _observe_one(config, name, source=source, debug=debug)
    if "frame" in steps and "frame" in config:
        frame = _first_rank(_frame_one, config, name)
        if steps == ("frame",) and synthesises(config):
            # The frame alone synthesises only the band it is made from.
            synthesis = _frame_settings(config, name)["synthesis"]
    return RunOutput(name, synthesis, results, frame)


def _check_files_apart(found: list, steps: tuple) -> None:
    """Refuse runs that would write different things to the same file."""
    syntheses, frames = {}, {}
    for run_name, config in found:
        if synthesises(config):
            path = synthesis_path(config, run_name).resolve()
            key = _settings_key(config)
            other = syntheses.setdefault(path, (run_name, key))
            if other[1] != key:
                raise ValueError(f"The runs {other[0]} and {run_name} would synthesise "
                                 f"differently into {path}; give each its own "
                                 f"'synthesis_file'.")
        if "frame" in config and "frame" in steps:
            settings = _frame_settings(config, run_name)
            path = settings["output"].resolve()
            other = frames.setdefault(path, run_name)
            if other != run_name:
                raise ValueError(f"The runs {other} and {run_name} would both write their "
                                 f"frame to {path}; give each its own 'frame.output'.")
            if "band" in settings:
                path, key = settings["synthesis"].resolve(), _settings_key(settings["band"])
                other = syntheses.setdefault(path, (run_name, key))
                if other[1] != key:
                    raise ValueError(f"The runs {other[0]} and {run_name} would synthesise "
                                     f"differently into {path}; give each its own "
                                     f"'frame.band_file'.")


def run_all(source, command: str, *, settings=(), only: Optional[str] = None,
            name: Optional[str] = None, debug: bool = False,
            **sections) -> list[RunOutput]:
    """
    Every run of a run file, through the steps of `command`.

    `command` is one of :data:`STEPS`. `sections` and `settings` change each
    run, after its own settings from `runs:` are laid over the file, except
    those of `runs:` itself, which change the list. `only` picks one run by
    its name in `runs:`. The runs are checked before any is started.
    """
    config, name, label = load(source, name=name)
    of_runs = [setting for setting in settings
               if setting.partition("=")[0].strip().split(".")[0] == "runs"]
    config = adjust(config, of_runs, {key: value for key, value in sections.items()
                                      if key == "runs"})
    settings = [setting for setting in settings if setting not in of_runs]
    sections = {key: value for key, value in sections.items() if key != "runs"}
    found = [(run_name, adjust(run_config, settings, sections))
             for run_name, run_config in runs(config, name)]
    if only is not None:
        picked = [(run_name, run_config) for run_name, run_config in found
                  if run_name == f"{name}_{only}"]
        if not picked:
            names = [run_name[len(name) + 1:] for run_name, _ in found]
            raise ValueError(f"No run named {only!r}" + (
                f"; the runs are {', '.join(names)}." if "runs" in config
                else ": the run file has no 'runs:'."))
        found = picked
    steps = STEPS[command]
    for run_name, run_config in found:
        check(run_config, run_name, steps)
        if command == "frame" and "frame" not in run_config:
            raise ValueError(f"The run {run_name} has no 'frame:' section saying what frame "
                             f"to make.")
    _check_files_apart(found, steps)
    return [run_steps(run_config, run_name, steps, source=label, debug=debug)
            for run_name, run_config in found]


# ----------------------------------------------------------------------
# From Python
# ----------------------------------------------------------------------
def run(source, *, settings=(), only: Optional[str] = None, name: Optional[str] = None,
        **sections) -> list[RunOutput]:
    """
    Do everything a run file describes, as `eclipse run` does.

    Synthesises the atmosphere if the file names one, unless its synthesis
    is made already, then observes it, and makes the full-CCD frame if the
    file has a `frame:` section.

    Parameters
    ----------
    source : str, Path or dict
        A run file, or its configuration.
    settings : sequence of str, optional
        Settings to change, each `key=value` with a dotted key, such as
        `"simulation.expos=[5 s, 20 s]"`.
    only : str, optional
        Only the run of this name in the file's `runs:`.
    name : str, optional
        Name of the run, which names its files. Default the file's name,
        or `"run"` for a dict.
    **sections
        Top-level settings or sections laid over the file's, such as
        `n_iter=10` or `simulation={"expos": "5 s"}`.

    Returns
    -------
    list of RunOutput
        The files of each run.
    """
    return run_all(source, "run", settings=settings, only=only, name=name, **sections)


def synthesise(source, *, settings=(), only: Optional[str] = None,
               name: Optional[str] = None, **sections) -> list[RunOutput]:
    """Synthesise a run file's atmosphere, as `eclipse synthesise` does; see :func:`run`."""
    return run_all(source, "synthesise", settings=settings, only=only, name=name, **sections)


def observe(source, *, settings=(), only: Optional[str] = None,
            name: Optional[str] = None, **sections) -> list[RunOutput]:
    """Observe what a run file describes, as `eclipse observe` does; see :func:`run`."""
    return run_all(source, "observe", settings=settings, only=only, name=name, **sections)


def make_frame(source, *, settings=(), only: Optional[str] = None,
               name: Optional[str] = None, **sections) -> list[RunOutput]:
    """Make a run file's full-CCD frame, as `eclipse frame` does; see :func:`run`."""
    return run_all(source, "frame", settings=settings, only=only, name=name, **sections)
