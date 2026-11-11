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
from pathlib import Path
from typing import NamedTuple, Optional

import astropy.units as u
import numpy as np
import yaml

from .utils import _parse_section, check_config_keys, load_yaml_config

#: Keys of the `frame:` section.
FRAME_KEYS = {"slit_position", "exposure", "slit_width", "band", "band_lines", "continuum",
              "n_workers", "output"}

# Synthesis options a run file gives elsewhere: the atmosphere at the top
# level, and where the synthesis goes with `synthesis_file`.
_NOT_SYNTHESIS_KEYS = {"help", "config", "atmosphere", "output_dir", "output_name"}

# The band of the SW detectors, for a frame's lines, in Angstrom.
_SW_BAND = (169.0, 213.5)

# Wavelengths of the continuum under a frame, finer than a detector row,
# inside the telescope's throughput tables.
_CONTINUUM_WAVELENGTHS = (170.0, 214.0, 10481)


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


def load(source, settings=(), name: Optional[str] = None, **sections) -> tuple[dict, str, str]:
    """
    A run file's configuration, its name and where it came from.

    Parameters
    ----------
    source : str, Path or dict
        A run file, or its configuration.
    settings : sequence of str, optional
        Settings to change, each `key=value` as for :func:`apply_setting`.
    name : str, optional
        Name of the run. Default the file's name without its suffix, or
        `"run"` for a configuration given as a dict.
    **sections
        Top-level keys or whole sections, laid over the file's with
        :func:`merge`.

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
    config = merge(config, sections)
    for setting in settings:
        config = apply_setting(config, setting)
    return config, name or "run", label


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
    """Whether a run synthesises an atmosphere before observing it."""
    return "atmosphere" in config


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


def synthesis_arguments(config: dict, name: str) -> list[str]:
    """
    `synthesise-spectra` command-line words for a run's synthesis.

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
    if not section.get("lines"):
        raise ValueError("The 'synthesis:' section names the 'lines' to synthesise, such as "
                         "lines: [Fe12_195.1190].")
    path = synthesis_path(config, name)
    if path.suffix.lower() != ".h5":
        raise ValueError(f"A run writes its synthesis to an HDF5 file ending in .h5, not "
                         f"{path}.")
    words = [f"--atmosphere={config['atmosphere']}", f"--output-dir={path.parent}",
             f"--output-name={path.name}"]
    for key, value in section.items():
        words += _option_words(key, options[key], value)
    return words


def _run_key(config: dict) -> str:
    """
    What a run's synthesis is made from, as one string.

    The atmosphere file, with its size and the time it was last changed, the
    synthesis settings and the version of ECLIPSE.
    """
    from . import __version__

    atmosphere = Path(config["atmosphere"]).resolve()
    status = atmosphere.stat()
    made_from = {"atmosphere": str(atmosphere), "size": status.st_size,
                 "modified": status.st_mtime_ns, "synthesis": config.get("synthesis") or {},
                 "eclipse": __version__}
    text = json.dumps(made_from, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _stored_key(path: Path) -> Optional[str]:
    """The key a run file stored in an HDF5 file it wrote, or None."""
    import h5py

    try:
        with h5py.File(path, "r") as file:
            key = file.attrs.get("run_key")
    except OSError:
        return None
    return None if key is None else str(key)


def _store_key(path: Path, key: str) -> None:
    import h5py

    with h5py.File(path, "a") as file:
        file.attrs["run_key"] = key


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

    words = synthesis_arguments(config, name)
    path = synthesis_path(config, name)
    if not Path(config["atmosphere"]).is_file():
        raise FileNotFoundError(f"Atmosphere file not found: {config['atmosphere']}")
    key = _run_key(config)
    if _reusable(path, key, "synthesis"):
        return path
    run_synthesis(parse_arguments(words))
    _store_key(path, key)
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
                raise ValueError(f"'atmosphere' and {key!r} cannot both be given: each says "
                                 f"what is observed.")
        lines = (config.get("synthesis") or {}).get("lines") or []
        lines = [lines] if isinstance(lines, str) else list(lines)
        del observed["atmosphere"]
        observed.pop("synthesis", None)
        observed["synthesis_file"] = str(synthesis_path(config, name))
        if "reference_line" not in observed and lines:
            observed["reference_line"] = lines[0]
    return observed


def check(config: dict, name: str) -> None:
    """
    Check a run's keys before anything is computed.

    So a mistake in the observation's settings is found before an
    atmosphere is synthesised, which can take hours.
    """
    from .main import _validate_config_keys

    if synthesises(config):
        synthesis_arguments(config, name)
    if "frame" in config:
        frame = config["frame"]
        if not isinstance(frame, dict):
            raise ValueError("The 'frame:' section must be a mapping of settings to values.")
        check_config_keys(frame, FRAME_KEYS, "'frame' section")
    observed = observation_config(config, name)
    instrument = str(observed.get("instrument", "SWC")).upper()
    _validate_config_keys(observed, instrument)


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
    if synthesises(config) and not Path(observed["synthesis_file"]).is_file():
        raise FileNotFoundError(f"The synthesis {observed['synthesis_file']} is not made yet: "
                                f"run 'eclipse synthesise' first, or 'eclipse run'.")
    return run_config(observed, name, source=source, debug=debug)


# ----------------------------------------------------------------------
# The full-CCD frame
# ----------------------------------------------------------------------
def _single(section: dict, key: str, where: str, default=None):
    """One value of a run's setting: a list of more than one is refused."""
    value = section.get(key, default)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"A frame takes one value of '{where}.{key}', got {value!r}. "
                             f"Give it in the 'frame:' section.")
        value = value[0]
    return value


def _band_lines(config: dict, frame: dict, synthesis: Path, n_workers: int):
    """Every line in the band, worked out or read back."""
    from . import __version__
    from .band import band_contribution_functions, read_band_lines, write_band_lines
    from .synthesis_file import read_synthesis_products

    products = read_synthesis_products(synthesis, keys=["logT_grid", "logN_grid", "config"])
    abundance = products["config"].get("abundance", "sun_coronal_2021_chianti")
    database = products["config"].get("hdf5_dbase_root")
    band = frame.get("band")
    band = (_SW_BAND if band is None
            else tuple(u.Quantity(edge).to_value(u.AA) for edge in band))
    path = Path(frame.get("band_lines", "run/input/band_lines.h5"))
    made_from = {"band": band, "abundance": abundance, "database": database,
                 "logT": np.asarray(products["logT_grid"]).tolist(),
                 "logN": np.asarray(products["logN_grid"]).tolist(), "eclipse": __version__}
    key = hashlib.sha256(json.dumps(made_from, sort_keys=True, default=str).encode()).hexdigest()
    if _reusable(path, key, "band's lines"):
        return read_band_lines(path)
    lines = band_contribution_functions(band, products["logT_grid"], products["logN_grid"],
                                        abundance=abundance, n_workers=n_workers,
                                        hdf5_dbase_root=database)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_band_lines(lines, path)
    _store_key(path, key)
    return lines


def _frame_one(config: dict, name: str) -> Path:
    """
    Make the full-CCD frame a run's `frame:` section describes.

    The instrument's settings come from the run's `simulation:`, `detector:`,
    `telescope:` and `filter:` sections, each with one value.

    Returns
    -------
    Path
        The frame file, `frame.output` or `run/result/<name>_frame.h5`.
    """
    from .config import AluminiumFilter, Detector_SWC, Telescope_EUVST
    from .continuum import compute_continuum_fiasco
    from .full_frame import full_ccd_frame, write_full_frame

    frame = config.get("frame")
    if not isinstance(frame, dict):
        raise ValueError("A run makes a frame from its 'frame:' section; this one has none.")
    check_config_keys(frame, FRAME_KEYS, "'frame' section")
    if str(config.get("instrument", "SWC")).upper() != "SWC":
        raise ValueError("A full-CCD frame is of the SW camera, so the instrument is SWC.")
    synthesis = (synthesis_path(config, name) if synthesises(config)
                 else Path(config.get("synthesis_file", "")))
    if not synthesis.is_file():
        raise FileNotFoundError(f"The synthesis {synthesis} is not there: a frame is made "
                                f"from a synthesis, given with 'synthesis_file', or made "
                                f"from the 'atmosphere'.")
    if "slit_position" not in frame:
        raise ValueError("The 'frame:' section needs a 'slit_position', such as 0 arcsec.")

    sections = {}
    for section in ("simulation", "detector", "telescope", "filter"):
        fixed, sweep = _parse_section(config.get(section) or {}, section)
        if sweep:
            raise ValueError(f"A frame takes one value of each setting, but "
                             f"'{section}.{next(iter(sweep))}' has several.")
        sections[section] = fixed
    simulation = sections["simulation"]
    exposure = u.Quantity(_single(frame, "exposure", "frame",
                                  _single(simulation, "expos", "simulation")))
    slit_width = frame.get("slit_width", simulation.get("slit_width", 0.2 * u.arcsec))
    n_workers = int(frame.get("n_workers", 0))
    telescope = Telescope_EUVST(filter=AluminiumFilter(**sections["filter"]),
                                **sections["telescope"])
    detector = Detector_SWC(**sections["detector"])
    options = {key: simulation[key] for key in ("psf", "spectral_psf", "noise",
                                                "psf_boundary", "slit_mask")
               if key in simulation}

    lines = _band_lines(config, frame, synthesis, n_workers)
    continuum = None
    if frame.get("continuum", False):
        wavelength = np.linspace(*_CONTINUUM_WAVELENGTHS) * u.AA
        free, two_photon = compute_continuum_fiasco(
            wavelength, lines.logT_grid, lines.logN_grid, abundance=lines.abundance,
            n_workers=n_workers)
        continuum = (wavelength, free, two_photon)
    made = full_ccd_frame(synthesis, lines, u.Quantity(frame["slit_position"]), exposure,
                          slit_width=u.Quantity(slit_width), det=detector, telescope=telescope,
                          continuum=continuum, n_workers=n_workers, **options)
    path = Path(frame.get("output", f"run/result/{name}_frame.h5"))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_full_frame(made, path)
    print(f"Wrote the full-CCD frame to {path}")
    return path


# ----------------------------------------------------------------------
# Running every step
# ----------------------------------------------------------------------
def _first_rank(step, *args):
    """Run `step` on the first MPI rank only, and wait for it on the others."""
    from .utils import _get_mpi_info

    comm, rank, size = _get_mpi_info()
    result = step(*args) if rank == 0 else None
    if size > 1:
        comm.Barrier()
    return result


#: Steps of each command. `run` makes a frame only for a run with a `frame:`
#: section.
STEPS = {"run": ("synthesise", "observe", "frame"), "synthesise": ("synthesise",),
         "observe": ("observe",), "frame": ("synthesise", "frame")}


def run_steps(config: dict, name: str, steps: tuple, source: Optional[str] = None,
              debug: bool = False) -> RunOutput:
    """
    One run's steps, from `"synthesise"`, `"observe"` and `"frame"`.

    The synthesis is made only if the run names an atmosphere, and reused
    if its file already holds it. The frame is made only if the run has a
    `frame:` section.
    """
    check(config, name)
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
    return RunOutput(name, synthesis, results, frame)


def run_all(source, command: str, *, settings=(), only: Optional[str] = None,
            name: Optional[str] = None, debug: bool = False,
            **sections) -> list[RunOutput]:
    """
    Every run of a run file, through the steps of `command`.

    `command` is one of :data:`STEPS`. `only` picks one run by its name in
    `runs:`. The runs are checked before any is started.
    """
    config, name, label = load(source, settings, name=name, **sections)
    found = runs(config, name)
    if only is not None:
        picked = [(run_name, run_config) for run_name, run_config in found
                  if run_name == f"{name}_{only}"]
        if not picked:
            names = [run_name[len(name) + 1:] for run_name, _ in found]
            raise ValueError(f"No run named {only!r}" + (
                f"; the runs are {', '.join(names)}." if "runs" in config
                else ": the run file has no 'runs:'."))
        found = picked
    for run_name, run_config in found:
        check(run_config, run_name)
        if command == "frame" and "frame" not in run_config:
            raise ValueError(f"The run {run_name} has no 'frame:' section saying what frame "
                             f"to make.")
    return [run_steps(run_config, run_name, STEPS[command], source=label, debug=debug)
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
