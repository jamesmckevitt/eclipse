"""
Main execution script for instrument response simulations.
"""

from __future__ import annotations
import argparse
import os
import sys
import warnings
from itertools import product as itertools_product
from pathlib import Path
import yaml
import astropy.units as u
import gzip
import h5py

from .config import AluminiumFilter, Detector_SWC, Detector_EIS, Telescope_EUVST, Telescope_EIS, Simulation, check_pinhole_lists, _load_throughput_table
from .data_processing import (_whole_pixels, load_atmosphere, rebin_atmosphere,
                              create_uniform_intensity_cube, pad_spectral_axis, rebin_spectra)
from .raster import AtmosphereSeries, RasterSynthesiser, SynthesisRaster, SynthesisSeries
from .atmosphere import _is_pickle
from .synthesis_file import (is_synthesis_file, read_synthesis, read_synthesis_products,
                             synthesis_line_names)
from .fitting import FitConfig, FitComponent, ground_truth_summary
from .results_file import _to_json, save_results
from .monte_carlo import expected_dn_uncertainty, monte_carlo
from .radiometric import spectral_optics_fwhm, spectral_psf_margin
from .utils import (
    parse_yaml_input, ensure_list, set_debug_mode, debug_break, debug_on_error,
    deduplicate_list, get_git_commit_id, _get_software_version,
    _parse_section, _params_to_key, _extract_config_params, _SECTION_LIST_FIELDS,
    rebin_slit_offchip, check_config_keys, load_yaml_config,
)
import dataclasses
import numpy as np


# Every key main() reads. Anything else in the file is not read at all, so it
# is rejected rather than ignored.
_TOP_LEVEL_KEYS = {
    "instrument", "n_iter", "ncpu",
    "uniform_intensity", "rest_wavelength", "thermal_width",
    "synthesis_file", "reference_line",
    "atmosphere_series", "synthesis_series", "synthesis", "raster",
    "pinhole_sizes", "pinhole_positions", "pinhole_positions_spectral",
    "offchip_bin_slit", "fit_signals",
    "simulation", "detector", "telescope", "filter", "fitting",
}

# What a time series of atmosphere files is synthesised with, and how a time
# series, of atmosphere or of synthesis files, is observed. The synthesis keys
# are those of synthesise-spectra; the raster keys are the observing plan of
# euvst_response.raster.RasterPlan.
_SYNTHESIS_KEYS = {"lines", "abundance", "vel_res", "vel_lim", "crop_y", "crop_z",
                   "precision", "mass_per_electron", "hdf5_dbase_root", "n_workers",
                   "goft_temperature_chunk"}
_RASTER_KEYS = {"start", "steps", "step", "repeats", "cadence", "centre", "direction"}

# Under MPI, a copy of the stderr a rank other than the first started with,
# before its output is silenced.
_STARTING_STDERR = None


def _cpu_list(text: str) -> list:
    """The CPUs of a Linux CPU list such as ``0-3,8,10-11``."""
    cpus = []
    for part in text.strip().split(","):
        if part:
            first, _, last = part.partition("-")
            cpus.extend(range(int(first), int(last or first) + 1))
    return cpus


def _cpuset():
    """The CPUs this process's cgroup cpuset allows, or None where there is none to read."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, path = line.split(":", 2)
            if hierarchy == "0" or "cpuset" in controllers.split(","):
                root = Path("/sys/fs/cgroup") if hierarchy == "0" else Path("/sys/fs/cgroup/cpuset")
                for name in ("cpuset.cpus.effective", "cpuset.cpus"):
                    listed = root / path.lstrip("/") / name
                    if listed.is_file() and listed.read_text().strip():
                        return _cpu_list(listed.read_text())
    except (OSError, ValueError):
        pass
    return None


def _allowed_cpus(environ, cpuset, affinity) -> list:
    """
    The CPUs this job step has on its node, for its ranks to share.

    SLURM confines a step to its CPUs with a cpuset, which may be wider than
    the *affinity* an MPI library narrowed a rank to, so the *cpuset* is
    taken where it is the step's: no more CPUs than SLURM_CPUS_ON_NODE says
    the job has there, and holding the affinity. Where SLURM does not
    confine steps, the cpuset is the whole node, and taking CPUs from it
    could land on another job's, so the affinity is kept, as it is where
    there is no cpuset or SLURM does not say.
    """
    on_node = environ.get("SLURM_CPUS_ON_NODE")
    if (cpuset and on_node is not None and on_node.isdigit()
            and len(cpuset) <= int(on_node) and set(affinity) <= set(cpuset)):
        return list(cpuset)
    return list(affinity)


def _rank_cpus(environ, allowed: list) -> list:
    """
    This MPI rank's CPUs: SLURM_CPUS_PER_TASK of *allowed*, taken in turn by its local task id.

    All of *allowed* where there are not enough for every task on the node
    to have its own, and none, to leave the affinity alone, where SLURM does
    not say how many a task has.
    """
    per_task, local = environ.get("SLURM_CPUS_PER_TASK"), environ.get("SLURM_LOCALID")
    if per_task is None or local is None:
        return []
    per_task, local = int(per_task), int(local)
    share = allowed[local * per_task:(local + 1) * per_task]
    return share if len(share) == per_task else list(allowed)


# The Simulation dataclass has more fields than this, but main() builds its
# Simulation objects itself and only takes these from the section. The rest
# (instrument, n_iter, ncpu, and the pinhole lists) are top-level keys, so
# writing one here would have been parsed and then dropped.
_SIMULATION_KEYS = {"slit_width", "expos", "vis_sl", "psf", "psf_boundary",
                    "spectral_psf", "noise", "enable_pinholes"}

_FITTING_KEYS = {"components", "primary_component",
                 "constrain_positive_intensity", "backend", "max_iter",
                 "bessel_correction", "save_iterations", "weighted"}
_FITTING_COMPONENT_KEYS = {"wavelength", "tie_center", "tie_width",
                           "amplitude_greater_than", "name"}


def _dataclass_keys(cls) -> set:
    """Constructor argument names of a config dataclass."""
    return {f.name for f in dataclasses.fields(cls) if f.init}


def _type_name(value) -> str:
    """Type of a config value for an error message; an empty YAML entry is None."""
    return "nothing" if value is None else type(value).__name__


def _require_mapping(value, what: str) -> None:
    """Raise unless a config section or fitting component is a mapping."""
    if not isinstance(value, dict):
        raise ValueError(
            f"{what} must be a mapping of parameter names to values, "
            f"got {_type_name(value)}."
        )


def _validate_config_keys(config: dict, instrument: str) -> None:
    """
    Reject any config key ECLIPSE does not read.

    Checked before anything is loaded or computed, so a config written against
    an older layout fails immediately rather than after an atmosphere load.

    Parameters
    ----------
    config : dict
        The whole parsed YAML config.
    instrument : str
        ``"SWC"`` or ``"EIS"``; the two have different detector and telescope
        parameters.

    Raises
    ------
    ValueError
        If the instrument is not supported, a key is not read, or a section
        that is present is not a mapping.
    """
    # Checked first because the valid keys depend on it, and main() treats
    # anything other than SWC as EIS.
    if instrument not in ("SWC", "EIS"):
        raise ValueError(
            f"Unknown instrument '{instrument}'. Supported values: 'SWC', 'EIS'."
        )

    det_keys = _dataclass_keys(Detector_EIS if instrument == "EIS"
                               else Detector_SWC)
    tel_keys = _dataclass_keys(Telescope_EIS if instrument == "EIS"
                               else Telescope_EUVST)
    fil_keys = _dataclass_keys(AluminiumFilter)

    # 'filter' is a Telescope_EUVST field, but main() builds the filter from
    # the top-level 'filter:' section and drops whatever is here, so it must
    # not look settable.
    tel_keys.discard("filter")
    if instrument == "EIS":
        # Warned about and ignored explicitly further down, so not a surprise.
        tel_keys.add("microroughness_sigma")

    sections = {
        "simulation": _SIMULATION_KEYS,
        "detector": det_keys,
        "telescope": tel_keys,
        "filter": fil_keys,
    }
    # Where else a name could have been meant, used for the suggestions. The
    # top level is included so that a section key written at the wrong depth
    # is named as such.
    elsewhere = {"": _TOP_LEVEL_KEYS, **sections}

    check_config_keys(config, _TOP_LEVEL_KEYS, "top-level", sections)

    # Sections are checked by presence, not value, so that a heading left
    # empty (which parses to None) gets a message here rather than an
    # AttributeError further on.
    for name, allowed in sections.items():
        if name not in config:
            continue
        _require_mapping(config[name], f"The '{name}:' section")
        others = {k: v for k, v in elsewhere.items() if k != name}
        check_config_keys(config[name], allowed, f"'{name}' section", others)

    for name, allowed in (("synthesis", _SYNTHESIS_KEYS), ("raster", _RASTER_KEYS)):
        if name in config:
            _require_mapping(config[name], f"The '{name}:' section")
            check_config_keys(config[name], allowed, f"'{name}' section")

    if "fitting" in config:
        fitting = config["fitting"]
        _require_mapping(fitting, "The 'fitting:' section")
        check_config_keys(fitting, _FITTING_KEYS, "'fitting' section")
        if "components" in fitting:
            components = fitting["components"]
            if not isinstance(components, list):
                raise ValueError(
                    f"'fitting.components' must be a list with one entry per "
                    f"Gaussian component, got {_type_name(components)}."
                )
            for idx, component in enumerate(components):
                where = f"'fitting.components[{idx}]'"
                _require_mapping(component, where)
                check_config_keys(component, _FITTING_COMPONENT_KEYS, where)


def _quantity_or_none(section: dict, key: str, unit, what: str):
    """A quantity of the kind *unit* from a config section, or None if absent."""
    if key not in section or section[key] is None:
        return None
    value = parse_yaml_input(section[key])
    if not isinstance(value, u.Quantity) or not value.unit.is_equivalent(unit):
        raise ValueError(f"'{what}.{key}' must be {unit.physical_type} with units, "
                         f"e.g. '{section[key]}' is not; got {section[key]!r}.")
    return value


def _range_or_none(section: dict, key: str, what: str):
    """A ``[low, high]`` range of lengths from a config section, or None."""
    if key not in section or section[key] is None:
        return None
    bounds = section[key]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"'{what}.{key}' must be a list of two lengths with units, "
                         f"e.g. ['0 Mm', '20 Mm'], got {bounds!r}.")
    low, high = (parse_yaml_input(b) for b in bounds)
    for value in (low, high):
        if not isinstance(value, u.Quantity) or not value.unit.is_equivalent(u.Mm):
            raise ValueError(f"'{what}.{key}' must hold lengths with units, got {bounds!r}.")
    return (low, high)


def _parse_synthesis_settings(config: dict):
    """The 'synthesis:' section as the settings a time series is synthesised with."""
    from .raster import SynthesisSettings

    section = config.get("synthesis")
    if section is None:
        raise ValueError("An 'atmosphere_series' needs a 'synthesis:' section naming at "
                         "least the 'lines' to synthesise.")
    lines = section.get("lines")
    if isinstance(lines, str):
        lines = [lines]
    if not isinstance(lines, list) or not lines or not all(isinstance(l, str) for l in lines):
        raise ValueError("'synthesis.lines' must be a list of line names such as "
                         "['Fe12_195.1190'].")
    settings = {"lines": tuple(lines)}
    if "abundance" in section:
        settings["abundance"] = str(section["abundance"])
    for key in ("vel_res", "vel_lim"):
        value = _quantity_or_none(section, key, u.km / u.s, "synthesis")
        if value is not None:
            settings[key] = value
    for key in ("crop_y", "crop_z"):
        settings[key] = _range_or_none(section, key, "synthesis")
    if "precision" in section:
        precision = str(section["precision"])
        if precision not in ("float32", "float64"):
            raise ValueError(f"'synthesis.precision' must be 'float32' or 'float64', "
                             f"got {precision!r}.")
        settings["precision"] = np.float32 if precision == "float32" else np.float64
    if section.get("mass_per_electron") is not None:
        value = section["mass_per_electron"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'synthesis.mass_per_electron' must be a number of atomic mass "
                             f"units, such as 1.16, got {value!r}.")
        settings["mass_per_electron"] = float(value)
    if section.get("hdf5_dbase_root") is not None:
        settings["hdf5_dbase_root"] = str(section["hdf5_dbase_root"])
    # Whole numbers, which int() would otherwise make of a fraction.
    for key, least in (("n_workers", 0), ("goft_temperature_chunk", 1)):
        value = section.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            raise ValueError(f"'synthesis.{key}' must be a whole number of {least} or more, "
                             f"got {value!r}.")
        settings[key] = value
    return SynthesisSettings(**settings)


def _parse_raster_plan(config: dict):
    """The 'raster:' section as an observing plan."""
    from .raster import RasterPlan

    section = config.get("raster")
    if section is None:
        raise ValueError("A time series needs a 'raster:' section saying at least when "
                         "the observation starts ('start').")
    start = _quantity_or_none(section, "start", u.s, "raster")
    if start is None:
        raise ValueError("'raster.start' is needed: the simulation time at which the "
                         "first exposure starts, with units, e.g. '3850 s'.")
    plan = {"start": start}
    for key in ("steps", "repeats"):
        if section.get(key) is not None:
            plan[key] = section[key]
    plan["step"] = _quantity_or_none(section, "step", u.arcsec, "raster")
    plan["cadence"] = _quantity_or_none(section, "cadence", u.s, "raster")
    plan["centre"] = _quantity_or_none(section, "centre", u.Mm, "raster")
    if section.get("direction") is not None:
        plan["direction"] = section["direction"]
    return RasterPlan(**plan)


def _series_paths(config: dict, key: str) -> list:
    """The files of 'atmosphere_series' or 'synthesis_series': a list of paths, or a glob pattern."""
    import glob

    value = config[key]
    if isinstance(value, str):
        paths = sorted(glob.glob(value))
        if not paths:
            raise FileNotFoundError(f"'{key}' matches no file: {value}")
        return paths
    if isinstance(value, list) and value and all(isinstance(p, str) for p in value):
        missing = [p for p in value if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"'{key}' names files that do not exist: {missing}")
        return list(value)
    kind = "atmosphere" if key == "atmosphere_series" else "synthesis"
    raise ValueError(f"'{key}' must be a glob pattern or a list of {kind} files, "
                     f"e.g. './data/bifrost/*.h5'.")


# The line observed when the configuration names none and the synthesis holds
# more than one.
DEFAULT_REFERENCE_LINE = "Fe12_195.1190"

# Where synthesise-spectra writes by default, and where older versions did.
DEFAULT_SYNTHESIS_FILE = "./run/input/synthesised_spectra.h5"
LEGACY_SYNTHESIS_FILE = "./run/input/synthesised_spectra.pkl"


def _reference_line(config: dict, synthesis_path) -> str:
    """
    'reference_line', or the line a synthesis file is observed in without one.

    Left empty, it is the file's first line, as older versions took it. Not
    given, it is the file's only line, or else the default line, which the
    file must then hold.
    """
    if config.get("reference_line") is not None:
        return config["reference_line"]
    names = synthesis_line_names(synthesis_path)
    if not names:
        # Refused when the file is read, as holding no lines.
        return DEFAULT_REFERENCE_LINE
    if "reference_line" in config or len(names) == 1:
        return names[0]
    if DEFAULT_REFERENCE_LINE not in names:
        raise ValueError(f"{synthesis_path} holds the lines {names} and no 'reference_line' "
                         f"says which to observe; the default, {DEFAULT_REFERENCE_LINE}, is "
                         f"not among them.")
    return DEFAULT_REFERENCE_LINE


def _parse_pinhole_config(config: dict) -> tuple:
    """
    Read the pinhole lists from a YAML config and validate them.

    The lists are paired, one entry per pinhole, so any of them without the
    sizes describes no pinhole at all and is always a mistake.

    Parameters
    ----------
    config : dict
        The whole parsed YAML config.

    Returns
    -------
    tuple
        ``(pinhole_sizes, pinhole_positions, pinhole_positions_spectral)``.
    """
    pinhole_sizes = []
    pinhole_positions = []
    if "pinhole_sizes" in config:
        pinhole_sizes = ensure_list(parse_yaml_input(config["pinhole_sizes"]))
    if "pinhole_positions" in config:
        pinhole_positions = ensure_list(config["pinhole_positions"])

    # Optional spectral positions, one per pinhole, as a fraction (0.0-1.0) of
    # the detector's spectral width.  Omit to project every pinhole to the
    # centre of the spectral window, which is what ECLIPSE always did.
    pinhole_positions_spectral = []
    if "pinhole_positions_spectral" in config:
        pinhole_positions_spectral = ensure_list(
            config["pinhole_positions_spectral"])

    pinhole_positions, pinhole_positions_spectral = check_pinhole_lists(
        pinhole_sizes, pinhole_positions, pinhole_positions_spectral)

    return pinhole_sizes, pinhole_positions, pinhole_positions_spectral


def _parse_fitting_config(config: dict) -> FitConfig | None:
    """
    Build the fit configuration from the ``fitting:`` block of a YAML config.

    A block with no components configures the single-Gaussian fit.
    FitConfig itself checks the values, including that there are either no
    components or at least two.

    Parameters
    ----------
    config : dict
        The whole parsed YAML config.

    Returns
    -------
    FitConfig or None
        None when the config has no ``fitting:`` block.
    """
    fitting_cfg = config.get("fitting", None)
    if fitting_cfg is None:
        return None

    components = []
    for idx, comp_dict in enumerate(fitting_cfg.get("components", [])):
        if "wavelength" not in comp_dict:
            raise ValueError(
                f"fitting.components[{idx}] is missing required field "
                f"'wavelength' (rest wavelength of this Gaussian component, "
                f"e.g. 'wavelength: 195.119 angstrom')."
            )
        components.append(FitComponent(
            wavelength=parse_yaml_input(comp_dict["wavelength"]),
            tie_center=comp_dict.get("tie_center", None),
            tie_width=comp_dict.get("tie_width", None),
            amplitude_greater_than=comp_dict.get("amplitude_greater_than", None),
            name=comp_dict.get("name", None),
        ))

    return FitConfig(
        components=components,
        primary_component=fitting_cfg.get("primary_component", 0),
        constrain_positive_intensity=fitting_cfg.get(
            "constrain_positive_intensity", False),
        backend=fitting_cfg.get("backend", None),
        max_iter=fitting_cfg.get("max_iter", FitConfig.max_iter),
        bessel_correction=fitting_cfg.get("bessel_correction", False),
        save_iterations=fitting_cfg.get("save_iterations", False),
        weighted=fitting_cfg.get("weighted", True),
    )


@debug_on_error
def main() -> None:
    """
    Run the simulation that the `eclipse` command runs, with its arguments from ``sys.argv``.

    The command checks its arguments and handles errors before calling this.
    It reads the configuration file named by ``--config``, runs every
    combination of its settings, and writes the results to
    ``run/result/<config name>.h5``.
    """

    # Suppress noisy astropy WCS warnings
    for msg in [
        "target cannot be converted to ICRS",
        "target cannot be converted to ICRS, so will not be set on SpectralCoord",
        "No observer defined on WCS, SpectralCoord will be converted without any velocity frame change",
    ]:
        warnings.filterwarnings("ignore", message=msg, category=UserWarning)
    warnings.filterwarnings("ignore", module="astropy.wcs.wcsapi.fitswcs", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="YAML config file", required=True)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    set_debug_mode(args.debug)
    if args.debug:
        print("Debug mode enabled - will break to IPython on errors")

    # MPI auto-detection: when launched via srun/mpirun with multiple tasks,
    # Monte Carlo iterations are distributed across ranks automatically.
    from .utils import _get_mpi_info
    _comm, _mpi_rank, _mpi_size = _get_mpi_info()
    if _mpi_size > 1:
        # Intel MPI pins each rank to cores, breaking joblib/loky. Widen it
        # to this rank's share of the CPUs the job step has on its node,
        # rather than to CPUs 0 to n, which every rank shared and which, on a
        # node shared with another job, need not be the job's at all.
        _cpus = _rank_cpus(os.environ, _allowed_cpus(os.environ, _cpuset(),
                                                     sorted(os.sched_getaffinity(0))))
        if _cpus:
            os.sched_setaffinity(0, _cpus)

        if _mpi_rank == 0:
            print(f"MPI distributed mode: {_mpi_size} processes "
                  f"(MC iterations will be split across ranks)")
        else:
            # Silence stdout and stderr on non-root ranks to avoid duplicated
            # output. Redirect at the OS file-descriptor level, not
            # just the Python objects, because SLURM captures fd 1/2 directly
            # and tqdm can bypass the Python sys.stderr object.
            # The stderr it started with is kept, for an error that has to be
            # said before the run is aborted.
            global _STARTING_STDERR
            _STARTING_STDERR = os.dup(2)
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull_fd, 1)  # redirect fd 1 (stdout)
            os.dup2(_devnull_fd, 2)  # redirect fd 2 (stderr)
            os.close(_devnull_fd)
            # Wrap the already-redirected fds rather than opening new devnull
            # handles; closefd=False prevents a ResourceWarning when these
            # objects are later garbage-collected.
            sys.stdout = open(1, "w", closefd=False)
            sys.stderr = open(2, "w", closefd=False)

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_yaml_config(config_path.read_text())

    # An empty file parses to None, which would otherwise fail later with an
    # AttributeError rather than saying the config is empty.
    if config is None:
        raise ValueError(f"Config file is empty: {args.config}")
    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must be a mapping of keys to values, got "
            f"{type(config).__name__}: {args.config}"
        )
    # The config is saved with the results, so a value the results file
    # cannot hold, such as a YAML alias inside itself, is refused now rather
    # than once the run is done.
    try:
        _to_json(config)
    except (TypeError, RecursionError) as error:
        reason = "a value holds itself" if isinstance(error, RecursionError) else error
        raise ValueError(f"{args.config} cannot be saved with the results: {reason}") from None

    # Top-level scalar settings. A 'simulation:' that is not a mapping is left
    # for _validate_config_keys to report.
    simulation_section = config.get("simulation")
    if isinstance(simulation_section, dict) and "instrument" in simulation_section:
        raise ValueError(
            "Set the instrument with the top-level 'instrument:' key, not "
            "inside the 'simulation:' section. Both Simulation objects are "
            "built from the top-level value, so one written here would be "
            "read and then ignored."
        )
    instrument = config.get("instrument", "SWC")
    if not isinstance(instrument, str):
        raise ValueError(f"'instrument' must be SWC or EIS, got {instrument!r}.")
    instrument = instrument.upper()

    # Before anything is loaded, so that a config written against an older
    # layout fails here rather than after the atmosphere load.
    _validate_config_keys(config, instrument)

    n_iter = config.get("n_iter", 25)
    # Not left to the run: 0 wrote empty results and said it had finished,
    # and '1e3', a string to PyYAML, failed after the ground truth was fitted.
    if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter < 1:
        raise ValueError(f"'n_iter' must be a whole number of iterations, 1 or more, got "
                         f"{n_iter!r}.")
    ncpu = config.get("ncpu", -1)
    # joblib and reproject both take it, and agree only on these. 0, which
    # means every CPU to the synthesis's n_workers, failed at the first fit.
    if isinstance(ncpu, bool) or not isinstance(ncpu, int) or (ncpu < 1 and ncpu != -1):
        raise ValueError(f"'ncpu' must be a whole number of CPUs, 1 or more, or -1 for all "
                         f"of them, got {ncpu!r}.")

    # In MPI mode, if a ncpu value specified, cap it to
    # the CPUs available to this rank so joblib doesn't oversubscribe.
    # Leave ncpu=-1 alone - joblib handles it natively via the OS affinity
    # mask (which I_MPI_PIN_DOMAIN=auto sets correctly).
    if _mpi_size > 1:
        if ncpu != -1:
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm_cpus is not None:
                available_cpus = int(slurm_cpus)
            else:
                available_cpus = os.cpu_count() or 1
            if ncpu > available_cpus:
                ncpu = available_cpus
        if _mpi_rank == 0:
            print(f"MPI: ncpu={ncpu} per rank")

    # Simulation mode. A time series is either atmosphere files, synthesised
    # in this run, or synthesis files, synthesised beforehand.
    uniform_intensity_mode = "uniform_intensity" in config
    if uniform_intensity_mode and "synthesis_file" in config:
        warnings.warn("'synthesis_file' is ignored: 'uniform_intensity' says what is observed.",
                      UserWarning, stacklevel=2)
    # The line of a uniform intensity; anything else observes the spectra it is given.
    for key in ("rest_wavelength", "thermal_width"):
        if key in config and not uniform_intensity_mode:
            warnings.warn(f"'{key}' is ignored: it describes the line of a 'uniform_intensity', "
                          f"and this run observes the spectra it is given.",
                          UserWarning, stacklevel=2)
    atmosphere_series_mode = "atmosphere_series" in config
    synthesis_series_mode = "synthesis_series" in config
    raster_mode = atmosphere_series_mode or synthesis_series_mode
    given = [key for key in ("uniform_intensity", "synthesis_file", "atmosphere_series",
                             "synthesis_series") if key in config]
    if raster_mode and len(given) > 1:
        raise ValueError(f"Give only one of {given}: each says what is observed.")
    if not atmosphere_series_mode and "synthesis" in config:
        raise ValueError("The 'synthesis:' section belongs to an 'atmosphere_series' run "
                         "and would not be read here.")
    if not raster_mode and "raster" in config:
        raise ValueError("The 'raster:' section belongs to an 'atmosphere_series' or "
                         "'synthesis_series' run and would not be read here.")

    if atmosphere_series_mode:
        series_paths = _series_paths(config, "atmosphere_series")
        synthesis_settings = _parse_synthesis_settings(config)
        raster_plan = _parse_raster_plan(config)
        # Left empty, as not given: the first line.
        reference_line = config.get("reference_line") or synthesis_settings.lines[0]
        if reference_line not in synthesis_settings.lines:
            raise ValueError(f"'reference_line' {reference_line!r} is not one of "
                             f"'synthesis.lines' {list(synthesis_settings.lines)}.")
        print("TIME SERIES MODE")
        print(f"  Atmosphere files: {len(series_paths)}")
        print(f"  Lines: {list(synthesis_settings.lines)}, reference {reference_line}")
        print(f"  Raster: {raster_plan.steps} step(s), {raster_plan.repeats} repeat(s), "
              f"starting at {raster_plan.start}")
    elif synthesis_series_mode:
        series_paths = _series_paths(config, "synthesis_series")
        synthesis_settings = None
        raster_plan = _parse_raster_plan(config)
        reference_line = _reference_line(config, series_paths[0])
        print("TIME SERIES MODE")
        print(f"  Synthesis files: {len(series_paths)}")
        print(f"  Reference line: {reference_line}")
        print(f"  Raster: {raster_plan.steps} step(s), {raster_plan.repeats} repeat(s), "
              f"starting at {raster_plan.start}")
    elif uniform_intensity_mode:
        uniform_intensity = parse_yaml_input(config["uniform_intensity"])
        if not hasattr(uniform_intensity, "unit"):
            raise ValueError(
                "uniform_intensity must include units, e.g. '5000 erg / (s cm2 sr)'"
            )
        uniform_rest_wavelength = parse_yaml_input(config.get("rest_wavelength", "195.119 AA"))
        uniform_thermal_width = parse_yaml_input(config.get("thermal_width", "20 km/s"))
        # Each a positive amount of its own kind: a negative width, or one of
        # zero, gave a line the fit made nonsense of, saved as a result.
        for key, value, kind, example in (
                ("uniform_intensity", uniform_intensity, u.erg / (u.s * u.cm**2 * u.sr),
                 "5000 erg / (s cm2 sr)"),
                ("rest_wavelength", uniform_rest_wavelength, u.AA, "195.119 AA"),
                ("thermal_width", uniform_thermal_width, u.km / u.s, "20 km/s")):
            if not (isinstance(value, u.Quantity) and value.unit.is_equivalent(kind)
                    and np.isfinite(value.value) and value.value > 0):
                raise ValueError(f"'{key}' must be a positive {kind.physical_type}, such as "
                                 f"'{example}', got {config.get(key)!r}.")
        print("UNIFORM INTENSITY MODE")
        print(f"  Intensity: {uniform_intensity}")
        print(f"  Rest wavelength: {uniform_rest_wavelength}")
        print(f"  Thermal width (1-sigma): {uniform_thermal_width}")
    else:
        synthesis_file = config.get("synthesis_file", DEFAULT_SYNTHESIS_FILE)
        # A run set up for an older version, whose synthesis wrote the pickle
        # and whose configuration leaves the file to the default, still finds
        # it.
        if "synthesis_file" not in config and Path(LEGACY_SYNTHESIS_FILE).is_file():
            if not Path(synthesis_file).is_file():
                synthesis_file = LEGACY_SYNTHESIS_FILE
            else:
                warnings.warn(f"Both {DEFAULT_SYNTHESIS_FILE} and {LEGACY_SYNTHESIS_FILE} exist "
                              f"and 'synthesis_file' names neither; observing "
                              f"{DEFAULT_SYNTHESIS_FILE}, where the synthesis now writes. "
                              f"Name the one to observe with 'synthesis_file'.",
                              UserWarning, stacklevel=2)
        if not Path(synthesis_file).is_file():
            raise FileNotFoundError(
                f"Synthesis file not found: {synthesis_file}. "
                "Please check the 'synthesis_file' path in your config file."
            )
        synthesis_is_hdf5 = is_synthesis_file(synthesis_file)
        if synthesis_is_hdf5:
            reference_line = _reference_line(config, synthesis_file)
        elif not _is_pickle(Path(synthesis_file)):
            raise ValueError(
                f"{synthesis_file} is neither a synthesis file, which is HDF5, nor a synthesis "
                f"pickle as older versions of ECLIPSE wrote. A synthesis from another code is "
                f"written as a synthesis file with euvst_response.write_synthesis.")
        else:
            warnings.warn(
                f"{synthesis_file} is a synthesis pickle, as older versions of ECLIPSE "
                f"wrote them. Pickles are deprecated and will not be read in a future "
                f"release: re-run the synthesis, or convert the file with "
                f"euvst_response.convert_synthesis_pickle.", FutureWarning, stacklevel=2)
            reference_line = config.get("reference_line", DEFAULT_REFERENCE_LINE)

    # Pinhole config (fixed paired lists, not swept)
    (pinhole_sizes, pinhole_positions,
     pinhole_positions_spectral) = _parse_pinhole_config(config)

    # Parse config sections
    sim_fixed, sim_sweep = _parse_section(config.get("simulation", {}), "simulation")
    det_fixed, det_sweep = _parse_section(config.get("detector", {}), "detector")
    tel_fixed, tel_sweep = _parse_section(config.get("telescope", {}), "telescope")
    fil_fixed, fil_sweep = _parse_section(config.get("filter", {}), "filter")

    # A run models frames taken with the shutter closed outside the exposure.
    # Without one, light falls on the CCDs while a frame is cleared and read,
    # and every line in the band smears into the window; the run synthesises
    # only the window, so it cannot model that. Refused here rather than
    # taken and ignored.
    if "shutter" in det_sweep or det_fixed.get("shutter", True) is False:
        raise ValueError(
            "detector.shutter must be true for an instrument run, which models frames taken "
            "with the shutter closed outside the exposure. Frames taken without it are "
            "modelled by euvst_response.readout, which takes the detector's settings.")
    # Nor does a run model the read-out, so the charge left behind in its
    # transfers would be taken and ignored; refused too.
    for name in ("cte_parallel", "cte_serial"):
        if name in det_sweep or det_fixed.get(name, 1.0) != 1.0:
            raise ValueError(
                f"detector.{name} must be 1 for an instrument run, which does not model the "
                f"read-out. Charge left behind as a frame is read is modelled by "
                f"euvst_response.readout, which takes the detector's settings.")

    # The wavelength grids are cached by slit width and detector sampling, and
    # are sized for the spectral PSF of the slit, which also depends on the
    # slit psf_params was measured with. That is one fact about the
    # telescope, as psf_params is, so it takes one value.
    if "psf_slit_width" in tel_sweep:
        raise ValueError(
            "telescope.psf_slit_width is the slit psf_params was measured with, "
            "so like psf_params it takes a single value and cannot be swept.")

    # Instrument-specific validation
    if instrument == "EIS":
        if config.get("filter"):
            warnings.warn(
                "The 'filter:' section describes SWC's filter and is ignored for EIS, whose "
                "filters are folded into its effective area tables.",
                UserWarning,
            )
            fil_fixed, fil_sweep = {}, {}
        vis_sl = [sim_fixed["vis_sl"]] if "vis_sl" in sim_fixed else sim_sweep.get("vis_sl", [])
        if any(np.any(u.Quantity(value).value > 0) for value in vis_sl):
            warnings.warn(
                "EIS's filters are not modelled for visible light, so vis_sl reaches its CCD "
                "as given, where for SWC it is the flux before the filter. Give EIS the flux "
                "after its filters.",
                UserWarning,
            )
        for key in ("microroughness_sigma",):
            if key in tel_fixed or key in tel_sweep:
                warnings.warn(
                    f"EIS does not support '{key}'. This parameter will be ignored.",
                    UserWarning,
                )
                tel_fixed.pop(key, None)
                tel_sweep.pop(key, None)
        # Any pinhole key at all, not just the sizes: a config carrying
        # positions alone was accepted here and then silently ignored.
        if (pinhole_sizes or pinhole_positions or pinhole_positions_spectral
                or sim_fixed.get("enable_pinholes")
                or any(sim_sweep.get("enable_pinholes", []))):
            raise ValueError(
                "Pinhole effects are not supported for EIS. Remove "
                "enable_pinholes, pinhole_sizes, pinhole_positions and "
                "pinhole_positions_spectral, or run this config against SWC."
            )

    # Parse fitting configuration
    fit_config = _parse_fitting_config(config)
    if fit_config is not None:
        if fit_config.backend == "mpfit":
            backend_label = "mpfit (forced)"
        elif fit_config.backend == "scipy":
            backend_label = "scipy (forced)"
        else:
            backend_label = "scipy (auto)"
        if fit_config.is_single:
            print(f"Single-Gaussian fitting: max_iter={fit_config.max_iter}, "
                  f"backend={backend_label}")
        else:
            print(f"Multi-component fitting enabled: {fit_config.n_components} components "
                  f"(primary={fit_config.primary_component}, "
                  f"{fit_config.n_full_params} params, "
                  f"backend={backend_label})")

    # Parse off-chip slit binning (ground-based spatial binning along the slit)
    offchip_bin_slits = ensure_list(config.get("offchip_bin_slit", [1]))

    # Parse which signals to fit (default: both DN and photon)
    fit_signals = config.get("fit_signals", "both")
    if fit_signals not in ("both", "dn", "photon"):
        raise ValueError(
            f"Unknown fit_signals value '{fit_signals}'. "
            f"Supported values: 'both', 'dn', 'photon'."
        )
    if fit_signals != "both":
        skipped = "photon" if fit_signals == "dn" else "dn"
        print(f"Fitting only '{fit_signals}' signal (skipping '{skipped}')")
    # Unless the fitting block says not, the DN fits are weighted by each
    # pixel's uncertainty, and the photon fits are those of greatest Poisson
    # likelihood. The ground truth is weighted as the DN fits are; with only
    # the photons fitted it is not, as no DN uncertainty is worked out.
    weighted_fits = fit_config is None or fit_config.weighted
    weighted_truth = weighted_fits and fit_signals != "photon"
    print("DN fits weighted by each pixel's uncertainty, photon fits by their Poisson "
          "likelihood" if weighted_fits else "Fits unweighted")

    def _fit_ground_truth(cube_binned, offchip_bin_slit, uncertainty=None):
        """The fit to the spectra with no noise, saying where it failed."""
        print(f"Fitting ground truth cube (offchip_bin_slit={offchip_bin_slit})...")
        truth = ground_truth_summary(cube_binned, fit_config, n_jobs=ncpu,
                                     uncertainty=uncertainty)
        truth_failed = truth["failed"]
        if truth_failed.any():
            print(f"  Ground truth fit failed in {np.count_nonzero(truth_failed)} "
                  f"of {truth_failed.size} pixels; their true velocity and "
                  f"width are NaN")
        return truth
    # ensure_list wraps scalars in a list; values are plain ints (no units),
    # which int() would have made of a fraction or a numeric string.
    if not offchip_bin_slits or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                                    for v in offchip_bin_slits):
        raise ValueError(f"'offchip_bin_slit' must be whole numbers of rows, 1 or more, got "
                         f"{config.get('offchip_bin_slit')!r}.")
    offchip_bin_slits = deduplicate_list(offchip_bin_slits, "offchip_bin_slit")
    if any(b > 1 for b in offchip_bin_slits):
        print(f"Off-chip slit binning values: {offchip_bin_slits} "
              f"(ground-based, all noise per pixel before summation)")

    # Apply defaults for required params not specified anywhere
    _sim_defaults = {
        "slit_width": 0.2 * u.arcsec,
        "expos": 1.0 * u.s,
        "vis_sl": 0.0 * u.photon / (u.s * u.cm**2),
        "psf": False,
        "psf_boundary": "replicate",
        "spectral_psf": "quadrature",
        "noise": True,
        "enable_pinholes": False,
    }
    _det_defaults = {
        "ccd_temperature": -60 * u.Celsius,
    }
    for attr, default in _sim_defaults.items():
        if attr not in sim_fixed and attr not in sim_sweep:
            sim_fixed[attr] = default
    for attr, default in _det_defaults.items():
        if attr not in det_fixed and attr not in det_sweep:
            det_fixed[attr] = default

    def _swept(*attrs):
        """Each of the simulation values *attrs*, as the list the sweep takes it through."""
        return [list(sim_sweep[attr]) if attr in sim_sweep else [sim_fixed[attr]]
                for attr in attrs]

    # Warnings
    psf_vals = list(sim_sweep.get("psf", [sim_fixed.get("psf", False)]))
    if any(psf_vals):
        if instrument == "SWC":
            warnings.warn(
                "The SWC PSF is the modelled PSF including simulations and some microroughness "
                "measurements. Final PSF will be measured before launch.",
                UserWarning,
            )
        elif instrument == "EIS":
            warnings.warn(
                "The EIS PSF is not well understood. We use a symmetrical Gaussian kernel with "
                "a FWHM of 3 pixels from Ugarte-Urra (2016) EIS Software Note 2.",
                UserWarning,
            )

    noise_vals = list(sim_sweep.get("noise", [sim_fixed.get("noise", True)]))
    if not all(noise_vals) and n_iter > 1:
        warnings.warn(
            f"noise is False, so every Monte Carlo iteration is identical. "
            f"n_iter is {n_iter}; set it to 1 to avoid repeating the same "
            f"deterministic run.",
            UserWarning,
        )

    enable_ph_vals = list(sim_sweep.get("enable_pinholes", [sim_fixed.get("enable_pinholes", False)]))
    if any(enable_ph_vals):
        warnings.warn(
            "Pinhole effects are only intended for use by the instrument team. "
            "Please contact MSSL for more information.",
            UserWarning,
        )
        if not pinhole_sizes:
            warnings.warn(
                "enable_pinholes is True but no pinhole_sizes specified. "
                "Pinhole effects will be disabled.",
                UserWarning,
            )
            if "enable_pinholes" in sim_sweep:
                sim_sweep["enable_pinholes"] = [False]
            else:
                sim_fixed["enable_pinholes"] = False

    # Build sweep dimensions: every list-valued entry in a section becomes a named sweep dimension.
    # Keys use "section.attribute" notation.
    sweep_dims = {}
    for attr, vals in sim_sweep.items():
        sweep_dims[f"simulation.{attr}"] = vals
    for attr, vals in det_sweep.items():
        sweep_dims[f"detector.{attr}"] = vals
    for attr, vals in tel_sweep.items():
        sweep_dims[f"telescope.{attr}"] = vals
    for attr, vals in fil_sweep.items():
        if instrument != "EIS":
            sweep_dims[f"filter.{attr}"] = vals

    dim_names = list(sweep_dims.keys())
    dim_values = [sweep_dims[n] for n in dim_names]
    total_combinations = 1
    for v in dim_values:
        total_combinations *= len(v)

    print(f"\nRunning {total_combinations} parameter combination(s).")
    if sweep_dims:
        print("Swept dimensions:")
        for dim, vals in sweep_dims.items():
            print(f"  {dim}: {vals}")

    # Load or create input cube
    raster = None
    raster_summed = {}
    raster_cubes = {}
    synthesis = None
    # What the cubes from a single synthesis file carry beyond its spectra,
    # as those from a pickle did: its dynamic mode.
    file_meta = {}
    if uniform_intensity_mode:
        cube_sim = None
        print("\nSkipping atmosphere loading (uniform intensity mode).")
    elif atmosphere_series_mode:
        # The cubes are synthesised per combination inside the loop, since
        # the slit width and the exposure time decide what the slit sees.
        cube_sim = None
        print("\nReading the atmosphere series...")
        series = AtmosphereSeries(series_paths)
        print(f"  {len(series)} snapshots from {series.times[0]:.3f} to {series.times[-1]:.3f}")
        # Before the contribution functions, which take minutes and gigabytes.
        RasterSynthesiser.check_plan(series, raster_plan, *_swept("slit_width", "expos"))
        raster = RasterSynthesiser(series, synthesis_settings)
    elif synthesis_series_mode:
        # The spectra are read per combination inside the loop, a strip of
        # columns at a time, and each column once.
        cube_sim = None
        print("\nReading the synthesis series...")
        series = SynthesisSeries(series_paths, reference_line)
        print(f"  {len(series)} snapshots from {series.times[0]:.3f} to {series.times[-1]:.3f}")
        print(f"  Lines in the window of {reference_line}: {', '.join(series.lines)}")
        SynthesisRaster.check_plan(series, raster_plan, *_swept("slit_width", "expos"))
        raster = SynthesisRaster(series)
    else:
        print(f"\nLoading the synthesis from {synthesis_file}...")
        print(f"Using '{reference_line}' as reference line for wavelength grid and metadata...")
        if synthesis_is_hdf5:
            # Only the lines that reach the reference line's window are read,
            # and added up on its wavelengths here. The sum is resampled onto
            # the detector grid per slit width inside the loop.
            synthesis = read_synthesis(synthesis_file, reference_line)
            print(f"  Lines in its window: {', '.join(synthesis.lines)}")
            if synthesis.source:
                print(f"  Source: {synthesis.source}")
            products = read_synthesis_products(synthesis_file, keys=("dynamic_mode",))
            dynamic_mode_info = products.get("dynamic_mode", {"enabled": False})
            summed_input = synthesis.summed(reference_line)
            # Kept in the results as the spectra the instrument observed, where
            # evenly spaced wavelengths let a WCS describe them.
            file_meta = {"dynamic_mode": dynamic_mode_info}
            cube_sim = (synthesis.summed_cube(reference_line, summed_input, file_meta)
                        if synthesis.evenly_spaced(reference_line) else None)
        else:
            cube_sim, dynamic_mode_info = load_atmosphere(synthesis_file, reference_line)

        is_dynamic_mode = dynamic_mode_info.get("enabled", False)
        if is_dynamic_mode:
            print("Synthesis was done in DYNAMIC MODE (time-varying atmosphere)")
            print(f"  Slit width: {dynamic_mode_info['slit_width']}")
            print(f"  Slit rest time: {dynamic_mode_info['slit_rest_time']}")
            print(f"  Timesteps used: {len(dynamic_mode_info['available_timesteps'])}")

            synth_slit_width = dynamic_mode_info["slit_width"]
            synth_rest_time = dynamic_mode_info["slit_rest_time"]

            slit_width_vals = sweep_dims.get(
                "simulation.slit_width", [sim_fixed["slit_width"]]
            )
            expos_vals = sweep_dims.get(
                "simulation.expos", [sim_fixed["expos"]]
            )

            if len(slit_width_vals) != 1:
                raise ValueError(
                    f"Dynamic mode synthesis requires exactly one slit width. "
                    f"Config specifies {len(slit_width_vals)}: {slit_width_vals}. "
                    f"Please provide only the synthesis slit width: {synth_slit_width}"
                )
            if not np.isclose(
                slit_width_vals[0].to_value(u.arcsec),
                synth_slit_width.to_value(u.arcsec),
                rtol=1e-6,
            ):
                raise ValueError(
                    f"Slit width mismatch: synthesis was done with {synth_slit_width}, "
                    f"but config specifies {slit_width_vals[0]}."
                )
            if len(expos_vals) != 1:
                raise ValueError(
                    f"Dynamic mode synthesis requires exactly one exposure time. "
                    f"Config specifies {len(expos_vals)}: {expos_vals}. "
                    f"Please provide only the synthesis slit rest time: {synth_rest_time}"
                )
            if not np.isclose(
                expos_vals[0].to_value(u.s),
                synth_rest_time.to_value(u.s),
                rtol=1e-6,
            ):
                raise ValueError(
                    f"Exposure time mismatch: synthesis was done with {synth_rest_time}, "
                    f"but config specifies {expos_vals[0]}."
                )
            print("  Dynamic mode parameters validated successfully!")

    # Main sweep loop
    all_results = {}
    # cube_reb_cache: keyed by (slit_width_arcsec, plate_scale, wvl_res)
    cube_reb_cache = {}
    # rebin_cache: keyed by (slit_width_arcsec, plate_scale, wvl_res, offchip_bin_slit)
    rebin_cache = {}
    # observed_cache: the cubes the Monte Carlo observes through the PSF,
    # keyed by the sampling and the PSF.
    observed_cache = {}
    # A synthesis series' raster, its metadata and its summed spectra, by sampling.
    raster_syntheses = {}
    # Keyed by slit_width_arcsec (first match) for convenient downstream access
    cube_reb_dict = {}

    # Add offchip_bin_slit as a sweep dimension if needed
    if len(offchip_bin_slits) > 1:
        sweep_dims["offchip_bin_slit"] = offchip_bin_slits
        dim_names = list(sweep_dims.keys())
        dim_values = [sweep_dims[n] for n in dim_names]
        total_combinations = 1
        for v in dim_values:
            total_combinations *= len(v)
        print(f"\nUpdated to {total_combinations} parameter combination(s) (including offchip_bin_slit sweep).")

    # Each raster of a time series is observed on its own, so a plan of several
    # rasters is a sweep over them, and a sit-and-stare a sweep over its
    # exposures.
    if raster_mode and raster_plan.repeats > 1:
        sweep_dims["raster.repeat"] = list(range(raster_plan.repeats))
        dim_names = list(sweep_dims.keys())
        dim_values = [sweep_dims[n] for n in dim_names]
        total_combinations = 1
        for v in dim_values:
            total_combinations *= len(v)
        print(f"\nUpdated to {total_combinations} parameter combination(s) (one per raster repeat).")

    def _combinations():
        """The sweep's combinations, afresh each time, one at a time rather than all at once."""
        return itertools_product(*dim_values) if dim_names else iter([()])

    def _combination(combo_values):
        """One combination: its sweep values, and its configuration objects, which check them."""
        combo = dict(zip(dim_names, combo_values)) if dim_names else {}
        offchip = combo.pop("offchip_bin_slit", offchip_bin_slits[0])
        repeat = combo.pop("raster.repeat", 0)

        # Merge sweep values with fixed values for this combination
        all_sim = {
            **sim_fixed,
            **{k[len("simulation."):]: v for k, v in combo.items() if k.startswith("simulation.")},
        }
        all_det = {
            **det_fixed,
            **{k[len("detector."):]: v for k, v in combo.items() if k.startswith("detector.")},
        }
        all_tel = {
            **tel_fixed,
            **{k[len("telescope."):]: v for k, v in combo.items() if k.startswith("telescope.")},
        }
        all_fil = {
            **fil_fixed,
            **{k[len("filter."):]: v for k, v in combo.items() if k.startswith("filter.")},
        }

        # Build config objects
        if instrument == "SWC":
            fil = AluminiumFilter(**all_fil) if all_fil else AluminiumFilter()
            tel_kwargs = {k: v for k, v in all_tel.items() if k != "filter"}
            tel_kwargs["filter"] = fil
            tel = Telescope_EUVST(**tel_kwargs)
            det = Detector_SWC(**all_det) if all_det else Detector_SWC()
        else:
            fil = None
            tel_kwargs = {k: v for k, v in all_tel.items() if k != "filter"}
            tel = Telescope_EIS(**tel_kwargs) if tel_kwargs else Telescope_EIS()
            det = Detector_EIS(**all_det) if all_det else Detector_EIS()

        enable = all_sim.get("enable_pinholes", False)
        sim = Simulation(
            expos=all_sim["expos"],
            n_iter=n_iter,
            slit_width=all_sim["slit_width"],
            ncpu=ncpu,
            instrument=instrument,
            vis_sl=all_sim.get("vis_sl", 0.0 * u.photon / (u.s * u.cm**2)),
            psf=all_sim.get("psf", False),
            psf_boundary=all_sim.get("psf_boundary", "replicate"),
            spectral_psf=all_sim.get("spectral_psf", "quadrature"),
            noise=all_sim.get("noise", True),
            enable_pinholes=enable,
            pinhole_sizes=pinhole_sizes if enable else [],
            pinhole_positions=pinhole_positions if enable else [],
            pinhole_positions_spectral=pinhole_positions_spectral if enable else [],
        )
        # The optics' spectral FWHM comes from the telescope and the detector
        # together, and was worked out only once the combination's scene was
        # laid onto the detector: every run needs it when psf_params is for a
        # given slit, and the PSF convolved with the slit always does.
        if tel.psf_slit_width is not None or (sim.psf and sim.spectral_psf == "convolution"):
            spectral_optics_fwhm(tel, det)
        return combo, offchip, repeat, fil, tel, det, sim

    # Every combination is checked before the first is run, so that a value
    # that cannot be run does not end the sweep part-way, and lose the
    # combinations run before it, which are saved only at the end.
    for combo_values in _combinations():
        _, _, _, _, tel, _, _ = _combination(combo_values)
    # The throughput tables are read now too, where one that was missing or
    # had a slip in it was found only when the first photons were counted.
    # They cannot be swept, so every combination has the last one's.
    if instrument == "SWC":
        for table in (tel.pm_table, tel.grating_table, tel.filter.al_table,
                      tel.filter.oxide_table, tel.filter.c_table):
            _load_throughput_table(table)

    def _check_offchip_bins(fov_along_slit):
        """Refuse an offchip_bin_slit that bins more rows than a combination's scene covers."""
        for combo_values in _combinations():
            _, n_bin, _, _, _, det, _ = _combination(combo_values)
            n_slit = _whole_pixels(fov_along_slit, det.plate_scale_angle)
            if n_bin > n_slit:
                raise ValueError(
                    f"offchip_bin_slit {n_bin} bins more rows than the {n_slit} along the "
                    f"slit, so there would be nothing left; bin at most {n_slit}.")

    # The off-chip binning is checked against the scene once it is first
    # rebinned; a uniform intensity is made with the rows it bins.
    offchip_checked = uniform_intensity_mode

    for combination_idx, combo_values in enumerate(_combinations(), start=1):
        (combo, offchip_bin_slit, raster_repeat, filter_obj, TEL, DET,
         SIM) = _combination(combo_values)

        # Extract core simulation params
        slit_width = SIM.slit_width
        expos = SIM.expos
        vis_sl = SIM.vis_sl
        psf = SIM.psf
        psf_boundary = SIM.psf_boundary
        spectral_psf = SIM.spectral_psf
        noise = SIM.noise
        enable_pinholes = SIM.enable_pinholes

        # Two-level rebinning cache: rebin_atmosphere does not depend on offchip_bin_slit,
        # so cube_reb_cache is keyed by the sampling alone to avoid redundant rebin calls
        # when sweeping multiple binning values at fixed spatial/spectral sampling.
        sampling_key = (
            slit_width.to_value(u.arcsec),
            DET.plate_scale_angle.to_value(u.arcsec / u.pixel),
            DET.wvl_res.to_value(u.cm / u.pixel),
        )
        # In uniform-intensity mode the cube is built with one slit pixel per binning
        # factor, so that rebin_slit_offchip has independent noise realisations to sum.
        # The cube therefore does depend on offchip_bin_slit, and the key must say so.
        # A time series is synthesised per exposure time as well as per slit width.
        if uniform_intensity_mode:
            cube_reb_key = (*sampling_key, offchip_bin_slit)
        elif raster_mode:
            cube_reb_key = (*sampling_key, expos.to_value(u.s), raster_repeat)
        else:
            cube_reb_key = sampling_key
        rebin_cache_key = (*cube_reb_key, offchip_bin_slit)

        if atmosphere_series_mode and cube_reb_key not in cube_reb_cache:
            print(f"\nSynthesising the time series as observed "
                  f"(slit_width={slit_width}, expos={expos})...")
            cube_sim = raster.summed_cube(raster_plan, slit_width, expos, reference_line,
                                          repeat=raster_repeat)
            raster_summed[cube_reb_key] = cube_sim
            print(f"  {cube_sim.data.shape[1]} exposures, {raster.strips_synthesised} "
                  f"strips synthesised so far")
        if synthesis_series_mode and cube_reb_key not in cube_reb_cache:
            print(f"\nReading the time series as observed "
                  f"(slit_width={slit_width}, expos={expos})...")
            # The exposures, as a synthesis whose columns they are, go onto
            # the detector as a single snapshot does.
            synthesis, raster_meta = raster.synthesis(raster_plan, slit_width, expos,
                                                      repeat=raster_repeat)
            summed_input = synthesis.summed(reference_line)
            cube_sim = None
            if synthesis.evenly_spaced(reference_line):
                cube_sim = synthesis.summed_cube(reference_line, summed_input, raster_meta)
            raster_summed[cube_reb_key] = cube_sim
            raster_syntheses[cube_reb_key] = (synthesis, raster_meta, summed_input)
            print(f"  {len(raster_meta['positions'])} exposures, {raster.strips_read} "
                  f"strips read so far")

        if cube_reb_key not in cube_reb_cache:
            print(
                f"\nRebinning atmosphere "
                f"(slit_width={slit_width}, "
                f"plate_scale={DET.plate_scale_angle}, "
                f"wvl_res={DET.wvl_res})..."
            )
            SIM_rebin = Simulation(
                expos=1.0 * u.s,
                n_iter=n_iter,
                slit_width=slit_width,
                ncpu=ncpu,
                instrument=instrument,
                psf=False,
            )
            if uniform_intensity_mode:
                cube_reb_cache[cube_reb_key] = create_uniform_intensity_cube(
                    total_intensity=uniform_intensity,
                    rest_wavelength=uniform_rest_wavelength,
                    thermal_width=uniform_thermal_width,
                    det=DET,
                    sim=SIM_rebin,
                    n_slit_pixels=offchip_bin_slit,
                    tel=TEL,
                )
            else:
                # A slit wider than the one psf_params is for spreads each line
                # further than the synthesis window's margin allows, so the
                # window is widened to hold what it spreads. Not for the
                # reference slit, whose window is as it was.
                if synthesis is None:
                    rebinned = rebin_atmosphere(cube_sim, DET, SIM_rebin)
                else:
                    rebinned = rebin_spectra(
                        synthesis, reference_line, DET, SIM_rebin, summed=summed_input,
                        meta=raster_meta if synthesis_series_mode else file_meta)
                cube_reb_cache[cube_reb_key] = pad_spectral_axis(
                    rebinned, spectral_psf_margin(TEL, DET, slit_width))
                if not offchip_checked:
                    # The scene is as long along the slit for every
                    # combination, so the rows each one's plate scale gives
                    # it are known now, before any combination has run.
                    _check_offchip_bins(cube_reb_cache[cube_reb_key].meta["fov_along_slit"])
                    offchip_checked = True

        cube_reb = cube_reb_cache[cube_reb_key]

        if rebin_cache_key not in rebin_cache:
            # Apply off-chip binning
            cube_reb_binned = rebin_slit_offchip(cube_reb, offchip_bin_slit)

            # Weighted, the ground truth's fit takes the uncertainties the DN
            # would have, which change with the exposure and the instrument,
            # so it is fitted for each combination below.
            ground_truth = (None if weighted_truth
                            else _fit_ground_truth(cube_reb_binned, offchip_bin_slit))
            rebin_cache[rebin_cache_key] = (cube_reb_binned, ground_truth)
            # Key by (slit_width_arcsec, offchip_bin_slit) so that sweeps over
            # multiple binning factors at fixed slit width all retain their cubes
            # (a single-key dict would silently keep only the first one). A
            # time series adds the exposure time, which changes the cube too.
            cube_key = ((sampling_key[0], expos.to_value(u.s), raster_repeat, offchip_bin_slit)
                        if raster_mode else (sampling_key[0], offchip_bin_slit))
            cube_reb_dict.setdefault(cube_key, cube_reb_binned)
            if raster_mode:
                raster_cubes[cube_key] = raster_summed[cube_reb_key]

        cube_reb_binned, ground_truth = rebin_cache[rebin_cache_key]

        # The cube the Monte Carlo observes. With the PSF on, the scene goes
        # onto the pixels through it, which is exact where blurring the pixels
        # afterwards moves a narrow line within its pixel; the ground truth
        # stays the scene without it. A uniform intensity is centred on a
        # pixel and uniform along the slit, and the Monte Carlo blurs it.
        cube_obs = cube_reb
        if psf and not uniform_intensity_mode:
            # The observed cube carries the telescope's throughput as well as
            # its PSF, so every setting of the telescope and filter is in the key.
            telescope_key = _params_to_key({
                **_extract_config_params(TEL, "telescope"),
                **(_extract_config_params(filter_obj, "filter") if filter_obj is not None else {})})
            psf_key = (*cube_reb_key, spectral_psf, psf_boundary, telescope_key)
            if psf_key not in observed_cache:
                print("Laying the scene onto the detector through the PSF...")
                SIM_obs = Simulation(expos=1.0 * u.s, n_iter=n_iter, slit_width=slit_width,
                                     ncpu=ncpu, instrument=instrument, psf=True,
                                     psf_boundary=psf_boundary, spectral_psf=spectral_psf)
                across = TEL.psf_across_slit
                extend = psf_boundary == "replicate"
                if atmosphere_series_mode:
                    scene = raster_summed[cube_reb_key] if across is None else raster.summed_cube(
                        raster_plan, slit_width, expos, reference_line, repeat=raster_repeat,
                        across_slit=across, extend=extend)
                    observed = rebin_atmosphere(scene, DET, SIM_obs, tel=TEL)
                elif synthesis_series_mode:
                    series_synthesis, series_meta, series_summed = raster_syntheses[cube_reb_key]
                    if across is not None:
                        series_synthesis, series_meta = raster.synthesis(
                            raster_plan, slit_width, expos, repeat=raster_repeat,
                            across_slit=across, extend=extend)
                        series_summed = None
                    observed = rebin_spectra(series_synthesis, reference_line, DET, SIM_obs,
                                             summed=series_summed, meta=series_meta, tel=TEL)
                elif synthesis is None:
                    observed = rebin_atmosphere(cube_sim, DET, SIM_obs, tel=TEL)
                else:
                    observed = rebin_spectra(synthesis, reference_line, DET, SIM_obs,
                                             summed=summed_input, meta=file_meta, tel=TEL)
                observed_cache[psf_key] = observed
            cube_obs = observed_cache[psf_key]

        # Progress output
        print(f"\n--- Combination {combination_idx}/{total_combinations} ---")
        for k, v in combo.items():
            print(f"  {k}: {v}")
        if offchip_bin_slit > 1:
            print(f"  offchip_bin_slit: {offchip_bin_slit}")
        if raster_mode and raster_plan.repeats > 1:
            print(f"  raster.repeat: {raster_repeat}")
        if not combo and offchip_bin_slit == 1 and not (raster_mode and raster_plan.repeats > 1):
            print("  (single combination - all parameters fixed)")
        print(f"  Calculated dark current: {DET.dark_current:.2e}")
        if instrument == "SWC":
            print(f"  Microroughness sigma: {TEL.microroughness_sigma}")
        if enable_pinholes and pinhole_sizes:
            print(f"  Pinhole sizes: {pinhole_sizes}")
            print(f"  Pinhole positions: {pinhole_positions}")

        if weighted_truth:
            # The ground truth is weighted by the uncertainty the DN would
            # have with no noise, so that it is the same fit as the measured
            # spectra's, of the same line.
            expected = expected_dn_uncertainty(cube_obs, expos, DET, TEL, SIM, offchip_bin_slit,
                                               uniform_mode=uniform_intensity_mode)
            if expected.shape != cube_reb_binned.data.shape:
                raise RuntimeError(
                    f"The detector's pixels, shaped {expected.shape}, do not match the "
                    f"spectra the ground truth is fitted to, shaped "
                    f"{cube_reb_binned.data.shape}.")
            ground_truth = _fit_ground_truth(cube_reb_binned, offchip_bin_slit, expected)

        # Run Monte Carlo
        first_dn_signal, dn_fit_stats, first_photon_signal, photon_fit_stats = monte_carlo(
            cube_obs, expos, DET, TEL, SIM,
            n_iter=SIM.n_iter,
            fit_config=fit_config,
            offchip_bin_slit=offchip_bin_slit,
            fit_signals=fit_signals,
            uniform_mode=uniform_intensity_mode,
        )

        # Only store results on rank 0 (MPI-aware)
        if _mpi_rank == 0 and first_dn_signal is not None:
            # Build parameters dict from actual config objects so all fields
            # (including those using class defaults) are recorded.
            parameters = {}
            parameters.update(_extract_config_params(SIM, "simulation"))
            parameters.update(_extract_config_params(DET, "detector"))
            if instrument == "SWC":
                parameters.update(_extract_config_params(filter_obj, "filter"))
            parameters.update(_extract_config_params(TEL, "telescope"))
            # Add offchip_bin_slit to the parameters dict
            parameters["offchip_bin_slit"] = offchip_bin_slit
            if raster_mode:
                parameters["raster.repeat"] = raster_repeat

            param_key = _params_to_key(parameters)

            all_results[param_key] = {
                "parameters": parameters,
                "config_objects": {
                    "detector": DET,
                    "telescope": TEL,
                    "simulation": SIM,
                },
                "first_dn_signal_data": first_dn_signal.data,
                "first_dn_signal_unit": first_dn_signal.unit,
                "first_photon_signal_data": first_photon_signal.data,
                "first_photon_signal_unit": first_photon_signal.unit,
                "first_signal_wcs": first_dn_signal.wcs,
                "dn_fit_stats": dn_fit_stats,
                "photon_fit_stats": photon_fit_stats,
                "ground_truth": ground_truth,
            }

            del first_dn_signal, first_photon_signal, dn_fit_stats, photon_fit_stats

    # Package results (rank 0 only in MPI mode)
    if _mpi_rank == 0:
        results = {
            "all_combinations": all_results,
            # All sweep dimensions and their value lists
            "sweep_dimensions": sweep_dims,
            # All fixed (non-swept) parameter values
            "fixed_params": {
                **{f"simulation.{k}": v for k, v in sim_fixed.items()},
                **{f"detector.{k}": v for k, v in det_fixed.items()},
                **{f"telescope.{k}": v for k, v in tel_fixed.items()},
                **(
                    {f"filter.{k}": v for k, v in fil_fixed.items()}
                    if instrument == "SWC"
                    else {}
                ),
                "offchip_bin_slit": offchip_bin_slits[0] if len(offchip_bin_slits) == 1 else None,
                **({"raster.repeat": 0} if raster_mode and raster_plan.repeats == 1 else {}),
            },
            "fit_config": fit_config,
            "fit_signals": fit_signals,
            # Whether the fits were weighted: the DN by each pixel's uncertainty,
            # the photons by their Poisson likelihood. Results made before this was
            # recorded were not.
            "fit_weighted": weighted_fits,
        }

        # Save
        git_commit_id = get_git_commit_id()
        software_version = _get_software_version()

        output_file = Path(f"run/result/{Path(args.config).stem}.h5")

        print(f"\nSaving results to {output_file}")
        save_data = {
            "results": results,
            "config": config,
            "instrument": instrument,
            "cube_sim": cube_sim,
            "cube_reb_dict": cube_reb_dict,
            "git_commit_id": git_commit_id,
            "software_version": software_version,
        }
        if raster_mode:
            # What the series was observed with, and the cube each combination
            # saw; cube_sim is the last combination's.
            save_data["raster"] = {
                "plan": raster_plan,
                "settings": synthesis_settings,
                "series": [str(p) for p in series.paths],
                "times": series.times,
                "hdf5_dbase_root": raster.goft_dbase_root if atmosphere_series_mode else None,
                "cubes": raster_cubes,
            }

        output_file = save_results(output_file, save_data)

        print(f"Saved results to {output_file} ({os.path.getsize(output_file) / 1e6:.1f} MB)")
        # A rerun replaced the pickle of the same name when results were
        # pickles, so one left from an older version is moved aside rather
        # than read in place of these results.
        stale = output_file.with_suffix(".pkl")
        if stale.is_file():
            # Beside any moved aside before, rather than over it, which may be
            # the only copy of still older results.
            aside = stale.with_name(stale.name + ".old")
            number = 0
            while aside.exists():
                number += 1
                aside = stale.with_name(f"{stale.name}.old.{number}")
            try:
                os.replace(stale, aside)
                print(f"Moved {stale}, from an older version, to {aside.name}")
            except OSError as error:
                print(f"Could not move {stale}, from an older version, aside ({error}); "
                      f"the results are {output_file}, not it")
        print(f"Software version: {software_version}  |  Git commit: {git_commit_id}")
        print(f"Instrument response simulation complete! Total combinations: {total_combinations}")


if __name__ == "__main__":
    main()
