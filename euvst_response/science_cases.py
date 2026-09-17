"""
Science cases from the Concept Study Report, and ECLIPSE configurations for them.

``data/science_cases.yaml`` holds every case with every line.  The functions
here turn the lines ECLIPSE can currently simulate into uniform-intensity
configurations, and the ``eclipse-science-cases`` command writes them to files.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
import yaml
from mendeleev import element

from .config import Telescope_EUVST, _load_throughput_table

# What every configuration gets unless the base settings say otherwise: the
# settings the cases were run with for the NASA PDR analysis.
DEFAULT_SETTINGS = {
    "n_iter": 512,
    "ncpu": -1,
    "offchip_bin_slit": [1, 2],
    "simulation": {"psf": True},
}

# Keys that each configuration takes from its line, so the base settings
# cannot set them.
_LINE_KEYS = ("instrument", "uniform_intensity", "rest_wavelength",
              "thermal_width", "synthesis_file", "reference_line")

# Two table entries for the same line can be written to different precision,
# e.g. 195.119 and 195.12, so a wavelength picks lines within this.
_WAVELENGTH_MATCH = 0.01 * u.AA

_ROMAN = {numeral: value for value, numeral in enumerate(
    "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX "
    "XXI XXII XXIII XXIV XXV XXVI XXVII XXVIII XXIX XXX".split(), start=1)}


@dataclass
class ScienceCaseConfig:
    """The ECLIPSE configuration for one line of one science case."""
    name: str
    case: dict
    line: dict
    config: dict

    def to_yaml(self) -> str:
        """The configuration as a file, with a header saying where it came from."""
        header = (
            f"# Science case {self.case['task']} ({self.case['name']}): "
            f"{self.case['description']}\n"
            f"# Line: {line_label(self.line)}, log T_max = {float(self.line['log_t_max']):.2f}, "
            f"intensity {self.line['intensity']} times filling factor "
            f"{self.case['filling_factor']}\n"
            f"# Written by eclipse-science-cases from the science case table\n\n"
        )
        return header + yaml.safe_dump(self.config, sort_keys=False,
                                       default_flow_style=None)


def load_science_cases(path: str | Path | None = None) -> list[dict]:
    """
    Read the science case table.

    Parameters
    ----------
    path : str or Path, optional
        A table to read instead of the one ECLIPSE includes.

    Returns
    -------
    list of dict
        One entry per case, as written in the table.
    """
    if path is None:
        source = files("euvst_response") / "data" / "science_cases.yaml"
    else:
        source = Path(path)
    return yaml.safe_load(source.read_text())


def line_label(line: dict) -> str:
    """A line as written in the table, e.g. ``Fe XII 195.119``."""
    return f"{line['ion']} {line['wavelength'].split()[0]}"


def line_id(line: dict) -> str:
    """A line as used in file names, e.g. ``fe12_195119``."""
    symbol, numeral = line["ion"].split()
    wavelength = u.Quantity(line["wavelength"]).to_value(u.AA)
    return f"{symbol.lower()}{_ROMAN[numeral]:02d}_{wavelength:.3f}".replace(".", "")


@lru_cache(maxsize=None)
def _atomic_weight(symbol: str) -> float:
    """Atomic weight in atomic mass units.  mendeleev's lookup is slow, so once per element."""
    return element(symbol).atomic_weight


def thermal_width(line: dict) -> u.Quantity:
    """The 1-sigma thermal velocity, sqrt(k T / m), at the line's T_max."""
    mass = _atomic_weight(line["ion"].split()[0]) * const.u
    temperature = 10.0 ** float(line["log_t_max"]) * u.K
    return np.sqrt(const.k_B * temperature / mass).to(u.km / u.s)


@lru_cache(maxsize=None)
def short_wavelength_range() -> tuple[u.Quantity, u.Quantity]:
    """
    The wavelengths ECLIPSE can simulate in the short wavelength channel.

    Where both the primary mirror and the grating efficiency tables have
    values, since the effective area is undefined outside them.
    """
    telescope = Telescope_EUVST()
    tables = [_load_throughput_table(table)[0]
              for table in (telescope.pm_table, telescope.grating_table)]
    return (max(t.min() for t in tables).to(u.AA),
            min(t.max() for t in tables).to(u.AA))


def instrument_for(line: dict) -> str | None:
    """The ECLIPSE instrument that can simulate *line*, or None if there is none yet."""
    low, high = short_wavelength_range()
    return "SWC" if low <= u.Quantity(line["wavelength"]) <= high else None


def _line_matches(line: dict, selector: str) -> bool:
    """Whether *selector*, an ion or an ion and a wavelength, picks *line*."""
    tokens = selector.split()
    if len(tokens) not in (2, 3):
        raise ValueError(
            f"Choose a line by ion, e.g. 'Fe XII', or by ion and wavelength in "
            f"Angstrom, e.g. 'Fe XII 195.119'; got {selector!r}."
        )
    if line["ion"] != " ".join(tokens[:2]):
        return False
    if len(tokens) == 2:
        return True
    wavelength = u.Quantity(line["wavelength"])
    return abs(wavelength - float(tokens[2]) * u.AA) <= _WAVELENGTH_MATCH


def _select_cases(table: list[dict], selectors: list[str] | None) -> list[dict]:
    if not selectors:
        return table
    unknown = [s for s in selectors
               if not any(s in (case["task"], case["name"]) for case in table)]
    if unknown:
        raise ValueError(
            f"No science case is called {unknown}. The cases are "
            f"{[case['task'] for case in table]}, or their names, e.g. "
            f"{table[0]['name']!r}."
        )
    return [case for case in table
            if any(s in (case["task"], case["name"]) for s in selectors)]


def _config(case: dict, line: dict, instrument: str, base: dict) -> dict:
    """One line's configuration: defaults, then the case, then the base settings."""
    value, unit = line["intensity"].split(maxsplit=1)
    intensity = float(value) * float(case["filling_factor"])

    settings = {**DEFAULT_SETTINGS, **base}
    config = {"instrument": instrument}
    config.update({k: v for k, v in settings.items() if not isinstance(v, dict)})
    config["uniform_intensity"] = f"{intensity:.6g} {unit}"
    config["rest_wavelength"] = line["wavelength"]
    config["thermal_width"] = f"{thermal_width(line).to_value(u.km / u.s):.2f} km/s"
    config["simulation"] = {
        **DEFAULT_SETTINGS["simulation"],
        "expos": case["exposure"],
        "slit_width": case["slit_width"],
        **base.get("simulation", {}),
    }
    config.update({k: v for k, v in settings.items()
                   if isinstance(v, dict) and k != "simulation"})
    return config


def science_case_configs(cases: list[str] | None = None,
                         lines: list[str] | None = None,
                         base: dict | None = None,
                         table: list[dict] | None = None,
                         ) -> tuple[list[ScienceCaseConfig], list[str]]:
    """
    ECLIPSE configurations for the science case lines ECLIPSE can simulate.

    Each uses uniform intensity mode: the line's intensity times the case's
    filling factor, the thermal width at the line's T_max, and the case's
    exposure time and slit width.

    Parameters
    ----------
    cases : list of str, optional
        Cases to include, by task (``I-1-1``) or name
        (``1.1.1-nanoflares_events``).  Default: every case.
    lines : list of str, optional
        Lines to include, by ion (``Fe XII``) or by ion and wavelength in
        Angstrom (``Fe XII 195.119``, matched to 0.01 Angstrom).  Default:
        every line.
    base : dict, optional
        Settings for every configuration, as in a configuration file.  They
        replace the defaults (512 iterations on every CPU, the PSF on, and
        off-chip binning of 1 and 2) and, where they set them, the case's
        exposure time and slit width.  They cannot set the instrument or the
        line.
    table : list of dict, optional
        Science cases to use instead of the table ECLIPSE includes.

    Returns
    -------
    configs : list of ScienceCaseConfig
        One per chosen line that ECLIPSE can simulate.
    skipped : list of str
        Chosen lines ECLIPSE cannot simulate yet, as ``<task> <line>``.
    """
    from .main import _validate_config_keys

    table = load_science_cases() if table is None else table
    base = {} if base is None else base
    for key in _LINE_KEYS:
        if key in base:
            raise ValueError(
                f"The base settings cannot set '{key}': every configuration "
                f"takes it from its line."
            )

    configs, skipped, matched = [], [], set()
    for case in _select_cases(table, cases):
        for line in case["lines"]:
            if lines:
                picked = {s for s in lines if _line_matches(line, s)}
                if not picked:
                    continue
                matched |= picked
            instrument = instrument_for(line)
            if instrument is None:
                skipped.append(f"{case['task']} {line_label(line)}")
                continue
            config = _config(case, line, instrument, base)
            _validate_config_keys(config, instrument)
            configs.append(ScienceCaseConfig(
                name=f"{case['name']}_{line_id(line)}",
                case=case, line=line, config=config))

    unmatched = [s for s in (lines or []) if s not in matched]
    if unmatched:
        raise ValueError(f"No line in the chosen cases matches {unmatched}.")
    return configs, skipped


def write_science_case_configs(out_dir: str | Path, **kwargs) -> tuple[list[Path], list[str]]:
    """
    Write the configurations from :func:`science_case_configs` to files.

    Parameters
    ----------
    out_dir : str or Path
        Folder to write to.  Each file is named after its case and line, so
        the results are too.
    **kwargs
        Passed to :func:`science_case_configs`.

    Returns
    -------
    paths : list of Path
        The files written.
    skipped : list of str
        Chosen lines ECLIPSE cannot simulate yet.
    """
    configs, skipped = science_case_configs(**kwargs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in configs:
        path = out_dir / f"{item.name}.yaml"
        path.write_text(item.to_yaml())
        paths.append(path)
    return paths, skipped


def main(argv: list[str] | None = None) -> None:
    """Command line entry point for ``eclipse-science-cases``."""
    parser = argparse.ArgumentParser(
        prog="eclipse-science-cases",
        description="Write ECLIPSE configurations for the science cases in "
                    "the Concept Study Report.",
    )
    parser.add_argument("--out", default="run/input",
                        help="folder to write the configurations to "
                             "(default: run/input)")
    parser.add_argument("--case", action="append", dest="cases", metavar="CASE",
                        help="a case to include, by task (I-1-1) or name "
                             "(1.1.1-nanoflares_events); repeat for more "
                             "(default: every case)")
    parser.add_argument("--line", action="append", dest="lines", metavar="LINE",
                        help="a line to include, by ion ('Fe XII') or by ion "
                             "and wavelength in Angstrom ('Fe XII 195.119'); "
                             "repeat for more (default: every line)")
    parser.add_argument("--base", metavar="YAML",
                        help="a configuration file whose settings go into every "
                             "configuration, e.g. a detector section or n_iter")
    parser.add_argument("--list", action="store_true",
                        help="list the chosen cases and lines instead of "
                             "writing anything")
    args = parser.parse_args(argv)

    if args.list:
        table = _select_cases(load_science_cases(), args.cases)
        for case in table:
            chosen = [line for line in case["lines"]
                      if not args.lines or any(_line_matches(line, s) for s in args.lines)]
            if not chosen:
                continue
            print(f"{case['task']}  {case['name']}: {case['description']}")
            for line in chosen:
                instrument = instrument_for(line) or "not simulated yet"
                print(f"    {line_label(line):<22} {instrument}")
        return

    base = {}
    if args.base:
        base = yaml.safe_load(Path(args.base).read_text()) or {}
        if not isinstance(base, dict):
            raise ValueError(f"{args.base} has to hold a mapping of settings.")

    paths, skipped = write_science_case_configs(
        args.out, cases=args.cases, lines=args.lines, base=base)
    low, high = short_wavelength_range()
    print(f"Wrote {len(paths)} configuration{'' if len(paths) == 1 else 's'} "
          f"to {args.out}")
    if skipped:
        print(f"Skipped {len(skipped)} line{'' if len(skipped) == 1 else 's'} "
              f"outside the {low.value:g} to {high.value:g} Angstrom ECLIPSE can "
              f"simulate yet; --list shows which")
