"""
Command line interface for ECLIPSE.
"""

from __future__ import annotations
import argparse
import os
import sys
import traceback
import warnings

from . import __version__
from .synthesis import main as run_synthesis


ASCII_LOGO = r"""
  ______ _____ _      _____ _____   _____ ______ 
 |  ____/ ____| |    |_   _|  __ \ / ____|  ____|
 | |__ | |    | |      | | | |__) | (___ | |__   
 |  __|| |    | |      | | |  ___/ \___ \|  __|  
 | |___| |____| |____ _| |_| |     ____) | |____ 
 |______\_____|______|_____|_|    |_____/|______|

ECLIPSE: Emission Calculation and Line Prediction for SOLAR-C EUVST

Contact: James McKevitt (jm2@mssl.ucl.ac.uk). See LICENSE for usage terms.
"""


def print_logo():
    """Print the ECLIPSE ASCII logo and info."""
    print(ASCII_LOGO)


_COMMANDS = {
    "run": "Do everything the run file describes: synthesise its atmosphere if it names "
           "one, unless the synthesis is made already, observe it, and make the full-CCD "
           "frame if it has a 'frame:' section.",
    "synthesise": "Synthesise the run file's atmosphere.",
    "observe": "Observe what the run file describes, and write the results.",
    "frame": "Make the full-CCD frame the run file's 'frame:' section describes, "
             "synthesising its atmosphere first if needed.",
}

_PASSED_ON = {
    "atmosphere": "Write or inspect atmosphere files; see 'eclipse atmosphere --help'.",
    "science-cases": "Write run files for the science cases; see "
                     "'eclipse science-cases --help'.",
}


def build_parser() -> argparse.ArgumentParser:
    """Command line parser for `eclipse`."""
    parser = argparse.ArgumentParser(
        prog="eclipse",
        description="ECLIPSE: Emission Calculation and Line Prediction for SOLAR-C EUVST",
        epilog="The documentation says how to write a run file: "
               "https://solarc-eclipse.readthedocs.io",
    )
    parser.add_argument("--version", action="version", version=f"ECLIPSE {__version__}")
    parser.add_argument("--config", type=str, default=None,
                        help="The old form of 'eclipse run FILE'")
    parser.add_argument("--logo-only", action="store_true", help="Just print the logo and exit")
    parser.add_argument("--debug", action="store_true",
                        help="Open an IPython debugger where an error happens")
    commands = parser.add_subparsers(dest="command", metavar="command")
    for command, text in _COMMANDS.items():
        sub = commands.add_parser(command, help=text, description=text)
        sub.add_argument("run_file", help="The run file (YAML)")
        sub.add_argument("--set", action="append", default=[], dest="settings",
                         metavar="KEY=VALUE",
                         help="Change a setting of the run file, with a dotted key and a "
                              "YAML value, such as --set simulation.expos='[5 s, 20 s]'. "
                              "Can be given more than once.")
        sub.add_argument("--only", metavar="RUN", default=None,
                         help="Only the run of this name in the file's 'runs:'")
        sub.add_argument("--debug", action="store_true", default=argparse.SUPPRESS,
                         help="Open an IPython debugger where an error happens")
    for command, text in _PASSED_ON.items():
        sub = commands.add_parser(command, help=text, add_help=False)
        sub.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def _first_word(argv: list) -> str | None:
    """The first command-line word that is not an option or its value."""
    words = iter(argv)
    for word in words:
        if word == "--config":
            next(words, None)
        elif not word.startswith("-"):
            return word
    return None


def main(argv=None):
    """Main CLI entry point."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Passed on whole, so their own options are not read as eclipse's.
    command = _first_word(argv)
    if command == "atmosphere":
        from .atmosphere import main as atmosphere_main
        atmosphere_main(argv[argv.index(command) + 1:])
        return
    if command == "science-cases":
        from .science_cases import main as science_cases_main
        science_cases_main(argv[argv.index(command) + 1:])
        return
    args = build_parser().parse_args(argv)

    # Always print the logo first
    print_logo()

    if args.logo_only:
        return

    if args.command is None:
        if not args.config:
            print("Usage: eclipse run <run file>")
            print("\nThe documentation says how to write one: "
                  "https://solarc-eclipse.readthedocs.io")
            print("\nFor more help: eclipse --help")
            return
        warnings.warn(f"'eclipse --config {args.config}' is the old form of 'eclipse run "
                      f"{args.config}', and will be removed in a future release.",
                      FutureWarning, stacklevel=1)
        args.command, args.run_file, args.settings, args.only = "run", args.config, [], None

    print(f"{args.command.capitalize()}: {args.run_file}")
    print("-" * 60)
    from .run_file import run_all
    try:
        run_all(args.run_file, args.command, settings=args.settings, only=args.only,
                debug=args.debug)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        # A run ECLIPSE cannot do, or a missing file, raises an error that says why.
        # Anything else is unexpected, so print its traceback.
        if not isinstance(e, (ValueError, FileNotFoundError)):
            traceback.print_exc()
        _fail(f"Error during {args.command}: {type(e).__name__}: {e}")


def _fail(message: str) -> None:
    """
    Print `message` and end the run on every MPI rank.

    Ranks other than the first have their output silenced, so the message
    goes to the stderr the rank started with. An MPI run is aborted, as the
    other ranks would otherwise wait for this one until the job's time runs
    out.
    """
    from .main import _STARTING_STDERR
    from .utils import _get_mpi_info

    comm, rank, size = _get_mpi_info()
    if size > 1:
        os.write(_STARTING_STDERR or 2, f"MPI rank {rank}: {message}\n".encode())
        comm.Abort(1)
    print(message)
    sys.exit(1)


if __name__ == "__main__":
    main()


def synthesis_main():
    """Entry point for the synthesis command line script."""
    run_synthesis()
