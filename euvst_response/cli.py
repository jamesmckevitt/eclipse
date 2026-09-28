"""
Command line interface for ECLIPSE.
"""

from __future__ import annotations
import argparse
import os
import sys
import traceback
from pathlib import Path

from . import __version__
from .main import main as run_simulation
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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="eclipse",
        description="ECLIPSE: Emission Calculation and Line Prediction for SOLAR-C EUVST",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"ECLIPSE {__version__}"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="YAML config file for instrument response simulation",
        required=False
    )
    
    parser.add_argument(
        "--logo-only",
        action="store_true",
        help="Just print the logo and exit"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode - drops to IPython on errors"
    )

    args = parser.parse_args()

    # Always print the logo first
    print_logo()

    if args.logo_only:
        return

    if not args.config:
        print("Usage: eclipse --config <config.yaml>")
        print("\nTo run instrument response simulation, provide a YAML config file.")
        print("The documentation says how to write one: https://solarc-eclipse.readthedocs.io")
        print("\nFor more help: eclipse --help")
        return

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    print(f"Running instrument response simulation with config: {args.config}")
    print("-" * 60)
    
    # Set up sys.argv for the main function (it expects argparse format)
    if args.debug:
        sys.argv = ["eclipse", "--config", args.config, "--debug"]
    else:
        sys.argv = ["eclipse", "--config", args.config]
    
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        # ECLIPSE refuses what it cannot run with a ValueError that says why;
        # anything else is unexpected, and its traceback says where.
        if not isinstance(e, (ValueError, FileNotFoundError)):
            traceback.print_exc()
        _fail(f"Error during simulation: {type(e).__name__}: {e}")


def _fail(message: str) -> None:
    """
    Say *message* and end the run, every MPI rank of it.

    A rank other than the first has its output silenced, so its message goes
    to the stderr it started with, and an MPI run is aborted, since the other
    ranks would otherwise wait for this one until the job's time runs out.
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
