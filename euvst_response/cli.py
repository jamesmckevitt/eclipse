"""
Command line interface for M-ECLIPSES.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

from .main import main as run_simulation


ASCII_LOGO = """
   __  __        ______ _____ _      _____ _____   _____ ______  _____ 
  |  \/  |      |  ____/ ____| |    |_   _|  __ \ / ____|  ____|/ ____|
  | \  / |______| |__ | |    | |      | | | |__) | (___ | |__  | (___  
  | |\/| |______|  __|| |    | |      | | |  ___/ \___ \|  __|  \___ \ 
  | |  | |      | |___| |____| |____ _| |_| |     ____) | |____ ____) |
  |_|  |_|      |______\_____|______|_____|_|    |_____/|______|_____/ 

M-ECLIPSES: MSSL Emission Calculation and Line Intensity Prediction for SOLAR-C EUVST-SW

Contact: James McKevitt (jm2@mssl.ucl.ac.uk). License: Contact for permission to use.
"""


def print_logo():
    """Print the M-ECLIPSES ASCII logo and info."""
    print(ASCII_LOGO)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="m-eclipses",
        description="M-ECLIPSES: MSSL Emission Calculation and Line Intensity Prediction for SOLAR-C EUVST-SW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="M-ECLIPSES 1.0.0"
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

    args = parser.parse_args()

    # Always print the logo first
    print_logo()

    if args.logo_only:
        return

    if not args.config:
        print("Usage: m-eclipses --config <config.yaml>")
        print("\nTo run instrument response simulation, provide a YAML config file.")
        print("Example config files can be found in the run/input/ directory.")
        print("\nFor more help: m-eclipses --help")
        return

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    print(f"Running instrument response simulation with config: {args.config}")
    print("-" * 60)
    
    # Set up sys.argv for the main function (it expects argparse format)
    sys.argv = ["m-eclipses", "--config", args.config]
    
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during simulation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
