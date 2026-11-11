#!/usr/bin/env python3
"""
Command line interface for synthesis script.
"""

import sys


def main():
    """Entry point for the synthesis command line script."""
    from .synthesis import _config_option, main as run_synthesis, parse_arguments

    words = sys.argv[1:]
    config = _config_option(words)
    # Given only a run file, the synthesis is the run's, made as 'eclipse
    # synthesise' makes it, so it is reused while it holds the same spectra.
    if config is not None and words in (["--config", config], [f"--config={config}"]):
        from .run_file import run_all

        run_all(config, "synthesise")
        return
    run_synthesis(parse_arguments(words))


if __name__ == "__main__":
    main()
