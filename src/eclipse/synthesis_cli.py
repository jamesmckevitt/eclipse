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
    # Given only a run file, the synthesis is the run file's, made as a run
    # makes it, so it is reused while it holds the same spectra.
    if config is not None and words in (["--config", config], [f"--config={config}"]):
        from .run_file import _synthesise_one, load, runs

        found = runs(*load(config)[:2])
        if len(found) != 1:
            raise ValueError(f"{config} has several runs; synthesise them with "
                             f"'eclipse synthesise {config}', which takes --only to pick one.")
        _synthesise_one(found[0][1], found[0][0])
        return
    run_synthesis(parse_arguments(words))


if __name__ == "__main__":
    main()
