#!/usr/bin/env python3
"""
Command line entry point for `synthesise-spectra`.
"""

import sys


def main():
    """Entry point for `synthesise-spectra`."""
    from .synthesis import _config_option, main as run_synthesis, parse_arguments

    words = sys.argv[1:]
    config = _config_option(words)
    # With only a run file, synthesise as 'eclipse synthesise' does, so the
    # synthesis is reused while its spectra are unchanged.
    if config is not None and words in (["--config", config], [f"--config={config}"]):
        from .run_file import load, run_all, runs

        if len(runs(*load(config)[:2])) != 1:
            raise ValueError(f"{config} has several runs; synthesise them with "
                             f"'eclipse synthesise {config}', which takes --only to pick one.")
        run_all(config, "synthesise")
        return
    run_synthesis(parse_arguments(words))


if __name__ == "__main__":
    main()
