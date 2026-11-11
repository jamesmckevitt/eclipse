#!/usr/bin/env python3
"""
Command line interface for synthesis script.
"""

import sys


def main():
    """Entry point for the synthesis command line script."""
    from .synthesis import _config_option, main as run_synthesis, parse_arguments

    words = sys.argv[1:]
    args = parse_arguments(words)
    run_synthesis(args)
    # Given only a run file, the synthesis is the run file's, so later runs
    # of it can reuse it.
    config = _config_option(words)
    if config is not None and words in (["--config", config], [f"--config={config}"]):
        from .run_file import _run_key, _store_key, load, runs, synthesis_path

        name, settings = runs(*load(config)[:2])[0]
        _store_key(synthesis_path(settings, name), _run_key(settings))


if __name__ == "__main__":
    main()
