# Contributing

## Instrument settings

Each instrument setting in `src/eclipse/config.py` gives the source of its default in its field metadata, as `source`. A setting that is a quantity also gives `unit`, the unit to show it in, such as `micron`. A plain number or a path takes no unit.

`docs/instrument-sources.md` lists the settings in tables. Each of these tables follows a comment line that names its instrument, `<!-- SWC settings: ... -->` or `<!-- EIS settings: ... -->`, with a blank line before and after it. In those tables, a row whose first cell names settings in backticks is a setting's row. Other rows, such as the dark current laws, are written by hand.

`python tools/update_instrument_sources.py` writes the Source cell of every setting's row. It also writes the Default cell of a row that names one setting when the cell is empty or holds only a value, such as `65535 DN` or `none`. Other Default cells are worded by hand. So are the rows of the settings in `NOT_SHOWN` in the script, which give design values or a description instead of the default. The script lists the hand-written cells that are empty or no longer show the code's defaults, and you edit those yourself. A list of zeros, such as no slit image tilt, reads as `none`.

To add a setting, give its field a `source`, and a `unit` if it is a quantity. Then add a row for it to a table of its instrument, with its key in backticks in the first cell, and run the script. For a row with one setting, you can leave the Default cell empty for the script to fill. For a row with several settings, or a default that needs words, write the Default cell by hand, starting with the defaults in the order of the settings. A setting that is not a fact about the instrument, such as a choice of a run, goes in `WITHOUT_SOURCE` in `tests/test_instrument_sources.py`, with the reason.

After changing a default or a source, run the script, and edit any Default cell it lists. The tests fail until the page agrees with the code.

## Releases

### Release when

- A bug fix changes output for the same input.
- A new feature or configuration key lands.
- The API or the configuration format changes.
- On demand, before a paper submission, or when downstream work needs a citable version to pin.

### Batch into the next release

- Documentation, CI, tests.
- Refactors that produce identical output.

### Version numbers

Three components, `vMAJOR.MINOR.PATCH`. Avoid four-component tags such as `v0.6.1.4` if possible.

- `patch` - fixes with no API change.
- `minor` - new features, new configuration keys, new extras; backwards compatible.
- `major` - breaking API or configuration changes.

Whatever the size of the bump, if output changes for identical input, say so prominently in the release notes.

### Making a release

1. Bump `__version__` in `src/eclipse/__init__.py`. `pyproject.toml` reads it with `version = {attr = "eclipse.__version__"}`, so that is the only place the version is written. `master` takes pull requests only, so this goes through one like anything else.
2. Tag `vX.Y.Z` on the merged bump, and push the tag.
3. `gh release create vX.Y.Z` with notes covering what merged, leading with any change to numerical output.

Publishing the release runs `.github/workflows/publish.yml`, which builds the sdists and wheels of `eclipse` and `solarc-eclipse` and uploads them to PyPI. Nothing is published by hand. The workflow refuses to upload if the tag disagrees with `__version__`, or if the wheel is missing the `eclipse/data` files, because a PyPI version cannot be replaced once it exists.

`legacy/solarc-eclipse` is `solarc-eclipse`, ECLIPSE's former name on PyPI. The workflow publishes it with every release, at the same version, and each release installs the `eclipse` of that version. It also holds the former package name `euvst_response` and the commands, so that upgrading an older installation keeps them. Keep its commands the same as those of `eclipse`: `tests/test_distributions.py` checks this.

Read the Docs builds its `stable` version from the latest tag, so a release is also what moves the published documentation forward.

PyPI authenticates the workflow by trusted publishing rather than an API token, configured against the repository, the `publish.yml` workflow and the `pypi` environment, on both the `eclipse` and `solarc-eclipse` projects.
