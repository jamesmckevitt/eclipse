# Contributing

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

Three components, `vMAJOR.MINOR.PATCH`. Avoid four-component tags such as
`v0.6.1.4` if possible.

- `patch` - fixes with no API change.
- `minor` - new features, new configuration keys, new extras; backwards compatible.
- `major` - breaking API or configuration changes.

Whatever the size of the bump, if output changes for identical input, say so prominently in the release notes.

### Making a release

1. Bump `__version__` in `euvst_response/__init__.py`. `pyproject.toml` reads it with `version = {attr = "euvst_response.__version__"}`, so that is the only place the version is written. `master` takes pull requests only, so this goes through one like anything else.
2. Tag `vX.Y.Z` on the merged bump, and push the tag.
3. `gh release create vX.Y.Z` with notes covering what merged, leading with any change to numerical output.

Publishing the release runs `.github/workflows/publish.yml`, which builds the sdist and wheel and uploads them to PyPI. Nothing is published by hand. The workflow refuses to upload if the tag disagrees with `__version__`, or if the wheel is missing the `euvst_response/data` files, because a PyPI version cannot be replaced once it exists.

Read the Docs builds its `stable` version from the latest tag, so a release is also what moves the published documentation forward.

PyPI authenticates the workflow by trusted publishing rather than an API token, configured against the repository, the `publish.yml` workflow and the `pypi` environment.
