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

1. Bump `__version__` in `euvst_response/__init__.py`. `pyproject.toml` reads it with `version = {attr = "euvst_response.__version__"}`, so that is the only place the version is written.
2. Commit the bump, tag `vX.Y.Z`, push the tag.
3. `gh release create vX.Y.Z` with notes covering what merged, leading with any change to numerical output.

Read the Docs builds its `stable` version from the latest tag, so a release is also what moves the published documentation forward.
