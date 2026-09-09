# Contributing

## Releases

Not every merge needs a release. Two things make releases costly here: the
repository has a Zenodo webhook on `release` events, so each one mints a
permanent citable DOI, and PyPI version numbers can never be reused.

The question that decides it is whether the change alters the numbers.

### Release when

- A bug fix changes output for the same input.
- A new feature or configuration key lands. Users cannot use what is not
  released.
- The API or the configuration format changes.
- On demand, before a paper submission, or when downstream work needs a citable
  version to pin. This is the easiest one to forget and the one that matters
  most for reproducibility.

### Batch into the next release

- Documentation, CI, tests.
- Refactors that produce identical output.

### Version numbers

Three components, `vMAJOR.MINOR.PATCH`. Avoid four-component tags such as
`v0.6.1.4`: they are not semantic versions and sort unpredictably.

- `patch` - fixes with no API change.
- `minor` - new features, new configuration keys, new extras; backwards
  compatible.
- `major` - breaking API or configuration changes.

Whatever the size of the bump, if output changes for identical input, say so
prominently in the release notes. That boundary is what tells someone their
earlier results are not comparable with later ones.

### Cutting a release

1. Bump `__version__` in `euvst_response/__init__.py`. `pyproject.toml` reads it
   with `version = {attr = "euvst_response.__version__"}`, so that is the only
   place the version is written.
2. Commit the bump, tag `vX.Y.Z`, push the tag.
3. `gh release create vX.Y.Z` with notes covering what merged, leading with any
   change to numerical output.

Read the Docs builds its `stable` version from the latest tag, so a release is
also what moves the published documentation forward.
