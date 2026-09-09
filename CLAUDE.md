# ECLIPSE

## Releases

Every merge into `master` gets a new GitHub release. Treat it as part of the
merge, not as a separate job to be picked up later.

Versions follow `vMAJOR.MINOR.PATCH`, for example `v1.1.1`. Propose the number
rather than picking one silently: patch for fixes and documentation, minor for
new features, major for breaking changes. Confirm it before tagging.

1. Bump `__version__` in `euvst_response/__init__.py`. `pyproject.toml` reads it
   with `version = {attr = "euvst_response.__version__"}`, so that is the only
   place the version is written.
2. Commit the bump, tag `vX.Y.Z`, push the tag.
3. `gh release create vX.Y.Z` with notes covering what merged.

The repository has a Zenodo webhook on `release` events, so every release mints
a new Zenodo deposition and its own DOI. Releases are not free - do not create
throwaway ones to test something.
