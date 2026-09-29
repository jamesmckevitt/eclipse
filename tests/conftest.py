"""Settings every test runs with."""
import os
from pathlib import Path

import matplotlib
import pytest

# The plotting tests draw without a display, whatever backend the
# environment asks for, since the package imports pyplot.
matplotlib.use("Agg", force=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "chianti: needs fiasco's CHIANTI database")


def _chianti_database():
    """Where fiasco reads the CHIANTI database, if it has been built there."""
    import fiasco

    path = Path(fiasco.defaults["hdf5_dbase_root"])
    return path if path.is_file() else None


def pytest_collection_modifyitems(config, items):
    """
    Skip the tests marked ``chianti`` where the database is not built, as in
    the main CI jobs, unless ECLIPSE_REQUIRE_CHIANTI is set, as in the job
    that builds it, where its absence is an error rather than a skip.
    """
    marked = [item for item in items if item.get_closest_marker("chianti")]
    if not marked or _chianti_database() is not None:
        return
    if os.environ.get("ECLIPSE_REQUIRE_CHIANTI"):
        import fiasco

        raise pytest.UsageError(
            f"ECLIPSE_REQUIRE_CHIANTI is set, but fiasco has no CHIANTI database at "
            f"{fiasco.defaults['hdf5_dbase_root']}.")
    skip = pytest.mark.skip(reason="needs fiasco's CHIANTI database")
    for item in marked:
        item.add_marker(skip)
