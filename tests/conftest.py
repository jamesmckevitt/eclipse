"""Settings every test runs with."""
import os
from pathlib import Path

import matplotlib
import pytest

# Force a backend that needs no display, since the package imports pyplot.
matplotlib.use("Agg", force=True)


def pytest_configure(config):
    config.addinivalue_line("markers", "chianti: needs fiasco's CHIANTI database")


def _chianti_database():
    """Path to fiasco's CHIANTI database, or None if it is not built."""
    import fiasco

    path = Path(fiasco.defaults["hdf5_dbase_root"])
    return path if path.is_file() else None


def pytest_collection_modifyitems(config, items):
    """
    Skip tests marked `chianti` if the database is not built.

    This is the case in the main CI jobs. If ECLIPSE_REQUIRE_CHIANTI is set,
    as in the job that builds the database, a missing database is an error
    instead.
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
