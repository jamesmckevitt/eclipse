"""Stops `pip install solarc-eclipse`, telling the user to install `eclipse`."""
import os

MESSAGE = """
solarc-eclipse is now called eclipse on PyPI. Install it with

    pip install eclipse

ECLIPSE needs Python 3.11 or later.

solarc-eclipse 0.12.1 and earlier can still be installed by giving the version,
such as `pip install solarc-eclipse==0.12.1`.
"""

# Set only by the release workflow, which builds this package's sdist.
if not os.environ.get("SOLARC_ECLIPSE_BUILD"):
    raise SystemExit(MESSAGE)

from setuptools import setup  # noqa: E402

setup()
