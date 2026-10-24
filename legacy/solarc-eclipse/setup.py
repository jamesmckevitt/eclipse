"""Gives `solarc-eclipse` the version of `eclipse`, and installs that version or later."""
import re
from pathlib import Path

from setuptools import setup

source = (Path(__file__).parent / "eclipse_init.py").read_text()
match = re.search(r'^__version__ = ([\'"])(.+?)\1', source, re.MULTILINE)
if match is None:
    raise RuntimeError("Unable to find the version of eclipse in eclipse_init.py.")
version = match.group(2)

setup(version=version, install_requires=[f"eclipse>={version}"])
