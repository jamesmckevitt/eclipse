"""Gives `solarc-eclipse` the version of `eclipse`, and installs that version."""
import re
from pathlib import Path

from setuptools import setup

source = (Path(__file__).parent / "eclipse_init.py").read_text()
version = re.search(r'^__version__ = "([^"]+)"', source, re.MULTILINE).group(1)

setup(version=version, install_requires=[f"eclipse=={version}"])
