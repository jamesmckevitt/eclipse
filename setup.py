"""
Setup script for ECLIPSE package.
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Get version from __init__.py
def get_version():
    """Extract version from __init__.py file."""
    init_file = this_directory / "euvst_response" / "__init__.py"
    content = init_file.read_text()
    match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name="solarc-eclipse",
    version=get_version(),
    author="James McKevitt",
    author_email="jm2@mssl.ucl.ac.uk",
    description="ECLIPSE: Emission Calculation and Line Prediction for SOLAR-C EUVST",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jamesmckevitt/eclipse",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "astropy",
        "ndcube",
        "specutils",
        "scipy",
        "matplotlib",
        "joblib",
        "tqdm",
        "dill",
        "pyyaml",
        "reproject",
        "dask",
        "psutil",
        "mendeleev",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "eclipse=euvst_response.cli:main",
            "solarc-eclipse=euvst_response.cli:main",
            "synthesise-spectra=euvst_response.synthesis_cli:main",
            "synthesise_spectra=euvst_response.synthesis_cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "euvst_response": ["data/**/*"],
    },
    zip_safe=False,
)
