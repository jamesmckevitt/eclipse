"""
Setup script for M-ECLIPSES package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="m-eclipses",
    version="1.0.0",
    author="James McKevitt",
    author_email="jm2@mssl.ucl.ac.uk",
    description="MSSL Emission Calculation and Line Intensity Prediction for SOLAR-C EUVST-SW",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jamesmckevitt/solc_euvst_sw_response",
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
            "m-eclipses=euvst_response.cli:main",
            "meclipses=euvst_response.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "euvst_response": ["data/**/*"],
    },
    zip_safe=False,
)
