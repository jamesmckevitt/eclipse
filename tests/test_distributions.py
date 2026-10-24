"""Installing `eclipse` or `solarc-eclipse` over an older `solarc-eclipse`.

ECLIPSE was on PyPI as `solarc-eclipse` until 0.13.0, and is now `eclipse`.
`solarc-eclipse` is kept as a package that installs `eclipse`. When pip
upgrades a package it deletes the files the old version installed, even ones
that another package has just installed. These tests check that the former
package name `euvst_response` and the commands survive both ways of moving over.
"""
import base64
import hashlib
import subprocess
import sys
import tomllib
import venv
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy" / "solarc-eclipse"
SHIM = ROOT / "src" / "euvst_response" / "__init__.py"


def _project(folder):
    return tomllib.loads((folder / "pyproject.toml").read_text())["project"]


def _wheel(folder, name, version, files, scripts, requires=()):
    """Write a wheel for `name` holding `files`, a dict of path to text."""
    tag = name.replace("-", "_")
    info = f"{tag}-{version}.dist-info"
    contents = dict(files)
    contents[f"{info}/METADATA"] = "\n".join(
        ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
        + [f"Requires-Dist: {r}" for r in requires]) + "\n"
    contents[f"{info}/WHEEL"] = ("Wheel-Version: 1.0\nGenerator: test\n"
                                 "Root-Is-Purelib: true\nTag: py3-none-any\n")
    contents[f"{info}/entry_points.txt"] = "[console_scripts]\n" + "".join(
        f"{command} = {target}\n" for command, target in scripts.items())
    record = []
    for path, text in contents.items():
        digest = hashlib.sha256(text.encode()).digest()
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        record.append(f"{path},sha256={encoded},{len(text.encode())}")
    contents[f"{info}/RECORD"] = "\n".join(record + [f"{info}/RECORD,,"]) + "\n"
    path = folder / f"{tag}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name_in_wheel, text in contents.items():
            archive.writestr(name_in_wheel, text)
    return path


@pytest.fixture
def old_install(tmp_path):
    """A virtual environment holding an older `solarc-eclipse`, and the new wheels."""
    scripts = _project(ROOT)["scripts"]
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    old = _wheel(tmp_path, "solarc-eclipse", "0.12.1",
                 {"euvst_response/__init__.py": "OLD = True\n",
                  "euvst_response/frame.py": "OLD = True\n"}, scripts)
    _wheel(wheels, "eclipse", "0.13.0",
           {"eclipse/__init__.py": "", "euvst_response/__init__.py": SHIM.read_text()},
           scripts)
    legacy = _project(LEGACY)
    _wheel(wheels, legacy["name"], legacy["version"],
           {"euvst_response/__init__.py": (LEGACY / "euvst_response" / "__init__.py").read_text()},
           legacy["scripts"], legacy["dependencies"])

    environment = tmp_path / "venv"
    venv.create(environment, with_pip=True)
    python = environment / "bin" / "python"
    _pip(python, old)
    return python, wheels


def _pip(python, *args):
    subprocess.run([python, "-m", "pip", "install", "--no-index", "--quiet",
                    "--disable-pip-version-check", *map(str, args)], check=True)


def _site_packages(python):
    return Path(subprocess.run(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True).stdout.strip())


def _check(python):
    installed = _site_packages(python) / "euvst_response" / "__init__.py"
    assert installed.read_text() == SHIM.read_text()
    for command in _project(ROOT)["scripts"]:
        assert (python.parent / command).exists(), command


@pytest.mark.skipif(sys.platform == "win32", reason="checks the scripts folder by its Unix name")
def test_upgrading_solarc_eclipse_keeps_the_old_name_and_the_commands(old_install):
    python, wheels = old_install
    _pip(python, "--find-links", wheels, "--upgrade", "solarc-eclipse")
    _check(python)


@pytest.mark.skipif(sys.platform == "win32", reason="checks the scripts folder by its Unix name")
def test_installing_eclipse_over_solarc_eclipse_gives_the_new_old_name(old_install):
    python, wheels = old_install
    _pip(python, "--find-links", wheels, "eclipse")
    _check(python)


def test_solarc_eclipse_installs_eclipse_with_the_same_commands_and_old_name():
    legacy = _project(LEGACY)
    main = _project(ROOT)
    assert legacy["scripts"] == main["scripts"]
    assert legacy["dependencies"] == [f"eclipse>={legacy['version']}"]
    assert (LEGACY / "euvst_response").resolve() == SHIM.parent
    assert (LEGACY / "LICENSE").resolve() == ROOT / "LICENSE"
