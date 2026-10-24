"""Installing `eclipse` or `solarc-eclipse` over an older `solarc-eclipse`.

ECLIPSE 0.12 and earlier were on PyPI as `solarc-eclipse`, and it is now
`eclipse`. `solarc-eclipse` is kept as a package that installs `eclipse`. When
pip upgrades a package it deletes the files the old version installed, even
ones that another package has just installed. These tests check that the former
package name `euvst_response` and the commands survive both ways of moving over.
"""
import base64
import hashlib
import re
import shutil
import subprocess
import sys
import tomllib
import venv
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="needs the symlinks in legacy/solarc-eclipse")
pytest.importorskip("ensurepip")

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy" / "solarc-eclipse"
SHIM = ROOT / "src" / "euvst_response" / "__init__.py"
VERSION = re.search(r'^__version__ = ([\'"])(.+?)\1',
                    (ROOT / "src" / "eclipse" / "__init__.py").read_text(), re.MULTILINE)[2]
SCRIPTS = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]


def _wheel(folder, name, version, files):
    """Write a wheel for `name` holding `files`, a dict of path to text, and the commands."""
    tag = name.replace("-", "_")
    info = f"{tag}-{version}.dist-info"
    contents = dict(files)
    contents[f"{info}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    contents[f"{info}/WHEEL"] = ("Wheel-Version: 1.0\nGenerator: test\n"
                                 "Root-Is-Purelib: true\nTag: py3-none-any\n")
    contents[f"{info}/entry_points.txt"] = "[console_scripts]\n" + "".join(
        f"{command} = {target}\n" for command, target in SCRIPTS.items())
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


@pytest.fixture(scope="module")
def wheels(tmp_path_factory):
    """The built `solarc-eclipse`, and an `eclipse` holding only what it shares with it."""
    folder = tmp_path_factory.mktemp("wheels")
    # Built from a copy, so that the build leaves nothing in the checkout.
    source = tmp_path_factory.mktemp("legacy") / "solarc-eclipse"
    shutil.copytree(LEGACY, source)
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--quiet",
                    "--disable-pip-version-check", "--wheel-dir", str(folder), str(source)],
                   check=True)
    _wheel(folder, "eclipse", VERSION,
           {"eclipse/__init__.py": "", "euvst_response/__init__.py": SHIM.read_text()})
    return folder


@pytest.fixture
def old_install(tmp_path):
    """A virtual environment holding an older `solarc-eclipse`."""
    old = _wheel(tmp_path, "solarc-eclipse", "0.12.0",
                 {"euvst_response/__init__.py": "OLD = True\n",
                  "euvst_response/frame.py": "OLD = True\n"})
    venv.create(tmp_path / "venv", with_pip=True)
    python = tmp_path / "venv" / "bin" / "python"
    _pip(python, old)
    return python


def _pip(python, *args):
    subprocess.run([python, "-m", "pip", "install", "--no-index", "--quiet",
                    "--disable-pip-version-check", *map(str, args)], check=True)


def _installed_shim(python):
    purelib = subprocess.run(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, check=True).stdout.strip()
    return (Path(purelib) / "euvst_response" / "__init__.py").read_text()


def test_upgrading_solarc_eclipse_keeps_the_old_name_and_the_commands(old_install, wheels):
    _pip(old_install, "--find-links", wheels, "--upgrade", "solarc-eclipse")
    assert _installed_shim(old_install) == SHIM.read_text()
    for command in SCRIPTS:
        assert (old_install.parent / command).exists(), command


def test_installing_eclipse_over_solarc_eclipse_gives_the_new_old_name(old_install, wheels):
    _pip(old_install, "--find-links", wheels, "eclipse")
    assert _installed_shim(old_install) == SHIM.read_text()


def test_solarc_eclipse_installs_eclipse_with_the_same_commands_and_old_name(wheels):
    archive = zipfile.ZipFile(next(wheels.glob("solarc_eclipse-*.whl")))
    info = f"solarc_eclipse-{VERSION}.dist-info"
    metadata = archive.read(f"{info}/METADATA").decode().splitlines()
    assert f"Version: {VERSION}" in metadata
    assert f"Requires-Dist: eclipse>={VERSION}" in metadata
    assert archive.read("euvst_response/__init__.py").decode() == SHIM.read_text()
    commands = archive.read(f"{info}/entry_points.txt").decode()
    assert dict(re.findall(r"^(\S+) = (\S+)$", commands, re.MULTILINE)) == SCRIPTS
