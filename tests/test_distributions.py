"""`pip install solarc-eclipse`, the former name on PyPI, says to install `eclipse`."""
import os
import subprocess
import sys
import tomllib
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[1] / "legacy" / "solarc-eclipse"


def test_installing_solarc_eclipse_stops_and_says_to_install_eclipse():
    env = {key: value for key, value in os.environ.items() if key != "SOLARC_ECLIPSE_BUILD"}
    result = subprocess.run([sys.executable, "setup.py", "egg_info"], cwd=LEGACY,
                            capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "    pip install eclipse\n" in result.stderr
    assert "pip install solarc-eclipse==0.12.1" in result.stderr


def test_solarc_eclipse_installs_nothing_and_is_newer_than_its_last_real_release():
    project = tomllib.loads((LEGACY / "pyproject.toml").read_text())
    assert project["tool"]["setuptools"]["packages"] == []
    assert "dependencies" not in project["project"]
    assert "scripts" not in project["project"]
    version = tuple(int(part) for part in project["project"]["version"].split("."))
    assert version > (0, 12, 1)
