"""
Check the packages built into dist/ before they are uploaded to PyPI.

Run it as `python tools/check_packages.py` after building them, or with
`--tag vX.Y.Z` to check `eclipse` has the version of that tag too. dist/ must
hold the sdist and wheel of `eclipse`, and may hold the sdist of
`solarc-eclipse`, the former name, whose install must stop with a message. The
wheel is read rather than the package imported, so this needs no runtime
dependency. A PyPI version cannot be replaced once it is uploaded, so the
release workflow runs this before uploading anything.
"""

import argparse
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
LEGACY = ROOT / "legacy" / "solarc-eclipse"


def _version(pattern, path):
    match = re.search(pattern, path.read_text(), re.MULTILINE)
    return None if match is None else match["version"]


def check(dist, tag=None):
    """Errors found in the packages in `dist`, as a list of messages."""
    if not dist.is_dir():
        return [f"no folder {dist}"]
    source = _version(r'^__version__ = ([\'"])(?P<version>.+?)\1', SOURCE / "eclipse" / "__init__.py")
    legacy = _version(r'^version = "(?P<version>.+?)"', LEGACY / "pyproject.toml")
    if source is None:
        return ["no __version__ found in src/eclipse/__init__.py"]
    if legacy is None:
        return [f"no version found in {LEGACY / 'pyproject.toml'}"]
    version = source if tag is None else tag.removeprefix("v")

    # Everything in dist/ is uploaded, so nothing else may be there.
    wheel = f"eclipse-{version}-py3-none-any.whl"
    stub = f"solarc_eclipse-{legacy}.tar.gz"
    expected = sorted([wheel, f"eclipse-{version}.tar.gz"])
    present = sorted(path.name for path in dist.iterdir())
    if sorted(set(present) - {stub}) != expected:
        return [f"expected {expected} and perhaps {stub} in {dist}, found {present}"]

    # Compared against the source tree rather than a list, so that this keeps
    # working when files are added, and a partial package fails.
    data = {path for path in (SOURCE / "eclipse" / "data").rglob("*") if path.is_file()}
    if not data:
        return ["no src/eclipse/data files in the checkout"]
    if not (SOURCE / "euvst_response" / "__init__.py").is_file():
        return ["no src/euvst_response/__init__.py in the checkout"]
    modules = set((SOURCE / "eclipse").rglob("*.py")) | set((SOURCE / "euvst_response").rglob("*.py"))
    files = {path.relative_to(SOURCE).as_posix() for path in data | modules}
    missing = sorted(files - set(zipfile.ZipFile(dist / wheel).namelist()))
    if missing:
        return [f"{len(missing)} of {len(files)} package files are absent from "
                f"{wheel}: {', '.join(missing)}"]
    print(f"all {len(files)} package files are in {wheel}")

    if stub in present:
        env = {key: value for key, value in os.environ.items() if key != "SOLARC_ECLIPSE_BUILD"}
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "--no-deps",
             "--disable-pip-version-check", str(dist / stub)],
            capture_output=True, text=True, env=env)
        if result.returncode == 0 or "pip install eclipse" not in result.stdout + result.stderr:
            print(result.stdout + result.stderr)
            return [f"installing {stub} does not stop with the message to install eclipse; "
                    "pip's output is above"]
        print(f"installing {stub} stops with the message to install eclipse")
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dist", type=Path, default=ROOT / "dist",
                        help="folder holding the packages (default: dist)")
    parser.add_argument("--tag", help="release tag the version of eclipse must match")
    args = parser.parse_args()
    errors = check(args.dist, args.tag)
    for error in errors:
        print(f"::error::{error}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
