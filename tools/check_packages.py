"""
Check the packages built into dist/ before they are uploaded to PyPI.

Run it as `python tools/check_packages.py` after building them, or with
`--tag vX.Y.Z` to check `eclipse` has the version of that tag too. dist/ must
hold the sdist and wheel of `eclipse` and the sdist of `solarc-eclipse`, the
former name. The wheel is read rather than the package imported, so this needs
no runtime dependency. A PyPI version cannot be replaced once it is uploaded,
so the release workflow runs this before uploading anything.
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


def check(dist, tag=None):
    """Errors found in the packages in `dist`, as a list of messages."""
    source = re.search(r'^__version__ = ([\'"])(.+?)\1',
                       (SOURCE / "eclipse" / "__init__.py").read_text(), re.MULTILINE)[2]
    version = source if tag is None else tag.removeprefix("v")
    legacy = re.search(r'^version = "(.+?)"',
                       (ROOT / "legacy" / "solarc-eclipse" / "pyproject.toml").read_text(),
                       re.MULTILINE)[1]

    # Everything in dist/ is uploaded, so nothing else may be there.
    expected = sorted([f"eclipse-{version}-py3-none-any.whl", f"eclipse-{version}.tar.gz",
                       f"solarc_eclipse-{legacy}.tar.gz"])
    present = sorted(path.name for path in dist.iterdir())
    if present != expected:
        return [f"expected {expected} in {dist}, found {present}"]

    # Compared against the source tree rather than a list, so that this keeps
    # working when files are added, and a partial package fails.
    wheel = dist / f"eclipse-{version}-py3-none-any.whl"
    data = {path for path in (SOURCE / "eclipse" / "data").rglob("*") if path.is_file()}
    if not data:
        return ["no src/eclipse/data files in the checkout"]
    modules = set((SOURCE / "eclipse").rglob("*.py")) | set((SOURCE / "euvst_response").rglob("*.py"))
    files = {path.relative_to(SOURCE).as_posix() for path in data | modules}
    missing = sorted(files - set(zipfile.ZipFile(wheel).namelist()))
    if missing:
        return [f"{len(missing)} of {len(files)} package files are absent from "
                f"{wheel.name}: {', '.join(missing)}"]
    print(f"{', '.join(expected)} are in {dist}, with all {len(files)} package files "
          f"in {wheel.name}")
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
