"""
Check the wheels of `eclipse` and `solarc-eclipse` built into dist/.

Run it as `python tools/check_packages.py`, after building both packages, or
with `--tag vX.Y.Z` to check they have the version of that tag too. The wheels
are read rather than the package imported, so it needs no runtime dependency.
A PyPI version cannot be replaced once it is uploaded, so the release workflow
runs this before uploading anything.
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


def _version(archive):
    metadata = next(name for name in archive.namelist()
                    if name.endswith(".dist-info/METADATA"))
    return next(line.split(": ", 1)[1].strip()
                for line in archive.read(metadata).decode().splitlines()
                if line.startswith("Version: "))


def check(dist, tag=None):
    """Errors found in the wheels in `dist`, as a list of messages."""
    errors = []
    # Everything in dist/ is uploaded, so nothing else may be there.
    present = sorted(path.name for path in dist.iterdir())
    wheels = {name: sorted(dist.glob(f"{name}-*.whl")) for name in ("eclipse", "solarc_eclipse")}
    sdists = sorted(dist.glob("*.tar.gz"))
    if [len(w) for w in wheels.values()] != [1, 1] or len(sdists) != 2 or len(present) != 4:
        return [f"expected one sdist and one wheel of each package in {dist}, found {present}"]
    eclipse = zipfile.ZipFile(wheels["eclipse"][0])
    legacy = zipfile.ZipFile(wheels["solarc_eclipse"][0])

    source = re.search(r'^__version__ = ([\'"])(.+?)\1',
                       (SOURCE / "eclipse" / "__init__.py").read_text(), re.MULTILINE)[2]
    expected = source if tag is None else tag.removeprefix("v")
    for archive in (eclipse, legacy):
        version = _version(archive)
        print(f"{archive.filename}: {version}")
        if version != expected:
            errors.append(f"{archive.filename} has version {version}, not {expected}")

    # Compared against the source tree rather than a list, so that this keeps
    # working when files are added, and a partial package fails.
    data = {path for path in (SOURCE / "eclipse" / "data").rglob("*") if path.is_file()}
    if not data:
        errors.append("no src/eclipse/data files in the checkout")
    modules = set((SOURCE / "eclipse").rglob("*.py")) | set((SOURCE / "euvst_response").rglob("*.py"))
    files = {path.relative_to(SOURCE).as_posix() for path in data | modules}
    missing = sorted(files - set(eclipse.namelist()))
    if missing:
        errors.append(f"{len(missing)} of {len(files)} package files are absent from "
                      f"{eclipse.filename}: {', '.join(missing)}")
    else:
        print(f"all {len(files)} package files are in {eclipse.filename}")

    for path in sorted((SOURCE / "euvst_response").rglob("*.py")):
        name = path.relative_to(SOURCE).as_posix()
        if name not in legacy.namelist():
            errors.append(f"{legacy.filename} has no {name}")
        elif legacy.read(name) != path.read_bytes():
            errors.append(f"{legacy.filename} holds a different {name}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dist", type=Path, default=ROOT / "dist",
                        help="folder holding the wheels (default: dist)")
    parser.add_argument("--tag", help="release tag the versions must match")
    args = parser.parse_args()
    errors = check(args.dist, args.tag)
    for error in errors:
        print(f"::error::{error}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
