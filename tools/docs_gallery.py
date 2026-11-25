"""
Run the docs gallery's scripts and save their figures.

Run from the repository with `python tools/docs_gallery.py`, or give the
scripts to run, such as `python tools/docs_gallery.py dark_current`.
"""

import contextlib
import pathlib
import runpy
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GALLERY = pathlib.Path(__file__).resolve().parents[1] / "docs" / "gallery"


def main(names: list[str]) -> None:
    # A name may be given as `dark_current`, `dark_current.py` or `dark-current`.
    scripts = ([GALLERY / (pathlib.Path(name).stem.replace("-", "_") + ".py") for name in names]
               or sorted(GALLERY.glob("*.py")))
    for script in scripts:
        plt.close("all")
        # Some scripts write files, which go in a scratch folder.
        with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(scratch):
            runpy.run_path(str(script), run_name="__main__")
        plt.gcf().savefig(GALLERY / "images" / f"{script.stem}.png", dpi=150)
        print(f"{script.stem}: saved")


if __name__ == "__main__":
    main(sys.argv[1:])
