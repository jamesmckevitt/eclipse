"""Every script in the docs gallery runs, and has a page, a figure and a card."""
import functools
import re
import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import eclipse

GALLERY = Path(__file__).resolve().parents[1] / "docs" / "gallery"
SCRIPTS = sorted(GALLERY.glob("*.py"))


def test_the_gallery_has_scripts():
    assert SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_a_gallery_script_has_a_page_a_figure_and_a_card(script):
    page = script.stem.replace("_", "-") + ".md"
    text = (GALLERY / page).read_text()
    assert f'--8<-- "docs/gallery/{script.name}"' in text
    assert (GALLERY / "images" / f"{script.stem}.png").is_file()
    # The card shows the page's figure, with the same text, and its title.
    title = text.splitlines()[0].removeprefix("# ")
    figure = re.search(rf"!\[[^\]]+\]\(images/{script.stem}\.png\)", text)
    assert figure
    index = (GALLERY / "index.md").read_text()
    assert f"[{figure.group()}]({page})" in index
    assert f"[{title}]({page})" in index


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_a_gallery_script_runs_and_draws_a_figure(script, monkeypatch, tmp_path):
    # Two iterations, fitted in this process, show the script runs; the
    # figures are made with the scripts' own settings by tools/docs_gallery.py.
    monte_carlo = eclipse.monte_carlo

    def quick(*args, n_iter=5, **kwargs):
        return monte_carlo(*args, n_iter=min(n_iter, 2), **kwargs)

    monkeypatch.setattr(eclipse, "monte_carlo", quick)
    monkeypatch.setattr(eclipse, "Simulation", functools.partial(eclipse.Simulation, ncpu=1))
    monkeypatch.setattr(eclipse, "observe", functools.partial(eclipse.observe, n_iter=2, ncpu=1))
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    # Some scripts write files, which go in a scratch folder.
    monkeypatch.chdir(tmp_path)
    plt.close("all")
    runpy.run_path(str(script), run_name="__main__")
    # tools/docs_gallery.py saves one figure for each script.
    assert len(plt.get_fignums()) == 1
    plt.close("all")
