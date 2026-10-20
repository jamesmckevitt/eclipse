"""The package's former name, euvst_response, still imports, with a warning."""
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import eclipse.config
from eclipse import results_file

MESSAGE = "euvst_response package is now called eclipse"


def _python(*args, cwd=None):
    """Run a new interpreter with `args`, importing the package under test."""
    source = str(Path(eclipse.config.__file__).resolve().parents[1])
    path = os.pathsep.join(filter(None, [source, os.environ.get("PYTHONPATH")]))
    return subprocess.run([sys.executable, "-W", "always::FutureWarning", *args],
                          capture_output=True, text=True, check=True, cwd=cwd,
                          env={**os.environ, "PYTHONPATH": path})


def _run(code, cwd=None):
    """Run `code` in a new interpreter that imports the package under test."""
    return _python("-c", code, cwd=cwd)


def test_the_old_name_warns_gives_the_new_modules_and_loads_old_pickles():
    """One interpreter: the warning, the module aliases, and a pickle naming the old modules."""
    old = pickle.dumps(eclipse.config.Detector_SWC(), protocol=0).replace(
        b"eclipse.config", b"euvst_response.config")
    assert b"euvst_response.config" in old
    result = _run(
        "import importlib, pickle\n"
        "import euvst_response, eclipse\n"
        "from euvst_response.frame import detect\n"
        "import euvst_response.config as old\n"
        "import eclipse.config as new\n"
        "assert old is new and detect is eclipse.frame.detect\n"
        "assert new.__spec__.name == 'eclipse.config'\n"
        "importlib.reload(new)\n"
        "assert euvst_response.Detector_SWC is eclipse.Detector_SWC\n"
        "assert euvst_response.__version__ == eclipse.__version__\n"
        f"det = pickle.loads({old!r})\n"
        "assert type(det).__name__ == 'Detector_SWC'\n"
        "importlib.reload(euvst_response)\n"
        "import sys\n"
        "assert sum(getattr(f, 'euvst_response_alias', False) for f in sys.meta_path) == 1\n")
    assert MESSAGE in result.stderr
    assert "<frozen importlib" not in result.stderr


def test_a_plain_import_warns_at_the_import_line():
    result = _run("x = 1\nimport euvst_response\n")
    assert f"<string>:2: FormerNameWarning: The {MESSAGE}" in result.stderr


def test_importing_the_new_name_does_not_warn_of_the_old():
    assert MESSAGE not in _run("import eclipse").stderr


@pytest.mark.parametrize("name", ["cli", "main", "synthesis_cli", "atmosphere"])
def test_the_commands_run_as_modules_under_either_name(name):
    for package in ("eclipse", "euvst_response"):
        assert "usage:" in _python("-m", f"{package}.{name}", "--help").stdout


def test_a_module_without_a_command_runs_under_the_old_name():
    _python("-m", "euvst_response.science_cases")


def test_the_modules_left_for_python_m_are_those_with_a_main_block():
    """Each of these is left out of `sys.modules` under the old name until imported."""
    source = Path(eclipse.config.__file__).parent
    runnable = {path.stem for path in source.glob("*.py")
                if 'if __name__ == "__main__"' in path.read_text()}
    left = _run("import euvst_response\nprint(sorted(euvst_response._COMMANDS))").stdout
    assert left.strip() == str(sorted(runnable))


@pytest.mark.parametrize("first", ["the module", "the function"])
def test_a_function_named_as_its_module_stays_the_function(first):
    """`monte_carlo` is a function of the package and the module that holds it."""
    imports = ["import euvst_response.monte_carlo", "from euvst_response import monte_carlo"]
    if first == "the function":
        imports.reverse()
    _run("import sys, types\n"
         "import euvst_response\n"
         "assert sys.modules['euvst_response.monte_carlo'] is sys.modules['eclipse.monte_carlo']\n"
         + "\n".join(imports) + "\n"
         "from euvst_response import monte_carlo, main\n"
         "import euvst_response.main\n"
         "assert isinstance(monte_carlo, types.FunctionType)\n"
         "assert euvst_response.monte_carlo is monte_carlo and euvst_response.main is main\n"
         "assert isinstance(main, types.FunctionType)\n")


def test_the_package_data_are_found_under_the_old_name(tmp_path):
    _run("from importlib.resources import files\n"
         "from pathlib import Path\n"
         "import euvst_response\n"
         "table = 'throughput/grating_reflection_efficiency.dat'\n"
         "text = (files('eclipse') / 'data' / table).read_text()\n"
         "assert text and (files('euvst_response') / 'data' / table).read_text() == text\n"
         "assert (Path(euvst_response.__file__).parent / 'data' / table).read_text() == text\n"
         "import sys\n"
         "assert sys.modules['euvst_response.config'] is sys.modules['eclipse.config']\n",
         cwd=tmp_path)


TABLE = Path("data/throughput/grating_reflection_efficiency.dat")


def _tables_read_back(tmp_path, recorded):
    """`grating_table` read back from a results pickle naming `recorded`, and from its conversion."""
    import dill

    telescope = eclipse.config.Telescope_EUVST()
    object.__setattr__(telescope, "grating_table", recorded)
    pickled = tmp_path / "old.pkl"
    with open(pickled, "wb") as handle:
        dill.dump({"results": {"all_combinations": {}, "fit_weighted": True},
                   "telescope": telescope}, handle)
    with pytest.warns(FutureWarning, match="results pickle"):
        from_pickle = results_file.load_results(pickled)["telescope"].grating_table
    converted = results_file.convert_results_pickle(pickled, tmp_path / "new.h5")
    return from_pickle, results_file.load_results(converted)["telescope"].grating_table


def test_a_packaged_table_named_under_the_old_name_is_read_from_this_installation(tmp_path):
    old = Path("/somewhere/site-packages/euvst_response") / TABLE
    installed = results_file._package_root() / TABLE
    assert _tables_read_back(tmp_path, old) == (installed, installed)


def test_a_table_of_ones_own_in_a_folder_named_as_the_old_package_is_kept(tmp_path):
    own = tmp_path / "euvst_response" / TABLE
    own.parent.mkdir(parents=True)
    own.write_bytes((results_file._package_root() / TABLE).read_bytes())
    assert results_file._jsonable(own) == {results_file.TAG: "path", "value": str(own)}
    assert _tables_read_back(tmp_path, own) == (own, own)


class _NamesAnOldModule:
    """Unpickles as a number, through a function of `euvst_response.utils`."""

    def __reduce__(self):
        return eclipse.utils._fwhm_to_sigma, (2.0,)


def _old_pickle(path, payload):
    """Write `payload` as a pickle naming the modules of `euvst_response`."""
    import dill

    text = dill.dumps({**payload, "old": _NamesAnOldModule()}, protocol=2)
    path.write_bytes(text.replace(b"eclipse.utils\n", b"euvst_response.utils\n"))
    assert b"euvst_response.utils" in path.read_bytes()
    return path


@pytest.mark.parametrize("read", [
    "from eclipse.analysis import load_instrument_response_results as read\n"
    "read(results)\n",
    "from eclipse.synthesis_file import convert_synthesis_pickle\n"
    "convert_synthesis_pickle(synthesis, synthesis.with_suffix('.h5'))\n",
    "from eclipse.data_processing import load_atmosphere\n"
    "load_atmosphere(synthesis)\n",
])
def test_an_old_pickle_is_read_without_the_warning_to_change_an_import(tmp_path, read):
    from test_synthesis_file import _line_cubes

    results = _old_pickle(tmp_path / "results.pkl",
                          {"results": {"all_combinations": {}, "fit_weighted": True}})
    synthesis = _old_pickle(tmp_path / "synthesis.pkl", {"line_cubes": _line_cubes()})
    result = _run("import sys\n"
                  "from pathlib import Path\n"
                  f"results, synthesis = Path({str(results)!r}), Path({str(synthesis)!r})\n"
                  + read +
                  "assert 'euvst_response.utils' in sys.modules\n")
    assert MESSAGE not in result.stderr
