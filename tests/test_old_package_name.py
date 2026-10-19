"""The package's former name, euvst_response, still imports, with a warning."""
import pickle
import subprocess
import sys
from pathlib import Path

import eclipse.config
from eclipse import results_file

MESSAGE = "euvst_response package is now called eclipse"


def _run(code):
    return subprocess.run([sys.executable, "-W", "always::FutureWarning", "-c", code],
                          capture_output=True, text=True, check=True)


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


def test_importing_the_new_name_does_not_warn_of_the_old():
    assert MESSAGE not in _run("import eclipse").stderr


def test_a_packaged_table_named_under_the_old_name_is_read_from_this_installation():
    old = (Path("/somewhere/site-packages/euvst_response/data/throughput")
           / "grating_reflection_efficiency.dat")
    assert results_file._jsonable(old) == {
        results_file.TAG: "resource", "value": "data/throughput/grating_reflection_efficiency.dat"}
    telescope = eclipse.config.Telescope_EUVST()
    object.__setattr__(telescope, "grating_table", old)
    results_file._give_earlier_values({"telescope": telescope})
    assert telescope.grating_table.is_file()
    assert results_file._jsonable(Path("/elsewhere/data/table.dat"))[results_file.TAG] == "path"
