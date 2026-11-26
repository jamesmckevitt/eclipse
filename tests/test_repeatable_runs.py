"""A seeded run gives the same fits each time, whatever Python's hash seed or the workers."""
import os
import subprocess
import sys
from pathlib import Path

import eclipse

RUN = """
import hashlib
import sys

import numpy as np

import eclipse

config = {"uniform_intensity": "5000 erg / (s cm2 sr)", "rest_wavelength": "195.119 AA",
          "thermal_width": "20 km/s", "n_iter": 4, "ncpu": int(sys.argv[1]),
          "offchip_bin_slit": [1, 2], "simulation": {"psf": [False, True], "expos": "20 s"}}
digest = hashlib.sha1()
hashed = 0


def add(name, value):
    # Every number in the fit statistics, however deeply they are nested.
    global hashed
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            add(f"{name}/{key}", value[key])
        return
    try:
        array = np.asarray(getattr(value, "value", value), dtype=float)
    except (TypeError, ValueError):
        return
    digest.update(name.encode() + np.ascontiguousarray(array).tobytes())
    hashed += 1


np.random.seed(5)
[output] = eclipse.observe(config, name="seeded")
results = eclipse.load_instrument_response_results(output.results)
for settings in eclipse.get_parameter_combinations(results):
    chosen = {key: settings[key] for key in ("offchip_bin_slit", "simulation.psf")}
    combination = eclipse.get_results_for_combination(results, **chosen)
    for key in ("dn_fit_stats", "photon_fit_stats"):
        add(f"{chosen}/{key}", combination[key])
print(hashed, digest.hexdigest())
"""


def test_a_seeded_run_repeats_in_new_processes(tmp_path):
    """The fits of a uniform line, binned and not, with and without the PSF, match exactly,
    from one worker with one hash seed and from two with another."""
    source = str(Path(eclipse.__file__).resolve().parents[1])
    outputs = []
    for hash_seed, ncpu in (("0", "1"), ("1", "2")):
        folder = tmp_path / hash_seed
        folder.mkdir()
        run = subprocess.run(
            [sys.executable, "-c", RUN, ncpu], cwd=folder, capture_output=True, text=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed,
                 "PYTHONPATH": os.pathsep.join(filter(None, (source,
                                                             os.environ.get("PYTHONPATH"))))})
        assert run.returncode == 0, run.stderr
        outputs.append(run.stdout.strip().splitlines()[-1].split())
    hashed = int(outputs[0][0])
    assert hashed > 50
    assert outputs[0] == outputs[1]
