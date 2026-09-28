"""What each MPI rank draws, runs on, and does when one fails.

Seeded alike, as the docs' recipe for comparing runs has it, every rank drew
the same numbers, so eight iterations on four ranks were two iterations four
times. Every rank was pinned to CPUs 0 to n, the same ones, which on a node
shared with another job need not be the job's. A rank that failed left the
others waiting in the gather until the job's time ran out, and a rank other
than the first failed without a word, its output being silenced.
"""
import importlib
import sys

import numpy as np
import pytest

from euvst_response import cli
from euvst_response.main import _cpu_list, _rank_cpus

# The package's own monte_carlo is the function, not the module.
monte_carlo = importlib.import_module("euvst_response.monte_carlo")


def test_each_rank_draws_its_own_numbers_and_again_the_same_ones():
    draws = {}
    for rank in (0, 1, 0):
        np.random.seed(1234)
        monte_carlo._own_stream(rank)
        draws.setdefault(rank, []).append(np.random.random(5))
    assert not np.array_equal(draws[0][0], draws[1][0])
    assert np.array_equal(draws[0][0], draws[0][1])


def test_a_cpu_list_is_read_as_linux_writes_it():
    assert _cpu_list("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]


@pytest.mark.parametrize("environ, cpus", [
    ({"SLURM_CPUS_PER_TASK": "4", "SLURM_LOCALID": "0"}, [16, 17, 18, 19]),
    ({"SLURM_CPUS_PER_TASK": "4", "SLURM_LOCALID": "1"}, [20, 21, 22, 23]),
    # Too few for every task to have its own: all of them, still the job's.
    ({"SLURM_CPUS_PER_TASK": "8", "SLURM_LOCALID": "1"}, list(range(16, 24))),
    ({}, []),
])
def test_each_rank_runs_on_its_share_of_the_jobs_cpus(environ, cpus):
    assert _rank_cpus(environ, list(range(16, 24))) == cpus


def test_a_failing_rank_says_so_and_ends_the_run(monkeypatch, capfd):
    aborted = []

    class Comm:
        def Abort(self, code):
            aborted.append(code)
            raise SystemExit(code)

    monkeypatch.setattr("euvst_response.utils._get_mpi_info", lambda: (Comm(), 3, 4))
    with pytest.raises(SystemExit):
        cli._fail("Error during simulation: out of memory")
    assert aborted == [1]
    assert "MPI rank 3: Error during simulation: out of memory" in capfd.readouterr().err


@pytest.mark.parametrize("ncpu", [0, -2, "2", 2.5, True])
def test_ncpu_is_a_whole_number_of_cpus_or_minus_one(ncpu, tmp_path, monkeypatch):
    import yaml
    from euvst_response.main import main

    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump({"instrument": "SWC", "ncpu": ncpu,
                                      "uniform_intensity": "5000 erg / (s cm2 sr)"}))
    monkeypatch.setattr(sys, "argv", ["eclipse", "--config", str(config)])
    with pytest.raises(ValueError, match="'ncpu' must be a whole number of CPUs"):
        main()
