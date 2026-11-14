# On a cluster

```bash
eclipse synthesise job.yaml
srun --mpi=pmi2 --kill-on-bad-exit=1 eclipse observe job.yaml
```

## Many cores with MPI

ECLIPSE can run as several MPI processes (ranks) on a SLURM cluster, started with `srun` or `mpirun` and `--ntasks-per-node`. It then shares the Monte Carlo iterations between the ranks, and collects the results on the first one. It detects MPI by itself, so nothing in the configuration changes. Without `mpi4py`, or with only one rank, it runs as a single process.

It needs `mpi4py` and Intel MPI (load it with `module load intel-mpi` before launching).

A submission script that runs one rank per node, with joblib using the cores within each rank:

```bash
#!/bin/bash
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=1-00:00:00

module load intel-mpi
export I_MPI_PMI_LIBRARY=/opt/slurm/slurm-21-08-5-1/lib/libpmi.so
export I_MPI_PIN_DOMAIN=auto
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK
source /path/to/venv/bin/activate

srun --mpi=pmi2 --kill-on-bad-exit=1 eclipse run ./run/input/my_run.yaml
```

MPI shares the iterations between the nodes, and joblib shares each rank's iterations between its cores. `--kill-on-bad-exit=1` stops every rank when one fails, rather than leaving the others waiting until the time limit. `I_MPI_PIN_DOMAIN=auto` lets each rank use its whole node, and `LOKY_MAX_CPU_COUNT` stops joblib starting more processes than that. Setting `ncpu` in the configuration is optional. Under MPI it is capped at `SLURM_CPUS_PER_TASK`, and `ncpu: -1` lets joblib find the CPUs it may use by itself.
