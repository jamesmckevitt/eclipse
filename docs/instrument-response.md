# Running a simulation

This is the second stage of a run. `eclipse` takes the spectra you made when you [synthesised them](index.md#how-eclipse-works), passes them through the telescope and detector, adds the noise, and fits the result as you would fit real data. The noise is random, so ECLIPSE repeats this many times (a Monte Carlo simulation). That gives the spread of the measured intensities, velocities and line widths, to compare with the known truth.

This page covers a run on a synthesis file. The same settings apply to [a single line of known intensity](uniform-intensity.md) and to a [time series](time-series.md).

## Running simulations

Run the simulation with:

```bash
eclipse --config ./run/input/config.yaml
```

**Command-line options:**

- `--config`: the configuration file (required)
- `--debug`: when an error happens, open an IPython debugger where it happened in ECLIPSE's code (optional)

The results are written to `run/result/<config name>.h5`, named after the configuration file. [Analysing the results](analysis.md) shows how to read them.

## Configuration file

The configuration file is YAML. Most settings go in four sections, `simulation`, `detector`, `telescope` and `filter`, and any field of the matching class in `config.py` can be set there. **In these four sections, a setting given as a list of more than one value is swept**: ECLIPSE runs every combination of the swept values. A list of one value is just that value.

There are two exceptions: `telescope.psf_params`, which is a list itself, and `telescope.psf_slit_width`, the slit it was measured with. Each always takes one value, never a sweep.

At the top level, only `offchip_bin_slit` is swept this way. The other lists there, such as the pinhole lists and a time series' files, are lists of things rather than sweeps, as are the fitting components.

The top level of the file holds the settings that are not about the instrument's hardware: which instrument to simulate (see [Instruments](instruments.md)), what to observe, and how many Monte Carlo iterations to run. The [configuration reference](configuration.md) lists every setting, with its default.

Here's a complete example configuration file:

```yaml
# Input
instrument: SWC
synthesis_file: ./run/input/synthesised_spectra.h5
reference_line: Fe12_195.1190

# Global settings (apply to all combinations)
n_iter: 500
ncpu: -1
offchip_bin_slit: [1, 2]  # sweep no-binning and 2-pixel off-chip binning

# Simulation parameters
# Any field listed as a list is swept over; all combinations are run.
simulation:
  slit_width: [0.2 arcsec, 0.4 arcsec]  # sweep over two slit widths
  expos: [5 s, 10 s, 20 s, 40 s, 80 s]  # sweep over five exposure times
  psf: True
  vis_sl: 0 photon / (s * cm^2)
  enable_pinholes: False

# Detector parameters
detector:
  ccd_temperature: -60 Celsius  # used to compute dark current via the CCD model

# Telescope parameters
telescope:
  microroughness_sigma: 0.3 nm  # RMS surface roughness

# Aluminium filter parameters (SWC only)
filter:
  al_thickness: 1485 angstrom
  oxide_thickness: 95 angstrom
  c_thickness: 40 angstrom
  mesh_throughput: 0.8
```

Any setting in the [configuration reference](configuration.md) can be added to its section in the same way. For example, to sweep over the detector's quantum efficiency:

```yaml
detector:
  ccd_temperature: -60 Celsius
  qe_euv: [0.5, 0.65, 0.76]   # sweep over three QE values
```

For recommended values, see [McKevitt et al. (2026), PASJ 78, 1524](https://academic.oup.com/pasj/article/78/4/1524/8731000).

!!! warning "Parameters must go inside their section"

    A setting written at the top level instead, such as `expos:` or `ccd_temperature:` directly under the document root, stops the run with an error naming the section it belongs in. Any other key ECLIPSE does not read, such as a misspelt name, stops the run in the same way.

## The point spread function

With `simulation.psf: True`, ECLIPSE blurs the spectra with the instrument's point spread function (PSF), along the slit and in wavelength, before they reach the detector's pixels. It is off by default.

### The spectral PSF and the slit

With `psf: True`, each line is blurred in wavelength by the optics and by the image of the slit, so the blur depends on the slit width. `telescope.psf_params` gives the PSF's FWHM in pixels, along the slit and in wavelength. For SWC, the one in wavelength is for the 0.2 arcsec slit.

`simulation.spectral_psf` sets how the slit is added to the optics' blur:

- `quadrature` (default): a Gaussian, whose FWHM is the optics' and the slit image's widths added in quadrature.
- `convolution`: the optics' Gaussian convolved with the slit's rectangular image.

```yaml
simulation:
  slit_width: [0.2 arcsec, 1.6 arcsec]
  psf: True
  spectral_psf: convolution
```

The EIS PSF is not tied to a slit, so its width in wavelength is the same for every slit, and `convolution` is refused for it. Setting `telescope.psf_slit_width`, the slit that `psf_params` was measured with, changes both.

### The edges of the atmosphere

With `psf: True`, the blur brings in light from beyond the edges of the atmosphere. `simulation.psf_boundary` sets what is there:

- `replicate` (default): the Sun beyond each edge is taken to be like the cells at the edge, so the pixels there are as bright as they would be in the middle of a larger atmosphere.
- `zero`: nothing is beyond the edges, so the light the blur carries out of the atmosphere is lost, and the pixels within a PSF width of an edge come out darker.

```yaml
simulation:
  psf: True
  psf_boundary: zero
```

## Off-chip slit binning

`offchip_bin_slit` adds together neighbouring pixels along the slit after read-out, so the binning is done on the ground rather than on the detector. Adding `n` pixels multiplies the signal by `n`, while the noise only grows as `sqrt(n)`. The signal to noise therefore goes up as `sqrt(n)`, at the cost of resolution along the slit.

```yaml
offchip_bin_slit: [1, 2, 4]   # swept like any other list
```

## Turning the noise off

`noise: False` turns the noise off: every random draw in the detector is replaced by its mean. Every iteration is then the same, so one is enough:

```yaml
simulation:
  noise: False

n_iter: 1     # every iteration would be identical
```

## Multi-node MPI parallelisation

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

srun --mpi=pmi2 --kill-on-bad-exit=1 eclipse --config ./run/input/my_run.yaml
```

MPI shares the iterations between the nodes, and joblib shares each rank's iterations between its cores. `--kill-on-bad-exit=1` stops every rank when one fails, rather than leaving the others waiting until the time limit. `I_MPI_PIN_DOMAIN=auto` lets each rank use its whole node, and `LOKY_MAX_CPU_COUNT` stops joblib starting more processes than that. Setting `ncpu` in the configuration is optional. Under MPI it is capped at `SLURM_CPUS_PER_TASK`, and `ncpu: -1` lets joblib find the CPUs it may use by itself.

