# Simulating a single snapshot

This is the second stage of a run. It takes the spectra you made when you [synthesised an atmosphere](index.md#how-eclipse-works), passes them through the telescope and detector, adds the noise, and fits the result as you would fit real data. The noise is random, so ECLIPSE repeats this many times (a Monte Carlo simulation). That gives the spread of the measured intensities, velocities and line widths, to compare with the known truth.

To observe a series of atmosphere or synthesis files instead, see [Simulating a time series](time-series.md). For spectra that another code synthesised, see [From another code](other-codes.md). The rest of this page applies to those too.

## Choosing an instrument

The `instrument:` key at the top level of the configuration file chooses the instrument:

- `SWC` - SOLAR-C/EUVST-SW, the short wavelength channel.
- `EIS` - Hinode/EIS.

Three settings apply to SWC only. Under `EIS`:

- `filter:` describes EUVST-SW's aluminium filter. EIS's effective area comes from its own calibration tables, which already include its filters, so for EIS the whole section is ignored, with a warning. See [EIS effective area](#eis-effective-area) below.
- `telescope.microroughness_sigma` is the roughness of EUVST's primary mirror. For EIS it is ignored, with a warning.
- Pinholes in the filter are modelled for EUVST-SW only. Any pinhole setting, `enable_pinholes` or a `pinhole_*` list, stops an EIS run with an error.

The EIS point spread function (PSF) is not well known. ECLIPSE uses a symmetrical Gaussian 3 pixels wide (FWHM), following Ugarte-Urra (2016), EIS Software Note 2, and prints a warning saying so whenever `psf: True` is set.

### EIS effective area

The EIS effective area comes from the instrument's own calibration tables, so it varies with wavelength and, for the in-flight calibrations, with the date of the observation:

```yaml
instrument: EIS
telescope:
  calibration: dz2025
  date: "2012-06-03"    # quoted, so YAML keeps it a string
```

| `calibration` | Source | Date |
|---|---|---|
| `ground` (default) | Pre-flight MSSL tables, `eis_ea.pro` | not used |
| `dz2013` | Del Zanna (2013), `eis_ltds.pro` | required |
| `warren2014` | Warren, Ugarte-Urra & Landi (2014) | required |
| `dz2025` | Del Zanna et al. (2025), `interpol_eis_ea.pro` | required |

The three in-flight calibrations need a `date`, and stop with an error without one. Both keys can be swept like any other setting:

```yaml
telescope:
  calibration: dz2025
  date: ["2008-01-01", "2013-01-01", "2018-01-01"]
```

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

## Configuration file

The configuration file is YAML. Most settings go in four sections, `simulation`, `detector`, `telescope` and `filter`, and any field of the matching class in `config.py` can be set there. **A setting given as a list of more than one value is swept**: ECLIPSE runs every combination of the swept values. A list of one value is just that value.

The one exception is `telescope.psf_params`, which is a list itself, so it is always one value, never a sweep.

**Top-level keys**:

- `instrument`: `SWC` (EUVST's short wavelength channel) or `EIS` (Hinode/EIS)
- `synthesis_file`: the synthesis file to observe, from ECLIPSE's own synthesis or [another code](other-codes.md) (default `./run/input/synthesised_spectra.h5`). Pickles written by ECLIPSE 0.11.0 and earlier are still read, with a warning
- `reference_line`: the line whose spectral window is observed (default: the file's only line if it has one, otherwise `Fe12_195.1190`, which is always the default for a pickle). Every line in the synthesis file that falls in that window is added in, each with its own brightness, so blends are included. To observe several windows, run once for each. Line names follow the [usual convention](synthesis.md#naming-spectral-lines).
- `n_iter`: how many Monte Carlo iterations to run
- `ncpu`: how many CPU cores to use (`-1` for all of them)
- `offchip_bin_slit`: how many pixels along the slit to bin together after read-out (default `1`); see [off-chip slit binning](#off-chip-slit-binning)
- `pinhole_sizes`, `pinhole_positions`, `pinhole_positions_spectral`: lists describing pinholes in the filter, one entry per pinhole (SWC only)
- `uniform_intensity`, `rest_wavelength`, `thermal_width`: a single line of known intensity, in place of a synthesis file; see [From a single intensity](uniform-intensity.md)
- `atmosphere_series`, `synthesis`, `raster`: a time series of atmosphere files, synthesised during the run, with the synthesis settings and the observing plan, in place of a synthesis file; see [Simulating a time series](time-series.md)
- `synthesis_series`: a time series of synthesis files, one per snapshot, observed with a `raster` plan in the same way, in place of a synthesis file; see [From synthesis files](time-series.md#from-synthesis-files)

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

Any field of the `Detector_SWC`, `Telescope_EUVST` or `AluminiumFilter` classes in `config.py` can be added to its section. For example, to sweep over the detector's quantum efficiency:

```yaml
detector:
  ccd_temperature: -60 Celsius
  qe_euv: [0.5, 0.65, 0.76]   # sweep over three QE values
```

For recommended values, see [McKevitt et al. (2026), PASJ 78, 1524](https://academic.oup.com/pasj/article/78/4/1524/8731000).

!!! warning "Parameters must go inside their section"

    A setting written at the top level instead, such as `expos:` or `ccd_temperature:` directly under the document root, stops the run with an error naming the section it belongs in. Any other key ECLIPSE does not read, such as a misspelt name, stops the run in the same way.

By default ECLIPSE fits two signals at every Monte Carlo iteration: the detector's output in DN, and the photons arriving at the detector. If you only need one, `fit_signals` saves time:

```yaml
fit_signals: dn   # "dn" fits only the DN signal
                  # "photon" fits only the photon signal
                  # "both" fits both, and is the default
```

To fit blended lines with several Gaussian components, add a `fitting` block:

```yaml
fitting:
  primary_component: 0           # index of the component whose velocity is reported
  constrain_positive_intensity: true  # keep every amplitude at zero or above during the fit
  backend: scipy                 # optimiser: "scipy" (default) or "mpfit"
  max_iter: 1000                 # optimiser iterations before it gives up
  components:
    - wavelength: 195.119 angstrom     # component 0: free centre, width, amplitude
      name: Fe XII 195.119
    - wavelength: 195.179 angstrom     # component 1: centre & width tied to component 0
      name: Fe XII 195.179
      tie_center: 0
      tie_width: 0
```

Each entry in `components` is one Gaussian, and needs a `wavelength`, its rest wavelength. It can also have:

- `tie_center: <i>`: fit this component at the same velocity as component *i*. Its centre is the other's scaled by the ratio of their rest wavelengths, so one velocity holds across the whole window.
- `tie_width: <i>`: fit this component with the same line width as component *i*.
- `amplitude_greater_than: <i>`: keep this component's amplitude above that of component *i*.
- `name: <text>`: the component's name in the results. It defaults to its rest wavelength, such as `195.1190 Angstrom`.

Without `components`, a single Gaussian is fitted.

`max_iter` limits how many iterations the optimiser may take on one spectrum. A fit that runs out, or fails for any other reason, is left out of the mean and standard deviation, and the run prints how many fits failed and in how many pixels.

`bessel_correction: true` computes the standard deviation over the Monte Carlo iterations with n - 1 in place of n (Bessel's correction).

`save_iterations: true` keeps every iteration's fitted values in the results, as well as their statistics. The results are then about `n_iter` times larger.

!!! warning "The primary component must be present in the data"

    ECLIPSE reports the velocity of `primary_component`. The fit starts with all of the components moved together, to where they best match the profile. So if the line you asked for is not in your data, the fit can settle a whole component spacing away: 92 km/s for Fe XII 195.119 and 195.179. A few per cent of the blend is enough to place it correctly.

    Also, two lines of similar brightness closer than about three line widths are often fitted as one broad component instead of two. The dip between them never falls below half the peak, so the fit starts with a width that covers the whole blend. `constrain_positive_intensity` does not help here, and can make it worse.

If you synthesised the spectra in the deprecated dynamic mode, the configuration must have:

- exactly one slit width, the one used in the synthesis
- exactly one exposure time, the one used in the synthesis

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

## Uniform intensity mode

Setting `uniform_intensity` replaces the atmosphere with a single line of known total intensity, and no `synthesis_file` is needed. See [From a single intensity](uniform-intensity.md).

## Running simulations

Run the simulation with:

```bash
eclipse --config ./run/input/config.yaml
```

**Command-line options:**

- `--config`: the configuration file (required)
- `--debug`: when an error happens, open an IPython debugger where it happened in ECLIPSE's code (optional)

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

## Common random numbers

When you compare two instrument configurations, the noise in each run can hide a small difference between them. Giving both runs the same random numbers, known as [common random numbers](https://en.wikipedia.org/wiki/Variance_reduction#Common_Random_Numbers_%28CRN%29), makes their noise go up and down together, so it mostly cancels when you compare them.

This only works if both runs use the same number of random values at every step, or every later step gets different values. By default that fails when the runs differ in brightness, because NumPy's photon-count sampler uses more or fewer values depending on how many photons are expected. ECLIPSE can instead draw its counts by [inverse-transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling), which uses exactly one value for each pixel however bright it is. There are two options for this:

- `photon_shot_inverse_transform`, for runs that differ in photon flux. It covers the photons arriving, the number the detector catches (its quantum efficiency), and the spread in the number of electrons each photon frees (the Fano noise).
- `dark_current_inverse_transform`, for runs that differ in dark current.

They are arguments to `monte_carlo`:

```python
import numpy as np
from euvst_response.monte_carlo import monte_carlo

np.random.seed(1234)   # the same value before each run being compared

first_dn, dn_stats, first_photon, photon_stats = monte_carlo(
    I_cube, t_exp, det, tel, sim, n_iter=500,
    photon_shot_inverse_transform=True,    # for runs differing in photon flux
    dark_current_inverse_transform=True,   # for runs differing in dark current
)
```

Both are off by default. They don't change a run's statistics, only how closely two runs follow each other, but they make the run slower.

They can only be switched on from Python, not in a config file run with `eclipse --config`. ECLIPSE doesn't seed NumPy's random number generator itself, so call `np.random.seed` with the same value before each run, as in the example. With MPI, both runs also need the same number of ranks, because each rank's random numbers come from the seed and its rank number.

## Output

Results are written to `run/result/<config name>.h5`. They include:

- The first iteration's signals, in DN and in photons
- For each fitted component, by name, in each pixel: its first fit, the mean and standard deviation of its intensity, velocity and width over the iterations, and the number of fits that failed
- The truth to compare against: the fit to the spectra on the detector's pixels, with no noise and no PSF
- The settings (`Detector`, `Telescope`, `Simulation`) of each combination
- The version of ECLIPSE and the git commit that made them

After loading the results, `summary_table(results)` shows every combination, the fitted components and the version. `list_fit_components` gives the components' names. `analyse_fit_statistics` and `create_sunpy_maps_from_combo` take `component=` to choose one, and default to the primary component.

??? note "Results files from ECLIPSE 0.11.0 and earlier"

    ECLIPSE 0.11.0 and earlier wrote the results as a pickle. `load_instrument_response_results` still reads one, with a warning, until a future release stops it. Rerunning `run.yaml` writes a new `run.h5`, and renames the old `run.pkl` to `run.pkl.old`, or `run.pkl.old.1` and so on if that name is taken. A script that loads `run.pkl` with ECLIPSE's functions then reads `run.h5` instead, with a warning. `euvst_response.convert_results_pickle("run.pkl")` converts an old pickle to the new format without rerunning.
