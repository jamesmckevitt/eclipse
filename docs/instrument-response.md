# Simulating the instrument

This is the second stage of a run. It takes the spectra produced when you [synthesised an atmosphere](index.md#how-eclipse-works), puts them through the telescope and detector, adds the noise, and fits the result the same way you would fit real data. Because the noise is random, a Monte Carlo simulation gives a distribution of measured intensities, velocities, and line widths to compare against the known truth.

## Choosing an instrument

The top-level `instrument:` key selects the instrument model:

- `SWC` - SOLAR-C/EUVST-SW (short wavelength channel).
- `EIS` - Hinode/EIS.

Three things are specific to `SWC` and are handled as follows under `EIS`:

- The `filter:` section describes the EUVST-SW aluminium filter. ECLIPSE treats the EIS effective area as one value and cannot vary engineering values for its aluminium filter, so the whole section is ignored with a warning for EIS.
- `telescope.microroughness_sigma` is an engineering parameter specific to the EUVST-primary mirror. For EIS, it is ignored with a warning.
- Pinhole effects are specific to EUVST-SW. Setting `pinhole_sizes`, or `simulation.enable_pinholes: True`, raises an error for EIS. Note that `pinhole_positions` on its own does not: without `pinhole_sizes` it is ignored for either instrument.

The EIS point spread function is not well characterised. ECLIPSE uses a symmetrical Gaussian with a FWHM of 3 pixels, following Ugarte-Urra (2016), EIS Software Note 2, and prints a warning saying so whenever `psf: True` is set.

## Configuration file

ECLIPSE uses YAML configuration files to specify simulation parameters. Parameters are organised into four sections - `simulation`, `detector`, `telescope`, and `filter` - each corresponding directly to a configuration class in `config.py`. Any field of those classes can be set here. **Any parameter given as a list of more than one value is automatically swept over** and the simulation runs every combination (Cartesian product). A single-element list is treated as a fixed value, not as a sweep of one.

There is one exception: `telescope.psf_params` is itself a list-valued parameter, so it is always taken as a single fixed value rather than as a sweep dimension.

**Top-level keys**:

- `instrument`: `SWC` (EUVST Short Wavelength) or `EIS` (Hinode/EIS)
- `synthesis_file`: path to the synthesised spectra pickle file
- `reference_line`: spectral line used as the wavelength-grid reference (default `Fe12_195.1190`). All lines in the synthesis file are interpolated onto this line's wavelength grid and summed, so this key effectively selects which spectral window is simulated, and any blends falling in that window are included. Run once per window. Line names follow the [usual convention](synthesis.md#naming-spectral-lines).
- `n_iter`: number of Monte Carlo iterations
- `ncpu`: CPU cores to use (`-1` = all available)
- `offchip_bin_slit`: off-chip slit binning factor (default `1`), see [off-chip slit binning](#off-chip-slit-binning)
- `pinhole_sizes`, `pinhole_positions`: fixed paired lists for pinhole diffraction tests (SWC only)
- `uniform_intensity`, `rest_wavelength`, `thermal_width`: uniform-intensity mode (alternative to synthesis file)

Here's a complete example configuration file:

```yaml
# Input
instrument: SWC
synthesis_file: ./run/input/synthesised_spectra.pkl
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

Any parameter from the `Detector_SWC`, `Telescope_EUVST`, or `AluminiumFilter`
dataclasses in `config.py` can be added to the corresponding section. For
example, to sweep over detector quantum efficiency:

```yaml
detector:
  ccd_temperature: -60 Celsius
  qe_euv: [0.5, 0.65, 0.76]   # sweep over three QE values
```

For guidance on recommended values, see
[McKevitt et al. (2026), PASJ 78, 1524](https://academic.oup.com/pasj/article/78/4/1524/8731000).

!!! warning "Parameters must go inside their section"

    Only `simulation`, `detector`, `telescope`, and `filter` are read as sections. A parameter written at the top level instead - `expos:` or `ccd_temperature:` directly under the document root - is **silently ignored**, and the run proceeds with the default value. There is no warning. If a sweep produces suspiciously identical results across combinations, check the indentation first.

By default, both the DN and photon signals are fitted at every Monte Carlo iteration. To speed up the simulation when only one is needed, use the `fit_signals` option:

```yaml
fit_signals: dn   # "dn" fits only the DN signal
                  # "photon" fits only the photon signal
                  # "both" fits both, and is the default
```

To fit blended spectral lines with multiple Gaussian components, add a `fitting` block:

```yaml
fitting:
  primary_component: 0           # index of the component whose velocity is reported
  constrain_positive_intensity: true  # reject fits with negative amplitudes
  backend: scipy                 # optimiser: "scipy" (default) or "mpfit"
  max_iter: 1000                 # optimiser iterations before it gives up
  components:
    - wavelength: 195.119 angstrom     # component 0: free centre, width, amplitude
    - wavelength: 195.179 angstrom     # component 1: centre & width tied to component 0
      tie_center: 0
      tie_width: 0
```

Each component requires a `wavelength` field giving its rest wavelength.

Each entry in `components` corresponds to one Gaussian. Optional per-component keys:

- `tie_center: <i>`: constrain this component to share component *i*'s velocity
- `tie_width: <i>`: constrain this component's line width to match component *i*
- `amplitude_greater_than: <i>`: constrain amplitude to exceed that of component *i*

Omitting the `fitting` block fits a single Gaussian.

`max_iter` caps how long the optimiser may work on one spectrum before it gives
up and returns wherever it reached, which it does silently. It is counted in
iterations, so it means the same thing on either backend and does not shrink as
components are added, and it applies to single-Gaussian fits as well as blends.
Fits converge in tens of iterations, so the default of 1000 is a safety net
rather than a tuning knob; raise it if a difficult blend looks under-converged.
For comparison, EISPAC uses 2000.

Without a `fitting` block there is nowhere to write it, so single-Gaussian runs
take the same 1000 by default.

#### Choosing the components

Three things about blends are worth knowing before you trust the velocities.

**Nominate a primary that has flux.** The reported velocity is read from
`primary_component`, and the initial guess places the whole comb by matching it
against the profile. If the primary is a line that is absent in your data, no
amount of fitting recovers its velocity, and the comb can settle a whole
component spacing away: for Fe XII 195.119 and 195.179 that is 92 km/s. As soon
as the primary carries even a few per cent of the blend the placement is
reliable again. Choose the line you actually want to measure, not the one that
happens to be first in the list.

**Comparable lines closer than about three line widths are hard.** The initial
width is estimated by walking out from the brightest pixel until the profile
falls below half its height. When two lines of similar brightness sit closer
than that, the dip between them never drops below half maximum, so the estimate
covers the whole blend and the fit can start twice as wide as the truth and
settle for one broad component instead of two. Enforcing
`constrain_positive_intensity` does not rescue it, and can make it worse by
pinning the second amplitude at zero. A blend of very unequal lines, or one
wider than a few line widths, is unaffected.

**`tie_center` ties velocities, not separations.** A common Doppler shift
stretches a blend rather than sliding it, because each line moves by an amount
proportional to its own wavelength. `tie_center` therefore scales each tied
centre by the ratio of the rest wavelengths, so the one fitted centre means one
velocity for every component in the group, whatever the width of the window.
`tie_width` is a plain equality, unchanged: thermal broadening does scale with
wavelength, but the instrumental width that dominates these windows does not.

If you synthesised data in dynamic mode, your configuration must specify:

- Exactly one slit width matching the synthesis slit width
- Exactly one exposure time matching the synthesis exposure time

## Off-chip slit binning

`offchip_bin_slit` sums adjacent pixels along the slit after read-out, so it is binning done on the ground rather than on the detector. Summing `n` pixels multiplies the signal by `n` while the noise only adds in quadrature. Signal to noise therefore goes up as `sqrt(n)`, with a reduction of spatial resolution along the slit.

```yaml
offchip_bin_slit: [1, 2, 4]   # swept like any other list-valued parameter
```

## Uniform intensity mode

Setting `uniform_intensity` replaces the atmosphere with a single spectral line of known integrated intensity, and no `synthesis_file` is needed. See [synthesis from a single intensity](uniform-intensity.md).

## Running simulations

Run the instrument response function using:

```bash
eclipse --config ./run/input/config.yaml
```

**Command-line options:**

- `--config`: Path to YAML configuration file (required)
- `--debug`: Enable debug mode with IPython breakpoints on errors (optional)

## Multi-node MPI parallelisation

When launched with multiple MPI ranks on a SLURM cluster (via `srun` or `mpirun`, and setting `--ntasks-per-node`), ECLIPSE automatically distributes Monte Carlo iterations across ranks and gathers results on rank 0. No code or configuration changes are needed - MPI is auto-detected at runtime. If `mpi4py` is not installed or only one rank is present, the code falls back to single-process mode.

Requirements: `mpi4py` and `intel-mpi` (load with `module load intel-mpi` before launching).

A working submission script, one rank per node with joblib using the cores inside each rank:

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

srun --mpi=pmi2 eclipse --config ./run/input/my_run.yaml
```

MPI spreads Monte Carlo iterations across nodes, and joblib parallelises within each rank. `I_MPI_PIN_DOMAIN=auto` gives each rank an affinity mask covering its whole node, and `LOKY_MAX_CPU_COUNT` stops joblib oversubscribing against that mask. Setting `ncpu` in the config is optional - in MPI mode it is capped to `SLURM_CPUS_PER_TASK`, while `ncpu: -1` lets joblib read the affinity mask itself.

## Common random numbers

If you want to compare two instrument configurations whose noise is lower than another noise source, the scatter in the results will make finding trends difficult. The solution is to use the same random numbers for their shared noise. This is variance reduction by common random numbers.

It does not work with `np.random.poisson`, which draws by rejection and so uses a different number of random values depending on the mean it is given. Two runs at different photon flux therefore become out of step.

Inverse-transform sampling uses one random value per pixel whatever the mean, so the runs stay in step. There is an option for each of the two Poisson stages:

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

Both are keyword-only and both default to `False`. The distribution is the same, so a run's statistics do not change and only the correlation between two runs does. Inverting the CDF is slower than the default sampler.

None of this has a configuration key, so this needs to be done with the Python API rather than run with `eclipse --config`. ECLIPSE does not seed NumPy's generator, so call `np.random.seed` with the same value before each run. Each MPI rank keeps its own generator state, so both runs also need the same number of ranks.

## Output

Results are saved as pickle files in the `run/result/` directory with the same base name as the configuration file. The output includes:

- Simulated detector signals (DN and photon counts)
- Fitted spectral line parameters (intensity, velocity, width)
- Statistical analysis of velocity precision vs. exposure time
- Ground truth comparisons
- Full config objects (`Detector`, `Telescope`, `Simulation`) for each parameter combination
- The git commit ID and software version used to produce the results

Use `summary_table(results)` after loading to see all parameter combinations and the run metadata.