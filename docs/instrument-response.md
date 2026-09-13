# Simulating the instrument

This is the second stage of a run. It takes the spectra produced when you [synthesised an atmosphere](index.md#how-eclipse-works), puts them through the telescope and detector, adds the noise, and fits the result the same way you would fit real data. Because the noise is random, a Monte Carlo simulation gives a distribution of measured intensities, velocities, and line widths to compare against the known truth.

## Choosing an instrument

The top-level `instrument:` key selects the instrument model:

- `SWC` - SOLAR-C/EUVST-SW (short wavelength channel).
- `EIS` - Hinode/EIS.

Three things are specific to `SWC` and are handled as follows under `EIS`:

- The `filter:` section describes the EUVST-SW aluminium filter. The EIS effective area comes from the instrument's own calibration tables, which already fold in its filters, so engineering values cannot be varied for it and the whole section is ignored with a warning for EIS. See [EIS effective area](#eis-effective-area) below.
- `telescope.microroughness_sigma` is an engineering parameter specific to the EUVST-primary mirror. For EIS, it is ignored with a warning.
- Pinhole effects are specific to EUVST-SW. Setting `pinhole_sizes`, or `simulation.enable_pinholes: True`, raises an error for EIS. Note that `pinhole_positions` on its own does not: without `pinhole_sizes` it is ignored for either instrument.

The EIS point spread function is not well characterised. ECLIPSE uses a symmetrical Gaussian with a FWHM of 3 pixels, following Ugarte-Urra (2016), EIS Software Note 2, and prints a warning saying so whenever `psf: True` is set.

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

The three in-flight calibrations raise without a `date`. Both keys can be swept like any other parameter:

```yaml
telescope:
  calibration: dz2025
  date: ["2008-01-01", "2013-01-01", "2018-01-01"]
```

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
- `pinhole_sizes`, `pinhole_positions`, `pinhole_positions_spectral`: paired lists describing filter pinholes (SWC only), covered in [pinhole stray light](pinholes.md)
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

- `tie_center: <i>`: fit this component at the same velocity as component *i*. Centres are scaled by the ratio of the two rest wavelengths rather than offset by a fixed wavelength, so a single velocity is correct across the whole window.
- `tie_width: <i>`: fit this component with the same line width as component *i*.
- `amplitude_greater_than: <i>`: constrain amplitude to exceed that of component *i*

Omitting the `fitting` block fits a single Gaussian, with `max_iter` at its default.

`max_iter` limits how many iterations the optimiser may take on one spectrum. If it runs out it returns whatever it has reached. There is no warning.

!!! warning "The primary component must be present in the data"

    The velocity ECLIPSE reports is the velocity of `primary_component`, and the initial guess positions all the components together by matching them against the profile. If the line you asked for is not in your data, the fit can settle a whole component spacing away - 92 km/s for Fe XII 195.119 and 195.179. A few per cent of the blend is enough to place it correctly.

    Separately, two lines of similar brightness closer than about three line widths are often fitted as one broad component instead of two, because the dip between them never falls below half maximum and the initial width then covers the whole blend. `constrain_positive_intensity` does not help here and can make it worse.

If you synthesised data in dynamic mode, your configuration must specify:

- Exactly one slit width matching the synthesis slit width
- Exactly one exposure time matching the synthesis exposure time

## Off-chip slit binning

`offchip_bin_slit` sums adjacent pixels along the slit after read-out, so it is binning done on the ground rather than on the detector. Summing `n` pixels multiplies the signal by `n` while the noise only adds in quadrature. Signal to noise therefore goes up as `sqrt(n)`, with a reduction of spatial resolution along the slit.

```yaml
offchip_bin_slit: [1, 2, 4]   # swept like any other list-valued parameter
```

## The edge of the raster

The spatial PSF has to assume something about the Sun beyond the ends of the slit. By default it continues the edge rows outward, which says the emission just outside the field looks much like the emission just inside it:

```yaml
simulation:
  psf: True
  psf_boundary: replicate   # or 'zero' for the old behaviour
```

`zero` treats everything outside the field as dark. That removes real signal from the outermost rows: with the default SWC spatial PSF the edge row loses about a third of the kernel's weight, the next row 8 per cent, and the one after 1 per cent. Rows further in are untouched either way.

Neither is measured, because nothing was observed out there. `replicate` is the better assumption of the two, but if you need the outer rows to be trustworthy, crop them.

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