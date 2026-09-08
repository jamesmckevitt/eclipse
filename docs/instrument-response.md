# Instrument response

## Configuration file

ECLIPSE uses YAML configuration files to specify simulation parameters.
Parameters are organised into four sections - `simulation`, `detector`, `telescope`, and `filter` - each corresponding directly to a configuration class in `config.py`.
Any field of those classes can be set here.
**Any parameter that is given as a list is automatically swept over** and the
simulation runs every combination (cartesian product).

**Top-level keys** (not sections):

- `instrument`: `SWC` (EUVST Short Wavelength) or `EIS` (Hinode/EIS)
- `synthesis_file`: path to the synthesised spectra pickle file
- `reference_line`: spectral line used as the wavelength-grid reference (default `Fe12_195.1190`)
- `n_iter`: number of Monte Carlo iterations
- `ncpu`: CPU cores to use (`-1` = all available)
- `offchip_bin_slit`: off-chip slit binning factor (default `1`)
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

For guidance on recommended values, see McKevitt et al. (2025) (in prep.).

By default, both the DN and photon signals are fitted at every Monte Carlo iteration. To speed up the simulation when only one is needed, use the `fit_signals` option:

```yaml
fit_signals: dn       # Fit only the DN signal
fit_signals: photon   # Fit only the photon signal
fit_signals: both     # Fit both (default)
```

To fit blended spectral lines with multiple Gaussian components, add a `fitting` block:

```yaml
fitting:
  primary_component: 0           # index of the component whose velocity is reported
  constrain_positive_intensity: true  # reject fits with negative amplitudes
  backend: scipy                 # optimiser: "scipy" (default) or "mpfit"
  components:
    - wavelength: 195.119 angstrom     # component 0: free centre, width, amplitude
    - wavelength: 195.179 angstrom     # component 1: centre & width tied to component 0
      tie_center: 0
      tie_width: 0
```

Each component requires a `wavelength` field giving its rest wavelength.

Each entry in `components` corresponds to one Gaussian. Optional per-component keys:

- `tie_center: <i>`: constrain this component's centre to match component *i*
- `tie_width: <i>`: constrain this component's line width to match component *i*
- `amplitude_greater_than: <i>`: constrain amplitude to exceed that of component *i*

Omitting the `fitting` block fits a single Gaussian (default behaviour).

If you synthesised data in dynamic mode, your configuration must specify:

- Exactly one slit width matching the synthesis slit width
- Exactly one exposure time matching the synthesis exposure time

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

## Output

Results are saved as pickle files in the `run/result/` directory with the same base name as the configuration file. The output includes:

- Simulated detector signals (DN and photon counts)
- Fitted spectral line parameters (intensity, velocity, width)
- Statistical analysis of velocity precision vs. exposure time
- Ground truth comparisons
- Full config objects (`Detector`, `Telescope`, `Simulation`) for each parameter combination
- The git commit ID and software version used to produce the results

Use `summary_table(results)` after loading to see all parameter combinations and the run metadata.
