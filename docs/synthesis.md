# Synthesising from an MHD simulation

The synthesis script converts 3D MHD simulation data into synthetic solar spectra. Contribution functions G(T, n_e) are computed on-the-fly using [fiasco](https://fiasco.readthedocs.io/) (a Python interface to the CHIANTI atomic database).

The output is a synthesis file, which is the input to the [instrument response](instrument-response.md) stage.

!!! note "Which simulations are supported"

    The reader currently expects MURaM output: separate binary files for temperature, density, and velocity, with the cube shape given on the command line. Support for other MHD codes is being added; until then, reducing your simulation to a [VDEM](vdem-synthesis.md) is the best way to use it.

    Synthesis here is optically thin. If you need optically thick lines, synthesise them with a code such as Lightweaver or RH1.5D, and bring the spectra in at the [instrument response](instrument-response.md) stage instead of this one. Reading externally synthesised spectra is coming soon.

## Basic usage

```bash
# Example using all available command line options
synthesise-spectra \
  --data-dir ./data/atmosphere \
  --lines Fe12_195.1190 Fe12_195.1790 \
  --abundance sun_coronal_2021_chianti \
  --n-workers 4 \
  --output-dir ./run/input \
  --output-name synthesised_spectra.asdf \
  --temp-file temp/eosT.0270000 \
  --rho-file rho/result_prim_0.0270000 \
  --vx-file vx/result_prim_1.0270000 \
  --vy-file vy/result_prim_3.0270000 \
  --vz-file vz/result_prim_2.0270000 \
  --cube-shape 512 768 256 \
  --voxel-dx "0.192 Mm" \
  --voxel-dy "0.192 Mm" \
  --voxel-dz "0.064 Mm" \
  --vel-res "5.0 km/s" \
  --vel-lim "300.0 km/s" \
  --integration-axis z \
  --crop-x "-50 Mm" "50 Mm" \
  --crop-y "-50 Mm" "50 Mm" \
  --crop-z "0 Mm" "20 Mm" \
  --downsample 1 \
  --precision float64 \
  --mean-mol-wt 1.29

# Show all available options
synthesise-spectra --help
```

## Command line options

**Input/Output Paths:**

- `--data-dir`: Directory containing simulation data (default: `data/atmosphere`)
- `--output-dir`: Output directory for results (default: `./run/input`)
- `--output-name`: Output filename (default: `synthesised_spectra.asdf`)

**Line and Abundance Selection:**

- `--lines`: Emission lines to synthesise, e.g., `--lines Fe12_195.1190 Fe12_195.1790` (required)
- `--abundance`: CHIANTI abundance dataset name (default: `sun_coronal_2021_chianti`)
- `--n-workers`: Number of parallel workers for the fiasco G(T, n_e) computation. Each distinct ion is computed in a separate process. `0` uses all available CPUs (default: `0`). Set to `1` for serial execution.
- `--hdf5-dbase-root`: CHIANTI HDF5 database to compute G(T, n_e) from. Defaults to whichever database fiasco is configured to use in `~/.fiasco/fiascorc`. Set this to run against a second CHIANTI version without changing that default for your other work.

**Simulation Files:**

- `--temp-file`: Temperature file relative to data-dir (default: `temp/eosT.0270000`)
- `--rho-file`: Density file relative to data-dir (default: `rho/result_prim_0.0270000`)
- `--vx-file`: X-velocity file (required if `--integration-axis x`)
- `--vy-file`: Y-velocity file (required if `--integration-axis y`)
- `--vz-file`: Z-velocity file (required if `--integration-axis z`)

**Grid Parameters:**

- `--cube-shape`: Cube dimensions as three integers (default: `512 768 256`)
- `--voxel-dx`, `--voxel-dy`, `--voxel-dz`: Voxel sizes with units (default: `"0.192 Mm"`, `"0.192 Mm"`, `"0.064 Mm"`)

**Velocity Grid:**

- `--vel-res`: Velocity resolution with units (default: `"5.0 km/s"`)
- `--vel-lim`: Half-range of the velocity grid, applied as +/- this value, with units (default: `"300.0 km/s"`)

**Integration and Viewing:**

- `--integration-axis`: Integration axis: `x`, `y`, or `z` (default: `z`)
    - `z`: Standard top-down view (integrates through height)
    - `x`: Side view from the left (integrates left-to-right)
    - `y`: Side view from the front (integrates front-to-back)

**Spatial Cropping (Heliocentric coordinates with units):**

- `--crop-x`: X-range to crop with units, e.g., `--crop-x "-50 Mm" "50 Mm"` (optional)
- `--crop-y`: Y-range to crop with units, e.g., `--crop-y "-50 Mm" "50 Mm"` (optional)
- `--crop-z`: Z-range to crop with units, e.g., `--crop-z "0 Mm" "20 Mm"` (optional)
- Omit any crop option to use the full range in that dimension

**Processing Options:**

- `--downsample`: Downsampling factor (default: `1` = no downsampling)
- `--precision`: Numerical precision `float32` or `float64` (default: `float64`)
- `--mean-mol-wt`: Mean molecular weight (default: `1.29`)

## Naming spectral lines

Lines are named `<Element><Stage>_<Wavelength>`, for example `Fe12_195.1190`:

- `Fe` - element symbol, capitalised as usual (`Fe`, `Si`, `S`, `O`).
- `12` - ionisation stage as an **arabic** numeral, in spectroscopic notation, so
  `Fe12` is Fe XII, not Fe XI or Fe XIII.
- `195.1190` - rest wavelength in Angstrom.

The same names are used by `--lines`, by the `reference_line` key in the
instrument configuration, and as the keys of `line_cubes` in the output file.

The wavelength does not have to be exact. ECLIPSE finds the nearest transition of
that ion in CHIANTI and prints both the requested and matched wavelengths:

```text
  Fe12_195.1190: requested 195.1190 Angstrom, matched 195.1190 Angstrom (delta=0.0000 Angstrom)
```

Check that line. A large difference means the transition you meant is not in the
database for that ion, and a neighbouring one was picked up instead. A name that
does not match the pattern at all raises `ValueError` immediately.

## Dynamic mode (time-varying atmospheres)

For simulating raster scans over evolving atmospheres, use dynamic mode which combines MHD timesteps based on instrument scanning:

```bash
synthesise-spectra \
  --data-dir ./data/atmosphere \
  --lines Fe12_195.1190 \
  --abundance sun_coronal_2021_chianti \
  --output-dir ./run/input \
  --slit-rest-time "40 s" \
  --slit-width "0.2 arcsec" \
  --temp-dir temp \
  --temp-filename eosT \
  --rho-dir rho \
  --rho-filename result_prim_0 \
  --vz-dir vz \
  --vz-filename result_prim_2 \
  --time-dir time \
  --time-filename tau_slice_0.100 \
  --cube-shape 512 768 256 \
  --voxel-dx "0.192 Mm" \
  --voxel-dy "0.192 Mm" \
  --voxel-dz "0.064 Mm" \
  --vel-res "5.0 km/s" \
  --vel-lim "300.0 km/s" \
  --integration-axis z
```

**Dynamic Mode Options:**

- `--slit-rest-time`: Slit rest time per position - enables dynamic mode
- `--slit-width`: Slit width
- `--temp-dir`, `--rho-dir`, `--vx-dir`, `--vy-dir`, `--vz-dir`, `--time-dir`: Directories containing timestep files
- `--temp-filename`, `--rho-filename`, `--vx-filename`, `--vy-filename`, `--vz-filename`, `--time-filename`: Filename prefix before timestep suffix

## Output

The synthesis produces a pickle file containing:

- `line_cubes`: Individual NDCube objects for each spectral line with proper WCS
- `config`: Runtime configuration for reproducibility
- Additional technical data for internal use

## Performance tips

- Use `--downsample 2` or `--downsample 4` for initial testing
- Use `--precision float32` to reduce memory usage (may affect accuracy)
- Use spatial cropping to focus on regions of interest and reduce computation time
- Monitor memory usage - full resolution synthesis can require 50+ GB RAM
- Side views (`--integration-axis x` or `y`) may require different velocity files

## Working with synthesis results

The synthesis results can be loaded and analysed using the package API:

```python
import euvst_response

# Load synthesis results - this sums all line cubes into a single cube.
# Returns a (cube, dynamic_mode_info) tuple, so unpack it.
# The second argument is the reference line whose wavelength grid the other
# lines are interpolated onto; omit it and the first line in the file is used.
cube, dynamic_mode_info = euvst_response.load_atmosphere(
    "./run/input/synthesised_spectra.asdf", "Fe12_195.1190"
)
print(f"Combined cube shape: {cube.data.shape}")

# Access individual line cubes if needed
from euvst_response.io import load_results
data = load_results("./run/input/synthesised_spectra.asdf")

# Access individual line cubes
fe12_195 = data["line_cubes"]["Fe12_195.1190"]
print(f"Fe XII 195.119 cube shape: {fe12_195.data.shape}")
print(f"Rest wavelength: {fe12_195.meta['rest_wav']}")

# List all available lines
print(f"Available spectral lines: {list(data['line_cubes'].keys())}")
```
