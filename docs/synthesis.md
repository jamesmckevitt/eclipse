# Synthesising from an MHD simulation

`synthesise-spectra` turns a 3D MHD simulation into the spectra it emits, taking the plasma to be optically thin. For each pixel of the image, it adds up the emission measure of the cells along the line of sight by their temperature and velocity. It then multiplies that by each line's contribution function, G(T, n_e), which says how much the line emits at each temperature and density. The contribution functions come from the CHIANTI atomic database, through [fiasco](https://fiasco.readthedocs.io/).

The spectra are written to a synthesis file, which `eclipse` then observes (see [Simulating a single snapshot](instrument-response.md)).

## Atmosphere files

`synthesise-spectra` reads the simulation from an atmosphere file, an HDF5 file laid out as below. You write it from your simulation's output, in Python with `write_atmosphere` or with any HDF5 library, and pass it with `--atmosphere`:

```bash
synthesise-spectra --atmosphere atmosphere.h5 --lines Fe12_195.1190 --output-dir ./run/input
```

### Layout

Root attributes:

| Attribute | Value |
| --- | --- |
| `format` | `eclipse-atmosphere` |
| `version` | `1` |
| `source` | A description of the simulation (optional). It is copied into the synthesis file. |

Datasets. Each one needs a `unit` attribute that astropy can read, such as `K`, `g / cm3`, `kg / m3`, `cm / s`, `km / s`, `Mm` or `km`. Any unit of the right kind will do.

| Dataset | Shape | Description |
| --- | --- | --- |
| `x_edges`, `y_edges`, `z_edges` | `(nx + 1,)`, `(ny + 1,)`, `(nz + 1,)` | Positions of the cell boundaries along each axis, in increasing order. |
| `temperature` | `(nz, ny, nx)` | |
| `mass_density` | `(nz, ny, nx)` | At least one of `mass_density` and `electron_density` is needed. |
| `electron_density` | `(nz, ny, nx)` | |
| `velocity_x`, `velocity_y`, `velocity_z` | `(nz, ny, nx)` | Velocity along each axis of the box, positive towards increasing coordinate. Only the component along `--integration-axis` is read: `velocity_z` for a view from above, `velocity_x` or `velocity_y` for a side view. |
| `time` | scalar | Time of the snapshot (optional). |

The cubes are stored in C order with z first, so `cube[k]` is a horizontal slice indexed `[y, x]`, and z points up. If your code stores its arrays in a different order, transpose them before writing.

The two axes that become the image must be evenly spaced. The axis along the line of sight can have cells of different sizes.

### Writing a file from Python

```python
import astropy.units as u
import numpy as np
from euvst_response import Atmosphere, write_atmosphere, edges_from_centres

atmosphere = Atmosphere(
    temperature=temperature * u.K,                # (nz, ny, nx)
    mass_density=density * u.g / u.cm**3,         # or electron_density=n_e / u.cm**3
    velocity_z=vz * u.km / u.s,                   # the component along the line of sight
    x_edges=np.arange(nx + 1) * 0.192 * u.Mm,     # an even grid
    y_edges=np.arange(ny + 1) * 0.192 * u.Mm,
    z_edges=edges_from_centres(z_centres * u.km), # an uneven one, from cell centres
    time=1250.0 * u.s,
    source="my simulation, snapshot 385",
)
write_atmosphere(atmosphere, "atmosphere.h5")
```

`Atmosphere` checks the shapes, units and edges when it is created. If your code only gives the cell centres, `edges_from_centres` puts each edge halfway between two neighbouring centres.

`read_atmosphere` reads a file back into an `Atmosphere`. To check what a file holds without loading the cubes, run

```bash
eclipse-atmosphere info atmosphere.h5
```

### Writing a file from other languages

Any HDF5 library can write the file, for example from Fortran or C inside the simulation code, or from IDL or Julia. Put the datasets and attributes listed above at the root of the file. A minimal file for a view from above looks like this in `h5dump`:

```text
HDF5 "atmosphere.h5" {
GROUP "/" {
   ATTRIBUTE "format"  { "eclipse-atmosphere" }
   ATTRIBUTE "version" { 1 }
   DATASET "x_edges"      { DATATYPE H5T_IEEE_F64LE DATASPACE SIMPLE { ( 513 ) }
                            ATTRIBUTE "unit" { "Mm" } }
   DATASET "y_edges"      { ... ( 257 ) ... ATTRIBUTE "unit" { "Mm" } }
   DATASET "z_edges"      { ... ( 769 ) ... ATTRIBUTE "unit" { "Mm" } }
   DATASET "temperature"  { DATATYPE H5T_IEEE_F32LE DATASPACE SIMPLE { ( 768, 256, 512 ) }
                            ATTRIBUTE "unit" { "K" } }
   DATASET "mass_density" { ... ( 768, 256, 512 ) ... ATTRIBUTE "unit" { "g / cm3" } }
   DATASET "velocity_z"   { ... ( 768, 256, 512 ) ... ATTRIBUTE "unit" { "cm / s" } }
}
}
```

Fortran stores arrays column by column, so an array declared `(nx, ny, nz)` in Fortran is written to HDF5 as `(nz, ny, nx)`, which is what ECLIPSE expects. The cubes can be float32, which halves the size of the file. The synthesis converts them to the precision set by `--precision`, float64 by default, when it reads them.

### Electron density

The contribution functions need the electron density. If your code calculates one, for example with non-equilibrium hydrogen ionisation, write it as `electron_density` and ECLIPSE will use it as it is.

If the file only has `mass_density`, ECLIPSE divides it by the mass per free electron. By default ECLIPSE works this out from the abundances chosen with `--abundance`, taking the plasma to be fully ionised. For coronal abundances it is about 1.16 atomic mass units per electron. You can set it yourself with `--mass-per-electron`.

The contribution functions are computed on a grid of densities, 0.3 apart in log10 n_e, that covers the densities of the plasma between 10^4 and 10^9 K. An atmosphere with a wide range of densities therefore takes longer and needs more memory.

### Cropping and downsampling

`--crop-x`, `--crop-y` and `--crop-z` are given in the coordinates of the file. A cell is kept if any part of it is inside the range. `--downsample N` keeps every N-th cell along each axis, and makes each one as big as the N cells it replaces, so the box keeps its size.

The synthesis file records the atmosphere file's path, its `source` and `time`, and the mass per electron that was used.

### Worked example: a MURaM flare

The [Hinode SDC Europe](https://sdc.uio.no/search/simulations) hosts snapshots of several MURaM and Bifrost simulations as FITS files, one file per variable. The values are in SI units, variables whose names start with `lg` are base-10 logarithms, and the heights of the cell centres are in the first FITS extension ([Carlsson et al. 2016](https://doi.org/10.1051/0004-6361/201527226), Sect. 5).

This example uses the flare simulation of [Cheung et al. (2019)](https://doi.org/10.1038/s41550-018-0629-3), run `ar098192`, at snapshot 300000, during the flare. Download the temperature, density and vertical velocity (400 MB each):

```bash
for variable in lgtg lgr uz; do
  curl -O https://sdc.uio.no/vol/simulations/ar098192/atmos/MURaM_ar098192_${variable}_300000.fits
done
```

Then write them to an atmosphere file:

```python
import astropy.units as u
import numpy as np
from astropy.io import fits
from euvst_response import Atmosphere, write_atmosphere, edges_from_centres

def variable(name, snapshot=300000):
    with fits.open(f"MURaM_ar098192_{name}_{snapshot}.fits") as hdul:
        return hdul[0].data, hdul[0].header, hdul[1].data  # cube (nz, ny, nx), header, z centres in Mm

lgtg, header, z = variable("lgtg")
lgr, _, _ = variable("lgr")
uz, _, _ = variable("uz")

nz, ny, nx = lgtg.shape
x = (header["CRVAL1"] + (np.arange(nx) + 1 - header["CRPIX1"]) * header["CDELT1"]) * u.Mm
y = (header["CRVAL2"] + (np.arange(ny) + 1 - header["CRPIX2"]) * header["CDELT2"]) * u.Mm

atmosphere = Atmosphere(
    temperature=10.0 ** lgtg.astype(np.float64) * u.K,
    mass_density=10.0 ** lgr.astype(np.float64) * u.kg / u.m**3,
    velocity_z=uz * u.m / u.s,
    x_edges=edges_from_centres(x), y_edges=edges_from_centres(y),
    z_edges=edges_from_centres(z * u.Mm),
    time=header["ELAPSED"] * u.s,
    source="MURaM ar098192 snapshot 300000, Hinode SDC Europe",
)
write_atmosphere(atmosphere, "muram_300000.h5")
```

The box is 98 by 49 Mm, and runs from 7.5 Mm below the surface to 42 Mm above it. There is no electron density in these files, so ECLIPSE works it out from the mass density as described [above](#electron-density). To synthesise the flare line Fe XXIV 192.028 from the surface upwards:

```bash
synthesise-spectra --atmosphere muram_300000.h5 \
  --lines Fe24_192.0280 \
  --crop-z "0 Mm" "42 Mm" \
  --vel-lim "1000 km/s" --vel-res "10 km/s" \
  --output-dir ./run/input
```

### Worked example: Bifrost quiet Sun

This example uses the enhanced-network run `en024048_hion` of [Carlsson et al. (2016)](https://doi.org/10.1051/0004-6361/201527226), at snapshot 385. It has an uneven z axis and carries its own electron density. Download the temperature, density, electron density and vertical velocity (480 MB each):

```bash
for variable in lgtg lgr lgne uz; do
  curl -O https://sdc.uio.no/vol/simulations/en024048_hion/atmos/BIFROST_en024048_hion_${variable}_385.fits
done
```

Then write them to an atmosphere file:

```python
import astropy.units as u
import numpy as np
from astropy.io import fits
from euvst_response import Atmosphere, write_atmosphere, edges_from_centres

def variable(name, snapshot=385):
    with fits.open(f"BIFROST_en024048_hion_{name}_{snapshot}.fits") as hdul:
        return hdul[0].data, hdul[0].header, hdul[1].data  # cube (nz, ny, nx), header, z centres in Mm

lgtg, header, z = variable("lgtg")
lgr, _, _ = variable("lgr")
lgne, _, _ = variable("lgne")
uz, _, _ = variable("uz")

nz, ny, nx = lgtg.shape
x = (header["CRVAL1"] + (np.arange(nx) + 1 - header["CRPIX1"]) * header["CDELT1"]) * u.Mm
y = (header["CRVAL2"] + (np.arange(ny) + 1 - header["CRPIX2"]) * header["CDELT2"]) * u.Mm

atmosphere = Atmosphere(
    temperature=10.0 ** lgtg.astype(np.float64) * u.K,
    mass_density=10.0 ** lgr.astype(np.float64) * u.kg / u.m**3,
    electron_density=10.0 ** lgne.astype(np.float64) * u.m**-3,
    velocity_z=uz * u.m / u.s,
    x_edges=edges_from_centres(x), y_edges=edges_from_centres(y),
    z_edges=edges_from_centres(z * u.Mm),
    time=header["ELAPSED"] * u.s,
    source="Bifrost en024048_hion snapshot 385, Hinode SDC Europe",
)
write_atmosphere(atmosphere, "bifrost_385.h5")
```

The box starts 2.4 Mm below the surface, so `--crop-z "0 Mm" "20 Mm"` keeps the part that emits:

```bash
synthesise-spectra --atmosphere bifrost_385.h5 \
  --lines Fe09_171.0730 \
  --crop-z "0 Mm" "20 Mm" \
  --output-dir ./run/input
```

## Basic usage

```bash
# The shortest run
synthesise-spectra \
  --atmosphere ./data/atmosphere.h5 \
  --lines Fe12_195.1190 Fe12_195.1790 \
  --output-dir ./run/input

# The same, with more of the options, most at their defaults
synthesise-spectra \
  --atmosphere ./data/atmosphere.h5 \
  --lines Fe12_195.1190 Fe12_195.1790 \
  --abundance sun_coronal_2021_chianti \
  --n-workers 4 \
  --output-dir ./run/input \
  --output-name synthesised_spectra.h5 \
  --vel-res "5.0 km/s" \
  --vel-lim "300.0 km/s" \
  --integration-axis z \
  --crop-x "-50 Mm" "50 Mm" \
  --crop-y "-50 Mm" "50 Mm" \
  --crop-z "0 Mm" "20 Mm" \
  --downsample 1 \
  --precision float64 \
  --mass-per-electron 1.16

# Show all available options
synthesise-spectra --help
```

## Command line options

**Input and output:**

- `--atmosphere`: The [atmosphere file](#atmosphere-files) to synthesise from (required, except in dynamic mode)
- `--output-dir`: The directory to write the synthesis file to (default: `./run/input`)
- `--output-name`: The synthesis file's name (default: `synthesised_spectra.h5`)

**Lines and abundances:**

- `--lines`: The lines to synthesise, for example `--lines Fe12_195.1190 Fe12_195.1790` (required)
- `--abundance`: The CHIANTI abundance set (default: `sun_coronal_2021_chianti`)
- `--n-workers`: How many processes compute the contribution functions, each taking one ion at a time, so there are never more than there are ions. `0`, the default, uses every CPU the job may use, as SLURM allocates them. `1` computes them one after another.
- `--hdf5-dbase-root`: The CHIANTI database for fiasco to read. The default is the one set in `~/.fiasco/fiascorc`. Use this to run with another CHIANTI version without changing that default.
- `--goft-temperature-chunk`: Compute the contribution functions for this many temperatures at a time, rather than the whole grid at once, to use less memory (default: the whole grid). With several ions and `--n-workers`, every worker needs that memory at the same time.

**Velocity grid:**

- `--vel-res`: The spacing of the velocity bins, with units (default: `"5.0 km/s"`). Each cell's emission is split between the two velocity bins either side of its velocity, and the two temperature bins either side of its temperature, so the emission's mean velocity is right at any spacing. A cell less than half a bin beyond the last bin goes into that bin.
- `--vel-lim`: How far the velocity grid reaches either side of zero, with units (default: `"300.0 km/s"`). Plasma faster than this is left out of the spectra, with a warning saying how much of the emission measure that is.

**Viewing direction:**

- `--integration-axis`: The axis to look along: `x`, `y`, or `z` (default: `z`)
    - `z`: The view from above (+z), looking down through the height
    - `x`: The side view from +x, looking towards decreasing x; a flow towards +x is blueshifted
    - `y`: The side view from -y, looking towards increasing y; a flow towards -y is blueshifted

**Cropping, in the file's coordinates, with units:**

- `--crop-x`: The range of x to keep, for example `--crop-x "-50 Mm" "50 Mm"`
- `--crop-y`: The range of y to keep, for example `--crop-y "-50 Mm" "50 Mm"`
- `--crop-z`: The range of z to keep, for example `--crop-z "0 Mm" "20 Mm"`
- Leave out a crop option to keep the whole range along that axis

**Processing:**

- `--downsample`: Keep every N-th cell along each axis. N must divide every dimension of the atmosphere (default: `1`, no downsampling)
- `--precision`: `float32` or `float64` (default: `float64`)
- `--mass-per-electron`: The mass per free electron in atomic mass units, used to get the electron density from the mass density (default: worked out from `--abundance` for a fully ionised plasma, about 1.16 for coronal abundances; see [Electron density](#electron-density)). Not used if the atmosphere file has an electron density. `--mean-mol-wt` is the old name for this option, which defaulted to 1.29 up to ECLIPSE 0.11.0.

## Naming spectral lines

Lines are named `<Element><Stage>_<Wavelength>`, for example `Fe12_195.1190`:

- `Fe` - element symbol, capitalised as usual (`Fe`, `Si`, `S`, `O`).
- `12` - ionisation stage as an **arabic** numeral, in spectroscopic notation, so `Fe12` is Fe XII, not Fe XI or Fe XIII.
- `195.1190` - rest wavelength in Angstrom.

The same names are used by `--lines`, by the `reference_line` key in the instrument configuration, and as the keys of `line_cubes` in the output file.

ECLIPSE picks the line of that ion whose CHIANTI wavelength, rounded to as many decimals as the name has, equals the name's wavelength. So `Fe12_195.119` and `Fe12_195.12` both name Fe XII 195.119. Lines that CHIANTI has only a theoretical wavelength for are named the same way, by that wavelength. If an observed and a theoretical line both match, the observed one is taken. If several transitions match, the brightest is taken. The line is synthesised at CHIANTI's wavelength, whatever the digits of the name.

If no line of the ion matches, as can happen with a name from another line list, ECLIPSE takes the nearest line with an observed wavelength, and warns. It skips theoretical wavelengths here, because many of them are weak transitions within a few mA of strong lines. It prints the requested and matched wavelength of every line:

```text
  Fe12_195.1190: requested 195.1190 Angstrom, matched 195.1190 Angstrom, observed (delta=0.0000 Angstrom)
```

Check these. A large difference means the line you meant is not in CHIANTI for that ion, and a neighbouring one was taken instead. Two names that match the same line are refused, since the line would be synthesised twice. A name that does not follow the pattern is refused straight away.

## Performance tips

- Use `--downsample 2` or `--downsample 4` for a first test
- Use `--precision float32` to use less memory, at some cost in accuracy
- Crop to the region you need, to save time
- Watch the memory: a full-resolution synthesis of a large box can need tens of GB
- Side views (`--integration-axis x` or `y`) need the atmosphere file to hold the velocity along that axis

## Working with synthesis results

The synthesis file holds each line's spectra over the image, which is what [`eclipse`](instrument-response.md) observes. It also holds what the synthesis worked out on the way: the DEM, the emission measure by temperature and velocity, the contribution functions, and the settings it ran with. `load_synthesis` reads all of it:

```python
import euvst_response

data = euvst_response.load_synthesis("./run/input/synthesised_spectra.h5")

# A line cube for each line, indexed [y, x, wavelength]
fe12_195 = data["line_cubes"]["Fe12_195.1190"]
print(f"Fe XII 195.119 cube shape: {fe12_195.data.shape}")
print(f"Rest wavelength: {fe12_195.meta['rest_wav']}")
print(f"Available spectral lines: {list(data['line_cubes'])}")

# What the synthesis worked out on the way
print(f"DEM map (y, x, logT): {data['dem_map'].shape}, on log T {data['logT_grid']}")
print(f"Settings: {data['config']}")
```

`read_synthesis` reads just the spectra, and `read_synthesis_products` just the rest, or only the parts named in `keys`. `load_synthesis` gives the spectra as line cubes, which need evenly spaced wavelengths, as ECLIPSE's own synthesis gives them. For a file from [another code](other-codes.md) with uneven wavelengths, use `read_synthesis`, which keeps each line's wavelengths as they are.

??? note "Synthesis files from ECLIPSE 0.11.0 and earlier"

    ECLIPSE 0.11.0 and earlier wrote the synthesis as a pickle. `eclipse` still reads one, and `synthesise-spectra` still writes one if `--output-name` ends in `.pkl`, both with a warning, until a future release stops them. `euvst_response.convert_synthesis_pickle("old.pkl", "new.h5")` rewrites a pickle as a synthesis file, keeping everything it held.

??? note "Reading MURaM's own files (deprecated)"

    Command lines from ECLIPSE 0.11.0 and earlier, which read MURaM's binary files directly, still run without `--atmosphere`, with a warning, until a future release removes them:

    ```bash
    synthesise-spectra \
      --data-dir ./data/atmosphere \
      --temp-file temp/eosT.0270000 \
      --rho-file rho/result_prim_0.0270000 \
      --vz-file vz/result_prim_2.0270000 \
      --cube-shape 512 768 256 \
      --voxel-dx "0.192 Mm" --voxel-dy "0.192 Mm" --voxel-dz "0.064 Mm" \
      --lines Fe12_195.1190 \
      --output-dir ./run/input
    ```

    - `--data-dir`: Directory the file names are relative to (default: `data/atmosphere`)
    - `--temp-file`, `--rho-file`: Temperature and density files (default: `temp/eosT.0270000`, `rho/result_prim_0.0270000`)
    - `--vx-file`, `--vy-file`, `--vz-file`: Velocity files; only the one along `--integration-axis` is read (default: `vx/result_prim_1.0270000`, `vy/result_prim_3.0270000`, `vz/result_prim_2.0270000`)
    - `--cube-shape`: Cube dimensions in the order the files store them, `(nx nz ny)` (default: `512 768 256`)
    - `--voxel-dx`, `--voxel-dy`, `--voxel-dz`: Cell sizes (default: `"0.192 Mm"`, `"0.192 Mm"`, `"0.064 Mm"`)

    In this mode x and y are centred on zero, and z = 0 is the centre of the bottom cell. `--crop-x`, `--crop-y` and `--crop-z` are measured from there.

??? note "Dynamic mode (deprecated)"

    Dynamic mode synthesises a raster over a time series of MURaM snapshots in `synthesise-spectra`, with the slit width and exposure fixed at synthesis. It still runs, with a warning, until a future release removes it. A time series of atmosphere files is now observed by the instrument run instead, as described in [Simulating a time series](time-series.md).

    ```bash
    synthesise-spectra \
      --data-dir ./data/atmosphere \
      --lines Fe12_195.1190 \
      --slit-rest-time "40 s" \
      --slit-width "0.2 arcsec" \
      --cube-shape 512 768 256 \
      --voxel-dx "0.192 Mm" --voxel-dy "0.192 Mm" --voxel-dz "0.064 Mm" \
      --output-dir ./run/input
    ```

    - `--slit-rest-time`: Time the slit rests at each position, which turns dynamic mode on
    - `--slit-width`: Slit width
    - `--temp-dir`, `--rho-dir`, `--vx-dir`, `--vy-dir`, `--vz-dir`, `--time-dir`: Directory of each quantity's files, relative to `--data-dir` (default: `temp`, `rho`, `vx`, `vy`, `vz` and `header`)
    - `--temp-filename`, `--rho-filename`, `--vx-filename`, `--vy-filename`, `--vz-filename`, `--time-filename`: File name before the snapshot suffix (default: `eosT`, `result_prim_0`, `result_prim_1`, `result_prim_3`, `result_prim_2` and `Header`)
    - `--cube-shape`, `--voxel-dx`, `--voxel-dy`, `--voxel-dz`: As for reading MURaM's own files above

    The instrument run on the synthesis file has to use the same slit width, and an exposure equal to `--slit-rest-time`.
