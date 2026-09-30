# Synthesising from an MHD simulation

`synthesise-spectra` turns a 3D MHD simulation into the spectra it emits, taking the plasma to be optically thin. For each pixel of the image, it adds up the emission measure of the cells along the line of sight by their temperature and velocity. It then multiplies that by each line's contribution function, G(T, n_e), which says how much the line emits at each temperature and density. The contribution functions come from the CHIANTI atomic database, through [fiasco](https://fiasco.readthedocs.io/).

The spectra are written to a synthesis file, which `eclipse` then observes (see [Running a simulation](instrument-response.md)).

## Atmosphere files

`synthesise-spectra` reads the simulation from an atmosphere file, an HDF5 file that you write from your simulation's output, and you pass it with `--atmosphere`:

```bash
synthesise-spectra --atmosphere atmosphere.h5 --lines Fe12_195.1190 --output-dir ./run/input
```

An atmosphere file holds the simulation's temperature, its mass density or electron density or both, the velocity along the line of sight, and the edges of its cells, each with its unit. It can also hold the snapshot's time. The cubes are indexed `[z, y, x]`, with z pointing up. The two axes that become the image must be evenly spaced, but the axis along the line of sight can have cells of different sizes. [Files](files.md#atmosphere-files) gives the full layout, for writing one with any HDF5 library, from Fortran, C, IDL or Julia.

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

- `--atmosphere`: The [atmosphere file](#atmosphere-files) to synthesise from (required, except in the deprecated modes on [Older versions](older-versions.md))
- `--output-dir`: The directory to write the synthesis file to (default: `./run/input`)
- `--output-name`: The synthesis file's name (default: `synthesised_spectra.h5`)

**Lines and abundances:**

- `--lines`: The lines to synthesise, for example `--lines Fe12_195.1190 Fe12_195.1790`, named as in [Naming spectral lines](line-names.md) (required)
- `--abundance`: The CHIANTI abundance set (default: `sun_coronal_2021_chianti`)
- `--n-workers`: How many processes compute the contribution functions, each taking one ion at a time, so there are never more than there are ions. `0`, the default, uses every CPU the job may use, as SLURM allocates them. `1` computes them one after another.
- `--hdf5-dbase-root`: The CHIANTI database for fiasco to read. The default is the one set in `~/.fiasco/fiascorc`. Use this to run with another CHIANTI version without changing that default.
- `--goft-temperature-chunk`: Compute the contribution functions for this many temperatures at a time, rather than the whole grid at once, to use less memory (default: the whole grid). With several ions and `--n-workers`, every worker needs that memory at the same time.

**Velocity grid:**

- `--vel-res`: The spacing of the velocity bins, with units (default: `"5.0 km/s"`). Each cell's emission is split between the two velocity bins either side of its velocity, and the two temperature bins either side of its temperature, so the emission's mean velocity is right at any spacing. A cell less than half a bin beyond the last bin goes into that bin.
- `--vel-lim`: How far the velocity grid reaches either side of zero, with units (default: `"300.0 km/s"`). Plasma faster than this is left out of the spectra, with a warning saying how much of the emission measure that is.

**Viewing direction:**

- `--integration-axis`: The axis to look along, and the side of the box to look from (default: `z`)
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

## Performance tips

- Use `--downsample 2` or `--downsample 4` for a first test
- Use `--precision float32` to use less memory, at some cost in accuracy
- Crop to the region you need, to save time
- Watch the memory: a full-resolution synthesis of a large box can need tens of GB
- Side views (`--integration-axis x`, `-x`, `y` or `+y`) need the atmosphere file to hold the velocity along that axis

## The output

`synthesise-spectra` writes a synthesis file. It holds each line's spectra over the image, which is what [`eclipse`](instrument-response.md) observes, and what the synthesis worked out on the way. [Files](files.md#synthesis-files) describes it, and how to read it in Python.
