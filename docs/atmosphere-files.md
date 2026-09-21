# Atmosphere files

`synthesise-spectra` reads the atmosphere it synthesises from one HDF5 file, whatever code produced it. The file holds the few things the synthesis needs, each with its units, so there is no reader to write for a new code: write the file from your own data with any HDF5 library, then

```bash
synthesise-spectra --atmosphere atmosphere.h5 --lines Fe12_195.1190 --output-dir ./run/input
```

Everything else on the [synthesis page](synthesis.md) works the same way: the lines, the velocity grid, the integration axis, cropping and downsampling. MURaM comes in through the same door: convert a snapshot once with [`eclipse-atmosphere from-muram`](#from-muram) and synthesise from the file.

## What the file holds

Root attributes:

| Attribute | Value |
| --- | --- |
| `format` | `eclipse-atmosphere` |
| `version` | `1` |
| `source` | Free text naming the simulation (optional). It is kept in the synthesis file. |

Datasets. Every one needs a `unit` attribute holding a unit string astropy can read, such as `K`, `g / cm3`, `cm / s`, `km / s`, `Mm` or `km`; the values can be in any unit of the right kind.

| Dataset | Shape | What it is |
| --- | --- | --- |
| `x_edges`, `y_edges`, `z_edges` | `(nx + 1,)`, `(ny + 1,)`, `(nz + 1,)` | The positions of the cell boundaries along each axis, increasing. |
| `temperature` | `(nz, ny, nx)` | |
| `mass_density` | `(nz, ny, nx)` | Give this, `electron_density`, or both. |
| `electron_density` | `(nz, ny, nx)` | |
| `velocity_x`, `velocity_y`, `velocity_z` | `(nz, ny, nx)` | The velocity along each of the box's own axes, positive towards increasing coordinate. Only the component along the axis you synthesise along (`--integration-axis`) is read, so a view from above needs `velocity_z` and a side view `velocity_x` or `velocity_y`. |
| `time` | scalar | The simulation time of the snapshot (optional). |

The cubes are stored `(nz, ny, nx)` in C order, so that `cube[k]` is a horizontal slice indexed `[y, x]` and z is height. If your code stores its arrays the other way round, transpose them before writing.

The two axes that end up as the image must be evenly spaced, because the image coordinates are written as a linear WCS. The axis along the line of sight may be stretched: the emission measure of each cell uses that cell's own size, so a chromosphere-to-corona grid whose cells grow with height integrates correctly seen from above. A stretched axis that would become an image axis is refused, so resample onto an even grid first if you want a side view of such a box.

## Writing one from Python

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
    z_edges=edges_from_centres(z_centres * u.km), # a stretched one, from cell centres
    time=1250.0 * u.s,
    source="my simulation, snapshot 385",
)
write_atmosphere(atmosphere, "atmosphere.h5")
```

`Atmosphere` checks the shapes, the units and the ordering of the edges as it is built, and `write_atmosphere` writes the layout above. `edges_from_centres` is for codes that know their cell centres rather than their cell boundaries: it places each edge halfway between two centres. Use your code's own boundaries when it has them, since on a stretched grid the two are not the same.

`read_atmosphere` reads a file back as an `Atmosphere`, and `eclipse-atmosphere info atmosphere.h5` prints what a file holds without loading all of it into Python.

## Writing one from anything else

Any HDF5 library can write the file: from Fortran or C inside a simulation code, from IDL, or from Julia. Create the datasets and attributes listed above at the root of the file. In `h5dump` terms, a minimal file for a top-down view looks like this:

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

Fortran stores arrays column-major, so an array declared `(nx, ny, nz)` in Fortran is written to HDF5 as `(nz, ny, nx)`, which is what ECLIPSE expects. Write the cubes as float32 if you want the file to stay small: the synthesis converts whatever it reads to the precision `--precision` asks for, which is float64 unless you say otherwise, and works in that throughout. The file's own precision only sets how exactly the values themselves were recorded.

## From MURaM

MURaM writes its output as separate binary files, one per variable, which `eclipse-atmosphere from-muram` reads and writes as an atmosphere file:

```bash
eclipse-atmosphere from-muram \
  --data-dir ./data/atmosphere \
  --snapshot 0270000 \
  --cube-shape 512 768 256 \
  --voxel-dx "0.192 Mm" --voxel-dy "0.192 Mm" --voxel-dz "0.064 Mm" \
  --velocities z \
  --output ./data/atmosphere_0270000.h5
```

`--cube-shape` is the file's own `(nx nz ny)` order. `--velocities` chooses which components to include; each is 400 MB at full resolution, and a view from above needs only `z`. The snapshot time is read from `header/Header.<snapshot>` if that file exists, or given with `--time`. The box is placed where ECLIPSE has always placed a MURaM box, x and y centred on zero and z = 0 at the centre of the bottom cell, so `--crop-x`, `--crop-y` and `--crop-z` mean to the synthesis what they always did.

Convert a snapshot once and synthesise from it as often as you like, cropping and downsampling at synthesis as before.

## A worked example: Bifrost from the Hinode SDC Europe

The [Hinode Science Data Centre Europe](https://sdc.uio.no/search/simulations) publishes Bifrost and MURaM snapshots as FITS files, one variable per file, in SI units, with `lg` variables as base-10 logarithms and the non-uniform z grid in a FITS extension ([Carlsson et al. 2016](https://doi.org/10.1051/0004-6361/201527226), Sect. 5). This builds an atmosphere file from the enhanced-network run `en024048_hion`, which has a stretched vertical grid and its own electron density:

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

The files put z increasing upwards, so `uz` is positive upwards as ECLIPSE expects; the granulation confirms it, with hot cells rising at the surface. The box starts 2.4 Mm below the surface, so `--crop-z "0 Mm" "20 Mm"` keeps the part that emits.

## Electron density

The contribution functions need the electron density. A code that carries one, for instance from non-equilibrium hydrogen ionisation, should write `electron_density`, and it is used as given.

With only a `mass_density`, ECLIPSE divides it by the mass of plasma per free electron. By default that is worked out from the abundance set the synthesis uses (`--abundance`) for a fully ionised plasma, which is what the EUV lines ECLIPSE synthesises form in: about 1.16 atomic mass units per electron for coronal abundances. Cells too cool to be fully ionised come out with too high an electron density, but they emit none of those lines. `--mass-per-electron` sets a value by hand instead.

The public Bifrost snapshot of the worked example above carries its own electron density, from non-equilibrium hydrogen ionisation. Above 100,000 K the density derived from its mass density with the coronal value is within 3 per cent of the one the code carries, where 1.29, the value for a neutral gas, is 8 per cent off; below 20,000 K the derived density is several times too high, as expected, and those cells emit nothing in the EUV lines.

## Cropping and downsampling

`--crop-x`, `--crop-y` and `--crop-z` are ranges in the file's own coordinates, and keep every cell that any part of the range covers; a bound that falls on a cell boundary does not keep the cell beyond it. `--downsample` keeps every n-th cell along each axis and gives each kept cell the boundaries of the block of cells it stands for, on an even grid and a stretched one alike, so the box keeps its extent and the columns their depth.

## What the synthesis file records

A synthesis from an atmosphere file records, under `atmosphere`, the file's path, its `source`, its time, its shape, which axes are stretched and whether it gave an electron density; and under `config`, the mass per electron used and where it came from.
