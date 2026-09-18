# Atmosphere files

`synthesise-spectra` can read the atmosphere it synthesises from one HDF5 file, whatever code produced it. The file holds the few things the synthesis needs, each with its units, so there is no reader to write for a new code: write the file from your own data with any HDF5 library, then

```bash
synthesise-spectra --atmosphere atmosphere.h5 --lines Fe12_195.1190 --output-dir ./run/input
```

Everything else on the [synthesis page](synthesis.md) works the same way: the lines, the velocity grid, the integration axis, cropping and downsampling. The MURaM file options are not needed, and giving one alongside `--atmosphere` is refused.

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
| `velocity_x`, `velocity_y`, `velocity_z` | `(nz, ny, nx)` | Positive towards increasing coordinate. Only the component along the line of sight is needed: `velocity_z` for the default top-down view. |
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

Fortran stores arrays column-major, so an array declared `(nx, ny, nz)` in Fortran is written to HDF5 as `(nz, ny, nx)`, which is what ECLIPSE expects. Write the cubes as float32 if you want the file to stay small; the synthesis works in its own precision.

## From MURaM

The MURaM reader is now a converter into this format, with the same defaults as the synthesis options had:

```bash
eclipse-atmosphere from-muram \
  --data-dir ./data/atmosphere \
  --snapshot 0270000 \
  --cube-shape 512 768 256 \
  --voxel-dx "0.192 Mm" --voxel-dy "0.192 Mm" --voxel-dz "0.064 Mm" \
  --velocities z \
  --output ./data/atmosphere_0270000.h5
```

`--cube-shape` is the file's own `(nx nz ny)` order, as for `synthesise-spectra`. `--velocities` chooses which components to include; each is 400 MB at full resolution, and a top-down view needs only `z`. The snapshot time is read from `header/Header.<snapshot>` if that file exists, or given with `--time`. The box is placed where ECLIPSE has always placed a MURaM box, x and y centred on zero and z = 0 at the centre of the bottom cell, so `--crop-x`, `--crop-y` and `--crop-z` mean what they did before, and a synthesis from the converted file gives the same result as one from the raw files.

`synthesise-spectra` still reads the raw MURaM files directly, so nothing has to change for a MURaM run.

## Electron density

The contribution functions need the electron density. A code that carries one, for instance from non-equilibrium hydrogen ionisation, should write `electron_density`, and it is used as given.

With only a `mass_density`, ECLIPSE divides it by the mass of plasma per free electron. By default that is worked out from the abundance set the synthesis uses (`--abundance`) for a fully ionised plasma, which is what the EUV lines ECLIPSE synthesises form in: about 1.17 atomic mass units per electron for coronal abundances. Cells too cool to be fully ionised come out with too high an electron density, but they emit none of those lines. `--mass-per-electron` sets a value by hand instead.

!!! warning "Changed from ECLIPSE 0.8.0"

    Up to 0.8.0 the conversion used `--mean-mol-wt`, with a default of 1.29. That is the mean molecular weight of a neutral solar gas, not the mass per electron of an ionised one, and it made every emission measure from a mass density about 20 per cent too small. The default is now derived from the abundances; the old option name still works and still sets the same quantity. To reproduce an older run exactly, give `--mass-per-electron 1.29`.

## Cropping and downsampling

`--crop-x`, `--crop-y` and `--crop-z` are ranges in the file's own coordinates, and keep every cell that any part of the range covers; a bound that falls on a cell boundary does not keep the cell beyond it. `--downsample` keeps every n-th cell along each axis and gives each kept cell the boundaries of the block of cells it stands for, on an even grid and a stretched one alike, so the box keeps its extent and the columns their depth.

## What the synthesis file records

A synthesis from an atmosphere file records, under `atmosphere`, the file's path, its `source`, its time, its shape, which axes are stretched and whether it gave an electron density; and under `config`, the mass per electron used and where it came from.
