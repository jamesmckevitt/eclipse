# Atmosphere files

An atmosphere file holds one snapshot of a simulation: its temperature, density and velocity on a grid of cells. [From an MHD simulation](../emission/mhd.md) synthesises the spectra it emits, and a [series of them](../observe/rasters.md) can be observed as the slit steps across.

## What one holds

This is the layout, for writing one with any HDF5 library. [Writing one in Python](#writing-one-in-python) does it for you.

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

## Writing one in Python

```python
import astropy.units as u
import numpy as np
from eclipse import Atmosphere, write_atmosphere, edges_from_centres

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

## Writing one from other languages

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

## Reading one

`read_atmosphere` reads a file back into an `Atmosphere`. To check what a file holds without loading the cubes, run

```bash
eclipse-atmosphere info atmosphere.h5
```
