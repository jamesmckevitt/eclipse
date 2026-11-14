# Atmosphere files

An atmosphere file holds one snapshot of a simulation: temperature, density and velocity on a grid of cells, for [synthesis](../emission/mhd.md) or a [raster](../observe/rasters.md).

## What one holds

Root attributes:

| Attribute | Value |
| --- | --- |
| `format` | `eclipse-atmosphere` |
| `version` | `1` |
| `source` | A description of the simulation (optional) |

Datasets, each with a `unit` attribute astropy can read, such as `K`, `g / cm3` or `km / s`:

| Dataset | Shape | Description |
| --- | --- | --- |
| `x_edges`, `y_edges`, `z_edges` | `(nx + 1,)`, `(ny + 1,)`, `(nz + 1,)` | Cell boundaries, increasing |
| `temperature` | `(nz, ny, nx)` | |
| `mass_density` | `(nz, ny, nx)` | This or `electron_density` is needed |
| `electron_density` | `(nz, ny, nx)` | |
| `velocity_x`, `velocity_y`, `velocity_z` | `(nz, ny, nx)` | Positive towards increasing coordinate; only those along the line of sight are needed |
| `time` | scalar | Time of the snapshot (optional) |

z points up. The two image axes must be evenly spaced; the line of sight need not be.

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

`edges_from_centres` puts each edge halfway between two centres.

## Writing one from other languages

Any HDF5 library can write it. A minimal file for a view from above, in `h5dump`:

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

A Fortran array `(nx, ny, nz)` is written as `(nz, ny, nx)`, as needed. float32 cubes halve the file.

## Reading one

`read_atmosphere` reads one into an `Atmosphere`. To see what a file holds:

```bash
eclipse-atmosphere info atmosphere.h5
```
