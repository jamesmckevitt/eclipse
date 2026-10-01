# Files

ECLIPSE works with three kinds of file, all of them HDF5:

- An **atmosphere file** holds one snapshot of a simulation. `synthesise-spectra`, and a [time series](time-series.md), read it.
- A **synthesis file** holds the spectra an atmosphere emits. `synthesise-spectra` writes one, or you can write one from [another code's spectra](other-codes.md), and `eclipse` observes it.
- A **results file** holds what `eclipse` measured. It is written to `run/result/<config name>.h5`.

## Atmosphere files

[From an MHD simulation](synthesis.md#atmosphere-files) shows how to write one in Python, with `write_atmosphere`. This is the layout, for writing one with any HDF5 library.

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

### Writing one from other languages

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

### Reading one

`read_atmosphere` reads a file back into an `Atmosphere`. To check what a file holds without loading the cubes, run

```bash
eclipse-atmosphere info atmosphere.h5
```

## Synthesis files

A synthesis file is laid out much like an atmosphere file. Its root has a `format` attribute of `eclipse-synthesis`, a `version` of `1`, and optionally a `source` saying where the spectra came from. Each dataset has a `unit` attribute that astropy can read.

| Dataset | Shape | What it holds |
| --- | --- | --- |
| `x_edges` | `(nx + 1,)` | The pixel boundaries across the slit, evenly spaced |
| `y_edges` | `(ny + 1,)` | The pixel boundaries along the slit, evenly spaced |
| `time` | scalar | The time of the snapshot, which only a time series needs |
| `lines/<name>/intensity` | `(ny, nx, n_wavelength)` | The spectral radiance at each pixel and wavelength |
| `lines/<name>/wavelength` | `(n_wavelength,)` | The wavelengths, increasing |
| `lines/<name>/rest_wavelength` | scalar | The wavelength the line's Doppler shifts are measured from |

Each group under `lines` holds one line, under the name that `reference_line` uses for it in the instrument configuration. A group can also hold a whole spectral window with its blends, as most codes give it. The blends are then fitted with a `fitting` block, as in [Fitting blended lines](fitting.md).

Files that ECLIPSE writes itself hold more: each line's `atom` and `ion` as attributes, an `integration_axis` attribute on the root for the view, which is the axis it looked along, with a sign such as `-x` if it looked from the other side of the box, and a `synthesis` group of intermediate results. A file from another code can leave all of these out.

The intensity can be in any unit of spectral radiance, per wavelength or per frequency, in energy or in photons, for example `erg / (s cm2 sr Angstrom)`, `W / (m2 sr Hz)` or `ph / (s cm2 sr nm)`. The wavelengths don't have to be evenly spaced, so you can use a grid that is finer in the line cores. Each wavelength stands for the interval halfway to its neighbours, so the spacing should change gradually.

x runs across the slit, the direction a raster steps in, and y runs along it. The edges can be lengths on the Sun, such as `Mm`, or angles as seen from 1 AU, such as `arcsec`. If your code gives pixel centres, `edges_from_centres` places the edges halfway between them.

[Spectra from another code](other-codes.md#writing-one) shows how to write one in Python, with `write_synthesis`.

### Reading one

A synthesis file from ECLIPSE holds each line's spectra over the image, and also what the synthesis worked out on the way: the DEM, the emission measure by temperature and velocity, the contribution functions, and the settings it ran with. `load_synthesis` reads all of it:

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

## Results files

`eclipse` writes its results to `run/result/<config name>.h5`. They include:

- The first iteration's signals, in DN and in photons
- For each fitted component, by name, in each pixel: its first fit, the mean and standard deviation of its intensity, velocity and width over the iterations, and the number of fits that failed
- The truth to compare against: the fit to the spectra on the detector's pixels, with no noise and no PSF, weighted as the DN fits are
- Whether the DN fits were weighted, as `fit_weighted` (see [Weighting](fitting.md#weighting))
- The settings (`Detector`, `Telescope`, `Simulation`) of each combination
- The version of ECLIPSE and the git commit that made them

Read them with `load_instrument_response_results`, as described in [Analysing the results](analysis.md).
