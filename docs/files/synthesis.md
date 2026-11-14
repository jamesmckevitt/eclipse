# Synthesis files

A synthesis file holds the spectra that leave the Sun, in each pixel of the image. [ECLIPSE's own synthesis](../emission/mhd.md) writes one, or you can write one from [another code's spectra](../emission/other-codes.md#writing-one), and `eclipse` observes it.

## What one holds

A synthesis file is laid out much like an atmosphere file. Its root has a `format` attribute of `eclipse-synthesis`, a `version` of `1`, and optionally a `source` saying where the spectra came from. Each dataset has a `unit` attribute that astropy can read. A synthesis made by a [run file](../run-files.md#reusing-the-synthesis) also has a `run_key` attribute, which records what it was made from so that later runs can reuse it.

| Dataset | Shape | What it holds |
| --- | --- | --- |
| `x_edges` | `(nx + 1,)` | The pixel boundaries across the slit, evenly spaced |
| `y_edges` | `(ny + 1,)` | The pixel boundaries along the slit, evenly spaced |
| `time` | scalar | The time of the snapshot, which only a time series needs |
| `lines/<name>/intensity` | `(ny, nx, n_wavelength)` | The spectral radiance at each pixel and wavelength |
| `lines/<name>/wavelength` | `(n_wavelength,)` | The wavelengths, increasing |
| `lines/<name>/rest_wavelength` | scalar | The wavelength the line's Doppler shifts are measured from |

Each group under `lines` holds one line, under the name that `reference_line` uses for it in the instrument configuration. A group can also hold a whole spectral window with its blends, as most codes give it. The blends are then fitted with a `fitting` block, as in [Fitting blended lines](../observe/fitting.md).

Files that ECLIPSE writes itself hold more: each line's `atom` and `ion` as attributes, an `integration_axis` attribute on the root for the view, which is the axis it looked along, with a sign such as `-x` if it looked from the other side of the box, and a `synthesis` group of intermediate results. A file from another code can leave all of these out. A file synthesised with `--continuum` also holds the continuum under its windows, as groups of their own named `continuum_<from>-<to>`, in Angstrom, each covering a set of overlapping windows. A file synthesised with `--band` holds every line between two wavelengths as one group, named `band_<from>-<to>`, in place of the lines.

The intensity can be in any unit of spectral radiance, per wavelength or per frequency, in energy or in photons, for example `erg / (s cm2 sr Angstrom)`, `W / (m2 sr Hz)` or `ph / (s cm2 sr nm)`. The wavelengths don't have to be evenly spaced, so you can use a grid that is finer in the line cores. Each wavelength stands for the interval halfway to its neighbours, so the spacing should change gradually.

x runs across the slit, the direction a raster steps in, and y runs along it. The edges can be lengths on the Sun, such as `Mm`, or angles as seen from 1 AU, such as `arcsec`. If your code gives pixel centres, `edges_from_centres` places the edges halfway between them.

[From another code](../emission/other-codes.md#writing-one) shows how to write one in Python, with `write_synthesis`.

## Reading one

A synthesis file from ECLIPSE holds each line's spectra over the image, and also what the synthesis worked out on the way: the DEM, the emission measure by temperature and velocity, the electron density at each temperature of each pixel, the contribution functions, and the settings it ran with. `load_synthesis` reads all of it:

```python
import eclipse

data = eclipse.load_synthesis("./run/input/muram_ar_fe12.h5")

# A line cube for each line, indexed [y, x, wavelength]
fe12_195 = data["line_cubes"]["Fe12_195.1190"]
print(f"Fe XII 195.119 cube shape: {fe12_195.data.shape}")
print(f"Rest wavelength: {fe12_195.meta['rest_wav']}")
print(f"Available spectral lines: {list(data['line_cubes'])}")

# What the synthesis worked out on the way
print(f"DEM map (y, x, logT): {data['dem_map'].shape}, on log T {data['logT_grid']}")
print(f"Settings: {data['config']}")
```

`read_synthesis` reads just the spectra, and `read_synthesis_products` just the rest, or only the parts named in `keys`. `load_synthesis` gives the spectra as line cubes, which need evenly spaced wavelengths, as ECLIPSE's own synthesis gives them. For a file from [another code](../emission/other-codes.md) with uneven wavelengths, use `read_synthesis`, which keeps each line's wavelengths as they are.
