# Synthesis files

A synthesis file holds the spectra leaving the Sun in each pixel, from [ECLIPSE](../emission/mhd.md) or [another code](../emission/other-codes.md#writing-one).

## What one holds

The root has `format` `eclipse-synthesis`, `version` `1` and an optional `source`. Each dataset has a `unit` attribute.

| Dataset | Shape | What it holds |
| --- | --- | --- |
| `x_edges` | `(nx + 1,)` | The pixel boundaries across the slit, evenly spaced |
| `y_edges` | `(ny + 1,)` | The pixel boundaries along the slit, evenly spaced |
| `time` | scalar | The snapshot's time, for a time series |
| `lines/<name>/intensity` | `(ny, nx, n_wavelength)` | The spectral radiance at each pixel and wavelength |
| `lines/<name>/wavelength` | `(n_wavelength,)` | The wavelengths, increasing |
| `lines/<name>/rest_wavelength` | scalar | The wavelength the line's Doppler shifts are measured from |

Each group under `lines` is named as `reference_line` names it, and can hold a whole window with its blends. ECLIPSE's own files also hold `continuum_<from>-<to>` groups with `--continuum`, one `band_<from>-<to>` group with `--band`, and a `synthesis` group of intermediate results.

The intensity can be any spectral radiance, such as `erg / (s cm2 sr Angstrom)` or `ph / (s cm2 sr nm)`. The wavelengths may be uneven, if the spacing changes gradually. x runs across the slit and y along it, as lengths on the Sun or angles from 1 AU.

## Reading one

`load_synthesis` reads the spectra and what the synthesis worked out on the way, such as the DEM and the settings:

```python
import eclipse

data = eclipse.load_synthesis("./run/input/muram_ar_fe12.h5")
fe12_195 = data["line_cubes"]["Fe12_195.1190"]   # indexed [y, x, wavelength]
dem = data["dem_map"]                            # indexed [y, x, log T]
```

For uneven wavelengths use `read_synthesis`, which reads just the spectra. `read_synthesis_products` reads just the rest.
