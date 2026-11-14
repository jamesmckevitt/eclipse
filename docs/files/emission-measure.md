# Emission measure files

An emission measure file holds a DEM or a VDEM over an image: how much plasma there is at each temperature, and velocity, in each pixel. [From a DEM or VDEM](../emission/dem.md) synthesises the spectra it emits.

[From a DEM or VDEM](../emission/dem.md#writing-an-emission-measure-file) shows how to write one in Python, with `write_emission_measure`. This is the layout, for writing one with any HDF5 library.

The root has a `format` attribute of `eclipse-emission-measure`, a `version` of `1`, and optionally a `source` saying where the emission measure came from. Each dataset needs a `unit` attribute that astropy can read.

| Dataset | Shape | Description |
| --- | --- | --- |
| `emission_measure` | `(ny, nx, n_temperatures)`, or `(ny, nx, n_temperatures, n_velocities)` for a VDEM | The emission measure n_e^2 dh in each bin, such as in `cm-5`. It is not per kelvin or per km/s. |
| `temperature` | `(n_temperatures,)` | The centres of the temperature bins, evenly spaced in log10(T). |
| `x_edges`, `y_edges` | `(nx + 1,)`, `(ny + 1,)` | The pixel boundaries across and along the slit, evenly spaced, as lengths on the Sun or angles seen from 1 AU. |
| `velocity` | `(n_velocities,)` | The centres of a VDEM's velocity bins, increasing. A positive velocity is a redshift. A DEM has none. |
| `electron_density` | scalar, `(ny, nx)` or `(ny, nx, n_temperatures)` | The electron density the contribution functions are taken at (optional). |
| `time` | scalar | Time of the snapshot (optional). |

`read_emission_measure` reads one back.
