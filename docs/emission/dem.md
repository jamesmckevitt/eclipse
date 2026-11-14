# From a DEM or VDEM

<div class="glance" markdown>
<div markdown>
**You need**

An [emission measure file](../files/emission-measure.md), written from your DEM or VDEM
</div>
<div markdown>
**You run**

`eclipse run`, or `eclipse synthesise` for this step alone
</div>
<div markdown>
**You get**

A [synthesis file](../files/synthesis.md) in `run/input/`
</div>
<div markdown>
**Next**

[Simulating an observation](../observe/simulate.md) or [Full-CCD frames](../observe/full-ccd-frames.md)
</div>
</div>

A differential emission measure (DEM) says how much plasma there is at each temperature, such as from an inversion of observations. A velocity DEM (VDEM) also splits it by line-of-sight velocity. ECLIPSE synthesises the lines either emits, taking the plasma to be optically thin.

## Writing an emission measure file

```python
import numpy as np
import astropy.units as u
from eclipse import EmissionMeasure, write_emission_measure

# A Gaussian DEM in cm^-5 K^-1; replace it with your own.
logT = np.linspace(4.0, 8.0, 101)
dem_per_K = 1.0e21 * np.exp(-0.5 * ((logT - 6.2) / 0.15) ** 2)

# The emission measure in each temperature bin, in cm^-5.
dlogT = logT[1] - logT[0]
em = dem_per_K * (10.0 ** (logT + dlogT / 2) - 10.0 ** (logT - dlogT / 2))

measure = EmissionMeasure(
    emission_measure=np.tile(em, (2, 2, 1)) / u.cm**5,   # (y, x, temperature)
    temperature=10.0 ** logT * u.K,
    x_edges=np.arange(3) * u.arcsec,
    y_edges=np.arange(3) * u.arcsec,
)
write_emission_measure(measure, "dem.h5")
```

The temperatures are bin centres, evenly spaced in log10(T). [Files](../files/emission-measure.md) gives the file's layout.

## Synthesising and observing it

A [run file](../run-files.md) names it with `emission_measure` in place of `atmosphere`:

```yaml
instrument: EIS
emission_measure: dem.h5
reference_line: Si10_258.3740

synthesis:
  lines: [Si10_258.3740, S10_264.2300]
  electron_density: 1e9 cm-3

simulation:
  slit_width: 1 arcsec
  expos: [10 s, 30 s, 60 s]
  psf: False
```

```bash
eclipse run dem.yaml
```

The contribution functions are taken at `electron_density`, or at each temperature's density for a `pressure` such as `1e15 K cm-3`. A density in the file, shaped `(y, x)` or `(y, x, temperature)`, is used if neither is given.

## With velocities: a VDEM

A VDEM has a `velocity`, the centres of its velocity bins, and an emission measure with one more axis:

```python
# vdem: your emission measure in cm^-5, shaped (y, x, temperature, velocity)
measure = EmissionMeasure(
    emission_measure=vdem / u.cm**5,
    temperature=10.0 ** logT * u.K,
    velocity=np.arange(-300.0, 305.0, 5.0) * u.km / u.s,
    x_edges=np.arange(3) * u.arcsec,
    y_edges=np.arange(3) * u.arcsec,
)
```

The velocities are bulk motions only, positive away from the observer. The synthesis adds the thermal width. Set `vel_lim` in the `synthesis:` section wide enough for the fastest plasma.

## A full-CCD frame

A run file with an `emission_measure` and a `frame:` section makes a [full-CCD frame](../observe/full-ccd-frames.md) from it, as from an atmosphere:

```bash
eclipse frame dem.yaml
```
