# Synthesis from a single intensity

The simplest input is no atmosphere at all: one spectral line of known integrated
intensity.

Unlike the [MHD](synthesis.md) and [DEM](dem-synthesis.md) routes, there is no
separate synthesis step and no synthesis file. Setting `uniform_intensity` in the
instrument configuration replaces the atmosphere entirely, so this is configured
at the [instrument response](instrument-response.md) stage and run with `eclipse`
directly.

```yaml
instrument: SWC
uniform_intensity: 5000 erg / (s cm2 sr)   # units are required
rest_wavelength: 195.119 AA                # default 195.119 AA
thermal_width: 20 km/s                     # 1-sigma velocity width, default 20 km/s

n_iter: 500
ncpu: -1

simulation:
  slit_width: [0.2 arcsec, 0.4 arcsec]
  expos: [5 s, 20 s, 80 s, 320 s]
  psf: True
```

```bash
eclipse --config configs/uniform.yaml
```

This builds a 1x1 pixel Gaussian line directly at the detector's spectral
resolution and runs the usual Monte Carlo over it.

## Why use it

Because the input intensity is exact and uniform, everything in the scatter of
the fitted results comes from the instrument. That makes it the cleanest way to
answer questions of the form "how precisely can this instrument measure a line
this bright, at this exposure?".

That is useful for building measurement uncertainty budgets - for example
propagating a line-intensity precision into the uncertainty on a FIP-bias ratio -
without committing to any particular atmosphere.

It is also much cheaper than the other two routes: there is no contribution
function to compute and no cube to carry around, so a broad sweep over exposure
time and slit width costs very little.
