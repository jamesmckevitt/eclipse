# Synthesis from a single intensity

The simplest input is one spectral line of known integrated intensity.

Unlike the [MHD](synthesis.md) and [DEM](dem-synthesis.md) routes, there is no separate synthesis step and no synthesis file. Setting `uniform_intensity` in the instrument configuration replaces the atmosphere, so this is configured at the [instrument response](instrument-response.md) stage and run with `eclipse` directly.

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

This builds a single-pixel Gaussian line directly at the detector's spectral resolution and runs the usual Monte Carlo over it.

## Off-chip binning

[`offchip_bin_slit`](instrument-response.md#off-chip-slit-binning) works here too, even though there is no atmosphere to bin. ECLIPSE builds the cube with one slit pixel per binning factor, each holding the same intensity, so that the pixels summed at the binning step carry independent noise:

```yaml
uniform_intensity: 5000 erg / (s cm2 sr)
offchip_bin_slit: [1, 2, 4]   # 1, 2 and 4 slit pixels binned on the ground
```

The signal-to-noise ratio improves by roughly `sqrt(n)`, exactly as it does for a synthesised atmosphere.

## Why use it

Because the input intensity is exact and uniform, everything in the scatter of the fitted results comes from the instrument.

That is useful for building measurement uncertainty budgets, for example propagating a line-intensity precision into the uncertainty on a FIP-bias ratio.

It is also much cheaper than the other two routes as there is no contribution function to compute so a broad sweep over instrument configurations and exposure times can be done quickly.