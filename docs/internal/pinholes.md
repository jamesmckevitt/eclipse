---
search:
  exclude: true
---

# Pinhole stray light

A pinhole is a defect in the aluminium filter. It lets through visible light, which the filter would otherwise block, and EUV without the filter's attenuation. ECLIPSE models both, for filter engineering rather than for science runs. Pinholes are modelled for SWC only: any pinhole setting stops an EIS run with an error.

## Configuration

```yaml
instrument: SWC

pinhole_sizes: [1 um, 5 um]             # diameters
pinhole_positions: [0.25, 0.5]          # fraction along the slit, 0 to 1
pinhole_positions_spectral: [0.1, 0.8]  # fraction along the spectral axis, 0 to 1

simulation:
  enable_pinholes: True
  vis_sl: 3.0e16 photon / (s * cm^2)    # before the filter; about 81 photons per pixel per second after it
```

The pinhole lists have one entry per pinhole, so they must be the same length. If `pinhole_positions_spectral` is left out, every pinhole is at the middle of the spectral window. A pinhole's position along the spectral axis, which is wavelength, decides which lines its light falls on, so give it if that matters.
