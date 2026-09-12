# Pinhole stray light

A pinhole is a defect in the aluminium filter. It admits visible light the filter would otherwise block, and lets EUV through without the attenuation applied everywhere else. ECLIPSE models both, to support filter engineering rather than science runs. Pinholes are SWC only: setting `pinhole_sizes`, or `simulation.enable_pinholes: True`, raises an error for EIS.

## Configuration

```yaml
instrument: SWC

pinhole_sizes: [1 um, 5 um]             # diameters
pinhole_positions: [0.25, 0.5]          # fraction along the slit, 0 to 1
pinhole_positions_spectral: [0.1, 0.8]  # fraction along the spectral axis, 0 to 1

simulation:
  enable_pinholes: True
  vis_sl: 8.1e1 photon / (s * cm^2)     # visible stray light before the filter
```

The pinhole lists are paired, one entry per pinhole, and must be the same length. `pinhole_positions_spectral` can be left out, in which case every pinhole lands at the centre of the spectral window. The spectral axis is wavelength, so that fraction is what decides which lines a pinhole contaminates - set it if that matters.

## Small pinholes put most of their light off the detector

The smaller the hole, the wider the pattern it projects. A 1 um hole spreads visible light over 183 mm at the 250 mm filter-to-detector distance, so a detector a few mm across catches very little of it:

| diameter | first Airy minimum | fraction landing on the detector |
| --- | --- | --- |
| 0.5 um | 366 mm | 5.2e-05 |
| 1 um | 183 mm | 2.1e-04 |
| 5 um | 36.6 mm | 5.2e-03 |
| 20 um | 9.15 mm | 7.9e-02 |
| 100 um | 1.83 mm | 0.73 |

For 128 x 256 pixels of 13.5 um. Only a hole big enough to bring the pattern inside the detector delivers most of its light.
