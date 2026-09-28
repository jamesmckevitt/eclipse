---
search:
  exclude: true
---

# Pinhole stray light

A pinhole is a defect in the aluminium filter. It admits visible light the filter would otherwise block, and lets EUV through without the attenuation applied everywhere else. ECLIPSE models both, to support filter engineering rather than science runs. Pinholes are SWC only: any pinhole setting raises an error for EIS.

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

Both positions are fractions of the simulated window, not of the whole detector: 0.5 along the spectral axis is the middle of the wavelength range the synthesis covers, wherever that is on the CCD.

## What a pinhole does

The filter is 250 mm in front of the detector (`filter_distance`), in the beam converging on it. The light for each point of the detector crosses the filter as a cone, whose section there is the SW pupil scaled down: half of the 280 mm primary, cut along the slit, about 2 mm in radius at f/62 (`beam_footprint_radius`, from the primary's diameter, the pixel size and the plate scale; the optical design has the same plate scale along the dispersion as along the slit).

EUV. A pinhole passes, for every point whose cone covers it, the share (hole area / cone area) of that point's light that crosses it, without the foil's attenuation. The points it serves make a half disc of the image, about 150 rows in radius, beside the hole on the long-wavelength side. The light through the hole is part of the same wave as the light through the foil around it, so the two interfere. With `t` the filter's amplitude transmission, including the phase the layers put on the light (`AluminiumFilter.amplitude_transmission`, from the layers' indices of refraction), the hole adds `1 - |t|^2` of the light through it. Of that, `|1 - t|^2` is diffracted into the hole's Airy pattern about the point the light was heading for, which is exact whatever the hole's size, since the wave converges on that point. The rest, `2 Re[conj(t) (1 - t)]`, stays in the image, in the pixel the light was heading for. The pinhole's light is taken from the image as the focusing optics leave it.

Visible. Where the visible stray light comes from is not known, so it is taken to reach the filter head-on as a plane wave. The hole's light is then its near-field diffraction pattern, centred under the hole, which a hole approaching sqrt(wavelength x distance) across, 390 um at 600 nm, needs in place of the far-field Airy pattern. The foil passes about 1e-9 of the visible, so its interference with the hole's light, at most 1e-4 of it, is left out.

Both patterns are integrated over the pixels they land on, and the light that falls outside the window is lost rather than moved onto it.
