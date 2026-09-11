# Pinhole stray light

!!! warning "Instrument team topic"

    Pinhole modelling exists to support EUVST-SW filter engineering, not
    general science runs. ECLIPSE warns whenever it is enabled. Contact MSSL
    before drawing conclusions from it.

A pinhole is a defect in the aluminium filter. It admits visible light that the
filter would otherwise block, and it lets EUV through without the attenuation
the filter applies everywhere else. ECLIPSE models both, for SWC only: setting
`pinhole_sizes` or `enable_pinholes` under `instrument: EIS` raises.

## Configuration

```yaml
instrument: SWC

enable_pinholes: True
pinhole_sizes: [1 um, 5 um]            # diameters
pinhole_positions: [0.25, 0.5]         # fraction along the slit, 0 to 1
pinhole_positions_spectral: [0.1, 0.8] # fraction along the spectral axis, 0 to 1

simulation:
  vis_sl: 8.1e1 photon / (s * cm^2)    # visible stray light before the filter
```

The three pinhole lists are paired and must be the same length, one entry per
pinhole. `pinhole_positions_spectral` may be omitted entirely, in which case
every pinhole projects to the centre of the spectral window.

Because the spectral axis is wavelength, the spectral fraction is what decides
which emission lines a pinhole contaminates. The centre-of-window default
cannot answer that question, so set it explicitly whenever the answer matters.

## Most of a small pinhole's light misses the detector

This is the thing that most often surprises people, and it is why absolute
photon numbers from a pinhole are not intuitive.

A pinhole is an aperture, so its light arrives as an Airy pattern whose size is
set by the diameter. The first minimum sits at `1.22 * lambda * L / D`. For a
1 micron hole at the 250 mm filter-to-detector distance, in visible light, that
is **183 mm** -- against a detector a few mm across. Almost all of the light
transmitted by that hole lands somewhere other than the detector.

ECLIPSE therefore scales the pattern by its absolute normalisation rather than
by its sum over the detector array. For an aperture of area `A` at distance `L`
the fraction of transmitted power reaching one pixel of area `a` is
`A * a / (lambda * L)^2`, and the pattern simply integrates to less than one
over the detector when the rest of it falls outside.

The fraction that lands on a 128 x 256 pixel detector of 13.5 micron pixels, in
visible light:

| diameter | first Airy minimum | fraction on the detector |
| --- | --- | --- |
| 0.5 micron | 366 mm | 5.2e-05 |
| 1 micron | 183 mm | 2.1e-04 |
| 5 micron | 36.6 mm | 5.2e-03 |
| 20 micron | 9.15 mm | 7.9e-02 |
| 100 micron | 1.83 mm | 0.73 |

Only once the hole is large enough to bring the Airy disc inside the detector
does most of the light arrive. A budget that assumes otherwise overstates
pinhole stray light by orders of magnitude for the small holes.

!!! note "Versions before this behaviour changed"

    ECLIPSE used to normalise the visible pattern by its sum over the detector,
    which placed every transmitted photon on the detector regardless of
    geometry. Visible pinhole stray light from those versions is overstated by
    the reciprocal of the last column above, a factor of nearly 5000 for a
    1 micron hole. Recompute anything that depended on it.

## The EUV path still normalises by the array sum

The EUV correction in `apply_euv_pinhole_diffraction` has not been changed and
still distributes the whole transmitted signal across the detector. In the EUV
the Airy pattern is roughly thirty times smaller than in the visible, so the
error is smaller, but it is the same error:

| diameter | first minimum at 195 Angstrom | fraction on the detector |
| --- | --- | --- |
| 0.5 micron | 11.9 mm | 4.8e-02 |
| 1 micron | 5.95 mm | 1.7e-01 |
| 5 micron | 1.19 mm | 0.85 |

Five to twenty times too much for the small holes. Treat EUV pinhole numbers as
an upper bound until this is addressed.
