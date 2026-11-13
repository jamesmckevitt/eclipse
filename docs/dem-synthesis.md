# Synthesising from a DEM or VDEM

A differential emission measure (DEM) says how much emitting plasma there is at each temperature. You might have one from an inversion of real observations. A velocity differential emission measure (VDEM) also splits the plasma by its velocity along the line of sight, and is made from an MHD simulation. From either, ECLIPSE can synthesise the lines the plasma emits and simulate how the instrument would measure them.

## How it works

ECLIPSE's own synthesis works in two steps. It first adds up the emission measure in each pixel by temperature and line-of-sight velocity, which is a VDEM. It then multiplies that by each line's contribution function, G(T, n_e), to get the spectrum. A DEM or a VDEM goes straight into the second step.

A DEM differs from an MHD simulation in two ways:

- **No velocities.** All of the emission is at rest, so the lines are not Doppler shifted.
- **No densities.** G(T, n_e) is taken at an electron density, or an electron pressure, that you choose.

A VDEM has the velocities, so only the second of these applies. It is covered [below](#with-velocities-a-vdem), after the DEM.

## Writing an emission measure file

Write the DEM as an emission measure file first. This one holds a made-up Gaussian DEM, the same in each of four pixels. Replace it with your own.

```python
import numpy as np
import astropy.units as u
from eclipse import EmissionMeasure, write_emission_measure

# 1. The observed DEM: a logT grid and DEM(T) in cm^-5 K^-1.
#    Replace this Gaussian with your own.
logT = np.linspace(4.0, 8.0, 101)
dem_per_K = 1.0e21 * np.exp(-0.5 * ((logT - 6.2) / 0.15) ** 2)

# 2. From DEM per kelvin to the emission measure in each log10(T) bin (cm^-5).
dlogT = logT[1] - logT[0]
bin_width = 10.0 ** (logT + dlogT / 2) - 10.0 ** (logT - dlogT / 2)
em = dem_per_K * bin_width

# 3. Copy the DEM into every pixel of a small scene, with 1 arcsec pixels.
ny, nx = 2, 2
measure = EmissionMeasure(
    emission_measure=np.tile(em, (ny, nx, 1)) / u.cm**5,   # (y, x, temperature)
    temperature=10.0 ** logT * u.K,
    x_edges=np.arange(nx + 1) * u.arcsec,
    y_edges=np.arange(ny + 1) * u.arcsec,
    source="A Gaussian DEM",
)
write_emission_measure(measure, "dem.h5")
```

[Files](files.md#emission-measure-files) gives the file's layout.

## Synthesising and observing it

A [run file](run-files.md) with `emission_measure` in place of `atmosphere` synthesises the file and observes the result. This one synthesises Si X 258.374 (low FIP) and S X 264.230 (high FIP), and observes the Si X window with EIS:

```yaml
# dem.yaml
instrument: EIS
emission_measure: dem.h5
reference_line: Si10_258.3740   # observe the Si X 258 window

synthesis:
  lines: [Si10_258.3740, S10_264.2300]
  electron_density: 1e9 cm-3

n_iter: 500
ncpu: -1

simulation:
  slit_width: 1 arcsec          # the same as the scene's pixels, and EIS's pixels along the slit
  expos: [10 s, 30 s, 60 s]     # each exposure is simulated in turn
  psf: False
```

```bash
eclipse run dem.yaml
```

The synthesis is an ordinary synthesis file, `run/input/dem.h5`, and the results can be analysed as in the [worked example](worked-example.ipynb). `synthesise-spectra` makes the same synthesis on its own, with `--emission-measure` in place of `--atmosphere`:

```bash
synthesise-spectra \
  --emission-measure dem.h5 \
  --lines Si10_258.3740 S10_264.2300 \
  --electron-density "1e9 cm-3"
```

## Key points

**Units.** An observed DEM is usually per kelvin (cm^-5 K^-1). ECLIPSE wants the emission measure in each temperature bin instead (cm^-5). Multiply by each bin's width in kelvin, as in step 2.

**Temperatures.** `temperature` holds the centres of the bins, which must be evenly spaced in log10(T). The contribution functions are computed at exactly these temperatures.

**Density.** `electron_density` in the `synthesis:` section is the density G(T, n_e) is taken at. `pressure`, such as `1e15 K cm-3`, gives each temperature the density at that electron pressure instead. A density that differs between pixels, or between the temperatures of each pixel, goes in the file, as the `electron_density` of the `EmissionMeasure`, shaped `(y, x)` or `(y, x, temperature)`. A density or pressure in the `synthesis:` section replaces one in the file.

**Line names.** Lines are named as in the MHD route: see [Naming spectral lines](line-names.md).

**Scene size.** A scene can be as small as one pixel, `nx, ny = 1, 1`, for a single DEM.

**Continuum.** `continuum: True` in the `synthesis:` section adds the continuum under the lines' windows.

**Abundance.** `abundance` chooses the element abundances, which set how bright the low-FIP lines are compared with the high-FIP ones. For FIP studies, synthesise once with each set, for example `sun_coronal_2021_chianti` and `sun_photospheric_2021_asplund`. The ratio of a line's intensity in the two is the FIP bias you are trying to measure.

**The wavelength window.** Each line's wavelengths are a velocity grid converted with `lambda_0 (1 + v/c)`: that of `vel_res` and `vel_lim` in the `synthesis:` section, +/- 300 km/s every 5 km/s by default. Each line is integrated over the wavelength bins, so all of its light is kept.

## With velocities: a VDEM

A VDEM is made from an MHD simulation by sorting every cell along the line of sight by its temperature and velocity. The [MHD route](synthesis.md) makes one itself on the way to the spectra. Giving ECLIPSE a VDEM skips that step. That is worth doing when you already have one: a VDEM is far smaller than the simulation it came from.

A VDEM fills the velocity bins in, so the lines come out Doppler shifted and broadened by the motions along the line of sight, as well as thermally.

!!! note "This is not the route for another code's spectra"

    A VDEM describes the plasma, and ECLIPSE still works out the light it emits, taking it to be optically thin. If another code, such as Lightweaver or RH1.5D, has already synthesised the spectra, write them as a [synthesis file](other-codes.md) instead. ECLIPSE then skips the synthesis and puts the instrument in front of your spectra.

An emission measure file holds a VDEM when it has a `velocity`, the centres of the velocity bins, and an emission measure with one more axis for them:

```python
# The VDEM: a logT grid, a velocity grid, and the emission measure in each (logT, v) bin.
# Replace this with your own code's output.
logT = np.linspace(4.0, 8.0, 101)
velocity = np.arange(-300.0, 300.0 + 5.0, 5.0) * u.km / u.s

ny, nx = 2, 2
vdem = my_code_output(ny, nx, logT, velocity)   # cm^-5, shape (ny, nx, nT, nv)

measure = EmissionMeasure(
    emission_measure=vdem / u.cm**5,
    temperature=10.0 ** logT * u.K,
    velocity=velocity,
    x_edges=np.arange(nx + 1) * u.arcsec,
    y_edges=np.arange(ny + 1) * u.arcsec,
)
write_emission_measure(measure, "vdem.h5")
```

It is synthesised and observed like a DEM, with `emission_measure: vdem.h5` in the run file.

### Key points for a VDEM

**Bulk velocity only.** The synthesis adds the thermal width itself, from each temperature bin's temperature and the element's atomic weight. Your VDEM should therefore hold only the bulk motion along the line of sight. If it already includes the thermal motions, the lines will come out too wide.

**Velocity sign.** Each bin is shifted to `lambda = lambda_0 (1 + v/c)`, so a positive velocity gives a longer wavelength, a redshift.

**Units.** The emission measure is that in each bin (cm^-5), not per kelvin or per km/s. If your VDEM is per kelvin and per km/s, multiply by each bin's widths, as in [step 2 of the DEM example](#writing-an-emission-measure-file).

**Bin centres.** Both `temperature` and `velocity` hold the centres of the bins, not their edges.

**The wavelength window.** Each line's wavelengths are those of `vel_res` and `vel_lim`, as for a DEM, whatever the VDEM's own velocity bins. Set `vel_lim` wide enough for the fastest plasma, plus several thermal widths. ECLIPSE warns if the VDEM has plasma beyond it.

## A full-CCD frame

A run file with an `emission_measure` and a `frame:` section makes a [full-CCD frame](full-ccd-frame.md#from-an-mhd-simulation) as one with an `atmosphere` does: every line in the band is synthesised from the DEM or VDEM under the slit, and the frame is made from that.

```yaml
instrument: SWC
emission_measure: dem.h5

synthesis:
  electron_density: 1e9 cm-3

simulation:
  expos: 60 s
  psf: True

frame:
  slit_position: 1 arcsec
  continuum: True
```

```bash
eclipse frame dem.yaml
```

The contribution functions of the band's lines are worked out on the DEM's own temperatures, which takes a few hours for the whole band, and kept in `run/input/`, in a file named for those temperatures, for later frames of DEMs on them.
