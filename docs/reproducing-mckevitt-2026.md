# Reproducing McKevitt et al. (2026)

[McKevitt et al. (2026)](https://academic.oup.com/pasj/article/78/4/1524/8731000) puts a
snapshot of the [Cheung et al. (2018)](https://www.nature.com/articles/s41550-018-0629-3)
flare simulation through EUVST-SW. This page reproduces two of its figures from nothing but
the published simulation data:

- the Fe XII 195.119 intensity of the model atmosphere, in heliocentric coordinates, as it
  leaves the Sun; and
- the intensity and Doppler velocity EUVST would measure, in helioprojective coordinates,
  with the 0.4 arcsec slit and 2-pixel binning along the slit.

It is one snapshot and one instrument configuration. The paper swept eight exposure times and
seven slit and binning combinations, and ran Hinode/EIS alongside for comparison, but that is
this same pipeline run more times.

## Before you start

| Resource | Needed |
| --- | --- |
| Download | 1.2 GB of simulation data |
| Disk | about 5 GB, most of it the synthesis file |
| Memory | around 60 GB for the synthesis, much less for the instrument run |
| Wall clock | a few hours on one node |

Contribution functions come from CHIANTI through
[fiasco](https://fiasco.readthedocs.io/), which builds a local HDF5 copy of the database the
first time it is used. That download is a few GB on its own and only happens once.

## 1. Get the simulation snapshot

The atmosphere is in the Stanford Digital Repository at
[purl.stanford.edu/dv883vb9686](https://purl.stanford.edu/dv883vb9686). The deposit is 146
files and 66 GB, of which you need three.

An optically thin calculation needs the temperature, the density, and the velocity along the
line of sight. Looking down on the box, that is the vertical velocity. The magnetic field, the
pressure, the internal energy and the two horizontal velocity components are not used, so
there is no reason to download them.

| File | Quantity | Unit |
| --- | --- | --- |
| `eosT.0270000` | temperature | K |
| `result_prim_0.0270000` | density | g/cm^3 |
| `result_prim_2.0270000` | vertical velocity | cm/s |

`0270000` is the first of the twelve snapshots in the deposit, and the one the paper used. The
directory names below are arbitrary; they only have to match the `--temp-file`, `--rho-file`
and `--vz-file` paths in the next step.

```bash
mkdir -p data/atmosphere/{temp,rho,vz}
BASE=https://stacks.stanford.edu/file/druid:dv883vb9686

curl -L -o data/atmosphere/temp/eosT.0270000           "$BASE/eosT.0270000"
curl -L -o data/atmosphere/rho/result_prim_0.0270000   "$BASE/result_prim_0.0270000"
curl -L -o data/atmosphere/vz/result_prim_2.0270000    "$BASE/result_prim_2.0270000"
```

Each file should come out at exactly 402653184 bytes, which is 512 x 768 x 256 cells of
single-precision float. A short file means the download was truncated, and the reader will
raise a reshape error rather than a helpful one.

!!! warning "Which file holds the vertical velocity"

    `result_prim_1`, `_2` and `_3` are the three velocity components, but which of them is
    vertical depends on how the run was laid out, and it is not the same in every MURaM
    deposit. In this one the vertical axis is the 768-cell one and the vertical velocity is
    `result_prim_2`. Getting this wrong is quiet: the synthesis runs, the intensity map looks
    much the same, and only the Doppler velocities are wrong.

### The grid

`--cube-shape` is the order the cells are written in, so `512 768 256` for these files. The
`--voxel-d*` options describe the axes after the reader has put them in heliocentric order:
`dx` for the 512-cell axis, `dy` for the 256-cell axis, and `dz` for the 768-cell vertical
one. At 0.192, 0.192 and 0.064 Mm that is a box 98.3 by 49.2 Mm across and 49.2 Mm tall, with
the photosphere 7.5 Mm above the bottom.

## 2. Synthesise the spectra

```bash
synthesise-spectra \
  --data-dir ./data/atmosphere \
  --lines Fe13_194.9800 Fe14_194.9910 Fe13_194.9970 Fe12_195.0040 Fe11_195.0250 \
          Fe09_195.0290 Mn10_195.0360 Fe11_195.0540 Fe12_195.0860 Fe12_195.1190 \
          Fe11_195.1470 Fe10_195.1510 Fe13_195.1600 Ni11_195.1600 Fe12_195.1790 \
          Fe09_195.2300 Fe14_195.2450 Fe11_195.2450 Fe10_195.2600 Fe10_195.2610 \
          Fe12_195.2660 \
  --abundance sun_coronal_1992_feldman_ext \
  --output-dir ./run/input \
  --output-name mckevitt2026.pkl \
  --temp-file temp/eosT.0270000 \
  --rho-file rho/result_prim_0.0270000 \
  --vz-file vz/result_prim_2.0270000 \
  --cube-shape 512 768 256 \
  --voxel-dx "0.192 Mm" \
  --voxel-dy "0.192 Mm" \
  --voxel-dz "0.064 Mm" \
  --vel-res "5.0 km/s" \
  --vel-lim "300.0 km/s" \
  --integration-axis z
```

The line list is Fe XII 195.119 together with every other transition falling in the same
spectral window, so the blends end up in the profile that gets fitted rather than being left
out of it. The one that matters most is Fe XII 195.179, close enough to 195.119 to shift the
centroid if it is ignored, which is why it appears again as a tied component in the fit later.

`--integration-axis z` is the top-down view, and is why only the vertical velocity was needed.
A side view (`x` or `y`) integrates through the box horizontally and wants the matching
`--vx-file` or `--vy-file` instead.

Abundances make a real difference to the intensities. The paper synthesised the same snapshot
with three sets and compared them against an EIS observation;
`sun_coronal_1992_feldman_ext` is the one behind the figures reproduced here. Swap the
`--abundance` value to see the others.

!!! note "Check the matched wavelengths"

    Synthesis prints the requested and matched wavelength for every line. A large difference
    means CHIANTI has no such transition for that ion and a neighbouring one was picked up
    instead. See [naming spectral lines](synthesis.md#naming-spectral-lines).

## 3. The heliocentric intensity map

This is the atmosphere before the instrument: the emission integrated along the line of sight
and across the line profile, on the simulation's own grid.

```python
import astropy.units as u
import dill
import matplotlib.pyplot as plt
import numpy as np

with open("run/input/mckevitt2026.pkl", "rb") as f:
    synthesis = dill.load(f)

line_cube = synthesis["line_cubes"]["Fe12_195.1190"]

# Integrate the line profile. The cube is indexed (y, x, wavelength) and holds a
# spectral radiance, so the sum over the last axis needs the wavelength spacing.
cube = line_cube.to(u.erg / u.s / u.cm**2 / u.sr / u.cm)
wavelengths = cube.axis_world_coords(-1)[0].to(u.cm)
d_lambda = np.gradient(wavelengths)
intensity = (cube.data * d_lambda.value).sum(axis=-1) * u.erg / u.s / u.cm**2 / u.sr

y = line_cube.axis_world_coords(0)[0].to(u.Mm)
x = line_cube.axis_world_coords(1)[0].to(u.Mm)

fig, ax = plt.subplots(figsize=(9, 5))
image = ax.imshow(
    np.log10(intensity.value),
    origin="lower",
    extent=[x[0].value, x[-1].value, y[0].value, y[-1].value],
    cmap="afmhot",
    vmin=1.5,
    vmax=5.0,
    aspect="equal",
)
ax.set_xlabel("Heliocentric X [Mm]")
ax.set_ylabel("Heliocentric Y [Mm]")
fig.colorbar(image, ax=ax, label=r"$\log_{10}$ intensity [erg/s/cm$^2$/sr]")
fig.savefig("heliocentric_intensity.png", dpi=200, bbox_inches="tight")
```

No transpose is needed anywhere. Cubes are indexed `[row, column, wavelength]`, so summing the
last axis leaves an array that plots the right way up, and `axis_world_coords(0)` and
`axis_world_coords(1)` are the coordinates of those two axes in that order.

## 4. Simulate the instrument

Write this as `run/input/mckevitt2026.yaml`:

```yaml
instrument: SWC
synthesis_file: ./run/input/mckevitt2026.pkl
reference_line: Fe12_195.1190

n_iter: 25
ncpu: -1
offchip_bin_slit: 2
fit_signals: dn

simulation:
  slit_width: 0.4 arcsec
  expos: 40 s
  psf: True

detector:
  ccd_temperature: -60 Celsius

filter:
  c_thickness: 40 angstrom

fitting:
  primary_component: 0
  backend: scipy
  constrain_positive_intensity: true
  components:
    - wavelength: 195.119 angstrom   # Fe XII 195.119
      amplitude_greater_than: 1      # brighter than the blend
    - wavelength: 195.179 angstrom   # Fe XII 195.179
      tie_center: 0                  # same velocity as component 0
      tie_width: 0                   # same width as component 0
```

```bash
eclipse --config ./run/input/mckevitt2026.yaml
```

The slit width and the binning factor are the two numbers this page is about.
`slit_width: 0.4 arcsec` sets both the slit and the raster step, and `offchip_bin_slit: 2`
sums pairs of pixels along the slit after read-out, so signal to noise goes up by sqrt(2) and
the spatial resolution along the slit halves. Both can be given as lists to sweep them, as the
paper did; here they are single values, so there is exactly one combination to select
afterwards.

Everything else is either the paper's value or the current default. Only the 40 Angstrom of
carbon contamination on the filter has to be written out, because the default is a clean
filter. The aluminium and oxide thicknesses, the mesh throughput, the microroughness and the
detector quantum efficiency are all at their defaults already.

`fit_signals: dn` fits only the digitised signal, which is what the maps below use, and roughly
halves the fitting time. `n_iter: 25` is well short of the paper's 500: the two maps on this
page come from the first Monte Carlo iteration and do not depend on it, but anything reporting
a spread across iterations - `velocity_std`, and the exposure time needed to reach a precision
- does, so raise it if you want those.

## 5. The helioprojective intensity and velocity maps

```python
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from euvst_response import (
    load_instrument_response_results,
    get_results_for_combination,
    create_sunpy_maps_from_combo,
    summary_table,
)

results = load_instrument_response_results("run/result/mckevitt2026.pkl")
summary_table(results)

combination = get_results_for_combination(results, **{
    "simulation.slit_width": 0.4 * u.arcsec,
    "simulation.expos": 40 * u.s,
    "offchip_bin_slit": 2,
})

maps = create_sunpy_maps_from_combo(
    combination,
    rest_wavelength=195.119 * u.AA,
    data_type="dn",
)

fig = plt.figure(figsize=(11, 5))

ax = fig.add_subplot(1, 2, 1, projection=maps["total_photons"])
image = maps["total_photons"].plot(
    axes=ax, norm=LogNorm(vmin=2e1, vmax=1.65e5), cmap="afmhot"
)
ax.set_title("Intensity")
fig.colorbar(image, ax=ax, label="Intensity [photons]")

ax = fig.add_subplot(1, 2, 2, projection=maps["velocity_from_fit"])
image = maps["velocity_from_fit"].plot(axes=ax, vmin=-20, vmax=20, cmap="RdBu_r")
ax.set_title("Doppler velocity")
fig.colorbar(image, ax=ax, label="Velocity [km/s]")

fig.savefig("helioprojective_maps.png", dpi=200, bbox_inches="tight")
```

`summary_table(results)` lists every combination in the file along with the exact parameter
names to select it by, which is worth running once before writing the
`get_results_for_combination` call. With a single combination in the file you can drop the
arguments altogether.

`create_sunpy_maps_from_combo` returns rather more than these two. `total_photons` is the
signal before the detector and `total_dn` the same after it; `velocity_from_fit` is the first
fit of the first iteration, `velocity_mean` and `velocity_std` the mean and spread across
iterations, and `velocity_err` the difference from the known truth. They are ordinary SunPy
maps, so `.peek()`, reprojection and coordinate overlays all work as usual.

## What will not match exactly

The figures you get will be close to the published ones, not identical to them.

- **The instrument model is revised.** ECLIPSE tracks EUVST as it is built and tested, so the
  effective area, the filter and the PSF are revised between releases. The point spread
  function currently in the code, for instance, came from a later specification than the paper
  ran against. Every results file records the version and commit that produced it, so
  `summary_table(results)` will tell you what you are actually running.
- **The noise is random.** Nothing is seeded, so each run draws a different realisation. The
  intensity and velocity maps here are a single Monte Carlo iteration, and differ visibly at
  the pixel level between runs. Statistics over many iterations are stable; individual pixels
  are not.
- **So is CHIANTI.** The contribution functions depend on the database version fiasco is
  configured to use, which `--hdf5-dbase-root` can override for a single run without changing
  the default for your other work.

If you need the published numbers themselves rather than a reproduction of the method, the
version that produced them is recorded in the paper.
