# From an MHD simulation

<div class="glance" markdown>
<div markdown>
**You need**

An [atmosphere file](../files/atmosphere.md), written from your simulation
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

ECLIPSE adds up the emission measure along each line of sight by temperature and velocity, and multiplies it by each line's contribution function from CHIANTI, through [fiasco](https://fiasco.readthedocs.io/). The plasma is taken to be optically thin.

## A minimal run

Name the atmosphere file and the lines in a [run file](../run-files.md):

```yaml
instrument: SWC
atmosphere: ./data/muram_ar.h5        # the simulation to synthesise

synthesis:
  lines: [Fe12_195.1190]              # the first is the reference line, unless one is given
  crop_z: [0 Mm, 20 Mm]
```

```bash
eclipse synthesise job.yaml
```

The synthesis goes to `run/input/job.h5`. The `synthesis:` section takes the [options below](#command-line-options), with underscores in place of dashes.

## The atmosphere file

An [atmosphere file](../files/atmosphere.md) holds the simulation's temperature, its mass or electron density, its velocities and its cell edges, indexed `[z, y, x]` with z up.

### Electron density

An `electron_density` in the file is used as it is. Otherwise the mass density is divided by the mass per electron, worked out from `--abundance` for a fully ionised plasma (about 1.16 u for coronal abundances), or given with `--mass-per-electron`.

## Cropping and downsampling

`--crop-x`, `--crop-y` and `--crop-z` keep a range of the box, in the file's coordinates. `--downsample N` keeps every N-th cell, each as big as the N it replaces.

## Seen from a position on the disk

`--position` puts the box's centre at a helioprojective x and y, instead of seeing it from straight above. `--rotation` turns it about the vertical.

```bash
synthesise-spectra --atmosphere atmosphere.h5 --lines Fe12_195.1190 \
  --position "600 arcsec" "-200 arcsec"
```

The line of sight is then slanted, so the atmosphere file needs the velocity along each axis it leans along. The box is taken to repeat sideways unless `--no-wrap` is given.

### The curved surface

`--curvature` lays the box on the curved surface, each column along its own vertical. It needs `velocity_x`, `velocity_y` and `velocity_z`.

```bash
synthesise-spectra --atmosphere atmosphere.h5 --lines Fe12_195.1190 \
  --position "600 arcsec" "-200 arcsec" --curvature
```

## From the command line

```bash
synthesise-spectra --atmosphere ./data/atmosphere.h5 --lines Fe12_195.1190 \
  --crop-z "0 Mm" "20 Mm" --output-dir ./run/input
```

### Command line options

**Input and output:**

- `--config`: A [run file](../run-files.md) to take the settings from; options given here override it
- `--atmosphere`: The [atmosphere file](../files/atmosphere.md)
- `--emission-measure`: In place of `--atmosphere`, an [emission measure file](dem.md)
- `--output-dir`: Where to write the synthesis file (default: `./run/input`)
- `--output-name`: The synthesis file's name (default: `synthesised_spectra.h5`)
- `--keep-vdem`: Also keep each pixel's emission measure by temperature and velocity, which is often larger than the spectra (default: off)

**Lines and abundances:**

- `--lines`: The lines, named as in [Naming spectral lines](../reference/line-names.md), for example `--lines Fe12_195.1190 Fe12_195.1790`
- `--band`: In place of `--lines`, every line between two wavelengths, for a [full-CCD frame](../observe/full-ccd-frames.md#the-bands-spectrum)
- `--band-step`: The band's wavelength spacing (default: `"0.002 Angstrom"`)
- `--band-lines`: Where to keep the band's lines (default: `band_lines.h5` in the output directory)
- `--abundance`: The CHIANTI abundance set (default: `sun_coronal_2021_chianti`)
- `--electron-density`, `--pressure`: With `--emission-measure`, the density or electron pressure to take the contribution functions at
- `--continuum`: Also synthesise the continuum (default: off)
- `--n-workers`: Processes for the contribution functions (default: `0`, every CPU)
- `--hdf5-dbase-root`: Another CHIANTI database for fiasco to read
- `--goft-temperature-chunk`: Temperatures to compute at a time, to use less memory

**Velocity grid:**

- `--vel-res`: The velocity bins' spacing (default: `"5.0 km/s"`)
- `--vel-lim`: How far the bins reach either side of zero (default: `"300.0 km/s"`); faster plasma is left out, with a warning

**Viewing direction:**

- `--integration-axis`: The axis to look along (default: `z`, from above): `x` from +x, `y` from -y, or a sign for the other side, such as `-x`
- `--position`: The helioprojective x and y of the box's centre; see [Seen from a position on the disk](#seen-from-a-position-on-the-disk)
- `--rotation`: With `--position`, turn the box about the vertical, from x pointing west towards north
- `--projection`: With `--position`, `two-step` (default) or `exact`, which is slower
- `--curvature`: With `--position`, lay the box on the [curved surface](#the-curved-surface)
- `--no-wrap`: With `--position`, do not take the box to repeat sideways

**Cropping and the image:**

- `--crop-x`, `--crop-y`, `--crop-z`: The range to keep along each axis, for example `--crop-z "0 Mm" "20 Mm"`
- `--image-x`: Keep only the image between two positions across the slit

**Processing:**

- `--downsample`: Keep every N-th cell (default: `1`)
- `--precision`: `float32` or `float64` (default: `float64`)
- `--mass-per-electron`: The mass per free electron in u; see [Electron density](#electron-density)

## Running it faster

Downsample or crop for a first test, and use `--precision float32` for less memory.

## Worked examples

### A MURaM active region

The [worked example](../worked-example.ipynb) writes an atmosphere file from a MURaM active region and synthesises it.

### A Bifrost quiet Sun

The enhanced-network run `en024048_hion` of [Carlsson et al. (2016)](https://doi.org/10.1051/0004-6361/201527226) is on [Hinode SDC Europe](https://sdc.uio.no/search/simulations) as one FITS file per variable, in SI units, with `lg` names as log10 and the cell heights in the first extension. Download snapshot 385:

```bash
for variable in lgtg lgr lgne uz; do
  curl -O https://sdc.uio.no/vol/simulations/en024048_hion/atmos/BIFROST_en024048_hion_${variable}_385.fits
done
```

and write it as an atmosphere file:

```python
import astropy.units as u
import numpy as np
from astropy.io import fits
from eclipse import Atmosphere, write_atmosphere, edges_from_centres

def variable(name, snapshot=385):
    with fits.open(f"BIFROST_en024048_hion_{name}_{snapshot}.fits") as hdul:
        return hdul[0].data, hdul[0].header, hdul[1].data  # cube (nz, ny, nx), header, z centres in Mm

lgtg, header, z = variable("lgtg")
lgr, _, _ = variable("lgr")
lgne, _, _ = variable("lgne")
uz, _, _ = variable("uz")

nz, ny, nx = lgtg.shape
x = (header["CRVAL1"] + (np.arange(nx) + 1 - header["CRPIX1"]) * header["CDELT1"]) * u.Mm
y = (header["CRVAL2"] + (np.arange(ny) + 1 - header["CRPIX2"]) * header["CDELT2"]) * u.Mm

atmosphere = Atmosphere(
    temperature=10.0 ** lgtg.astype(np.float64) * u.K,
    mass_density=10.0 ** lgr.astype(np.float64) * u.kg / u.m**3,
    electron_density=10.0 ** lgne.astype(np.float64) * u.m**-3,
    velocity_z=uz * u.m / u.s,
    x_edges=edges_from_centres(x), y_edges=edges_from_centres(y),
    z_edges=edges_from_centres(z * u.Mm),
    time=header["ELAPSED"] * u.s,
    source="Bifrost en024048_hion snapshot 385, Hinode SDC Europe",
)
write_atmosphere(atmosphere, "bifrost_385.h5")
```

The box starts 2.4 Mm below the surface, so crop it to the part that emits:

```bash
synthesise-spectra --atmosphere bifrost_385.h5 --lines Fe09_171.0730 \
  --crop-z "0 Mm" "20 Mm" --output-dir ./run/input
```
