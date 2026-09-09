# Synthesising lines from an observed DEM

The [line synthesis](synthesis.md) page starts from a 3D MHD cube. If instead you
already have a differential emission measure (DEM) - from an inversion of real
observations, for example - you can feed it into ECLIPSE directly and forward
model how the instrument would measure it.

This is the route used to study how EIS instrument effects bias FIP-bias
measurements: take observed DEMs, synthesise a low-FIP and a high-FIP line from
each, and run the instrument response to see how well the ratio can be recovered.

## How it fits together

Internally the MHD path builds a `DEM(x, y, T)` map and then synthesises spectra
from `EM(T, v) * G(T)`. An observed DEM is the same object, so it can be injected
at that step. Two things differ from the MHD case:

- **No velocity information.** All the emission goes in the zero-velocity bin.
- **No density information.** `G(T, n_e)` is evaluated at a single assumed
  electron density.

## Minimal example

This synthesises Si X 258.375 (low FIP) and S X 264.230 (high FIP) from a single
analytic DEM profile. Swap the profile for your own inversion output.

```python
import numpy as np
import astropy.units as u
import dill

from euvst_response.synthesis import (
    compute_goft_fiasco,
    interpolate_g_on_dem,
    synthesise_spectra,
    create_line_cube,
    create_atmosphere_ndcube,
)
from euvst_response.utils import angle_to_distance

INTENSITY_UNIT = u.erg / u.s / u.cm ** 2 / u.sr / u.cm


def main():
    # 1. The observed DEM: a logT grid and DEM(T) in cm^-5 K^-1.
    #    Replace this Gaussian with your own inversion output.
    logT = np.arange(4.0, 8.0 + 0.04, 0.04)
    dem_perK = 1.0e21 * np.exp(-0.5 * ((logT - 6.2) / 0.15) ** 2)

    # 2. DEM per kelvin -> emission measure per log10(T) bin (cm^-5).
    dlogT = float(np.mean(np.diff(logT)))
    dT_lin = 10.0 ** (logT + dlogT / 2.0) - 10.0 ** (logT - dlogT / 2.0)
    em_bin = dem_perK * dT_lin

    # 3. Lay the profile out as a small scene of independent pixels.
    nx, ny = 2, 2
    em_scene = np.tile(em_bin, (nx, ny, 1))

    # 4. Contribution functions, on exactly the DEM temperature grid.
    lines = ["Si10_258.3750", "S10_264.2300"]
    goft, logT_goft, logN_grid = compute_goft_fiasco(
        lines,
        abundance="sun_coronal_2021_chianti",
        logT_min=float(logT[0]),
        logT_max=float(logT[-1]),
        nT=len(logT),
        n_workers=2,
    )
    assert np.allclose(logT_goft, logT, atol=1e-6)

    # 5. Evaluate G(T, n_e) at one assumed electron density.
    ne_map = np.full((nx, ny, len(logT)), 10.0 ** 9.0)
    interpolate_g_on_dem(goft, ne_map, logT, logN_grid, logT_goft, np.float64)

    # 6. Put all the emission in the zero-velocity bin.
    vel_grid = np.arange(-300.0, 300.0 + 5.0, 5.0) * u.km / u.s
    iv0 = int(np.argmin(np.abs(vel_grid.value)))
    em_tv = np.zeros((nx, ny, len(logT), len(vel_grid)))
    em_tv[:, :, :, iv0] = em_scene

    # 7. Synthesise.
    synthesise_spectra(goft, em_tv, vel_grid.to(u.cm / u.s), logT)

    # 8. Wrap as ECLIPSE line cubes and save in the synthesis format.
    plate_scale = 1.0 * u.arcsec * (1.0 + 50.0 * np.finfo(float).eps)
    voxel = angle_to_distance(plate_scale).to(u.Mm)
    reference = create_atmosphere_ndcube(
        np.zeros((nx, ny, 1)) * u.K, voxel, voxel, voxel
    )
    line_cubes = {
        name: create_line_cube(name, info, reference, INTENSITY_UNIT,
                               integration_axis="z")
        for name, info in goft.items()
    }

    with open("./run/input/dem_synth.pkl", "wb") as f:
        dill.dump({
            "line_cubes": line_cubes,
            "dynamic_mode": {"enabled": False},
            "config": {"lines": lines,
                       "abundance": "sun_coronal_2021_chianti"},
        }, f)


if __name__ == "__main__":
    main()
```

!!! warning "Keep the `if __name__ == \"__main__\":` guard"

    `compute_goft_fiasco` parallelises over ions using the `spawn` start method,
    so each worker re-imports the main module. Without the guard, every worker
    re-runs the script top to bottom and spawns more workers, which forks without
    bound until the machine gives up.

## Things that matter

**Units.** An observed DEM is usually per kelvin (cm^-5 K^-1), while ECLIPSE's
`em_tv` is emission measure summed per log10(T) bin (cm^-5). Convert with the
linear temperature width of each bin, as in step 2. Getting this wrong scales
every intensity.

**Grid alignment.** Force the `G(T)` grid to match the DEM grid exactly by
passing the DEM's `logT_min`, `logT_max`, and `nT` to `compute_goft_fiasco`, then
assert they agree. If they do not line up, `G(T)` and `DEM(T)` are silently
sampled at different temperatures.

**Line names.** The same `<Element><Stage>_<Wavelength>` convention as the MHD
route - see [naming spectral lines](synthesis.md#naming-spectral-lines).

**Scene size.** `create_line_cube` needs at least two pixels along each spatial
axis, so a single DEM profile still has to be laid out on a 2x2 or larger scene.

**Abundance.** The `abundance` argument sets the FIP treatment. For FIP work,
synthesise once per abundance set (for example `sun_coronal_2021_chianti` and
`sun_photospheric_2021_asplund`); the ratio between them is the FIP enhancement
you are trying to recover.

## Running the instrument response

The pickle is in the normal synthesis format, so the
[instrument response](instrument-response.md) stage consumes it unchanged. One
run per spectral window:

```yaml
# configs/eis_si10.yaml
instrument: EIS
synthesis_file: ./run/input/dem_synth.pkl
reference_line: Si10_258.3750   # selects the Si X 258 window

n_iter: 500
ncpu: -1

simulation:
  slit_width: 1 arcsec          # 1"/pix plate scale, so no spatial binning
  expos: [10 s, 30 s, 60 s]     # swept: one run per exposure
  psf: False
```

```bash
eclipse --config configs/eis_si10.yaml
```

Then load the results and compare the fitted intensities against the truth, as
in the [analysis tutorial](tutorial.ipynb).
