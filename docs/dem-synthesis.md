# Synthesising lines from an observed DEM

If you have a differential emission measure (DEM), for example from an inversion of real observations, ECLIPSE can synthesise the lines it emits and simulate how the instrument would measure them.

## How it works

ECLIPSE's own synthesis works in two steps. It first adds up the emission measure in each pixel by temperature and line-of-sight velocity. It then multiplies that by each line's contribution function, G(T, n_e), to get the spectrum. An observed DEM is the first of these without the velocities, so it can go straight into the second step. That makes two differences from an MHD simulation:

- **No velocities.** All of the emission goes in the zero-velocity bin, so the lines are not Doppler shifted.
- **No densities.** G(T, n_e) is taken at an electron density you choose.

## Minimal example

This synthesises Si X 258.374 (low FIP) and S X 264.230 (high FIP) from a made-up Gaussian DEM. Replace it with your own.

```python
from pathlib import Path

import numpy as np
import astropy.units as u

from euvst_response.synthesis import (
    compute_goft_fiasco,
    interpolate_g_on_dem,
    synthesise_spectra,
    create_line_cube,
    create_atmosphere_ndcube,
)
from euvst_response.synthesis_file import write_line_cubes
from euvst_response.utils import angle_to_distance

INTENSITY_UNIT = u.erg / u.s / u.cm ** 2 / u.sr / u.cm


def main():
    # 1. The observed DEM: a logT grid and DEM(T) in cm^-5 K^-1.
    #    Replace this Gaussian with your own.
    logT = np.arange(4.0, 8.0 + 0.04, 0.04)
    dem_perK = 1.0e21 * np.exp(-0.5 * ((logT - 6.2) / 0.15) ** 2)

    # 2. From DEM per kelvin to the emission measure in each log10(T) bin (cm^-5).
    dlogT = float(np.mean(np.diff(logT)))
    dT_lin = 10.0 ** (logT + dlogT / 2.0) - 10.0 ** (logT - dlogT / 2.0)
    em_bin = dem_perK * dT_lin

    # 3. Copy the DEM into every pixel of a small scene.
    nx, ny = 2, 2
    em_scene = np.tile(em_bin, (ny, nx, 1))

    # 4. Contribution functions, on exactly the DEM's temperatures.
    lines = ["Si10_258.3740", "S10_264.2300"]
    goft, logT_goft, logN_grid = compute_goft_fiasco(
        lines,
        abundance="sun_coronal_2021_chianti",
        logT_min=float(logT[0]),
        logT_max=float(logT[-1]),
        nT=len(logT),
        n_workers=2,
    )
    assert np.allclose(logT_goft, logT, atol=1e-6)

    # 5. Take G(T, n_e) at one electron density.
    ne_map = np.full((ny, nx, len(logT)), 10.0 ** 9.0)
    interpolate_g_on_dem(goft, ne_map, logT, logN_grid, logT_goft, np.float64)

    # 6. Put all of the emission in the zero-velocity bin.
    vel_grid = np.arange(-300.0, 300.0 + 5.0, 5.0) * u.km / u.s
    iv0 = int(np.argmin(np.abs(vel_grid.value)))
    em_tv = np.zeros((ny, nx, len(logT), len(vel_grid)))
    em_tv[:, :, :, iv0] = em_scene

    # 7. Synthesise.
    synthesise_spectra(goft, em_tv, vel_grid.to(u.cm / u.s), logT)

    # 8. Make ECLIPSE line cubes, with 1 arcsec pixels, and write them and the emission measure to a synthesis file.
    voxel = angle_to_distance(1.0 * u.arcsec).to(u.Mm)
    reference = create_atmosphere_ndcube(
        np.zeros((1, ny, nx)) * u.K, voxel, voxel, voxel
    )
    line_cubes = {
        name: create_line_cube(name, info, reference, INTENSITY_UNIT,
                               integration_axis="z")
        for name, info in goft.items()
    }

    write_line_cubes(line_cubes, Path("./run/input/dem_synth.h5"), products={
        "em_tv": em_tv,
        "logT_grid": logT,
        "vel_grid": vel_grid,
        "config": {"lines": lines, "abundance": "sun_coronal_2021_chianti"},
    })


if __name__ == "__main__":
    main()
```

!!! warning "Keep the `if __name__ == '__main__':` guard"

    `compute_goft_fiasco` computes the ions in separate worker processes, and each worker starts by importing your script. Without the guard, each worker would run the whole script again and start workers of its own.

## Key points

**Units.** An observed DEM is usually per kelvin (cm^-5 K^-1). ECLIPSE wants the emission measure in each temperature bin instead (cm^-5), with the bins evenly spaced in log10(T). Multiply by each bin's width in kelvin, as in step 2.

**Temperatures.** Compute the contribution functions on exactly the DEM's temperatures, by passing the DEM's `logT_min`, `logT_max` and `nT` to `compute_goft_fiasco`, and check that the two grids agree, as the `assert` does. On a different grid, G(T) would be interpolated linearly onto the DEM's temperatures, which is less accurate for a sharply peaked G(T), and taken as zero outside its own range.

**Line names.** Lines are named as in the MHD route: see [naming spectral lines](synthesis.md#naming-spectral-lines).

**Scene size.** A scene can be as small as one pixel, `nx, ny = 1, 1`, for a single DEM.

**Abundance.** `abundance` chooses the element abundances, which set how bright the low-FIP lines are compared with the high-FIP ones. For FIP studies, synthesise once with each set, for example `sun_coronal_2021_chianti` and `sun_photospheric_2021_asplund`. The ratio of a line's intensity in the two is the FIP bias you are trying to measure.

## Running the instrument response

The output is an ordinary synthesis file, so `eclipse` observes it as it would any other (see [Simulating a single snapshot](instrument-response.md)). Each run observes one spectral window, chosen by `reference_line`:

```yaml
# configs/eis_si10.yaml
instrument: EIS
synthesis_file: ./run/input/dem_synth.h5
reference_line: Si10_258.3740   # observe the Si X 258 window

n_iter: 500
ncpu: -1

simulation:
  slit_width: 1 arcsec          # the same as the scene's pixels, and EIS's pixels along the slit
  expos: [10 s, 30 s, 60 s]     # each exposure is simulated in turn
  psf: False
```

```bash
eclipse --config configs/eis_si10.yaml
```

The results can then be analysed as in the [worked example](worked-example.ipynb).
