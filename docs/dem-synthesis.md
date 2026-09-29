# Synthesising from a DEM or VDEM

A differential emission measure (DEM) says how much emitting plasma there is at each temperature. You might have one from an inversion of real observations. A velocity differential emission measure (VDEM) also splits the plasma by its velocity along the line of sight, and is made from an MHD simulation. From either, ECLIPSE can synthesise the lines the plasma emits and simulate how the instrument would measure them.

## How it works

ECLIPSE's own synthesis works in two steps. It first adds up the emission measure in each pixel by temperature and line-of-sight velocity, which is a VDEM. It then multiplies that by each line's contribution function, G(T, n_e), to get the spectrum. A DEM or a VDEM can go straight into the second step.

A DEM differs from an MHD simulation in two ways:

- **No velocities.** All of the emission goes in the zero-velocity bin, so the lines are not Doppler shifted.
- **No densities.** G(T, n_e) is taken at an electron density you choose.

A VDEM has the velocities, so only the second of these applies. It is covered [below](#with-velocities-a-vdem), after the DEM.

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

**Line names.** Lines are named as in the MHD route: see [Naming spectral lines](line-names.md).

**Scene size.** A scene can be as small as one pixel, `nx, ny = 1, 1`, for a single DEM.

**Abundance.** `abundance` chooses the element abundances, which set how bright the low-FIP lines are compared with the high-FIP ones. For FIP studies, synthesise once with each set, for example `sun_coronal_2021_chianti` and `sun_photospheric_2021_asplund`. The ratio of a line's intensity in the two is the FIP bias you are trying to measure.

## With velocities: a VDEM

A VDEM is made from an MHD simulation by sorting every cell along the line of sight by its temperature and velocity. The [MHD route](synthesis.md) makes one itself on the way to the spectra. Giving ECLIPSE a VDEM skips that step. That is worth doing when you already have one: a VDEM is far smaller than the simulation it came from, and quick to synthesise again with other lines or abundances.

A VDEM fills the velocity bins in, so the lines come out Doppler shifted and broadened by the motions along the line of sight, as well as thermally.

!!! note "This is not the route for another code's spectra"

    A VDEM describes the plasma, and ECLIPSE still works out the light it emits, taking it to be optically thin. If another code, such as Lightweaver or RH1.5D, has already synthesised the spectra, write them as a [synthesis file](other-codes.md) instead. ECLIPSE then skips the synthesis and puts those spectra straight through the instrument.

### The example with a VDEM

This is the [example above](#minimal-example) with two changes.

**Replace steps 1 to 3** with your VDEM, laid out as `vdem[y, x, logT, v]` in cm^-5, one value for each temperature and velocity bin:

```python
# 1. The VDEM: a logT grid, a velocity grid, and the emission measure in each (logT, v) bin.
#    Replace this with your own code's output.
logT = np.arange(4.0, 8.0 + 0.04, 0.04)
vel_grid = np.arange(-300.0, 300.0 + 5.0, 5.0) * u.km / u.s

nx, ny = 2, 2
vdem = my_code_output(nx, ny, logT, vel_grid)   # cm^-5, shape (ny, nx, nT, nv)
```

**Replace step 6**, which puts all of the emission in the zero-velocity bin, with the VDEM itself:

```python
# 6. The VDEM is already EM(y, x, T, v), so it goes straight in.
em_tv = vdem
```

Then carry on from step 7, `synthesise_spectra`, unchanged.

### Key points for a VDEM

**Bulk velocity only.** `synthesise_spectra` adds the thermal width itself, from each temperature bin's temperature and the element's atomic weight. Your VDEM should therefore hold only the bulk motion along the line of sight. If it already includes the thermal motions, the lines will come out too wide.

**Velocity sign.** Each bin is shifted to `lambda = lambda_0 (1 + v/c)`, so a positive velocity in the grid gives a longer wavelength, a redshift.

**Units.** `em_tv` is the emission measure in each bin (cm^-5), not per kelvin or per km/s. If your VDEM is per kelvin and per km/s, multiply by each bin's widths, as in [step 2 of the DEM example](#minimal-example).

**Bin centres.** Both `logT` and `vel_grid` hold the centres of the bins, not their edges.

**An even velocity grid.** The velocity grid must be evenly spaced and increasing. ECLIPSE refuses any other.

**Temperatures.** As for a DEM, compute the contribution functions on exactly your temperatures, by passing `logT_min`, `logT_max` and `nT` to `compute_goft_fiasco`, and check that the two grids agree.

**The velocity range sets the wavelength window.** Each line's wavelengths are its velocity grid converted with `lambda_0 (1 + v/c)`, so a line near the end of the velocity range loses the far side of its profile. Leave room for the fastest motion you need, plus several thermal widths. `synthesise-spectra` uses +/- 300 km/s by default, which is wide for coronal lines.

## Running the instrument response

The output, from a DEM or a VDEM, is an ordinary synthesis file, so `eclipse` observes it as it would any other (see [Running a simulation](instrument-response.md)). Each run observes one spectral window, chosen by `reference_line`:

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
