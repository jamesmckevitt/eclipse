# Synthesising from a VDEM

A velocity differential emission measure (VDEM) is a DEM that is also split by line-of-sight velocity. It says how much emitting plasma there is at each temperature and each velocity. It is made from an MHD simulation by sorting every cell along the line of sight by its temperature and velocity. If you already have one, ECLIPSE can synthesise lines from it and simulate how the instrument would measure them.

The [MHD route](synthesis.md) makes a VDEM itself, and then multiplies it by each line's contribution function, G(T, n_e). Giving ECLIPSE a VDEM skips the first of those steps. That is worth doing when you already have one: a VDEM is far smaller than the simulation it came from, and quick to synthesise again with other lines or abundances.

A DEM has no velocities, so the [DEM route](dem-synthesis.md) puts all of the emission in the zero-velocity bin. A VDEM fills the velocity bins in, so the lines come out Doppler shifted and broadened by the motions along the line of sight, as well as thermally.

!!! note "This is not the route for another code's spectra"

    A VDEM describes the plasma, and ECLIPSE still works out the light it emits, taking it to be optically thin. If another code, such as Lightweaver or RH1.5D, has already synthesised the spectra, write them as a [synthesis file](other-codes.md) instead. ECLIPSE then skips the synthesis and puts those spectra straight through the instrument.

## Minimal example

This is the [DEM example](dem-synthesis.md#minimal-example) with two changes.

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

## Key points

**Bulk velocity only.** `synthesise_spectra` adds the thermal width itself, from each temperature bin's temperature and the element's atomic weight. Your VDEM should therefore hold only the bulk motion along the line of sight. If it already includes the thermal motions, the lines will come out too wide.

**Velocity sign.** Each bin is shifted to `lambda = lambda_0 (1 + v/c)`, so a positive velocity in the grid gives a longer wavelength, a redshift.

**Units.** `em_tv` is the emission measure in each bin (cm^-5), not per kelvin or per km/s. If your VDEM is per kelvin and per km/s, multiply by each bin's widths, as in [step 2 of the DEM example](dem-synthesis.md#minimal-example).

**Bin centres.** Both `logT` and `vel_grid` hold the centres of the bins, not their edges.

**An even velocity grid.** The velocity grid must be evenly spaced and increasing. ECLIPSE refuses any other.

**Temperatures.** As on the DEM page, compute the contribution functions on exactly your temperatures, by passing `logT_min`, `logT_max` and `nT` to `compute_goft_fiasco`, and check that the two grids agree.

**The velocity range sets the wavelength window.** Each line's wavelengths are its velocity grid converted with `lambda_0 (1 + v/c)`, so a line near the end of the velocity range loses the far side of its profile. Leave room for the fastest motion you need, plus several thermal widths. `synthesise-spectra` uses +/- 300 km/s by default, which is wide for coronal lines.

## Running the instrument response

The output is an ordinary synthesis file, so `eclipse` observes it as it would any other (see [Simulating a single snapshot](instrument-response.md)), and the results are analysed as in the [worked example](worked-example.ipynb).
