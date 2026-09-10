# Synthesising from a VDEM

A velocity differential emission measure (VDEM) is a DEM resolved in line-of-sight velocity, containing information of how much emitting material there is at each temperature *and* each velocity. It is computed from an MHD simulation, by binning every voxel along the line of sight by its temperature and velocity. If you already have one, ECLIPSE can synthesise lines from it and forward model how the instrument would measure them.

This is ECLIPSE's native internal format. The [MHD route](synthesis.md) does that binning itself, and produces this object before folding it with `G(T, n_e)`. Providing one directly just skips that first step.

Two reasons to take this route rather than the [MHD route](synthesis.md):

- **Your simulation is not MURaM.** ECLIPSE's reader currently expects MURaM output. Reducing your own simulation to a VDEM is the way to use it in the meantime.
- **The reduction is already done.** A VDEM is far smaller than the MHD cubes it came from, and easy to re-synthesise from with different lines or abundances.

A DEM carries no velocity information, so the [DEM route](dem-synthesis.md) puts all the emission in the zero-velocity bin. A VDEM fills the velocity bins in, and the synthesised lines come out Doppler shifted and broadened by the bulk motions along the line of sight, as well as thermally.

!!! note "This is not the route for another code's spectra"

    A VDEM describes the *plasma*, and ECLIPSE still does the radiative transfer on it, optically thin. If another code has already synthesised the *spectra* from an atmosphere - Lightweaver, RH1.5D, or anything else - that is a different input. The synthesis stage is skipped altogether and the spectra go straight into the [instrument response](instrument-response.md). Reading those directly is coming soon.

## Minimal example

This recipe is as in the [DEM example](dem-synthesis.md#minimal-example) with two changes.

**Replace steps 1 to 3** with your VDEM, laid out as
`vdem[x, y, logT, v]` in cm^-5, one value per temperature and velocity bin:

```python
# 1. The VDEM: a logT grid, a velocity grid, and EM in each (logT, v) bin.
#    Replace this with your own code's output.
logT = np.arange(4.0, 8.0 + 0.04, 0.04)
vel_grid = np.arange(-300.0, 300.0 + 5.0, 5.0) * u.km / u.s

nx, ny = 2, 2
vdem = my_code_output(nx, ny, logT, vel_grid)   # cm^-5, shape (nx, ny, nT, nv)
```

**Replace step 6** - the one that collapses everything into the zero-velocity
bin - with the VDEM itself:

```python
# 6. The VDEM is already EM(x, y, T, v), so it goes straight in.
em_tv = vdem
```

Then continue from step 7 (`synthesise_spectra`) unchanged.

## Key points

**Bulk velocity only.** `synthesise_spectra` adds thermal broadening. For each temperature bin it computes the thermal width from that bin's temperature and the ion's atomic weight, then places that Gaussian at the velocity bin's Doppler-shifted centre. Your VDEM must therefore describe only the distribution of *bulk* line-of-sight velocity. If it has already been convolved with a thermal profile, the synthesised lines will be too wide.

**Velocity sign convention.** ECLIPSE shifts each bin to `lambda = lambda_0 (1 + v/c)`, so positive velocity in the grid means longer wavelength, that is a redshift.

**Units and binning.** `em_tv` is emission measure *summed within* each bin (cm^-5), not a density per unit temperature or per unit velocity. If your code gives a VDEM per kelvin and per km/s, multiply by each bin's width, as in [step 2 of the DEM example](dem-synthesis.md#minimal-example).

**Bin centres, not edges.** Both `logT` and `vel_grid` hold bin centres.

**Use a uniformly spaced velocity grid.** Non-uniform spacing is not supported. Bin edges are derived by applying the *first* spacing to the whole grid, and the output cube's WCS is written with a single linear `CDELT` taken from the first wavelength step. An unevenly spaced grid therefore gets both its bin widths and its wavelength coordinates silently wrong.

**Grid alignment.** As on the DEM page, force the `G(T)` grid to match your temperature grid by passing `logT_min`, `logT_max`, and `nT` to `compute_goft_fiasco`, and make sure they agree.

**The velocity range sets the wavelength window.** The synthesised wavelength grid is the velocity grid mapped through `lambda_0 (1 + v/c)`, so a line sitting near the edge of the velocity range has its far wing cut off. Leave room for the fastest bulk motion you need *plus* several thermal widths beyond it. The default +/- 300 km/s is a wide assumption for coronal lines.

## Running the instrument response

The output is an ordinary synthesis file, so the [instrument response](instrument-response.md) stage can use it, and the results are analysed exactly as in the [analysis tutorial](tutorial.ipynb).