# Synthesising from a VDEM

A velocity differential emission measure (VDEM) is a DEM resolved in
line-of-sight velocity: how much emitting material there is at each temperature
*and* each velocity. It is computed from an MHD simulation, by binning every
voxel along the line of sight by its temperature and velocity. If you already
have one, ECLIPSE can synthesise lines from it and forward model how the
instrument would measure them.

This is ECLIPSE's native internal representation. The
[MHD route](synthesis.md) does exactly that binning itself, and produces this
object before folding it with `G(T, n_e)`. Handing one over directly just skips
the first step.

Two reasons to take this route rather than the [MHD route](synthesis.md):

- **Your simulation is not MURaM.** ECLIPSE's reader currently expects MURaM
  output. Reducing your own simulation to a VDEM is the way to use it in the
  meantime.
- **The reduction is already done.** A VDEM is far smaller than the cube it came
  from, and cheap to re-synthesise from with different lines or abundances.

It also follows that the VDEM route is the [DEM route](dem-synthesis.md) without
the DEM route's main limitation. A DEM carries no velocity information, so that
page puts all the emission in the zero-velocity bin; a VDEM fills the velocity
bins in, and the synthesised lines come out Doppler shifted and broadened by the
bulk motions as well as thermally.

!!! note "This is not the route for another code's spectra"

    A VDEM describes the *plasma*, and ECLIPSE still does the radiative transfer
    on it, optically thin. If another code has already synthesised the *spectra*
    from an atmosphere - Lightweaver, RH1.5D, or anything else - that is a
    different input: the synthesis stage is skipped altogether and the spectra go
    straight into the [instrument response](instrument-response.md). Reading
    those directly is coming soon.

## Minimal example

The recipe is the [DEM example](dem-synthesis.md#minimal-example) with two
changes. Everything else - the contribution functions, the grid alignment
check, the density assumption, wrapping the result as line cubes, and saving -
is identical.

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

## Things that matter

**Bulk velocity only.** `synthesise_spectra` adds thermal broadening itself: for
each temperature bin it computes the thermal width from that bin's temperature
and the ion's atomic weight, then places that Gaussian at the velocity bin's
Doppler-shifted centre. Your VDEM must therefore describe only the distribution
of *bulk* line-of-sight velocity. If it has already been convolved with a
thermal profile, the synthesised lines will be too wide.

**Velocity sign convention.** ECLIPSE shifts each bin to
`lambda = lambda_0 (1 + v/c)`, so positive velocity in the grid means longer
wavelength, that is a redshift. Check this against the convention of whichever
code produced the VDEM. A flipped sign is invisible in a symmetric profile and
reverses the answer in an asymmetric one.

**Units and binning.** `em_tv` is emission measure *summed within* each bin
(cm^-5), not a density per unit temperature or per unit velocity. If your code
gives a VDEM per kelvin and per km/s, multiply by each bin's width, as in
[step 2 of the DEM example](dem-synthesis.md#minimal-example).

**Bin centres, not edges.** Both `logT` and `vel_grid` hold bin centres. The
velocity grid does not have to be uniform - edges are derived from the centres -
but it does have to be monotonic.

**Grid alignment.** As on the DEM page, force the `G(T)` grid to match your
temperature grid by passing `logT_min`, `logT_max`, and `nT` to
`compute_goft_fiasco`, and assert they agree.

**The velocity range sets the wavelength window.** The synthesised wavelength
grid is the velocity grid mapped through `lambda_0 (1 + v/c)` and nothing wider,
so a line sitting near the edge of the velocity range has its far wing cut off.
Leave room for the fastest bulk motion you care about *plus* several thermal
widths beyond it. The default +/- 300 km/s is generous for coronal lines, where
the thermal width is of order 20 km/s.

## Running the instrument response

The output is an ordinary synthesis file, so the
[instrument response](instrument-response.md) stage consumes it unchanged, and
the results are analysed exactly as in the [analysis tutorial](tutorial.ipynb).
