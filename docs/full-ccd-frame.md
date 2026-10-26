# A full-CCD frame

`eclipse` observes one spectral window at a time. A full-CCD frame is what the SW camera reads out at one slit position: both CCDs, with every line in the band and the continuum, and the prescan and overscan of each output.

A frame is made from a synthesis file from ECLIPSE's [synthesis of an MHD simulation](synthesis.md), which keeps the emission measure in each bin of temperature and velocity of every pixel, and the electron density at each temperature.

## 1. Every line in the band

`band_contribution_functions` works out the contribution function of every CHIANTI line in the band, on the synthesis's grids and with its abundances, and `write_band_lines` keeps them in a file:

```python
from eclipse import band_contribution_functions, read_synthesis_products, write_band_lines

if __name__ == "__main__":
    products = read_synthesis_products("synthesis.h5", keys=["logT_grid", "logN_grid", "config"])
    lines = band_contribution_functions((169.0, 213.5), products["logT_grid"],
                                        products["logN_grid"],
                                        abundance=products["config"]["abundance"])
    write_band_lines(lines, "band_lines.h5")
```

On the default grids this takes about two hours and 150 GB of memory on 16 cores, so it is done once. A synthesis's densities are always steps of 0.3 in log10 n_e, so lines worked out over a wider run of those steps serve any synthesis whose densities fall within it.

## 2. The continuum

The continuum is optional. It is worked out on wavelengths finer than a detector row, which is about 0.017 A, inside the telescope's throughput tables, from 170 to 214 A:

```python
import astropy.units as u
import numpy as np
from eclipse.continuum import compute_continuum_fiasco

if __name__ == "__main__":
    wavelength = np.linspace(170.0, 214.0, 10481) * u.AA
    free, two_photon = compute_continuum_fiasco(wavelength, lines.logT_grid, lines.logN_grid,
                                                abundance=lines.abundance)
```

This takes about 6 minutes on 16 cores.

## 3. The frame

```python
from eclipse import full_ccd_frame, read_band_lines, write_full_frame

lines = read_band_lines("band_lines.h5")
frame = full_ccd_frame("synthesis.h5", lines, slit_position=0 * u.arcsec,
                       exposure=10 * u.s, slit_width=0.2 * u.arcsec,
                       continuum=(wavelength, free, two_photon))
write_full_frame(frame, "frame.h5")
```

`slit_position` is where the middle of the slit is, across it, in the synthesis file's x coordinates. Along the slit, the middle of the image is at the middle of the slit, which is the middle column of the CCDs.

With the PSF on, the blur along the slit carries light out of the columns at the ends of the scene. `psf_boundary` sets what the blur brings in from beyond the scene, as in a run. With `"replicate"`, the default, the Sun beyond the scene is taken to look like the scene's edges, and the columns beyond the scene get no light. With `"zero"`, the Sun beyond the scene is dark, so the columns at the ends of the scene lose up to about a third of their light to the columns just beyond. `psf_boundary` sets the same across the slit, for a slit near the synthesis's x edges. With `"replicate"`, a slit partly beyond those edges gets light only over its part inside them.

With `slit_mask=True`, no light reaches the columns beyond the ends of the 300 arcsec slit or behind its fiducials, as in a run (see [The slit's fiducials](instrument-response.md#the-slits-fiducials)). The CCD's 2048 columns cover 326 arcsec along the slit, so about 80 columns at each end are beyond the slit. It is off by default.

The slit width, PSF and noise can instead be taken from a `Simulation`, as in an instrument-response run:

```python
from eclipse import Simulation

sim = Simulation(slit_width=0.2 * u.arcsec, psf=True, noise=True)
frame = full_ccd_frame("synthesis.h5", lines, slit_position=0 * u.arcsec,
                       exposure=10 * u.s, sim=sim, continuum=(wavelength, free, two_photon))
```

The frame uses the `Simulation`'s `slit_width`, `psf`, `spectral_psf`, `noise`, `psf_boundary` and `slit_mask`. A `Simulation` has the PSF off unless it is made with `psf=True`. When one of these six arguments of `full_ccd_frame` equals its default, the `Simulation`'s value is used, even if the argument was typed out. An argument with any other value must agree with the `Simulation`. The exposure is set by `exposure`, not by the `Simulation`'s `expos`. The number of workers is set by `n_workers`. A frame does not model pinholes or visible stray light. `vis_sl` must be left at zero and no pinholes enabled. `instrument` must be `"SWC"`.

For the MURaM flare in these docs, with 36 densities and 201 velocity bins, a frame takes about 20 minutes and 55 GB of memory on 16 cores. By default `full_ccd_frame` runs a worker process on every CPU it may use, and each adds to the memory; set `n_workers` to use fewer.

Every part of each line, and of the continuum, is counted in photons at its own wavelength, with the telescope's throughput there, and the photons in each pixel free the electrons of their own energy.

The lines come only from plasma within the synthesis's velocity grid, so set `--vel-lim` wide enough for the fastest flows. The continuum is not Doppler shifted, and comes from all the plasma.

## What a frame holds

For each CCD, `"left"` and `"right"`:

- `rate`: the photons per second reaching each image pixel;
- `photons`: the photons in each pixel over the exposure;
- `dn`: the frame as it is read out, in DN;
- `wavelength`: the wavelength each row records;
- `photon_wavelength`: the wavelength of the mean energy of each pixel's photons;
- `photon_rms_wavelength`: the wavelength of the root mean square energy of each pixel's photons, which sets how far their electrons spread.

`dn` holds each row as the two outputs read it: the left output's prescan, the first half of the image, the left output's overscan, the right output's overscan, the second half of the image and the right output's prescan. The parallel overscan rows follow the image's rows.

## How a frame differs from a run

A frame and an `eclipse` run share the instrument's throughput, PSF and detector settings, but they do not yet model all the same effects:

| Effect | `eclipse` run | Full-CCD frame |
| --- | --- | --- |
| Pinholes in the filter | with `simulation.enable_pinholes` | not modelled |
| Visible stray light | with `simulation.vis_sl` | not modelled |
| Dark current | over the clear, the exposure and the read-out of its own rows alone, unless `simulation.readout_dark_current` is `False` | over the clear, the exposure and the read-out of every row |
| Wavelength of each pixel | a fixed step, `detector.wvl_res`, or with `simulation.focal_plane_dispersion` the focal plane's step where the line lands | the focal plane's dispersion, from 17.0 to 16.8 mA per row |
| The 1 mm gap between the CCDs | not modelled: a window sees an unbroken spectrum | no rows from 198.26 to 199.51 Angstrom |
| Rows outside the 170.0 to 212.3 Angstrom band | lit | dark |
| Smear without a shutter, and charge transfer | refused | modelled |
| Prescan and overscan | none | both |
