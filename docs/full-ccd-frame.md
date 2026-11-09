# A full-CCD frame

`eclipse` observes one spectral window at a time. A full-CCD frame is what the SW camera reads out at one slit position: both CCDs, with every line in the band and the continuum, and the prescan and overscan of each output.

A frame is made from a synthesis file from ECLIPSE's [synthesis of an MHD simulation](synthesis.md), which keeps the emission measure in each bin of temperature and velocity of every pixel, and the electron density at each temperature.

This page makes the frame for the active region of the [worked example](worked-example.ipynb), with the slit through its core:

![On the left, Fe XII 195.12 over the active region, with the slit and its fiducials marked; on the right, what both CCDs read out at that slit position](images/example-slit-and-frame.png)

On the left is the Fe XII map of the region, observed without noise, with the slit in red. The triangles mark the slit's two fiducials. On the right is everything the two CCDs record through that slit in one 60 s exposure. Each vertical streak is one spectral line, seen all the way along the slit. The bright band across them is the core of the region, where the slit crosses it. The grey background is the detector's dark current and noise.

## 1. Every line in the band

Each step below starts worker processes, so in a script its code goes under `if __name__ == "__main__":`.

`band_contribution_functions` works out the contribution function of every CHIANTI line in the band, on the synthesis's grids and with its abundances, and `write_band_lines` keeps them in a file:

```python
from eclipse import band_contribution_functions, read_synthesis_products, write_band_lines

if __name__ == "__main__":
    products = read_synthesis_products("run/input/muram_ar_fe12.h5",
                                       keys=["logT_grid", "logN_grid", "config"])
    lines = band_contribution_functions((169.0, 213.5), products["logT_grid"],
                                        products["logN_grid"],
                                        abundance=products["config"]["abundance"])
    write_band_lines(lines, "band_lines.h5")
```

This only needs doing once. A synthesis's densities are always steps of 0.3 in log10 n_e, so lines worked out over a wider run of those steps serve any synthesis whose densities fall within it.

## 2. The continuum

The continuum is optional. It is worked out on wavelengths finer than a detector row, which is about 0.017 A, inside the telescope's throughput tables, from 170 to 214 A:

```python
import astropy.units as u
import numpy as np
from eclipse import read_band_lines
from eclipse.continuum import compute_continuum_fiasco

if __name__ == "__main__":
    lines = read_band_lines("band_lines.h5")
    wavelength = np.linspace(170.0, 214.0, 10481) * u.AA
    free, two_photon = compute_continuum_fiasco(wavelength, lines.logT_grid, lines.logN_grid,
                                                abundance=lines.abundance)
```

## 3. The frame

```python
from eclipse import full_ccd_frame, write_full_frame

if __name__ == "__main__":
    frame = full_ccd_frame("run/input/muram_ar_fe12.h5", lines, slit_position=0 * u.arcsec,
                           exposure=60 * u.s, slit_width=0.4 * u.arcsec, slit_mask=True,
                           n_workers=16, continuum=(wavelength, free, two_photon))
    write_full_frame(frame, "frame.h5")
```

The result is the frame at the top of this page. `dn` holds it as it is read out, with the prescan and overscan of each output, and `rate` holds the photons reaching each pixel. The median of the image columns in each row gives the spectrum of the whole band:

```python
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pre, over = frame.settings["serial_prescan"], frame.settings["serial_overscan"]
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    for ccd in ("left", "right"):
        rows = frame.wavelength[ccd].to_value(u.AA)
        half = frame.rate[ccd].shape[1] // 2
        columns = np.r_[pre:pre + half, pre + half + 2 * over:pre + 2 * half + 2 * over]
        order = np.argsort(rows)
        spectrum = np.median(frame.dn[ccd][:rows.size, columns], axis=1)
        ax.plot(rows[order], spectrum[order], label=f"{ccd} CCD")
    ax.set_xlim(169, 213)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength [Angstrom]")
    ax.set_ylabel("DN per pixel")
    ax.legend()
    plt.show()
```

![The spectrum of the band in the frame, from 170 to 212 Angstrom, with the median over the slit in each row](images/example-frame-spectrum.png)

`slit_position` is where the middle of the slit is, across it, in the synthesis file's x coordinates. Along the slit, the middle of the image is at the middle of the slit, which is the middle column of the CCDs.

A frame takes the synthesis file's y to increase towards solar north, as in nominal observations. Each CCD's columns count from its left output, as the camera reads them. On the left CCD, that is from the north end of the slit. The right CCD faces the other way, with its serial register on its own outer edge, so its columns count from the south end. `field_angle` gives the angle of each column along the slit.

With the PSF on, the blur along the slit carries light out of the columns at the ends of the scene. `psf_boundary` sets what the blur brings in from beyond the scene, as in a run. With `"replicate"`, the default, the Sun beyond the scene is taken to look like the scene's edges, and the columns beyond the scene get no light. With `"zero"`, the Sun beyond the scene is dark, so the columns at the ends of the scene lose up to about a third of their light to the columns just beyond. `psf_boundary` sets the same across the slit, for a slit near the synthesis's x edges. With `"replicate"`, a slit partly beyond those edges gets light only over its part inside them.

With `slit_mask=True`, no light reaches the columns beyond the ends of the 280 arcsec slit or behind its fiducials, as in a run (see [The slit's fiducials](instrument-model.md#the-slits-fiducials)). The CCD's 2048 columns cover 326 arcsec along the slit, so about 144 columns at each end are beyond the slit. It is off by default.

The slit width, PSF and noise can instead be taken from a `Simulation`, as in an instrument-response run:

```python
from eclipse import Simulation

if __name__ == "__main__":
    sim = Simulation(slit_width=0.4 * u.arcsec, psf=True, noise=True, slit_mask=True)
    frame = full_ccd_frame("run/input/muram_ar_fe12.h5", lines, slit_position=0 * u.arcsec,
                           exposure=60 * u.s, sim=sim, n_workers=16,
                           continuum=(wavelength, free, two_photon))
```

The frame uses the `Simulation`'s `slit_width`, `psf`, `spectral_psf`, `noise`, `psf_boundary` and `slit_mask`. A `Simulation` has the PSF off unless it is made with `psf=True`. When one of these six arguments of `full_ccd_frame` equals its default, the `Simulation`'s value is used, even if the argument was typed out. An argument with any other value must agree with the `Simulation`. The exposure is set by `exposure`, not by the `Simulation`'s `expos`. The number of workers is set by `n_workers`. A frame does not model pinholes or visible stray light. `vis_sl` must be left at zero and no pinholes enabled. `instrument` must be `"SWC"`.

By default `full_ccd_frame` runs a worker process on every CPU it may use, and each adds to the memory; set `n_workers` to use fewer.

Every part of each line, and of the continuum, is counted in photons at its own wavelength, with the telescope's throughput there, and the photons in each pixel free the electrons of their own energy.

The lines come only from plasma within the synthesis's velocity grid, so set `--vel-lim` wide enough for the fastest flows. The continuum is not Doppler shifted, and comes from all the plasma.

## What a frame holds

For each CCD, `"left"` and `"right"`:

- `rate`: the photons per second reaching each image pixel;
- `photons`: the photons in each pixel over the exposure;
- `dn`: the frame as it is read out, in DN;
- `wavelength`: the wavelength each row records;
- `field_angle`: the angle along the slit of each column, from the slit's centre, positive towards solar north in nominal observations;
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
