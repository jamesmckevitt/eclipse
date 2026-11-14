# A full-CCD frame

A full-CCD frame is what the SW camera reads out at one slit position: both CCDs, with their prescan and overscan. It holds only the spectra in a synthesis file, so for a frame of the whole band ECLIPSE first synthesises every line in it.

![On the left, Fe XII 195.12 over the active region, with the slit and its fiducials marked; on the right, what both CCDs read out at that slit position](images/example-slit-and-frame.png)

This is the frame of the [worked example](worked-example.ipynb)'s active region in a 60 s exposure, with the slit, in red on the left, through its core.

## From an MHD simulation

Write a [run file](run-files.md#a-full-ccd-frame) with an `atmosphere` and a `frame:` section, such as `slit_core.yaml`:

```yaml
instrument: SWC
atmosphere: muram_ar_1160000.h5

simulation:
  slit_width: 0.4 arcsec
  expos: 60 s
  psf: True
  slit_mask: True

frame:
  slit_position: 0 arcsec
  continuum: True
```

```bash
eclipse frame slit_core.yaml
```

This synthesises the band under the slit, then writes the frame to `run/result/slit_core_frame.h5`. The band's lines take a few hours to work out the first time, and are kept in `run/input/band_lines.h5` for later frames.

## From your own spectra

With a `synthesis_file` in place of the `atmosphere`, the frame holds that file's spectra as they are, from [another code](other-codes.md) or from ECLIPSE:

```yaml
instrument: SWC
synthesis_file: my_spectra.h5

simulation:
  expos: 60 s

frame:
  slit_position: 0 arcsec
```

## The band's spectrum

`synthesise-spectra --band` synthesises every line between two wavelengths as one spectrum. `--image-x` keeps a strip of the image, as the whole image can take tens of gigabytes:

```bash
synthesise-spectra --atmosphere muram_ar_1160000.h5 --continuum \
  --band "169 Angstrom" "213.5 Angstrom" --image-x "-1 Mm" "1 Mm"
```

Set `--vel-lim` wide enough for the fastest flows.

## From Python

```python
import astropy.units as u
from eclipse import full_ccd_frame, write_full_frame

frame = full_ccd_frame("run/input/muram_ar_band.h5", slit_position=0 * u.arcsec,
                       exposure=60 * u.s, slit_width=0.4 * u.arcsec, slit_mask=True)
write_full_frame(frame, "frame.h5")
```

`slit_mask=True` blocks the light beyond the ends of the slit and behind its fiducials. `sim=Simulation(...)` takes the slit width, PSF and noise from a `Simulation` instead.

The median of the image columns in each row gives the spectrum of the band:

```python
import matplotlib.pyplot as plt
import numpy as np

pre, over = frame.settings["serial_prescan"], frame.settings["serial_overscan"]
fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
for ccd in ("left", "right"):
    rows = frame.wavelength[ccd].to_value(u.AA)
    half = frame.rate[ccd].shape[1] // 2
    columns = np.r_[pre:pre + half, pre + half + 2 * over:pre + 2 * half + 2 * over]
    order = np.argsort(rows)
    spectrum = np.median(frame.dn[ccd][:rows.size, columns], axis=1)
    ax.plot(rows[order], spectrum[order], label=f"{ccd} CCD")
ax.set_yscale("log")
ax.set_xlabel("Wavelength [Angstrom]")
ax.set_ylabel("DN per pixel")
ax.legend()
plt.show()
```

![The spectrum of the band in the frame, from 170 to 212 Angstrom, with the median over the slit in each row](images/example-frame-spectrum.png)

## What a frame holds

For each CCD, `"left"` and `"right"`:

- `rate`: photons per second reaching each image pixel
- `photons`: photons in each pixel over the exposure
- `dn`: the frame as read out, in DN, with the prescan and overscan of each output
- `wavelength`: the wavelength each row records
- `field_angle`: each column's angle along the slit, positive towards solar north
- `photon_wavelength`, `photon_rms_wavelength`: the wavelengths of the mean and root mean square energies of each pixel's photons

Each CCD's columns count from its left output: from the north end of the slit on the left CCD, and the south end on the right. `settings["spectra"]` names the entries of the synthesis file the frame holds.

## How a frame differs from a run

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
