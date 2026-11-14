# Full-CCD frames

<div class="glance" markdown>
<div markdown>
**You need**

An [atmosphere file](../files/atmosphere.md), an [emission measure file](../files/emission-measure.md) or a [synthesis file](../files/synthesis.md)
</div>
<div markdown>
**You run**

`eclipse frame`
</div>
<div markdown>
**You get**

A [frame file](../files/frame.md) in `run/result/`
</div>
<div markdown>
**Next**

[Analysing a full-CCD frame](../analyse/frame.md)
</div>
</div>

A full-CCD frame is what the SW camera reads out at one slit position: both CCDs, with prescan and overscan. It holds the spectra of a synthesis file.

![On the left, Fe XII 195.12 over the active region, with the slit and its fiducials marked; on the right, what both CCDs read out at that slit position](../images/example-slit-and-frame.png)

## From an MHD simulation { data-routes="mhd" }

Write a [run file](../run-files.md#a-full-ccd-frame) with an `atmosphere` and a `frame:` section, such as `slit_core.yaml`:

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

This synthesises the band under the slit and writes `run/result/slit_core_frame.h5`. The band's lines take a few hours the first time, and are kept in a [band lines file](../files/band-lines.md). An `emission_measure` works [the same way](../emission/dem.md#a-full-ccd-frame).

## From your own spectra { data-routes="spectra" }

With a `synthesis_file`, the frame holds that file's spectra as they are:

```yaml
instrument: SWC
synthesis_file: my_spectra.h5

simulation:
  expos: 60 s

frame:
  slit_position: 0 arcsec
```

## The band's spectrum { data-routes="mhd dem" }

`synthesise-spectra --band` synthesises every line between two wavelengths as one spectrum. `--image-x` keeps a strip of the image, which otherwise can take tens of gigabytes:

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

`sim=Simulation(...)` takes the slit width, PSF and noise from a `Simulation` instead.

## How a frame differs from a simulated observation

| Effect | `eclipse` run | Full-CCD frame |
| --- | --- | --- |
| Pinholes in the filter | with `simulation.enable_pinholes` | not modelled |
| Visible stray light | with `simulation.vis_sl` | not modelled |
| Dark current read out | its own rows | every row |
| Wavelength of each pixel | `detector.wvl_res`, or the focal plane's with `simulation.focal_plane_dispersion` | the focal plane's dispersion |
| The gap between the CCDs | not modelled | no rows from 198.26 to 199.51 Angstrom |
| Rows outside the 170.0 to 212.3 Angstrom band | lit | dark |
| Smear without a shutter, and charge transfer | refused | modelled |
| Prescan and overscan | none | both |
